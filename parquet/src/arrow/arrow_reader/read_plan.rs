// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`ReadPlan`] and [`ReadPlanBuilder`] for determining which rows to read
//! from a Parquet file

use crate::arrow::array_reader::ArrayReader;
use crate::arrow::array_reader::RowGroups;
use crate::arrow::array_reader::primitive_array::coerce_array;
use crate::arrow::arrow_reader::selection::RowSelectionPolicy;
use crate::arrow::arrow_reader::selection::RowSelectionStrategy;
use crate::arrow::arrow_reader::{
    ArrowPredicate, ParquetRecordBatchReader, RowSelection, RowSelectionCursor, RowSelector,
};
use crate::arrow::buffer::bit_util::sign_extend_be;
use crate::arrow::buffer::offset_buffer::OffsetBuffer;
use crate::arrow::in_memory_row_group::InMemoryRowGroup;
use crate::arrow::schema::{ParquetField, parquet_to_arrow_field};
use crate::basic::Encoding;
use crate::basic::Type;
use crate::column::page::Page;
use crate::data_type::BoolType;
use crate::data_type::DoubleType;
use crate::data_type::FloatType;
use crate::data_type::Int32Type;
use crate::data_type::Int64Type;
use crate::encodings::decoding::Decoder;
use crate::encodings::decoding::PlainDecoder;
use crate::errors::{ParquetError, Result};
use arrow_array::BinaryArray;
use arrow_array::BooleanArray;
use arrow_array::Decimal128Array;
use arrow_array::Decimal256Array;
use arrow_array::Float32Array;
use arrow_array::Float64Array;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::{Array, ArrayRef};
use arrow_buffer::i256;
use arrow_schema::DataType as ArrowType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_select::filter::prep_null_mask_filter;
use bytes::Bytes;
use std::collections::VecDeque;

/// A builder for [`ReadPlan`]
#[derive(Clone, Debug)]
pub struct ReadPlanBuilder {
    batch_size: usize,
    /// Which rows to select. Includes the result of all filters applied so far
    selection: Option<RowSelection>,
    /// Policy to use when materializing the row selection
    row_selection_policy: RowSelectionPolicy,
}

impl ReadPlanBuilder {
    /// Create a `ReadPlanBuilder` with the given batch size
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            selection: None,
            row_selection_policy: RowSelectionPolicy::default(),
        }
    }

    /// Set the current selection to the given value
    pub fn with_selection(mut self, selection: Option<RowSelection>) -> Self {
        self.selection = selection;
        self
    }

    /// Configure the policy to use when materialising the [`RowSelection`]
    ///
    /// Defaults to [`RowSelectionPolicy::Auto`]
    pub fn with_row_selection_policy(mut self, policy: RowSelectionPolicy) -> Self {
        self.row_selection_policy = policy;
        self
    }

    /// Returns the current row selection policy
    pub fn row_selection_policy(&self) -> &RowSelectionPolicy {
        &self.row_selection_policy
    }

    /// Returns the current selection, if any
    pub fn selection(&self) -> Option<&RowSelection> {
        self.selection.as_ref()
    }

    /// Specifies the number of rows in the row group, before filtering is applied.
    ///
    /// Returns a [`LimitedReadPlanBuilder`] that can apply
    /// offset and limit.
    ///
    /// Call [`LimitedReadPlanBuilder::build_limited`] to apply the limits to this
    /// selection.
    pub(crate) fn limited(self, row_count: usize) -> LimitedReadPlanBuilder {
        LimitedReadPlanBuilder::new(self, row_count)
    }

    /// Returns true if the current plan selects any rows
    pub fn selects_any(&self) -> bool {
        self.selection
            .as_ref()
            .map(|s| s.selects_any())
            .unwrap_or(true)
    }

    /// Returns the number of rows selected, or `None` if all rows are selected.
    pub fn num_rows_selected(&self) -> Option<usize> {
        self.selection.as_ref().map(|s| s.row_count())
    }

    /// Returns the [`RowSelectionStrategy`] for this plan.
    ///
    /// Guarantees to return either `Selectors` or `Mask`, never `Auto`.
    pub(crate) fn resolve_selection_strategy(&self) -> RowSelectionStrategy {
        match self.row_selection_policy {
            RowSelectionPolicy::Selectors => RowSelectionStrategy::Selectors,
            RowSelectionPolicy::Mask => RowSelectionStrategy::Mask,
            RowSelectionPolicy::Auto { threshold, .. } => {
                let selection = match self.selection.as_ref() {
                    Some(selection) => selection,
                    None => return RowSelectionStrategy::Selectors,
                };

                // total_rows: total number of rows selected / skipped
                // effective_count: number of non-empty selectors
                let (total_rows, effective_count) =
                    selection.iter().fold((0usize, 0usize), |(rows, count), s| {
                        if s.row_count > 0 {
                            (rows + s.row_count, count + 1)
                        } else {
                            (rows, count)
                        }
                    });

                if effective_count == 0 {
                    return RowSelectionStrategy::Mask;
                }

                if total_rows < effective_count.saturating_mul(threshold) {
                    RowSelectionStrategy::Mask
                } else {
                    RowSelectionStrategy::Selectors
                }
            }
        }
    }

    /// Evaluates an [`ArrowPredicate`], updating this plan's `selection`
    ///
    /// If the current `selection` is `Some`, the resulting [`RowSelection`]
    /// will be the conjunction of the existing selection and the rows selected
    /// by `predicate`.
    ///
    /// Note: pre-existing selections may come from evaluating a previous predicate
    /// or if the [`ParquetRecordBatchReader`] specified an explicit
    /// [`RowSelection`] in addition to one or more predicates.
    pub fn with_predicate(
        mut self,
        array_reader: Box<dyn ArrayReader>,
        predicate: &mut dyn ArrowPredicate,
    ) -> Result<Self> {
        let reader = ParquetRecordBatchReader::new(array_reader, self.clone().build());
        let mut filters = vec![];
        for maybe_batch in reader {
            let maybe_batch = maybe_batch?;
            let input_rows = maybe_batch.num_rows();
            let filter = predicate.evaluate(maybe_batch)?;
            // Since user supplied predicate, check error here to catch bugs quickly
            if filter.len() != input_rows {
                return Err(arrow_err!(
                    "ArrowPredicate predicate returned {} rows, expected {input_rows}",
                    filter.len()
                ));
            }
            match filter.null_count() {
                0 => filters.push(filter),
                _ => filters.push(prep_null_mask_filter(&filter)),
            };
        }

        let raw = RowSelection::from_filters(&filters);
        self.selection = match self.selection.take() {
            Some(selection) => Some(selection.and_then(&raw)),
            None => Some(raw),
        };
        Ok(self)
    }

    /// Applies an encoded predicate selection, updating this plan's `selection`.
    ///
    /// If the current `selection` is `Some`, the resulting [`RowSelection`]
    /// will be the conjunction of the existing selection and the rows selected
    /// by `selection`.
    pub fn with_encoded_selection(mut self, selection: RowSelection) -> Result<Self> {
        self.selection = match self.selection.take() {
            Some(existing) => Some(existing.and_then(&selection)),
            None => Some(selection),
        };
        Ok(self)
    }

    /// Evaluates an [`ArrowPredicate`] against encoded data in `row_group`,
    /// updating this plan's `selection`.
    ///
    /// This is intended for dictionary-encoded columns where the dictionary
    /// can be decoded once, the predicate applied to dictionary values, and
    /// then used to filter the RLE dictionary indices.
    ///
    /// `fields` is used to resolve Arrow logical types for the projected
    /// column so the predicate sees Arrow-typed values. If `fields` is
    /// `None`, the Arrow type is inferred from the Parquet schema.
    pub(crate) fn with_encoded_predicate(
        self,
        row_group: &InMemoryRowGroup<'_>,
        predicate: &mut dyn ArrowPredicate,
        fields: Option<&ParquetField>,
    ) -> Result<Self> {
        let context = Self::encoded_predicate_context(row_group, predicate, fields)?;
        let EncodedPredicateContext::Supported {
            column_idx,
            arrow_type,
        } = context
        else {
            return Err(ParquetError::General(
                "Encoded predicate evaluation is not supported for this projection".to_string(),
            ));
        };

        let existing_selection = self.selection.as_ref();
        let selection = Self::evaluate_dictionary_predicate(
            row_group,
            column_idx,
            predicate,
            &arrow_type,
            existing_selection,
        )?;
        self.with_encoded_selection(selection)
    }

    /// Returns true if an encoded predicate evaluation is supported for this
    /// row group and predicate projection.
    ///
    /// This checks that the predicate projects exactly one non-nested,
    /// non-nullable column, and that the Parquet physical type can be mapped
    /// to the resolved Arrow logical type.
    pub(crate) fn encoded_predicate_supported(
        row_group: &InMemoryRowGroup<'_>,
        predicate: &dyn ArrowPredicate,
        fields: Option<&ParquetField>,
    ) -> bool {
        matches!(
            Self::encoded_predicate_context(row_group, predicate, fields),
            Ok(EncodedPredicateContext::Supported { .. })
        )
    }

    /// Create a final `ReadPlan` the read plan for the scan
    pub fn build(mut self) -> ReadPlan {
        // If selection is empty, truncate
        if !self.selects_any() {
            self.selection = Some(RowSelection::from(vec![]));
        }

        // Preferred strategy must not be Auto
        let selection_strategy = self.resolve_selection_strategy();

        let Self {
            batch_size,
            selection,
            row_selection_policy: _,
        } = self;

        let selection = selection.map(|s| s.trim());

        let row_selection_cursor = selection
            .map(|s| {
                let trimmed = s.trim();
                let selectors: Vec<RowSelector> = trimmed.into();
                match selection_strategy {
                    RowSelectionStrategy::Mask => {
                        RowSelectionCursor::new_mask_from_selectors(selectors)
                    }
                    RowSelectionStrategy::Selectors => RowSelectionCursor::new_selectors(selectors),
                }
            })
            .unwrap_or(RowSelectionCursor::new_all());

        ReadPlan {
            batch_size,
            row_selection_cursor,
        }
    }

    fn evaluate_dictionary_predicate(
        row_group: &InMemoryRowGroup<'_>,
        column_idx: usize,
        predicate: &mut dyn ArrowPredicate,
        arrow_type: &ArrowType,
        existing_selection: Option<&RowSelection>,
    ) -> Result<RowSelection> {
        let row_group_metadata = row_group.row_group_metadata();
        let column = row_group_metadata.column(column_idx);
        let column_descr = column.column_descr();

        if column_descr.max_def_level() > 0 || column_descr.max_rep_level() > 0 {
            return Err(ParquetError::General(
                "Encoded predicate evaluation does not support nested or nullable columns"
                    .to_string(),
            ));
        }

        let mut page_readers = row_group.column_chunks(column_idx)?;
        let mut page_reader = page_readers.next().transpose()?.ok_or_else(|| {
            ParquetError::General("Missing page reader for column chunk".to_string())
        })?;

        let mut dict_page: Option<(Bytes, u32, Encoding)> = None;
        let mut data_pages: Vec<Page> = Vec::new();

        while let Some(page) = page_reader.get_next_page()? {
            match &page {
                Page::DictionaryPage {
                    buf,
                    num_values,
                    encoding,
                    ..
                } => {
                    dict_page = Some((buf.clone(), *num_values, *encoding));
                }
                Page::DataPage { .. } | Page::DataPageV2 { .. } => data_pages.push(page),
            }
        }

        let (dict_buf, dict_values, dict_encoding) = dict_page.ok_or_else(|| {
            ParquetError::General("Missing dictionary page for encoded predicate".to_string())
        })?;

        if !matches!(dict_encoding, Encoding::PLAIN | Encoding::PLAIN_DICTIONARY) {
            return Err(ParquetError::General(format!(
                "Unsupported dictionary encoding {dict_encoding:?} for encoded predicate"
            )));
        }

        let dict_array =
            Self::decode_dictionary_page(column_descr, arrow_type, dict_buf, dict_values)?;
        let dict_schema = Schema::new(vec![Field::new(
            column_descr.name(),
            dict_array.data_type().clone(),
            false,
        )]);
        let dict_batch = RecordBatch::try_new(std::sync::Arc::new(dict_schema), vec![dict_array])
            .map_err(|err| ParquetError::General(err.to_string()))?;

        let dict_filter = predicate
            .evaluate(dict_batch)
            .map_err(|err| ParquetError::General(err.to_string()))?;

        if dict_filter.len() != dict_values as usize {
            return Err(ParquetError::General(format!(
                "ArrowPredicate returned {} rows for dictionary with {dict_values} values",
                dict_filter.len()
            )));
        }

        let dict_filter = if dict_filter.null_count() == 0 {
            dict_filter
        } else {
            prep_null_mask_filter(&dict_filter)
        };

        let dict_allowed: Vec<bool> = (0..dict_filter.len())
            .map(|idx| dict_filter.value(idx))
            .collect();

        let mut filters = Vec::with_capacity(data_pages.len());
        for page in data_pages {
            let (buf, num_values, encoding, def_levels_len, rep_levels_len) = match page {
                Page::DataPage {
                    buf,
                    num_values,
                    encoding,
                    ..
                } => (buf, num_values, encoding, 0_u32, 0_u32),
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    ..
                } => (
                    buf,
                    num_values,
                    encoding,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                ),
                Page::DictionaryPage { .. } => continue,
            };

            if !matches!(
                encoding,
                Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
            ) {
                return Err(ParquetError::General(format!(
                    "Unsupported data page encoding {encoding:?} for encoded predicate"
                )));
            }

            if def_levels_len != 0 || rep_levels_len != 0 {
                return Err(ParquetError::General(
                    "Encoded predicate evaluation does not support definition or repetition levels"
                        .to_string(),
                ));
            }

            let data = if def_levels_len == 0 && rep_levels_len == 0 {
                buf
            } else {
                let start = (def_levels_len + rep_levels_len) as usize;
                buf.slice(start..)
            };

            if data.is_empty() {
                return Err(ParquetError::General(
                    "Missing dictionary index data in page".to_string(),
                ));
            }

            let bit_width = data[0];
            let mut decoder = crate::encodings::rle::RleDecoder::new(bit_width);
            decoder.set_data(data.slice(1..))?;

            let mut indices = vec![0i32; num_values as usize];
            let read = decoder.get_batch(&mut indices)?;
            if read != indices.len() {
                return Err(ParquetError::General(
                    "Not enough dictionary indices for data page".to_string(),
                ));
            }

            let mut values = Vec::with_capacity(indices.len());
            for index in indices {
                let index = index as usize;
                let allowed = dict_allowed.get(index).copied().ok_or_else(|| {
                    ParquetError::General(format!(
                        "Dictionary index {index} out of bounds for dictionary with {} values",
                        dict_allowed.len()
                    ))
                })?;
                values.push(allowed);
            }
            filters.push(BooleanArray::from(values));
        }

        let raw = RowSelection::from_filters(&filters);
        match existing_selection {
            None => Ok(raw),
            Some(selection) => {
                let filter_rows: usize = filters.iter().map(|filter| filter.len()).sum();
                if filter_rows == selection.row_count() {
                    Ok(raw)
                } else {
                    let mut selection_for_filters = selection.clone();
                    let total_rows =
                        selection_for_filters.row_count() + selection_for_filters.skipped_row_count();
                    if filter_rows > total_rows {
                        let mut selectors: Vec<RowSelector> =
                            selection_for_filters.iter().cloned().collect();
                        selectors.push(RowSelector::skip(filter_rows - total_rows));
                        selection_for_filters = RowSelection::from(selectors);
                    }

                    let total_rows =
                        selection_for_filters.row_count() + selection_for_filters.skipped_row_count();
                    if filter_rows == total_rows {
                        Self::apply_selection_to_filters(&filters, &selection_for_filters)
                    } else {
                        Err(ParquetError::General(format!(
                            "Encoded predicate evaluation produced {filter_rows} rows, expected {total_rows} or {}",
                            selection.row_count()
                        )))
                    }
                }
            }
        }
    }

    fn decode_dictionary_page(
        column_descr: &crate::schema::types::ColumnDescriptor,
        arrow_type: &ArrowType,
        buf: Bytes,
        num_values: u32,
    ) -> Result<ArrayRef> {
        let num_values = num_values as usize;
        match column_descr.physical_type() {
            Type::BOOLEAN => {
                Self::decode_plain::<BoolType>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(BooleanArray::from(values)) as ArrayRef)
                    .and_then(|array| coerce_array(array, arrow_type))
            }
            Type::INT32 => {
                Self::decode_plain::<Int32Type>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(Int32Array::from(values)) as ArrayRef)
                    .and_then(|array| coerce_array(array, arrow_type))
            }
            Type::INT64 => {
                Self::decode_plain::<Int64Type>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(Int64Array::from(values)) as ArrayRef)
                    .and_then(|array| coerce_array(array, arrow_type))
            }
            Type::FLOAT => {
                Self::decode_plain::<FloatType>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(Float32Array::from(values)) as ArrayRef)
                    .and_then(|array| coerce_array(array, arrow_type))
            }
            Type::DOUBLE => {
                Self::decode_plain::<DoubleType>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(Float64Array::from(values)) as ArrayRef)
                    .and_then(|array| coerce_array(array, arrow_type))
            }
            // Type::INT96 => {
            //     let values =
            //         Self::decode_plain::<Int96Type>(buf, num_values, column_descr.type_length())?;
            //     let buffer = values.into_buffer(arrow_type);
            //     let array = Int64Array::new(ScalarBuffer::new(buffer, 0, num_values), None);
            //     coerce_array(std::sync::Arc::new(array) as ArrayRef, arrow_type)
            // }
            Type::BYTE_ARRAY => {
                Self::decode_byte_array_dictionary(arrow_type, buf, num_values as u32)
            }
            other => Err(ParquetError::General(format!(
                "Encoded predicate evaluation does not support physical type {other:?}"
            ))),
        }
    }

    fn decode_plain<T: crate::data_type::DataType>(
        buf: Bytes,
        num_values: usize,
        type_length: i32,
    ) -> Result<Vec<T::T>> {
        let mut decoder = PlainDecoder::<T>::new(type_length);
        decoder.set_data(buf, num_values)?;
        let mut values = vec![T::T::default(); num_values];
        let read = decoder.get(&mut values)?;
        if read != num_values {
            return Err(ParquetError::General(
                "Unexpected number of values decoded from dictionary page".to_string(),
            ));
        }
        Ok(values)
    }

    fn decode_byte_array_dictionary(
        arrow_type: &ArrowType,
        buf: Bytes,
        num_values: u32,
    ) -> Result<ArrayRef> {
        if matches!(arrow_type, ArrowType::Dictionary(_, _)) {
            return Err(ParquetError::General(
                "Encoded predicate evaluation does not support dictionary Arrow types".to_string(),
            ));
        }

        let validate_utf8 = matches!(arrow_type, ArrowType::Utf8 | ArrowType::LargeUtf8);
        let len = num_values as usize;

        match arrow_type {
            ArrowType::LargeUtf8 | ArrowType::LargeBinary => {
                Self::decode_byte_array_dictionary_impl::<i64>(arrow_type, buf, len, validate_utf8)
            }
            ArrowType::Utf8
            | ArrowType::Binary
            | ArrowType::Decimal128(_, _)
            | ArrowType::Decimal256(_, _) => {
                Self::decode_byte_array_dictionary_impl::<i32>(arrow_type, buf, len, validate_utf8)
            }
            _ => Err(ParquetError::General(format!(
                "Encoded predicate evaluation does not support Arrow type {arrow_type:?}"
            ))),
        }
    }

    fn decode_byte_array_dictionary_impl<I: arrow_array::OffsetSizeTrait>(
        arrow_type: &ArrowType,
        buf: Bytes,
        len: usize,
        validate_utf8: bool,
    ) -> Result<ArrayRef> {
        let mut buffer = OffsetBuffer::<I>::default();
        let mut decoder = crate::arrow::array_reader::byte_array::ByteArrayDecoderPlain::new(
            buf,
            len,
            Some(len),
            validate_utf8,
        );
        decoder.read(&mut buffer, usize::MAX)?;

        match arrow_type {
            ArrowType::Decimal128(p, s) => {
                let array = buffer.into_array(None, ArrowType::Binary);
                let binary = array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .ok_or_else(|| {
                        ParquetError::General(
                            "Failed to build binary array from dictionary values".to_string(),
                        )
                    })?;
                let decimal = Decimal128Array::from_unary(binary, |x| match x.len() {
                    0 => i128::default(),
                    _ => i128::from_be_bytes(sign_extend_be(x)),
                })
                .with_precision_and_scale(*p, *s)?;
                Ok(std::sync::Arc::new(decimal))
            }
            ArrowType::Decimal256(p, s) => {
                let array = buffer.into_array(None, ArrowType::Binary);
                let binary = array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .ok_or_else(|| {
                        ParquetError::General(
                            "Failed to build binary array from dictionary values".to_string(),
                        )
                    })?;
                let decimal = Decimal256Array::from_unary(binary, |x| match x.len() {
                    0 => i256::default(),
                    _ => i256::from_be_bytes(sign_extend_be(x)),
                })
                .with_precision_and_scale(*p, *s)?;
                Ok(std::sync::Arc::new(decimal))
            }
            _ => Ok(buffer.into_array(None, arrow_type.clone())),
        }
    }

    /// Applies an existing selection to a list of page-aligned boolean filters,
    /// reducing them to only the rows already selected.
    ///
    /// This is required when evaluating a predicate on encoded data after
    /// a prior selection has already been applied, but the encoded evaluation
    /// was performed on full row group pages.
    fn apply_selection_to_filters(
        filters: &[BooleanArray],
        selection: &RowSelection,
    ) -> Result<RowSelection> {
        let total_rows = selection.row_count() + selection.skipped_row_count();
        let filter_rows: usize = filters.iter().map(|filter| filter.len()).sum();
        if total_rows != filter_rows {
            return Err(ParquetError::General(format!(
                "Encoded predicate evaluation produced {filter_rows} rows, expected {total_rows}"
            )));
        }

        let mut values = Vec::with_capacity(filter_rows);
        for filter in filters {
            for idx in 0..filter.len() {
                values.push(filter.value(idx));
            }
        }

        let mut reduced = Vec::with_capacity(selection.row_count());
        let mut pos = 0usize;
        for selector in selection.iter() {
            if selector.skip {
                pos = pos.saturating_add(selector.row_count);
                continue;
            }

            for _ in 0..selector.row_count {
                reduced.push(values[pos]);
                pos += 1;
            }
        }

        if pos != values.len() {
            return Err(ParquetError::General(
                "Encoded predicate selection did not consume all rows".to_string(),
            ));
        }

        Ok(RowSelection::from_filters(&[BooleanArray::from(reduced)]))
    }
}

enum EncodedPredicateContext {
    Supported {
        column_idx: usize,
        arrow_type: ArrowType,
    },
    Unsupported,
}

impl ReadPlanBuilder {
    fn encoded_predicate_context(
        row_group: &InMemoryRowGroup<'_>,
        predicate: &dyn ArrowPredicate,
        fields: Option<&ParquetField>,
    ) -> Result<EncodedPredicateContext> {
        let row_group_metadata = row_group.row_group_metadata();
        let projection = predicate.projection();
        let projected_columns: Vec<usize> = (0..row_group_metadata.num_columns())
            .filter(|column_idx| projection.leaf_included(*column_idx))
            .collect();

        if projected_columns.len() != 1 {
            return Ok(EncodedPredicateContext::Unsupported);
        }

        let column_idx = projected_columns[0];
        let column = row_group_metadata.column(column_idx);
        let column_descr = column.column_descr();

        if column_descr.max_def_level() > 0 || column_descr.max_rep_level() > 0 {
            return Ok(EncodedPredicateContext::Unsupported);
        }

        let Some(page_encoding_stats_mask) = column.page_encoding_stats_mask() else {
            return Ok(EncodedPredicateContext::Unsupported);
        };

        let mut saw_dictionary = false;
        for encoding in page_encoding_stats_mask.encodings() {
            match encoding {
                Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => {
                    saw_dictionary = true;
                }
                _ => return Ok(EncodedPredicateContext::Unsupported),
            }
        }

        if !saw_dictionary {
            return Ok(EncodedPredicateContext::Unsupported);
        }

        let arrow_type = match Self::arrow_type_for_column(fields, column_idx) {
            Some(arrow_type) => arrow_type.clone(),
            None => parquet_to_arrow_field(column_descr)
                .map(|f| f.data_type().clone())
                .map_err(|err| ParquetError::General(err.to_string()))?,
        };

        if !Self::supports_encoded_dictionary_type(column_descr.physical_type(), &arrow_type) {
            return Ok(EncodedPredicateContext::Unsupported);
        }

        Ok(EncodedPredicateContext::Supported {
            column_idx,
            arrow_type,
        })
    }

    fn arrow_type_for_column<'a>(
        field: Option<&'a ParquetField>,
        column_idx: usize,
    ) -> Option<&'a ArrowType> {
        let field = field?;
        match &field.field_type {
            crate::arrow::schema::ParquetFieldType::Primitive { col_idx, .. } => {
                (*col_idx == column_idx).then_some(&field.arrow_type)
            }
            crate::arrow::schema::ParquetFieldType::Group { children } => children
                .iter()
                .find_map(|child| Self::arrow_type_for_column(Some(child), column_idx)),
            crate::arrow::schema::ParquetFieldType::Virtual(_) => None,
        }
    }

    fn supports_encoded_dictionary_type(physical: Type, arrow_type: &ArrowType) -> bool {
        if matches!(arrow_type, ArrowType::Dictionary(_, _)) {
            return false;
        }

        match physical {
            Type::BOOLEAN => matches!(arrow_type, ArrowType::Boolean),
            Type::INT32 => matches!(
                arrow_type,
                ArrowType::UInt8
                    | ArrowType::Int8
                    | ArrowType::UInt16
                    | ArrowType::Int16
                    | ArrowType::Int32
                    | ArrowType::UInt32
                    | ArrowType::Date32
                    | ArrowType::Date64
                    | ArrowType::Time32(_)
                    | ArrowType::Timestamp(_, _)
                    | ArrowType::Decimal32(_, _)
                    | ArrowType::Decimal64(_, _)
                    | ArrowType::Decimal128(_, _)
                    | ArrowType::Decimal256(_, _)
            ),
            Type::INT64 => matches!(
                arrow_type,
                ArrowType::Int64
                    | ArrowType::UInt64
                    | ArrowType::Date64
                    | ArrowType::Time64(_)
                    | ArrowType::Duration(_)
                    | ArrowType::Timestamp(_, _)
                    | ArrowType::Decimal64(_, _)
                    | ArrowType::Decimal128(_, _)
                    | ArrowType::Decimal256(_, _)
            ),
            Type::FLOAT => matches!(arrow_type, ArrowType::Float32),
            Type::DOUBLE => matches!(arrow_type, ArrowType::Float64),
            Type::INT96 => matches!(arrow_type, ArrowType::Timestamp(_, _)),
            Type::BYTE_ARRAY => matches!(
                arrow_type,
                ArrowType::Binary
                    | ArrowType::LargeBinary
                    | ArrowType::Utf8
                    | ArrowType::LargeUtf8
                    | ArrowType::Decimal128(_, _)
                    | ArrowType::Decimal256(_, _)
            ),
            _ => false,
        }
    }
}

/// Builder for [`ReadPlan`] that applies a limit and offset to the read plan
///
/// See [`ReadPlanBuilder::limited`] to create this builder.
pub(crate) struct LimitedReadPlanBuilder {
    /// The underlying builder
    inner: ReadPlanBuilder,
    /// Total number of rows in the row group before the selection, limit or
    /// offset are applied
    row_count: usize,
    /// The offset to apply, if any
    offset: Option<usize>,
    /// The limit to apply, if any
    limit: Option<usize>,
}

impl LimitedReadPlanBuilder {
    /// Create a new `LimitedReadPlanBuilder` from the existing builder and number of rows
    fn new(inner: ReadPlanBuilder, row_count: usize) -> Self {
        Self {
            inner,
            row_count,
            offset: None,
            limit: None,
        }
    }

    /// Set the offset to apply to the read plan
    pub(crate) fn with_offset(mut self, offset: Option<usize>) -> Self {
        self.offset = offset;
        self
    }

    /// Set the limit to apply to the read plan
    pub(crate) fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    /// Apply offset and limit, updating the selection on the underlying builder
    /// and returning it.
    pub(crate) fn build_limited(self) -> ReadPlanBuilder {
        let Self {
            mut inner,
            row_count,
            offset,
            limit,
        } = self;

        // If the selection is empty, truncate
        if !inner.selects_any() {
            inner.selection = Some(RowSelection::from(vec![]));
        }

        // If an offset is defined, apply it to the `selection`
        if let Some(offset) = offset {
            inner.selection = Some(match row_count.checked_sub(offset) {
                None => RowSelection::from(vec![]),
                Some(remaining) => inner
                    .selection
                    .map(|selection| selection.offset(offset))
                    .unwrap_or_else(|| {
                        RowSelection::from(vec![
                            RowSelector::skip(offset),
                            RowSelector::select(remaining),
                        ])
                    }),
            });
        }

        // If a limit is defined, apply it to the final `selection`
        if let Some(limit) = limit {
            inner.selection = Some(
                inner
                    .selection
                    .map(|selection| selection.limit(limit))
                    .unwrap_or_else(|| {
                        RowSelection::from(vec![RowSelector::select(limit.min(row_count))])
                    }),
            );
        }

        inner
    }
}

/// A plan reading specific rows from a Parquet Row Group.
///
/// See [`ReadPlanBuilder`] to create `ReadPlan`s
#[derive(Debug)]
pub struct ReadPlan {
    /// The number of rows to read in each batch
    batch_size: usize,
    /// Row ranges to be selected from the data source
    row_selection_cursor: RowSelectionCursor,
}

impl ReadPlan {
    /// Returns a mutable reference to the selection selectors, if any
    #[deprecated(since = "57.1.0", note = "Use `row_selection_cursor_mut` instead")]
    pub fn selection_mut(&mut self) -> Option<&mut VecDeque<RowSelector>> {
        if let RowSelectionCursor::Selectors(selectors_cursor) = &mut self.row_selection_cursor {
            Some(selectors_cursor.selectors_mut())
        } else {
            None
        }
    }

    /// Returns a mutable reference to the row selection cursor
    pub fn row_selection_cursor_mut(&mut self) -> &mut RowSelectionCursor {
        &mut self.row_selection_cursor
    }

    /// Return the number of rows to read in each output batch
    #[inline(always)]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::ProjectionMask;
    use crate::basic::{Encoding, EncodingMask, Repetition, Type as PhysicalType};
    use crate::file::metadata::{ColumnChunkMetaData, FileMetaData, ParquetMetaData, RowGroupMetaData};
    use crate::schema::types::{SchemaDescriptor, Type as SchemaType};
    use arrow_array::{BooleanArray, RecordBatch};
    use arrow_schema::ArrowError;
    use std::sync::Arc;

    fn builder_with_selection(selection: RowSelection) -> ReadPlanBuilder {
        ReadPlanBuilder::new(1024).with_selection(Some(selection))
    }

    struct TestPredicate {
        projection: ProjectionMask,
    }

    impl TestPredicate {
        fn new(projection: ProjectionMask) -> Self {
            Self { projection }
        }
    }

    impl ArrowPredicate for TestPredicate {
        fn projection(&self) -> &ProjectionMask {
            &self.projection
        }

        fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
            Ok(BooleanArray::from(vec![true; batch.num_rows()]))
        }
    }

    fn build_metadata(
        page_encoding_stats_mask: Option<EncodingMask>,
    ) -> (ParquetMetaData, Arc<SchemaDescriptor>) {
        let schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("a", PhysicalType::INT32)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap();

        let schema_descr = Arc::new(SchemaDescriptor::new(Arc::new(schema)));
        let column_descr = schema_descr.column(0);

        let mut column_builder = ColumnChunkMetaData::builder(column_descr)
            .set_num_values(10)
            .set_data_page_offset(0)
            .set_dictionary_page_offset(Some(4))
            .set_encodings_mask(EncodingMask::new_from_encodings(
                [Encoding::PLAIN_DICTIONARY].iter(),
            ));

        if let Some(mask) = page_encoding_stats_mask {
            column_builder = column_builder.set_page_encoding_stats_mask(mask);
        }

        let column_meta = column_builder.build().unwrap();
        let row_group = RowGroupMetaData::builder(schema_descr.clone())
            .set_num_rows(10)
            .add_column_metadata(column_meta)
            .build()
            .unwrap();

        let file_metadata = FileMetaData::new(1, 10, None, None, schema_descr.clone(), None);
        let parquet_metadata = ParquetMetaData::new(file_metadata, vec![row_group]);

        (parquet_metadata, schema_descr)
    }

    #[test]
    fn encoded_predicate_requires_page_encoding_stats_mask() {
        let (metadata, schema_descr) = build_metadata(None);
        let projection = ProjectionMask::leaves(&schema_descr, [0]);
        let predicate = TestPredicate::new(projection);
        let row_group = InMemoryRowGroup {
            offset_index: None,
            column_chunks: vec![None; schema_descr.num_columns()],
            row_count: 10,
            row_group_idx: 0,
            metadata: &metadata,
        };

        assert!(!ReadPlanBuilder::encoded_predicate_supported(
            &row_group,
            &predicate,
            None
        ));
    }

    #[test]
    fn encoded_predicate_rejects_non_dictionary_page_encodings() {
        let mask = EncodingMask::new_from_encodings(
            [Encoding::PLAIN_DICTIONARY, Encoding::PLAIN].iter(),
        );
        let (metadata, schema_descr) = build_metadata(Some(mask));
        let projection = ProjectionMask::leaves(&schema_descr, [0]);
        let predicate = TestPredicate::new(projection);
        let row_group = InMemoryRowGroup {
            offset_index: None,
            column_chunks: vec![None; schema_descr.num_columns()],
            row_count: 10,
            row_group_idx: 0,
            metadata: &metadata,
        };

        assert!(!ReadPlanBuilder::encoded_predicate_supported(
            &row_group,
            &predicate,
            None
        ));
    }

    #[test]
    fn encoded_predicate_accepts_dictionary_only_page_encodings() {
        let mask = EncodingMask::new_from_encodings([Encoding::RLE_DICTIONARY].iter());
        let (metadata, schema_descr) = build_metadata(Some(mask));
        let projection = ProjectionMask::leaves(&schema_descr, [0]);
        let predicate = TestPredicate::new(projection);
        let row_group = InMemoryRowGroup {
            offset_index: None,
            column_chunks: vec![None; schema_descr.num_columns()],
            row_count: 10,
            row_group_idx: 0,
            metadata: &metadata,
        };

        assert!(ReadPlanBuilder::encoded_predicate_supported(
            &row_group,
            &predicate,
            None
        ));
    }

    #[test]
    fn preferred_selection_strategy_prefers_mask_by_default() {
        let selection = RowSelection::from(vec![RowSelector::select(8)]);
        let builder = builder_with_selection(selection);
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Mask
        );
    }

    #[test]
    fn preferred_selection_strategy_prefers_selectors_when_threshold_small() {
        let selection = RowSelection::from(vec![RowSelector::select(8)]);
        let builder = builder_with_selection(selection)
            .with_row_selection_policy(RowSelectionPolicy::Auto { threshold: 1 });
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Selectors
        );
    }
}
