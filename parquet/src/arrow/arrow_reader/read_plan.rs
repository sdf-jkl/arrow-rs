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
use crate::arrow::arrow_reader::RowGroups;
use crate::arrow::arrow_reader::selection::RowSelectionPolicy;
use crate::arrow::arrow_reader::selection::RowSelectionStrategy;
use crate::arrow::arrow_reader::{
    ArrowPredicate, ParquetRecordBatchReader, RowSelection, RowSelectionCursor, RowSelector,
};
use crate::arrow::in_memory_row_group::InMemoryRowGroup;
use crate::basic::Encoding;
use crate::basic::Type;
use crate::column::page::Page;
use crate::data_type::BoolType;
use crate::data_type::ByteArray;
use crate::data_type::ByteArrayType;
use crate::data_type::DoubleType;
use crate::data_type::FloatType;
use crate::data_type::Int32Type;
use crate::data_type::Int64Type;
use crate::encodings::decoding::Decoder;
use crate::encodings::decoding::PlainDecoder;
use crate::errors::{ParquetError, Result};
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::BinaryArray;
use arrow_array::BooleanArray;
use arrow_array::Float32Array;
use arrow_array::Float64Array;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
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
    pub(crate) fn with_encoded_predicate(
        self,
        row_group: &InMemoryRowGroup<'_>,
        predicate: &mut dyn ArrowPredicate,
    ) -> Result<Self> {
        let row_group_metadata = row_group.row_group_metadata();
        let projection = predicate.projection();
        let projected_columns: Vec<usize> = (0..row_group_metadata.num_columns())
            .filter(|column_idx| projection.leaf_included(*column_idx))
            .collect();

        if projected_columns.is_empty() {
            return Ok(self);
        }

        if projected_columns.len() > 1 {
            return Err(ParquetError::General(
                "Encoded predicate evaluation only supports a single column".to_string(),
            ));
        }

        let column_idx = projected_columns[0];
        let selection = Self::evaluate_dictionary_predicate(row_group, column_idx, predicate)?;
        self.with_encoded_selection(selection)
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

        let dict_array = Self::decode_dictionary_page(column_descr, dict_buf, dict_values)?;
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

            let values: Vec<bool> = indices
                .iter()
                .map(|index| {
                    let index = *index as usize;
                    dict_allowed.get(index).copied().unwrap_or(false)
                })
                .collect();
            filters.push(BooleanArray::from(values));
        }

        Ok(RowSelection::from_filters(&filters))
    }

    fn decode_dictionary_page(
        column_descr: &crate::schema::types::ColumnDescriptor,
        buf: Bytes,
        num_values: u32,
    ) -> Result<ArrayRef> {
        let num_values = num_values as usize;
        match column_descr.physical_type() {
            Type::BOOLEAN => {
                Self::decode_plain::<BoolType>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(BooleanArray::from(values)) as ArrayRef)
            }
            Type::INT32 => {
                Self::decode_plain::<Int32Type>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(Int32Array::from(values)) as ArrayRef)
            }
            Type::INT64 => {
                Self::decode_plain::<Int64Type>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(Int64Array::from(values)) as ArrayRef)
            }
            Type::FLOAT => {
                Self::decode_plain::<FloatType>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(Float32Array::from(values)) as ArrayRef)
            }
            Type::DOUBLE => {
                Self::decode_plain::<DoubleType>(buf, num_values, column_descr.type_length())
                    .map(|values| std::sync::Arc::new(Float64Array::from(values)) as ArrayRef)
            }
            Type::BYTE_ARRAY => {
                Self::decode_plain::<ByteArrayType>(buf, num_values, column_descr.type_length())
                    .map(|values| {
                        let bytes = values.iter().map(ByteArray::data);
                        std::sync::Arc::new(BinaryArray::from_iter_values(bytes)) as ArrayRef
                    })
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

    fn builder_with_selection(selection: RowSelection) -> ReadPlanBuilder {
        ReadPlanBuilder::new(1024).with_selection(Some(selection))
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
