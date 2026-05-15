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

//! ALP (Adaptive Lossless floating-Point) encoder.
//!
//! Based on the draft Parquet spec: <https://github.com/apache/parquet-format/pull/557>

mod vector;

use crate::basic::Encoding;
use crate::data_type::DataType;
use crate::encodings::alp::{AlpFloat, AlpHeader};
use crate::errors::{ParquetError, Result};
use bytes::Bytes;
use std::fmt::Formatter;

use super::Encoder;
use vector::{InProgressVector, VectorFinishResult, VectorPutResult, EncodingParams};

/// Vector size in bits
const ALP_LOG_VECTOR_SIZE: u8 = 10;
const ALP_VECTOR_SIZE: usize = 1 << ALP_LOG_VECTOR_SIZE;
const ALP_COMPRESSION_MODE: u8 = 0;
const ALP_INTEGER_ENCODING_FOR_BIT_PACK: u8 = 0;

/// ALP encoder for `f32` / `f64` columns.
///
pub(crate) struct AlpEncoder<T: DataType>
where
    T::T: AlpFloat,
{
    /// In progress buffer.
    ///
    /// The final ALP page is incrementally constructed in-place in this buffer.
    ///
    /// Page format:
    ///
    /// ```text
    /// +-------------+-----------------------------+--------------------------------------+
    /// |   Header    |        Offset Array         |            Vector Data               |
    /// |  (7 bytes)  |   (num_vectors * 4 bytes)   |            (variable)                |
    /// +-------------+------+------+-----+---------+----------+----------+-----+----------+
    /// | Page Header | off0 | off1 | ... | off N-1 | Vector 0 | Vector 1 | ... | Vec N-1  |
    /// |  (7 bytes)  | (4B) | (4B) |     |  (4B)   |(variable)|(variable)|     |(variable)|
    /// +-------------+------+------+-----+---------+----------+----------+-----+----------+
    /// ```
    ///
    buffer: Vec<u8>,
    /// Currently in progress vector
    vector_state: VectorState<T::T>,
    /// Total number of values encoded, NOT including the currently in progress vector
    count: usize,
}

/// State machine that tracks the currently in progress vector being built
#[derive(Default, Debug)]
enum VectorState<T> {
    /// Default value, temporarily left in place during state transition
    #[default]
    Placeholder,
    /// No vector in progress
    None(Scratch<T>),
    /// Vector started,
    InProgress(InProgressVector<T>),
}

/// Buffers to reuse for next vector
struct Scratch<T> {
    /// Encoding parameters (maybe not known until we see a sample of the data)
    encoding_params: Option<EncodingParams>,
    exception_positions: Vec<usize>,
    exception_values: Vec<T>,
}

impl<T> std::fmt::Debug for Scratch<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scratch")
            .field(
                "exception_positions",
                &format!(
                    "Vec<usize> with capacity {}",
                    self.exception_positions.capacity()
                ),
            )
            .field(
                "exception_values",
                &format!("Vec<T> with capacity {}", self.exception_values.capacity()),
            )
            .finish()
    }
}

impl<T: DataType> AlpEncoder<T>
where
    T::T: AlpFloat,
{
    /// Create a new encoder with
    pub(crate) fn new() -> Self {
        let expected_num_values = 1024;
        // TODO allocate with expected size
        // Leave space for an header
        let mut buffer = vec![0; AlpHeader::SERIALIZED_SIZE];

        let scratch = Scratch {
            encoding_params: None,
            exception_positions: Vec::with_capacity(expected_num_values),
            exception_values: Vec::with_capacity(expected_num_values),
        };

        Self {
            buffer,
            vector_state: VectorState::None(scratch),
            count: 0,
        }
    }
}

impl<T: DataType> Encoder<T> for AlpEncoder<T>
where
    T::T: AlpFloat,
{
    /// Write as many values from `values` as possible into the underlying vector.
    /// Will
    fn put(&mut self, mut values: &[T::T]) -> Result<()> {
        let mut done = false;
        loop {
            // leave Default value in self.vector_state temporarily
            let current_state = std::mem::take(&mut self.vector_state);
            self.vector_state = match current_state {
                VectorState::Placeholder => {
                    return Err(general_err!(
                        "Internal Error: ALP encoder called after error"
                    ));
                }
                // begin a new vector
                VectorState::None(scratch) => {
                    let in_progress = InProgressVector::new(&mut self.buffer, scratch);
                    // Will encode on the next loop through
                    VectorState::InProgress(in_progress)
                }
                VectorState::InProgress(mut in_progress) => {
                    match in_progress.put(&mut self.buffer, values)? {
                        // Consumed enough to complete the vector
                        VectorPutResult::Finished {
                            encoded_len,
                            finish_result,
                        } => {
                            let VectorFinishResult {
                                vector_len: vector_size,
                                scratch,
                            } = finish_result;
                            self.count += vector_size;
                            values = &values[encoded_len..];
                            VectorState::None(scratch)
                        }
                        // Consumed all input
                        VectorPutResult::StillInProgress(in_progress) => {
                            done = true;
                            VectorState::InProgress(in_progress)
                        }
                    }
                }
            };
            if done {
                return Ok(());
            }
        }
    }

    fn encoding(&self) -> Encoding {
        Encoding::ALP
    }

    fn estimated_data_encoded_size(&self) -> usize {
        // TODO add estimated data size of in progress vector
        self.buffer.len()
    }

    fn estimated_memory_size(&self) -> usize {
        // TODO add data size of in progess vector
        self.buffer.capacity()
    }

    fn flush_buffer(&mut self) -> Result<Bytes> {
        // finish up the last vector if needed
        let current_state = std::mem::take(&mut self.vector_state);
        self.vector_state = if let VectorState::InProgress(in_progress) = current_state {
            let VectorFinishResult {
                vector_len,
                scratch,
            } = in_progress.finish(&mut self.buffer)?;
            self.count += vector_len;
            VectorState::None(scratch)
        } else {
            current_state
        };

        // update page header in place now that we know the final value count
        let value_count: i32 = self.count.try_into().map_err(|_| {
            general_err!("ALP can encode at most i32::MAX values, got {}", self.count)
        })?;

        let header = AlpHeader {
            compression_mode: ALP_COMPRESSION_MODE,
            integer_encoding: ALP_INTEGER_ENCODING_FOR_BIT_PACK,
            //log_vector_size: ALP_LOG_VECTOR_SIZE,
            log_vector_size: 0, // TODO support something more
            num_elements: value_count,
        };

        header.serialize(&mut self.buffer);

        // reset internal fields for next time
        self.count = 0;
        let mut buffer = Vec::with_capacity(self.buffer.capacity());
        std::mem::swap(&mut buffer, &mut self.buffer);

        Ok(Bytes::from(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::Type as PhysicalType;
    use crate::data_type::{DoubleType, FloatType};
    use crate::encodings::decoding::get_decoder;
    use crate::schema::types::{ColumnDescPtr, ColumnDescriptor, ColumnPath, Type as SchemaType};
    use std::sync::Arc;

    fn col_desc(t: PhysicalType) -> ColumnDescPtr {
        let ty = SchemaType::primitive_type_builder("c", t)
            .with_length(0)
            .build()
            .unwrap();
        Arc::new(ColumnDescriptor::new(
            Arc::new(ty),
            0,
            0,
            ColumnPath::new(vec![]),
        ))
    }

    /// Compare floats by bit pattern so NaN, ±0.0, and ±Inf are distinguished.
    trait BitsAsU64 {
        fn bits(&self) -> u64;
    }
    impl BitsAsU64 for f32 {
        fn bits(&self) -> u64 {
            self.to_bits() as u64
        }
    }
    impl BitsAsU64 for f64 {
        fn bits(&self) -> u64 {
            self.to_bits()
        }
    }

    fn check_roundtrip<T: DataType>(values: &[T::T])
    where
        T::T: AlpFloat + BitsAsU64 + std::fmt::Debug,
    {
        let descr = col_desc(T::get_physical_type());
        let mut encoder = AlpEncoder::<T>::new();
        encoder.put(values).unwrap();
        let bytes = encoder.flush_buffer().unwrap();

        let mut decoder = get_decoder::<T>(descr, Encoding::ALP).unwrap();
        decoder.set_data(bytes, values.len()).unwrap();
        let mut out = vec![T::T::default(); values.len()];
        let read = decoder.get(&mut out).unwrap();
        assert_eq!(read, values.len());

        for (i, (got, want)) in out.iter().zip(values).enumerate() {
            assert_eq!(
                got.bits(),
                want.bits(),
                "bit mismatch at index {i}: got={got:?}, want={want:?}"
            );
        }
    }

    #[test]
    fn alp_encoder_reports_alp_encoding() {
        let encoder = AlpEncoder::<FloatType>::new();
        assert_eq!(encoder.encoding(), Encoding::ALP);
        let encoder = AlpEncoder::<DoubleType>::new();
        assert_eq!(encoder.encoding(), Encoding::ALP);
    }

    #[test]
    fn alp_encoder_roundtrip_f32() {
        check_roundtrip::<FloatType>(&[
            1.23,
            4.56,
            7.89,
            0.12,
            f32::NAN,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]);
    }

    #[test]
    fn alp_encoder_roundtrip_f64() {
        check_roundtrip::<DoubleType>(&[
            1.23,
            4.56,
            7.89,
            0.12,
            f64::NAN,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ]);
    }

    #[test]
    fn alp_encoder_roundtrip_multi_vector() {
        // 1024 + 58: one full vector plus a partial trailing vector.
        let values: Vec<f32> = (0..1082).map(|i| i as f32 * 0.1).collect();
        check_roundtrip::<FloatType>(&values);
    }

    #[test]
    fn alp_encoder_roundtrip_empty() {
        check_roundtrip::<FloatType>(&[]);
        check_roundtrip::<DoubleType>(&[]);
    }
}
