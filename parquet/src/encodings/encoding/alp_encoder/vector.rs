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

use crate::data_type::DataType;
use crate::encodings::alp::{AlpExact, AlpFloat, AlpInfo};
use crate::encodings::encoding::alp_encoder::{ALP_VECTOR_SIZE, Scratch};
use std::mem;

pub(super) enum VectorPutResult<T> {
    /// All values were encoded, and the vector is still in progress (not full)
    StillInProgress(InProgressVector<T>),
    /// After `encoded` values from `values` Vector were written, the Vector was
    /// full and fully written to the buffer.
    Finished {
        /// Number of values from the input that were encoded
        encoded_len: usize,
        /// Vector information
        finish_result: VectorFinishResult<T>,
    },
}

/// result of finishing a Vector
#[derive(Debug)]
pub(super) struct VectorFinishResult<T> {
    /// Total number of values encoded in the output vector (can be fewer than
    /// the target vector size for the last vector)
    pub(super) vector_len: usize,
    /// Returned scratch buffers
    pub(super) scratch: Scratch<T>,
}

/// Accumulates data for the Vector currently being encoded, buffering if necessary.
///
/// ```text
/// +-------------------+-----------------+-------------------+---------------------+-------------------+
/// |      AlpInfo      |     ForInfo     |   PackedValues    | ExceptionPositions  | ExceptionValues   |
/// |     (4 bytes)     | (5B or 9B)      |    (variable)     |     (variable)      |    (variable)     |
/// +-------------------+-----------------+-------------------+---------------------+-------------------+
/// ```
#[derive(Debug)]
pub(super) struct InProgressVector<F> {
    /// Start position of the vector in the page (points to AlpInfo)
    start_pos: usize,
    /// Number of values in the vector so far
    count: usize,
    /// positions of values in exception_values in original vector
    exception_positions: Vec<usize>,
    /// values that could not be encoded
    /// or are being accumulated before the page is done
    exception_values: Vec<F>,
}

impl<F: AlpFloat> InProgressVector<F> {
    /// Creates a new in progress vector, writing space for the eventual header
    /// to buffer
    ///
    /// Uses buffers from scratch
    pub(super) fn new(buffer: &mut Vec<u8>, scratch: Scratch<F>) -> InProgressVector<F> {
        let Scratch {
            mut exception_positions,
            mut exception_values,
        } = scratch;
        exception_positions.clear();
        exception_values.clear();

        let start_pos = buffer.len();
        // reserve space for the header. ForInfo is `frame_of_reference`
        // (4 bytes for f32 / 8 bytes for f64 — same as `T::get_type_size()`)
        // plus a 1-byte `bit_width`.
        let header_len = AlpInfo::SERIALIZED_SIZE + mem::size_of::<F>() + 1;
        buffer.resize(buffer.len() + header_len, 0);

        InProgressVector {
            start_pos,
            count: 0,
            exception_positions,
            exception_values,
        }
    }

    /// Encode as many values as possible, writing directly to `buffer` when possible
    ///
    /// Returns [`VectorPutResult`] which distinguishes between the vector being
    /// complete or still have space.
    pub(super) fn put(
        mut self,
        buffer: &mut Vec<u8>,
        values: &[F],
    ) -> crate::errors::Result<VectorPutResult<F>> {
        // If there are enough values to find an exponent and scale, do it and then encode
        if (self.exception_values.len() + values.len()) >= ALP_VECTOR_SIZE {
            // TODO find frame of reference and bit width, write header, encode values, update count
        } else {
            let num = values.len().min(ALP_VECTOR_SIZE);
            // don't have enough values, treat all as exceptions
            self.exception_values.extend(&values[0..num]);
            self.exception_positions
                .extend(self.count..(self.count + num));
        }
        todo!();
    }

    /// Force flush the remaining values to the buffer
    pub(super) fn finish(
        self,
        buffer: &mut Vec<u8>,
    ) -> crate::errors::Result<VectorFinishResult<F>> {
        todo!();
        // TODO; finalize the header
    }
}
