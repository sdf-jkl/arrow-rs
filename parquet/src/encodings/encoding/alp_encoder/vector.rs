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

use crate::encodings::alp::{AlpFloat, AlpInfo};
use crate::encodings::encoding::alp_encoder::{ALP_VECTOR_SIZE, Scratch};
use std::mem;

pub(super) enum VectorPutResult<F> {
    /// All values were encoded, and the vector is still in progress (not full)
    StillInProgress(InProgressVector<F>),
    /// After `encoded` values from `values` Vector were written, the Vector was
    /// full and fully written to the buffer.
    Finished {
        /// Number of values from the input that were encoded
        encoded_len: usize,
        /// Vector information
        finish_result: VectorFinishResult<F>,
    },
}

/// result of finishing a Vector
#[derive(Debug)]
pub(super) struct VectorFinishResult<F> {
    /// Total number of values encoded in the output vector (can be fewer than
    /// the target vector size for the last vector)
    pub(super) vector_len: usize,
    /// Returned scratch buffers/encoding parameters
    pub(super) scratch: Scratch<F>,
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
    /// Encoding parameters (maybe not known until we see a sample of the data)
    encoding_params: Option<EncodingParams>,
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
    /// Uses buffers from scratch and optional pre-known encoding paramaters
    pub(super) fn new(buffer: &mut Vec<u8>,
                      scratch: Scratch<F>) -> InProgressVector<F> {
        let Scratch {
            encoding_params,
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
            encoding_params,
            exception_positions,
            exception_values,
        }
    }

    /// Encode as many values as possible, writing directly to `dst` when possible
    ///
    /// Returns [`VectorPutResult`] which distinguishes between the vector being
    /// complete or still have space.
    pub(super) fn put(
        mut self,
        dst: &mut Vec<u8>,
        values: &[F],
    ) -> crate::errors::Result<VectorPutResult<F>> {


        // Phase 1: Determine encoding parameters from first batch if needed
        let encoding_params = self.encoding_params
            .take()
            // TODO: handle case when a small first batch is pushed (maybe buffer)
            .unwrap_or_else(|| EncodingParams::from_sample(values));

        // If we can encode an entire vector do so
        let space_left = ALP_VECTOR_SIZE - self.count;
        let num_to_encode  = values.len().min(ALP_VECTOR_SIZE).min(space_left);

        // TODO: actually encode that many values (TODO)
        // for now just treat them all as exceptions
        self.exception_values.extend(&values[0..num_to_encode]);
            self.exception_positions
                .extend(self.count..(self.count + num_to_encode));

        // Update counters
        self.count += num_to_encode;
        self.encoding_params = Some(encoding_params);
        if self.count < ALP_VECTOR_SIZE {
            Ok(VectorPutResult::StillInProgress(self))
        } else {
            Ok(VectorPutResult::Finished {
                encoded_len: num_to_encode,
                finish_result: self.finish(dst)?
            })
        }
    }

    /// Finalize this vector and write the remaining values to the `dst` buffer
    pub(super) fn finish(
        self,
        dst: &mut Vec<u8>,
    ) -> crate::errors::Result<VectorFinishResult<F>> {
        todo!();
        // TODO; finalize the header
    }
}


/// Encoding Parameters
#[derive(Debug)]
pub(super) struct EncodingParams {
    exponent: u8,
    factor: u8,
}

impl EncodingParams {
    /// Create encoding parameters from a sample of the data.
    ///
    /// Algorithm: TODO
    fn from_sample<F: AlpFloat>(values: &[F]) -> EncodingParams {
        // TEMP hard code
        let exponent = 0;
        let factor = 5;
        EncodingParams {
            exponent,
            factor,
        }
    }
}