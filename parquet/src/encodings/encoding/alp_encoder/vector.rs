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

use crate::errors::{ParquetError, Result};
use crate::encodings::alp::{AlpFloat, AlpInfo, ForInfo};
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
    vector_len: usize,
    /// Encoding parameters (maybe not known until we see a sample of the data)
    encoding_params: Option<EncodingParams>,
    /// positions of values in exception_values in original vector. u16 so
    /// we can copy them directly to the output.
    exception_positions: Vec<u16>,
    /// TODO can avoid this copy when we have an entire vector.
    /// values that could not be encoded
    /// or are being accumulated before the page is done
    /// Since we don't know how many exceptions there will be we buffer them here until the end of the vector, then write them all at once to the output buffer.
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
        let header_len = AlpInfo::SERIALIZED_SIZE + ForInfo::<F::Exact>::serialized_size();
        buffer.resize(buffer.len() + header_len, 0);

        InProgressVector {
            start_pos,
            vector_len: 0,
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
    ) -> Result<VectorPutResult<F>> {


        // Phase 1: Determine encoding parameters from first batch if needed
        let encoding_params = self.encoding_params
            .take()
            // TODO: handle case when a small first batch is pushed (maybe buffer)
            .unwrap_or_else(|| EncodingParams::from_sample(values));

        // If we can encode an entire vector do so
        let space_left = ALP_VECTOR_SIZE - self.vector_len;
        let num_to_encode  = values.len().min(ALP_VECTOR_SIZE).min(space_left);

        // TODO: actually encode that many values (TODO)
        // for now just treat them all as exceptions
        self.exception_values.extend(&values[0..num_to_encode]);

        // TODO check overflow
        let vector_len = self.vector_len as u16;
        self.exception_positions.extend(vector_len..vector_len + num_to_encode as u16);

        // Update counters
        self.vector_len += num_to_encode;
        self.encoding_params = Some(encoding_params);
        if self.vector_len < ALP_VECTOR_SIZE {
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
    ) -> Result<VectorFinishResult<F>> {
        let Self{ start_pos, vector_len, encoding_params, exception_positions, exception_values } = self;

        // If we had no encoding parameters, no values were written (zero length vector)
        let Some(encoding_params) = encoding_params else {
            return Err(general_err!("Internal error: ALP Vector finished with no values written"));
        };

        // Output is like this (starting at start_pos) (see diagram on InProgressVector)
        // AlpInfo
        // ForInfo
        // PackedValues (already written)
        // ExceptionPositions (write from exception_positions)
        // ExceptionValues (write from exception_values)

        let num_exceptions = exception_values.len();
        let num_exceptions: u16 = num_exceptions.try_into()
            .map_err(|_| general_err!("More than u16::MAX exceptions in ALP Vector: {num_exceptions}"))?;

        // TODO move ALPInfo and FOR creation into the encoding params
        let alp_info = AlpInfo::new(encoding_params.exponent, encoding_params.factor, num_exceptions);
        alp_info.serialize(&mut dst[start_pos..]);
        let frame_of_reference = Default::default();
        let bit_width = 0; // TODO actually compute bitwidth (from encoding params)
        let for_info = ForInfo::<F::Exact>::new(frame_of_reference, bit_width);
        for_info.serialize(&mut dst[start_pos + AlpInfo::SERIALIZED_SIZE..]);

        // ExceptionPositions (all uint16)
        // TODO make this faster
        dst.extend(exception_positions.iter().flat_map(|pos| pos.to_le_bytes()));
        // ExceptionValues (all T)
        //dst.extend(exception_values.iter().flat_map(|val| val.to_le_bytes()));
        // temp just write zeros (the traits are getting messy)
        let num_bytes = exception_values.len() * mem::size_of::<F::Exact>();
        dst.extend(std::iter::repeat(0).take(num_bytes));

        Ok(VectorFinishResult {
            vector_len,
            scratch: Scratch {
                encoding_params: Some(encoding_params),
                exception_positions,
                exception_values,
            },
        })
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