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

use std::collections::VecDeque;

/// Tracks the remaining pages to read for each selected column
///
/// Used to track which pages have been read for each selected column
/// and if there are any remaining pages to read.
#[derive(Debug, Clone)]
pub(crate) struct RemainingPages {
    /// offsets[i] represents the current position in column i
    ///
    /// Note that certain pages may have been pruned earlier if they
    /// contain no selected rows, so the length of offsets[i] may be less
    /// than the total number of pages in column `i` and there may be gaps
    ///
    /// TODO diagram
    offsets: Vec<VecDeque<u64>>,
}

/// Try and read the next chunk of the page layout for all selected columns
pub(crate) enum PageChunk {
    /// Can read at least N rows from column chunk
    AtLeast(usize),
    /// Must skip N rows from column chunk as at least one page is missing (was pruned earlier)
    Skip(usize),
}
impl RemainingPages {
    /// The starting offsets of pages for each selected column if known. Otherwise
    /// `None` (all pages are read)
    ///
    /// `page_offsets[i][j]` corresponds to the start offset of page j in column
    /// chunk `i`.
    pub(crate) fn try_new(page_start_offsets: Option<Vec<Vec<u64>>>) -> Option<Self> {
        let offsets = page_start_offsets?
            .into_iter()
            .map(VecDeque::from)
            .collect();

        Some(Self { offsets })
    }
    /// Returns the number of values that can be read from the next chunk across
    /// all selected columns given the available pages.
    pub fn next_chunk(&self, position: usize) -> Option<PageChunk> {
        //let mut min_rows = usize::MAX;
        todo!();
    }

    /// Advances the cursor to the specified absolute offset
    pub fn advance_to_position(&mut self, position: usize) {
        // Advance each column's offset queue to the the next page that has the
        // specified position
        for offset in &mut self.offsets {
            offset.pop_front();
        }
        todo!()
    }
}
