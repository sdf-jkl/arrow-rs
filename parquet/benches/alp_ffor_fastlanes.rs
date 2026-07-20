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

//! FFOR decode throughput: arrow-rs bit-packing vs the FastLanes transposed
//! layout.
//!
//! # What this measures
//!
//! ALP stores each 1024-value vector as frame-of-reference (FOR) deltas, bit
//! packed at a per-vector width. Decoding that vector back to integers is the
//! "unffor" step: bit-unpack the deltas, then add the frame. This is the step a
//! FastLanes-ordered `integer_encoding` would change, so the benchmark isolates
//! it and excludes the ALP decimal reconstruction (the two multiplies), which is
//! identical for both layouts and would only dilute the comparison.
//!
//! Two implementations of the same unffor, over the same values:
//!   - `arrow`     - [`BitReader::get_batch`], the real arrow bit-unpack (which
//!                   dispatches to the vectorised fixed-width `unpack`), then a
//!                   scalar `+ frame` pass. Standard LSB-first bit layout.
//!   - `fastlanes` - [`fastlanes::FoR::unfor_pack`], a fused unpack + `+ frame`.
//!                   FastLanes transposed bit layout.
//!
//! # Why FastLanes is faster here (it is not about data dependencies)
//!
//! FOR has no cross-value dependency, so the usual "FastLanes parallelises
//! sequential codecs" argument (delta, RLE) does not apply. The speedup comes
//! from the bit-unpack itself. In the standard layout, value `i` sits at bit
//! offset `i * W`, so extracting the 1024 values needs a *different* shift per
//! value plus per-value word-straddle handling - which unrolls to a long
//! shift/mask/or chain rather than clean SIMD. The FastLanes transpose stores
//! the fields so that every SIMD lane extracts its value with the *same* shift,
//! so one vector shift+mask yields one value per lane. That is width-insensitive;
//! the standard layout is fastest at byte-aligned widths (8/16/32) and slowest at
//! the non-aligned widths ALP typically produces (e.g. 11).
//!
//! # Correctness
//!
//! Before timing, the two are asserted to recover byte-identical values in
//! identical (original) order - the FastLanes transpose lives only in the packed
//! bytes; `unfor_pack` returns normal row-major order. So this is a drop-in swap
//! for the bit-packing layer, changing only the on-disk byte order.
//!
//! # Running and reading the output
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" \
//!   cargo bench -p parquet --features experimental --bench alp_ffor_fastlanes
//! ```
//!
//! `--features experimental` is required: it exposes `parquet::util::bit_util`.
//! `target-cpu=native` lets the arrow unpack use the widest SIMD available (a
//! fair comparison - FastLanes ships its own SIMD).
//!
//! Compare the two implementations by their **throughput** (`thrpt`, in
//! `Gelem/s`): higher is better, so `fastlanes/W` beating `arrow/W` is the
//! result. Criterion's separate "improved/regressed" line compares a benchmark
//! against *its own previous run* (baselines cached under `target/criterion/`),
//! not arrow against fastlanes - ignore it here, or `rm -rf target/criterion` to
//! reset baselines between runs.

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fastlanes::FoR;
use parquet::util::bit_util::{BitReader, BitWriter};
use std::hint::black_box;

const VEC: usize = 1024;
const NVEC: usize = 128; // ~one rowgroup, stays in L2

/// xorshift, so no rand dependency and results are deterministic.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// One vector's data in both packed representations, plus its frame.
struct Prepared {
    arrow: Bytes,
    fastlanes: Vec<u64>,
    frame: u64,
}

fn run<const W: usize, const B: usize>(c: &mut Criterion) {
    assert_eq!(B, VEC * W / 64);
    let mask = if W == 64 { u64::MAX } else { (1u64 << W) - 1 };
    let frame = 1_000_000u64;
    let mut rng = Rng(0x9E3779B97F4A7C15 ^ (W as u64));

    let mut prepared = Vec::with_capacity(NVEC);
    for _ in 0..NVEC {
        // encoded values = frame + delta, delta in [0, 2^W)
        let values: Vec<u64> = (0..VEC).map(|_| frame + (rng.next() & mask)).collect();
        let deltas: Vec<u64> = values.iter().map(|&v| v - frame).collect();

        // arrow: pack the FOR deltas (get_batch is the inverse of the writer).
        let mut w = BitWriter::new(VEC * 8);
        for &d in &deltas {
            w.put_value(d, W);
        }
        let arrow = Bytes::from(w.consume());

        // fastlanes: for_pack subtracts the reference and transposes.
        let values_arr: [u64; VEC] = values.try_into().unwrap();
        let mut packed = [0u64; B];
        <u64 as FoR>::for_pack::<W, B>(&values_arr, frame, &mut packed);

        prepared.push(Prepared {
            arrow,
            fastlanes: packed.to_vec(),
            frame,
        });
    }

    // Correctness: both recover identical values in identical (original) order.
    {
        let mut a = [0u64; VEC];
        let mut f = [0u64; VEC];
        for p in &prepared {
            let mut reader = BitReader::new(p.arrow.clone());
            reader.get_batch::<u64>(&mut a, W);
            for x in a.iter_mut() {
                *x = x.wrapping_add(p.frame);
            }
            let fl: &[u64; B] = p.fastlanes.as_slice().try_into().unwrap();
            <u64 as FoR>::unfor_pack::<W, B>(fl, p.frame, &mut f);
            assert_eq!(a, f, "arrow/fastlanes disagree at W={W}");
        }
    }

    let mut group = c.benchmark_group("alp_ffor_decode");
    group.throughput(Throughput::Elements((NVEC * VEC) as u64));

    let mut out = [0u64; VEC];
    group.bench_with_input(BenchmarkId::new("arrow", W), &prepared, |b, prepared| {
        b.iter(|| {
            for p in prepared {
                let mut reader = BitReader::new(black_box(p.arrow.clone()));
                reader.get_batch::<u64>(&mut out, W);
                let fr = p.frame;
                for x in out.iter_mut() {
                    *x = x.wrapping_add(fr);
                }
                black_box(&out);
            }
        })
    });
    group.bench_with_input(BenchmarkId::new("fastlanes", W), &prepared, |b, prepared| {
        b.iter(|| {
            for p in prepared {
                let fl: &[u64; B] = black_box(p.fastlanes.as_slice()).try_into().unwrap();
                <u64 as FoR>::unfor_pack::<W, B>(fl, p.frame, &mut out);
                black_box(&out);
            }
        })
    });
    group.finish();
}

fn bench(c: &mut Criterion) {
    // Representative ALP bit widths, from tight (temperatures) to wide (prices).
    run::<8, { VEC * 8 / 64 }>(c);
    run::<11, { VEC * 11 / 64 }>(c);
    run::<16, { VEC * 16 / 64 }>(c);
    run::<24, { VEC * 24 / 64 }>(c);
    run::<32, { VEC * 32 / 64 }>(c);
}

criterion_group!(benches, bench);
criterion_main!(benches);
