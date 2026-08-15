<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Apache Parquet ALP benchmark

This benchmark evaluates the Apache Parquet implementation of
[ALP (Adaptive Lossless floating-Point encoding)][alp] for the
[Parquet ALP blog post][blog-pr]. It compares columns of `f64` values using:

- PLAIN encoding without compression
- PLAIN encoding with ZSTD compression
- BYTE_STREAM_SPLIT encoding with ZSTD compression
- ALP encoding without an additional block compressor

It reports compression speed, decompression speed, and compressed size for all
30 double-precision datasets in the CWI ALP corpus, plus a focused random-access
comparison on `city_temperature_f`.

## Results preview

Example benchmark run. Speed is machine-dependent; compressed size is
deterministic. The speed and size rows are arithmetic means of the 30
per-dataset results.

```text
MACHINE

CPU             AMD Ryzen AI 9 HX PRO 470 w/ Radeon 890M
ARCHITECTURE    x86_64
SIMD ISA        AVX-512F, AVX2, AVX
LOGICAL CPUS    24
OS / KERNEL     Linux 6.19.10-300.fc44.x86_64
CPU GOVERNOR    performance
RUST            rustc 1.96.1 (31fca3adb 2026-06-26)
LLVM            22.1.2
RUSTFLAGS       -C target-cpu=native

AVERAGE OF ALL 30 DATASETS

                     COMPRESSION     DECOMPRESSION     COMPRESSED SIZE
PLAIN                 65.547 GB/s     76.644 GB/s       64.01 bits/value
PLAIN + ZSTD           1.401 GB/s      3.236 GB/s       22.75 bits/value
BYTE_STREAM_SPLIT + ZSTD
                       1.786 GB/s      5.132 GB/s       32.76 bits/value
ALP                    1.273 GB/s     33.805 GB/s       24.27 bits/value

100 RANDOM ROWS FROM city_temperature_f

PLAIN                      2.819 µs
PLAIN + ZSTD          74,317.077 µs
BYTE_STREAM_SPLIT + ZSTD
                      27,025.798 µs
ALP                       10.007 µs
```

<details>
<summary>Full results for all 30 datasets</summary>

| Dataset | Parquet choice | Compression (GB/s) | Decompression (GB/s) | Compressed size (bits/value) |
|---|---|---:|---:|---:|
| arade4 | PLAIN | 63.775 | 68.205 | 64.01 |
| arade4 | PLAIN + ZSTD | 0.574 | 1.499 | 37.39 |
| arade4 | BYTE_STREAM_SPLIT + ZSTD | 1.875 | 4.980 | 54.58 |
| arade4 | ALP | 1.698 | 31.860 | 24.99 |
| basel_temp_f | PLAIN | 33.559 | 57.828 | 64.01 |
| basel_temp_f | PLAIN + ZSTD | 0.458 | 1.656 | 23.07 |
| basel_temp_f | BYTE_STREAM_SPLIT + ZSTD | 1.181 | 2.080 | 54.59 |
| basel_temp_f | ALP | 0.534 | 28.058 | 29.23 |
| basel_wind_f | PLAIN | 52.641 | 76.402 | 64.01 |
| basel_wind_f | PLAIN + ZSTD | 0.591 | 1.733 | 18.53 |
| basel_wind_f | BYTE_STREAM_SPLIT + ZSTD | 1.405 | 2.191 | 54.12 |
| basel_wind_f | ALP | 0.576 | 29.778 | 29.87 |
| bird_migration_f | PLAIN | 65.238 | 93.070 | 64.01 |
| bird_migration_f | PLAIN + ZSTD | 0.420 | 1.778 | 23.49 |
| bird_migration_f | BYTE_STREAM_SPLIT + ZSTD | 1.210 | 10.407 | 45.82 |
| bird_migration_f | ALP | 0.164 | 27.212 | 20.24 |
| bitcoin_f | PLAIN | 91.371 | 222.346 | 64.07 |
| bitcoin_f | PLAIN + ZSTD | 0.585 | 1.660 | 50.01 |
| bitcoin_f | BYTE_STREAM_SPLIT + ZSTD | 1.345 | 19.512 | 48.79 |
| bitcoin_f | ALP | 0.047 | 30.608 | 27.18 |
| bitcoin_transactions_f | PLAIN | 61.862 | 76.608 | 64.01 |
| bitcoin_transactions_f | PLAIN + ZSTD | 1.081 | 1.985 | 47.96 |
| bitcoin_transactions_f | BYTE_STREAM_SPLIT + ZSTD | 1.412 | 2.580 | 56.65 |
| bitcoin_transactions_f | ALP | 0.572 | 21.192 | 41.27 |
| city_temperature_f | PLAIN | 74.404 | 75.291 | 64.01 |
| city_temperature_f | PLAIN + ZSTD | 0.561 | 1.368 | 17.67 |
| city_temperature_f | BYTE_STREAM_SPLIT + ZSTD | 0.962 | 3.569 | 16.64 |
| city_temperature_f | ALP | 1.972 | 37.725 | 10.80 |
| cms1 | PLAIN | 64.257 | 62.655 | 64.01 |
| cms1 | PLAIN + ZSTD | 0.627 | 1.600 | 26.84 |
| cms1 | BYTE_STREAM_SPLIT + ZSTD | 0.670 | 1.651 | 38.64 |
| cms1 | ALP | 1.061 | 18.514 | 35.19 |
| cms25 | PLAIN | 70.317 | 71.847 | 64.01 |
| cms25 | PLAIN + ZSTD | 0.798 | 1.829 | 58.11 |
| cms25 | BYTE_STREAM_SPLIT + ZSTD | 1.287 | 4.345 | 56.72 |
| cms25 | ALP | 1.516 | 21.901 | 41.17 |
| cms9 | PLAIN | 65.334 | 67.735 | 64.01 |
| cms9 | PLAIN + ZSTD | 0.668 | 1.436 | 11.71 |
| cms9 | BYTE_STREAM_SPLIT + ZSTD | 2.151 | 5.502 | 10.07 |
| cms9 | ALP | 1.932 | 34.393 | 12.16 |
| food_prices | PLAIN | 71.786 | 75.631 | 64.01 |
| food_prices | PLAIN + ZSTD | 0.574 | 1.356 | 18.13 |
| food_prices | BYTE_STREAM_SPLIT + ZSTD | 0.744 | 1.865 | 25.47 |
| food_prices | ALP | 0.887 | 20.747 | 23.20 |
| gov10 | PLAIN | 67.983 | 70.573 | 64.01 |
| gov10 | PLAIN + ZSTD | 0.486 | 1.259 | 29.12 |
| gov10 | BYTE_STREAM_SPLIT + ZSTD | 0.648 | 1.740 | 37.31 |
| gov10 | ALP | 1.119 | 24.621 | 29.88 |
| gov26 | PLAIN | 66.105 | 67.432 | 64.01 |
| gov26 | PLAIN + ZSTD | 10.325 | 23.010 | 0.20 |
| gov26 | BYTE_STREAM_SPLIT + ZSTD | 8.112 | 18.208 | 0.24 |
| gov26 | ALP | 1.876 | 84.789 | 1.40 |
| gov30 | PLAIN | 61.089 | 64.308 | 64.01 |
| gov30 | PLAIN + ZSTD | 2.038 | 5.186 | 4.52 |
| gov30 | BYTE_STREAM_SPLIT + ZSTD | 1.695 | 4.155 | 6.14 |
| gov30 | ALP | 1.020 | 33.276 | 17.88 |
| gov31 | PLAIN | 70.238 | 71.923 | 64.01 |
| gov31 | PLAIN + ZSTD | 3.821 | 8.994 | 1.65 |
| gov31 | BYTE_STREAM_SPLIT + ZSTD | 3.780 | 9.643 | 2.47 |
| gov31 | ALP | 1.659 | 50.153 | 6.77 |
| gov40 | PLAIN | 74.751 | 75.933 | 64.01 |
| gov40 | PLAIN + ZSTD | 9.232 | 17.740 | 0.43 |
| gov40 | BYTE_STREAM_SPLIT + ZSTD | 6.477 | 13.890 | 0.62 |
| gov40 | ALP | 1.879 | 77.866 | 2.59 |
| medicare1 | PLAIN | 73.791 | 65.523 | 64.01 |
| medicare1 | PLAIN + ZSTD | 0.552 | 1.508 | 31.68 |
| medicare1 | BYTE_STREAM_SPLIT + ZSTD | 0.800 | 2.166 | 45.27 |
| medicare1 | ALP | 0.958 | 19.660 | 40.46 |
| medicare9 | PLAIN | 73.908 | 75.228 | 64.01 |
| medicare9 | PLAIN + ZSTD | 0.706 | 1.471 | 11.86 |
| medicare9 | BYTE_STREAM_SPLIT + ZSTD | 2.126 | 5.717 | 10.19 |
| medicare9 | ALP | 1.938 | 36.311 | 12.82 |
| neon_air_pressure | PLAIN | 70.685 | 73.563 | 64.01 |
| neon_air_pressure | PLAIN + ZSTD | 0.789 | 2.051 | 11.85 |
| neon_air_pressure | BYTE_STREAM_SPLIT + ZSTD | 0.782 | 2.268 | 28.51 |
| neon_air_pressure | ALP | 1.876 | 36.770 | 16.48 |
| neon_bio_temp_c | PLAIN | 63.818 | 69.314 | 64.01 |
| neon_bio_temp_c | PLAIN + ZSTD | 0.522 | 1.513 | 16.84 |
| neon_bio_temp_c | BYTE_STREAM_SPLIT + ZSTD | 1.240 | 2.885 | 35.40 |
| neon_bio_temp_c | ALP | 1.941 | 35.578 | 10.81 |
| neon_dew_point_temp | PLAIN | 72.097 | 73.614 | 64.01 |
| neon_dew_point_temp | PLAIN + ZSTD | 0.465 | 1.638 | 23.73 |
| neon_dew_point_temp | BYTE_STREAM_SPLIT + ZSTD | 1.503 | 2.370 | 48.00 |
| neon_dew_point_temp | ALP | 1.931 | 32.863 | 13.63 |
| neon_pm10_dust | PLAIN | 51.533 | 70.722 | 64.01 |
| neon_pm10_dust | PLAIN + ZSTD | 0.848 | 1.689 | 7.79 |
| neon_pm10_dust | BYTE_STREAM_SPLIT + ZSTD | 0.634 | 1.682 | 22.21 |
| neon_pm10_dust | ALP | 0.927 | 38.427 | 8.41 |
| neon_wind_dir | PLAIN | 54.631 | 60.583 | 64.01 |
| neon_wind_dir | PLAIN + ZSTD | 0.432 | 1.312 | 24.41 |
| neon_wind_dir | BYTE_STREAM_SPLIT + ZSTD | 1.387 | 3.518 | 42.31 |
| neon_wind_dir | ALP | 1.883 | 47.114 | 15.94 |
| nyc29 | PLAIN | 71.342 | 71.984 | 64.01 |
| nyc29 | PLAIN + ZSTD | 0.611 | 1.539 | 24.67 |
| nyc29 | BYTE_STREAM_SPLIT + ZSTD | 0.939 | 3.768 | 36.91 |
| nyc29 | ALP | 1.679 | 24.137 | 40.43 |
| poi_lat | PLAIN | 58.440 | 50.294 | 64.01 |
| poi_lat | PLAIN + ZSTD | 0.635 | 1.793 | 57.78 |
| poi_lat | BYTE_STREAM_SPLIT + ZSTD | 2.711 | 5.929 | 55.30 |
| poi_lat | ALP | 1.052 | 11.874 | 88.19 |
| poi_lon | PLAIN | 61.546 | 73.497 | 64.01 |
| poi_lon | PLAIN + ZSTD | 0.850 | 1.945 | 60.44 |
| poi_lon | BYTE_STREAM_SPLIT + ZSTD | 2.467 | 5.826 | 57.24 |
| poi_lon | ALP | 1.180 | 14.614 | 79.12 |
| ssd_hdd_benchmarks_f | PLAIN | 66.698 | 112.905 | 64.02 |
| ssd_hdd_benchmarks_f | PLAIN + ZSTD | 0.813 | 1.802 | 12.98 |
| ssd_hdd_benchmarks_f | BYTE_STREAM_SPLIT + ZSTD | 1.265 | 2.850 | 17.42 |
| ssd_hdd_benchmarks_f | ALP | 0.114 | 35.975 | 16.04 |
| stocks_de | PLAIN | 68.397 | 71.924 | 64.01 |
| stocks_de | PLAIN + ZSTD | 0.664 | 1.689 | 10.07 |
| stocks_de | BYTE_STREAM_SPLIT + ZSTD | 0.881 | 2.278 | 33.46 |
| stocks_de | ALP | 1.224 | 35.921 | 11.20 |
| stocks_uk | PLAIN | 60.600 | 64.531 | 64.01 |
| stocks_uk | PLAIN + ZSTD | 0.608 | 1.413 | 11.29 |
| stocks_uk | BYTE_STREAM_SPLIT + ZSTD | 1.180 | 4.006 | 14.89 |
| stocks_uk | ALP | 0.948 | 32.869 | 12.75 |
| stocks_usa_c | PLAIN | 64.214 | 67.851 | 64.01 |
| stocks_usa_c | PLAIN + ZSTD | 0.692 | 1.634 | 8.24 |
| stocks_usa_c | BYTE_STREAM_SPLIT + ZSTD | 0.712 | 2.390 | 26.89 |
| stocks_usa_c | ALP | 2.028 | 39.329 | 7.95 |
| **ALL AVG.** | **PLAIN** | **65.547** | **76.644** | **64.01** |
| **ALL AVG.** | **PLAIN + ZSTD** | **1.401** | **3.236** | **22.75** |
| **ALL AVG.** | **BYTE_STREAM_SPLIT + ZSTD** | **1.786** | **5.132** | **32.76** |
| **ALL AVG.** | **ALP** | **1.273** | **33.805** | **24.27** |

</details>

## Run

Requirements are a Rust toolchain, `curl`, `unzip`, and either `sha256sum` or
`shasum`.

```shell
./parquet/examples/alp_compression_stats.sh \
  > target/alp-compression-and-speed-results.md
```

The script downloads and verifies the 6.7 GiB CWI bundle, extracts the 30 `f64`
datasets, builds with `-C target-cpu=native`, and writes Markdown results to
stdout. Downloads resume if interrupted, and later runs reuse the extracted
files.

The extracted corpus occupies approximately 15 GiB. Approximately 22 GiB is
needed while the archive is also present; the archive is deleted after
extraction by default.

Configuration:

- `ALP_DATASET_DIR=/path` changes the durable dataset location.
- `ALP_DOWNLOAD_DIR=/path` changes the archive download location.
- `ALP_KEEP_ARCHIVE=1` retains the verified archive.
- An existing `RUSTFLAGS` overrides `-C target-cpu=native`.

The Rust example also accepts an individual raw little-endian `f64` `.bin`
file, a one-value-per-line `.csv` file, or a recursively searched directory:

```shell
cargo run --quiet --release -p parquet \
  --example alp_compression_stats \
  --features arrow,zstd,experimental -- /path/to/data
```

The `experimental` feature exposes internal page APIs to the example; ALP
itself is not gated by that feature.

## What is measured

- **Compressed size:** An `ArrowWriter` with dictionary encoding disabled
  supplies the compressed column-chunk size. It includes data-page headers and
  excludes the file footer.
- **Speed:** Every value is encoded and decoded in pages of at most 131,072
  values. GB/s uses the uncompressed input size, and file I/O is excluded.
  Both ZSTD choices use compression level 1 and include both pipeline stages.
  ALP compression includes first-page parameter sampling once per default
  1,048,576-value row group. Short pages are repeated and normalized for stable
  timing.
- **Random access:** A fixed seed selects the same 100 rows from
  `city_temperature_f` on every run. PLAIN and ALP skip to and decode one value.
  The ZSTD choices decompress the complete in-memory target page before their
  encoded lookup. File I/O and page discovery are excluded.

The complete output contains all 120 dataset/encoding combinations and explains
the units and averaging beside the tables.

## Reproducibility and privacy

The wrapper records a privacy-safe environment table: timestamp, commit and
worktree state, CPU, architecture, SIMD ISA, logical CPU count, governor when
available, OS/kernel, Rust/Cargo/LLVM versions, safe compiler flags, and the
dataset archive digest.

It does not print hostnames, usernames, local paths, network information, Git
remotes, the complete environment, or raw `/proc/cpuinfo`. Compiler flags that
contain paths or shell characters are reported as set but omitted. Review
generated results before publishing them.

Run publication measurements on an otherwise idle machine. Throughput varies
with hardware, CPU frequency, thermal state, and background load.

Benchmark implementation:
[`alp_compression_stats.rs`](parquet/examples/alp_compression_stats.rs) ·
[`alp_compression_stats.sh`](parquet/examples/alp_compression_stats.sh)

[alp]: https://ir.cwi.nl/pub/33334/33334.pdf
[blog-pr]: https://github.com/apache/parquet-site/pull/195
