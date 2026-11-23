# CUDA backend: done, open and future tasks

Status on 2026-11-24 of the tasks on the CUDA backend, where `half_t` and `bhalf_t` wrap
`__half` and `__nv_bfloat16`. Machine: A100 (sm_80), nvcc 12.6 with GCC 12.2, build
`build_gpu` (Debug, CUDA + Serial, `Kokkos_ENABLE_COMPLEX_ALIGN`). The full record is in
`complex_half.md`; this file is the summary. For the std half backend (C++23) see
`c++23-tests.md`.

## Done

All verified on the A100, host (Serial) and device (Cuda):

| Step | What | Job |
|---|---|---|
| B2 (and its revision) | `half_t`/`bhalf_t` stream extraction as C++23 specifies | 10394586, 10400777 |
| B3 | half types in the scalar promotion traits | 10396174 |
| B1/H16 | numeric traits, Serial1 and Cuda1 | 10397183 |
| CH1 | `complex<half_t>`, `complex<bhalf_t>` compile and work | 10396565 |
| CH2 | explicit narrowing conversions into half complex types | 10396632 |
| CH3 | half complex arithmetic and functions computed in float; accuracy ≤ 1.4 u (`pow` 9.4 u), no spurious inf/NaN | 10397002 (accuracy and benchmark jobs before it) |
| H17 | `bool` conversions of `__half`/`__nv_bfloat16` | 10397129 |
| H18 tests | assigning numeric-trait constants (identity casts exist on CUDA) | 10404035 |
| Declared casts | every declared cast linked and correct, host and kernel (`*_cast_overloads`) | 10404595 |
| B4a, CH5 | CUDA unchanged by the std-backend `constexpr` (macro empty) | 10415251, 10415253 |
| A8 | `std::complex` interop deprecated except the converting constructor; 27/27 complex tests on Serial and Cuda, including the deprecated-interop test for `half_t`/`bhalf_t`; 0 nvcc warnings | 10415439 |
| B4b-CUDA, CH5 on CUDA | `constexpr half_t`/`bhalf_t` and `constexpr complex<half_t>` on CUDA: software conversions in constant evaluation, bit-identical to the hardware on every pattern, midpoint and 64-bit integer (host and device); B4a/CH5 `static_assert`s hold in both nvcc passes; `*half*` 14 pass, complex 27/27, numeric traits 14 | 10415931, 10415932, 10415933 |
| B4b-CUDA prototype | software `constexpr` conversions for `__half`/`__nv_bfloat16`: 0 mismatches over all floats, all bit patterns and all 65536² arithmetic pairs (non-NaN) | 10396298, 10396345, 10396374 |
| CH6 benchmark | paired `__half2` arithmetic: no gain. It halves the instruction count (nvcc does not pair the scalar operations), but the measured kernels are bound by memory bandwidth or latency; results bit-identical | 10397058 |

Every one of these runs was the **Debug** build, and the test runs used **filters**:
`*complex*:*atomic*`, `*half*` or `*numeric_traits*`.

## Open

Both can run on the A100.

### 1. Complete CUDA core test suite

**Why.** The branch changed `Kokkos_Complex.hpp` and the half headers, which much of
Kokkos includes. The A100 runs checked only the complex, half and numeric-trait
tests. A complete run covers everything else that uses complex or half values:
reductions, atomics, views, math functions and so on. On the std backend the
complete suite already passes (Aurora job 9117446).

**How.** Build and run every core test executable of `build_gpu`:
`Kokkos_CoreUnitTest_Cuda_SmokeTest`, `Cuda1`, `Cuda2`, `Cuda3`, `Cuda_ViewSupport`,
`Serial_SmokeTest`, `Serial1`, `Serial2`, `Serial_ViewSupport` and `Default`, without a
filter. Compile on the login node
(`make -j8 -f <target>/build.make <objects>`), then link and run in a GPU job, as the
existing `sub_*_A100.sh` scripts do. The job script doesn't exist yet.
*Caution:* calling `build.make` directly does not notice changed headers, so delete
the objects first (or run `cmake --build` in the job).

**Expected:** all pass, apart from tests Kokkos itself skips. Compare any failure
with a run on the branch's base commit `a360acf37` before suspecting the branch.

### 2. Optimized CUDA build (Release)

**Why.** `build_gpu` is a Debug build. On the device, nvcc contracts `a*b + c` into FMAs
by default (`--fmad=true`), also in Debug builds, which the device tests have
therefore already seen. The host side is only optimized in a Release build. That is
where the Aurora Release run showed GCC fusing half expressions (`c++23-tests.md`).
Bit-exact tests such as `complex_half_compute_in_float` and `complex_multiplication`
compare host and device paths.

**How.** A second build directory, e.g. `build_gpu_release`
(`-DCMAKE_BUILD_TYPE=Release`, otherwise as `build_gpu`). Then the filtered tests
(`*complex*:*atomic*`, `*half*`, `*numeric_traits*`) on Serial and Cuda. A new configure
needs many files: check the group's file quota on `/work` first, which is nearly full.

**Expected:** all pass. A bit-exact test that fails only in Release would point to
host-side contraction.

## Future work

| Task | Why not now |
|---|---|
| **B4c:** numeric traits as `constexpr half_t` rather than `BitComparisonWrapper` | needs B4b; changes public API |
| **CH6 revisit:** paired `__half2` arithmetic | only if a benchmark of instruction-bound code (not memory-bound) shows a gain |
| **Pre-Ampere bf16 double rounding:** before sm_80, `double` and 64-bit integers convert to `bfloat16` through `float`, rounding twice (`Kokkos_Cuda_Half_Conversion.hpp`) | known gap, unfixed; needs an NVIDIA GPU older than sm_80, which none of the available machines has |
| **C++23 with nvcc** (CUDA ≥ 13.3, per Kokkos 5.2.0) | CUDA 12.6 here. With the CUDA backend `half_t` stays `__half`, so this is mainly a compile check |

Not CUDA, so listed elsewhere: HIP and SYCL validation, and B4b on HIP and SYCL
(`complex_half.md` §8.3, §8.4, §8.5).

## Scripts and tools

| File | Purpose |
|---|---|
| `sub_complex_A100.sh` | link and run `*complex*:*atomic*` on Serial and Cuda |
| `sub_half_A100.sh` | link and run `Kokkos_CoreUnitTest_Default` `*half*` |
| `sub_numtraits_A100.sh` | `*numeric_traits*` on Serial1 and Cuda1 |
| `tmp/claude_builds/complex_a6/` | the CH3 accuracy/benchmark, CH6 benchmark, B4b prototype and probe sources, with their `sub_*_A100.sh` jobs |

Objects are compiled on the login node, with modules `gcc/12.2.0 cuda/12.6 cmake/3.24.0`.
Executables link only on GPU nodes, where `libcuda` exists, and the jobs are submitted
with `pjsub`.
