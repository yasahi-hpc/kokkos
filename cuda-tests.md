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
| Optimized CUDA build | `build_gpu_release` (Release, `-O3 -DNDEBUG`, no `Kokkos_ENABLE_DEBUG`; otherwise as `build_gpu`), at `f83f9ae81` (`sub_release_A100.sh`): `*complex*:*atomic*` 27/27 on Serial and Cuda, `*numeric_traits*` 14 on Serial1 and Cuda1, `*half*` 14 (the std-only comparisons skip): the Debug counts. The bit-exact tests (`complex_half_compute_in_float`, `complex_multiplication`, `*_constexpr_path_matches_runtime`) hold at `-O3` on host and device | 10430071 |
| Complete CUDA core test suite | all 10 core test executables of `build_gpu`, unfiltered, at `82a06f7a8` (`sub_full_A100.sh`): `Cuda1` 224, `Cuda2` 129, `Cuda3` 62, `Cuda_ViewSupport` 37, `Serial_SmokeTest` 128, `Serial1` 217, `Serial2` 126, `Serial_ViewSupport` 37, `Default` 76 pass; `Cuda_SmokeTest` without `cuda_graph.sized_functor_launch`: 126 pass (`sub_cudasmoke_A100.sh`). Skips are Kokkos's own. `sized_functor_launch` segfaults, as it did in this build before the branch (below) | 10421072, 10421610 |
| B4c | the half numeric traits are `constexpr half_t`/`bhalf_t` on CUDA (bit patterns on HIP/SYCL): compile-time checks hold in both nvcc passes; `*half*` 14, complex 27/27, numeric traits 14 on Serial and Cuda | 10420748, 10420749, 10420750 |
| B4b-CUDA prototype | software `constexpr` conversions for `__half`/`__nv_bfloat16`: 0 mismatches over all floats, all bit patterns and all 65536² arithmetic pairs (non-NaN) | 10396298, 10396345, 10396374 |
| CH6 benchmark | paired `__half2` arithmetic: no gain. It halves the instruction count (nvcc does not pair the scalar operations), but the measured kernels are bound by memory bandwidth or latency; results bit-identical | 10397058 |

Every one of these runs was the **Debug** build, and the test runs used **filters**:
`*complex*:*atomic*`, `*half*` or `*numeric_traits*`.

## Open

None on this machine (the future work below needs other hardware or a decision).

### Not from this work: `cuda_graph.sized_functor_launch` segfaults

`Kokkos_CoreUnitTest_Cuda_SmokeTest` segfaults (exit 139) in
`cuda_graph.sized_functor_launch`, which launches an `int` functor padded to 100,
6000 and 100,000 bytes through a CUDA Graph. It did so in this `build_gpu` before the
branch: ctest recorded `***Exception: SegFault` on 2026-10-22, 10-26 and 10-27
(`sub_tests_A100.sh.*.out`). The first of those is before the base commit `a360acf37`
(2026-10-23). The branch changes nothing in the graph code. Not investigated: CUDA
12.6, Debug, A100; possibly the launch path for large functors.

## Future work

| Task | Why not now |
|---|---|
| **CH6:** paired `__half2` arithmetic (future work, decided 2026-11-24) | only if a benchmark of instruction-bound code (not memory-bound) shows a gain |
| ~~**Pre-Ampere bf16 double rounding:** before sm_80, `double` and 64-bit integers convert to `bfloat16` through `float`, rounding twice (`Kokkos_Cuda_Half_Conversion.hpp`)~~ | fixed 2026-11-25 (found with the SYCL double-rounding fix): the pre-Ampere branch now uses `__double2bfloat16` and the `_rn` integer intrinsics, which round once on every architecture in CUDA ≥ 12.2. Compile-checked for sm_70. To run: `pjsub sub_bf16fix_A100.sh`, whose `v_sm70` is built for sm_70; the A100 JIT-compiles its PTX, so the pre-Ampere code runs. Expected line (corrected): `1 4e81 4e81 1 1`. **Job 10432508:** `*half*` 14 pass, 2 skip (`*_io_vs_std`), including the new tie-adjacent integers against `__ll2bfloat16_rn`/`__int2bfloat16_rn` and the changed `double` compound assignments; `*complex*:*atomic*` 27/27 on Serial and Cuda. `v_sm70` did not run: it was under `/home`, which the compute nodes cannot see. It is now in `/work/.../tmp/claude_builds/complex_a6/`. **Job 10432529:** the same test results, and `v_sm70` prints `1 4e81 4e81 3f80 1`. So on the pre-Ampere path `bhalf_t(double)` and `bhalf_t` from 64- and 32-bit integers near a tie round once (through float they would give `0`, `4e80`, `4e80`), and `half_t(double)` too. The fourth value was a wrong expectation, not a bug: `x = 1; x += 1 + (2^-8 + 2^-60)` adds in double, the sum rounds to the tie 1 + 2^-8, and the tie goes to even, `0x3f80` (as for `std::bfloat16_t`). The check now uses `x = 0; x += 2^-134 (1 + 2^-52)` instead (exact in double; expected `0x1`); optional rerun |
| **C++23 with nvcc** (CUDA ≥ 13.3, per Kokkos 5.2.0) | CUDA 12.6 here. With the CUDA backend `half_t` stays `__half`, so this is mainly a compile check |

Not CUDA, so listed elsewhere: HIP and SYCL validation, and B4b on HIP and SYCL
(`complex_half.md` §8.3, §8.4, §8.5).

## Scripts and tools

| File | Purpose |
|---|---|
| `sub_complex_A100.sh` | link and run `*complex*:*atomic*` on Serial and Cuda |
| `sub_half_A100.sh` | link and run `Kokkos_CoreUnitTest_Default` `*half*` |
| `sub_numtraits_A100.sh` | `*numeric_traits*` on Serial1 and Cuda1 |
| `sub_full_A100.sh` | every core test executable, unfiltered (objects compiled on the login node first, from the `/work` path) |
| `sub_release_A100.sh` | the filtered tests in `build_gpu_release` (the optimized build) |
| `sub_cudasmoke_A100.sh` | `Cuda_SmokeTest` without the pre-existing `sized_functor_launch` segfault |
| `tmp/claude_builds/complex_a6/` | the CH3 accuracy/benchmark, CH6 benchmark, B4b prototype and probe sources, with their `sub_*_A100.sh` jobs |

Objects are compiled on the login node, with modules `gcc/12.2.0 cuda/12.6 cmake/3.24.0`.
Executables link only on GPU nodes, where `libcuda` exists, and the jobs are submitted
with `pjsub`.
