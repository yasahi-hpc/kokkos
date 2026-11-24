# C++23 (std half backend): open and future tasks

Status on 2026-11-24 of the tasks that need GCC ≥ 13 in C++23 mode, where `half_t` and
`bhalf_t` wrap `std::float16_t` and `std::bfloat16_t`. The full record is in
`complex_half.md` (§8.2 for the runs, §8.5 for the task table); this file is the summary.

## Done

All on the std half backend, all with 0 warnings and every test passing:

| What | Where (job) |
|---|---|
| Committed complex/half work: complex, numeric traits, half tests | Grace, GCC 15 (4260499); Aurora, GCC 14 (9116902 and later) |
| Native half arithmetic (`__FLT_EVAL_METHOD_TS_18661_3__` = 16) | Aurora, `Kokkos_ARCH_SPR` (9116933); Grace, `Kokkos_ARCH_ARMV9_GRACE` (4264666, 4271100) |
| libstdc++'s complex `*` vs Kokkos's (D5): libstdc++ overflows under native half, Kokkos computes in float | both machines |
| bf16 warnings 279 → 0; H18 (missing identity casts) fixed; every declared cast linked (`*_cast_overloads`) | Aurora (9117159, 9117024, 9117216) |
| GNU dialect `-std=gnu++23`, Kokkos's default for GCC | Aurora (9117216), Grace (4264666) |
| Traits vs `std::numeric_limits`; stream I/O vs libstdc++ | Aurora (9117303) |
| Complete core test suite, Serial + OpenMP (`FULL=ON`) | Aurora (9117446) |
| Optimized builds (`BUILD_TYPE=Release`, `-O3`), four builds | Aurora (9120562) |
| B4a, CH5: `constexpr half_t`/`bhalf_t` and `constexpr complex<half_t>`/`complex<bhalf_t>`: compile-time checks evaluated for both types in four builds; CUDA unchanged | Aurora (9121273); A100 (10415251, 10415253) |
| A8 (D11), half part: the deprecated `std::complex` interop for `half_t`/`bhalf_t` on libstdc++ (deprecated code ON: 27 complex tests), and the Kokkos 6 state (`DEPRECATED_CODE=OFF`: 26), four builds each, 0 warnings | Aurora (9121388, 9121428) |

All of these are Debug builds (`-O0`) except the Release row.

**What the optimized builds showed.** On a CPU with half FMA (Sapphire Rapids, the
`native` and `gnu-native` builds) GCC contracts `a*b - c` into one half FMA, rounded
once, in strict `-std=c++23` as well as `-std=gnu++23`. Probe values change:

| probe (`native`, `gnu-native`) | Debug | Release |
|---|---|---|
| `std::float16_t` and `half_t` `256*256-65000` | inf | 544 |
| `std::complex<std::float16_t>` `(269,111.5)²` | (inf, 60000) | (59936, 60000) |
| `Kokkos::complex<half_t>` `(269,111.5)²` | (59936, 60000) | (59936, 60000) |

- `half_t` arithmetic is fused across its inlined operators like `float` and `double`
  arithmetic; so in optimized code it is not rounded after each operation. That is
  GCC's default (`-ffp-contract=fast`), not a Kokkos bug; `-ffp-contract=off` restores
  per-operation rounding.
- libstdc++'s complex `*` overflows under native half only without contraction.
- Kokkos's complex results and all tests are unchanged.

Details: `complex_half.md` §8.2 ("Release run") and the D5 qualifier in the CH3 notes.

## Open

### 1. Optional: optimized builds on Grace

The Aurora result should carry over: Grace's native builds have half FMA, and the
stand-alone check of `aarch64_tests.md` (step 3) already showed a half `fmadd` at
`-O2`. On aarch64 even the builds without `ARMV9_GRACE` have float FMA (it is part of
base aarch64), so `default` and `gnu` may contract float arithmetic too.

```bash
qsub -l walltime=2:00:00 -v BUILD_TYPE=Release run_std_half_fp16_grace.sh
```

Expected: all tests pass; in `native` and `gnu-native` the probe values of the Aurora
Release run (`256*256-65000` = 544, libstdc++ `(269,111.5)²` = (59936, 60000)); Kokkos
`(269,111.5)²` = (59936, 60000) in all four.

### 2. Optional: complex functions vs libstdc++

**What.** Compare `Kokkos::complex<half_t>` with `std::complex<std::float16_t>` over the
`ch3acc` inputs, for `exp`, `log`, `sqrt`, the trigonometric functions and their
inverses, and `pow`, under both evaluation methods. Expected: equal up to the last bit,
since both compute in float. Multiplication and division are already settled.

**Status.** The program is not written: `ch3acc` is a CUDA program on the A100 machine.
A host-only version with a `std::complex<std::float16_t>` reference column is needed
first. Low priority.

### 3. B4c on the std backend: the half numeric traits as `constexpr` half values

**What.** B4c (2026-11-24; `complex_half.md`, B4c status) makes the 9 value traits of
`half_t` and `bhalf_t` (`infinity`, `epsilon`, `finite_max`, ...) `constexpr` half values
wherever half is `constexpr`, which includes the std backend. Verified on CUDA, and for
`half_t` with the GCC 12 shim; `bhalf_t` on the std backend needs GCC >= 13.

**Run on Aurora:** `qsub run_std_half_fp16.sh`. The check is compile-time
(`half_value_traits_are_constexpr_halves`, in `Kokkos_CoreUnitTest_Serial1`), so a
successful build means it passes. Expected: all builds `ok (0 warnings)`, and
`*numeric_traits*` 15 (`numeric_traits_half_vs_std_numeric_limits` included, now
comparing `constexpr` half traits with `std::numeric_limits`), `*complex*:*atomic*` 27,
`*half*` 16 tests, 16 passing (the two `*_constexpr_path_matches_runtime` tests run on
the std backend too).

## Future work

| Task | Why not now |
|---|---|
| GCC 13 (only 14.2 and 15.2 tested) | no machine with GCC 13 |
| Locale branch of `half_io`: reading with a decimal comma through the stream's locale | not a priority; no machine has `de_DE`/`fr_FR` UTF-8 locales (a user-built one via `localedef`/`LOCPATH` would do) |
| Report the libstdc++ extraction deviation to GCC: on empty or whitespace-only input `operator>>(_Float16&)` assigns an uninitialized float, where C++23 leaves the value unchanged (GCC 14, 15, master) | external |
| **CH4**: converting constructor from `std::complex<std::float16_t>` (constructor only, D11) | Kokkos's minimum is C++20, and `std::float16_t` needs C++23 (and GCC >= 13); pick up once C++23 is the minimum |

Not listed here because they don't depend on C++23: HIP and SYCL validation, B4b on
HIP/SYCL, the pre-Ampere bf16 gap, CH6.

## Scripts

| Script | Machine | Purpose |
|---|---|---|
| `run_std_half_fp16.sh` | Aurora (PBS header), any x86-64 or aarch64 | four builds (`default`, `native`, `gnu`, `gnu-native`), probes, tests; settings `CONFIGS`, `BUILD_TYPE`, `ENABLE_OPENMP`, `FULL`, `ARCH`, `CXX` |
| `run_std_half_fp16_grace.sh` | Grace | PBS header, modules and binutils for Grace, then `run_std_half_fp16.sh` |
| `run_std_half_gcc13.sh` | Grace | the first std-backend run (job 4260499), default target only |
| `run_h18_std_half.sh` | Aurora | H18 link check |
| `run_bf16_warning_probe.sh` | Aurora | the `__bfloat16` warning reproducer |
| `aarch64_tests.md` | Grace | step-by-step native-half checks |
