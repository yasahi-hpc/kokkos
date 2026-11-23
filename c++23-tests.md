# C++23 (std half backend): open and future tasks

Status on 2026-11-23 of the tasks that need GCC ≥ 13 in C++23 mode, where `half_t` and
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

Every one of these runs was a **Debug** build (`-O0`).

## Open

### 1. Optimized builds (Release, `-O3`)

**Why.** With optimization GCC contracts `a * b + c` into a fused multiply-add (one
rounding instead of two), where the CPU has FMA and contraction is allowed. GCC's manual
gives `-ffp-contract=off` as the default only for C in a strict ISO mode, and `fast`
otherwise, so C++ probably contracts in both dialects (not checked; the run will show).
On Aurora only the native builds have FMA:

| build | dialect | FMA available | contraction possible |
|---|---|---|---|
| `default` | `-std=c++23` | no (x86-64 baseline) | no |
| **`native`** | `-std=c++23` | yes (Sapphire Rapids) | **yes** |
| `gnu` | `-std=gnu++23` | no | no |
| **`gnu-native`** | `-std=gnu++23` | yes | **yes** |

Grace already showed the effect: at `-O2`, the stand-alone check of `aarch64_tests.md`
fused `a * a - c` into a half `fmadd` and printed 544 instead of inf. In Kokkos it could
change the result of complex multiplication, and break tests that expect bit-exact
results: `complex_half_compute_in_float` (half path vs float path) and
`complex_multiplication` (vs the textbook formula). Either outcome is worth knowing.

**Run on Aurora**, from the Kokkos checkout:

```bash
qsub -q capacity -l walltime=2:00:00 -v BUILD_TYPE=Release run_std_half_fp16.sh
```

- **What it does:** builds `build_std_half_<config>_release` for the four configs,
  so the Debug builds are kept. `Kokkos_ENABLE_DEBUG` is OFF for this build type.
- **Queue:** `capacity`, because four optimized builds may not fit in the debug queue's
  hour.
- **Optional, longer:** add the complete test executables and OpenMP:

  ```bash
  qsub -q capacity -l walltime=3:00:00 \
       -v 'BUILD_TYPE=Release,ENABLE_OPENMP=ON,FULL=ON,CONFIGS=default gnu-native' \
       run_std_half_fp16.sh
  ```

**What to look for** in `std_half_fp16_logs.<job>/job.out`:

| | expected |
|---|---|
| `optimization:` line of each build | `-O3` |
| probe values in `default`, `gnu` | as in the Debug runs (job 9117216) |
| probe values in `native`, `gnu-native` | may differ: `std_float16_t 256*256-65000` can print 544 instead of inf, if GCC fuses it |
| `kokkos_complex (269,111.5)^2` | (59936, 60000) everywhere; a different value means contraction changed Kokkos's result |
| tests | all pass. A failure in `complex_half_compute_in_float` or `complex_multiplication` in a native build would point to contraction |

Then on Grace the same with `qsub -v BUILD_TYPE=Release run_std_half_fp16_grace.sh`.
There every build can contract float arithmetic (FMA is part of base aarch64), and the
native builds half arithmetic too: the stand-alone check already showed a half
`fmadd`. Grace's walltime limit for the 1-hour header may need raising
(`qsub -l walltime=2:00:00 …`).

### 2. Optional: complex functions vs libstdc++

**What.** Compare `Kokkos::complex<half_t>` with `std::complex<std::float16_t>` over the
`ch3acc` inputs, for `exp`, `log`, `sqrt`, the trigonometric functions and their
inverses, and `pow`, under both evaluation methods. Expected: equal up to the last bit,
since both compute in float. Multiplication and division are already settled.

**Status.** The program is not written: `ch3acc` is a CUDA program on the A100 machine.
A host-only version with a `std::complex<std::float16_t>` reference column is needed
first. Low priority.

## Future work

| Task | Why not now |
|---|---|
| GCC 13 (only 14.2 and 15.2 tested) | no machine with GCC 13 |
| Locale branch of `half_io`: reading with a decimal comma through the stream's locale | not a priority; no machine has `de_DE`/`fr_FR` UTF-8 locales (a user-built one via `localedef`/`LOCPATH` would do) |
| Report the libstdc++ extraction deviation to GCC: on empty or whitespace-only input `operator>>(_Float16&)` assigns an uninitialized float, where C++23 leaves the value unchanged (GCC 14, 15, master) | external |
| **B4a**: `constexpr` half on the std backend | library change (deferred with B4) |
| **CH5** (std part): enable the half `static_assert`s in `TestComplex.hpp` | needs B4a |
| **CH4**: converting constructor from `std::complex<std::float16_t>` (constructor only, D11) | library change |
| **A8**, half part: the deprecated-code test with `half_t`/`bhalf_t` on libstdc++, and a build with `-DKokkos_ENABLE_DEPRECATED_CODE_5=OFF` | needs A8, the deprecation of the `std::complex` interop (D11); A8 itself is developed and tested on the A100 machine |

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
