# SYCL backend: done, open and future tasks

Status on 2026-11-25 of the tasks on the SYCL backend, where `half_t` and `bhalf_t` wrap
`sycl::half` and `sycl::ext::oneapi::bfloat16`. The full record is in `complex_half.md`
(§8.4, and §8.5 for the task table); this file is the summary. For the other backends see
`cuda-tests.md` and `c++23-tests.md`.

**Where to test:** Aurora. Its GPUs are Intel Data Center GPU Max (Ponte Vecchio),
which have the FP64 that these tests need (`Kokkos_ARCH_INTEL_PVC`). The plan's §8.4
and §8.5 were written before Aurora was available and say "an Intel GPU with FP64".

**Aurora environment:** `icpx` (oneAPI) is available by default; only
`module load cmake` is needed. Runs over 1 hour go to the `capacity` queue.

## Done

Nothing on the SYCL backend has run yet. Done so far, on this machine:

| What | How |
|---|---|
| The premise of B4b-SYCL, compile-time only | `tmp/claude_builds/complex_a6/b4sycl.cpp`, `icpx -fsycl -fsyntax-only` (DPC++ 2024.0), host and `-fsycl-device-only` passes. `constexpr sycl::half(1.5f) + sycl::half(2.5f) == 4` holds, so `sycl::half` is natively `constexpr`. `constexpr bfloat16(1.5f)` fails: bfloat16 is not. `__builtin_bit_cast` to and from both types is a constant expression |
| The half code paths no longer use `double` on the device (CH3) | by construction: the half `<numbers>` constants are instantiated only in discarded branches |

## Open: validate the committed work on SYCL

No code changes; one Aurora job:

```bash
qsub run_sycl_aurora.sh
```

`run_sycl_aurora.sh` (PBS, `capacity` queue, 2 hours, `-A kokkos_math`):
- loads cmake, uses `icpx`, prints `sycl-ls`;
- configures `-DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON
  -DKokkos_ARCH_INTEL_PVC=ON`, tests on, Debug by default (`BUILD_TYPE`), in
  `build_sycl_pvc`;
- prints the half macros as the tests see them (`KOKKOS_HAS_NATIVE_HALF_TYPE`,
  `..._BHALF_...`, `KOKKOS_IMPL_HALF_IS_CONSTEXPR`, `*_T_IS_FLOAT`);
- builds `Serial_SmokeTest`, `Serial1`, `SYCL_SmokeTest`, `SYCL1B` and `Default`. The complex
  tests are in the smoke tests, the numeric traits in group 1B (`SYCL1B`, `Serial1`),
  the half tests in `Default`;
- runs the filters below and prints a summary. Logs: `sycl_logs.<job id>/`.

`FULL=ON` also builds and runs every core test executable (part 2; needs more time:
`qsub -l walltime=6:00:00 -v FULL=ON run_sycl_aurora.sh`).

### 1. The complex, half and numeric-trait tests, Serial and SYCL

| Filter | Executables | Expected |
|---|---|---|
| `*complex*:*atomic*` | `Serial_SmokeTest`, `SYCL_SmokeTest` | 27 pass, as on CUDA. They include A8's interop tests: `complex_std_interop_deprecated` also runs for the native `half_t`/`bhalf_t` |
| `*numeric_traits*` | `Serial1`, `SYCL1B` | 14 pass, 1 skip (the `std::numeric_limits` comparison is std-backend only) |
| `*half*` | `Default` (default execution space = SYCL) | 16 tests, 12 pass. 4 skip: `half_io_vs_std` and `bhalf_io_vs_std` (std backend only), and `half_constexpr_path_matches_runtime` and `bhalf_...` (half is not `constexpr` on SYCL yet, B4b-SYCL) |

Things to check that are specific to SYCL (§8.4):
- **`KOKKOS_HAS_NATIVE_BHALF_TYPE`:** defined only with
  `<sycl/ext/oneapi/bfloat16.hpp>` (`Kokkos_Setup_SYCL.hpp:32-34`). Without it `bhalf_t`
  is `float`, and the bhalf tests reduce to float. The job should print it, as the
  std-backend scripts print the half macros.
- **FP64:** the double instantiations of every complex test, and
  `complex_half_conversions`, use `double` in kernels. On a GPU without FP64 they fail
  for reasons unrelated to half; Aurora's Max GPUs have FP64.
- **Device math:** `Kokkos_MathematicalFunctions.hpp` calls `sycl::` functions; CH3's
  half complex functions rely on the float ones there.
- **`Kokkos::bit_cast` is `sycl::bit_cast`** (`Kokkos_BitManipulation.hpp:319-327`), used
  by `half_same`, `TestHalfIO`'s round trip, and the cast and constexpr-path tests.
- **Declared casts** (`*_cast_overloads`, the H18 pattern): on SYCL the casts come from
  `Kokkos_SYCL_Half_Conversion.hpp`, which no test run has compiled yet.
- **B4c:** on SYCL the numeric traits stay bit patterns (`BitComparisonWrapper`),
  because half is not `constexpr` there. The tests confirm nothing else depends on that.

### 2. Optional: the complete core test suite on SYCL

As done for CUDA (`sub_full_A100.sh`) and the std backend (`FULL=ON`): every core test
executable, without a filter, on Serial and SYCL: `qsub -l walltime=6:00:00 -v FULL=ON
run_sycl_aurora.sh` (the walltime is a guess). Expect Kokkos's own skips; compare
any failure with a run of the branch's base commit `a360acf37` before suspecting the
branch.

## Future work

| Task | Notes |
|---|---|
| **B4b-SYCL:** `constexpr` `half_t`/`bhalf_t` on SYCL | First recheck the premise with Aurora's oneAPI version (`b4sycl.cpp`). Then as B4b-CUDA: `KOKKOS_IMPL_HALF_CONSTEXPR`, `KOKKOS_IMPL_HALF_IS_CONSTEXPR` and `KOKKOS_IMPL_HALF_CONSTANT_EVALUATED()` also for SYCL. The cast functions in `Kokkos_SYCL_Half_Conversion.hpp` become `constexpr`, with the constant-evaluation branch into `Impl::half_constexpr_from`/`_to` (the software path; needed for bfloat16, harmless for `sycl::half`). The wrapper's native-operator branches already have the fallback (`KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH` is always defined on SYCL). Tests: the compile-time checks of B4a and CH5, and `*_constexpr_path_matches_runtime` against the Intel GPU's conversions |
| **CH5 on SYCL:** `constexpr complex<half_t>` | needs B4b-SYCL; then only the existing `static_assert`s, as on CUDA |
| **B4c on SYCL:** the numeric traits as `constexpr` half values | follows from B4b-SYCL automatically: the traits are half values wherever `KOKKOS_IMPL_HALF_IS_CONSTEXPR` is defined |

Not SYCL, listed elsewhere: HIP (`complex_half.md` §8.3), CUDA (`cuda-tests.md`), the std
backend (`c++23-tests.md`).

## Scripts and tools

| File | Purpose |
|---|---|
| `tmp/claude_builds/complex_a6/b4sycl.cpp` | the B4b-SYCL premise check (compile only) |
| `run_sycl_aurora.sh` | part 1 (and part 2 with `FULL=ON`) on Aurora |
