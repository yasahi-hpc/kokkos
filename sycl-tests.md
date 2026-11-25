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

Part 1 is done: **Aurora job 9127366** (2026-11-25, `efa57b218`, icpx 2026.1, Intel Data
Center GPU Max 1550, Debug, `-fp-model=precise`). All builds succeed; the half macros are as
expected (`NATIVE_HALF=1`, `NATIVE_BHALF=1`, `HALF_IS_CONSTEXPR=0`,
`HALF_T_IS_FLOAT=false`, `BHALF_T_IS_FLOAT=false`); the only warnings are Intel's 10,715
"generic address space" messages.

| Filter | Serial | SYCL |
|---|---|---|
| `*complex*:*atomic*` | 27 pass | 27 pass |
| `*numeric_traits*` | 14 pass, 1 skip | 14 pass, 1 skip |
| `*half*` (`Default`) | — | 12 pass, 4 skip (the two std-only `*_io_vs_std`, the two `*_constexpr_path_*` until B4b-SYCL) |

It took four runs. They found and fixed, in Kokkos: H17 on SYCL (the `bool` casts), DPC++'s
host `sycl::half(float)` rounding (Kokkos rounds in software on the SYCL host), and the Intel
GPU's signed zeros of `sin` and `atan2`. In the tests: the special-value reference (now glibc's
fixed table, as the CUDA and std backends) and finite-math guards. In the job script:
`-fp-model=precise` and the probe. Details in the run sections below.

Done before, on this machine:

| What | How |
|---|---|
| The premise of B4b-SYCL, compile-time only | `tmp/claude_builds/complex_a6/b4sycl.cpp`, `icpx -fsycl -fsyntax-only` (DPC++ 2024.0), host and `-fsycl-device-only` passes. `constexpr sycl::half(1.5f) + sycl::half(2.5f) == 4` holds, so `sycl::half` is natively `constexpr`. `constexpr bfloat16(1.5f)` fails: bfloat16 is not. `__builtin_bit_cast` to and from both types is a constant expression |
| The half code paths no longer use `double` on the device (CH3) | by construction: the half `<numbers>` constants are instantiated only in discarded branches |

### First run: job 9126968 (2026-11-25, `715baddcc`)

Aurora, icpx 2026.1.0, Intel Data Center GPU Max 1550, Debug, default compiler flags.

| Result | Cause | Fix |
|---|---|---|
| `*numeric_traits*`: 14 pass on `Serial1` and `SYCL1B` (1 std-only skip) | — | — |
| `Default` (the half tests) did not build: "SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute", at the bool and cast-overload tests | **H17 on SYCL**: the wrapper header declares `cast_to_half(bool)`, `cast_from_half<bool>`, `cast_to_bhalf(bool)` and `cast_from_bhalf<bool>`, but `Kokkos_SYCL_Half_Conversion.hpp` never defined them (as CUDA before H17). Pre-existing; caught by `half_bool_conversion` and `*_cast_overloads` | the four casts defined, through float as on CUDA (2026-11-25) |
| `*complex*:*atomic*`: 4 failures on Serial, 5 on SYCL, all with infinities, NaNs or a signed zero (`(inf,nan) * (1,0)`, `arg(inf,-1)` gave `+0`) | icpx's default `-fp-model=fast` does not honour them; Kokkos's CI builds with `-fp-model=precise` whenever it uses icpx | `run_sycl_aurora.sh` passes `-fp-model=precise` (setting `CXX_FLAGS`) |
| The half-macro probe printed nothing | `icpx -fsycl -E` fails (several passes) | the probe preprocesses with `-fsycl-device-only` and keeps its output |
| About 4,500 "additional control flow due to presence of generic address space operations" warnings per SYCL test target | Intel's GPU backend, not this work | — |

Found with it: one Kokkos CI job builds with clang and `-ffinite-math-only`, and Kokkos's math
tests guard their NaN/infinity checks with `#if !__FINITE_MATH_ONLY__`. The branch's complex
special-value tests had no such guard. They have it now (`complex_half.md` §8.2). It does not
change the icpx run: icpx's fast model leaves `__FINITE_MATH_ONLY__` at 0.

### Second run: job 9127242 (2026-11-25, `37f5148b7`, `-fp-model=precise`)

`Default` builds and the bool and cast-overload tests pass (H17 on SYCL fixed);
`complex_multiplication` and `complex_division` pass (`-fp-model=precise`);
`*numeric_traits*` 14 on `Serial1` and `SYCL1B`. Still failing, and the fixes
(2026-11-25), aligned with the CUDA and std backends:

| Result | Cause | Fix |
|---|---|---|
| 3 special-value tests on Serial and SYCL (`exp`/`sqrt`/`log`, hyperbolic, inverse) | the tests compared with `std::complex`, which under icpx uses Intel's math library: it gives other signs where Annex G leaves them unspecified (`exp(-inf,-inf)`, `cosh(0,inf)`, `acos(inf,nan)`) and C99's NaN where C11 gives a zero (`tanh(+0+i inf) = +0+iNaN`, `acosh(+-0+iNaN) = NaN+i pi/2`). Kokkos's results were the C11 ones | the expected kind and sign of every special-value result are fixed to glibc's, the reference the CUDA and std backends were verified against (a 15-function table in `TestComplex.hpp`, generated by `tmp/claude_builds/complex_a6/gen_complex_special_ref.cpp`); `std::complex` still provides the finite values |
| SYCL device: `complex_arg_proj` (`arg(inf,-1)` gave `+0`), signed zeros of `log(inf,-1)` and `cosh(x, -0)` | the Intel GPU's `sycl::atan2(-1, inf)` returns `+0` and `sycl::sin(-0)` returns `+0`, where C and IEEE 754 give `-0` | `Kokkos::sin` and `Kokkos::atan2` on SYCL restore the signs (`sin(+-0) = +-0`; `atan2(y, x)` has the sign of `y`), through the namespace `Kokkos::Impl::sycl_math`; every complex function that uses them follows |
| `half_io`: 1974 values do not survive the text round trip | DPC++'s host `sycl::half(float)` rounds wrongly below 2^-13: it truncates, and ties do not go to even (measured with oneAPI 2024.0: about 2,050 neighbours and 1024 ties wrong, in the subnormals and the smallest normal binade). An upstream bug, worth reporting to Intel | on the SYCL host, `cast_to_half`/`cast_to_bhalf` from `float`/`double` round with Kokkos's software conversion (`Impl::half_constexpr_from`), and the wrapper's operators compute in float (`KOKKOS_IMPL_HALF_SOFTWARE_PATH()`, true in the SYCL host pass); the device keeps the hardware conversions |
| The half-macro probe failed again | a compile definition holds a quoted string with spaces, which the unquoted expansion split | the flags are expanded with `eval` |

### Third run: job 9127338 (2026-11-25, `b2dfe3313`)

The half-macro probe works now: `NATIVE_HALF=1`, `NATIVE_BHALF=1`, `HALF_IS_CONSTEXPR=0`,
`HALF_FULL_TYPE=1`, `HALF_T_IS_FLOAT=false`, `BHALF_T_IS_FLOAT=false`, as expected.
Every build failed in `kokkoscore`: the `sycl_math` namespace was declared as
`namespace Kokkos::Impl::sycl_math` at a place already inside `namespace Kokkos`, so it
became `Kokkos::Kokkos::Impl::sycl_math`, and that inner `Kokkos` also hid
`Kokkos::Impl::promote_2_t` (my error; the local check had put the namespace at global
scope). Fixed: `namespace Impl::sycl_math` inside `namespace Kokkos`. Checked with the
local icpx 2024.0: the real `Kokkos_MathematicalFunctions.hpp` compiles for SYCL host and
device (`-DKOKKOS_ENABLE_SYCL`, `-fsyntax-only`), and every SYCL math function goes
through `::Kokkos::Impl::sycl_math`. The SYCL half headers could not be checked locally
(they need the SYCL backend's configuration); they compile for the first time with the
fourth run.

## Open

### 0. Verify B4b-SYCL and CH5 on SYCL (implemented 2026-11-25)

`constexpr` `half_t`/`bhalf_t` on SYCL, and with it CH5 (`constexpr complex<half_t>`) and
B4c (the numeric traits as half values). As B4b-CUDA:
- `KOKKOS_IMPL_HALF_CONSTEXPR` and `KOKKOS_IMPL_HALF_IS_CONSTEXPR` are now defined on SYCL;
  `KOKKOS_IMPL_HALF_SOFTWARE_PATH()` is `__builtin_is_constant_evaluated()` in the device
  pass and stays `true` in the host pass (the host rounding fix).
- Every cast in `Kokkos_SYCL_Half_Conversion.hpp` is `constexpr`, and each non-identity one
  starts with the software branch (`Impl::half_constexpr_from`/`_to`). The wrapper needed
  no change.
- Compile-checked here with icpx 2024.0 in both passes, with a fake SYCL configuration
  (`claude_builds/complex/fakesycl/t.cpp`): `static_assert`s on both types pass, including
  a `constexpr` product inside a kernel, and a planted false assertion fails. Not run: no
  device here has fp16.

Rerun part 1 unchanged: `qsub run_sycl_aurora.sh`. Expected now:

| Check | Expected (was) |
|---|---|
| probe | `HALF_IS_CONSTEXPR=1` (0); the other macros unchanged |
| build | succeeds. The B4a, CH5 and B4c `static_assert`s (`TestHalfConversion.hpp`, `TestComplex.hpp`, `TestNumericTraits.hpp`) compile in both passes |
| `*complex*:*atomic*` | 27 pass on Serial and SYCL (unchanged; CH5 is compile-time only) |
| `*numeric_traits*` | 14 pass, 1 skip (unchanged) |
| `*half*` | 14 pass, 2 skip (12 + 4): `half_constexpr_path_matches_runtime` and `bhalf_...` now run. Their device part compares the software path with the Intel GPU's conversions bit for bit (every pattern, the midpoints, 64-bit integers); on the host both sides use the software path |

If a `*_constexpr_path_*` test fails on the device, the GPU's conversion disagrees with
round-to-nearest-even somewhere (the log counts the mismatches); compare with the host
rounding bug, which was the same kind of finding.

### Part 1, the job script and what it checks (done above)

One Aurora job:

```bash
qsub run_sycl_aurora.sh
```

`run_sycl_aurora.sh` (PBS, `capacity` queue, 2 hours, `-A kokkos_math`):
- loads cmake, uses `icpx`, prints `sycl-ls`;
- configures `-DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON
  -DKokkos_ARCH_INTEL_PVC=ON`, tests on, Debug by default (`BUILD_TYPE`), with
  `CMAKE_CXX_FLAGS=-fp-model=precise` (as Kokkos's CI; setting `CXX_FLAGS`), in
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
| `*half*` | `Default` (default execution space = SYCL) | 16 tests, 12 pass. 4 skip: `half_io_vs_std` and `bhalf_io_vs_std` (std backend only), and `half_constexpr_path_matches_runtime` and `bhalf_...` (until B4b-SYCL; with it 14 pass, 2 skip, see 0.) |

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
- **B4c:** before B4b-SYCL the numeric traits stayed bit patterns
  (`BitComparisonWrapper`) on SYCL; with it they are half values (see 0.).

### 2. Optional: the complete core test suite on SYCL

As done for CUDA (`sub_full_A100.sh`) and the std backend (`FULL=ON`): every core test
executable, without a filter, on Serial and SYCL: `qsub -l walltime=6:00:00 -v FULL=ON
run_sycl_aurora.sh` (the walltime is a guess). Expect Kokkos's own skips; compare
any failure with a run of the branch's base commit `a360acf37` before suspecting the
branch.

## Future work

| Task | Notes |
|---|---|
| (B4b-SYCL, CH5 and B4c on SYCL: implemented 2026-11-25, moved to Open, 0.) | |

Not SYCL, listed elsewhere: HIP (`complex_half.md` §8.3), CUDA (`cuda-tests.md`), the std
backend (`c++23-tests.md`).

## Scripts and tools

| File | Purpose |
|---|---|
| `tmp/claude_builds/complex_a6/b4sycl.cpp` | the B4b-SYCL premise check (compile only) |
| `/home/i18048/claude_builds/complex/fakesycl/` | the B4b-SYCL compile check: `t.cpp`, and `inc/` (the host config with `KOKKOS_ENABLE_SYCL`, `Kokkos_Setup_SYCL.hpp` in the setup header); `icpx -std=c++20 -fp-model=precise -fsycl [-fsycl-device-only] -fsyntax-only -Iinc -I<kokkos>/core/src t.cpp` after `module purge; module load intel/2024.0.0` |
| `run_sycl_aurora.sh` | part 1 (and part 2 with `FULL=ON`) on Aurora |
