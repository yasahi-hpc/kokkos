# HIP backend: remaining tasks

Status on 2026-11-28 of the tasks on the HIP backend, where `half_t` and `bhalf_t` wrap
ROCm's `__half` and `__hip_bfloat16` (`HIP/Kokkos_HIP_Half_Impl_Type.hpp`). Nothing has
been built or run with HIP yet. The plan's HIP notes are in `complex_half.md` §8.3 and in
the B4 per-backend table. For the other backends see `cuda-tests.md`, `sycl-tests.md` and
`c++23-tests.md`.

**Where to test:** an AMD GPU (access available as of 2026-11-28). Needed before writing the
job script:
- the scheduler and its job header (account, queue, walltime limits);
- the GPU and its Kokkos architecture, e.g. MI250X → `Kokkos_ARCH_AMD_GFX90A`, MI300A →
  `AMD_GFX942_APU`, MI300X → `AMD_GFX942`;
- the ROCm module(s) and versions;
- which file systems the compute nodes can see (the A100 lesson: `/home` was invisible).

**The ROCm version matters.** `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH` is defined in the device
pass always, but on the host only for HIP ≥ 6.4 (`setup/Kokkos_Setup_HIP.hpp:55-59`). So
the host takes the float path below 6.4 and `__half`'s own operators from 6.4 on. Both
paths are worth a run if both ROCm versions are installed.

## Summary

| # | Task | Kind | Expected |
|---|---|---|---|
| 1 | Validate the committed work: the complex, half and numeric-trait tests on Serial and HIP | required | passes, except possibly the special values and signed zeros (as on SYCL) |
| 2 | Run the software/hardware rounding comparison on HIP too, then fix HIP's double rounding | required (a predicted bug) | `half_t(double)` rounds twice, through float; `bhalf_t` from 64-bit integers maybe too |
| 3 | The complete core test suite on HIP, Release; compare a failure with the base `a360acf37` | required | passes as on the base |
| 4 | B4b-HIP: `constexpr` `half_t`/`bhalf_t` on HIP (with it CH5 and B4c on HIP) | optional, uncertain | try the workaround with the real ROCm headers; else stays future work |
| 5 | Benchmarks: the float path (CH3) and paired `__half2` (CH6) on AMD | optional | the float path costs nothing, as on the A100 |
| 6 | Both ROCm sides of 6.4 (the host full-type switch) | optional | the same results |

## 1. Validate the committed work

Configure `-DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_<GPU>=ON
-DKokkos_ENABLE_TESTS=ON` with `hipcc` or `amdclang++`, Debug first. Then print the half
macros as the tests see them: `KOKKOS_HAS_NATIVE_HALF_TYPE`, `..._BHALF_...`,
`KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH` (host and device pass), `KOKKOS_IMPL_HALF_IS_CONSTEXPR`
(expected 0) and `*_T_IS_FLOAT` (expected `false`). Use the same probe as
`run_sycl_aurora.sh`, with `--cuda-device-only`/`--offload-device-only` for the device pass.

| Filter | Executables | Expected |
|---|---|---|
| `*complex*:*atomic*` | `Serial_SmokeTest`, `HIP_SmokeTest` | 27 pass, as on CUDA and SYCL. It includes A8's interop tests: `complex_std_interop_deprecated` with the native `half_t`/`bhalf_t` |
| `*numeric_traits*` | `Serial1`, `HIP` | 14 pass, 1 skip (the `std::numeric_limits` comparison is std-backend only) |
| `*half*` | `Default` (default execution space = HIP) | 16 tests: 12 pass, 4 skip (the two `*_io_vs_std`, std only; the two `*_constexpr_path_*`, not `constexpr` on HIP). With task 2: 14 pass, 2 skip |

Points specific to HIP:
- **CH1 constants:** `T(infinity_v<T>)` goes through `Kokkos::bit_cast` into HIP's
  union-based `__half` in device code.
- **CH3 on the device:** the float path. `half_same` in `complex_half_compute_in_float`
  (same bits, or both NaN) must hold on AMD hardware, whose NaN payloads may differ.
- **CH2:** `half_t` → `float` is `explicit` on every backend, so the explicit/implicit
  table should match: `tmp/claude_builds/complex_a6/ch2probe.cpp`.
- **Declared casts** (`*_cast_overloads`, the H18 pattern): HIP's casts come from
  `Kokkos_HIP_Half_Conversion.hpp`, which no test run has compiled yet. H17 doesn't apply:
  HIP defines the `bool` casts (`:19`, `:188`).
- **B4c:** the numeric traits stay bit patterns (`BitComparisonWrapper`) on HIP;
  `half_assign_trait_constants` covers assigning them.
- **Lessons from SYCL, to check deliberately:**
  - **Special values:** the three `*_special_values` tests compare with glibc's fixed
    table. A device math library (ROCm's OCML) may differ in the sign of zero or infinity.
    On SYCL, `sin(-0)` and `atan2(-1, inf)` returned `+0` on the device and needed wrappers
    (`Kokkos_MathematicalFunctions.hpp`, `Impl::sycl_math`). The tests report the failing
    function and input.
  - **Floating-point flags:** HIP clang contracts `a*b+c` into FMA on the device by default
    (`-ffp-contract=fast`), as nvcc does. CUDA passes this way, so keep the defaults first.
    Kokkos's `-ffinite-math-only` CI job is covered by the `__FINITE_MATH_ONLY__` guards.
  - **Host conversions:** DPC++'s host `sycl::half(float)` rounded wrongly. HIP's host
    `__float2half` is ROCm's software code; the half tests cover it (`half_conversion`, and
    task 2 exhaustively).

## 2. Double rounding on HIP (predicted, from the SYCL findings)

On SYCL the comparison of the software conversions (`Impl::half_constexpr_from`/`_to`,
round to nearest even, once) with the device found `half_t(double)` rounding twice,
through float (Aurora job 9127596; fixed in `e667e7fbd`). HIP's conversions
(`HIP/Kokkos_HIP_Half_Conversion.hpp`) look the same:
- `cast_to_half(double)`: `__float2half(static_cast<float>(val))`, on the host and the
  device. 2^-25 (1 + 2^-52) gives 0 instead of 2^-24.
- `cast_to_bhalf(double)`: `__hip_bfloat16(val)`. It depends on whether ROCm's bf16 has a
  `double` constructor or converts through float.
- `cast_to_bhalf(long long)`, `(unsigned long long)`: through `double` (`// FIXME_HIP`,
  `:207`, `:213`). That rounds twice above 2^53 near a tie, e.g. 2^62 + 2^54 + 1.
- `cast_to_half` from the integers on the host: through float. That is exact for every
  integer within half's range, so it's fine.
- Since `e667e7fbd` the wrapper's `double` compound assignments go through these casts,
  so they inherit the result.

The test that finds this, `half_constexpr_path_matches_runtime` / `bhalf_...`
(`TestHalfConversion.hpp`), is gated on `KOKKOS_IMPL_HALF_IS_CONSTEXPR` and skips on HIP.
But its run-time comparison needs no `constexpr`:
1. Make the run-time part run wherever `half_t`/`bhalf_t` are not `float`. The
   `half_constexpr_from`/`_to` templates are usable at run time on every backend: they
   read and write the bits with `__builtin_bit_cast`, which is fine at run time even for
   HIP's union. It reports per kind (decode, encode, midpoints, integers) and the first
   failing pattern's bits.
2. Run it on HIP. Expected: midpoint failures for `half_t` (one per non-NaN pattern but
   four, as on SYCL), and integer failures for `bhalf_t` if the `long long` path
   double-rounds.
3. Fix as on SYCL. Use ROCm's correctly rounded conversions if it has them
   (`__double2half`; bf16 from `double`, `long long`); otherwise use the software
   conversion. Then rerun task 1.

## 3. The complete core test suite on HIP

Every core test executable without a filter, on Serial and HIP, in **Release** (the SYCL
Debug run was too large to link and hit unrelated backend bugs; see `sycl-tests.md`,
"Unrelated bugs"). Expected: as the base `a360acf37` with the same settings, plus the
branch's 49 new tests. Run the base too if anything fails; a failure in both is not the
branch's. Keep the SYCL script's safeguards: line-buffered output, per-run timeouts,
results written as they finish, and a disk-space check.

## 4. B4b-HIP (optional)

`constexpr` half on HIP needs a way to read `__half`'s bits during constant evaluation. ROCm
keeps the value in a `protected` anonymous union (`_Float16 data; unsigned short __x;`,
`amd_hip_fp16.h`; bf16 likewise). Clang rejects `__builtin_bit_cast` of a union in a
constant expression, and `operator __half_raw()` is not `constexpr`.
- **Workaround to try:** a derived `struct : __half { constexpr std::uint16_t bits()
  const { return __x; } }`. It is valid only while `__x` is the active member, so every
  compile-time value must come from the raw constructor `__half(__half_raw)`. On a mock
  with HIP's layout (clang 17) this worked; reading after the defaulted constructor did
  not. Check with the real ROCm headers and hipcc first: a compile-only probe, then
  `half_constexpr_checks` (B4a's checks).
- **If it works:** as B4b-SYCL. `KOKKOS_IMPL_HALF_CONSTEXPR`/`IS_CONSTEXPR` on HIP,
  `KOKKOS_IMPL_HALF_SOFTWARE_PATH()` = `__builtin_is_constant_evaluated()`, `constexpr`
  casts with the software branch. CH5 (`constexpr complex<half_t>`) and B4c (traits as half
  values) follow automatically.
- **If not:** it stays future work. It is better solved upstream: a `constexpr` bits
  accessor or `constexpr` conversions in ROCm would make it as simple as SYCL.

## 5. Benchmarks (optional)

- **CH3:** the float path for half complex arithmetic. Port
  `tmp/claude_builds/complex_a6/ch3bench.cpp` (Kokkos only) and compare with the A100
  result: no cost.
- **CH6:** paired `__half2` arithmetic. `ch6bench.cpp` uses CUDA's `__half2` intrinsics;
  HIP has the same names (`hip_fp16.h`). CH6 is future work unless instruction-bound code
  gains (`cuda-tests.md`).

## 6. Both sides of ROCm 6.4 (optional)

If both versions are installed: task 1 with ROCm < 6.4 (host: float path) and ≥ 6.4 (host:
`__half`'s operators). The results should be identical; a difference points to ROCm's host
half arithmetic.

## Future work

- B4b-HIP, if task 4's workaround fails (see there).
- Report the double rounding to ROCm if their conversion functions are the cause (task 2).

## Files

| File | Purpose |
|---|---|
| `core/src/HIP/Kokkos_HIP_Half_Impl_Type.hpp`, `..._Conversion.hpp`, `..._MathematicalFunctions.hpp` | HIP's half types, casts and math |
| `core/src/setup/Kokkos_Setup_HIP.hpp:52-59` | native half/bf16, the host full-type switch at HIP 6.4 |
| `run_sycl_aurora.sh` | template for the HIP job script (probe, filters, safeguards) |
| `tmp/claude_builds/complex_a6/ch2probe.cpp`, `ch3bench.cpp`, `ch6bench.cpp` | CH2 probe, CH3/CH6 benchmarks |
