# Plan: `Kokkos::complex<half_t>` aligned with `std::complex<std::float16_t>`

Branch `refactor-complex` at `a360acf37`. Line numbers refer to that tree (before any
Part A change). **Status (2026-11-20): A1 implemented in the working tree, not committed;
verified on A100 (job 10375120: 10/10 `*complex*` tests pass on Serial and Cuda); see
the A1 status note in §5. A1 is committed as `6a0a264a8`. A2 was revised to delegate to
`std::complex` (job 10375888, which tested the first version, was cancelled) and is
verified (job 10376616). §3b (`cuda::std::complex` survey) added A5, A6,
A3b and CH6, and the gaps C21/C22.** The earlier test-side investigation
(on branch `half-complex`) is in `half-complex.md`. This document covers the library
side.

Goal: `Kokkos::complex<Kokkos::Experimental::half_t>` (and `bhalf_t`) should behave
like `std::complex<std::float16_t>` (`std::bfloat16_t`), as specified by C++23
[complex.numbers] + P1467 (extended floating-point types), wherever that is
possible on device. Where Kokkos deliberately differs (device support, mixed
precision), the difference should be documented.

Kokkos requires C++20 (`cmake/kokkos_test_cxx_std.cmake:89`). So `explicit(bool)`,
`std::bit_cast`, `std::is_constant_evaluated()` and `requires` can be used
throughout.

Status legend: **[bug]** = defect independent of half support; **[gap]** =
missing for half; **[align]** = differs from `std::complex<std::float16_t>`.

---

## 1. Reference: what `std::complex<std::float16_t>` is

| Aspect | C++23 `std::complex<std::float16_t>` |
|---|---|
| Allowed `T` | any cv-unqualified floating-point type, including extended ones ([complex.numbers.general]/2) |
| Value ctor | `constexpr complex(const T& re = T(), const T& im = T())` |
| Converting ctor | `template<class X> constexpr explicit(rank(T) < rank(X)) complex(const complex<X>&)`: implicit only when not narrowing (P1467) |
| Scalar → `T` | implicit narrowing into an extended type is **ill-formed** (`std::float16_t x = 1.0;` is rejected by GCC 13), so `complex<std::float16_t> z(1.5, 2.5)` does not compile; `1.5f16` is required |
| Members | `real()`, `imag()` get/set, `=`, `+= -= *= /=` with `T` and with `complex<X>`; all `constexpr` (C++20) |
| Non-members | unary `+ -`; binary `+ - * /` for `(complex,complex)`, `(complex,T)`, `(T,complex)`, **same `T` only**; `==` (complex/complex, complex/T; `!=` rewritten); all `constexpr` |
| Values | `real imag abs arg norm conj proj polar` |
| Transcendentals | `exp log log10 pow sqrt sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh` |
| Scalar overloads | `real imag arg norm conj proj` for arithmetic **and floating-point** `T`: integers → `double`, FP types (incl. extended) keep their type |
| `pow` mixed | `pow(complex<T>, U)` etc. promote to the common FP type (`complex<std::float16_t>` with `double` → `complex<double>`) |
| I/O | `<<` writes `(re,im)` using the stream's flags; `>>` accepts `r`, `(r)`, `(r,i)` |
| Layout | array-oriented access: `reinterpret_cast<T(&)[2]>(z)` |
| Tuple protocol | `get<I>`, `tuple_size`, `tuple_element` (C++26, P2819) |

Not verified on this system: libstdc++ ≥ 13 behaviour for `std::complex<std::float16_t>`
(it needs GCC ≥ 13; only gcc ≤ 12.2 is installed). Where this plan says "libstdc++
does X", check it on a GCC 13+ machine before relying on it.

---

## 2. `Kokkos::complex` today (`core/src/Kokkos_Complex.hpp`)

### 2.1 Gap list

| # | Where | Today | vs. std / half | Kind |
|---|---|---|---|---|
| C1 | `:35-38` | `static_assert(std::is_floating_point_v<RealType> ...)` | rejects `half_t`/`bhalf_t` (class types) | gap |
| C2 | includes `:10-18` | does not include `<Kokkos_Half.hpp>` | unqualified `fabs`, `hypot`, `atan2`, ... in templates are found by ordinary lookup at definition, or by ADL in `Kokkos::Experimental::Impl` (the wrapper's namespace) only. If `Kokkos_Complex.hpp` is included before `Kokkos_Half.hpp`, `Kokkos::fabs(half_t)` is not found. Works today only because `Kokkos_Core.hpp:34` includes Half before Complex (`:54`) | gap |
| C3 | `:208-232` | `operator/=(const std::complex<RealType>&)` reads `y.re_`/`y.im_` | `std::complex` has no such members; compiles only because it is never instantiated | **bug** |
| C4 | `:829` | `const RealType1 y_scaled_abs` in `operator/(complex<R1>, complex<R2>)` | should be `common_real_type`; `complex<float>/complex<double>` loses precision (measured: 3.1e-8 relative error vs 2.2e-16) | **bug** |
| C5 | `:824` (also `:193`, `:219` in `/=`) | upfront `if (s == 0.0)` branch, `s = \|c\| + \|d\|`, then divide by `s` | The branch is what makes `z / 0` give `inf` instead of `NaN`. Kokkos also gets infinite denominators wrong, and `s` itself overflows when `\|c\|+\|d\|` > max: `(1e308,1e308)/(1e308,1e308)` is NaN, not `(1,0)`. See D10 | **bug** + align |
| C6 | `:52-59` | converting ctor implicit whenever `is_convertible` | std: `explicit` when narrowing (`complex<half>` ← `complex<float>`) | align |
| C7 | `:89-95` | separate `(val)` and `(re, im)` ctors; no default args | std has `(re = T(), im = T())`. Equivalent except `complex<T>{}` paths; keep | — |
| C8 | `:421-528`, `:800-845` | non-member `+ - * /` are **not** `constexpr` | std: `constexpr` | align |
| C9 | same | mixed-type operators (`complex<R1>` op `complex<R2>` / `R2`) returning `complex<common_type_t>` | std: same `T` only. Kokkos extension, keep. With `half_t` the result follows `std::common_type` specialisations in `Kokkos_Half_CommonType.hpp` (`half`+`double` → `double`) | keep, document |
| C10 | `:530-554`, `:636-653` | scalar `real imag conj norm` use `Impl::promote_t`, which is undefined for `half_t` | `conj(half_t)` / `norm(half_t)` do not compile; `real`/`imag(half_t)` exist only via `Kokkos_Half_MathematicalFunctions.hpp:558-565` | gap |
| C11 | — | no `arg`, no `proj` (complex or scalar) | std has both | gap (all types) |
| C12 | `:588-607` | mixed `pow` gated by `std::is_arithmetic_v` / `Impl::promote_2_t` | excludes half | gap |
| C13 | `:556-561` | `polar(r, theta)` with `KOKKOS_EXPECTS(r >= 0)` | std: UB for negative/NaN `r`; fine | — |
| C14 | `:483-490`, `:165-172`, `:182-206`, `:810-836` | multiply/divide computed in `RealType` | for half, products overflow at 65504 and lose ~3 bits; see D5 | align/accuracy |
| C15 | `:61-85`, `:105-113` | interop with `std::complex<RealType>` only (D11, 2026-11-22: all of it except the converting ctor is to be deprecated, A8) | for half this is `std::complex<half_t>` (a wrapper class): formally unspecified, works with libstdc++ (verified on CUDA, `half-complex.md` §3.2, and with GCC 15 where `half_t` wraps `std::float16_t`, job 4260499). Does **not** interoperate with `std::complex<std::float16_t>` | align |
| C16 | `:847-860` | `<<`/`>>` go through `std::complex<RealType>`; `>>` assigns even on failure (comment `:858` says otherwise) | needs `std::complex<half_t>` (see C15); assign-on-failure is a **bug** | bug + align |
| C17 | `:862-866` | `reduction_identity<complex<T>>` returns `0`/`1` (int) | works for half via `half_t(int)` | — |
| C18 | `:31-33` | `alignas(2*sizeof(RealType))` under `KOKKOS_ENABLE_COMPLEX_ALIGN` | `complex<half_t>`: align 4 vs `std::complex<std::float16_t>`: 2. Same kind of difference already exists for float/double; keep | — |
| C19 | `:657-789` | transcendentals use `RealType(0.5)`, `RealType(2.0)`, ... | with `half_t` these are fine (exact) | — |
| C20 | `:22-28` | doc says only `float`, `double`, `long double` | update | doc |
| C21 | `operator*` (`:483-490` in the original numbering), `*=` (`:165-172`) | naive `(ac−bd, ad+bc)`, no Annex G recovery: `(inf,nan)*(1,0)` = (NaN,NaN) | libstdc++ (`__muldc3`), libc++ and libcu++ (§3b) give (inf,NaN) | align (found 2026-11-20 via §3b; verified with a host prototype) |
| C22 | `exp`, `sqrt`, `log`, inverse hyperbolics, … (`:655-789` in the original numbering) | naive formulas: `exp(inf,0)` = (inf,NaN), `exp(-inf,inf)` = (NaN,NaN), `sqrt(1,inf)` = (inf,NaN) | libstdc++ gives (inf,0), (0,0), (inf,inf); C99 Annex G G.6 tables | align (found 2026-11-20 via §3b) |
| C23 | `sqrt` (`i < 0` tests), inherited by `asin acos asinh acosh` | finite inputs on a branch cut with imaginary part −0 land on the wrong side: `sqrt(−1 − 0i) = (0, +1)`, `asin(2.5 + 0i)` has the wrong imaginary sign | libstdc++/glibc: (0, −1); C99 G.6 requires continuity from the side of the signed zero | **bug** (found 2026-11-21 by the A6 grid; fixed in `sqrt` by A6 batch 1) |
| C24 | `tanh` (`sinh(z)/cosh(z)`), inherited by `tan` | finite results inaccurate componentwise: `tanh(20+i)` imag 5.2e-17 instead of 7.7e-18 (up to 9e33 eps for x ∈ [5,40]); NaN for \|x\| ≳ 710 | glibc: Kahan's formula, ≤ 3.3 eps componentwise | **bug** (found 2026-11-21 by A6 batch 2; fixed there) |
| C25 | `asinh acosh atanh asin acos atan` finite formulas (`log(z + sqrt(z² ± 1))` etc.) | inaccurate componentwise for ordinary inputs: up to 9e3 eps at \|z\| < 2, 1e12 for tiny \|z\|, non-finite for \|z\| ~ 1e200 (`z²` overflow), 1e10 near ±1/±i | glibc/FreeBSD: a few ulps (Hull–Fairgrieve–Tang); libc++/libcu++ share the naive formulas | accuracy (found 2026-11-21 by A6 batch 3; large negative real parts fixed there; rest fixed by A7: ≤ 2.4 eps componentwise) |

### 2.2 What already works for half (no change needed)

Data members (`RealType re_{}`), trivially copyable (`is_trivially_copyable_v<complex<half_t>>`
holds), tuple protocol (`:250-305`), `==`/`!=` (`:315-416`, via `common_type`), all
transcendentals (they call `Kokkos::` scalar functions that exist for half), and
`reduction_identity`. The earlier branch (`half-complex.md`) built and ran all
complex tests for half/bhalf on A100 with only C1 relaxed.

---

## 3. `Kokkos::Experimental::half_t` today (`core/src/impl/Kokkos_Half_*.hpp`)

`half_t`/`bhalf_t` = `Impl::floating_point_wrapper<impl_type>`
(`Kokkos_Half_FloatingPointWrapper.hpp:270`), with `impl_type` = `__half` /
`__nv_bfloat16` (CUDA), HIP/SYCL equivalents, or `std::float16_t` /
`std::bfloat16_t` with C++23 support and no GPU backend (`Kokkos_Macros.hpp:693-717`,
`Kokkos_STD_Half_Impl_Type.hpp`). Without native support `half_t` is `float`
(`KOKKOS_HALF_T_IS_FLOAT`).

| # | Where | Today | vs. `std::float16_t` | Relevance for complex |
|---|---|---|---|---|
| H1 | wrapper `:341-368` | implicit ctors from `float`, `double`, all integers; only `(impl_type)` is `constexpr` | std: narrowing into float16 is **not implicit**; conversions are `constexpr` | `complex<half_t>{1.5, 2.5}` works in Kokkos, not in std. Keep implicit for backward compatibility; document. `constexpr` see D6 |
| H2 | wrapper `:282-288` | default ctor and `(bit_comparison_type)` ctor not `constexpr` | std: trivial / `constexpr` | blocks `constexpr complex<half_t>{}` |
| H3 | wrapper `:292-325` | conversions to arithmetic types all `explicit`, not `constexpr` | std: implicit widening `float16_t → float/double` | `static_cast<double>(h)` needed; affects gtest macros (`ASSERT_FLOAT_EQ` rejects `half_t`) |
| H4 | wrapper `:590-760` | arithmetic in `impl_type` when `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH`, else via `float`; `half` op `float`/`double` → `float`/`double` (upcast templates); integers convert to half | matches the usual arithmetic conversions for `float16_t` | ok |
| H5 | wrapper `:740-860` | comparisons not `constexpr` | std `constexpr` | blocks `constexpr` `==` (half-complex.md §1b) |
| H6 | wrapper `:866-878` | `<<` = `std::to_string(double)` (fixed, 6 decimals, ignores stream flags); `>>` = whitespace token + `std::stod` | libstdc++ (to verify): `<<` formats via `float`/`double` honouring flags; `>>` reads a number and stops at `)`/`,` | breaks `(2)`/`(3,4)` parsing and makes output type-dependent (half-complex.md §2.1). Also: `>>` **throws** (`stod`) on bad input or at end of input instead of setting failbit; `<<` uses the C global locale, not the stream's; values below 5e-7 print as `0.000000`. Examples and causes in D7 |
| H7 | `Kokkos_Half_CommonType.hpp` | `common_type<half, float|double|long double>` = wider type; `half`/`bhalf` have none | matches P1467 ranks (`float16` < `float`; `float16` and `bfloat16` unordered) | ok; use it for mixed complex ops |
| H8 | `Kokkos_MathematicalFunctions.hpp:31-50` | `Impl::promote` defined only for `float`, `double`, `long double`, integers | std treats extended FP types as FP | cause of C10/C12 |
| H9 | `Kokkos_Half_MathematicalFunctions.hpp:186-299` | `abs fabs fmod ... exp log sqrt hypot sin ... atan2 copysign`: compute in `float`, round to half (CUDA/HIP/SYCL/std override with native versions via `impl_<FUNC><false>`); mixed `(half, double)` → `double` | ok | all scalar functions `Kokkos::complex` needs are present; **no `fma`** (commented `// fma`, `:192`) |
| H10 | `Kokkos_Half_MathematicalFunctions.hpp:558-565` | `real(half_t)`, `imag(half_t)` defined here | std: in `<complex>` | `conj`, `norm`, `arg`, `proj` for scalar half are missing |
| H11 | `Kokkos_Half_NumericTraits.hpp:60-184` | `infinity_v`, `epsilon_v`, ... are `constexpr BitComparisonWrapper`, converting to `half_t` at runtime | std `numeric_limits<float16_t>` is `constexpr T` | `epsilon_v<half_t>` cannot be used directly as a double (half-complex step 6) (B4c, 2026-11-24: `constexpr half_t` where half is `constexpr`, std backend and CUDA) |
| H12 | `Kokkos_Half_NumericTraits.hpp:324-333` | `digits_v<bhalf_t> = 2`, `digits10_v<bhalf_t> = 1` | `numeric_limits<std::bfloat16_t>`: `digits = 8`, `digits10 = 2` | **bug**; any tolerance based on `digits` is wrong for bhalf |
| H13 | `Kokkos_STD_Half_Conversion.hpp:18-60` | `cast_to_half`/`cast_from_half` are `inline`, not `constexpr`, though they are plain `std::float16_t` conversions | could be `constexpr` at no cost | enables `constexpr` on the std backend |
| H14 | `Cuda/Kokkos_Cuda_Half_Conversion.hpp:15-50` | intrinsics `__float2half`, ... | CUDA 12.6: only `__half(const __half_raw&)` is `constexpr`; `operator __half_raw()` is not | `constexpr` on CUDA needs a software path (D6) |
| H15 | `Kokkos_BitManipulation.hpp:319` | `Kokkos::bit_cast` not `constexpr` | `std::bit_cast` is | needed for a `constexpr` software path |
| H16 | `Kokkos_Half_NumericTraits.hpp` | `max_digits10`, `min_exponent10`, `max_exponent10` were missing for `half_t`/`bhalf_t` (the trait had no `value`) | `numeric_limits<std::float16_t>`: 5, −4, 4; `<std::bfloat16_t>`: 4, −37, 38 | API gap (found 2026-11-21 while auditing for B1; added) |

---

## 3b. Reference: `cuda::std::complex` (libcu++, CUDA backend)

Inspected on 2026-11-20:
- **CCCL 2.5.0 source** shipped with CUDA 12.6 (the toolkit of our A100 builds):
  `cuda/std/detail/libcxx/include/complex` and `cuda/std/__complex/{vector_support,nvfp16,nvbf16}.h`.
- **CCCL `main`**, via GitHub. There the header is split into
  `cuda/std/__complex/{complex,tuple,literals,nvfp16,nvbf16,...}.h`. I read only
  `complex.h`, `nvfp16.h` and `tuple.h`, through excerpts and summaries rather than in
  full; anything taken only from `main` is marked "(main)".
- **The docs page**
  <https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api/numerics_library/complex.html>.

It is a fork of libc++'s `<complex>`, made host/device. Features:

| Area | `cuda::std::complex` | `Kokkos::complex` (after A1, A2) | Use for Kokkos |
|---|---|---|---|
| Value types | `float`, `double`; `long double` host only. `__half` / `__nv_bfloat16` since libcu++ 2.4.0, when CUDA ≥ 12.2 and the `CCCL_DISABLE_FP16_SUPPORT` / `CCCL_DISABLE_BF16_SUPPORT` macros are not defined. Gate: `__is_complex_float<T> = is_floating_point<T> \|\| __is_nvfp16<T> \|\| __is_nvbf16<T>` | `is_floating_point` only (C1) | **confirms D1**: the same trait design, with `half_t`/`bhalf_t` in place of `__half`/`__nv_bfloat16` |
| `operator*` | naive formula, then **Annex G recovery** when both parts are NaN: `inf·(nan, …)` is recovered as `inf`, as in libstdc++'s `__muldc3` | naive only (new gap **C21**) | **adopt**: A5 |
| `operator/` | compiler-rt `__divdc3` structure: scale the denominator by `logb`/`scalbn`, scale the result back, then Annex G recovery. Same in 2.5 and (main) | Smith + libgcc range guards + recovery (A1) | **keep Kokkos's**: D10 measured this structure as worse (≈10¹⁵ eps with mixed exponents, spurious overflow near max). Kokkos is now more robust than `cuda::std` here |
| Opting out of the recovery | `LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_{OPERATIONS,MULTIPLICATION,DIVISION}` skip it for speed | none | **optional**, A5: a matching Kokkos macro |
| `constexpr` | C++20 `constexpr <complex>` already in C++14. `*` and `/` have an `is_constant_evaluated()` branch that returns the Annex G answers for zero/inf/NaN, and use `__constexpr_isnan/isinf/copysign/logb/scalbn` | `/` cannot be constant-evaluated (A3 note) | **adopt the pattern** in A3: a constant-evaluation branch with `constexpr` helpers |
| Special values in functions | libc++ algorithms: `exp`, `sqrt` (through `polar`), `asinh`/`acosh`/`atanh`/…, `polar` (NaN or negative `rho` → NaN), `norm` (`inf` if either part is infinite), `proj`, all following the Annex G tables; `pow` = `exp(y·log x)`; `__sqr` for `z²` | naive formulas: `exp(inf,0)` = (inf,NaN), `exp(-inf,inf)` = (NaN,NaN), `sqrt(1,inf)` = (inf,NaN); libstdc++ gives (inf,0), (0,0), (inf,inf) (new gap **C22**). `norm(inf,nan)` = NaN in Kokkos **and libstdc++**; only libc++/libcu++ return inf | **adopt selectively**: A6 |
| Scalar overloads | `real imag arg norm conj proj` for integers (→ `double`) and FP types, through `__libcpp_complex_overload_traits`. 2.5 specialises it for `__half` (`<__half,false,false>`) so that scalar `conj`/`norm`/`real`/`imag` work; it also adds `arg(__half)` | `real imag conj norm` only (C10, C11) | **confirms D8/B3**: specialising the promotion trait for half is the approach libcu++ takes |
| Mixed `pow` | `pow(complex<T>, U)` etc. return `complex<__promote_t<T,U>>`, gated by `__is_complex_arithmetic` (which includes half) | gated by `is_arithmetic` (C12) | **confirms** the CH1 gate change |
| Converting ctor | 2.5: implicit iff `T{declval<U>()}` is non-narrowing list-init. (main): `__is_non_narrowing_convertible<T,U>`, **specialised as true for `__half`↔`float`/`double` in both directions** (a workaround for `__CUDA_NO_HALF_CONVERSIONS__`), so `complex<__half>` ← `complex<float>` is **implicit** | always implicit | see the D4 note: libcu++ does *not* follow P1467 for half; brace-init detection cannot work for class-type halves |
| `std::complex` interop | ctor/assignment from `::std::complex<U>` for **any** `U`, by `reinterpret_cast` of `U[2]`, host+device; conversion to `::std::complex<T>` host-only | only `std::complex<RealType>` | possible simplification of CH4 (see CH4); under D11 only libcu++'s constructor is a model, not its conversion to `std::complex` |
| Stream I/O | delegates to `std::complex<T>` (`complex<__half>`: through `std::complex<float>`). `>>` assigns the temporary **even on failure**: the same defect as Kokkos C16, still present in (main) | delegates to `std::complex<RealType>` too (A2), but assigns only on success (A1.4); for half through standard proxy types (CH1) | the same delegation design; Kokkos fixes the assignment defect |
| `complex<__half>` layout | explicit specialisation storing a `__half2`, `alignas(alignof(__half2)) = 4`; `+= -=` via `__hadd2/__hsub2`, `==` via `__hbeq2`. In 2.5, `*` and `/` take their partial products from `__half2` vector multiplies (`__type_to_vector<__half> = __half2`), i.e. the arithmetic is **in half precision, not promoted** | two `half_t` members, `alignas(4)` only with `KOKKOS_ENABLE_COMPLEX_ALIGN` (C18) | optional CH6 (performance), behind a benchmark |
| Half transcendentals | 2.5: `asinh`, `acosh`, `atanh`, `acos` for `complex<__half>` computed through `complex<float>`; everything else in half. (main): the specialisations are not in `nvfp16.h` (where they live now was not checked) | — | **supports D5**: libcu++ needed float for at least these four |
| Tuple protocol | (main) `get<I>` (4 overloads, `constexpr noexcept`) in `__complex/tuple.h`; the fetched file has no `tuple_size`/`tuple_element` | full P2819 already (`:250-305`) | nothing to adopt |
| Literals | `_i`, `_if`, `_il`, but documented as **not supported with NVCC** | none (D9) | keep out of scope |
| `volatile` `real()`/`imag()` | kept "to help with reductions in thrust" | none | not needed (Kokkos reductions do not use `volatile`) |

Key takeaways:
- libcu++ already does several things Kokkos plans: the half-aware value-type trait (D1), promotion
  traits specialised for half (D8), mixed `pow` including half (C12), and float
  computation for some half transcendentals (D5).
- Its division is *less* robust than Kokkos after A1, and its I/O has the C16 defect.
- Worth adopting: Annex G recovery in `*` (A5); a constant-evaluation branch for `*` and
  `/` (A3); special-value handling in the elementary functions (A6); possibly
  `__half2`-backed storage for `complex<half_t>` on CUDA (CH6).
- Not verified here: how the (main) `complex<__half>` multiplies, i.e. whether the partial
  products are still `__half2` vector ops.

---

## 4. Design decisions

**D1: value-type trait instead of `is_floating_point`.** Add to `Kokkos_Complex.hpp`
(or a small `impl/Kokkos_Complex_Traits.hpp`):

```cpp
namespace Kokkos::Impl {
template <class T>
inline constexpr bool is_complex_value_type_v =
    std::is_floating_point_v<T> ||
    Kokkos::Experimental::Impl::is_float16<T>::value ||
    Kokkos::Experimental::Impl::is_bfloat16<T>::value;
}
```

`is_float16`/`is_bfloat16` already exist (`Kokkos_Half_FloatingPointWrapper.hpp:24-34`,
`:52`, `:120`) and are false when `half_t` is `float`. That is fine, because `float` passes
`is_floating_point`. Use it in the class `static_assert`, the scalar overloads (D8)
and mixed `pow` (C12). This needs `#include <Kokkos_Half.hpp>` in
`Kokkos_Complex.hpp`, which also fixes C2.

**D2: `std::complex` interop target. — SUPERSEDED by D11 (2026-11-22).** *The rest of
D2 (keep everything; add an explicit conversion operator to
`std::complex<std::float16_t>`) no longer applies: the only interop kept is the
converting constructor, and CH4 shrinks to a constructor.* Original text: Keep the existing interop with
`std::complex<RealType>` for all types. For half it is `std::complex<half_t>`: it works
with libstdc++, and removing it would break code. Additionally, when
`KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT` / `..._BHALF_...` is defined (host-only
C++23 builds), add **explicit** conversions to/from `std::complex<half_t::impl_type>`
(= `std::complex<std::float16_t>`). That is the standard-conforming type, and the closest
match to the goal. Do not add it for CUDA/HIP/SYCL `impl_type`s (`std::complex<__half>`
is meaningless).

**D11: `std::complex` interop: keep only the converting constructor (decided
2026-11-22).** Deprecate every other interop between `Kokkos::complex` and
`std::complex` in Kokkos 5 (`KOKKOS_ENABLE_DEPRECATED_CODE_5`), and remove it in
Kokkos 6. Kept: `complex(const std::complex<RealType>&)`, i.e. `std::complex` →
`Kokkos::complex`. Reasons:
- `std::complex` is not device-callable, so all the other interop is host-only, in a
  type meant for device code (each overload has a "cannot be called in a CUDA device
  function" comment);
- the overloads are an asymmetric, accidental set: `std * Kokkos` exists, `Kokkos *
  std` and `+ - /` do not; `==`/`!=` exist both ways; `exp` alone of all functions
  takes a `std::complex` (with the naive formula, not A6's special values);
- the implicit `operator std::complex<RealType>()` (marked "TODO: make explicit" since
  2019) creates `std::complex<half_t>` for half (unspecified, C15), and makes mixed
  expressions ambiguous;
- the converting constructor alone covers the common uses: it is implicit, and every
  member operator taking `const complex<RealType>&` (`= += -= *= /=`) and every
  non-template parameter accept a `std::complex<RealType>` through it.
Implementation step: A8. Interop with `std::complex<std::float16_t>` (CH4) follows the
same policy: a converting constructor only.

**D3: mixed precision stays a Kokkos extension.** `std::complex` forbids
`complex<float16_t> + double`. Kokkos allows it, returning `complex<common_type_t>`.
Keep it, since removing it is a breaking change. Document that
`complex<half_t> op double` yields `complex<double>` (H7). This is also the trap
noted in half-complex.md §3.4: literals like `3.` silently promote.

**D4: explicitness of the converting constructor (C6).** Mirror P1467 with
`explicit(bool)`, but **only where it is not a breaking change**:

```cpp
template <class RType>
  requires std::is_convertible_v<RType, RealType>
KOKKOS_INLINE_FUNCTION constexpr
explicit(Impl::is_narrowing_fp_conversion_v<RType, RealType> &&
         Impl::is_half_v<RealType>)            // new: only into half types
complex(const complex<RType>& other) noexcept;
```

`is_narrowing_fp_conversion_v<From, To>` = `sizeof`/rank-based: `float`/`double`
→ `half_t` is narrowing; `half_t` ↔ `bhalf_t` has no conversion at all (H7). For
`double → float` Kokkos is implicit today while `std::complex<float>(complex<double>)`
is explicit. Changing that is a separate, deprecation-cycle decision. Record it as
a known divergence.

*Note from §3b (libcu++).* `cuda::std::complex` decides implicit vs. explicit by
non-narrowing list-initialisation. In 2.5 that is `T{declval<U>()}` well-formed; (main)
uses `__is_non_narrowing_convertible`. Two consequences for Kokkos:
1. **Brace-init detection cannot be reused.** `half_t` is a class with converting
   constructors, so `half_t{double}` is not a narrowing error. libcu++ hit the same problem
   with `__half` and, in (main), specialises the trait by hand. So
   `is_narrowing_fp_conversion_v` must be rank-based, as written above.
2. **libcu++ chose the opposite policy for half.** It marks `__half` ↔ `float`/`double` as
   non-narrowing in **both** directions, so `cuda::std::complex<__half>` ← `complex<float>`
   is implicit. This is the precedent for keeping Kokkos implicit, which costs no breakage.
   CH2's explicit-into-half follows `std::complex<std::float16_t>` (P1467) instead.
   Decide CH2 with both precedents in view.

**D5: intermediate precision for half arithmetic (C14).** Define

```cpp
template <class T> struct complex_compute_type { using type = T; };
template <> struct complex_compute_type<Kokkos::Experimental::half_t>  { using type = float; };
template <> struct complex_compute_type<Kokkos::Experimental::bhalf_t> { using type = float; };
```

Compute `*`, `/`, `*=`, `/=`, `abs`, and the transcendental compositions
(`asinh`, `acosh`, `atanh`, `atan`, `tan`, `tanh`, `pow`) in `compute_type`, and round
once to `RealType`. Rationale:
- Half products overflow early (`|a|·|c| > 65504`), and each rounding costs a lot in an
  11-bit (8-bit for bhalf) significand.
- The scalar half math already computes in `float` (H9), so this is consistent.
- GCC evaluates `_Float16` arithmetic with excess precision on most targets. Checking
  what libstdc++'s `complex<_Float16>` does is a to-do (needs GCC ≥ 13).

Cost: two conversions per operand. It needs measuring on device, where native `__half`
arithmetic exists (`KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH`). Provide it as the default for
`/` and transcendentals, and benchmark `*` before deciding.

**D6: `constexpr` (H1, H2, H5, H13-H15).** Two tiers:
- *std backend (C++23 `std::float16_t`)*: mark the wrapper ctors/operators and
  `Kokkos_STD_Half_Conversion.hpp` functions `constexpr`. They are plain native
  conversions, so `static_assert(Kokkos::complex<half_t>{42., 43.} == ...)` becomes
  possible there.
- *CUDA/HIP/SYCL*: add an `if (std::is_constant_evaluated())` branch to
  `cast_to_half`/`cast_from_half` (and bhalf) that does round-to-nearest-even in integer
  arithmetic. Build the result with the `constexpr __half(__half_raw)` ctor and read bits with
  `std::bit_cast<uint16_t>` (H14, H15). Then make the wrapper ctors and comparisons
  `constexpr`.
  - Risk: nvcc/hipcc acceptance of `std::bit_cast` on these types in constant evaluation.
  - Needs a prototype; separate PR. Until then the `constexpr` complex tests stay
    float/double-only, with runtime twins for half (half-complex.md step 2).

**D7: I/O (H6, C15, C16).** Stream I/O of `complex<half_t>` has two layers:

1. **The complex layer:** how `(re,im)` is written and parsed. **Done for float/double:**
   - A1.4 made `>>` assign only on success.
   - A2 (revised) delegates the bodies to `std::complex<RealType>`. Kokkos keeps its own
     thin overloads, because `std::complex`'s cannot be found through the conversions; see
     A2.
   - For half, CH1 swaps in a **standard** proxy: `std::complex<float>` for output (exact)
     and `std::complex<double>` for input. This avoids `std::complex<half_t>`, which the
     standard leaves unspecified. *(Superseded 2026-11-22 by the B2 revision: input now
     parses as `std::complex<std::float16_t>` does, reading each part with the C++23 half
     extractor, through float, clamping at ±max.)*
2. **The scalar layer:** how one `half_t` is written and read. **Still broken; this is
   the subject of D7 / B2.**
   - Today `complex<half_t>` reaches these operators through `std::complex<half_t>`: its
     `operator>>` calls `is >> half_t`, and `operator<<` calls `s << half_t`. So every
     scalar defect below is also a `complex<half_t>` defect (half-complex.md §2.1).
   - With the CH1 proxy, `complex<half_t>` I/O **no longer touches them**, because the
     standard library parses and formats `float`/`double`. This decouples Part C from B2.
   - B2 is still needed for `half_t` itself (`std::cout << h`, gtest messages,
     `is >> h`).
   - The **choice of proxy types** follows the same reasoning as the B2 fix below:
     `float` for output, `double` for input. *(Superseded by the B2 revision: C++23
     specifies `float` for input, double rounding included, with clamping at ±max; see
     the B2 finding in Part B.)*

*Current scalar code* (`Kokkos_Half_FloatingPointWrapper.hpp:866-878`, used by both
`half_t` and `bhalf_t`):

```cpp
friend std::ostream& operator<<(std::ostream& os, const floating_point_wrapper& x) {
  const std::string out = std::to_string(static_cast<double>(x));
  os << out;
  return os;
}
friend std::istream& operator>>(std::istream& is, floating_point_wrapper& x) {
  std::string in;
  is >> in;
  x = std::stod(in);
  return is;
}
```

*Examples where it goes wrong.* These are reproduced with a host program that copies the
two bodies above and models `half_t` by GCC 12's IEEE-binary16 `_Float16`; the operators
only see `static_cast<double>(x)`, so the result is the same. `complex_io` on A100 showed
the first input row for real (half-complex.md §2.1, job 10022380).

| # | Code (`h`/`z` are `half_t` / `complex<half_t>`) | Current result | Expected (stream rules, as for `float`) |
|---|---|---|---|
| O1 | `os << z` with `z = {3.14, 1.41}` | `(3.140625,1.410156)` | `(3.14062,1.41016)` (precision 6, like `complex<float>` holding the same values) |
| O2 | `os << std::setprecision(3) << z` | `(3.140625,1.410156)`: precision ignored | `(3.14,1.41)` |
| O3 | `os << std::scientific << h`, also `hexfloat`, `showpos`, `uppercase` | ignored, always fixed with 6 decimals | formatted as requested |
| O4 | `os << complex<half_t>{denorm_min, 1e-5}` | `(0.000000,0.000010)`: the first value is lost, the second has 1 significant digit | `(5.96046e-08,1.00136e-05)` |
| O5 | `setlocale(LC_NUMERIC, "de_DE.UTF-8")`, then `os.imbue(std::locale::classic()); os << z` | `(3,140625,1,410156)`: ambiguous, cannot be parsed back | `(3.14062,1.41016)`: the stream's locale decides |
| I1 | `ss.str("1 (2) (3,4)"); ss >> z >> z >> z` | 1st `(1,0)`; 2nd **fails** | `(1,0)`, `(2,0)`, `(3,4)` |
| I2 | `ss.str("x"); ss >> h` (or `ss >> z`) | **throws `std::invalid_argument`** | `ss.fail()`, `h` unchanged |
| I3 | `ss.str(""); ss >> h` (reading at end of input) | **throws `std::invalid_argument`** | `ss.fail()` (and `eof()`) |
| I4 | `ss.str("1e999"); ss >> h` | **throws `std::out_of_range`** | `ss.fail()`, as for `float` |
| I5 | `while (ss >> h) ...` loop over a file | ends with an exception, not by `fail()`; see I3 | ends normally |

The A2 parity test (`complex_io_like_std`) covers float and double only. It passes
because the defects are all in the half scalar operators.

*Why each one happens:*
- **O1–O3:** `std::to_string(double)` is specified as `std::sprintf("%f")`. That is
  always fixed notation with 6 decimals, and it is computed before the stream is involved.
  So `precision`, `floatfield` (`scientific`/`fixed`/`hexfloat`), `showpos`, `showpoint`
  and `uppercase` cannot have any effect. Only `width`/`fill` still apply, because they
  apply to the finished string.
- **O4:** "6 decimals" is an absolute resolution of 1e-6.
  - half has subnormals down to 5.96e-8, and normal values below 1e-4 with 3–4 significant
    digits. Every half below 5e-7 prints as `0.000000`, and values near 1e-5 keep 1 digit.
  - Printing is therefore not invertible: writing a small value and reading it back gives
    a different value, often 0. With stream formatting, the default precision 6 is ≥
    `max_digits10` (5 for half, 4 for bhalf), so default output round-trips.
  - For bhalf (range up to 3.4e38) `%f` also prints up to 39 integer digits.
- **O5:** `sprintf` uses the C global locale (`setlocale`), not the stream's
  `std::locale`. The stream's `imbue` is bypassed, and a German C locale puts `,` as the
  decimal separator, the same character as the complex separator.
- **I1:** `is >> std::string` reads a whole whitespace-delimited token: `"2)"` for the
  second value.
  - `stod("2)")` parses the prefix `2` and silently drops the rest, so `)` is consumed.
  - The complex parser then reads its "closing" character from the *next* token (`(`
    of `(3,4)`), sees neither `)` nor `,`, and sets failbit.
  - So any composite format that does not surround the value with whitespace breaks: the
    complex `(re,im)` form, CSV, `key=value`.
  - `"( 2 ) ( 3 , 4 )"` happens to work, which is the workaround half-complex.md §2.1 had
    to use in tests.
- **I2–I5:** `std::stod` reports errors by exceptions, while streams report them with
  `failbit`.
  - An empty token (end of input, or a failed `is >> in`) and non-numeric text throw
    `invalid_argument`; overflow throws `out_of_range`.
  - The stream's state is never set. The idiomatic `while (is >> h)` loop therefore
    cannot terminate normally, and code compiled without exceptions aborts.
  - The return value of `is >> in` is not checked either, so a stream that is already in
    a failed state still calls `stod`.

*Proposed scalar fix (B2).* Changed from the earlier draft: `>>` reads through `double`,
not `float`.

```cpp
template <class CharT, class Traits>
friend std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os, const floating_point_wrapper& x) {
  return os << static_cast<float>(x);  // exact; honours flags, precision, locale
}
template <class CharT, class Traits>
friend std::basic_istream<CharT, Traits>& operator>>(
    std::basic_istream<CharT, Traits>& is, floating_point_wrapper& x) {
  double d;
  if (is >> d) x = d;  // num_get: stops at ')' and ','; failbit, no exceptions
  return is;
}
```

Why this works:
- **`<<` through `float`:** every half and bhalf value is exactly representable in `float`.
  So the stream formats the exact stored value with all its own rules (O1–O5): `num_put`
  uses the stream's flags, precision and imbued locale.
- **`>>` through `num_get`:** reading a `double` stops at the first character that cannot
  belong to a number (`)` and `,` stay in the stream, I1). It reports errors with failbit
  (I2–I5) and uses the stream's locale. Assigning only on success matches A1.4.
- **`double`, not `float`, for `>>`:** decimal → float → half rounds twice.
  - Example: `"1.000488281250001"` lies just above the midpoint `1 + 2⁻¹¹` between the
    halves `1` and `1 + 2⁻¹⁰`.
  - Through float it rounds first to the midpoint exactly, then ties-to-even gives
    **`1.0`**. The correctly rounded result is `1.0009765625`, which is what reading
    through double gives (emulated with `_Float16`).
  - Double rounding through double remains possible only for inputs within ~1e-16
    relative of a midpoint, i.e. more than 17 significant digits.
  - Exactly rounded parsing would need `std::from_chars` for `std::float16_t` (GCC ≥ 13),
    or a dedicated parser.
  - The `x = d` conversion (`cast_to_half(double)`, e.g. `__double2half` on CUDA) must
    round correctly; check this per backend.
- **`CharT`/`Traits` templates:** as in A2. Without them, `std::wostream << h` does not
  compile for a `half_t` `h`. `complex<half_t>` is not affected once CH1 delegates through
  `std::complex<float>`.

Consequences and compatibility:
- Printed values change: `half_t(3.14)` gives `3.14062` instead of `3.140625`, and
  `half_t(100)` gives `100` instead of `100.000000`. Check `TestHalf*` and any user code
  that parses the old text.
- Code that relied on exceptions from `>>` now has to check `fail()`, as for every other
  arithmetic type.
- Scalar `half_t` output then matches the `float` holding the same value, which is
  consistent with `complex<half_t>` output through the CH1 proxy. That output equals
  `complex<float>`'s by construction, without B2; `cuda::std::complex<__half>` does the
  same (§3b).
- To verify on GCC ≥ 13: whether libstdc++ prints `std::complex<std::float16_t>` the
  same way (P1467 specifies insertion of extended types through `float`/`double`), and
  whether extraction into `std::float16_t` exists at all.

**D8: scalar overloads and missing functions (C10, C11, H8, H10).**
- Specialise `Kokkos::Impl::promote` for `half_t`/`bhalf_t` (`using type = half_t`) in
  `Kokkos_Half_MathematicalFunctions.hpp`.
  - This is safe: every generic template in `Kokkos_MathematicalFunctions.hpp` that uses
    `promote_2_t`/`promote_3_t` is also gated on `std::is_arithmetic_v` (e.g. `:195-204`),
    so no new overloads appear.
  - `conj(half_t)` and `norm(half_t)` then work.
- Remove the duplicate `real`/`imag(half_t)` from the half header, or keep them. Non-templates
  win overload resolution, so there is no ambiguity either way.
- Add `arg(complex<T>)` = `atan2(imag, real)` and `proj(complex<T>)` = map infinities to
  `(inf, copysign(0, imag))`, plus scalar `arg`/`proj`. This applies to all types.
- Change the `std::is_arithmetic_v` gate in mixed `pow` (`:588`, `:595`) to "arithmetic or
  half", and define `promote_2` for half pairs via `std::common_type`.

**D10: complex division algorithm (C5, C14).**

*What the Kokkos branch does.* `operator/` (`:810-836`) and `operator/=` (`:182-206`)
compute `s = |c| + |d|`, then `x/s`, `conj(y)/s`, and divide by the squared modulus.
Only when `s == 0` do they return `(a/s, b/s)`. Without that branch, `y/s = 0/0 = NaN`
makes every result NaN. With it, `(1,1)/(0,0) = (inf, inf)`, as C99 Annex G G.5.2
requires for nonzero/zero. So the branch is needed for that behaviour; only its
`0.0` literal is cosmetic.

*What LLVM does* (llvm-project `main`, checked 2026-11-18):
- **libc++** `<complex>`: for floating-point `T`, `operator/` forwards to the compiler's
  `_Complex` division (`__lhs.__builtin() / __rhs.__builtin()`). For any other `T`,
  including a class type like Kokkos `half_t`, it uses the naive formula
  `((ac+bd)/(c²+d²), (bc−ad)/(c²+d²))` with **no** zero or infinity handling. libc++ has
  no `_Float16` specialisation.
- **compiler-rt `__divdc3`**, the routine clang calls for `_Complex double` division
  (Annex G):
  1. Scale `c, d` by `2^-ilogb(max(|c|,|d|))` with `logb`/`scalbn`. Power-of-two scaling
     is exact.
  2. Compute the naive formula and scale back.
  3. **Only if both results are NaN**, recover three cases: zero denominator
     (`copysign(inf, c) * a`, `... * b`), infinite numerator with finite denominator,
     and infinite denominator with finite numerator (→ signed zeros).
  - The common path has no zero test at all.
- clang `-fcomplex-arithmetic=basic|improved|promoted|full` (clang ≥ 18; to verify)
  selects naive, Smith's algorithm, a wider-type computation, or `__divXc3`.

*Differences between Kokkos and `__divdc3`:*
- **Infinite denominators:** `(1,1)/(inf, 0)` gives NaN in Kokkos (`inf/inf`), and
  `(0, 0)` per Annex G.
- **Rounding:** division by `s` rounds, `scalbn` does not.
- **Zero denominator:** `__divdc3` takes the sign from `c` (`copysign(inf, c)`), while
  Kokkos always divides by `+0` (`s = |c|+|d| ≥ 0`).
- **Cost:** Kokkos pays a test on every division; `__divdc3` pays only when the result
  is NaN.

*What GCC does* (the reference our host tests compare against): libstdc++ `operator/`
uses `_Complex` division. GCC lowers it to libgcc's `__divdc3`, which since GCC 12 is
**Smith's algorithm** with range guards: halve all operands when the larger denominator
component is ≥ max/2, multiply by `1/eps` when it is tiny, and use a different evaluation
order when `ratio` is subnormal. The same post-hoc Annex G recovery follows. For float
(`__divsc3`) and half, libgcc instead computes the naive formula in double / float.

*Recommendation (revised 2026-11-20, implemented in A1.3):* Smith's algorithm with
libgcc's range guards, plus the Annex G recovery. One helper,
`Impl::complex_divide<T>`, serves `operator/` and `operator/=`. Measured with a host
prototype against libstdc++ (GCC 12.2 libgcc) and a long-double reference (2·10⁶ random
operands per row; "spurious" = non-finite result although the true result is representable):

| Algorithm | worst error, ordinary range | worst error, mixed exponents (double) | spurious inf/NaN near max | special-value grid mismatches vs libstdc++ (double) |
|---|---|---|---|---|
| Kokkos today | 2.46 eps | 1.87 eps | 0 | 7636 |
| compiler-rt `__divdc3` (the earlier recommendation) | 1.99 eps | **~10¹⁵ eps** | 77 (double), 1793 (float) | 3442 |
| power-of-two scaling of both operands + recovery | 1.99 eps | 1.54 eps | 450–3579 (`inf·0`) | 1125 |
| **Smith + libgcc guards + recovery (chosen)** | **1.59 eps** | **1.16 eps** | **0** | **464**: all are overflowing results or non-finite inputs; none is finite/finite → finite |

Why the earlier recommendation was dropped:
- compiler-rt scales only the denominator and scales the result back afterwards. So
  `a·c' + b·d'` overflows when the numerator is near max: `(1e308,1e308)/(1e308,1e308)`
  gives `(inf, 0)`.
- The scale-back underflows for mixed exponents.
- It also needs `logb` and 4×`scalbn`. Smith needs 3 divisions (Kokkos today: 6) and a
  few `fabs` comparisons.

Remaining divergence: like libgcc's double path, Smith overflows when `|a|+|b|` is near
max even though the result is finite, e.g. `(FLT_MAX,FLT_MAX)/(1,1)`. Kokkos today gets
that case right. libgcc avoids it for float only because it computes in double. Not
fixed in Part A, because double is slow on many GPUs.

`cuda::std::complex` (§3b) uses the compiler-rt structure in CCCL 2.5 (CUDA 12.6) and in
(main), with an `is_constant_evaluated()` branch on top. So the weaknesses measured in the
"compiler-rt" row also apply to libcu++ on device, and Kokkos's A1 division is the more
robust of the two.

Not verified: this transcription was compared with the libgcc binary, not the libgcc
source. compiler-rt was not run: clang 13 here links libgcc's `__divdc3`.

- **half_t (Part C, CH3):** unchanged. Compute in `float` with the naive formula, with no
  scaling and no upfront test. For finite half inputs `c² + d² ≤ 2·65504² ≈ 8.6e9` and
  `≥ (2^-24)² ≈ 3.6e-15`, so it cannot overflow or underflow in float. This is also what
  libgcc's `__divhc3` does. Keep the post-hoc recovery for zero/infinite operands.
- **bhalf_t:** its range equals float's, so the naive float squares *can* overflow.
  Convert to float and call `Impl::complex_divide<float>` (Smith); double intermediates
  are not needed. Note that `complex_divide<T>` takes its thresholds from `constexpr`
  `finite_max_v<T>`/`norm_min_v<T>`/`epsilon_v<T>`. For half types these are
  `BitComparisonWrapper` (H11), so half/bhalf **must** go through the float compute type
  and never instantiate `complex_divide<half_t>` directly.

**D9: out of scope.** Complex literals (`_i`), `fma` for half (H9: add it when
needed), `long double` on device.

---

## 5. Modification plan (by file, in PR-sized steps)

The work is ordered in three parts:

- **Part A:** align `Kokkos::complex` with `std::complex`. It does not touch half
  types, and every step is tested with float/double.
- **Part B:** align `half_t`/`bhalf_t` with `std::float16_t`/`std::bfloat16_t`. It
  does not touch `Kokkos::complex`, and is tested with the existing half tests.
- **Part C:** add `complex<half_t>`/`complex<bhalf_t>` support on top of A and B.
  Its steps are named CH1–CH6, to avoid confusion with the gap IDs C1–C22 of §2.1.

A and B are independent of each other and can be developed in parallel. C needs
both:
- CH1 needs A2, whose delegation CH1 extends to half through standard proxy types. Since the A2 revision it no longer needs B2, because `complex<half_t>` I/O no longer calls `half_t`'s stream operators.
- CH5 needs B4, for `constexpr`.

```text
A1 ─ A2 ─ A3 ─ A4 ──┐
                    ├─ CH1 ─ CH2 ─ CH3 ─ (CH4: future) ─ (CH5: future, with B4) ─ (CH6: no gain, not done)
B1 ─ B2 ─ B3 ─ B4 ──┘
A8 (D11, deprecation of the std::complex interop) is independent of B and C.
A5, A6 (from §3b) are independent of B and C. A5 should precede CH3, whose half `*`
reuses Impl::complex_multiply. A6 can come later.
```

### Part A: align `Kokkos::complex` with `std::complex` (`Kokkos_Complex.hpp`)

**A1. Bug fixes.**
1. C3: rewrite `operator/=(const std::complex<RealType>&)` as
   `return *this /= complex(y);` (host-only, since it takes `std::complex`), or delete it.
   Add a test that instantiates it.
2. C4: `const common_real_type y_scaled_abs`. Test: `complex<float>/complex<double>`
   matches the double result to double precision.
3. C5/D10: replace the upfront `s == 0` branch in `operator/` and `operator/=` with the
   compiler-rt `__divdc3` structure: compute first, recover only when both results are NaN.
   Tests against `std::complex<double>` division for `(1,1)/(0,0)`, `(inf,0)/(1,1)`,
   `(1,1)/(inf,0)` (must be `(0,0)`; NaN today), and a large/small pair such as
   `(1e300,1e300)/(1e300,1e-300)` for scaling.
4. C16: `operator>>` must not assign on failure. Test: after a failed `ss >> z`, `z` is
   unchanged and `ss.fail()` is set.

*A1 status (2026-11-20): implemented in the working tree, not committed.*
- **C3:** `return *this /= complex(y.real(), y.imag());`, and the overload is now
  **host-only**: `KOKKOS_INLINE_FUNCTION` was removed, matching `operator=(std::complex)`.
  Keeping it device-marked produced nvcc warnings #20013/#20015 (host `constexpr`
  `std::complex::real()` called from `__host__ __device__`), because
  `--expt-relaxed-constexpr` is opt-in (`Kokkos_ENABLE_CUDA_CONSTEXPR`,
  `cmake/kokkos_arch.cmake:219`) and off in our builds. Test:
  `complex_divide_assign_std_complex`.
- **C4:** fixed as a side effect of item 3: everything is computed in `common_real_type`.
  Test: `complex<float>/complex<double>` must be bit-identical to
  `complex<double>(x)/y`, and within 4 eps(double) of the long-double reference.
- **Item 3:** changed from the compiler-rt structure to Smith + libgcc guards; see D10
  for the measurements.
  - Behaviour change: `(1,1)/(-0,0)` is now `(-inf,-inf)` (sign from `real(y)`, as in
    libstdc++). It was `(+inf,+inf)` before.
  - The plan's scaling test `(1e300,1e300)/(1e300,1e-300)` was dropped because today's
    code already passes it. It is replaced by `(max/2,max/2)/(max/2,max/2) = (1,0)`,
    which is NaN today, and by a denominator at `norm_min`.
  - Test: `complex_division` (float and double, both `/` and `/=`, on the test
    execution space).
- **C16:** `if (is >> x_std) x = x_std;`. Test: `complex_io_failure` with
  `"x"`, `"(x)"`, `"(1,x)"`, `"(1;2)"`, `"(1,2"`.
- The existing tests are unchanged.
- Verification:
  - nvcc builds with 0 errors and 0 warnings; clang-format 17 is clean.
  - Job 10375120 (A100, 2026-11-20): all 10 `*complex*` tests pass on Serial and Cuda.
    These are the 7 existing tests plus `complex_division`, `complex_io_failure` and
    `complex_divide_assign_std_complex`.
  - The new special-value cases fail with the old algorithm: (1,1)/(inf,0) and
    (max/2,max/2)/(max/2,max/2) give NaN, and (1,1)/(-0,0) gives +inf. This was shown
    with the host prototype, not by rerunning the old header.

**A2. I/O by delegation to `std::complex` (D7, complex layer).** *Revised 2026-11-20.*
The first version of A2 re-implemented libstdc++'s `<<`/`>>` so that only `RealType`'s
stream operators were needed. It was replaced before being committed, because the same
goal is reached more simply by delegating to a **standard** `std::complex`:
`std::complex<RealType>` for float/double, and a standard proxy type for half (CH1).

Kokkos still needs its own overloads; the conversions alone are not enough (checked with
GCC 12):
- `os << z`: `std::complex`'s `operator<<` is a template on `complex<T>`, and `T` cannot
  be deduced from `Kokkos::complex` because implicit conversions are not considered during
  deduction. The result is "no match for `operator<<`", even with the implicit
  `operator std::complex<RealType>()`.
- `is >> z`: this needs an lvalue `std::complex<T>&`. A conversion yields a temporary, and
  the converting constructor goes the other way.

Only the *bodies* delegate:

```cpp
template <class RealType, class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const complex<RealType>& x) {
  return os << std::complex<RealType>(x.real(), x.imag());
}
template <class RealType, class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits>& is,
                                              complex<RealType>& x) {
  std::complex<RealType> x_std;
  if (is >> x_std) x = complex<RealType>(x_std.real(), x_std.imag());
  return is;
}
```

Why delegation instead of the re-implementation:
- **Exact by construction:** output, parsing, stream state and the unread remainder are
  those of the platform's `std::complex`, under libstdc++ *and* libc++. The re-implementation
  copied libstdc++, so under libc++ it diverged on what malformed input consumes.
- **Less code:** about 40 lines of parser are gone. Nothing new is needed on the
  `RealType` side.
- **Half does not need the scalar fix:** with a standard proxy type (CH1), `complex<half_t>`
  I/O never calls `half_t`'s stream operators (H6, D7). B2 is still worth doing for
  `half_t` itself, but Part C no longer depends on it.

*A2 status (2026-11-20): verified on A100 (job 10376616): 13/13 `*complex*:*atomic*`
tests pass on Serial and Cuda, with 0 errors/warnings in the nvcc build. Committed after
A1.*
- **Kept from A1.4:** assign only on success. `cuda::std::complex` still lacks this (§3b).
- **Signatures:** templated on `CharT`/`Traits`, like `std::complex`'s, instead of only
  `std::ostream`/`std::istream`. Source compatible: `std::ostream` deduces exactly as
  before, and wide streams now work.
- **Includes:** unchanged (`<iosfwd>`; `<complex>` provides the stream classes).
- **Assignment:** `x` is assigned from `x_std.real()`/`imag()` instead of `x = x_std`.
  `x = x_std` goes through `operator=(const std::complex<RealType>&)` and its
  `reinterpret_cast`, and does not carry over to a proxy type for half.
- **Test:** `complex_io_like_std` for `<float,char>`, `<double,char>` and
  `<double,wchar_t>`.
  - Output must equal `std::complex`'s character for character under 8 formatting states.
  - Input must give the same value, `fail()`, `eof()` and unread remainder for 15 inputs,
    well-formed and malformed.
  - The earlier `__GLIBCXX__` guard on the remainder check is removed, since delegation
    makes it library-independent.
  - `complex_io` and `complex_io_failure` are unchanged.

**A3. `constexpr` non-member operators (C8).** Mark unary/binary `+ - * /` and the mixed
overloads `constexpr`. They are inline templates, so this has no ABI impact. Test with
`static_assert`s for float/double, e.g. `complex<double>{1,2} * complex<double>{3,4} == complex<double>{-5,10}`.
`/` and `/=` (complex divisor) cannot be constant-evaluated: `Impl::complex_divide` calls
`Kokkos::fabs`/`isnan`/`copysign`, which are not `constexpr`. The same was true before
A1, because `/=` was already `constexpr` and called `fabs`. Making division usable in
constant expressions needs an `if (std::is_constant_evaluated())` path; leave it out of A3
and do not `static_assert` on division.

*A3 status (2026-11-20): verified on A100 (job 10377355: 13/13 `*complex*:*atomic*` tests
pass on Serial and Cuda); 0 errors/warnings in the nvcc build; committed.*

**Correction to the plan:** the complex-divisor overloads are **not** marked `constexpr`
in A3. The plan said they "can be marked `constexpr`", but that would be wrong rather than
harmless. Under C++20 (Kokkos's language level), a `constexpr` function template none of
whose specialisations can be a constant expression is ill-formed, no diagnostic required
([dcl.constexpr]/6; C++23 P2448 drops that rule). So `operator/(complex, complex)` and
`operator/(scalar, complex)` get `constexpr` in A3b, together with the
constant-evaluation branch that makes it true.

The existing member `constexpr operator/=(const complex&)` has the same problem, and had
it before A1 too. It is left as is, and A3b fixes it as well.

Marked `constexpr` (13 overloads, all inline templates, so no ABI impact):
- unary `+ -`;
- binary `+ - *` for (complex, complex), (complex, scalar) and (scalar, complex),
  including mixed precision;
- `complex / scalar`;
- the host-only `std::complex * complex`.

Test: file-scope `static_assert`s in `TestComplex.hpp`.
- `test_arithmetic_in_constant_expression<float|double>()` covers every overload above,
  with results exact in both types.
- Mixed-precision checks: `complex<float> * complex<double>`, `int * complex<double>` and
  `complex<float> - double`, plus the result type of `complex<float> + complex<double>`.
- `std::complex<double> * complex<double>`.
- Negative check: compiled against the A2 header, the same file fails with 56 errors
  ("expression must have a constant value").

*Extension from §3b (libcu++).* libcu++ makes `*` and `/` constant-evaluable even in C++14:
- `*` and `/` are `constexpr`, and a branch `if (__libcpp_is_constant_evaluated())` returns
  the Annex G answers for zero/inf/NaN operands without doing the invalid floating-point
  operations.
- `isnan`, `isinf`, `copysign`, `logb` and `scalbn` are replaced by `__constexpr_*`
  helpers.

Kokkos can do the same as an optional second half of A3 (**A3b**):
- Add `constexpr` helpers in `Kokkos::Impl`: `isnan(x)` as `x != x`, `isinf` as a
  comparison with `infinity_v`, `fabs` as `x < 0 ? -x : x` (with signed zeros from
  `copysign`), and `copysign` via `std::bit_cast`.
- Use them in `complex_divide` under `if (std::is_constant_evaluated())`, keeping the
  `Kokkos::` math calls on the runtime path.

Then `static_assert(complex<double>{1,1} / complex<double>{0,0} == ...)` and similar
become possible. Smith's algorithm needs no `logb`/`scalbn`, so A3b is smaller than
libcu++'s version.

*A3b status (2026-11-20): verified on A100 (job 10377468): 14/14 `*complex*:*atomic*` tests
pass on Serial and Cuda, including `complex_division`, so the runtime path is unchanged;
0 errors/warnings in the nvcc build; committed.*

**Changes to the sketch above, and why:**

1. **Compile-time detection.** The branch uses `if consteval` under C++23 and
   `__builtin_is_constant_evaluated()` otherwise, not `std::is_constant_evaluated()`.
   - `std::is_constant_evaluated()` is a host-only function. Calling it from a
     `KOKKOS_INLINE_FUNCTION` gives nvcc warning #20013, because `--expt-relaxed-constexpr`
     is off; see A1.
   - nvcc 12.6 accepts the builtin in `__host__ __device__ constexpr` code with no warning.
     Tested: a `static_assert` takes the compile-time branch, while host and device code
     (checked in the PTX) take the runtime branch.
   - The builtin is available on all Kokkos minimum compilers: GCC ≥ 9, Clang ≥ 9,
     MSVC ≥ 19.25 and IntelLLVM. The same `if consteval` pattern is in
     `Kokkos_BitManipulation.hpp` (`dispatch_helper`).
2. **The runtime path is unchanged.** It still uses `Kokkos::fabs`/`isnan`/`copysign`,
   which A1 verified on A100. The portable helpers are used *only* during constant
   evaluation. Smith's algorithm moved unchanged into
   `Impl::complex_divide_smith<Fabs>`, so the two paths share it.
3. **Compile-time scope: finite operands and a nonzero divisor only.**
   - Compilers reject constant-evaluated floating-point operations that produce a NaN
     (Clang: `0/0`, `inf − inf`, `inf·0`) or divide by zero (GCC, nvcc). So the IEEE code,
     including the Annex G recovery, cannot simply run at compile time with special
     operands.
   - For finite operands and a nonzero divisor, Smith's algorithm never produces a NaN:
     each numerator has one product and the denominators are nonzero. So the compile-time
     path needs no recovery, and the result equals the runtime one up to
     floating-point contraction.
   - Special operands at compile time call a deliberately non-`constexpr` function,
     `Impl::complex_division_by_zero_or_of_inf_or_nan_is_not_a_constant_expression()`.
     Verified with nvcc 12.6: `static_assert(C(1,1)/C(0,0) == ...)` and an infinite
     operand fail with "cannot call non-constexpr function
     `...is_not_a_constant_expression`", while finite division compiles.
   - The difference from libcu++: its constant-evaluation branch returns fixed answers for
     special operands and drops the signs, e.g. always (inf, inf), so its compile-time and
     runtime results can differ. Kokkos refuses instead.
   - Possible follow-up, if special values are needed at compile time: IEEE-emulating
     `constexpr` add/mul/div wrappers that return NaN/inf constants for invalid
     operations. The sign of zero then needs `__builtin_bit_cast`, which GCC 10.4 (the
     Kokkos minimum) lacks; nvcc 12.6 accepts it in `constexpr` device code.
4. **Only `fabs` is needed at compile time.** It is used only in comparisons, so
   `x < 0 ? -x : x` is enough, and the sign of a zero result does not matter. No
   `constexpr` `isnan`/`copysign`/`bit_cast` is needed, and the GCC 10.4 limitation
   above does not arise.
5. **`constexpr` added:**
   - `Impl::complex_divide` and `complex_divide_smith`;
   - `operator/(complex, complex)` and `operator/(scalar, complex)`.

   The member `operator/=(const complex&)` was already marked `constexpr`, and is now
   genuinely usable in constant expressions. This resolves the C++20 "no constant
   specialisation" problem noted in A3.

Tests (`TestComplex.hpp`):
- `test_division_in_constant_expression<float|double>()` covers:
  - both Smith branches (`|c|<|d|` and `|c|>=|d|`), with exact results;
  - `scalar / complex` and `/=`;
  - the halving guard (`(max/2,max/2)/(max/2,max/2) = (1,0)`);
  - the scale-up guard (a divisor at `norm_min`, result `(0.3, 0.1)/m` within bounds).
- Mixed precision: `complex<float> / complex<double>`.
- Runtime TEST `complex_division_constexpr_matches_runtime`: a compile-time quotient equals
  the runtime one (inputs read through `volatile`) within 2 eps.
- The special-value refusal was checked manually (above). It cannot be a test: a
  `static_assert` on a non-constant expression does not compile.

**A4. Missing API: `arg`, `proj` (C11, D8).** Add `arg(complex<T>)` = `atan2(imag, real)`
and `proj(complex<T>)` (infinities → `(inf, copysign(0, imag))`), plus scalar `arg`/`proj`
for arithmetic types via `Impl::promote_t`, as for `conj`/`norm`. Tests against
`std::arg`/`std::proj` for float/double.

*A4 status (2026-11-20): verified on A100 (job 10377921): 15/15 `*complex*:*atomic*` tests
pass on Serial and Cuda; 0 errors/warnings in the nvcc build; committed.*

The code agreed with the plan: neither function existed anywhere in Kokkos. Three
precisions were added:
- **Scalar semantics come from the standard.** [cmplx.over] treats a real argument as
  `complex<T>(x)`:
  - `arg(x)` is `atan2(+0, x)`, so `arg(-1) = π` and also `arg(-0.) = π`;
  - `proj(x)` is `proj(complex(x))`, so `proj(-inf) = (inf, +0)`;
  - integers promote to `double`.

  libc++ and libcu++ (§3b) do the same.
- **Not `constexpr`:** `std::arg`/`std::proj` are not `constexpr`, and `Kokkos::atan2`,
  `isinf` and `copysign` are not either, so marking them would hit the C++20 rule
  noted in A3.
- **nvcc pitfall:** passing `Kokkos::infinity_v<T>` directly to the
  `complex(const RealType&, ...)` constructor binds a reference to it ("odr-uses" it).
  That is a host-only variable, so it gives "identifier `Kokkos::infinity_v<double>` is
  undefined in device code". Copy it into a local `constexpr` first, as
  `complex_divide` does. The same applies to every numeric trait used in A5, A6 and
  CH3.

Overloads: `arg(complex<T>)`, `arg(Arithmetic)`, `proj(complex<T>)` and
`proj(Arithmetic)`. The scalar ones are removed by SFINAE for non-arithmetic arguments,
through the `promote_t` return type, as for `conj`/`norm`.

Tests:
- `complex_arg_proj` runs on the test execution space for float and double, over 13
  inputs (signed zeros in all four quadrants, infinities, `(-inf,-inf)`, `(-0,-3)`).
  - `arg` must match `std::arg` within a few eps, with the same sign: device `atan2` may
    differ from host libm in the last bit.
  - `proj` must match `std::proj` exactly, including the sign of zero.
  - The scalar overloads are checked on the device as well: `arg(-2)`, `arg(-0.)`,
    `arg(3)`, `proj(-inf)`, `proj(2)`.
- `complex_operations_arithmetic_types_overloads` gains the return types of `arg`/`proj`
  for integer, float, double and long double arguments.

**A5. Annex G recovery in `operator*` (C21; from §3b).** Implement it once in an
`Impl::complex_multiply` helper (as `complex_divide`), used by `operator*` and `*=`. After
the naive formula, if both parts are NaN, recompute the way `__muldc3`/libc++/libcu++ do:
replace infinite operands by `copysign(1 or 0)` and NaN partners by `copysign(0)`, set
`recalc`, and multiply by `inf`.
- Cost: one `isnan(x) && isnan(y)` test on the common path.
- Optional: a macro to skip the recovery, like libcu++'s
  `LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_{OPERATIONS,MULTIPLICATION,DIVISION}`. Its name and
  whether it becomes a CMake option (`Kokkos_ENABLE_...`) is a Kokkos-team decision; it
  should cover `/` as well.
- Test: `(inf,nan)*(1,0)` must equal libstdc++'s (inf, NaN), plus the Annex G
  `inf·finite` cases, through `*` and `*=` on device.
- Ordinary values must stay bit-identical. The recovery runs only when both parts are
  NaN, and `complex_basic_math` must pass unchanged.

*A5 status (2026-11-21): verified on A100 (job 10384375): 16/16 `*complex*:*atomic*` tests
pass on Serial and Cuda, including the bit-identity check of ordinary products on the GPU;
0 errors/warnings in the nvcc build; committed.*

**Changes to the plan, and why:**
1. **A compile-time branch was needed.** The plan predates A3, which made `operator*`
   `constexpr` and added `static_assert`s on it. The recovery uses
   `Kokkos::isnan`/`isinf`/`copysign`, which are not `constexpr`, so A5 uses the A3b
   pattern.
   - During constant evaluation (`if consteval` or `__builtin_is_constant_evaluated()`),
     operands must be finite and the usual formula is used.
   - Other operands call
     `Impl::complex_multiplication_of_inf_or_nan_is_not_a_constant_expression()`.
     Verified with nvcc 12.6: `C(inf,0) * C(1,0)` in a `static_assert` fails with that
     name, while a finite product compiles.
   - The A3 `static_assert`s still pass.
2. **Finite operands never reach the recovery,** so compile-time and runtime results
   agree whenever compile-time evaluation is allowed.
   - The recovery runs only when *both* parts are NaN. For finite operands that needs
     `ac` and `bd` to overflow with the same sign while `ad` and `bc` overflow with
     opposite signs.
   - That is impossible: sign(a)sign(c) = sign(b)sign(d) and
     sign(a)sign(d) = −sign(b)sign(c) multiply to 1 = −1.
3. **Bit-identical ordinary results.**
   - `x = a*c − b*d` and `y = a*d + b*c` are kept as single expressions, as before. The
     partial products for the third recovery case (overflowing products) are recomputed
     only inside the rare branch.
   - Storing them once and reusing them in the test would give each product two uses,
     and nvcc fuses a multiply into an FMA only when its result has a single use. So the
     rounding of ordinary values on the GPU could change.
   - The test checks bit-identity against the textbook formula in the same kernel.
4. **Scope:**
   - `Impl::complex_multiply` serves the member `*=(complex)`,
     `operator*(complex, complex)`, including mixed precision (computed in the common
     type, the same as before), and the host-only `operator*(std::complex, complex)`.
   - `complex * scalar` and `scalar * complex` stay componentwise. `std::complex` does
     no recovery for a real factor either.
5. **No opt-out macro.** A libcu++-style `..._SIMPLIFIED_COMPLEX_...` switch remains a
   Kokkos-team decision. The common-path cost is the two `isnan` tests.

Test `complex_multiplication` (float and double, `*` and `*=`, on the test execution space):
- the three recovery branches: an infinite left operand (`(inf,nan)*(1,0)` and
  `(inf,inf)*(1,0)`), an infinite right operand (`(1,0)*(inf,nan)`), and overflowing
  partial products (`(max/4,nan)*(max/4,1)` = (inf, inf));
- a case that must stay (NaN, NaN): `(nan,0)*(inf,0)`;
- comparison with libstdc++ `std::complex` by value and sign, or both NaN, and with the
  recovered values directly;
- an ordinary product, bit-identical to the textbook formula;
- `std::complex * Kokkos::complex` on the host.

Before A5, `(inf,nan)*(1,0)` was (NaN, NaN) (host prototype, §3b / C21).

**A6. Special values in elementary functions (C22; from §3b; larger, lower priority).**
- Port the special-value handling of the libc++/libcu++ algorithms, which follow the
  Annex G G.6 tables. Start with `exp`, `sqrt`, `log`, `polar` and `proj` (A4), then the
  inverse hyperbolics (`asinh`, `acosh`, `atanh`) and through them `asin`, `acos`, `atan`.
- Keep the Kokkos formulas for finite arguments unless the libc++ ones are measurably
  better, because `complex_special_funtions` pins today's finite results.
- Each function is one small PR with a test against libstdc++ on the special-value grid
  (`{0, -0, 1, inf, -inf, nan}²`).
- Known library disagreement: `norm(inf, nan)` is NaN in libstdc++ but inf in
  libc++/libcu++. Follow Annex G (inf) only if it is decided explicitly.

*A6 status (2026-11-21): batch 1 (`exp`, `sqrt`, `polar`) verified on A100 (job 10388058:
17/17 tests pass on Serial and Cuda, including the 81-input grid for float and double on the
GPU and the unchanged `complex_special_funtions`); committed. Batch 2: see below.*

**Measurement first.** A host program built against the real `Kokkos_Complex.hpp` (the
build's nvcc_wrapper and flags, no Kokkos runtime) compares every function with
libstdc++ on the grid `{0, −0, 1, −1, 0.5, 2.5, inf, −inf, NaN}²` (81 inputs). On Linux,
libstdc++ forwards these functions to glibc's C99 functions (`cexp`, `csqrt`, …), so it
is the Annex G reference.

For each function, the table counts the results whose class differs from libstdc++
(NaN vs number, inf vs finite).

| Function | Before | After batch 1 | Note |
|---|---|---|---|
| `exp` | 10 | **0** (signs too) | |
| `sqrt` | 14, plus 3 sign errors | **0** (signs too) | the sign errors were on the branch cut |
| `log`, `log10` | **0** | 0 | already correct |
| `sin`, `cos`, `sinh`, `cosh` | 18 each | 18 | batch 2 |
| `tan`, `tanh` | 26 each | 26 | batch 2 |
| `asin`, `acos` | 34 each | 30 and 27 | batch 3 |
| `atan`, `atanh` | 34 each | 34 | batch 3 |
| `asinh` | 34 | 26 | batch 3 |
| `acosh` | 22 | 6 | batch 3; improved through `sqrt` |

The float grid gives the same result for `exp`, `sqrt` and `log`.

**Corrections to the plan, from this measurement:**
1. **`log` needs no change.** `log(abs(z)) + i·atan2(imag, real)` with `hypot` already
   gives the Annex G values: `log(0,0) = (−inf, 0)`, `log(inf, NaN) = (inf, NaN)`.
2. **New gap C23: wrong side of the branch cut for finite inputs with a signed-zero
   imaginary part.**
   - `sqrt(−1 − 0i)` was (0, **+1**); libstdc++ gives (0, −1). `sqrt` tested `i < 0`,
     which is false for −0.
   - The inverse functions inherit this, e.g. `asin(2.5 + 0i)` and `acos(2.5 + 0i)` had
     the wrong sign of the imaginary part.
   - This affects finite results, not only special values, so it matters more than C22.
   - Fixed for `sqrt` (`copysign(…, i)`). The inverse functions are batch 3, re-measured
     after it.

**Batch 1 changes** (finite non-special results unchanged: same operations as before):
- **`exp`:** zero imaginary part → `(exp(re), imag)`, which keeps the sign of the zero
  and gives `exp(inf + 0i) = (inf, 0)` instead of (inf, NaN). Then the infinite real
  part, as in libc++:
  - `(−inf, inf or NaN)` → (±0, ±0). The signs are unspecified; `copysign(1, y)` matches
    glibc's (0, −0) for `−inf`, where libc++ uses (0, +0).
  - `(+inf, inf or NaN)` → (±inf, NaN).
- **`sqrt`:** `imag = ±inf` → (inf, imag); `re = +inf` → (inf, ±0) or (inf, NaN);
  `re = −inf` → (+0, ±inf) or (NaN, ±inf). The finite formula is unchanged except that
  the sign comes from `copysign(…, i)`, which fixes C23.
- **`polar`:** `polar(inf, ±0)` gave (inf, NaN), because `inf·sin(0)` is NaN. NaN parts
  are now replaced by 0, with the sign of `theta` for the imaginary part. This follows
  libc++ and libcu++.
  - **Library disagreement:** libstdc++ computes `inf·sin(0)` literally and returns
    (inf, NaN). That case is within `std::polar`'s precondition (rho non-negative and not
    NaN, theta finite), and the value with magnitude inf and phase 0 is (inf, 0). So the
    test checks the expected values rather than libstdc++ here.
  - `KOKKOS_EXPECTS(r >= 0)` is kept. Negative or NaN `rho` and infinite `theta` are
    precondition violations in the standard, so they are not handled.
- Same nvcc pitfall as A4: `polar(const T&, …)` with a static `constexpr` member as
  argument odr-uses a host variable in device code, so the test uses a local copy.

Test `complex_exp_sqrt_log_special_values` (float and double, on the test execution space):
- `exp`, `sqrt` and `log` on the whole 81-input grid. Each part must have the same class
  and sign as libstdc++, and finite parts must agree within 8 eps, since device math may
  differ in the last bits.
- `sqrt(−1 ∓ 0i) = (0, ∓1)` on both sides of the cut.
- `polar(inf, ±0) = (inf, ±0)`, `polar(inf, 1) = (inf, inf)`, and an ordinary `polar`.

*A6 batch 2 (2026-11-21): `sinh`, `cosh`, `tanh`, and `sin`, `cos`, `tan` through the
identities; verified on A100 (job 10393803: 19/19 tests pass on Serial and Cuda, including
the unchanged `complex_special_funtions`); committed.*

The grid tool and the accuracy programs now live in
`/work/.../tmp/claude_builds/complex_a6/` (`build.sh <name>` builds `<name>.cpp` against
the real header). They were in `/tmp` before, which was purged during this step.

Result on the 81-input grid, float and double:

| Function | Before | After batch 2 |
|---|---|---|
| `sinh`, `cosh`, `tanh`, `tan` | 18, 18, 26, 26 wrong-class results | **0**, signs included |
| `sin`, `cos` | 18 each | **0** wrong-class; 6 and 2 sign differences, all unspecified by Annex G (see below) |

**Changes:**
- **`sinh`, `cosh`:** special values first (G.6.2.4/5), then the unchanged finite formula.
  - libc++ is **wrong** for `cosh(−0 ± 0i)`. It returns (1, imag), but `cosh` is even
    and conjugate-symmetric, so the zero's sign is sign(re)·sign(im). The plain formula
    `sinh(±0)·sin(±0)` already gives that sign, so that libc++ special case was dropped.
  - `cosh(±inf ± 0i)` likewise gets imaginary part `copysign(0, re)·im`.
  - Where Annex G leaves a sign unspecified, glibc's choice is used, since it is the test
    reference: `sinh(−inf + i·inf or NaN)` has +inf, and `cosh(±0 + i·inf or NaN)` has +0.
- **`sin`, `cos`:** computed as `−i·sinh(iz)` and `cosh(iz)`, which is how Annex G defines
  them. For finite arguments these are the same products as before, so results are
  bit-identical.
  - The remaining sign differences are all Annex G "unspecified" cases:
    `sin(±inf + i·inf)`, `sin(NaN ± i·inf)` and `cos(±0 + i·NaN)`.
  - glibc's `csin`/`ccos` choose those signs independently of its `csinh`/`ccosh`, so no
    implementation can match both glibc's `sinh` and glibc's `sin` there. Following the
    identity is the consistent choice.
  - The test checks `sin`/`cos`/`tan` against glibc for class and value, and bitwise
    against the identities.
- **`tanh`:** special values (G.6.2.6), plus three additions libc++ lacks:
  - `tanh(±0 + i·inf or NaN) = (±0, NaN)`, required since C11;
  - a real argument gives `(tanh(re), ±0)`;
  - `tanh(NaN + iy) = (NaN, NaN)` for `y ≠ 0`.

  In the finite path, **Kahan's formula replaces `sinh(z)/cosh(z)`**; see C24 below. It
  is the formula glibc uses:
  (sinh x·cosh x + i·sin y·cos y) / (sinh²x + cos²y).

  Beyond (max_exponent − 1)·ln 2 / 2 (≈ 354.9 for double, ≈ 44.0 for float),
  `(±1, 4·sin y·cos y·e^{−2|x|})` is used.
- **`tan`:** computed as `−i·tanh(iz)`, so it inherits all of the above.
- `Kokkos_Complex.hpp` now includes `Kokkos_MathematicalConstants.hpp` (for `ln2_v`)
  instead of relying on `Kokkos_Core.hpp`'s include order (cf. C2).

**New gap C24 (accuracy of finite `tanh`/`tan`, fixed).**
- `sinh(z)/cosh(z)` cancels once `sinh ≈ cosh`. `tanh(20 + i)` had an imaginary part
  of 5.2e-17 instead of 7.7e-18; float `tanh(8 + i)` was 16% off in the imaginary part.
  Beyond |x| ≈ 710 it gave NaN (inf/inf).
- My first comparison used a *normwise* error, which hides a tiny component being wrong
  (it reported 2.8 eps). The componentwise measurement:

| Range | `sinh/cosh` (before) | libc++ formula | Kahan (now) |
|---|---|---|---|
| \|x\|, \|y\| < 2 | (3.8, **40.8**) eps | (4.9e4, 4.9e4) | (2.7, 3.2) |
| x ∈ [5, 40] | (3.4, **9.2e33**) | (2.5, 1.8) | (3.0, 3.2) |
| near the pole iπ/2 | (4.2, 4.6) | (5.1e10, 5.1e10) | (2.9, 3.1) |

- libc++/libcu++'s `(sinh 2x, sin 2y)/(cosh 2x + cos 2y)` is unusable near the poles,
  so `cuda::std::tanh` has this problem (§3b).
- After the change, `tanh(20 + i)` and a point near the pole match glibc exactly.

Tests:
- `complex_hyperbolic_special_values` (float and double, on the test execution space):
  - `sinh`, `cosh`, `tanh` on the whole grid, for class, sign and value;
  - `sin`, `cos`, `tan` for class and value, and bitwise equal to their identities.
- `complex_tanh_large_argument`:
  - `tanh(±800 ± i)` and `tan(1 + 800i)` (100 for float) give (±1, ±0) and (0, 1);
  - `tanh(20 + i)` (float: 8 + i), componentwise within 8 eps of glibc;
  - a point near the pole iπ/2, componentwise within 8 eps.

*A6 batch 3 (2026-11-21): the inverse functions. The first A100 run (job 10393972) failed
on Cuda only; fixed (below), rerun job 10394120: 20/20 tests pass on Serial and Cuda;
committed. A6 is complete.*

**Failure in job 10393972, and the portability lesson.**
- `cuda.complex_inverse_special_values` failed for `atan(±inf + i·NaN)`: the GPU gave
  imaginary part −0, while Serial and glibc gave +0.
- `atan(z) = −i·atanh(iz)` passes `−NaN` as the real part of `iz`, and `atanh`'s branch
  for an infinite imaginary part returned `copysign(0, xr)`, which reads the **sign bit
  of a NaN**.
- On x86, negating a NaN flips that bit. On NVIDIA GPUs, NaN results of arithmetic are
  unspecified and may be canonical (PTX `neg`), so host and device chose different signs.
- The sign is unspecified by Annex G (`catanh(NaN + i∞) = ±0 + iπ/2`), so neither was
  wrong, but a result must not depend on NaN sign bits.

The fix and audit:
- **Fixed:** `atanh` returns +0 there for a NaN real part, and `cosh`'s `(NaN, ±0)` branch
  returns `i` instead of `copysign(0, r)·i`, the same issue reached via `cos = cosh(iz)`.
- **Audited:** all functions that receive `iz = (−imag, real)` (`sinh`, `cosh`, `tanh`,
  `asinh`, `atanh`). The other sites either exclude a NaN real part first or only
  propagate it. Remaining `copysign(…, NaN)` uses only read the sign of a NaN that came
  directly from the caller's input, which host and device see identically.
- **Test:** `atan` now joins `asin`, `sin` and `cos`. It is compared with glibc for class
  and value; its unspecified signs are fixed by the bitwise identity check.
- **Grid:** `atan` has 2 deterministic sign differences from glibc, both unspecified, and
  `cos` now has none.
- **General rule for A7 and Part C:** never let a returned zero's sign depend on
  `copysign(…, x)` for an `x` that may be a NaN produced inside Kokkos.

**Measurement before the change.** Besides wrong special values, the inverse functions
were badly inaccurate for *ordinary finite* inputs. The table gives the worst
componentwise error against glibc long double, as (real / imaginary); X means some
results were not finite.

| Regime | `asinh` | `acosh` | `atanh` | `asin` | `acos` | `atan` |
|---|---|---|---|---|---|---|
| \|re\|, \|im\| < 2 | 3.1e3 / 7 | 2.1e3 / 1.9 | 2e3 / 9.1 | 6.8 / 9.3e3 | 1.9e3 / 1.9e3 | 22 / 7.3e3 |
| tiny (1e-8) | 4.6e11 / 1.5 | 2.5e12 / 1.2 | 2.8e11 / 0.8 | 1.2 / 1.7e12 | 0.5 / 1.7e12 | 0.8 / 7.2e12 |
| large (1e8) | 1.6e15 / 7.6e18 X | 0.6 / 2 | 5e12 / 0.3 | 8.9e18 / 1.6e15 X | 1.4e18 / 1.4e15 X | 0.3 / 2.6e11 |
| huge (1e200) | X | 0.3 / 1.9 | X | X | X | X |
| near ±1, ±i | 8.7e6 / 650 | 4.8e6 / 2.6 | 7.5e10 / 1.1e7 | 540 / 8.6e6 | 9e6 / 9e6 | 1.1e7 / 1.2e10 |

libc++/libcu++ use the same naive log formulas (`log(z + sqrt(z² ± 1))`,
`log((1+z)/(1−z))/2`), so `cuda::std` has the same problem, and following the plan's
"port libc++" would not fix accuracy.

**Decision (user, 2026-11-21):** batch 3 fixes special values and branch cuts, plus the
cheap symmetry fix. Accuracy is recorded as **C25** and scheduled as **A7**.

**Changes:**
- **`asinh`, `atanh`** (odd and conjugate-symmetric) get the libc++ special values first.
  Ordinary values come from the existing Kokkos formula evaluated on (|x|, |y|), with
  signs restored by `copysign`. That makes the branch cuts follow the sign of zero (C23),
  and removes the cancellation of `z + sqrt(z² + 1)` for negative real parts. At 1e8,
  `asinh` went from 1.6e15 / 7.6e18 X to 0.5 / 2.
- **`acosh`** (conjugate-symmetric, real part ≥ 0) gets the special values, then the
  existing Kahan formula on (x, |y|), giving (|re|, `copysign(im, y)`).
- **Exact axis cases**, which the complex formulas computed as log(≈1) = 1e-16 instead of
  an exact zero:
  - `asinh(x ± 0i)` = (asinh x, ±0);
  - `asinh(±0 + iy)` = (±0, asin y) for |y| ≤ 1;
  - `acosh(x ± 0i)` = (+0, ±acos x) for |x| ≤ 1, and (acosh x, ±0) for x > 1.
- **`asin`** stays `−i·asinh(iz)`, and **`atan`** becomes `−i·atanh(iz)`. That replaces
  Kokkos's own `atan` formula, which is the same mathematics; the accuracy table is
  unchanged in class.
- **`acos`** gets the special values, then `π/2 − asin(z)`, which inherits the corrected
  branch cuts of `asin`.
- **Unspecified signs, set to glibc's choice:** `asinh(NaN ± i·inf)` gives +inf, and
  `acos(±inf + i·NaN)` gives −inf. `acosh(±0 + i·NaN)` = (NaN, π/2), as in glibc.
- **`Kokkos::numbers::pi_v`** comes from `Kokkos_MathematicalConstants.hpp`, which batch 2
  already includes.

**Result** on the 81-input grid, float and double: **0 wrong-class results** for all six
functions, signs included. The exception is two `asin` signs that Annex G leaves
unspecified, `asin(±inf + i·NaN)`. glibc's `casin` chooses them independently of its
`casinh`, so no implementation matches both (as for `sin` in batch 2). The finite grid
values are within 3 eps of glibc.

The accuracy table is the same after batch 3 except for these entries:

| Entry | Before | After |
|---|---|---|
| `asinh` at 1e8 | 1.6e15 / 7.6e18 X | 0.52 / 2 |
| `asin` at 1e8 | 8.9e18 / 1.6e15 X | 1.9 / 0.52 |
| `acos` at 1e8 | 1.4e18 / 1.4e15 X | 1.2e3 / 0.52 |
| `atan` at 1e8, imaginary part | 2.6e11 | 4.3e12 |
| `asinh` at \|z\| < 2, real part | 3.1e3 | 9.8e3 |

The last two got worse, but they are in the same cancellation class, log(1 + small), that
C25 covers.

Test `complex_inverse_special_values` (float and double, on the test execution space):
- the six functions on the grid: class, sign and value within 16 eps (`asin` without
  sign), and `asin`/`atan` bitwise against their identities;
- both sides of each branch cut: `asin`/`acos(2.5 ± 0i)`, `atanh(2 ± 0i)`,
  `acosh(−2 ± 0i)`, `asinh(−0 + 2.5i)`;
- `asinh(−1e8 + 0.5i)`. On the host all of these match glibc to within 0.76 eps.

**A6 is then complete for special values.** The remaining finite inaccuracy of the
inverse functions is A7.

**A7. Accurate inverse functions (C25; scheduled 2026-11-21).**
- Replace the finite formulas of `asinh`, `acosh`, `atanh` (and through the identities
  `asin`, `acos`, `atan`) by an algorithm that is accurate in every regime: tiny |z|
  (`log1p`), huge |z| (no `z²` overflow), and near the branch points ±1, ±i.
- Candidates: Hull, Fairgrieve and Tang (1997), as implemented in FreeBSD/musl
  `catrig.c` (BSD-licensed; porting it needs the Kokkos team's OK), or glibc's
  `__kernel_casinh` (LGPL, not portable into Kokkos). Alternatively, write it from the
  paper.
- Keep the batch 3 special-value handling and symmetry wrappers. Measure with
  `claude_builds/complex_a6/invacc.cpp` (the regimes above); target a few eps
  componentwise everywhere.

*A7 status (2026-11-21): verified on A100 (job 10394308): 21/21 tests pass on Serial and
Cuda, including `complex_inverse_accuracy` on the GPU; committed. Part A is complete.*

**Route:** written from the published papers. No implementation's code (FreeBSD/musl
`catrig.c`, glibc) was consulted or copied, so there is no license question.
- **`asin`/`acos` core:** `Impl::complex_asin_acos(x, y, re_asin, re_acos, im)` for
  finite x, y ≥ 0, after Hull, Fairgrieve and Tang (ACM TOMS 23, 1997).
  - With r = |z+1|, s = |z−1|, A = (r+s)/2 and B = x/A, the real parts are `asin(B)` and
    `acos(B)`. Above B = 0.6417 they become `atan` of a ratio computed without
    cancellation (one form each for x ≤ 1 and x > 1).
  - The imaginary part is `log1p(Am1 + sqrt(Am1·(A+1)))` up to A = 1.5, with A−1 formed
    without cancellation. Beyond that it is `log(A + sqrt((A−1)(A+1)))`, or
    `log A + ln 2` for A > 1/eps.
  - For x or y > max/4, where r+s could overflow, `(atan2(x,y), log(2|z|))` is used.
  - For y small compared with 1−x, `im = y/√(1−x²)`.
- **Used by:** `asinh` (the core at (|y|, |x|)), `acosh` (at (|x|, |y|), with
  `acos(−z) = π − acos(z)`) and `acos` directly. `asin` and `atan` stay as identities.
- **`atanh`:** Kahan (1987).
  - re = ¼·log1p(4a/((1−a)² + b²)) and im = ½·atan2(2b, (1−a)(1+a) − b²), where 1−a is
    exact near 1.
  - For a or b > 1/eps: (a/|z|/|z|, π/2).
- **Kept from batch 3:** the special-value branches, the symmetry wrappers and the exact
  axis cases. The grid result is unchanged (only the four unspecified signs).

**One bug found by the wider measurement, and fixed.**
- My first small-y condition, y² < eps·(1−x²), allowed `y/√(1−x²)` where it is invalid.
  At `asin(1 − 1.1e-16 + 2e-16·i)` the imaginary part was 1.32e-8 instead of 1.07e-8,
  1e15 eps off.
- The approximation's relative error is of order (y/(1−x))², so the correct condition
  is y² < eps·(1−x)².
- That still avoids the underflow of y²: since 1−x ≥ eps/2, the right-hand side cannot
  underflow.
- The first measurement (offsets ≥ 1e-6 from the branch points) did not catch it; the
  wider one (offsets down to 1e-15) did.

**Result: at most 2.4 eps componentwise everywhere, for float and double.** Before A7 it
was up to 9e18 eps, or not finite. Measured on the host with `invacc.cpp` and
`invacc2.cpp` against glibc long double, in these regimes:
- \|z\| < 2, tiny (1e-8, 1e-300 / 1e-35), large (1e8), huge (1e200);
- log-uniform exponents over the whole range;
- offsets 1e-15 … 1e-1 from ±1 and ±i;
- both sides of every cut.

| Regime (double) | Before A7 (worst of the 6 functions) | After A7 |
|---|---|---|
| \|z\| < 2 | 9.8e3 | 1.9 |
| tiny | 7.2e12 | 1.8 |
| large (1e8) | 4.3e12 | 1.8 |
| huge (1e200) | non-finite | 2.0 |
| near ±1, ±i | 1.1e10 | 2.1 (offsets ≥ 1e-6); 2.3 (offsets down to 1e-15) |

The non-finite `atanh` results on the cut are all at exactly |x| = 1, a pole, where
(±inf, ±0) is correct.

Test `complex_inverse_accuracy` (float and double, on the test execution space):
- 64 fixed inputs per regime (\|z\| < 2, tiny, huge, large, near the branch points),
  plus the near-branch-point case above;
- all six functions, componentwise within 8 eps of `std::complex<long double>`. On the
  host the worst case is 1.6 eps; 8 eps leaves room for device math differences.

**C25 is resolved by A7,** and `cuda::std::complex` (§3b) has the inaccurate formulas
that Kokkos no longer has.

Cost: the core makes 2 `hypot` calls, plus one `asin`/`acos` or `atan` call and one
`log1p` or `log` call. That is comparable to before (complex `sqrt` and `log`, which
themselves use `hypot`, `sqrt`, `log` and `atan2`), but not benchmarked on the A100.

Not in Part A: the converting-ctor explicitness for float/double (C6). Kokkos is
implicit for `complex<float>` ← `complex<double>` while `std` is explicit. Changing it
breaks users and needs a deprecation cycle. Record it as a known divergence; the
half-specific part is CH2.

**A8. Deprecate the `std::complex` interop except the converting constructor (D11;
planned 2026-11-22; IMPLEMENTED 2026-11-24).** Independent of Parts B and C; tested with float/double.

*A8 status (2026-11-24): implemented as designed below (`d7304e50c`); verified here,
on the A100, and on the std half backend (Aurora).*
- **Verified:**
  - host, GCC 12 (`half_t` = float): deprecated code ON: 27 `*complex*:*atomic*` tests
    pass (25 before, plus the two new ones), 0 warnings; OFF (Kokkos 6 state): 26 pass,
    0 warnings, so nothing outside the deprecated test needs the removed interop;
  - CUDA: the complex test objects compile with 0 `warning #` (nvcc honours the
    suppression); A100 job 10415439: 27/27 on Serial and on Cuda, including
    `complex_std_interop_deprecated` for the native `half_t`/`bhalf_t`;
  - std half backend, Aurora, GCC 14.2, `d7304e50c`, the four builds of
    `run_std_half_fp16.sh` each: job 9121388 (deprecated code ON): 0 warnings,
    `*complex*:*atomic*` 27 / `*numeric_traits*` 15 / `*half*` 14 pass, including
    `complex_std_interop_deprecated` for `half_t`/`bhalf_t` through libstdc++'s
    `std::complex<half_t>`; job 9121428 (`DEPRECATED_CODE=OFF`, fresh
    `build_std_half_<config>_nodep`): 0 warnings, 26 / 15 / 14 pass. So the deprecated
    interop works until Kokkos 6, no deprecated use escapes the suppressed blocks, and
    the core builds and passes without it.
- **Code** (`Kokkos_Complex.hpp`): I1, I4, I5, I6, I7 under
  `#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5` with `KOKKOS_DEPRECATED_WITH_COMMENT`
  naming the replacement. I2 and I3: the non-template overloads are removed; under
  `KOKKOS_ENABLE_DEPRECATED_CODE_5`, deprecated templates take a `std::complex<U>` with
  `U != RealType` that converts implicitly to `std::complex<RealType>`. I0 unchanged.
- **Probe, real header** (GCC 12, `a8probe.cpp`, the 21 cases of the table below):
  deprecated code ON: cases 1-6 and 19-21 compile silently, 7-16 with exactly one
  deprecation warning each, 17-18 do not compile (as today); deprecated code OFF
  (`-DKokkos_ENABLE_DEPRECATED_CODE_5=OFF`): exactly 7-16 fail. As designed.
- **Tests** (`TestComplex.hpp`):
  - `complex_construction`'s interop block is gone; its checks moved to the two new
    tests;
  - `CHECK_COMPLEX_NEAR` converts a `Kokkos::complex` reference from its parts
    (`complex_near_ref`);
  - `Kokkos::exp(std::complex)` became `Kokkos::exp(Kokkos::complex<double>(a))`;
  - the `std::complex * Kokkos::complex` checks (multiplication recovery,
    `static_assert`) are under `KOKKOS_ENABLE_DEPRECATED_CODE_5` with
    `KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH/POP`;
  - new `complex_std_interop_converting_ctor` (float, double): I0 in copy-init,
    direct-init, as a call argument, `= += -= *= /=`, and the braced assignment;
  - new `complex_std_interop_deprecated` (deprecated code only, warnings suppressed):
    I1 (copy-, direct-init, assignment), I6, I4/I5 both orders, I7 (float, double; with
    a tolerance, since its textbook formula differs from `exp(complex)`), and the mixed
    I2/I3 (double from `std::complex<float>`); for `half_t`/`bhalf_t` too where they
    are native types.
  - The test file compiles without a warning with deprecated code ON and OFF (GCC 12).
- **CHANGELOG: not edited.** Reconciled with the repository: Kokkos writes
  `CHANGELOG.md` at release time from the merged PRs (every entry carries a PR number;
  this branch has never edited it). The entry for the PR description instead:
  "Deprecate the interop between `Kokkos::complex` and `std::complex` except the
  converting constructor: the conversion operator to `std::complex`, `==`/`!=` with a
  `std::complex` operand, `std::complex * Kokkos::complex`, `exp(std::complex)`, and
  assignment and division assignment from a `std::complex` of another value type".
- **Script:** `run_std_half_fp16.sh` takes `DEPRECATED_CODE=OFF` (build directories
  `..._nodep`) for the Kokkos 6 state on the std half backend.

*Inventory* (`core/src/Kokkos_Complex.hpp` at `a6ad59faa`; the only other user in the
repository is `core/unit_test/TestComplex.hpp`):

| # | Interop | Line | Today | After removal | A8 |
|---|---|---|---|---|---|
| I0 | `complex(const std::complex<RealType>&)` (implicit, host+device) | `:426` | — | — | **keep** |
| I1 | `operator std::complex<RealType>() const` (implicit, host) | `:442` | `SD s = kz;`, `sd = kz;`, `SD s(kz);` | all three break | deprecate |
| I2 | `operator=(const std::complex<RealType>&)` (host) | `:481` | `kz = sd;`, and mixed `kz = sf;` | same type still compiles (I0 + copy assignment); **mixed breaks** (two user-defined conversions) | replace by a deprecated template for the mixed case only |
| I3 | `operator/=(const std::complex<RealType>&)` (host, `constexpr`) | `:561` | `kz /= sd;`, mixed `kz /= sf;` | same type still compiles (I0 + member `/=`); **mixed breaks** | same as I2 |
| I4 | `operator==(std::complex<R1>, complex<R2>)` and reverse | `:659`, `:668` | `sd == kz`, `kz == sd` | break | deprecate |
| I5 | `operator!=(std::complex<R1>, complex<R2>)` and reverse | `:710`, `:719` | `kz != sd` | break | deprecate |
| I6 | `operator*(std::complex<R1>, complex<R2>)` (host, `constexpr`) | `:844` | `sd * kz` | breaks (no reverse overload exists either) | deprecate |
| I7 | `exp(const std::complex<RealType>&)` → `Kokkos::complex` | `:1616` | `Kokkos::exp(sd)` | breaks | deprecate |

Not interop, unchanged: stream I/O builds `std::complex` objects internally (A2), and
the tests use `std::complex` as a reference.

*Measured* (GCC 12, C++20, `-fsyntax-only` against the real header; probe
`tmp/claude_builds/complex_a6/a8probe.cpp`, one case per `-DCASE=n`; "removed" = I1–I7
deleted, "A8" = the design below, both as edited copies of the header):

| Expression (`KD = Kokkos::complex<double>`, `sd`/`sf` = `std::complex<double/float>`) | today | removed | A8 |
|---|---|---|---|
| `KD a = sd;`, `take(sd)` with `take(KD)`, `KD a(sf);` (I0) | ok | ok | ok, no warning |
| `kz = sd;`, `kz /= sd;`, `kz += sd; kz *= sd;` | ok | ok | ok, no warning |
| `kz = sf;`, `kz /= sf;` | ok | error | ok, 1 deprecation warning |
| `sd == kz`, `kz == sd`, `kz != sd`, `sd * kz`, `Kokkos::exp(sd)` | ok | error | ok, 1 warning |
| `SD s = kz;`, `sd = kz;`, `SD s(kz);` | ok | error | ok, 1 warning |
| `kz * sd`, `kz + sd` | error | error | error (unchanged) |
| `kz = {3.0, 4.0};`, `kz /= {3.0, 4.0};` | **error (ambiguous with I2/I3)** | ok | ok |

So A8 warns exactly where the Kokkos 6 removal will break code, never where code keeps
compiling, and as a side effect fixes the braced assignment.

*Design:*
- I1, I4, I5, I6, I7: wrap in `#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5` and mark
  `KOKKOS_DEPRECATED_WITH_COMMENT(...)` (warns only with
  `Kokkos_ENABLE_DEPRECATION_WARNINGS`, ON by default). Messages give the replacement:
  - I1: "Use std::complex<T>(z.real(), z.imag()) instead.";
  - I4/I5: "Convert the std::complex operand to Kokkos::complex.";
  - I6: "Use Kokkos::complex(x) * y instead.";
  - I7: "Use Kokkos::exp(Kokkos::complex(z)) or std::exp(z) instead."
- I2, I3: remove the non-template overloads unconditionally (I0 takes over the same-type
  case, with identical results). Under `KOKKOS_ENABLE_DEPRECATED_CODE_5` add deprecated
  templates for the mixed case:
  ```cpp
  template <class U>
    requires(!std::is_same_v<U, RealType> &&
             std::is_convertible_v<const std::complex<U>&, std::complex<RealType>>)
  KOKKOS_DEPRECATED_WITH_COMMENT("Convert to Kokkos::complex<RealType> first.")
  complex& operator=(const std::complex<U>& src) noexcept;   // and operator/=
  ```
  The constraint accepts exactly what the old overload accepted through
  `std::complex`'s implicit widening (`std::complex<float>` → `<double>`).
- I0 stays as is (implicit, `KOKKOS_INLINE_FUNCTION`, `reinterpret_cast` reading).
  Whether it should become `explicit(...)` by the D4 rule is a separate decision
  (Part A note on C6).
- CHANGELOG: under "Deprecations", one entry listing I1–I7 and the replacements.

*Tests* (`TestComplex.hpp`; the prototype gives warnings at exactly these lines):
- `:83-88` (`std::complex<RealType> sa(a)`, `r = a`, `r = sb * a`: I1, I6): rewrite
  without interop, and move one use of each into a deprecated-code test (below);
- `:522` (`CHECK_COMPLEX_NEAR` converts its `ref` to `std::complex<double>`: I1 when
  `ref` is a `Kokkos::complex`): convert explicitly from the parts;
- `:559` (`Kokkos::exp(a)` with a `std::complex<double>`: I7): use
  `Kokkos::exp(Kokkos::complex<double>(a))`;
- `:999` and `:1983` (`std::complex * Kokkos::complex`: I6): move into the
  deprecated-code test;
- `:784-792` (`complex_divide_assign_std_complex`): keep. It now tests the same-type
  path through I0, without a warning; add the mixed case to the deprecated-code test;
- new `complex_std_interop_converting_ctor` (always compiled): I0 in copy-init,
  direct-init, call arguments, `= += -= *= /=` with `std::complex<RealType>`, and the
  braced assignment; float and double only (no new `std::complex<half_t>` tests: it is
  unspecified, and CH4 is the half route);
- new `complex_std_interop_deprecated` under `#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5`,
  between `KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()`/`POP()`: one check per
  I1–I7, including the mixed I2/I3 templates. The I1 and I6 checks moved from
  `complex_construction:79-99` keep `half_t`/`bhalf_t` (under
  `KOKKOS_HAS_NATIVE_HALF_TYPE`/`..._BHALF_...`, as today), so the deprecated paths keep
  working for half until Kokkos 6.

*Verification:*
- A100 (`build_gpu`, deprecated code ON): `*complex*` on Serial and Cuda; the Complex
  objects build without deprecation warnings (the tests push/pop them), nvcc included;
- host build with `-DKokkos_ENABLE_DEPRECATED_CODE_5=OFF`: the tests compile, i.e.
  nothing outside the deprecated-code test needs I1–I7 (the Kokkos 6 state);
- a compile-only check that each deprecated use warns (the probe above with
  `-Werror=deprecated-declarations`, expecting failure).

*Risk:* downstream users (Kokkos Kernels, Trilinos) may rely on I1 to pass
`Kokkos::complex` to host APIs taking `std::complex`. Not checkable here; the
deprecation cycle is what gives them notice.

### Part B: align `half_t` with `std::float16_t` (`core/src/impl/Kokkos_Half_*.hpp`)

**B1. Numeric traits bug (H12).** `Kokkos_Half_NumericTraits.hpp:324-333`:
`digits_v<bhalf_t> = 8`, `digits10_v<bhalf_t> = 2`. Update `TestNumericTraits.hpp`
expectations.

*B1 status (2026-11-21): verified by compilation (Serial1 and Cuda1 NumericTraits objects, and
a negative check); committed.*
- **Fix:** `digits_v<bhalf_t>` 2 → 8 (7 fractional bits plus the implicit 1) and
  `digits10_v<bhalf_t>` 1 → 2 (⌊7·log₁₀2⌋), with corrected comments. The old comment
  "7 − 1 · log10(2)" described neither value.
- **Correction to the plan: there were no expectations to update.** For the half types,
  `TestNumericTraits.hpp` only checks that each trait can be used: it takes its address
  on the host and the device and never compares a value. That is why the wrong value went
  unnoticed.
  - B1 therefore **adds** file-scope `static_assert`s for every integer trait of `half_t`
    and `bhalf_t` (`digits`, `digits10`, `radix`, `min_exponent`, `max_exponent`) against
    the IEEE binary16 / bfloat16 values, which are those of
    `numeric_limits<std::float16_t/std::bfloat16_t>`.
  - They are guarded by `!KOKKOS_HALF_T_IS_FLOAT` / `!KOKKOS_BHALF_T_IS_FLOAT`, like the
    specialisations; otherwise `half_t` is `float`.
- **Audit of all half traits:** infinity, finite min/max, epsilon (2⁻¹⁰ / 2⁻⁷),
  round_error, norm_min, denorm_min, the NaNs, radix and min/max_exponent are all
  correct. Only `digits` and `digits10` of `bhalf_t` were wrong.
- **No runtime effect:** nothing in Kokkos uses `digits`/`digits10` of a half type. The
  other users, in `Kokkos_BitManipulation.hpp` and its tests, apply it to unsigned
  integers only.
- **Verification:** the `Serial1` and `Cuda1` NumericTraits objects compile with 0 errors
  and 0 warnings. Negative check: against the old header, exactly the two `bhalf_t`
  assertions fail.
  - The runtime `numeric_traits_*` tests (in `Kokkos_CoreUnitTest_Serial1`/`Cuda1`) were
    not rerun at the time. They only odr-use the traits, which B1 does not change.
  - **Run on 2026-11-22 (job 10397183), together with H16:** all 71 `Serial1` and 72
    `Cuda1` objects were rebuilt (0 errors, 0 warnings, so the `static_assert`s hold),
    and all 14 `*numeric_traits*` tests pass on Serial and Cuda.
- **New gap H16:** `max_digits10`, `min_exponent10` and `max_exponent10` had no
  specialisation for `half_t`/`bhalf_t`, so `Kokkos::max_digits10<half_t>` had no
  `value`. **Added on request (2026-11-21), committed separately after B1.**

| Trait | Definition | `half_t` | `bhalf_t` |
|---|---|---|---|
| `max_digits10` | ⌈1 + digits·log₁₀2⌉ | 5 | 4 |
| `min_exponent10` | ⌈log₁₀ norm_min⌉ | −4 | −37 |
| `max_exponent10` | ⌊log₁₀ finite_max⌋ | 4 | 38 |

These are the values of `numeric_limits<std::float16_t/std::bfloat16_t>`.
- **Test:** `static_assert`s on the values. `half_t`/`bhalf_t` are also added to the
  runtime `numeric_traits_max_digits10` and `numeric_traits_min_max_exponent10` tests,
  which check the traits are usable on host and device.
- **Verification:** both objects compile with 0 errors and 0 warnings. Negative check:
  without the specialisations, the build fails with "class … has no member `value`".
- **Runtime tests:** run on 2026-11-22 with B1 (job 10397183). All 14
  `*numeric_traits*` tests pass on Serial and Cuda, including `numeric_traits_max_digits10`
  and `numeric_traits_min_max_exponent10` with the half types.

**B2. Scalar I/O (H6, D7 scalar part).** `Kokkos_Half_FloatingPointWrapper.hpp:866-878`.
- Write with `os << static_cast<float>(x)`, which honours flags, precision and locale.
- Read with `double d; if (is >> d) x = d;`. It stops at `)` and `,`, sets failbit instead
  of throwing, and reads through **double, not float**, to avoid the double rounding shown
  in D7.
- Template both on `CharT`/`Traits`. See D7 for the code and the failing examples O1–O5 and
  I1–I5.
- This changes the printed form of `half_t` from `std::to_string` (`3.140625`) to the
  stream format (`3.14062`). Check `TestHalf*` tests and any users that parse the old
  output.
- Tests, for `half_t` and `bhalf_t`: one per row of the D7 table.
  - The output rows compare with `static_cast<float>(h)` printed under the same flags.
  - I1 is `"1 (2) (3,4)"` through `complex<half_t>`.
  - I2–I4 check `fail()`, no exception, and that the value is unchanged.
  - A round trip `os << h; is >> h2; h2 == h` at default precision covers every finite
    half value (all 63488 bit patterns are enumerable).

*B2 status (2026-11-22): done. Verified on the A100 (job 10394586): all 6 `*half*` tests of `Kokkos_CoreUnitTest_Default` pass, including `half_io` and `bhalf_io` (CUDA build, half_t/bhalf_t are real 16-bit types, so the rounding and round-trip checks ran). Whether the comma-locale check ran on the compute node depends on its installed locales (the login node has `de_DE`); if none exists the check is skipped silently.*
- **Code** (`Kokkos_Half_FloatingPointWrapper.hpp`): the two friend operators, as in D7.
  - `<<` is `os << static_cast<float>(x)`.
  - `>>` is `double d; if (is >> d) x = d;`.
  - Both are templated on `CharT`/`Traits`.
  - Includes: `<iosfwd>` and `<string>` are replaced by `<istream>`/`<ostream>`.
    `std::to_string`/`std::stod` were the only uses of `<string>` in the header, and the
    Default test object still compiles.
- **Where the tests live** (the plan said `TestHalf*`): no test of the half stream
  operators existed. The half tests are compiled into `Kokkos_CoreUnitTest_Default`
  (`core/unit_test/default/TestDefaultDeviceType.cpp` includes `TestHalfConversion.hpp`
  and `TestHalfOperators.hpp`; 19 objects). B2 adds `core/unit_test/TestHalfIO.hpp`,
  included there.
  - Nothing else in Kokkos printed or parsed half values, so no user of the old format
    was found.
- **Test `half_io` / `bhalf_io`** (host only, as stream I/O is), one check per row of
  the D7 table:
  - output equal to the float's under 7 formatting states, for 8 values including
    `denorm_min`, 1e-5, −0 and `finite_max` (O1–O4);
  - the stream's own locale (O5);
  - `"2,3"` and `"1.5)"` stopping at the separator (I1);
  - `"x"`, `""` and `"±1e999"`: failbit, no exception, value unchanged (I2–I4);
  - a `while (is >> h)` loop ending normally (I5);
  - wide streams;
  - single rounding of a decimal just above a midpoint, for half and bhalf, which also
    checks `cast_to_half(double)`;
  - a bit-exact round trip of **every finite value** of `half_t` and `bhalf_t` at the
    default precision.
- **nvcc pitfall (independent of Kokkos):** a user-defined `std::numpunct` facet
  segfaults in libstdc++ (`locale::facet::_M_cow_shim`, `dynamic_cast`) when compiled by
  nvcc 12.6 with GCC 12.2. The same file compiled with g++ works. The locale check
  therefore uses a named locale (`de_DE.UTF-8` / `fr_FR.UTF-8`) and is skipped if none is
  installed.
- **Host runs** (the test compiled with the build's flags and gtest, run on the login
  node):
  - with the new operators: both tests pass, including the ~127k-value round trip;
  - with the old operators: wide streams do not compile ("no operator"). Without them,
    110 formatting failures (e.g. `setprecision(3)` printed `3.140625`), the
    `0.000000` check, the locale check, and uncaught `std::invalid_argument` from
    `stod` that abort the input tests.

*B2 finding (2026-11-22): diverges from C++23, and its test fails where `half_t` is
`float`. Decision: option (a), follow the standard; done (see "B2 revision" below).
Verified on the A100 (job 10400777): all 25 `*complex*:*atomic*` tests on Serial and
Cuda, and all 8 `*half*` tests. Also verified where `half_t`/`bhalf_t` are `float`
(GCC 12, C++23, host-only trial of `run_std_half_gcc13.sh`): 25 + 14 + 8 tests pass,
0 warnings.* Found by the trial run of
`run_std_half_gcc13.sh` with GCC 12 in C++23 mode. There `half_t`/`bhalf_t` are `float`
(no native half), which is the default for every host-only build without C++23
`std::float16_t`. `half_io` and `bhalf_io` then fail 3 checks each (`TestHalfIO.hpp:128`):
`"x"` leaves 0, and `"1e999"`/`"-1e999"` leave ±3.40282347e+38, where the test expects 7
unchanged. That is float extraction behaving as specified: since C++11, `num_get` stores 0
on a parse failure and ±max on overflow, with failbit. On the A100 the test passes only
because `half_t` is the wrapper there.

The C++23 rule for the extended types (`[istream.formatted.arithmetic]/4-6`, checked in
the current draft, eel.is/c++draft) is as if:

```cpp
FP fval;  // float if rank <= float (float16_t, bfloat16_t), else double, else long double
use_facet<num_get<...>>(loc).get(*this, 0, *this, state, fval);
if (fval < -numeric_limits<X>::max())      { state |= failbit; val = -numeric_limits<X>::max(); }
else if (numeric_limits<X>::max() < fval)  { state |= failbit; val =  numeric_limits<X>::max(); }
else                                        { val = static_cast<X>(fval); }
```

Note 2 there accepts the double rounding this causes and points to `from_chars` for maximum
accuracy. B2 differs in three ways:
1. It reads through `double`, not `float`. D7 chose `double` to avoid the double rounding,
   but the standard specifies `float`.
2. On a parse failure it leaves the value unchanged; std stores 0.
3. Above half's range it gives inf without failbit (`"1e5"`), or keeps the value
   (`"1e999"`); std stores ±`max()` (65504 for half) and sets failbit.

The CH1 complex proxy reads `std::complex<double>`. `std::complex<std::float16_t>`
extraction reads each part with `is >> float16_t`, i.e. through float with the same
clamping.

Options:
- **(a)** follow the standard exactly (read float, clamp to ±max with failbit, 0 on a
  parse failure), in the wrapper and in the CH1 proxy, and fix the tests;
- **(b)** keep the double reading (more accurate), but fix the test so that it also holds
  where `half_t` is `float`.

Either way the committed test must change: as it is, it fails in every build without a
native half type.

*B2 revision (option (a)):*
- **Scalar extractor** (`Kokkos_Half_FloatingPointWrapper.hpp`): the C++23 fragment.
  - sentry, then `num_get<CharT>::get` into a `float`;
  - beyond ±`max()`: failbit and ±`max()`;
  - otherwise convert (num_get gives 0 on a parse failure);
  - a failed sentry leaves the value unchanged.
- **Complex extractor** (`Kokkos_Complex.hpp`): for half types it no longer reads a
  `std::complex<double>`. It parses `u`, `(u)`, `(u,v)` with the libstdc++ algorithm,
  reading each part with the half extractor. That is what `std::complex<std::float16_t>`
  does, so a part beyond the range fails at that part: the value is unchanged and the
  rest is not read. `Impl::complex_istream_t` is removed; float/double still delegate to
  `std::complex<RealType>` (A2).
- **Insertion is unchanged.** C++23 formats the extended types as if
  `static_cast<double>(val)`, and `os << float` goes through `double` too.
- **`TestHalfIO.hpp`:**
  - errors: `"x"` → 0, `"±1e999"` → ±`max()` (failbit), `""` unchanged;
  - in range: equal to `H(strtof(s))` (compared through the float bits, so −0
    counts);
  - rounding cases: now 1, because through float the input becomes the midpoint,
    which rounds to even;
  - new range cases: half `"±1e5"`, bhalf `"±3.4e38"` → ±`max()`;
  - holds also where `half_t` is `float`.
- **`TestComplex.hpp`:** `complex_io_like_std` compares half against
  `std::complex<float>`; new `testComplexIOHalfRange` checks failure, unchanged value
  and the unread rest for out-of-range parts.

**B3. Treat half as a floating-point type in the scalar promotion traits (H8, D8).**
Specialise `Kokkos::Impl::promote` for `half_t`/`bhalf_t` (`using type = half_t`) in
`Kokkos_Half_MathematicalFunctions.hpp`, and define `promote_2` for half pairs via
`std::common_type`.
- This is safe for the generic math: every template in `Kokkos_MathematicalFunctions.hpp`
  that uses `promote_2_t`/`promote_3_t` is also gated on `std::is_arithmetic_v`
  (e.g. `:195-204`), so no new overloads appear.
- Effect visible immediately: `Kokkos::norm(half_t)` and `Kokkos::arg(half_t)` (after A4)
  compile.
- `conj(half_t)` returns `complex<half_t>`, so it only becomes usable after CH1.

*B3 status (2026-11-22): done. Verified on the A100 (job 10396174): all 22 `*complex*:*atomic*` SmokeTests pass on Serial and Cuda, including the new `complex_half_scalar_promotion`.*
Reconciled with the source before starting; the plan text above was wrong in three
places:
- **`promote_2`/`promote_3` do not follow from `promote`.** Both had their own
  `std::is_arithmetic_v` gate (`Kokkos_MathematicalFunctions.hpp:51-69`), false for the
  half class types, so specialising `promote` alone leaves `promote_2<half_t, …>` empty.
  They are now defined whenever every argument has a `promote_t` and those have a
  `std::common_type`: `common_type_t<promote_t<T>, promote_t<U>>`. That is the P1467
  rule, using the existing `Kokkos_Half_CommonType.hpp` (H7): half + float → float,
  half + int → double (integers promote to double first), half + bhalf → none. The
  old `decltype(promote_t<T>() + promote_t<U>())` is the same for arithmetic types, but
  would have picked a type for half + bhalf through their implicit `float` conversion.
  The callers in the math header keep their own `is_arithmetic` gates, so the generic
  math overloads are unchanged, as the plan said.
- **The mixed `pow` gates needed changing in B3.** `pow(T, complex<U>)` and
  `pow(complex<T>, U)` (`Kokkos_Complex.hpp:806,813`) were gated on `is_arithmetic`. They
  are now gated on `promote_t` (D8's last bullet), so `pow(complex<float>, half_t)` →
  `complex<float>` and `pow(half_t, complex<double>)` → `complex<double>` work now:
  they never need `complex<half_t>`. `complex<T>` itself has no `promote`, so it is
  still excluded.
- **`conj`/`proj(half_t)` would have become a hard error.** With `promote<half_t>`,
  their return type `complex<half_t>` hits the `static_assert` in `complex` (`:270`).
  That happens even in an unevaluated `requires`/`is_invocable` check, which then fails
  to compile instead of yielding `false` (measured with a probe). Both are now
  constrained on `Impl::is_complex_value_type_v<promote_t<T>>`, the same trait the
  `static_assert` now uses. CH1 widens that one trait to enable `complex<half_t>`, and
  `conj`/`proj(half_t)` with it.

Measured (probe against the real headers, before → after): `promote_t<half_t>`,
`norm(half_t)`, `arg(half_t)`, `promote_2<half_t, half_t | float | int>`,
`pow(complex<float>, half_t)`, `pow(half_t, complex<float>)` 0 → 1 (also for bhalf);
`promote_2<half_t, bhalf_t>`, `conj(half_t)`, `proj(half_t)` stay 0;
`real`/`imag(half_t)` stay 1 (the non-template half overloads still win);
`pow(complex<float>, int)` → `complex<double>` is unchanged.
Test: `complex_half_scalar_promotion` in `TestComplex.hpp`. It checks the promotion
results and return types statically, the absence of `conj`/`proj(half_t)` and of
`promote_2<half_t, bhalf_t>`, and runs `norm`, `arg` (including `arg(-0) = pi`) and
mixed `pow` on the host and in a kernel.

**B4. `constexpr` half (H1, H2, H5, H13-H15, D6). — B4a, B4b-CUDA and B4c IMPLEMENTED (2026-11-24); B4b-SYCL IMPLEMENTED (2026-11-25, to verify on Aurora); B4b-HIP FUTURE WORK.**
*B4a status (2026-11-24): implemented (`545a6aa5a`), verified on Aurora (job 9121273, see below).*
- **Code:** `Kokkos_Half_FloatingPointWrapper.hpp` defines `KOKKOS_IMPL_HALF_CONSTEXPR`
  (`constexpr` when `KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT` or `..._BHALF_...` is
  defined, else empty) and the feature macro `KOKKOS_IMPL_HALF_IS_CONSTEXPR`. Every
  function of the wrapper (constructors, conversions, unary, compound and binary
  operators, comparisons, the `BitComparisonWrapper` comparisons),
  `cast_to_wrapper`/`cast_from_wrapper`, every declaration of the cast functions (C++
  requires `constexpr` on all declarations of a function) and the float fallback carry
  it; the definitions in `Kokkos_STD_Half_Conversion.hpp` are `constexpr`. The
  bit-pattern constructor (numeric-trait constants) uses `__builtin_bit_cast` on the std
  backend, since `Kokkos::bit_cast` (`memcpy`) is not `constexpr`. The stream operators
  stay non-`constexpr`. On CUDA, HIP and SYCL the macro is empty: no change there.
- **Test:** `half_constexpr_checks<H>()` in `TestHalfConversion.hpp`, `static_assert`ed
  for `half_t` and `bhalf_t` under `KOKKOS_IMPL_HALF_IS_CONSTEXPR`: arithmetic,
  comparisons (also with `float`/`double`), conversions to `int`/`double`/`bool`, `!`,
  compound assignment and `++`, the trait constants `infinity_v`/`finite_max_v`, and the
  cast functions.
- **Checked here** with a GCC 12 shim that selects the std backend for `half_t`
  (`-D__STDCPP_FLOAT16_T__=1` and a stand-in `<stdfloat>` with `std::float16_t =
  _Float16` and float-based `<cmath>` overloads, which GCC 12's libstdc++ lacks): the
  B4a and CH5 `static_assert`s compile (a planted false assertion fails, so they are
  evaluated), and `*complex*:*atomic*` passes (25). `bhalf_t` is `float` in the shim
  (GCC 12's `__bf16` has no arithmetic), so its part needs GCC >= 13.
- **Verified on Aurora (PBS job 9121273, 2026-11-24, `545a6aa5a`, GCC 14.2):** all four
  builds of `run_std_half_fp16.sh` (`default`, `native`, `gnu`, `gnu-native`; Debug,
  Serial) with 0 warnings, 25 / 15 / 14 tests pass. `half_t` and `bhalf_t` both wrap the
  std types there, and the builds recompiled `TestDefaultDeviceType.cpp` (B4a checks)
  and `TestSerial_Complex.cpp` (CH5 checks): so the `static_assert`s were evaluated and
  hold for both types, `bhalf_t` for the first time. CUDA unchanged (macro empty): A100
  jobs 10415251 (`*half*`: 12 pass, the 2 `*_io_vs_std` skip as without the std
  backend) and 10415253 (`*complex*:*atomic*`: 25/25 on Serial and Cuda).

*B4b-CUDA status (2026-11-24): implemented and verified on the A100 (jobs 10415931-33).*
Reconciled with the plan: the plan put the software path in the wrapper only, and
said that adding it to each backend's `cast_to_half` would leave the 21 full-type
branches non-`constexpr`. Both are needed. The declarations of the cast functions
carry `KOKKOS_IMPL_HALF_CONSTEXPR`, so on CUDA their definitions must be `constexpr`
too. And the full-type branches (device pass only) need their own fallback. So:
- **Macros** (`Kokkos_Half_FloatingPointWrapper.hpp`): `KOKKOS_IMPL_HALF_CONSTEXPR` and
  `KOKKOS_IMPL_HALF_IS_CONSTEXPR` now also on CUDA. The new
  `KOKKOS_IMPL_HALF_CONSTANT_EVALUATED()` is `__builtin_is_constant_evaluated()` on CUDA
  and `false` elsewhere, so the std backend is unchanged. HIP and SYCL are unchanged.
- **Software conversions** (`Kokkos::Experimental::Impl`, in the wrapper header, from the
  prototype): `half_constexpr_from<H>(T)` (round to nearest even, once, from
  `float`/`double`/`bool`/any integer) and `half_constexpr_to<T>(H)` (exact to
  `float`/`double`, toward zero to integers, as CUDA's `_rz`). They read and write the
  bits with `__builtin_bit_cast`. NaN results are the canonical `0x7fff`.
- **CUDA casts** (`Kokkos_Cuda_Half_Conversion.hpp`): all 66 non-identity
  `cast_to_*`/`cast_from_*`, including the pre-Ampere bf16 ones, are `constexpr` and
  begin with `if (KOKKOS_IMPL_HALF_CONSTANT_EVALUATED()) return Impl::half_constexpr_...;`.
  At run time they still use the intrinsics.
- **Wrapper:** the 21 full-type branches (`__half`'s own operators) run only outside
  constant evaluation; there the float path is used (one rounding: exact for
  `+ - * /`). The 8 compound assignments with `float`/`double` use the cast functions
  during constant evaluation. The bit-pattern constructor uses `__builtin_bit_cast` on
  CUDA too.
- **Tests:** the B4a checks (`half_constexpr_checks`) and CH5
  (`test_half_complex_in_constant_expression`) now also run on CUDA, being under
  `KOKKOS_IMPL_HALF_IS_CONSTEXPR`. New `half_constexpr_path_matches_runtime` /
  `bhalf_...` (`TestHalfConversion.hpp`, in `*half*`) compare the software path with the
  run-time conversions bit for bit, on the host and in a kernel: all 65536 patterns
  decoded and encoded back, the midpoints between neighbours and the doubles next to
  them, and 64-bit integers. NaNs are not compared.
- **Checked here:**
  - nvcc 12.6 (`build_gpu` flags): a check file with the B4a assertions, CUDA-specific
    cases (65520 → inf, the 64-bit bf16 case without double rounding) and a
    `constexpr complex<half_t>` product compiles in both passes. The 221 objects of the
    test executables compile with 0 `warning #`.
  - The GCC 12 shim (std backend for `half_t`): unchanged, 27 complex tests pass, and
    `half_constexpr_path_matches_runtime` passes against the native `_Float16`
    conversions.
  - Host (`half_t` = float): unchanged.
- **Verified on the A100** (jobs 10415931, 10415932, 10415933):
  - `*half*` 16 tests: 14 pass, including `half_constexpr_path_matches_runtime` and
    `bhalf_constexpr_path_matches_runtime` on the host and the device; the 2
    `*_io_vs_std` skip, as without the std backend. So the software path agrees with the
    A100's conversions bit for bit on every pattern, midpoint and 64-bit integer.
  - `*complex*:*atomic*` 27/27 on Serial and Cuda; `*numeric_traits*` 14 pass on
    Serial1 and Cuda1 (the `std::numeric_limits` comparison skips without the std
    backend).
  - The executables were built from the new objects, so the B4a and CH5
    `static_assert`s held in both nvcc passes.
  - The first submission (jobs 10415877-79) failed before compiling: `cmake --build`
    run from the `/home` symlink path had written that path into `build_gpu`'s build
    files, which the compute nodes cannot see. Fixed by regenerating from `/work`.

*B4b-SYCL status (2026-11-25): implemented; compile-checked here (DPC++ 2024.0), to verify on Aurora (`run_sycl_aurora.sh`).*
As B4b-CUDA, with one SYCL-specific difference: the host pass already uses the
software conversions and the float path at run time (the DPC++ host rounding bug, H17
notes), so only the device pass switches on constant evaluation.
- **Macros:** `KOKKOS_IMPL_HALF_CONSTEXPR` (`constexpr`) and
  `KOKKOS_IMPL_HALF_IS_CONSTEXPR` now also on SYCL. `KOKKOS_IMPL_HALF_SOFTWARE_PATH()`
  (renamed from `KOKKOS_IMPL_HALF_CONSTANT_EVALUATED()` with the host fix) is
  `__builtin_is_constant_evaluated()` in the SYCL device pass (was `false`) and stays
  `true` in the host pass.
- **SYCL casts** (`Kokkos_SYCL_Half_Conversion.hpp`): all `cast_to_*`/`cast_from_*`
  are `constexpr`; every non-identity one begins with the software branch
  (`Impl::half_constexpr_from`/`_to`), so also `bfloat16`, whose `bfloat16(float)` is
  not `constexpr`, and the integer conversions. At run time on the device they still
  use `sycl::half`/`bfloat16`'s own conversions.
- **Wrapper:** no change; the fallbacks of B4b-CUDA cover SYCL
  (`KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH` is always defined there).
- **Follows automatically:** the B4a checks, CH5 (`constexpr complex<half_t>`) and B4c
  (the numeric traits as half values) on SYCL; `half_constexpr_path_matches_runtime`
  and `bhalf_...` now run instead of skipping. Their device part compares the software
  path with the Intel GPU's conversions; on the host both sides use the software path.
- **Checked here** (icpx 2024.0, a fake SYCL configuration: the host build's config
  with `KOKKOS_ENABLE_SYCL` and `Kokkos_Setup_SYCL.hpp`;
  `claude_builds/complex/fakesycl/t.cpp`): `static_assert`s on `half_t` and `bhalf_t`
  arithmetic, conversions to `int`/`bool`, comparisons, unary minus, compound
  assignment with `float`/`double`, `epsilon`, subnormal ties, and a `constexpr`
  product inside a kernel, all pass in the host and `-fsycl-device-only` passes with 0
  warnings (`-fp-model=precise`). A planted false assertion fails, so the checks are
  evaluated. Not run: the CPU OpenCL device here has no `fp16` aspect. The host, shim
  and CUDA builds are unaffected: the changes are inside SYCL-only preprocessor
  branches, plus comments.

*The analysis and the verified CUDA prototype below are the basis of B4b-CUDA.*
- **B4a, std backend.** Mark these `constexpr`:
  - the `Kokkos_STD_Half_Conversion.hpp:18-60` conversions;
  - the wrapper ctors (`:282-368`), comparisons (`:740-860`) and arithmetic, under
    `KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT`.

  Test: `static_assert(half_t(1.5) + half_t(2.5) == half_t(4))` on a GCC ≥ 13 host build.
- **B4b, CUDA/HIP/SYCL (optional, risky; revised below: per backend, B4b-HIP is a future task).** Add an `if (std::is_constant_evaluated())`
  software round-to-nearest-even branch in `cast_to_half`/`cast_from_half` (and bhalf), using
  the `constexpr __half(__half_raw)` ctor and `std::bit_cast`. Needs a prototype under
  nvcc/hipcc/icpx first.
- **B4c (optional).** Numeric traits as `constexpr half_t` (H11), once B4b exists. Today's
  `BitComparisonWrapper` return type is public API, so this needs care.

  *B4c status (2026-11-24): DONE (`c0e6e93ee`); verified on CUDA (A100 jobs 10420748-50) and on the std backend (Aurora job 9123196).*
  - **The care, reconciled with the source:** `BitComparisonWrapper` lives in
    `Kokkos::Experimental::Impl`. It is reachable as `half_t::bit_comparison_type` and
    is the type of `infinity_v<half_t>` and the other value traits. Nothing in Kokkos
    depends on that type: the library uses the traits only through comparison and
    conversion (`Kokkos_BitManipulation.hpp`, the half math functions, the reduction
    identities, the stream extraction). Both work the same with a `half_t`. The bit
    masks (`exponent_mask`, `fraction_mask`) are not numeric traits and stay
    `BitComparisonWrapper`. User code that reads `.value` (the raw bits) of a trait
    would break on the backends below.
  - **Change** (`Kokkos_Half_NumericTraits.hpp`): the 9 value traits of `half_t` and of
    `bhalf_t` (`infinity`, `finite_min`, `finite_max`, `epsilon`, `round_error`,
    `norm_min`, `denorm_min`, `quiet_NaN`, `signaling_NaN`) are, through
    `KOKKOS_IMPL_HALF_TRAIT_VALUE`:
    - `static constexpr half_t value{bit_comparison_type{bits}}` where
      `KOKKOS_IMPL_HALF_IS_CONSTEXPR` (std backend, CUDA);
    - the bit pattern as before on HIP and SYCL, where half values cannot be
      `constexpr` (B4b-HIP/-SYCL).

    So the type of `infinity_v<half_t>` depends on the backend, as `constexpr` half
    itself does. The values and bits are the same.
  - **Gained** (H11): `static_cast<double>(epsilon_v<half_t>)`, arithmetic on the
    constants (`finite_max_v<half_t> / 2`) and their use in constant expressions,
    as for `float`/`double`; and `bhalf_t x = infinity_v<bhalf_t>;` no longer
    converts between classes, so GCC's bf16 false positive (§8.2) does not apply to it.
  - **Test:** `half_value_traits_are_constexpr_halves<H>()` (`TestNumericTraits.hpp`,
    `static_assert`ed for `half_t`/`bhalf_t` under `KOKKOS_IMPL_HALF_IS_CONSTEXPR`):
    each trait has type `H`, has the IEEE value (as `double`), the NaNs are unordered,
    and arithmetic on the constants works. The existing trait tests (`*numeric_traits*`,
    `numeric_traits_half_vs_std_numeric_limits`) check the bits.
  - **Checked:**
    - CUDA: 212 objects compile with 0 `warning #`; A100 jobs 10420748 (`*half*` 14
      pass, 2 std-only skips), 10420749 (`*complex*:*atomic*` 27/27 on Serial and
      Cuda), 10420750 (`*numeric_traits*` 14 pass on Serial1 and Cuda1, the std-only
      comparison skips). So the traits work as `constexpr` half values in both nvcc
      passes and in device code.
    - GCC 12 shim (std backend for `half_t`): the B4c checks compile; 41 of 42
      `*numeric_traits*:*mathematical_functions*` tests pass. The exception is
      `numeric_traits_half_vs_std_numeric_limits`, because GCC 12's libstdc++ has no
      `numeric_limits<_Float16>`: a shim artifact; that test passes with GCC 14
      (Aurora). 27 complex and 9 conversion tests pass.
    - Host (`half_t` = float): unchanged, all pass, 0 warnings.
    - std backend, Aurora job 9123196 (GCC 14.2, `c0e6e93ee`, the four builds of
      `run_std_half_fp16.sh`): 0 warnings; `*complex*:*atomic*` 27, `*numeric_traits*`
      15, `*half*` 16 pass in each. `half_t` and `bhalf_t` wrap the std types and
      `TestSerial_NumericTraits.cpp` was recompiled, so the B4c checks hold for both;
      `numeric_traits_half_vs_std_numeric_limits` passes, so the `constexpr` traits
      equal `std::numeric_limits` bit for bit. The same run is the first of
      `half_`/`bhalf_constexpr_path_matches_runtime` on the std backend: the software
      conversions (B4b-CUDA) agree with GCC's native `std::float16_t`/`std::bfloat16_t`
      conversions in all four builds.

*B4 status (2026-11-22): B4a blocked here (no GCC ≥ 13). B4b prototype verified on
host and A100 (jobs 10396298, 10396345, 10396374). `constexpr` was checked per backend
(table below): B4b-CUDA is ready, B4b-SYCL is feasible (compile-time checks only),
B4b-HIP is a future task. No Kokkos source changed yet; implementing B4b-CUDA/SYCL
awaits a decision.* Reconciled with the source:
- **B4a cannot be built on this system.** The std backend is selected only when the
  compiler defines `__STDCPP_FLOAT16_T__` (`Kokkos_Macros.hpp:696-700`), i.e. GCC ≥ 13 in
  C++23 mode. Measured: GCC 12.2 (`-std=c++23`) and icpx 2024.0 do not define it, and
  no newer GCC module exists. B4a is a mechanical `constexpr` on plain `std::float16_t`
  conversions; it should be done with a GCC ≥ 13 CI job, not blind.
- **B4b must not use `std::is_constant_evaluated()` or `std::bit_cast`.** Both are host
  `constexpr` functions in libstdc++. In `__host__ __device__` code they need
  `--expt-relaxed-constexpr`, which this build does not use (nvcc #20013, seen in A3).
  Use `if consteval` / `__builtin_is_constant_evaluated()` as `Kokkos_BitManipulation.hpp:28-36`
  and A3 already do, and `__builtin_bit_cast` (accepted by nvcc's front end and GCC).
  H15 (`Kokkos::bit_cast` not `constexpr`) can be fixed the same way.
- **The software path belongs in the wrapper, not in each backend's `cast_to_half`.**
  Every ctor and operator funnels through `cast_to_wrapper`/`cast_from_wrapper`, and
  arithmetic either goes through `float` or, under `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH`
  (21 branches), through `impl_type`'s own operators, which are not `constexpr`
  either. A constant-evaluated branch in the wrapper (decode/encode the bits, compute in
  `float`, round once) covers all of them, independent of the backend. The plan's
  "(and bhalf) in `cast_to_half`/`cast_from_half`" would have left the 21 full-type
  arithmetic branches non-`constexpr`.
- **Round once from the source type.** Converting `double` or a 64-bit integer through
  `float` rounds twice. Example: `(1ll << 62) + (1ll << 54) + 1` → `bfloat16` must round up,
  but through `float`/`double` lands on the midpoint and rounds to even. The prototype
  rounds `sig · 2^exp` with a sticky bit directly. Arithmetic via `float` and one
  rounding is exact for both types: 24 ≥ 2p + 2 for p = 11 and p = 8 (Figueroa).
  New gap: the pre-Ampere CUDA path (`Kokkos_Cuda_Half_Conversion.hpp:104-228`)
  converts `double` and integers to `bfloat16` through `float` at run time, so it
  double-rounds; sm_80+ uses `__double2bfloat16`/`__ll2bfloat16_rn`.
- **`constexpr` is not CUDA-only (checked 2026-11-22).** Every backend can support it,
  but each needs a different mechanism, and each `impl_type` decides whether the
  software path is needed:

  | Backend | `impl_type` (half / bhalf) | Native `constexpr`? | Can the wrapper read and write the bits in constant evaluation? | Way to `constexpr` | Evidence | Testable here? |
  |---|---|---|---|---|---|---|
  | std, C++23 (GCC ≥ 13) | `std::float16_t` / `std::bfloat16_t` | **yes**: arithmetic types, so conversions, arithmetic and comparisons are core-language `constexpr` | not needed | **B4a**: mark the wrapper and `Kokkos_STD_Half_Conversion.hpp` `constexpr`; no software path | the standard (P1467); not compiled here | **no**: needs GCC ≥ 13 with `-std=c++23`. Clang provides `_Float16` but no `std::float16_t`/`<stdfloat>`, so with clang `half_t` stays `float` and nothing is needed |
  | CUDA (nvcc 12.6) | `__half` / `__nv_bfloat16` | no: only the `__half_raw` ctor and `operator bool` (H14) | **yes**: `__half(__half_raw)` writes, `__builtin_bit_cast` reads | **B4b-CUDA**: software path in the wrapper (prototype above) | prototype: compile-time asserts, host and A100 exhaustive checks | yes |
  | SYCL (DPC++ 2024.0) | `sycl::half` / `sycl::ext::oneapi::bfloat16` | half: **yes**. `half_type.hpp:26-34` makes ctor, arithmetic, comparisons and `float` conversion `constexpr` when `__builtin_bit_cast` exists. bf16: **no** (its `bfloat16(float)` ctor, `bfloat16.hpp:92`, is not `constexpr`) | **yes** for both (`__builtin_bit_cast` to and from them is a constant expression) | **B4b-SYCL**: half: only mark the wrapper `constexpr` (native, like std); bhalf: the CUDA software path | `b4sycl.cpp`, `icpx -fsycl -fsyntax-only`, host and `-fsycl-device-only` passes: `constexpr sycl::half(1.5f) + sycl::half(2.5f) == 4` OK; `constexpr bfloat16(1.5f)` fails; bit_cast both OK | compile-time only (no Intel GPU here) |
  | HIP (ROCm `clr`, `develop`, 2026-11-22) | `__half` / `__hip_bfloat16` | no: only `__half(const __half_raw&)` / `__hip_bfloat16(const __hip_bfloat16_raw&)` and `operator bool` (bf16) | **write yes, read no**. The value is a `protected` anonymous union (`_Float16 data; unsigned short __x;`, `amd_hip_fp16.h:73-81`; bf16 likewise, `amd_hip_bf16.h:164-168`). Clang rejects `__builtin_bit_cast` of a type containing a union in constant evaluation. `operator __half_raw()` is not `constexpr` and reads `data`, the inactive member after a raw construction. Nothing public exposes the bits | **B4b-HIP**, a workaround only. For example, store a `struct : __half { constexpr std::uint16_t bits() const { return __x; } }`, valid only while `__x` is the active member. Every value built at compile time would have to come from the raw ctor; the defaulted ctor activates `data` | headers read from github.com/ROCm/clr. The three union rules were measured with icpx 2024.0 (clang 17) on a mock with HIP's layout (`_Float16 data; unsigned short __x;`): `__builtin_bit_cast` → "bit_cast from a union type is not allowed in a constant expression"; the derived `bits()` after the raw ctor → OK; after the defaulted ctor → "read of member '__x' of union with active member 'data'" | **no** (no ROCm here; the mock is not HIP) |

  Conclusion: `constexpr` half is possible on the std, CUDA and SYCL backends. It is
  native on std and for `sycl::half`, and needs the software path for CUDA and SYCL
  bf16. HIP is the outlier: it needs an access workaround that depends on union
  active-member rules and cannot be tested here. So B4b stays a current, optional step,
  not a future task, but it is split per backend:
  - **B4b-CUDA:** ready (prototype verified).
  - **B4b-SYCL:** half is trivial; bf16 reuses the CUDA path. Only compile-time tests are
    possible here. *Implemented 2026-11-25 (status above): the software path for both
    types, which also covers the host pass.*
  - **B4b-HIP:** **future task.** It is unverifiable here, and it is better solved
    upstream: a `constexpr` bits accessor, or `constexpr` conversions in ROCm's
    `amd_hip_fp16.h`, would make it as easy as SYCL. Until then, keep the HIP path
    non-`constexpr`.

  So `constexpr` `half_t` would be backend-dependent. A per-backend macro, e.g.
  `KOKKOS_IMPL_HALF_CONSTEXPR` (defined for std, CUDA and SYCL; empty for HIP), marks
  the functions. A feature macro tells tests and CH5 (`constexpr complex<half_t>`)
  when they may `static_assert`.

*Prototype* (`tmp/claude_builds/complex_a6/b4proto.cu`, standalone, nvcc 12.6 + GCC 12.2,
`-std=c++20 -arch=sm_80`, no `--expt-relaxed-constexpr`):
- compiles; 17 `static_assert`s on rounding edge cases (overflow to inf at 65520,
  subnormal ties, ties to even, signed zero, bf16 range, the 64-bit integer case above)
  and a literal class holding `__half` (`constexpr W w = W(1.5f) + W(2.5f)`) pass;
- the same `constexpr` evaluation works inside a `__global__` function, and the runtime
  branch there still compiles to the hardware conversions (`F2F`/`F2FP` in SASS);
- host, exhaustive against CUDA's own host conversions: all 2³² floats → half and → bf16,
  all 65536 patterns → float (both types), doubles at/next to every half midpoint:
  **0 mismatches**;
- device, A100 (job 10396298): all 2³² floats against `__float2half_rn`/`__float2bfloat16_rn`
  and all 65536 patterns → float: **0 mismatches**. The sampled arithmetic grid
  (65536 × 262) reported 2, because that check did not skip NaN results.
- device, full grid (job 10396345, `b4arith.cu`): `__hadd`/`__hmul`/`__hdiv` against
  float + one rounding on all 65536² pairs with non-NaN inputs and results:
  **0 mismatches**. So the 2 above are NaN encodings only: ±inf · (+0) is in the sample
  (j = 0). The hardware returns the canonical NaN `0x7fff`, the prototype a
  sign-carrying quiet NaN. Confirmed (job 10396374, `b4nan.cu`): hardware
  `inf·0`, `−inf·0` and `__float2half_rn(−NaN)` all give `0x7fff`; the prototype gives
  `0xfe00` for both products (x86's `inf·0` is the negative default NaN). For the
  implementation, emit `0x7fff` for every half NaN in the
  constant-evaluated path, so it matches CUDA bit for bit (bf16: check `__float2bfloat16_rn(NaN)` first; not measured).
  Results must not depend on NaN bits anyway (A6: GPUs canonicalise NaN). A
  `constexpr` value and the same expression at run time then agree on every non-NaN
  result.

Not in Part B: rejecting implicit narrowing into `half_t` like `std::float16_t` (H1). It
would break existing `half_t` users; keep it as a documented divergence.

### Part C: `complex<half_t>` support (depends on A and B)

**CH1. Allow half value types (minimum viable).**

| File | Change |
|---|---|
| `Kokkos_Complex.hpp` | `#include <Kokkos_Half.hpp>` (C2); D1 trait; `static_assert(Impl::is_complex_value_type_v<RealType> && std::is_same_v<RealType, std::remove_cv_t<RealType>>, "... cv-unqualified floating point type (including Kokkos::Experimental::half_t/bhalf_t)")`; update doc comment (C20) |
| `Kokkos_Complex.hpp` | *(revised after B3)* B3 added `Impl::is_complex_value_type_v` (used by the `static_assert`) and constrained the scalar `conj`/`proj` on it; `norm`/`arg`/`real`/`imag` already work for half through `promote`, and the mixed `pow` gate is already `promote_t`-based (C12). CH1 only widens `is_complex_value_type_v` to the half types (the D1 trait), which enables `complex<half_t>`, `conj`/`proj(half_t)` and flips the `!has_scalar_conj<half_t>` assertions in `complex_half_scalar_promotion` |
| `core/unit_test/TestComplex.hpp` | port the verified test work from `half-complex.md` (steps 1–9): templated tests, runtime twins for `constexpr` checks, `CHECK_COMPLEX_NEAR`, file-scope `static_assert`s. Thanks to A2 plus the half proxy below, `complex_io` uses the compact input `"1 (2) (3,4)"` for all types, with no space-separated workaround and no `FIXME_HALF`. The output expectation for half is `complex<float>`'s output for the same values |
| `Kokkos_Complex.hpp` | **I/O for half (D7, revised A2):** in `<<`/`>>`, replace `std::complex<RealType>` by a standard proxy, e.g. `Impl::complex_io_t<RealType>`: `RealType` for standard FP types; for half types `float` when writing (exact) and `double` when reading (avoids the float double rounding shown in D7). Read as `std::complex<double> t; if (is >> t) x = complex<half_t>(half_t(t.real()), half_t(t.imag()));`. This is what `cuda::std::complex<__half>` does (§3b), minus its assign-on-failure defect. Test: `complex_io_like_std` for half, compared against `complex<float>` holding the same values |

Check that `Kokkos_Half.hpp` does not include `Kokkos_Complex.hpp` (it does not today),
so there is no include cycle.

*CH1 status (2026-11-22): done. Verified on the A100 (job 10396565): all 23 `*complex*:*atomic*` SmokeTests pass on Serial and Cuda, with the complex tests instantiated for float, double, half_t and bhalf_t.* Reconciled with the source:
- **Include cycle:** none. `Kokkos_Half.hpp` and its `impl/` headers include only half,
  backend and `Kokkos_{NumericTraits,ReductionIdentity,MathematicalFunctions,BitManipulation}`
  headers.
- **The plan missed that widening the trait is not enough.** The Part A algorithms hold
  their constants in `constexpr` locals: `constexpr T inf = infinity_v<T>`, the
  `complex_divide_smith` thresholds, `pi`/`ln2`, the HFT crossovers. A probe that calls
  every member and free function of `complex<half_t>`/`complex<bhalf_t>` on host and
  device (`tmp/claude_builds/complex_a6/ch1probe.cpp`) gave 89 errors, from three causes:
  1. **Numeric traits.** For half they are `BitComparisonWrapper`, which converts to
     `half_t` only at run time (H11, and B4 is future work).
  2. **`<numbers>`.** `std::numbers::pi_v<half_t>` is ill-formed: libstdc++ defines the
     constants only for floating-point types.
  3. **`half_t /= int`.** It is ambiguous: the wrapper has `/=` for `float` and `double`,
     but not for `int`.

  Fix: `Impl::complex_{infinity,quiet_nan,epsilon,finite_max,norm_min,pi,ln2}<T>()`.
  These return the `constexpr` trait for standard types, and for half the trait
  converted at run time (`<numbers>` constants rounded once from `double`). The
  algorithms now use `const` locals, and the captureless `finite` lambdas capture `inf`.
  Constant evaluation of `/` for float/double (A3) is unaffected: locals of a
  `constexpr` function need not be `constexpr`. A `const` local copy still avoids the
  device odr-use issue of A4. After the fix the probe compiles with 0 errors and
  0 warnings.
- **I/O proxy** as planned: `Impl::complex_ostream_t` (`float`) and
  `Impl::complex_istream_t` (`double`) for non-standard types.
- **Test port:** the `half-complex` branch's `TestComplex.hpp` (commits `8d97050ee`,
  `1deb33b3b`, base `a360acf37`) was three-way merged into the current file. There was one
  conflict: Part A had added tests after the old one-line `TEST(complex_io)`, which the
  branch templated. Then, as planned, `FIXME_HALF` and the space-separated input were
  removed, so half uses `"1 (2) (3,4)"`. `complex_io_like_std` runs for half and bhalf
  against `std::complex<float>` (output) and `std::complex<double>` + one rounding
  (input). The B3 assertions `!has_scalar_conj<half_t>`/`!has_scalar_proj<half_t>`
  became `conj(half_t)` → `complex<half_t>`.
- **Alignment (CH6 caveat):** `complex<half_t>` follows the same rule as the other types
  (`alignas(2*sizeof)` only with `KOKKOS_ENABLE_COMPLEX_ALIGN`). That matches
  `std::complex<std::float16_t>` (alignment 2) and the existing ABI rule, so CH6 would
  need that option or an ABI change.
- **New gap (H17), fixed 2026-11-22; verified on the A100 (job 10397129: all 8 `*half*` tests of `Kokkos_CoreUnitTest_Default` pass).** On CUDA,
  `half_t`/`bhalf_t` ↔ `bool` conversions did not link in device code ("ptxas:
  Unresolved extern function"). The cause was missing definitions in
  `Kokkos_Cuda_Half_Conversion.hpp` for functions declared in
  `Kokkos_Half_FloatingPointWrapper.hpp:57,83,125,151`:
  - `cast_to_half(bool)` and `cast_from_half<bool>`: missing for every arch;
  - `cast_to_bhalf(bool)` and `cast_from_bhalf<bool>`: defined only in the pre-Ampere
    section (`< 80`), missing for sm_80+.

  All four were confirmed failing on sm_80 with a probe
  (`tmp/claude_builds/complex_a6/h17probe.cpp`). HIP and the std backend define them.

  **C++23 supports these conversions**, so this is a defect, not a divergence:
  - `bool` is an integer type ([basic.fundamental]);
  - [conv.fpint]/2 converts any integer type to any floating-point type, which since
    C++23 includes `std::float16_t`/`std::bfloat16_t` ([basic.extended.fp]), and maps
    false/true to 0/1;
  - P1467 restricts only floating → floating conversions;
  - [conv.bool] gives the other direction (±0 → false, anything else, including NaN,
    → true).

  `_Float16 a = true; bool b = a;` compiles implicitly with GCC 12 and icpx 2024.0.

  Fix:
  - `cast_to_half(bool)` and `cast_to_bhalf(bool)` (sm_80+) go through
    `float(val)`, which is exact;
  - `cast_from_half<bool>` and `cast_from_bhalf<bool>` (sm_80+) return
    `float(val) != 0`.

  The existing pre-Ampere definitions already do the same (`static_cast<bool>(float)`).
  Remaining divergence: Kokkos's `half_t(bool)` is `explicit`, while the standard
  conversion is implicit. Making it implicit would be a separate API decision.

  Test: `half_bool_conversion` and `bhalf_bool_conversion` in `TestHalfConversion.hpp`
  (`Kokkos_CoreUnitTest_Default`). They check both directions, through the
  constructor/operator and through the cast functions, on host and device: 0, −0, 0.5,
  −2, 1e-3, the smallest subnormal, ±inf, NaN. The existing conversion tests loop over
  the arithmetic types but skip `bool`, which is why H17 went unnoticed.
- **New gap (H18), std backend, found 2026-11-22 (Aurora, job 9116988); fixed and
  verified on the std backend (Aurora, job 9117024, `run_h18_std_half.sh`: GCC 14.2,
  with and without `Kokkos_ARCH_SPR`, the assignment program links with no undefined
  `cast_to_half`/`cast_to_bhalf`, its bits equal direct initialization: half_t
  `0x7c00 0x1400 0x7bff 0x0400`, bhalf_t `0x7f80 0x3c00 0x7f7f 0x0080` for
  infinity/epsilon/finite_max/norm_min; `*half*` 10/10).** `Kokkos_Half_FloatingPointWrapper.hpp:79,147` declare
  `cast_to_half(half_t)` and `cast_to_bhalf(bhalf_t)`; CUDA, HIP and SYCL define them
  (identity), `Kokkos_STD_Half_Conversion.hpp` did not. They are reached by the
  template `floating_point_wrapper::operator=(T)` (through `cast_to_wrapper`) for a `T`
  that converts to the half type, e.g. a numeric-trait constant:
  `h = Kokkos::infinity<half_t>::value;`. GCC 14 then warns "inline function
  `cast_to_bhalf(bhalf_t)` used but never defined", and the program cannot link.
  Copy- and direct-initialization take another path (the `bit_comparison_type`
  constructor), so the existing tests never hit it. Pre-existing, not from this branch.
  - **Fix:** define both identity overloads in `Kokkos_STD_Half_Conversion.hpp`, as the
    other backends do.
  - **Test:** `half_assign_trait_constants` / `bhalf_assign_trait_constants` in
    `TestHalfConversion.hpp` (`Kokkos_CoreUnitTest_Default`, in the `*half*` filter):
    assign `infinity`, `epsilon`, `finite_max`, `norm_min` on the host and in a kernel,
    and compare the bits with direct initialization. Passes where `half_t` is `float`
    (GCC 12 host trial, 10/10 `*half*`); the std-backend run is
    `run_std_half_fp16.sh` on Aurora.

**CH2. Explicit narrowing into half (C6, D4).** `explicit(bool)` on the converting ctor,
explicit only for conversions *into* half types. Tests:
`static_assert(!std::is_convertible_v<complex<double>, complex<half_t>>)` and
`std::is_constructible_v<complex<half_t>, complex<double>>`.

*CH2 status (2026-11-22): done. Verified on the A100 (job 10396632): all 24
`*complex*:*atomic*` SmokeTests pass on Serial and Cuda, including the new
`complex_half_conversions`.* Reconciled with the source (probe `tmp/claude_builds/complex_a6/ch2probe.cpp`,
`ch2mixed.cpp`, measured after CH1):
- **Into half was implicit, as assumed.** `complex<half_t>` ← `complex<float|double>`
  was convertible, through the `std::is_convertible_v<RType, RealType>`-constrained
  ctor.
- **Out of half was missing; D4 did not mention it.** `half_t`'s conversion to `float`
  is `explicit` (wrapper `:294`), so `is_convertible_v<half_t, float>` is false.
  `complex<float|double>(complex<half_t>)`, which std makes implicit (widening), was
  neither convertible nor constructible. It also broke B3's mixed `pow` with a
  `complex<half_t>` operand: `pow(complex<half_t>, complex<float>)` failed in its body
  at `complex<float>(x)`. The B3 tests only covered `pow(complex<float>, half_t)`. Mixed
  arithmetic and `==` (`complex<half_t> + complex<float>`, `* float`, `/ double`)
  worked, because it is computed part by part.
- **Assignment needed an operator.** `zh = zf` worked only through the implicit ctor.
  `std::complex` has `template <class X> complex& operator=(const complex<X>&)`, which
  stays usable when the ctor is explicit; Kokkos had none.

Change:
- The ctor is constrained on `std::is_constructible_v<RealType, RType>`, initialises
  with `static_cast`, and is now `constexpr`. It is `explicit(Impl::complex_conversion_is_explicit_v<RType, RealType>)`:
  explicit when converting into `half_t`/`bhalf_t` from another type, implicit
  otherwise. This is the D4 rule, written with `Impl::is_half_type_v` rather than a rank
  trait: every other type that converts into half has a higher rank, and half ↔ bhalf
  has no conversion.
- Added a templated `operator=(const complex<RType>&)` (for `RType != RealType`), as in
  std.
- `float` ↔ `double` stays implicit both ways (the known divergence).

Measured after the change: into half explicit but constructible; out of half implicit
(`float`, `double`); half ↔ bhalf neither, also not assignable; `float` ↔ `double`
unchanged; `pow(complex<half_t>, complex<float>)` → `complex<float>`. All existing
complex tests still compile: none relied on implicit narrowing into half.
Test: `complex_half_conversions` (static traits; rounding into half and exactness out
of half, on host and in a kernel; assignment; mixed `pow`), plus file-scope
`static_assert`s for float/double (implicit both ways; `constexpr` conversion) and
half ↔ bhalf (none).

**CH3. Numerics for half (D5).** `Impl::complex_compute_type`, used in `operator*`,
`operator/`, `*=`, `/=`, `abs`, `asinh/acosh/atanh/atan/tan/tanh/pow`. Tests:
- intermediate overflow: `z * z` with `z = complex<half_t>(269, 111.5)` (both exact
  in half). The true result ≈ (59928.75, 59987), which is representable (< 65504), but
  `269 * 269 = 72361` overflows. In half arithmetic the real part becomes `inf`. With
  `float` intermediates it must be finite and within one half ULP.
- division: the half path is the float naive formula plus post-hoc recovery (D10). Test
  `(1,1)/(0,0)`, `(1,1)/(inf,0)` and a near-overflow case, e.g. `(60000,60000)/(1,1)`
  (`a·c + b·d = 120000` overflows half but not float). For bhalf, test that float squares
  do not overflow with the chosen intermediate type, e.g. `(1e38,1e38)/(1e38,1e38) = (1,0)`.
- accuracy: the error bounds already used in `complex_basic_math` (half-complex §3.1)
  should hold with a smaller gamma.

Benchmark `*` on A100 native `__half` vs. through float before enabling it for `*`.

*CH3 status (2026-11-22): done. Verified on the A100 (job 10397002): all 25
`*complex*:*atomic*` SmokeTests pass on Serial and Cuda, including the new
`complex_half_compute_in_float` (job 10396960 aborted on Serial because of a test bug:
its Views were in CudaSpace).* Measured before changing the source (`tmp/claude_builds/complex_a6/ch3acc.cpp`,
`ch3bench.cpp`, A100, jobs 10396863/10396864/10396919):
- **Benchmark** (2²⁶ `complex<half_t>`, sm_80 native `__half` arithmetic, i.e.
  `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH`):

  | op | case | native half | via float |
  |---|---|---|---|
  | `*` | stream `c = a*b` | 0.603 ms (1336 GB/s) | 0.603 ms (1336 GB/s) |
  | `*` | chain of 256 | 1.007 ms | 0.889 ms (0.88×) |
  | `/` | stream | 0.621 ms | 0.632 ms |
  | `/` | chain of 256 | 5.233 ms | 5.326 ms (1.02×) |

  So float costs nothing, and `*` uses it too; the plan had made that conditional on
  this benchmark.
- **Accuracy and overflow** of every operation, against `complex<double>`. 4096 random
  representable inputs; normwise error in u (unit roundoff), with |ref| floored at
  `norm_min` so that unavoidable subnormal results do not count. Results are identical
  on host and device.

  | | before (CH2) | after (CH3) |
  |---|---|---|
  | half, parts in 2^±4 | ≤ 5.7 u (log); pow 70 u; spurious inf/NaN in exp, sin, cos, sinh, cosh, pow (4-7 each) | ≤ 1.4 u; none |
  | half, parts in 2^±14 | tanh 13.7, log 10, tan 8.2 u; pow 4084 u; **asin/acos/asinh/acosh: 489 of 4096 spurious inf/NaN**; mul, sqrt, sin, cos 1 each | ≤ 1.4 u, pow 9.4 u; none |
  | bhalf, parts in 2^±4 | ≤ 6.8 u; pow 104 u | ≤ 0.99 u |
  | `(269,111.5)²` | `(inf, 60000)` | `(59936, 60000)` (exact, rounded once) |
  | `(60000,60000)/(1,1)` | `(inf, 0)` | `(60000, 0)` |

  The remaining 9.4 u of `pow` is at `pow((3892, 0.02), (1.3e-4, -9696))`: |y log x| ≈ 8e4,
  so float's rounding of the exponent (≈ 5e-3 absolute) is about 10 half ulps in the
  result. Only a double intermediate would remove it.

Reconciled with the plan:
- **`abs` needs nothing.** It measured ≤ 0.98 u with no overflow
  (`abs(40000, 40000)` = 56576), because half `hypot` already computes in float (H9).
- **Not only `asinh/acosh/atanh/atan/tan/tanh/pow` need float.** `exp`, `sin`, `cos`,
  `sinh` and `cosh` produced spurious inf/NaN, and `log`/`log10` had errors up to 10 u.
  So every transcendental (`exp log log10 sqrt sin cos tan sinh cosh tanh asin acos atan
  asinh acosh atanh pow`) computes in float.
- **Implementation:**
  - `Impl::complex_compute_t<T>` is `float` for the half types.
  - `Impl::complex_multiply`/`complex_divide` dispatch to the A3/A5 algorithms
    (`*_impl`) in the compute type and round once. The recovery logic is reused, not
    duplicated, as the §3b note asked.
  - Each transcendental starts with
    `if constexpr (Impl::is_half_type_v<T>) return complex<T>(f(complex<float>(x)));`.
    The half instantiation of the original body is then discarded, never compiled.
  - Scalar ops (`z * s`, `z / s`) stay in half: each part is a single correctly
    rounded operation.

Test: `complex_half_compute_in_float`. For half_t and bhalf_t:
- every one of 22 operations and functions on 64 input pairs (ordinary, overflowing
  squares, branch cuts) gives the same bits as `complex<float>` rounded once, on host
  and in a kernel;
- the plan's cases `(269,111.5)²`, `(60000,60000)/(1,1)`, `(1,1)/(0,0)`, `(1,1)/(inf,0)`,
  `(inf,NaN)·(1,0)`, and bhalf `(1e38,1e38)/(1e38,1e38)` = (1,0);
- `asin(16384, 16384)` is finite.

*Note from §3b.* `cuda::std::complex<__half>` (CCCL 2.5) computes `*` and `/` in half, with
`__half2` vector partial products and no promotion. It goes through `complex<float>` only
for `asinh`, `acosh`, `atanh` and `acos`. So `cuda::std` overflows on the
`z = (269, 111.5)` test above, while D5 does not. That makes the test a useful
cross-check: if Kokkos ever matches `cuda::std` on it, the promotion is not active.
Reuse `Impl::complex_multiply` (A5) and `Impl::complex_divide<float>` for the promoted
path, rather than duplicating the recovery logic.

*CH3 vs. C++23 and libstdc++ (checked 2026-11-22).* Is computing `complex<half_t>` in
float what C++23 does for `std::complex<std::float16_t>`?
- **The standard does not fix it.** [complex.numbers] specifies what the operations
  compute, not the precision of intermediates. [expr.pre] lets floating-point
  expressions be evaluated "in greater precision and range than that required by the
  type". So float intermediates conform, as do half ones.
- **The reference implementation is libstdc++ (GCC ≥ 13),** the only one with
  `std::float16_t`; clang/libc++ and MSVC have none. Source read:
  `libstdc++-v3/include/std/complex` and `libgcc/libgcc2.c`, `releases/gcc-13` branch.

  | operation | libstdc++ 13 for `std::complex<std::float16_t>` | Kokkos `complex<half_t>` (CH3) |
  |---|---|---|
  | `abs`, `arg` | `_Float16(__builtin_cabsf(z))`, `cargf` (`complex:602-609`) | `hypot`/`atan2` of half, computed in float (H9) |
  | `exp log sqrt sin cos tan sinh cosh tanh pow` | `static_cast<__complex__ _Float16>(__builtin_cexpf(z))` etc. (`complex:611-649`) | `complex<float>` function, rounded once |
  | `asin acos atan asinh acosh atanh` | same pattern with `cacosf` … (`complex:2038-`) | same |
  | `/` | the compiler's `__complex__ _Float16` division. libgcc `__divhc3` computes in `SFtype` (`AMTYPE`, `libgcc2.c:1860-1863, 2026-2050`, "Half precision is handled with float precision"), with the plain formula; on x86 without AVX512-FP16 the whole expression is evaluated in float | Smith's algorithm (A3) in float, rounded once |
  | `*` | the compiler's `__complex__ _Float16` multiplication, **target-dependent**: evaluated in float where `__FLT_EVAL_METHOD_TS_18661_3__` is 0 (x86 without AVX512-FP16), in half where it is 16 (`-mavx512fp16`, i.e. Sapphire Rapids and later) | A5 formula in float, rounded once |

  The bf16 functions follow the same float pattern (`complex:806-`). bf16 arithmetic
  was not measured.
- **Measured with GCC 12** (`_Complex _Float16` in C; the same compiler arithmetic
  libstdc++ uses; GCC 12's C++ front end rejects `__complex__ _Float16`):
  - x86 without AVX512-FP16 (the login node): `__FLT_EVAL_METHOD_TS_18661_3__` = 0,
    for every `-fexcess-precision` and `-O` level. `(269,111.5)²` = `(59936, 60000)` and
    `(60000,60000)/(1,1)` = `(60000, 0)`, the same as Kokkos after CH3. The code widens
    to float (`__extendhfsf2`, `mulss`).
  - `-mavx512fp16` (compiled, not run; this CPU lacks it): the method becomes 16, `*`
    uses `vmulsh` (each product rounded to half, so `269²` overflows to inf, as
    Kokkos did before CH3), and `/` calls `__divhc3` (float).
- **Conclusion:** CH3 matches libstdc++ for all functions and `/` everywhere, and for `*`
  on targets that evaluate `_Float16` in float. On FP16-native targets libstdc++'s `*`
  is less accurate and can overflow, while Kokkos stays in float; the standard allows
  both. The bits may still differ in the last place: Kokkos uses its own float
  algorithms, libstdc++ glibc's `cexpf`/`casinf`…
- **GCC 15 on aarch64 (PBS job 4260499, 2026-11-22; §8.2 "First run"):**
  `__FLT_EVAL_METHOD_TS_18661_3__` = 0 (and `__FLT_EVAL_METHOD__` = 0) with the tests'
  flags, i.e. Kokkos's default target options, no `Kokkos_ARCH_*`. GCC then evaluates
  `_Float16` in float, so libstdc++'s `*` is float too: **on that configuration CH3
  matches libstdc++ for every operation.** With `Kokkos_ARCH_ARMV9_GRACE`
  (`-mcpu=neoverse-v2`; job 4264666, 2026-11-23, §8.2 "Grace run") the method is 16,
  as with `-mavx512fp16` on x86: a half `*` compiles to `fmul h…`, libstdc++'s
  `(269,111.5)²` overflows to `(inf, 60000)`, and Kokkos gives `(59936, 60000)`.
- **Aurora, GCC 14 (jobs 9116902 and 9116933; §8.2): the `*` row measured.** With
  `-march=sapphirerapids` (`Kokkos_ARCH_SPR`) the method is 16 and a half `*` compiles
  to `vmulsh`. The probe (job 9116933), real `std::complex<std::float16_t>` from
  libstdc++ 14:

  | | method 0 (no `Kokkos_ARCH`) | method 16 (`Kokkos_ARCH_SPR`) |
  |---|---|---|
  | `std::complex<std::float16_t>` `(269,111.5)²` | (59936, 60000) | **(inf, 60000)** |
  | `Kokkos::complex<half_t>` `(269,111.5)²` | (59936, 60000) | (59936, 60000) |
  | both: `(60000,60000)/(1,1)` | (60000, 0) | (60000, 0) |

  So libstdc++'s `*` follows the evaluation method (each product rounded to half under
  16, and `269²` overflows), its `/` is float under both, and Kokkos is the same under
  both. **The conclusion above holds, now measured:** CH3 matches libstdc++ everywhere
  except `*` on FP16-native targets, where Kokkos is more accurate; both conform.
- **Qualifier: optimized builds contract (job 9120562, 2026-11-24, §8.2 "Release
  run").** The measurements above are Debug (`-O0`). At `-O3`, on a CPU with half FMA,
  GCC fuses `a*b - c` into one half FMA (one rounding), in strict `-std=c++23` as well
  as `-std=gnu++23`. Then libstdc++'s `(269,111.5)²` is `(59936, 60000)`, like Kokkos's,
  because the product no longer overflows on its own. So libstdc++'s `*` overflows under
  method 16 only without contraction (Debug, or `-ffp-contract=off`); with it, its
  result depends on the inputs and on the optimizer. Kokkos's complex results are the
  same in Debug and Release: CH3 computes in float.
- **Not done (optional):** compare `std::complex<std::float16_t>` with Kokkos over the
  `ch3acc` inputs, for the functions (expected: the same up to the last bit, since both
  compute in float).

**CH4. Interop with `std::complex<std::float16_t>` on C++23 host builds (D2). — FUTURE WORK (decided 2026-11-22; reason revised 2026-11-24).**
*Not implemented. Kokkos's minimum is C++20, and `std::float16_t` exists only from
C++23 on, and only where the compiler provides it (GCC >= 13; not clang/libc++ or
MSVC). So the constructor would exist in some builds only; pick it up once C++23 is
Kokkos's minimum standard.* Under
`KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT`:

```cpp
complex(const std::complex<typename RealType::impl_type>&) noexcept;   // host
```

*Revised 2026-11-22 (D11):* only the constructor. The conversion operator to
`std::complex<std::float16_t>` that D2 proposed is dropped, since A8 deprecates the
existing conversion operator. `std::complex<std::float16_t>` → `complex<half_t>` is not
narrowing, so the constructor can be implicit like I0.

Constrained with `requires Impl::is_half_v<RealType> && std::is_floating_point_v<typename RealType::impl_type>`.
Tests are host-only, need GCC ≥ 13, and are not possible on this system.

*Alternative from §3b.* libcu++ takes `::std::complex<U>` for **any** `U` (a template
constructor and assignment reading `reinterpret_cast<const U(&)[2]>`), plus a host-only
conversion to `std::complex<T>`.

The Kokkos equivalent would be
`template <class U> requires std::is_constructible_v<RealType, U> explicit(...) complex(const std::complex<U>&)`.
It covers `std::complex<std::float16_t>` → `complex<half_t>` without a half-specific
overload, and also `std::complex<float>` → `complex<double>`.

Trade-offs:
- It widens the overload set for all types. `explicit(bool)` must use the D4 rule, so that
  `complex<float>` ← `std::complex<double>` does not become implicit.
- Overload resolution must stay unambiguous with the existing
  `complex(const std::complex<RealType>&)`; the template replaces that one.
- Prefer it to the half-only version if the D4 trait is in place.

**CH5. `constexpr complex<half_t>` (needs B4a, and B4b for GPU backends). — std and CUDA parts IMPLEMENTED (2026-11-24); SYCL part with B4b-SYCL (2026-11-25, to verify on Aurora); HIP FUTURE WORK, with B4b.** Enable
the `static_assert(... <half_t>())` lines in `TestComplex.hpp` under
`KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT` (and for GPU backends once B4b exists). Keep the
runtime twins elsewhere.
*Status (2026-11-24):* `test_half_complex_in_constant_expression<T>()` in
`TestComplex.hpp`, `static_assert`ed for `half_t` and `bhalf_t` under
`KOKKOS_IMPL_HALF_IS_CONSTEXPR`: the existing constant-expression checks
(construction, assignment, comparison, `norm`, `conj`, `+ - * /` with complex and real
operands), the conversions to and from `complex<float>`/`complex<double>` (CH2), and
`test_half_division_in_constant_expression<T>()`: the division check with tolerances
for half (the exact Smith branches, `big/big` at `finite_max`, and the scaled divisor
at `norm_min`; the float/double version's range checks are tighter than half precision,
and `finite_max_v<T> / 2` does not compile for a `BitComparisonWrapper`). No library
change was needed: `Kokkos::complex` was already `constexpr`, and with B4a its half
arithmetic (in float, CH3) and the constants `complex_infinity<T>()` etc. are constant
expressions. The run-time twins (`complex_constexpr_checks_at_runtime`) stay for every
backend. Compile-checked with the GCC 12 shim (B4a); verified on Aurora for `half_t` and
`bhalf_t` (job 9121273, with B4a).

**CH6 (optional, performance; from §3b). Paired-half arithmetic on CUDA. — FUTURE WORK (decided 2026-11-24).** libcu++ stores
`complex<__half>` as a `__half2` (`alignas(4)`) and uses `__hadd2`, `__hsub2` and
`__hbeq2` for `+= -= ==`. A Kokkos equivalent:
- Keep the two-member layout, which keeps the tuple protocol and `real()&` references
  working, but force `alignas(2*sizeof(half_t))` for half types, even without
  `KOKKOS_ENABLE_COMPLEX_ALIGN`. The current layout is already compatible with `__half2`.
- Under `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH`, implement `+ - += -= ==` for `complex<half_t>`
  with `__half2` intrinsics through `bit_cast`.

Do this only if a benchmark on A100 shows a gain. nvcc may already pair the two scalar
`__hadd`s. Do not apply it to `*` and `/`, which compute in float (D5).
- ABI caveat: changing the alignment of `complex<half_t>` is an ABI change. That is
  acceptable only if CH1, which introduces the type, already includes the alignment.

*CH6 status (2026-11-22): not implemented. The benchmark shows no gain, the plan's
condition for doing it (job 10397058, `tmp/claude_builds/complex_a6/ch6bench.cpp`,
A100, 2²⁶ `complex<half_t>`):*

| op | case | scalar (current) | paired `__half2` |
|---|---|---|---|
| `+` | stream `c = a + b` | 0.601 ms | 0.601 ms |
| `+` | chain of 256 per thread | 0.100 ms | 0.102 ms (1.02×) |
| `-` | chain of 256 per thread | 0.100 ms | 0.102 ms (1.02×) |
| `==` | count over all pairs | 0.454 ms | 0.453 ms |

Results are bit-identical (all 2²⁶ sums; same `==` counts).
- **nvcc does not pair the scalar operations.** The SASS of the chain kernels has 128
  `HADD2` (scalar) vs. 32 (paired) for the same unrolled loop. So the paired form halves
  the instruction count.
- **That does not show in time.** These kernels are bound by memory bandwidth (stream,
  `==`) or by the latency of the dependent chain. A gain could appear only in code
  bound by instruction issue; that was not demonstrated.
- **Revisit with a benchmark of such code before implementing.** It would also need
  the pack/unpack form (no `bit_cast`, see above) and a `KOKKOS_IF_ON_DEVICE` branch
  under `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH`.

Findings from the benchmark setup, reconciled with the source:
- **The alignment is already there by default.** `Kokkos_ENABLE_COMPLEX_ALIGN` defaults
  to ON (`cmake/kokkos_enable_options.cmake:128`) and is ON in this build, so
  `complex<half_t>` has `alignas(4)`: same size and alignment as `__half2`
  (static_assert in the benchmark). Forcing it would only matter for builds that turn
  the option off. Since CH1 is not in a release, that would still not be an ABI break.
- **`bit_cast` to `__half2` is impossible.** `__half2` is not trivially copyable in
  CUDA 12.6: it has a user-provided copy ctor and assignment (`cuda_fp16.hpp:771-791`),
  so `Kokkos::bit_cast` (and `std::bit_cast`) reject it. The lanes must be packed and
  unpacked with `__halves2half2`/`__low2half`/`__high2half`. The benchmark measures that
  form.

---

## 6. Feasibility summary

| Item | Feasible? | Notes |
|---|---|---|
| `complex<half_t>` compiles and runs, all backends | yes | CH1; already demonstrated on A100 (half-complex.md) |
| Same results as `std::complex<std::float16_t>` for `+ - * /` and functions | approximately | CH3/D5; bit-exactness with libstdc++ not a goal (Kokkos's own algorithms) |
| `constexpr` `complex<half_t>` on std backend | yes | B4a + CH5; future work (B4 deferred 2026-11-22) |
| `constexpr` `complex<half_t>` on CUDA/HIP/SYCL | CUDA, SYCL: yes; HIP: only with a workaround | B4b + CH5; future work. See the B4 per-backend table (CUDA prototype verified; `sycl::half` native; HIP union blocks reading bits) |
| `explicit` narrowing conversions like std | partly | CH2: only into half types; float/double unchanged for compatibility (Part A note) |
| Rejecting `complex<half_t>(1.5, 2.5)` like std | **no (deliberately)** | H1, Part B note: would break existing `half_t` users |
| Mixed-type complex arithmetic like std (forbidden) | **no (deliberately)** | D3 |
| Interop with `std::complex<std::float16_t>` | host C++23 only; constructor only (D11) | CH4 |
| `std::complex` interop beyond the converting constructor | **no (deliberately), deprecated** | D11, A8: deprecated in Kokkos 5, removed in Kokkos 6 |
| Annex G special values like libstdc++/libc++/libcu++ (§3b) | yes | `/` done (A1); `*` in A5; elementary functions in A6 |
| `constexpr` division like libcu++ | yes | A3b: a constant-evaluation branch with `constexpr` helpers |
| `std::complex<half_t>` portability (non-libstdc++) | unknown; not pursued | formally unspecified. Works with libstdc++ (CUDA; GCC 15 std backend, job 4260499). A2 plus the CH1 proxy remove the I/O use; A8 deprecates everything except the converting ctor, which is kept but not newly tested for half; the standard route for half is CH4 (`std::complex<std::float16_t>`) (C15, D11) |

## 7. Verification

- Per step: build `Kokkos_CoreUnitTest_{Serial,Cuda}_SmokeTest` Complex objects with
  `make -B -f .../build.make ...` (see half-complex.md §5), and `grep "warning #"`.
  Then run `--gtest_filter='*complex*'` on A100.
- Linking with `make -f .../build.make <exe>` also recompiles every object older than
  its source. On 2026-11-20 that was all 17 objects per executable, so use `-j` and run
  it in the background. The link itself cannot run on the login node, because
  `/usr/lib64/libcuda.so` exists only on GPU nodes. So compile the objects on the login
  node (`make -j8 -f .../build.make <all .o from link.txt>`), then link and run in the job
  (`sub_complex_A100.sh`: link both SmokeTests, then `--gtest_filter='*complex*'`).
- Correction to half-complex.md §5: in this build `build.make` *does* track header
  dependencies (via `compiler_depend.make`). After a change to `Kokkos_Complex.hpp`, all
  16 other SmokeTest objects are out of date and are recompiled before the link. `-B` is
  only needed to force a rebuild.
- Header-only changes (all of Part A) risk an ODR mix: if another linked object or
  `libkokkoscore.a` still holds an old weak copy of the changed template, the linker may
  keep it. Before trusting a partial rebuild, check with
  `nm -C <obj/lib> | grep 'Kokkos::operator/'`. For A1 no other SmokeTest object or
  library defined complex division. For A2, `Test*_Atomics.cpp.o` holds weak copies of
  complex `operator<<`, so the job also runs `*atomic*`.
- Part B steps: also run the half tests (`TestHalfOperators`, `TestHalfConversion`,
  `TestNumericTraits`). Part A must not change any float/double test expectation.
- Not available here: GCC ≥ 13 for the std backend (B4a, CH4, CH5), and HIP/SYCL builds.
  These need another machine; see §8.

## 8. Testing on other machines (written 2026-11-22)

Everything committed so far (A1-A7, B1-B3, CH1-CH3) was verified only with **GCC 12.2 +
nvcc 12.6, Serial and Cuda backends, on an A100 (sm_80)**. There, `half_t`/`bhalf_t` wrap
`__half`/`__nv_bfloat16`. Three other machines are available: one with GCC ≥ 13, one
with an AMD GPU and one with an Intel GPU. Each exercises a different half
implementation, selected as follows:

| Machine | Backend(s) | `half_t` / `bhalf_t` wrap | Selected by |
|---|---|---|---|
| this one (A100) | Serial + Cuda | `__half` / `__nv_bfloat16` | `Kokkos_Setup_Cuda.hpp:72-75` |
| GCC ≥ 13 | Serial (or OpenMP), **C++23** | `std::float16_t` / `std::bfloat16_t` ("std backend") | `__STDCPP_FLOAT16_T__` / `__STDCPP_BFLOAT16_T__`, `Kokkos_Macros.hpp:693-717`. Needs `-DCMAKE_CXX_STANDARD=23`; with C++20, `half_t` is `float` and the half tests test nothing new |
| AMD GPU | Serial + HIP | `__half` / `__hip_bfloat16` | `Kokkos_Setup_HIP.hpp:52-58`. `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH` on the host only for HIP ≥ 6.4 |
| Intel GPU | Serial + SYCL (icpx) | `sycl::half` / `sycl::ext::oneapi::bfloat16` | `Kokkos_Setup_SYCL.hpp:25-38`. `bhalf_t` is `float` unless `<sycl/ext/oneapi/bfloat16.hpp>` exists |

### 8.1 Which tests to run, and where they live

| Test | Source | Executable | Covers |
|---|---|---|---|
| all `*complex*` (23 tests; 25 with `*atomic*`) | `core/unit_test/TestComplex.hpp` | `Kokkos_CoreUnitTest_<Backend>_SmokeTest` and `..._Serial_SmokeTest` | A1-A7, B3, CH1-CH3 |
| `*atomic*` | `Test*_Atomics.cpp` | same | weak copies of complex `operator<<` (§7) |
| `half_io`, `bhalf_io` | `TestHalfIO.hpp` | `Kokkos_CoreUnitTest_Default` | B2 |
| `*half*` (conversion, operators) | `TestHalfOperators.hpp`, `TestHalfConversion.hpp` | `Kokkos_CoreUnitTest_Default` | regression for B1-B3 |
| `numeric_traits*` | `TestNumericTraits.hpp` | `Kokkos_CoreUnitTest_<Backend>1`, `..._Serial1` | B1, H16 (also the compile-time `static_assert`s) |
| `TestHalfCommonType` | `TestHalfCommonType.cpp` | compile-only (`COMPILE_ONLY_SOURCES`, `core/unit_test/CMakeLists.txt:36-41`): building it is the test | H7, B3's `promote_2` relies on it |

Suggested command per machine, after building the tests (`-DKokkos_ENABLE_TESTS=ON`):

```sh
for e in core/unit_test/Kokkos_CoreUnitTest_{Serial,<Backend>}_SmokeTest; do $e --gtest_filter='*complex*:*atomic*'; done
core/unit_test/Kokkos_CoreUnitTest_Default --gtest_filter='*half*'
for e in core/unit_test/Kokkos_CoreUnitTest_{Serial,<Backend>}1; do $e --gtest_filter='*numeric_traits*'; done
```

(On the GCC ≥ 13 machine `<Backend>` is `Serial`/`OpenMP` only.) Also run `locale -a | grep -E 'de_DE|fr_FR'`.
The comma-locale check in `half_io` is skipped silently without one of these
locales (B2), so the run shows whether it was exercised.

### 8.2 GCC ≥ 13 (std backend, C++23)

Configure host-only: `-DCMAKE_CXX_STANDARD=23 -DKokkos_ENABLE_SERIAL=ON`, optionally
`-DKokkos_ENABLE_OPENMP=ON`. Then check `KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT`
(defined iff the compiler defines `__STDCPP_FLOAT16_T__`, e.g.
`echo | g++ -std=c++23 -dM -E -x c++ - | grep STDCPP`).

**Job script: `run_std_half_gcc13.sh`** (repository root; not committed). It is a PBS
job script for the GCC ≥ 13 machine:
- header `-q regular-g`, `select=1`, walltime 1:00:00, `group_list=jh220036`;
- submit it with `qsub run_std_half_gcc13.sh` from a checkout of `refactor-complex`;
- it changes to `$PBS_O_WORKDIR` and loads `cmake/3.31.1 gcc/15.2.0` (override with
  `MODULES`);
- it can also run directly with `bash`, from the current directory.

Its first line is `#!/bin/bash -e`, followed by `set +e`, so that a failing step still
reaches the summary. It:
1. records the environment: compiler, commit, comma-decimal locales;
2. stops unless GCC ≥ 13 defines `__STDCPP_FLOAT16_T__` with `-std=c++23` (bf16 only
   warns);
3. configures Serial (`ENABLE_OPENMP=ON` adds OpenMP) with `CMAKE_CXX_STANDARD=23` and
   `CMAKE_CXX_EXTENSIONS=OFF`. Strict ISO, because Kokkos otherwise uses `-std=gnu++23`,
   whose excess-precision default may differ;
4. preprocesses with the tests' own flags to show which half backend Kokkos selected
   (`KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT`, `*_T_IS_FLOAT`,
   `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH`, `__FLT_EVAL_METHOD__`), and stops unless it is
   the std backend;
5. builds `Kokkos_CoreTestCompileOnly` (contains `TestHalfCommonType`),
   `Kokkos_CoreUnitTest_Serial_SmokeTest`, `..._Serial1` and `..._Default`, counting
   warnings;
6. runs the §8.1 filters (`*complex*:*atomic*`, `*numeric_traits*`, `*half*`; `FULL=ON`:
   also everything) and prints a summary, with a nonzero exit status on any failure.

Settings are environment variables (`KOKKOS_SRC`, `BUILD_DIR`, `CXX`, `BUILD_TYPE`,
`JOBS`, `MODULES`, `FULL`; with PBS: `qsub -v NAME=value`); logs go to
`$BUILD_DIR/std_half_logs/`, and the job's output (`-j oe`) is the summary.
The compiler is chosen after `module purge`/`module load`: a `CXX` from the submit
environment would name the purged compiler (this was the first run's `CXX: unbound
variable`), so `CXX` is used only with `MODULES=`. `TERM` is set to `dumb` for the
`module` command; the remaining `tput: No value for $TERM` lines at the top of the job
output come from the job shell's startup, before the script, and are harmless.

*First run (PBS job 4260499, 2026-11-22): all pass.* `mg0959` (aarch64), GCC 15.2.0,
CMake 3.31.1, commit `2379511da`, Debug, Serial, `-std=c++23`:
- the probe: `STD_HALF=1`, `STD_BHALF=1`, `HALF_FULL_TYPE=1`, `HALF_T_IS_FLOAT=false`,
  `BHALF_T_IS_FLOAT=false`, `FLT_EVAL_METHOD=0`, `FLT_EVAL_METHOD_TS_18661_3=0`. Both
  half types wrap the std types, so the tests below exercise real 16-bit types on the
  host for the first time;
- builds: the 4 targets, 0 warnings (so `TestHalfCommonType` passes);
- runs: `*complex*:*atomic*` 25/25 (including `complex_half_scalar_promotion`,
  `complex_half_conversions`, `complex_half_compute_in_float`), `*numeric_traits*`
  14/14, `*half*` 8/8 (including H17's `half_`/`bhalf_bool_conversion`);
- not covered: the node has no `de_DE`/`fr_FR` locale, so the locale branch of
  `half_io` did not run; OpenMP and `FULL=ON` were not run.

**Job script `run_std_half_fp16.sh`** (repository root): the same tests on a CPU with
native half arithmetic. Written for Aurora (PBS, `-A kokkos_math`, `module load cmake`,
GCC 14 as `/usr/bin/g++-14`; a non-GCC `CXX` such as oneAPI's `icpx` is ignored). It
builds twice: `default` (no `Kokkos_ARCH_*`) and `native` (`Kokkos_ARCH_SPR`,
`-march=sapphirerapids`), and for each prints the evaluation-method macros, the code of
one half multiplication at `-O2`, and a probe program whose results differ between
float and half evaluation (`std::float16_t` scalar, `std::complex<std::float16_t>` and
`Kokkos::complex<half_t>` `*` and `/`). Its output is also saved as
`std_half_fp16_logs.<job>/job.out`.
*Extended 2026-11-23:* two more builds in Kokkos's default GCC dialect, `gnu` and
`gnu-native` (`CMAKE_CXX_EXTENSIONS=ON`, `-std=gnu++23`; the probe prints the
`-std=` flag and `__STRICT_ANSI__`), so `CONFIGS` defaults to
`default native gnu gnu-native`; a build now fails on GCC's "used but never defined".

*Four-build run (PBS job 9117216, 2026-11-23, commit `417e5844d`): all pass.* Aurora,
GCC 14.2, Xeon Max 9470C:

| | `default` | `native` | `gnu` | `gnu-native` |
|---|---|---|---|---|
| dialect | `-std=c++23` (`__STRICT_ANSI__`) | `-std=c++23` | `-std=gnu++23` | `-std=gnu++23` |
| method / half `*` | 0 / through float | 16 / `vmulsh` | 0 / through float | 16 / `vmulsh` |
| warnings, 5 targets | 0 | 0 | 0 | 0 |
| "used but never defined" | none | none | none | none |
| `*complex*:*atomic*` / `*numeric_traits*` / `*half*` | 25 / 14 / 12 | 25 / 14 / 12 | 25 / 14 / 12 | 25 / 14 / 12 |

- **GNU dialect:** the probe values equal the strict builds' line for line
  (`std::float16_t` `256*256-65000`: 544 under method 0, inf under 16;
  `std::complex<std::float16_t>` `(269,111.5)²`: (59936, 60000) / (inf, 60000);
  Kokkos (59936, 60000) in all four), and all tests pass. So the different
  `-fexcess-precision` default of GNU mode changes nothing for these expressions and
  tests; that is the extent of the check, not a proof for every half expression.
- **Declared casts:** `half_cast_overloads` and `bhalf_cast_overloads` pass in all four
  builds, and no build reports "used but never defined". Every cast declared in
  `Kokkos_Half_FloatingPointWrapper.hpp`, and every wrapper operation that reaches one,
  is defined on the std backend: H18 was the only gap of its kind.

*libstdc++ deviates from C++23 on extraction when the sentry fails (found
2026-11-23, from the source).* [istream.formatted.arithmetic] does the extraction only
if the sentry succeeds, so on empty input, or input that is only whitespace, the value
stays unchanged; Kokkos does that (B2 revision, `half_io`). libstdc++'s
`operator>>(_Float16&)` and `operator>>(__bfloat16_t&)` (`<istream>`, GCC 14 and 15 and
current master, unchanged) declare `float __flt;`, call `_M_extract(__flt)`, which
does not write it when the sentry fails, and then assign
`__f = static_cast<_Float16>(__flt)` unconditionally: the result is an indeterminate
value. So `half_io_vs_std` does not compare such inputs. Worth reporting to GCC.

*aarch64 (added 2026-11-23):* the script detects the CPU: on aarch64 the native builds
use `Kokkos_ARCH_ARMV9_GRACE`, the CPU check is `fphp` in `/proc/cpuinfo`, and the code
probe counts `fmul h…` (native) and `fcvt s…, h…` (through float). On the Grace
machine submit `run_std_half_fp16_grace.sh`, which sets that machine's PBS header and
modules and runs `run_std_half_fp16.sh`. Step by step: `aarch64_tests.md`.

*Grace run (PBS job 4264666, 2026-11-23, commit `48e69227b`): all pass, native half
confirmed.* `mg0639`, Neoverse-V2 (`fphp`, `asimdhp`), GCC 15.2.0, CMake 3.31.1, Debug,
Serial. The builds need binutils ≥ 2.41: the `gcc/15.2.0` module uses the system
binutils 2.35, whose assembler rejects `-mcpu=neoverse-v2` ("unknown architecture
armv9-a+…"), so the job ran with gcc-toolset-14's binutils 2.41 first on `PATH`
(`qsub -v PATH=…`; only binutils, not that toolset's GCC 14; `aarch64_tests.md`
Prerequisites). Since then `run_std_half_fp16_grace.sh` does this itself: if the
assembler cannot compile `-mcpu=neoverse-v2`, it puts a symlink directory to the newest
gcc-toolset binutils that can first on `PATH` (override: `BINUTILS_DIR`). Checked with a
plain `qsub` of commit `bfab679c6` and an empty build root (job 4271100,
`CONFIGS=native`, `BUILD_ROOT=…/grace_fresh_build`): it chose gcc-toolset-14's
binutils 2.41, compiled the 5 targets from scratch (157 files, 0 warnings), the probe
values equal job 4264666's, and 25 / 15 / 14 tests pass. (The earlier job 4265289 is not
a check of the fix: it reused job 4264666's `build_std_half_native`, so nothing but the
probes was compiled.)

| | `default` | `native` | `gnu` | `gnu-native` |
|---|---|---|---|---|
| dialect | `-std=c++23` | `-std=c++23`, `ARMV9_GRACE` | `-std=gnu++23` | `-std=gnu++23`, `ARMV9_GRACE` |
| method / half `*` | 0 / through float (`fcvt`) | 16 / `fmul h…` | 0 / through float | 16 / `fmul h…` |
| `std::float16_t` `256*256-65000` | 544 | inf | 544 | inf |
| `std::complex<std::float16_t>` `(269,111.5)²` | (59936, 60000) | **(inf, 60000)** | (59936, 60000) | **(inf, 60000)** |
| `Kokkos::complex<half_t>` `(269,111.5)²` | (59936, 60000) | (59936, 60000) | (59936, 60000) | (59936, 60000) |
| warnings, 5 targets | 0 | 0 | 0 | 0 |
| `*complex*:*atomic*` / `*numeric_traits*` / `*half*` | 25 / 15 / 14 | 25 / 15 / 14 | 25 / 15 / 14 | 25 / 15 / 14 |

`/` gives (60000, 0) in all four. So Grace with `ARMV9_GRACE` behaves as Sapphire
Rapids with `Kokkos_ARCH_SPR` (job 9117216): method 16, native half multiplication,
libstdc++'s complex `*` overflows while Kokkos's stays in float, and every test passes
unchanged. Note on the stand-alone check (`aarch64_tests.md` step 3): at `-O2`, GCC's
default `-ffp-contract=fast` fuses `a * a - c` into a half `fmadd` (one rounding, no
overflow), which prints 544 under method 16; with `-ffp-contract=off` it prints inf. The
job's probes use the tests' Debug flags and do not contract.

*Finite-math guards (2026-11-25).* One Kokkos CI job builds with clang and
`-ffinite-math-only`; Kokkos's math tests skip their NaN/infinity checks there
(`#if !__FINITE_MATH_ONLY__`). The tests this branch added did not. Now
`complex_division`, `complex_multiplication`, `complex_arg_proj`,
`complex_exp_sqrt_log_special_values`, `complex_hyperbolic_special_values`,
`complex_tanh_large_argument` and `complex_inverse_special_values` skip under
`__FINITE_MATH_ONLY__`; `complex_half_compute_in_float` leaves out its special-value
lines, `half_bool_conversion` its infinity and NaN inputs, and `half_io` its
`"+-1e999"` inputs. Found while checking the icpx flags of the first SYCL run.

*Fixed special-value reference (2026-11-25).* The special-value tests
(`complex_exp_sqrt_log_special_values`, `complex_hyperbolic_special_values`,
`complex_inverse_special_values`) take the expected kind and sign of each result from
`complex_special_ref` in `TestComplex.hpp`: glibc's, generated with GCC 12 by
`tmp/claude_builds/complex_a6/gen_complex_special_ref.cpp` (the same for float and
double), and checked against the CUDA and std backends, which matched glibc. Before,
they compared with the host's `std::complex`, which under icpx is Intel's library.
`std::complex` still provides the finite values (within the tolerance). Checked here:
GCC 12 host 25 complex tests pass, `-ffinite-math-only` 18 pass and 7 skip. Not related, found
with the GCC 12 shim: `TestHalfOperators.hpp:413` assumes `bhalf_t` is a wrapper whenever
`half_t` is, which fails where `std::float16_t` exists without `std::bfloat16_t` (no
machine tested has that).

*Options added 2026-11-23:* `ENABLE_OPENMP=ON` also enables the OpenMP backend and
builds and runs `Kokkos_CoreUnitTest_OpenMP_SmokeTest`/`OpenMP1` (separate build
directories `build_std_half_<config>_omp`, `OMP_NUM_THREADS` default 8). `FULL=ON` also
builds `<backend>2` and `<backend>_ViewSupport` and runs every test executable without
a filter; it needs more than 1 hour, and runs longer than 1 hour must use the
`capacity` queue (`qsub -q capacity -l walltime=3:00:00 -v FULL=ON,...`). A
skipped test of the std-backend tests (`*_vs_std*`) fails the summary; other skips are
Kokkos's own (death tests, graphs, shared spaces) and are only counted.

*Runs of `f0740e516` (2026-11-23, Aurora, GCC 14.2; both jobs at the same time, they
share no build directories).*
- **Job 9117303** (`default native gnu gnu-native`, Serial): all pass, 0 warnings;
  `*complex*:*atomic*` 25, `*numeric_traits*` 15, `*half*` 14 in each build. The new
  tests ran (not skipped) and pass in all four: `numeric_traits_half_vs_std_numeric_limits`
  (every trait equals `std::numeric_limits` of `std::float16_t`/`std::bfloat16_t`), and
  `half_io_vs_std`/`bhalf_io_vs_std` (Kokkos's extraction and insertion equal
  libstdc++'s for every compared input and format setting).
- **Job 9117309** (`default gnu-native`, `ENABLE_OPENMP=ON`, `FULL=ON`, `capacity`
  queue, `OMP_NUM_THREADS=208` from the environment): every built executable passes in
  both builds, 0 warnings: Serial_SmokeTest 126, Serial1 217, Serial2 127,
  Serial_ViewSupport 37, OpenMP_SmokeTest 125, OpenMP_ViewSupport 37, Default 74 tests;
  the new tests pass here too. Two script errors, fixed since: the OpenMP test
  executable is `Kokkos_CoreUnitTest_OpenMP` (not `OpenMP1`/`OpenMP2`, which do not
  exist; so it was not built or run), and the skips it reported (7 in Serial1, 3 in
  Default, …, each counted twice) are Kokkos's own, not failures.
- **Job 9117446** (rerun at `ef0391887`, same settings, `capacity` queue): all pass in
  both builds, all 11 targets with 0 warnings:

  | executable | tests |
  |---|---|
  | `Serial_SmokeTest` / `Serial1` / `Serial2` / `Serial_ViewSupport` | 126 / 217 / 127 / 37 |
  | `OpenMP_SmokeTest` / `OpenMP` / `OpenMP_ViewSupport` / `OpenMPInterOp` | 125 / 333 / 37 / 1 |
  | `Default` | 74 |
  | filtered: `*complex*:*atomic*` (Serial, OpenMP) / `*numeric_traits*` (Serial1, OpenMP) / `*half*` | 25, 25 / 15, 15 / 14 |

  `numeric_traits_half_vs_std_numeric_limits` passes on `serial` and `openmp`, and
  `half_io_vs_std`/`bhalf_io_vs_std` pass, in both builds. The skips (4-9 per Serial
  executable, 5 in `OpenMP_SmokeTest`, 3 in `Default`, 27 in `OpenMP`: 10 death tests and
  17 infrastructure tests such as partitioning, execution-space thread safety and
  allocation corner cases) are Kokkos's own; none involves complex, half or the numeric
  traits. **So the committed complex/half work passes the complete Kokkos core test
  suite on the std half backend, Serial and OpenMP, in the GNU dialect and with native
  half arithmetic.**

*Release run (PBS job 9120562, 2026-11-24, commit `35bac0593`, `BUILD_TYPE=Release`,
`-O3`, `capacity` queue): all pass.* Aurora, GCC 14.2, the four builds in
`build_std_half_<config>_release`: 0 warnings, 25 / 15 / 14 tests pass in each,
including `complex_half_compute_in_float` and `complex_multiplication`. The probe values
of the builds without FMA (`default`, `gnu`) equal the Debug runs; the native builds
(`native`, `gnu-native`) differ:

| probe (`native`, `gnu-native`) | Debug, `-O0` (job 9117216) | Release, `-O3` (job 9120562) |
|---|---|---|
| `std::float16_t` `256*256-65000` | inf | **544** |
| `half_t` `256*256-65000` | inf | **544** |
| `std::complex<std::float16_t>` `(269,111.5)²` | (inf, 60000) | **(59936, 60000)** |
| `Kokkos::complex<half_t>` `(269,111.5)²` | (59936, 60000) | (59936, 60000) |
| `(60000,60000)/(1,1)`, both | (60000, 0) | (60000, 0) |

- **Cause: GCC's floating-point contraction** (its default `-ffp-contract=fast`, in
  strict `-std=c++23` too, which disproves the "probably off for strict C++" guess). The
  job's binaries could not be disassembled here (binutils 2.30 and LLVM 13 predate
  AVX512-FP16), so it was reproduced with GCC 12, `-O3 -march=sapphirerapids`: a plain
  `_Float16` `a*a - c`, the same split over two statements, and a wrapper whose `*` and
  `-` each return a half (as `half_t`'s) all compile to one `vfmsub132sh`, with
  `-std=c++20` and `-std=gnu++20`; with `-ffp-contract=off`, `vmulsh` + `vsubsh`.
- **So `half_t` is not rounded per operation in optimized code on FMA hardware:** GCC
  fuses across the inlined wrapper operators, as it does for `float` and `double`. Not
  a Kokkos bug, but half results then depend on the optimization level; code that needs
  per-operation rounding builds with `-ffp-contract=off`. The probe's label for
  `half_t` said "rounded per operation: inf"; now "inf unless contracted".
- **Kokkos's complex results do not change** (CH3 computes in float, and the bit-exact
  tests pass). D5 gets a qualifier (CH3 notes).

*Run on Aurora (PBS job 9116902, 2026-11-22): all tests pass.* Xeon Max 9470C
(`avx512_fp16`), GCC 14.2.0, commit `b5d91eb67`, Debug, Serial:

| | `default` | `native` (`Kokkos_ARCH_SPR`) |
|---|---|---|
| `__FLT_EVAL_METHOD_TS_18661_3__` | 0 | **16** |
| `__AVX512FP16__` | — | 1 |
| `HALF_T_IS_FLOAT`, `BHALF_T_IS_FLOAT` | false, false | false, false |
| one half `*` (`std::float16_t` and `half_t`) | through float: `__extendhfsf2`, `mulss`, `__truncsfhf2` | **`vmulsh`** |
| tests `*complex*:*atomic*` / `*numeric_traits*` / `*half*` | 25 / 14 / 8 pass | 25 / 14 / 8 pass |
| build warnings | 0 | 279, one kind (below); 7 after the fix (job 9116933) |

So with native half arithmetic the scalar `half_t` operators compute in half, and the
complex tests still pass, including `complex_half_compute_in_float` (bit-identical to the
float path): CH3's `*`, `/` and functions do not depend on the evaluation method. The
probe program did not compile in this run (`complex<H> k(H(v269), H(v111))` parsed as a
function declaration; fixed with braces).

*Rerun (PBS job 9116933, 2026-11-22, commit `3c70cfdca`): all tests pass again, and the
probe runs.* Same node type and compiler. Probe output:

| | `default` (method 0) | `native` (method 16) |
|---|---|---|
| `std::float16_t` `256*256-65000` | 544 (float: 65536 − 64992, as 65000 rounds to 64992 in half) | inf |
| `half_t` `256*256-65000` | inf (each operator rounds to half) | inf |
| `std::complex<std::float16_t>` `(269,111.5)²` | (59936, 60000) | **(inf, 60000)** |
| `Kokkos::complex<half_t>` `(269,111.5)²` | (59936, 60000) | (59936, 60000) |
| `(60000,60000)/(1,1)`, std and Kokkos | (60000, 0) | (60000, 0) |

This closes the D5 multiplication question (see "CH3 vs. C++23 and libstdc++").

*The 279 warnings* are GCC ≥ 13's "`__bfloat16` is redefined from typedef `short` to
real `__bf16` since GCC 13.1 …", emitted wherever the x86 BF16 intrinsics are visible.
They come from the `short`/`unsigned short` overloads of `cast_to_bhalf`/`cast_from_bhalf`
in `Kokkos_STD_Half_Conversion.hpp` (predating this branch, `c7d070bc6`), and GCC also
reports them at six test lines that construct `bhalf_t` (`TestNumericTraits.hpp:114,123`,
`TestMathematicalFunctions.hpp:2965-2967`, `TestReduce.hpp:789`,
`TestHalfOperators.hpp:972`). The results are right (`bhalf_conversion` and the numeric
traits pass). *Fix (commit `3c70cfdca`, verified by job 9116933):* those four overloads
now convert through `float`, which holds every `short`, `unsigned short` and
`std::bfloat16_t` value exactly, so the results are those of the direct conversion.
279 → 7 warnings; none from `Kokkos_STD_Half_Conversion.hpp`, all tests pass.

*The 7 remaining warnings* are the six test lines above (`TestNumericTraits.hpp:114`
twice). Each initializes a `bhalf_t` from a numeric-trait constant
(`infinity`/`epsilon`/`finite_max`… of `bhalf_t`, a `BitComparisonWrapper`). That goes
through `floating_point_wrapper(bit_comparison_type)`, which copies the bits with
`Kokkos::bit_cast` (`memcpy`), so there is no value conversion, and the values are
checked by the passing tests. Status: **harmless false positives; no library change.**

*Cause, from `run_bf16_warning_probe.sh` (PBS job 9116988, GCC 14.2, Xeon Max).* The
script compiles 10 small cases under four flag sets (none, `-mavx512bf16`,
`-mavx512fp16`, `-march=sapphirerapids`) and counts the warning:

| case | code | warnings (none / bf16 / fp16 / spr) |
|---|---|---|
| M1 | struct{`uint16_t`} → struct{`__bf16`}, copy-init through a constructor | 0 / 1 / 0 / 1 |
| M2 | the same, direct-init | 0 / 0 / 0 / 0 |
| M3 | copy-init from a 4-byte struct | 0 / 0 / 0 / 0 |
| M4 | `short` → `std::bfloat16_t` (the case fixed in `3c70cfdca`) | 0 / 2 / 0 / 2 |
| M5 | `float` → `std::bfloat16_t` | 0 / 0 / 0 / 0 |
| K1 | `bhalf_t x = infinity<bhalf_t>::value;` (the test lines) | 0 / 1 / 0 / 1 |
| K2 | `bhalf_t x(infinity<bhalf_t>::value);` | 0 / 0 / 0 / 0 |
| K3 | `x = infinity<bhalf_t>::value;` | 0 / 0 / 0 / 0, but see H18 |
| K4 | `half_t x = infinity<half_t>::value;` | 0 / 0 / 0 / 0 |
| K5 | `bhalf_t(short(1))` (after `3c70cfdca`) | 0 / 0 / 0 / 0 |

So GCC warns whenever BF16 is enabled and a copy-initialization converts between a
2-byte struct holding a `uint16_t` (`BitComparisonWrapper`) and one holding a `__bf16`
(`bhalf_t`), although the constructor only copies bits: it treats the two structs like
`short` and `__bf16`. Direct initialization does not convert and stays silent. User code
`bhalf_t x = Kokkos::infinity_v<bhalf_t>;` gets the same warning with
`-march=sapphirerapids`; `bhalf_t x(Kokkos::infinity_v<bhalf_t>);` avoids it. Not
explained: why the conversion inside `operator=(T)` (K3) does not warn. K3 instead
showed H18 ("`cast_to_bhalf(bhalf_t)` used but never defined").

*The warning in detail (seen again in job 9117024, `native` build).*

```text
core/unit_test/TestHalfOperators.hpp: In instantiation of 'void Test::_test_half_operators(half_type, half_type)
    [with half_type = Kokkos::Experimental::Impl::floating_point_wrapper<__bf16>]':
core/unit_test/TestHalfOperators.hpp:1035:34:   required from here
core/unit_test/TestHalfOperators.hpp:972:13: warning: '__bfloat16' is redefined from typedef 'short'
    to real '__bf16' since GCC 13.1, be careful of implicit conversion between '__bf16' and 'short';
    an explicit bitcast may be needed here
```

- **Source line:** `half_type epsilon = Kokkos::epsilon<half_type>::value;` in
  `_test_half_operators<bhalf_t>`. Job 9117024 rebuilt only `Kokkos_CoreUnitTest_Default`,
  so only this one of the 7 appeared; the other sites are in `Kokkos_CoreUnitTest_Serial1`
  and `..._SmokeTest`.
- **All 7 have the same form**, a copy-initialization of `bhalf_t` from a trait constant:

  | site | code |
  |---|---|
  | `TestHalfOperators.hpp:972` | `half_type epsilon = Kokkos::epsilon<half_type>::value;` |
  | `TestNumericTraits.hpp:114` (twice: two instantiations) | `T const inf = infinity<T>::value;` |
  | `TestNumericTraits.hpp:123` | `T const eps = epsilon<T>::value;` |
  | `TestMathematicalFunctions.hpp:2965` | `const FP16Type pos_max = finite_max<FP16Type>::value;` |
  | `TestMathematicalFunctions.hpp:2966` | `const FP16Type neg_max = finite_min<FP16Type>::value;` |
  | `TestMathematicalFunctions.hpp:2967` | `const FP16Type pos_inf = infinity<FP16Type>::value;` |
  | `TestReduce.hpp:789` | `ScalarType inf = Kokkos::infinity_v<ScalarType>;` |

- **When it appears:** GCC ≥ 13, `bhalf_t` on the std backend (`std::bfloat16_t`), and
  BF16 instructions enabled (`-mavx512bf16`, or `-march=sapphirerapids` through
  `Kokkos_ARCH_SPR`). Never for `half_t`, nor without BF16.
- **What it means here:** nothing. The conversion is
  `BitComparisonWrapper` → `bhalf_t` through
  `floating_point_wrapper(bit_comparison_type)`, which copies the bits
  (`Kokkos::bit_cast`, `memcpy`). GCC sees a 2-byte struct holding a `uint16_t` converted
  into a 2-byte struct holding a `__bf16`, and treats it like the `short` → `__bf16`
  conversion it is meant to catch. The values are right: the IEEE bit patterns are
  checked by `numeric_traits_*` and, since H18, by `bhalf_assign_trait_constants`.

*Ways to suppress it* (checked 2026-11-22):

| option | works? | cost |
|---|---|---|
| `-Wno-…` flag or `#pragma GCC diagnostic ignored` | **no**: GCC emits it with no option name (no `[-W…]` tag in the message), so neither can target it | — |
| `-w` | yes | silences every warning; not acceptable |
| **Direct-initialize at the 6 test lines**: `T const inf(infinity<T>::value);` | **yes** (probe K2: silent under all four flag sets) | 6 one-line test edits; same value, same code path (the `bit_comparison_type` constructor) |
| Library: make `BitComparisonWrapper` not look like a 2-byte integer, e.g. 4 bytes with padding (probe M3: silent), and bit-cast `rhs.value` in the constructor | likely, also for user code | changes an `Impl` type's size; needs a check on all backends (CUDA/HIP/SYCL device code uses it). Not worth it for a false positive |
| Library: an explicit constructor or a conversion operator instead | no / breaks code | `explicit` would break `bhalf_t x = infinity_v<bhalf_t>;` in user code; adding a conversion operator makes copy-initialization ambiguous |

**Recommendation:** the test-side change only, i.e. direct initialization at the 6
lines. That removes the warning from the Kokkos build and tests nothing less. Leave the
library as is. User code with `bhalf_t x = Kokkos::infinity_v<bhalf_t>;` under
`-march=sapphirerapids` will see the same harmless warning, and
`bhalf_t x(Kokkos::infinity_v<bhalf_t>);` avoids it. **Applied 2026-11-22** at the 6
lines (commit `664a10b78`; the table above shows the old code). Built and run here
where `half_t` is `float`, and compiled for CUDA (A100 build). **Verified on Aurora
(PBS job 9117159, `run_std_half_fp16.sh`, commit `664a10b78`): 0 warnings in both
builds**, `native` included. That build recompiled every object containing the 6 lines
(`TestSerial_NumericTraits`, `TestSerial_MathematicalFunctions1-3`,
`TestSerial_Reductions`, `TestDefaultDeviceType`), and all tests pass:
`*complex*:*atomic*` 25, `*numeric_traits*` 14, `*half*` 10 (with the H18 tests), in
both builds; the probe gives the same values as job 9116933.

*Warnings of the `-march=sapphirerapids` build, summary:* 279 (job 9116902) → 7 after
the `short` ↔ bf16 conversions in `Kokkos_STD_Half_Conversion.hpp` went through float
(`3c70cfdca`, job 9116933) → 0 after the test lines used direct initialization
(`664a10b78`, job 9117159).

*Validation of committed work* (first done by the run above; the points below all
passed there):
- **All tests in §8.1.** Particular points:
  - **CH1 constants:** `T(infinity_v<T>)` from `BitComparisonWrapper` with a
    `std::float16_t` `impl_type`.
  - **CH1 I/O proxy and B2:** reading through `double`, then
    `cast_to_half(double)` = `std::float16_t(double)`. This must round once; the
    `half_io` rounding cases check it.
  - **CH3:** with `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH` the wrapper's scalar operators use
    `std::float16_t` arithmetic. GCC may evaluate `_Float16` with excess precision
    (`FLT_EVAL_METHOD`, `-fexcess-precision`). `complex_half_compute_in_float` compares
    bits with the float path, so it shows whether any half arithmetic is left in the
    complex code; the scalar tests may need checking.
  - **B1/H16:** the trait values must hold for `std::float16_t`/`std::bfloat16_t` too.
    The `static_assert`s compare against the exact IEEE values, and `std::numeric_limits`
    of the std types can be used as an independent reference.
- ~~**`std::complex<half_t>` interop**~~ (half-complex.md §3.2). *Done 2026-11-22, job
  4260499 (GCC 15.2 libstdc++).* There `std::complex<half_t>` is
  `std::complex<floating_point_wrapper<std::float16_t>>`, unspecified but handled by
  libstdc++'s generic template. `complex_construction` for `half_t`/`bhalf_t`
  (`TestComplex.hpp:79-99`) passed. It covers the conversion to `std::complex` (I1:
  `std::complex<half_t> sa(a)`, `r = a`), `std::complex * Kokkos::complex` (I6:
  `r = sb * a`) and `std::complex<half_t>`'s own assignment; not the converting ctor
  from `std::complex<half_t>` (I0), which no half test uses. GCC 13 and 14 were not
  run (the generic template is long-standing code; low risk). No further work: A8
  deprecates I1 and I6, keeps their half coverage in its deprecated-code test, and
  does not add tests constructing from `std::complex<half_t>`; for half on C++23 the
  standard route is CH4.
- ~~**D5 to-do**~~ (*answered 2026-11-22*; see "CH3 vs. C++23 and libstdc++"): libstdc++
  computes all functions and `/` in float; `*` follows the evaluation method: float
  under method 0 (aarch64 default, job 4260499; x86 default, job 9116933), half under
  method 16 (`-march=sapphirerapids`, job 9116933: `(269,111.5)²` = `(inf, 60000)`, while
  Kokkos gives `(59936, 60000)`). Optional: compare the functions with Kokkos over the
  `ch3acc` inputs.

*Future tasks that become testable here:*
- **B4a:** `constexpr` std backend; native conversions, so no software path.
- **CH4:** converting constructor from `std::complex<std::float16_t>` (D11).
- **CH5** (std part): enable the half `static_assert`s in `TestComplex.hpp` under
  `KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT`.

### 8.3 AMD GPU (HIP)

Configure `-DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_<AMD_GFX...>=ON` with hipcc/amdclang++.
Record the ROCm version: it decides `KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH` on the host
(HIP ≥ 6.4).

*Validation of committed work:*
- **All tests in §8.1** on Serial and HIP. Particular points:
  - **CH1 constants:** `T(infinity_v<T>)` goes through `Kokkos::bit_cast` into HIP's
    union-based `__half` (B4 table) in device code.
  - **CH3 on device:** the float path. Also check that `half_same` in
    `complex_half_compute_in_float` (bits, or both NaN) holds on AMD hardware; NaN
    payloads may differ (A6 lesson).
  - **CH2:** `half_t` → `float` is `explicit` in the wrapper on every backend, so the
    explicit/implicit table should be identical. Check it with `ch2probe.cpp`.
  - **H17 does not apply:** HIP defines `cast_to_half(bool)` and `cast_to_bhalf(bool)`
    (`Kokkos_HIP_Half_Conversion.hpp:19,188`).
- **Optional benchmarks** (CH3 decision, CH6): port `ch3bench.cpp` (it uses only Kokkos)
  and `ch6bench.cpp` (needs the HIP `__half2` intrinsics) to confirm that the float path
  costs nothing there either.

*Future task:* **B4b-HIP.** It needs a way to read `__half` bits in constant evaluation
(B4 table). First test the derived-class workaround with the real ROCm headers; the
probe here used a mock with clang 17.

### 8.4 Intel GPU (SYCL)

Configure `-DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_<GEN>=ON` with icpx. Check whether
`KOKKOS_HAS_NATIVE_BHALF_TYPE` is defined: without `<sycl/ext/oneapi/bfloat16.hpp>`,
`bhalf_t` is `float` and the bhalf tests reduce to float.

*Validation of committed work:*
- **All tests in §8.1** on Serial and SYCL. Particular points:
  - **FP64 requirement:** the tests use `double` in kernels: `complex_half_conversions`,
    and the double instantiations of every complex test. Use a GPU with FP64 (Data
    Center GPU Max); Arc and most integrated GPUs lack it, and those kernels then fail
    for reasons unrelated to half. After CH3 the half code paths themselves no longer
    use `double` on the device: the half `<numbers>` constants are instantiated only in
    the discarded half branches.
  - **Device math:** `Kokkos_MathematicalFunctions.hpp` uses the `sycl::` namespace;
    CH3 relies on the float `complex` functions there.
  - **`Kokkos::bit_cast` is `sycl::bit_cast`** (`Kokkos_BitManipulation.hpp:319-327`),
    used by `half_same` and `TestHalfIO`'s round trip.
- **Compile-time check of B4b-SYCL's premise** with the installed DPC++ version:
  `b4sycl.cpp` (`sycl::half` native `constexpr`, bfloat16 not).

~~*Future task:* **B4b-SYCL.**~~ Implemented 2026-11-25 (B4 notes, B4b-SYCL status);
to verify on Aurora.

### 8.5 Remaining tasks that cannot be tested on this machine

| Task | Kind | Where it can be tested |
|---|---|---|
| ~~Validation of A1-A7, B1-B3, CH1-CH3, H17 with the std half backend~~ | done 2026-11-22, job 4260499: 25 + 14 + 8 tests pass, 0 warnings (GCC 15, aarch64; Serial only) | GCC ≥ 13, C++23 (§8.2) |
| ~~`std::complex<half_t>` interop with libstdc++ ≥ 13 (half-complex.md §3.2)~~ | done 2026-11-22, job 4260499 (GCC 15; 13/14 not run); no further work (§8.2, A8) | GCC ≥ 13 |
| ~~D5 to-do: libstdc++ `complex<std::float16_t>` intermediate precision~~ | done 2026-11-22 (jobs 4260499, 9116902, 9116933): float except `*` under method 16, where libstdc++ overflows and Kokkos does not; optional: functions over the `ch3acc` inputs | GCC ≥ 13 |
| ~~Remaining 7 `__bfloat16` warnings at test lines (`-march=sapphirerapids`)~~ | done 2026-11-22, job 9116988: GCC false positive on struct-to-struct bit copies; harmless, no change (§8.2) | GCC ≥ 13 on Aurora |
| ~~H18: `cast_to_half(half_t)`/`cast_to_bhalf(bhalf_t)` missing on the std backend~~ | done: fixed `1220b11de`, verified on Aurora (job 9117024); CUDA run: job 10404035 | GCC ≥ 13 |
| ~~Direct-initialize the 6 test lines that trigger the `__bfloat16` false positive~~ | done: `664a10b78`, verified on Aurora (job 9117159: 0 warnings in both builds, all tests pass) | GCC ≥ 13 on Aurora |
| ~~Optimized CUDA build (Release, A100)~~ | done 2026-11-25: `build_gpu_release`, job 10430071: the complex, numeric-trait and half tests pass with the Debug counts on Serial and Cuda (`cuda-tests.md`) | A100 |
| ~~Complete CUDA core test suite (A100)~~ | done 2026-11-24 at `82a06f7a8`: every core test executable passes (jobs 10421072, 10421610), except `cuda_graph.sized_functor_launch`, which segfaulted in this build before the branch too (`cuda-tests.md`) | A100 |
| Validation of the committed work with HIP half/bhalf, host and device | verification | AMD GPU (§8.3) |
| ~~Validation of the committed work with SYCL half/bhalf, host and device~~ | done 2026-11-25, Aurora job 9127366 (Intel GPU Max 1550, icpx 2026.1): complex 27 on Serial and SYCL, numeric traits 14, half 12 (4 expected skips). Fixed on the way: H17 on SYCL, DPC++'s host `sycl::half` rounding, the Intel GPU's signed zeros of `sin`/`atan2`, the special-value reference (`sycl-tests.md`) | Aurora |
| ~~B4a: `constexpr` half, std backend~~ | done 2026-11-24 (`545a6aa5a`), verified on Aurora (job 9121273, four builds, `half_t` and `bhalf_t`); CUDA unchanged (A100 jobs 10415251, 10415253) | GCC ≥ 13 |
| CH4: converting constructor from `std::complex<std::float16_t>` (D11) | **future work** (2026-11-24): `std::float16_t` needs C++23, Kokkos's minimum is C++20; pick up once C++23 is the minimum | GCC ≥ 13 |
| ~~CH5: `constexpr complex<half_t>`, std part~~ | done 2026-11-24 (tests only), verified on Aurora (job 9121273) | GCC ≥ 13 |
| B4b-HIP (and CH5 on HIP) | future work, needs a workaround | AMD GPU |
| B4b-SYCL (and CH5 on SYCL) | implemented 2026-11-25, compile-checked here (icpx 2024.0, both passes); to verify: `qsub run_sycl_aurora.sh` (`HALF_IS_CONSTEXPR=1`; `*half*` 14 + 2 skipped) | Aurora |
| Locale branch of `half_io` (a decimal comma through the stream's locale) | **future work** (decided 2026-11-23, not a priority); never run: no machine has `de_DE`/`fr_FR` UTF-8 locales | any machine with those locales, or a user-built one (`localedef`, `LOCPATH`) |
| ~~**GNU dialect** `-std=gnu++23`, Kokkos's default for GCC~~ | done 2026-11-23, job 9117216: same probe values and test results as strict `-std=c++23`, with and without native half (§8.2) | Aurora |
| ~~**Declared but undefined cast functions** (the H18 pattern)~~ | done 2026-11-23: `half_cast_overloads`/`bhalf_cast_overloads` pass on the std backend (job 9117216, four builds) and on CUDA (job 10404595); none missing. HIP/SYCL: run the same tests there (§8.3, §8.4) | Aurora, A100; HIP/SYCL pending |
| ~~aarch64 with native half arithmetic: `Kokkos_ARCH_ARMV9_GRACE` (`-mcpu=neoverse-v2`)~~ | done 2026-11-23, job 4264666: Neoverse V2 has `fphp`/`asimdhp`; the native builds evaluate `_Float16` in half (method 16, `fmul h…`), with the Sapphire Rapids probe results (libstdc++ `(269,111.5)²` = (inf, 60000), Kokkos (59936, 60000)); all four builds 0 warnings, 25 / 15 / 14 tests pass. Needs binutils ≥ 2.41 on `PATH` (the module's 2.35 rejects armv9-a) (§8.2, `aarch64_tests.md`) | Grace machine |
| ~~Numeric traits of `half_t`/`bhalf_t` against `std::numeric_limits<std::float16_t/bfloat16_t>`~~ | done 2026-11-23: `numeric_traits_half_vs_std_numeric_limits` passes in all four builds (job 9117303) and in `default`/`gnu-native` with OpenMP (job 9117309); all traits agree | GCC ≥ 13 |
| ~~Stream I/O of `half_t`/`bhalf_t` against libstdc++'s `std::float16_t`/`std::bfloat16_t` extractor and inserter~~ | done 2026-11-23: `half_io_vs_std`/`bhalf_io_vs_std` pass in all four builds (job 9117303) and with OpenMP (job 9117309); Kokkos matches libstdc++ for every compared input and format. Not compared: sentry failures, where libstdc++ deviates from C++23 (§8.2) | GCC ≥ 13 |
| Report the libstdc++ extraction deviation (sentry failure assigns an uninitialized float; GCC 14, 15, master) to GCC | **future work** (decided 2026-11-23), external | GCC Bugzilla |
| ~~`ENABLE_OPENMP=ON` and `FULL=ON` (complete test executables)~~ | done 2026-11-23: job 9117446 (`default` and `gnu-native`, Serial + OpenMP): every core test executable builds with 0 warnings and passes, including `Kokkos_CoreUnitTest_OpenMP` (333) and `OpenMPInterOp` (§8.2) | Aurora, `capacity` queue |
| GCC 13 (only 14.2 and 15.2 tested) | **future work** (decided 2026-11-23): no machine with GCC 13 is available | a machine with GCC 13 |
| ~~**Optimized builds** (`BUILD_TYPE=Release`, `-O3`) on Aurora~~ | done 2026-11-24, job 9120562: all pass, 0 warnings; the native builds contract half expressions into FMAs (strict and GNU dialect), which changes `std::float16_t`/`half_t` probe values but not Kokkos's complex results (§8.2) | Aurora |
| Optional: optimized builds on Grace (`qsub -l walltime=2:00:00 -v BUILD_TYPE=Release run_std_half_fp16_grace.sh`); expected the same picture (half `fmadd`, seen at `-O2` in `aarch64_tests.md` step 3) | optional | Grace |
| Optional: complex functions vs `std::complex<std::float16_t>` over the `ch3acc` inputs (expected equal up to the last bit) | measurement (proposed) | GCC ≥ 13, both evaluation methods |
| ~~A8 (D11), half part: the deprecated-code test with `half_t`/`bhalf_t` on libstdc++, and the Kokkos 6 state (`DEPRECATED_CODE=OFF`)~~ | done 2026-11-24: Aurora jobs 9121388 (ON: 27 / 15 / 14 pass, 0 warnings) and 9121428 (OFF: 26 / 15 / 14, 0 warnings), four builds each | GCC ≥ 13 (Aurora) |
| Pre-Ampere bf16 double rounding at run time (B4 notes) | known gap, unfixed | none of the available machines: needs an NVIDIA GPU older than sm_80 |

Still open but testable **on this machine** (no other machine needed):
- **A8** (D11): deprecation of the `std::complex` interop except the converting
  constructor; implemented and verified 2026-11-24 (A100 job 10415439; host with
  deprecated code ON and OFF; std half backend: Aurora jobs 9121388, 9121428).
- ~~B1/H16 runtime tests in `Kokkos_CoreUnitTest_{Serial,Cuda}1`~~: done 2026-11-22, job
  10397183, all 14 `*numeric_traits*` tests pass on Serial and Cuda.
- **H17:** fixed on CUDA (see CH1 notes).
- ~~**B4b-CUDA** (and CH5 on CUDA)~~: done 2026-11-24, verified on the A100 (jobs 10415931-33).
- **CH6 (future work, decided 2026-11-24):** a benchmark of code bound by instruction issue, if paired `__half2` should be
  reconsidered.
