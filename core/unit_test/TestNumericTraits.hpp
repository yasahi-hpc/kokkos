// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>

// Suppress "'long double' is treated as 'double' in device code"
// The suppression needs to happen before Kokkos_NumericTraits.hpp is included
// to be effective
#ifdef KOKKOS_COMPILER_NVCC
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20208
#else
#ifdef __CUDA_ARCH__
#pragma diagnostic push
#pragma diag_suppress 3245
#endif
#endif
#endif

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <type_traits>
#include <limits>
#include <cstdint>
#if defined(KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT) || \
    defined(KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT)
#include <stdfloat>
#endif

struct extrema {
#define DEFINE_EXTREMA(T, m, M)                 \
  KOKKOS_FUNCTION static T min(T) { return m; } \
  KOKKOS_FUNCTION static T max(T) { return M; }

  DEFINE_EXTREMA(char, CHAR_MIN, CHAR_MAX)
  DEFINE_EXTREMA(signed char, SCHAR_MIN, SCHAR_MAX)
  DEFINE_EXTREMA(unsigned char, 0, UCHAR_MAX)
  DEFINE_EXTREMA(short, SHRT_MIN, SHRT_MAX)
  DEFINE_EXTREMA(unsigned short, 0, USHRT_MAX)
  DEFINE_EXTREMA(int, INT_MIN, INT_MAX)
  DEFINE_EXTREMA(unsigned, 0U, UINT_MAX)
  DEFINE_EXTREMA(long, LONG_MIN, LONG_MAX)
  DEFINE_EXTREMA(unsigned long, 0UL, ULONG_MAX)
  DEFINE_EXTREMA(long long, LLONG_MIN, LLONG_MAX)
  DEFINE_EXTREMA(unsigned long long, 0ULL, ULLONG_MAX)
#if !KOKKOS_HALF_T_IS_FLOAT
  DEFINE_EXTREMA(Kokkos::Experimental::half_t,
                 Kokkos::Experimental::half_t(-65504.0f),
                 Kokkos::Experimental::half_t(65504.0f))
#endif
#if !KOKKOS_BHALF_T_IS_FLOAT
  DEFINE_EXTREMA(Kokkos::Experimental::bhalf_t,
                 Kokkos::Experimental::bhalf_t(-3.38953139e38f),
                 Kokkos::Experimental::bhalf_t(3.38953139e38f))
#endif
  DEFINE_EXTREMA(float, -FLT_MAX, FLT_MAX)
  DEFINE_EXTREMA(double, -DBL_MAX, DBL_MAX)
  DEFINE_EXTREMA(long double, -LDBL_MAX, LDBL_MAX)

#undef DEFINE_EXTREMA
};

// clang-format off
struct Infinity { template <class T> using trait = Kokkos::infinity<T>; };
struct Epsilon { template <class T> using trait = Kokkos::epsilon<T>; };
struct FiniteMin { template <class T> using trait = Kokkos::finite_min<T>; };
struct FiniteMax { template <class T> using trait = Kokkos::finite_max<T>; };
struct RoundError { template <class T> using trait = Kokkos::round_error<T>; };
struct NormMin { template <class T> using trait = Kokkos::norm_min<T>; };
struct DenormMin { template <class T> using trait = Kokkos::denorm_min<T>; };
struct Digits { template <class T> using trait = Kokkos::digits<T>; };
struct Digits10 { template <class T> using trait = Kokkos::digits10<T>; };
struct MaxDigits10 { template <class T> using trait = Kokkos::max_digits10<T>; };
struct Radix { template <class T> using trait = Kokkos::radix<T>; };
struct MinExponent { template <class T> using trait = Kokkos::min_exponent<T>; };
struct MaxExponent { template <class T> using trait = Kokkos::max_exponent<T>; };
struct MinExponent10 { template <class T> using trait = Kokkos::min_exponent10<T>; };
struct MaxExponent10 { template <class T> using trait = Kokkos::max_exponent10<T>; };
struct QuietNaN { template <class T> using trait = Kokkos::quiet_NaN<T>; };
struct SignalingNaN { template <class T> using trait = Kokkos::signaling_NaN<T>; };
// clang-format on

template <class T>
KOKKOS_FUNCTION T* take_address_of(T& arg) {
  return &arg;
}

template <class T>
KOKKOS_FUNCTION void take_by_value(T) {}

template <class Space, class T, class Tag>
struct TestNumericTraits {
  template <class U>
  using trait = typename Tag::template trait<U>;

  Kokkos::View<T, Space> compare;
  TestNumericTraits() {
    compare = Kokkos::View<T, Space>("C");
    run();
  }

  void run() const {
    int errors = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Space, Tag>(0, 1), *this,
                            errors);
    ASSERT_EQ(errors, 0);
    (void)take_address_of(trait<T>::value);  // use on host
  }

  KOKKOS_FUNCTION void operator()(Infinity, int, int& e) const {
    using Kokkos::infinity;
    T const inf(infinity<T>::value);
    T const zero = 0;
    e += (int)!(inf + inf == inf);
    e += (int)!(inf != zero);
    use_on_device();
  }

  KOKKOS_FUNCTION void operator()(Epsilon, int, int& e) const {
    using Kokkos::epsilon;
    T const eps(epsilon<T>::value);
    T const one = 1;
    // Avoid higher precision intermediate representation
    compare() = one + eps;
    e += (int)!(compare() != one);
    compare() = one + eps / 2;
    e += (int)!(compare() == one);
    use_on_device();
  }

  KOKKOS_FUNCTION void operator()(FiniteMin, int, int& e) const {
    using Kokkos::finite_max;
    using Kokkos::finite_min;
    auto const min = finite_min<T>::value;
    auto const max = finite_max<T>::value;
    e += (int)!(min == extrema::min(T{}));
    e += (int)!(max == extrema::max(T{}));
    use_on_device();
  }

  // clang-format off
  KOKKOS_FUNCTION void operator()(FiniteMax, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(RoundError, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(NormMin, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(DenormMin, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(Digits, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(Digits10, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(MaxDigits10, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(Radix, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(MinExponent, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(MaxExponent, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(MinExponent10, int, int&) const { use_on_device(); }
  KOKKOS_FUNCTION void operator()(MaxExponent10, int, int&) const { use_on_device(); }
  // clang-format on
  KOKKOS_FUNCTION void operator()(QuietNaN, int, int& e) const {
    using Kokkos::quiet_NaN;
    constexpr auto nan = quiet_NaN<T>::value;
    auto const zero    = T(0);
    e += (int)!(nan != nan);
    e += (int)!(nan != zero);
    use_on_device();
  }
  KOKKOS_FUNCTION void operator()(SignalingNaN, int, int& e) const {
    using Kokkos::signaling_NaN;
    constexpr auto nan = signaling_NaN<T>::value;
    auto const zero    = T(0);
    e += (int)!(nan != nan);
    e += (int)!(nan != zero);
    use_on_device();
  }

  KOKKOS_FUNCTION void use_on_device() const {
#if defined(KOKKOS_COMPILER_NVCC) || defined(KOKKOS_ENABLE_OPENACC)
    take_by_value(trait<T>::value);
#else
    (void)take_address_of(trait<T>::value);
#endif
  }
};

#if (defined(KOKKOS_COMPILER_NVCC) && defined(KOKKOS_ENABLE_CUDA)) || \
    defined(KOKKOS_ENABLE_SYCL)
template <class Tag>
struct TestNumericTraits<
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::Cuda,
#elif defined(KOKKOS_ENABLE_SYCL)
    Kokkos::SYCL,
#else
#error "logic error"
#endif
    long double, Tag> {
  template <class T>
  using trait = typename Tag::template trait<T>;
  TestNumericTraits() {
    (void)take_address_of(trait<long double>::value);
    // Do nothing on the device.
    // According to the doc
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-variables
    // the traits member constant value cannot be directly used in device code.
  }
};
#endif

// NOLINTBEGIN(bugprone-unused-raii)
TEST(TEST_CATEGORY, numeric_traits_infinity) {
#if __FINITE_MATH_ONLY__
  GTEST_SKIP() << "skipping when compiling with -ffinite-math-only";
#endif
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, Infinity>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, Infinity>();
  TestNumericTraits<TEST_EXECSPACE, float, Infinity>();
  TestNumericTraits<TEST_EXECSPACE, double, Infinity>();
  TestNumericTraits<TEST_EXECSPACE, long double, Infinity>();
}

TEST(TEST_CATEGORY, numeric_traits_epsilon) {
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, Epsilon>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, Epsilon>();
  TestNumericTraits<TEST_EXECSPACE, float, Epsilon>();
  TestNumericTraits<TEST_EXECSPACE, double, Epsilon>();
  TestNumericTraits<TEST_EXECSPACE, long double, Epsilon>();
}

TEST(TEST_CATEGORY, numeric_traits_round_error) {
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, RoundError>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t,
                    RoundError>();
  TestNumericTraits<TEST_EXECSPACE, float, RoundError>();
  TestNumericTraits<TEST_EXECSPACE, double, RoundError>();
  TestNumericTraits<TEST_EXECSPACE, long double, RoundError>();
}

TEST(TEST_CATEGORY, numeric_traits_norm_min) {
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, NormMin>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, NormMin>();
  TestNumericTraits<TEST_EXECSPACE, float, NormMin>();
  TestNumericTraits<TEST_EXECSPACE, double, NormMin>();
  TestNumericTraits<TEST_EXECSPACE, long double, NormMin>();
}

TEST(TEST_CATEGORY, numeric_traits_denorm_min) {
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, DenormMin>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, DenormMin>();
  TestNumericTraits<TEST_EXECSPACE, float, DenormMin>();
  TestNumericTraits<TEST_EXECSPACE, double, DenormMin>();
  TestNumericTraits<TEST_EXECSPACE, long double, DenormMin>();
}

TEST(TEST_CATEGORY, numeric_traits_finite_min_max) {
  TestNumericTraits<TEST_EXECSPACE, char, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, char, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, signed char, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, signed char, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned char, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, unsigned char, FiniteMax>();

  TestNumericTraits<TEST_EXECSPACE, short, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, short, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned short, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, unsigned short, FiniteMax>();

  TestNumericTraits<TEST_EXECSPACE, int, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, int, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned int, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, unsigned int, FiniteMax>();

  TestNumericTraits<TEST_EXECSPACE, long, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, long, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long, FiniteMax>();

  TestNumericTraits<TEST_EXECSPACE, long long, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, long long, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long long, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long long, FiniteMax>();

  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, float, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, float, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, double, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, double, FiniteMax>();
  TestNumericTraits<TEST_EXECSPACE, long double, FiniteMin>();
  TestNumericTraits<TEST_EXECSPACE, long double, FiniteMax>();
}

TEST(TEST_CATEGORY, numeric_traits_digits) {
  TestNumericTraits<TEST_EXECSPACE, bool, Digits>();
  TestNumericTraits<TEST_EXECSPACE, char, Digits>();
  TestNumericTraits<TEST_EXECSPACE, signed char, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned char, Digits>();
  TestNumericTraits<TEST_EXECSPACE, short, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned short, Digits>();
  TestNumericTraits<TEST_EXECSPACE, int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, long int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, long long int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long long int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, Digits>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, Digits>();
  TestNumericTraits<TEST_EXECSPACE, float, Digits>();
  TestNumericTraits<TEST_EXECSPACE, double, Digits>();
  TestNumericTraits<TEST_EXECSPACE, long double, Digits>();
}

TEST(TEST_CATEGORY, numeric_traits_digits10) {
  TestNumericTraits<TEST_EXECSPACE, bool, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, char, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, signed char, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned char, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, short, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned short, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, long int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, long long int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long long int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, float, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, double, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, long double, Digits10>();
}

TEST(TEST_CATEGORY, numeric_traits_max_digits10) {
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t,
                    MaxDigits10>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t,
                    MaxDigits10>();
  TestNumericTraits<TEST_EXECSPACE, float, MaxDigits10>();
  TestNumericTraits<TEST_EXECSPACE, double, MaxDigits10>();
  TestNumericTraits<TEST_EXECSPACE, long double, MaxDigits10>();
}
TEST(TEST_CATEGORY, numeric_traits_radix) {
  TestNumericTraits<TEST_EXECSPACE, bool, Radix>();
  TestNumericTraits<TEST_EXECSPACE, char, Radix>();
  TestNumericTraits<TEST_EXECSPACE, signed char, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned char, Radix>();
  TestNumericTraits<TEST_EXECSPACE, short, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned short, Radix>();
  TestNumericTraits<TEST_EXECSPACE, int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, long int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, long long int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long long int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, Radix>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, Radix>();
  TestNumericTraits<TEST_EXECSPACE, float, Radix>();
  TestNumericTraits<TEST_EXECSPACE, double, Radix>();
  TestNumericTraits<TEST_EXECSPACE, long double, Radix>();
}

TEST(TEST_CATEGORY, numeric_traits_min_max_exponent) {
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t,
                    MinExponent>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t,
                    MaxExponent>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t,
                    MinExponent>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t,
                    MaxExponent>();
  TestNumericTraits<TEST_EXECSPACE, float, MinExponent>();
  TestNumericTraits<TEST_EXECSPACE, float, MaxExponent>();
  TestNumericTraits<TEST_EXECSPACE, double, MinExponent>();
  TestNumericTraits<TEST_EXECSPACE, double, MaxExponent>();
  TestNumericTraits<TEST_EXECSPACE, long double, MinExponent>();
  TestNumericTraits<TEST_EXECSPACE, long double, MaxExponent>();
}

TEST(TEST_CATEGORY, numeric_traits_min_max_exponent10) {
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t,
                    MinExponent10>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t,
                    MaxExponent10>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t,
                    MinExponent10>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t,
                    MaxExponent10>();
  TestNumericTraits<TEST_EXECSPACE, float, MinExponent10>();
  TestNumericTraits<TEST_EXECSPACE, float, MaxExponent10>();
  TestNumericTraits<TEST_EXECSPACE, double, MinExponent10>();
  TestNumericTraits<TEST_EXECSPACE, double, MaxExponent10>();
  TestNumericTraits<TEST_EXECSPACE, long double, MinExponent10>();
  TestNumericTraits<TEST_EXECSPACE, long double, MaxExponent10>();
}

KOKKOS_IMPL_DISABLE_UNREACHABLE_WARNINGS_PUSH()
TEST(TEST_CATEGORY, numeric_traits_quiet_and_signaling_nan) {
#ifdef KOKKOS_COMPILER_NVHPC  // FIXME_NVHPC
  GTEST_SKIP() << "This test is known to fail with the NVHPC compiler";
#endif

  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t, QuietNaN>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::half_t,
                    SignalingNaN>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t, QuietNaN>();
  TestNumericTraits<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t,
                    SignalingNaN>();
  TestNumericTraits<TEST_EXECSPACE, float, QuietNaN>();
  TestNumericTraits<TEST_EXECSPACE, float, SignalingNaN>();
  TestNumericTraits<TEST_EXECSPACE, double, QuietNaN>();
  TestNumericTraits<TEST_EXECSPACE, double, SignalingNaN>();
  TestNumericTraits<TEST_EXECSPACE, long double, QuietNaN>();
  TestNumericTraits<TEST_EXECSPACE, long double, SignalingNaN>();
}
// NOLINTEND(bugprone-unused-raii)
KOKKOS_IMPL_DISABLE_UNREACHABLE_WARNINGS_POP()

namespace NumericTraitsSFINAE {

struct HasNoSpecialization {};

#define CHECK_TRAIT_IS_SFINAE_FRIENDLY(TRAIT)                  \
  template <class T>                                           \
  using TRAIT##_value_t = decltype(Kokkos::TRAIT<T>::value);   \
  template <class T>                                           \
  using has_##TRAIT = Kokkos::is_detected<TRAIT##_value_t, T>; \
  static_assert(!has_##TRAIT<HasNoSpecialization>::value);

CHECK_TRAIT_IS_SFINAE_FRIENDLY(infinity)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(finite_min)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(finite_max)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(epsilon)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(round_error)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(norm_min)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(denorm_min)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(quiet_NaN)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(signaling_NaN)

CHECK_TRAIT_IS_SFINAE_FRIENDLY(digits)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(digits10)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(max_digits10)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(radix)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(min_exponent)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(min_exponent10)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(max_exponent)
CHECK_TRAIT_IS_SFINAE_FRIENDLY(max_exponent10)

}  // namespace NumericTraitsSFINAE

// Example detecting presence or absence of values
template <class T>
using infinity_value_t = decltype(Kokkos::infinity<T>::value);

template <class T>
using has_infinity = Kokkos::is_detected<infinity_value_t, T>;

template <class T, std::enable_if_t<has_infinity<T>::value>* = nullptr>
constexpr T legacy_std_numeric_limits_infinity() {
  return Kokkos::infinity<T>::value;
}

template <class T, std::enable_if_t<!has_infinity<T>::value>* = nullptr>
constexpr T legacy_std_numeric_limits_infinity() {
  return T();
}

TEST(TEST_CATEGORY, numeric_traits_sfinae_friendly) {
  ASSERT_EQ(legacy_std_numeric_limits_infinity<int>(), 0);
}

// On the std backend half_t and bhalf_t wrap std::float16_t and
// std::bfloat16_t, so std::numeric_limits of those types is an independent
// reference for every half trait: the values bit for bit, the NaNs by kind
// (the payload is unspecified), and the integer traits.
#if defined(KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT) || \
    defined(KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT)
template <class H, class F>
void test_half_traits_vs_numeric_limits() {
  using L = std::numeric_limits<F>;
  static_assert(sizeof(H) == 2 && sizeof(F) == 2);
  auto bits = [](auto x) { return Kokkos::bit_cast<std::uint16_t>(x); };
  // Direct initialization: the traits are half values on the std backend
  // (B4c), bit patterns (BitComparisonWrapper) elsewhere.
  EXPECT_EQ(bits(H(Kokkos::infinity<H>::value)), bits(L::infinity()));
  EXPECT_EQ(bits(H(Kokkos::epsilon<H>::value)), bits(L::epsilon()));
  EXPECT_EQ(bits(H(Kokkos::round_error<H>::value)), bits(L::round_error()));
  EXPECT_EQ(bits(H(Kokkos::finite_min<H>::value)), bits(L::lowest()));
  EXPECT_EQ(bits(H(Kokkos::finite_max<H>::value)), bits(L::max()));
  EXPECT_EQ(bits(H(Kokkos::norm_min<H>::value)), bits(L::min()));
  EXPECT_EQ(bits(H(Kokkos::denorm_min<H>::value)), bits(L::denorm_min()));

  // NaNs: all exponent bits set, a nonzero fraction, and the quiet bit (the
  // most significant fraction bit) set for quiet_NaN, clear for
  // signaling_NaN.
  const std::uint16_t fraction = (std::uint16_t(1) << (L::digits - 1)) - 1;
  const std::uint16_t exponent = 0x7fff & ~fraction;
  const std::uint16_t quiet    = std::uint16_t(1) << (L::digits - 2);
  auto kind                    = [&](std::uint16_t b) {
    if ((b & exponent) != exponent || (b & fraction) == 0) return 0;  // no NaN
    return (b & quiet) ? 1 : 2;  // 1: quiet, 2: signaling
  };
  const std::uint16_t kq = bits(H(Kokkos::quiet_NaN<H>::value));
  const std::uint16_t sq = bits(L::quiet_NaN());
  const std::uint16_t ks = bits(H(Kokkos::signaling_NaN<H>::value));
  const std::uint16_t ss = bits(L::signaling_NaN());
  EXPECT_EQ(kind(kq), 1) << std::hex << kq;
  EXPECT_EQ(kind(sq), 1) << std::hex << sq;
  EXPECT_EQ(kind(ks), 2) << std::hex << ks;
  EXPECT_EQ(kind(ss), 2) << std::hex << ss;

  EXPECT_EQ(Kokkos::digits<H>::value, L::digits);
  EXPECT_EQ(Kokkos::digits10<H>::value, L::digits10);
  EXPECT_EQ(Kokkos::max_digits10<H>::value, L::max_digits10);
  EXPECT_EQ(Kokkos::radix<H>::value, L::radix);
  EXPECT_EQ(Kokkos::min_exponent<H>::value, L::min_exponent);
  EXPECT_EQ(Kokkos::max_exponent<H>::value, L::max_exponent);
  EXPECT_EQ(Kokkos::min_exponent10<H>::value, L::min_exponent10);
  EXPECT_EQ(Kokkos::max_exponent10<H>::value, L::max_exponent10);
}
#endif

// B4c: where half values are constexpr (std backend, CUDA, SYCL), the
// distinguished values of half_t and bhalf_t are constexpr half values, like
// those of float and double (H11); on HIP they stay bit patterns.
#ifdef KOKKOS_IMPL_HALF_IS_CONSTEXPR
template <class H>
constexpr bool half_value_traits_are_constexpr_halves() {
  using namespace Kokkos::Experimental;
  constexpr bool bf16 = Kokkos::Experimental::Impl::is_bfloat16<H>::value;
  auto is_h           = [](auto x) {
    return std::is_same_v<std::remove_cv_t<decltype(x)>, H>;
  };
  static_assert(is_h(infinity_v<H>) && is_h(finite_min_v<H>) &&
                is_h(finite_max_v<H>) && is_h(epsilon_v<H>) &&
                is_h(round_error_v<H>) && is_h(norm_min_v<H>) &&
                is_h(denorm_min_v<H>) && is_h(quiet_NaN_v<H>) &&
                is_h(signaling_NaN_v<H>));
  static_assert(static_cast<double>(epsilon_v<H>) == (bf16 ? 0x1p-7 : 0x1p-10));
  static_assert(static_cast<double>(round_error_v<H>) == 0.5);
  static_assert(static_cast<double>(finite_max_v<H>) ==
                (bf16 ? 0x1.fep127 : 65504.0));
  static_assert(finite_min_v<H> == -finite_max_v<H>);
  static_assert(static_cast<double>(norm_min_v<H>) ==
                (bf16 ? 0x1p-126 : 0x1p-14));
  static_assert(static_cast<double>(denorm_min_v<H>) ==
                (bf16 ? 0x1p-133 : 0x1p-24));
  static_assert(infinity_v<H> > finite_max_v<H>);
  static_assert(-infinity_v<H> < finite_min_v<H>);
  static_assert(quiet_NaN_v<H> != quiet_NaN_v<H>);
  static_assert(signaling_NaN_v<H> != signaling_NaN_v<H>);
  // Arithmetic directly on the constants (H11).
  static_assert(finite_max_v<H> / H(2) < finite_max_v<H>);
  static_assert(H(1) + epsilon_v<H> > H(1));
  return true;
}
#if !KOKKOS_HALF_T_IS_FLOAT
static_assert(
    half_value_traits_are_constexpr_halves<Kokkos::Experimental::half_t>());
#endif
#if !KOKKOS_BHALF_T_IS_FLOAT
static_assert(
    half_value_traits_are_constexpr_halves<Kokkos::Experimental::bhalf_t>());
#endif
#endif

TEST(TEST_CATEGORY, numeric_traits_half_vs_std_numeric_limits) {
#if !defined(KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT) && \
    !defined(KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT)
  GTEST_SKIP() << "needs the std half backend (std::float16_t, C++23)";
#else
#ifdef KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT
  test_half_traits_vs_numeric_limits<Kokkos::Experimental::half_t,
                                     std::float16_t>();
#endif
#ifdef KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT
  test_half_traits_vs_numeric_limits<Kokkos::Experimental::bhalf_t,
                                     std::bfloat16_t>();
#endif
#endif
}

// Compare to std::numeric_limits
template <int V1, int V2>
struct AssertIntEquality {
  static constexpr bool value = false;
};
template <int V>
struct AssertIntEquality<V, V> {
  static constexpr bool value = true;
};
#define CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(T, TRAIT)           \
  static_assert(AssertIntEquality<Kokkos::TRAIT<T>::value,               \
                                  std::numeric_limits<T>::TRAIT>::value, \
                "")
#define CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(T, TRAIT) \
  static_assert(Kokkos::TRAIT<T>::value == std::numeric_limits<T>::TRAIT(), "")

CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(float, infinity);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(double, infinity);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(long double, infinity);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(float, epsilon);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(double, epsilon);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(long double, epsilon);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(float, round_error);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(double, round_error);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(long double, round_error);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(float, denorm_min);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(double, denorm_min);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(long double, denorm_min);

// clang-format off
static_assert(Kokkos::norm_min<float      >::value == std::numeric_limits<      float>::min());
static_assert(Kokkos::norm_min<double     >::value == std::numeric_limits<     double>::min());
static_assert(Kokkos::norm_min<long double>::value == std::numeric_limits<long double>::min());
// integer types
static_assert(Kokkos::finite_min<char                  >::value == std::numeric_limits<                  char>::min());
static_assert(Kokkos::finite_min<signed char           >::value == std::numeric_limits<           signed char>::min());
static_assert(Kokkos::finite_min<unsigned char         >::value == std::numeric_limits<         unsigned char>::min());
static_assert(Kokkos::finite_min<short                 >::value == std::numeric_limits<                 short>::min());
static_assert(Kokkos::finite_min<unsigned short        >::value == std::numeric_limits<        unsigned short>::min());
static_assert(Kokkos::finite_min<int                   >::value == std::numeric_limits<                   int>::min());
static_assert(Kokkos::finite_min<unsigned int          >::value == std::numeric_limits<          unsigned int>::min());
static_assert(Kokkos::finite_min<long int              >::value == std::numeric_limits<              long int>::min());
static_assert(Kokkos::finite_min<unsigned long int     >::value == std::numeric_limits<     unsigned long int>::min());
static_assert(Kokkos::finite_min<long long int         >::value == std::numeric_limits<         long long int>::min());
static_assert(Kokkos::finite_min<unsigned long long int>::value == std::numeric_limits<unsigned long long int>::min());
static_assert(Kokkos::finite_max<char                  >::value == std::numeric_limits<                  char>::max());
static_assert(Kokkos::finite_max<signed char           >::value == std::numeric_limits<           signed char>::max());
static_assert(Kokkos::finite_max<unsigned char         >::value == std::numeric_limits<         unsigned char>::max());
static_assert(Kokkos::finite_max<short                 >::value == std::numeric_limits<                 short>::max());
static_assert(Kokkos::finite_max<unsigned short        >::value == std::numeric_limits<        unsigned short>::max());
static_assert(Kokkos::finite_max<int                   >::value == std::numeric_limits<                   int>::max());
static_assert(Kokkos::finite_max<unsigned int          >::value == std::numeric_limits<          unsigned int>::max());
static_assert(Kokkos::finite_max<long int              >::value == std::numeric_limits<              long int>::max());
static_assert(Kokkos::finite_max<unsigned long int     >::value == std::numeric_limits<     unsigned long int>::max());
static_assert(Kokkos::finite_max<long long int         >::value == std::numeric_limits<         long long int>::max());
static_assert(Kokkos::finite_max<unsigned long long int>::value == std::numeric_limits<unsigned long long int>::max());
// floating point types
static_assert(Kokkos::finite_min<float      >::value == -std::numeric_limits<      float>::max());
static_assert(Kokkos::finite_min<double     >::value == -std::numeric_limits<     double>::max());
static_assert(Kokkos::finite_min<long double>::value == -std::numeric_limits<long double>::max());
static_assert(Kokkos::finite_max<float      >::value ==  std::numeric_limits<      float>::max());
static_assert(Kokkos::finite_max<double     >::value ==  std::numeric_limits<     double>::max());
static_assert(Kokkos::finite_max<long double>::value ==  std::numeric_limits<long double>::max());
// clang-format on

CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(bool, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(char, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(signed char, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned char, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(short, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned short, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(int, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned int, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long int, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned long int, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long long int, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned long long int, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(float, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(double, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long double, digits);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(bool, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(char, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(signed char, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned char, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(short, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned short, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(int, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned int, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long int, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned long int, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long long int, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned long long int, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(float, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(double, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long double, digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(float, max_digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(double, max_digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long double, max_digits10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(bool, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(char, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(signed char, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned char, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(short, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned short, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(int, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned int, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long int, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned long int, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long long int, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(unsigned long long int, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(float, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(double, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long double, radix);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(float, min_exponent);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(float, max_exponent);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(double, min_exponent);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(double, max_exponent);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long double, min_exponent);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long double, max_exponent);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(float, min_exponent10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(float, max_exponent10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(double, min_exponent10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(double, max_exponent10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long double, min_exponent10);
CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT(long double, max_exponent10);

#undef CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION
#undef CHECK_SAME_AS_NUMERIC_LIMITS_MEMBER_CONSTANT

#define CHECK_NAN_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(T, TRAIT)   \
  static_assert(Kokkos::TRAIT<T>::value != Kokkos::TRAIT<T>::value); \
  static_assert(std::numeric_limits<T>::TRAIT() !=                   \
                std::numeric_limits<T>::TRAIT());                    \
  static_assert(Kokkos::TRAIT<T>::value != std::numeric_limits<T>::TRAIT())

// Workaround compiler issue error: expression must have a constant value
// See kokkos/kokkos#4574
// FIXME_NVHPC FIXME_NVCC
#if !defined(KOKKOS_COMPILER_NVHPC) && \
    !(defined(KOKKOS_COMPILER_NVCC) && !defined(KOKKOS_ENABLE_CUDA))
CHECK_NAN_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(float, quiet_NaN);
CHECK_NAN_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(double, quiet_NaN);
CHECK_NAN_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(long double, quiet_NaN);
CHECK_NAN_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(float, signaling_NaN);
CHECK_NAN_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(double, signaling_NaN);
CHECK_NAN_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION(long double, signaling_NaN);
#endif

#undef CHECK_NAN_SAME_AS_NUMERIC_LIMITS_MEMBER_FUNCTION

#define CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(T, TRAIT)                    \
  static_assert(Kokkos::TRAIT<T const>::value == Kokkos::TRAIT<T>::value);    \
  static_assert(Kokkos::TRAIT<T volatile>::value == Kokkos::TRAIT<T>::value); \
  static_assert(Kokkos::TRAIT<T const volatile>::value ==                     \
                Kokkos::TRAIT<T>::value)

#define CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(TRAIT) \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(float, TRAIT);              \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(double, TRAIT);             \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(long double, TRAIT)

#define CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_INTEGRAL(TRAIT)      \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(bool, TRAIT);              \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(char, TRAIT);              \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(signed char, TRAIT);       \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(unsigned char, TRAIT);     \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(short, TRAIT);             \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(unsigned short, TRAIT);    \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(int, TRAIT);               \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(unsigned int, TRAIT);      \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(long int, TRAIT);          \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(unsigned long int, TRAIT); \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(long long int, TRAIT);     \
  CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES(unsigned long long int, TRAIT)

CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(infinity);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(finite_min);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_INTEGRAL(finite_min);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(finite_max);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_INTEGRAL(finite_max);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(epsilon);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(round_error);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(norm_min);

CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(digits);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_INTEGRAL(digits);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(digits10);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_INTEGRAL(digits10);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(max_digits10);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(radix);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_INTEGRAL(radix);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(min_exponent);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(min_exponent10);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(max_exponent);
CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(max_exponent10);

#undef CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_INTEGRAL
#undef CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT
#undef CHECK_INSTANTIATED_ON_CV_QUALIFIED_TYPES

#define CHECK_NAN_INSTANTIATED_ON_CV_QUALIFIED_TYPES(T, TRAIT)                \
  static_assert(Kokkos::TRAIT<T>::value != Kokkos::TRAIT<T>::value);          \
  static_assert(Kokkos::TRAIT<T const>::value != Kokkos::TRAIT<T>::value);    \
  static_assert(Kokkos::TRAIT<T volatile>::value != Kokkos::TRAIT<T>::value); \
  static_assert(Kokkos::TRAIT<T const volatile>::value !=                     \
                Kokkos::TRAIT<T>::value)

#define CHECK_NAN_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(TRAIT) \
  CHECK_NAN_INSTANTIATED_ON_CV_QUALIFIED_TYPES(float, TRAIT);              \
  CHECK_NAN_INSTANTIATED_ON_CV_QUALIFIED_TYPES(double, TRAIT);             \
  CHECK_NAN_INSTANTIATED_ON_CV_QUALIFIED_TYPES(long double, TRAIT)

CHECK_NAN_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(quiet_NaN);
CHECK_NAN_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT(signaling_NaN);

#undef CHECK_NAN_INSTANTIATED_ON_CV_QUALIFIED_TYPES_FLOATING_POINT
#undef CHECK_NAN_INSTANTIATED_ON_CV_QUALIFIED_TYPES

#define CHECK_BACKWARD_COMPATIBILITY(T, TRAIT)          \
  static_assert(Kokkos::TRAIT<T>::value ==              \
                Kokkos::Experimental::TRAIT<T>::value); \
  static_assert(Kokkos::TRAIT##_v<T> == Kokkos::Experimental::TRAIT##_v<T>)

#define CHECK_BACKWARD_COMPATIBILITY_INTEGRAL(TRAIT)      \
  CHECK_BACKWARD_COMPATIBILITY(bool, TRAIT);              \
  CHECK_BACKWARD_COMPATIBILITY(char, TRAIT);              \
  CHECK_BACKWARD_COMPATIBILITY(signed char, TRAIT);       \
  CHECK_BACKWARD_COMPATIBILITY(unsigned char, TRAIT);     \
  CHECK_BACKWARD_COMPATIBILITY(short, TRAIT);             \
  CHECK_BACKWARD_COMPATIBILITY(unsigned short, TRAIT);    \
  CHECK_BACKWARD_COMPATIBILITY(int, TRAIT);               \
  CHECK_BACKWARD_COMPATIBILITY(unsigned int, TRAIT);      \
  CHECK_BACKWARD_COMPATIBILITY(long int, TRAIT);          \
  CHECK_BACKWARD_COMPATIBILITY(unsigned long int, TRAIT); \
  CHECK_BACKWARD_COMPATIBILITY(long long int, TRAIT);     \
  CHECK_BACKWARD_COMPATIBILITY(unsigned long long int, TRAIT)

#define CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(TRAIT) \
  CHECK_BACKWARD_COMPATIBILITY(float, TRAIT);              \
  CHECK_BACKWARD_COMPATIBILITY(double, TRAIT);             \
  CHECK_BACKWARD_COMPATIBILITY(long double, TRAIT)

#define CHECK_BACKWARD_COMPATIBILITY_ARITHMETIC(TRAIT) \
  CHECK_BACKWARD_COMPATIBILITY_INTEGRAL(TRAIT);        \
  CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(TRAIT)

CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(infinity);
CHECK_BACKWARD_COMPATIBILITY_ARITHMETIC(finite_min);
CHECK_BACKWARD_COMPATIBILITY_ARITHMETIC(finite_max);
CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(epsilon);
CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(round_error);
CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(norm_min);
CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(denorm_min);
// NOTE: These do compare equal.  Keeping them as is for bookkeeping and not
// bothering with more checks.
// CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(quiet_NaN);
// CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(signaling_NaN);

CHECK_BACKWARD_COMPATIBILITY_ARITHMETIC(digits);
CHECK_BACKWARD_COMPATIBILITY_ARITHMETIC(digits10);
CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(max_digits10);
CHECK_BACKWARD_COMPATIBILITY_ARITHMETIC(radix);
CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(min_exponent);
CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(min_exponent10);
CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(max_exponent);
CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT(max_exponent10);

#undef CHECK_BACKWARD_COMPATIBILITY_ARITHMETIC
#undef CHECK_BACKWARD_COMPATIBILITY_FLOATING_POINT
#undef CHECK_BACKWARD_COMPATIBILITY_INTEGRAL
#undef CHECK_BACKWARD_COMPATIBILITY

// Values of the integer traits of the half types, as for
// std::numeric_limits<std::float16_t> and std::numeric_limits<std::bfloat16_t>
// (IEEE binary16: 10 + 1 mantissa bits, 5 exponent bits; bfloat16: 7 + 1
// mantissa bits, 8 exponent bits).  The tests above only check that the traits
// can be used, not their values: digits_v<bhalf_t> was 2 and digits10_v 1.
#if !KOKKOS_HALF_T_IS_FLOAT
static_assert(Kokkos::Experimental::digits_v<Kokkos::Experimental::half_t> ==
              11);
static_assert(Kokkos::Experimental::digits10_v<Kokkos::Experimental::half_t> ==
              3);
static_assert(Kokkos::Experimental::radix_v<Kokkos::Experimental::half_t> == 2);
static_assert(
    Kokkos::Experimental::min_exponent_v<Kokkos::Experimental::half_t> == -13);
static_assert(
    Kokkos::Experimental::max_exponent_v<Kokkos::Experimental::half_t> == 16);
static_assert(
    Kokkos::Experimental::max_digits10_v<Kokkos::Experimental::half_t> == 5);
static_assert(
    Kokkos::Experimental::min_exponent10_v<Kokkos::Experimental::half_t> == -4);
static_assert(
    Kokkos::Experimental::max_exponent10_v<Kokkos::Experimental::half_t> == 4);
#endif
#if !KOKKOS_BHALF_T_IS_FLOAT
static_assert(Kokkos::Experimental::digits_v<Kokkos::Experimental::bhalf_t> ==
              8);
static_assert(Kokkos::Experimental::digits10_v<Kokkos::Experimental::bhalf_t> ==
              2);
static_assert(Kokkos::Experimental::radix_v<Kokkos::Experimental::bhalf_t> ==
              2);
static_assert(
    Kokkos::Experimental::min_exponent_v<Kokkos::Experimental::bhalf_t> ==
    -125);
static_assert(
    Kokkos::Experimental::max_exponent_v<Kokkos::Experimental::bhalf_t> == 128);
static_assert(
    Kokkos::Experimental::max_digits10_v<Kokkos::Experimental::bhalf_t> == 4);
static_assert(
    Kokkos::Experimental::min_exponent10_v<Kokkos::Experimental::bhalf_t> ==
    -37);
static_assert(
    Kokkos::Experimental::max_exponent10_v<Kokkos::Experimental::bhalf_t> ==
    38);
#endif

#ifdef KOKKOS_COMPILER_NVCC
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diagnostic pop
#else
#ifdef __CUDA_ARCH__
#pragma diagnostic pop
#endif
#endif
#endif
