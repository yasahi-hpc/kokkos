// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef TESTHALFCONVERSION_HPP_
#define TESTHALFCONVERSION_HPP_

namespace Test {

template <class T>
void test_half_conversion_type() {
  // When truncating mantissa to 10bits (like f16), 3.3 becomes 3.298828125
  // 3.3 - 3.298828125 < 1.1719e-3, so conversion error should be smaller
  double epsilon                 = KOKKOS_HALF_T_IS_FLOAT ? 3e-7 : 1.1719e-3;
  T base                         = static_cast<T>(3.3);
  Kokkos::Experimental::half_t a = Kokkos::Experimental::cast_to_half(base);
  T b                            = Kokkos::Experimental::cast_from_half<T>(a);
  ASSERT_NEAR(b, base, epsilon);

  using MemorySpace = typename TEST_EXECSPACE::memory_space;
  Kokkos::View<T, MemorySpace> b_v("b_v");
  Kokkos::parallel_for(
      "TestHalfConversion", Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
      KOKKOS_LAMBDA(int) {
        Kokkos::Experimental::half_t d_a =
            Kokkos::Experimental::cast_to_half(base);
        b_v() = Kokkos::Experimental::cast_from_half<T>(d_a);
      });

  Kokkos::deep_copy(b, b_v);
  ASSERT_NEAR(b, base, epsilon);
}

template <class T>
void test_bhalf_conversion_type() {
  // When truncating mantissa to 7bits (like b16), 3.3 becomes 3.296875
  // 3.3 - 3.296875 < 3.125e-3, so conversion error should be smaller
  double epsilon                  = KOKKOS_BHALF_T_IS_FLOAT ? 3e-7 : 3.125e-3;
  T base                          = static_cast<T>(3.3);
  Kokkos::Experimental::bhalf_t a = Kokkos::Experimental::cast_to_bhalf(base);
  T b                             = Kokkos::Experimental::cast_from_bhalf<T>(a);
  ASSERT_NEAR(b, base, epsilon);

  using MemorySpace = typename TEST_EXECSPACE::memory_space;
  Kokkos::View<T, MemorySpace> b_v("b_v");
  Kokkos::parallel_for(
      "TestHalfConversion", Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
      KOKKOS_LAMBDA(int) {
        Kokkos::Experimental::bhalf_t d_a =
            Kokkos::Experimental::cast_to_bhalf(base);
        b_v() = Kokkos::Experimental::cast_from_bhalf<T>(d_a);
      });

  Kokkos::deep_copy(b, b_v);
  ASSERT_NEAR(b, base, epsilon);
}

void test_half_conversion() {
  test_half_conversion_type<float>();
  test_half_conversion_type<double>();
  test_half_conversion_type<short>();
  test_half_conversion_type<int>();
  test_half_conversion_type<long>();
  test_half_conversion_type<long long>();
  test_half_conversion_type<unsigned short>();
  test_half_conversion_type<unsigned int>();
  test_half_conversion_type<unsigned long>();
  test_half_conversion_type<unsigned long long>();
}

void test_bhalf_conversion() {
  test_bhalf_conversion_type<float>();
  test_bhalf_conversion_type<double>();
  test_bhalf_conversion_type<short>();
  test_bhalf_conversion_type<int>();
  test_bhalf_conversion_type<long>();
  test_bhalf_conversion_type<long long>();
  test_bhalf_conversion_type<unsigned short>();
  test_bhalf_conversion_type<unsigned int>();
  test_bhalf_conversion_type<unsigned long>();
  test_bhalf_conversion_type<unsigned long long>();
}

// bool converts to 0 and 1, and a half value converts to bool as any
// floating-point value does: false for +-0, true otherwise (also NaN).  In
// device code this needs cast_to_half(bool), cast_from_half<bool>, and the
// bhalf_t equivalents to be defined for the backend.
template <class H>
void test_half_bool_conversion(float denorm_min) {
  constexpr int n        = 8;
  const float values[n]  = {0.f,   -0.f,       0.5f,     -2.f,
                            1e-3f, denorm_min, INFINITY, NAN};
  const bool expected[n] = {false, false, true, true, true, true, true, true};
  using MemorySpace      = typename TEST_EXECSPACE::memory_space;
  Kokkos::View<H[n], MemorySpace> in("in");
  Kokkos::View<bool[n], MemorySpace> to_bool("to_bool");
  Kokkos::View<float[2], MemorySpace> from_bool("from_bool");
  auto h_in = Kokkos::create_mirror_view(in);
  for (int i = 0; i < n; ++i) h_in(i) = H(values[i]);
  Kokkos::deep_copy(in, h_in);

  // Host.
  EXPECT_EQ(static_cast<float>(H(false)), 0.f);
  EXPECT_EQ(static_cast<float>(H(true)), 1.f);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(static_cast<bool>(h_in(i)), expected[i]) << values[i];
  }

  // Device.
  Kokkos::parallel_for(
      "TestHalfBoolConversion", Kokkos::RangePolicy<TEST_EXECSPACE>(0, n),
      KOKKOS_LAMBDA(int i) {
        to_bool(i) = static_cast<bool>(in(i));
        if (i < 2) from_bool(i) = static_cast<float>(H(i == 1));
      });
  auto h_to_bool =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), to_bool);
  auto h_from_bool =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), from_bool);
  EXPECT_EQ(h_from_bool(0), 0.f);
  EXPECT_EQ(h_from_bool(1), 1.f);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(h_to_bool(i), expected[i]) << values[i];
  }
}

// The cast functions themselves, on the host and in a kernel.
inline void test_half_bool_cast_functions() {
  using Kokkos::Experimental::half_t;
  EXPECT_EQ(static_cast<float>(Kokkos::Experimental::cast_to_half(true)), 1.f);
  EXPECT_FALSE(Kokkos::Experimental::cast_from_half<bool>(half_t(0.f)));
  int errors = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
      KOKKOS_LAMBDA(int, int &err) {
        if (!(static_cast<float>(Kokkos::Experimental::cast_to_half(true)) ==
              1.f))
          ++err;
        if (!Kokkos::Experimental::cast_from_half<bool>(half_t(0.25f))) ++err;
      },
      errors);
  EXPECT_EQ(errors, 0);
}

inline void test_bhalf_bool_cast_functions() {
  using Kokkos::Experimental::bhalf_t;
  EXPECT_EQ(static_cast<float>(Kokkos::Experimental::cast_to_bhalf(true)), 1.f);
  EXPECT_FALSE(Kokkos::Experimental::cast_from_bhalf<bool>(bhalf_t(0.f)));
  int errors = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
      KOKKOS_LAMBDA(int, int &err) {
        if (!(static_cast<float>(Kokkos::Experimental::cast_to_bhalf(true)) ==
              1.f))
          ++err;
        if (!Kokkos::Experimental::cast_from_bhalf<bool>(bhalf_t(0.25f))) ++err;
      },
      errors);
  EXPECT_EQ(errors, 0);
}

// Assigning a numeric-trait constant (a BitComparisonWrapper where H is a half
// type) goes through floating_point_wrapper::operator=(T), which needs
// cast_to_half(half_t)/cast_to_bhalf(bhalf_t); the std backend lacked them, so
// this failed to link. The result must have the bits of the constant.
template <class H>
void test_half_assign_trait_constants() {
  using Bits = std::conditional_t<sizeof(H) == 2, std::uint16_t, std::uint32_t>;
  constexpr int n = 4;
  // Reference: direct initialization, which takes the constant's bits as is.
  const H ref[n] = {H(Kokkos::infinity<H>::value), H(Kokkos::epsilon<H>::value),
                    H(Kokkos::finite_max<H>::value),
                    H(Kokkos::norm_min<H>::value)};
  auto assign    = [](int i) {
    H x;
    switch (i) {
      case 0: x = Kokkos::infinity<H>::value; break;
      case 1: x = Kokkos::epsilon<H>::value; break;
      case 2: x = Kokkos::finite_max<H>::value; break;
      default: x = Kokkos::norm_min<H>::value; break;
    }
    return x;
  };
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Kokkos::bit_cast<Bits>(assign(i)), Kokkos::bit_cast<Bits>(ref[i]))
        << i;
  }

  using MemorySpace = typename TEST_EXECSPACE::memory_space;
  Kokkos::View<H[n], MemorySpace> out("out");
  Kokkos::parallel_for(
      "TestHalfAssignTraitConstants", Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
      KOKKOS_LAMBDA(int) {
        out(0) = Kokkos::infinity<H>::value;
        out(1) = Kokkos::epsilon<H>::value;
        out(2) = Kokkos::finite_max<H>::value;
        out(3) = Kokkos::norm_min<H>::value;
      });
  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Kokkos::bit_cast<Bits>(h_out(i)), Kokkos::bit_cast<Bits>(ref[i]))
        << i;
  }
}

// Every cast function declared in Kokkos_Half_FloatingPointWrapper.hpp, and
// every floating_point_wrapper operation that reaches one through
// cast_to_wrapper/cast_from_wrapper, called once with a value of 1. A backend
// that declares one of them without defining it fails to link here (H18 was
// such a case). Returns the number of wrong results.
KOKKOS_FUNCTION inline Kokkos::Experimental::half_t cast_to(
    Kokkos::Experimental::half_t *, Kokkos::Experimental::half_t x) {
  return Kokkos::Experimental::cast_to_half(x);
}
template <class T>
KOKKOS_FUNCTION Kokkos::Experimental::half_t cast_to(
    Kokkos::Experimental::half_t *, T x) {
  return Kokkos::Experimental::cast_to_half(x);
}
template <class T>
KOKKOS_FUNCTION T cast_from(Kokkos::Experimental::half_t x) {
  return Kokkos::Experimental::cast_from_half<T>(x);
}
#if !KOKKOS_BHALF_T_IS_FLOAT
KOKKOS_FUNCTION inline Kokkos::Experimental::bhalf_t cast_to(
    Kokkos::Experimental::bhalf_t *, Kokkos::Experimental::bhalf_t x) {
  return Kokkos::Experimental::cast_to_bhalf(x);
}
template <class T>
KOKKOS_FUNCTION Kokkos::Experimental::bhalf_t cast_to(
    Kokkos::Experimental::bhalf_t *, T x) {
  return Kokkos::Experimental::cast_to_bhalf(x);
}
template <class T>
KOKKOS_FUNCTION T cast_from(Kokkos::Experimental::bhalf_t x) {
  return Kokkos::Experimental::cast_from_bhalf<T>(x);
}
#endif

template <class H, class T>
KOKKOS_FUNCTION int check_half_cast_overloads_for() {
  int err        = 0;
  const H one    = H(1.f);
  const double d = 1.0;
  // The cast functions themselves.
  err += static_cast<double>(cast_to(static_cast<H *>(nullptr), T(1))) != d;
  err += cast_from<T>(one) != T(1);
  // Wrapper paths: constructor and operator=(T) (cast_to_wrapper), conversion
  // (cast_from_wrapper).
  H c(T(1));
  err += static_cast<double>(c) != d;
  H a;
  a = T(1);
  err += static_cast<double>(a) != d;
  err += static_cast<T>(one) != T(1);
  if constexpr (!std::is_same_v<T, bool>) {
    // Mixed binary operators.
    err += static_cast<double>(one + T(1)) != 2.0;
    err += static_cast<double>(T(1) + one) != 2.0;
    err += static_cast<double>(one - T(1)) != 0.0;
    err += static_cast<double>(T(1) - one) != 0.0;
    err += static_cast<double>(one * T(1)) != d;
    err += static_cast<double>(T(1) * one) != d;
    err += static_cast<double>(one / T(1)) != d;
    err += static_cast<double>(T(1) / one) != d;
    // Compound assignment exists only with float and double (both
    // directions); integers would be ambiguous between them.
    if constexpr (std::is_floating_point_v<T>) {
      T x = T(1);
      x += one;
      err += x != T(2);
      x -= one;
      err += x != T(1);
      x *= one;
      err += x != T(1);
      x /= one;
      err += x != T(1);
      H h = one;
      h += T(1);
      err += static_cast<double>(h) != 2.0;
    }
  }
  return err;
}

template <class H>
KOKKOS_FUNCTION int check_half_cast_overloads() {
  int err = 0;
  err += check_half_cast_overloads_for<H, bool>();
  err += check_half_cast_overloads_for<H, float>();
  err += check_half_cast_overloads_for<H, double>();
  err += check_half_cast_overloads_for<H, short>();
  err += check_half_cast_overloads_for<H, int>();
  err += check_half_cast_overloads_for<H, long>();
  err += check_half_cast_overloads_for<H, long long>();
  err += check_half_cast_overloads_for<H, unsigned short>();
  err += check_half_cast_overloads_for<H, unsigned int>();
  err += check_half_cast_overloads_for<H, unsigned long>();
  err += check_half_cast_overloads_for<H, unsigned long long>();
  // The identity casts.
  const H one = H(1.f);
  err += static_cast<double>(cast_to(static_cast<H *>(nullptr), one)) != 1.0;
  H a;
  a = one;
  err += static_cast<double>(a) != 1.0;
  return err;
}

template <class H>
void test_half_cast_overloads() {
  EXPECT_EQ(check_half_cast_overloads<H>(), 0);
  int errors = 0;
  Kokkos::parallel_reduce(
      "TestHalfCastOverloads", Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
      KOKKOS_LAMBDA(int, int &err) { err += check_half_cast_overloads<H>(); },
      errors);
  EXPECT_EQ(errors, 0);
}

TEST(TEST_CATEGORY, half_cast_overloads) {
  test_half_cast_overloads<Kokkos::Experimental::half_t>();
}

TEST(TEST_CATEGORY, bhalf_cast_overloads) {
  test_half_cast_overloads<Kokkos::Experimental::bhalf_t>();
}

TEST(TEST_CATEGORY, half_assign_trait_constants) {
  test_half_assign_trait_constants<Kokkos::Experimental::half_t>();
}

TEST(TEST_CATEGORY, bhalf_assign_trait_constants) {
  test_half_assign_trait_constants<Kokkos::Experimental::bhalf_t>();
}

TEST(TEST_CATEGORY, half_bool_conversion) {
  test_half_bool_conversion<Kokkos::Experimental::half_t>(0x1p-24f);
  test_half_bool_cast_functions();
}

TEST(TEST_CATEGORY, bhalf_bool_conversion) {
  test_half_bool_conversion<Kokkos::Experimental::bhalf_t>(0x1p-133f);
  test_bhalf_bool_cast_functions();
}

TEST(TEST_CATEGORY, half_conversion) { test_half_conversion(); }

TEST(TEST_CATEGORY, bhalf_conversion) { test_bhalf_conversion(); }

}  // namespace Test
#endif
