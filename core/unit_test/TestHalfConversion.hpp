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
