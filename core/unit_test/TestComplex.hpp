// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>

// Suppress "'long double' is treated as 'double' in device code"
// The suppression needs to happen before Kokkos_Complex.hpp is included to be
// effective
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

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <complex>
#include <sstream>

namespace {
template <typename... Ts>
KOKKOS_FUNCTION constexpr void maybe_unused(Ts &&...) noexcept {}
}  // namespace

namespace Test {

// Test construction and assignment

template <class ExecSpace, class RealType>
struct TestComplexConstruction {
  Kokkos::View<Kokkos::complex<RealType> *, ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<RealType> *,
                        ExecSpace>::host_mirror_type h_results;

  void testit() {
    d_results = Kokkos::View<Kokkos::complex<RealType> *, ExecSpace>(
        "TestComplexConstruction", 10);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    ASSERT_EQ(h_results(0).real(), 1.5);
    ASSERT_EQ(h_results(0).imag(), 2.5);
    ASSERT_EQ(h_results(1).real(), 1.5);
    ASSERT_EQ(h_results(1).imag(), 2.5);
    ASSERT_EQ(h_results(2).real(), 0.0);
    ASSERT_EQ(h_results(2).imag(), 0.0);
    ASSERT_EQ(h_results(3).real(), 3.5);
    ASSERT_EQ(h_results(3).imag(), 0.0);
    ASSERT_EQ(h_results(4).real(), 4.5);
    ASSERT_EQ(h_results(4).imag(), 5.5);
    ASSERT_EQ(h_results(5).real(), 1.5);
    ASSERT_EQ(h_results(5).imag(), 2.5);
    ASSERT_EQ(h_results(6).real(), 4.5);
    ASSERT_EQ(h_results(6).imag(), 5.5);
    ASSERT_EQ(h_results(7).real(), 7.5);
    ASSERT_EQ(h_results(7).imag(), 0.0);
    ASSERT_EQ(h_results(8).real(), RealType(8));
    ASSERT_EQ(h_results(8).imag(), 0.0);

    // Copy construction conversion between
    // Kokkos::complex and std::complex doesn't compile
    // Compare through double: ASSERT_FLOAT_EQ does not accept half_t, and all
    // values below are exactly representable down to bhalf_t.
    Kokkos::complex<RealType> a(1.5, 2.5), b(3.25, 5.25), r_kk;
    std::complex<RealType> sa(a), sb(3.25, 5.25), r;
    r    = a;
    r_kk = a;
    ASSERT_EQ(static_cast<double>(r.real()), static_cast<double>(r_kk.real()));
    ASSERT_EQ(static_cast<double>(r.imag()), static_cast<double>(r_kk.imag()));
    r    = sb * a;
    r_kk = b * a;
    ASSERT_EQ(static_cast<double>(r.real()), static_cast<double>(r_kk.real()));
    ASSERT_EQ(static_cast<double>(r.imag()), static_cast<double>(r_kk.imag()));
    ASSERT_EQ(static_cast<double>(r.real()), -8.25);
    ASSERT_EQ(static_cast<double>(r.imag()), 16.);
    r    = sa;
    r_kk = a;
    ASSERT_EQ(static_cast<double>(r.real()), static_cast<double>(r_kk.real()));
    ASSERT_EQ(static_cast<double>(r.imag()), static_cast<double>(r_kk.imag()));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    Kokkos::complex<RealType> a(1.5, 2.5);
    d_results(0) = a;
    Kokkos::complex<RealType> b(a);
    d_results(1)                = b;
    Kokkos::complex<RealType> c = Kokkos::complex<RealType>();
    d_results(2)                = c;
    Kokkos::complex<RealType> d(3.5);
    d_results(3) = d;
    Kokkos::complex<RealType> a_v(4.5, 5.5);
    d_results(4) = a_v;
    Kokkos::complex<RealType> b_v(a);
    d_results(5) = b_v;
    Kokkos::complex<RealType> e(a_v);
    d_results(6) = e;

    d_results(7) = RealType(7.5);
    d_results(8) = int(8);
  }
};

TEST(TEST_CATEGORY, complex_construction) {
  TestComplexConstruction<TEST_EXECSPACE, float> f;
  f.testit();

  TestComplexConstruction<TEST_EXECSPACE, double> d;
  d.testit();

#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  TestComplexConstruction<TEST_EXECSPACE, Kokkos::Experimental::half_t> h;
  h.testit();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  TestComplexConstruction<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t> bh;
  bh.testit();
#endif
}

// Test Math FUnction

template <class ExecSpace, class RealType>
struct TestComplexBasicMath {
  Kokkos::View<Kokkos::complex<RealType> *, ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<RealType> *,
                        ExecSpace>::host_mirror_type h_results;

  void testit() {
    d_results = Kokkos::View<Kokkos::complex<RealType> *, ExecSpace>(
        "TestComplexBasicMath", 24);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    std::complex<double> a(1.5, 2.5);
    std::complex<double> b(3.25, 5.75);
    std::complex<double> d(1.0, 2.0);
    double c = 9.3;

    // gamma_n = (n * eps) / (1 - n * eps)
    double eps =
        static_cast<double>(static_cast<RealType>(Kokkos::epsilon_v<RealType>));
    auto gamma = [&](int n) {
      return (static_cast<RealType>(n) * eps) /
             (RealType(1) - static_cast<RealType>(n) * eps);
    };

    auto addition = [&](double a_real, double a_imag, double b_real,
                        double b_imag, RealType r_real, RealType r_imag,
                        double ref_real, double ref_imag) {
      double error_bound_real = eps * Kokkos::abs(a_real + b_real);
      double error_bound_imag = eps * Kokkos::abs(a_imag + b_imag);
      double error_real       = Kokkos::abs(ref_real - r_real);
      double error_imag       = Kokkos::abs(ref_imag - r_imag);
      return std::make_tuple(error_real, error_imag, error_bound_real,
                             error_bound_imag);
    };

    auto subtraction = [&](double a_real, double a_imag, double b_real,
                           double b_imag, RealType r_real, RealType r_imag,
                           double ref_real, double ref_imag) {
      double error_bound_real = eps * Kokkos::abs(a_real - b_real);
      double error_bound_imag = eps * Kokkos::abs(a_imag - b_imag);
      double error_real       = Kokkos::abs(ref_real - r_real);
      double error_imag       = Kokkos::abs(ref_imag - r_imag);
      return std::make_tuple(error_real, error_imag, error_bound_real,
                             error_bound_imag);
    };

    auto multiplication = [&](double a_real, double a_imag, double b_real,
                              double b_imag, RealType r_real, RealType r_imag,
                              double ref_real, double ref_imag) {
      double error_bound_real = gamma(3) * (Kokkos::abs(a_real * b_real) +
                                            Kokkos::abs(a_imag * b_imag));
      double error_bound_imag = gamma(3) * (Kokkos::abs(a_real * b_imag) +
                                            Kokkos::abs(a_imag * b_real));
      double error_real       = Kokkos::abs(ref_real - r_real);
      double error_imag       = Kokkos::abs(ref_imag - r_imag);
      return std::make_tuple(error_real, error_imag, error_bound_real,
                             error_bound_imag);
    };

    auto division = [&](double a_real, double a_imag, double b_real,
                        double b_imag, RealType r_real, RealType r_imag,
                        double ref_real, double ref_imag) {
      double S_r = Kokkos::abs(a_real * b_real) + Kokkos::abs(a_imag * b_imag);
      double S_i = Kokkos::abs(a_imag * b_real) + Kokkos::abs(a_real * b_imag);
      double S_D = Kokkos::abs(b_real) * Kokkos::abs(b_real) +
                   Kokkos::abs(b_imag) * Kokkos::abs(b_imag);
      double N_r              = a_real * b_real + a_imag * b_imag;
      double N_i              = a_imag * b_real - a_real * b_imag;
      double error_bound_real = (gamma(3) * S_r + Kokkos::abs(N_r) * gamma(2) +
                                 eps * (Kokkos::abs(N_r) + gamma(3) * S_r)) /
                                (S_D * (1 - gamma(2)));
      double error_bound_imag = (gamma(3) * S_i + Kokkos::abs(N_i) * gamma(2) +
                                 eps * (Kokkos::abs(N_i) + gamma(3) * S_i)) /
                                (S_D * (1 - gamma(2)));
      double error_real = Kokkos::abs(ref_real - r_real);
      double error_imag = Kokkos::abs(ref_imag - r_imag);

      return std::make_tuple(error_real, error_imag, error_bound_real,
                             error_bound_imag);
    };

    std::complex<double> r;
    // Testing z1 op z2 (op: +, -, *, /)
    {
      r = a + b;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          addition(a.real(), a.imag(), b.real(), b.imag(), h_results(0).real(),
                   h_results(0).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = a - b;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          subtraction(a.real(), a.imag(), b.real(), b.imag(),
                      h_results(1).real(), h_results(1).imag(), r.real(),
                      r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = a * b;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          multiplication(a.real(), a.imag(), b.real(), b.imag(),
                         h_results(2).real(), h_results(2).imag(), r.real(),
                         r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = a / b;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          division(a.real(), a.imag(), b.real(), b.imag(), h_results(3).real(),
                   h_results(3).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    // Testing z1 op z2 (op: +=, -=, *=, /=)
    {
      r = d + a;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          addition(d.real(), d.imag(), a.real(), a.imag(), h_results(4).real(),
                   h_results(4).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = d - a;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          subtraction(d.real(), d.imag(), a.real(), a.imag(),
                      h_results(5).real(), h_results(5).imag(), r.real(),
                      r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = d * a;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          multiplication(d.real(), d.imag(), a.real(), a.imag(),
                         h_results(6).real(), h_results(6).imag(), r.real(),
                         r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = d / a;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          division(d.real(), d.imag(), a.real(), a.imag(), h_results(7).real(),
                   h_results(7).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    // Testing z1 op r1 (op: +, -, *, /)
    {
      r = a + c;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          addition(a.real(), a.imag(), c, 0.0, h_results(8).real(),
                   h_results(8).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = a - c;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          subtraction(a.real(), a.imag(), c, 0.0, h_results(9).real(),
                      h_results(9).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = a * c;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          multiplication(a.real(), a.imag(), c, 0.0, h_results(10).real(),
                         h_results(10).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = a / c;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          division(a.real(), a.imag(), c, 0.0, h_results(11).real(),
                   h_results(11).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    // Testing z1 op r1 (op: +=, -=, *=, /=)
    {
      r = d + c;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          addition(d.real(), d.imag(), c, 0.0, h_results(12).real(),
                   h_results(12).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = d - c;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          subtraction(d.real(), d.imag(), c, 0.0, h_results(13).real(),
                      h_results(13).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = d * c;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          multiplication(d.real(), d.imag(), c, 0.0, h_results(14).real(),
                         h_results(14).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = d / c;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          division(d.real(), d.imag(), c, 0.0, h_results(15).real(),
                   h_results(15).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    // Testing r1 op z1 (op: +, -, *, /)
    {
      r = c + a;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          addition(c, 0.0, a.real(), a.imag(), h_results(16).real(),
                   h_results(16).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = c - a;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          subtraction(c, 0.0, a.real(), a.imag(), h_results(17).real(),
                      h_results(17).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = c * a;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          multiplication(c, 0.0, a.real(), a.imag(), h_results(18).real(),
                         h_results(18).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      r = c / a;
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          division(c, 0.0, a.real(), a.imag(), h_results(19).real(),
                   h_results(19).imag(), r.real(), r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    // Testing z1 op i1 (op: +, -, *, /)
    int e = 2;
    r     = a;
    {
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          addition(a.real(), a.imag(), e, 0.0, h_results(20).real(),
                   h_results(20).imag(), r.real() + e, r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          subtraction(a.real(), a.imag(), e, 0.0, h_results(21).real(),
                      h_results(21).imag(), r.real() - e, r.imag());
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          multiplication(a.real(), a.imag(), e, 0.0, h_results(22).real(),
                         h_results(22).imag(), r.real() * e, r.imag() * e);
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }

    {
      auto [error_real, error_imag, error_bound_real, error_bound_imag] =
          division(a.real(), a.imag(), e, 0.0, h_results(23).real(),
                   h_results(23).imag(), r.real() / e, r.imag() / e);
      ASSERT_LE(error_real, error_bound_real);
      ASSERT_LE(error_imag, error_bound_imag);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    Kokkos::complex<RealType> a(1.5, 2.5);
    Kokkos::complex<RealType> b(3.25, 5.75);
    // Basic math complex / complex
    d_results(0) = a + b;
    d_results(1) = a - b;
    d_results(2) = a * b;
    d_results(3) = a / b;
    d_results(4).real(1.0);
    d_results(4).imag(2.0);
    d_results(4) += a;
    d_results(5) = Kokkos::complex<RealType>(1.0, 2.0);
    d_results(5) -= a;
    d_results(6) = Kokkos::complex<RealType>(1.0, 2.0);
    d_results(6) *= a;
    d_results(7) = Kokkos::complex<RealType>(1.0, 2.0);
    d_results(7) /= a;

    // Basic math complex / scalar
    RealType c    = 9.3;
    d_results(8)  = a + c;
    d_results(9)  = a - c;
    d_results(10) = a * c;
    d_results(11) = a / c;
    d_results(12).real(1.0);
    d_results(12).imag(2.0);
    d_results(12) += c;
    d_results(13) = Kokkos::complex<RealType>(1.0, 2.0);
    d_results(13) -= c;
    d_results(14) = Kokkos::complex<RealType>(1.0, 2.0);
    d_results(14) *= c;
    d_results(15) = Kokkos::complex<RealType>(1.0, 2.0);
    d_results(15) /= c;

    // Basic math scalar / complex
    d_results(16) = c + a;
    d_results(17) = c - a;
    d_results(18) = c * a;
    d_results(19) = c / a;

    int e         = 2;
    d_results(20) = a + e;
    d_results(21) = a - e;
    d_results(22) = a * e;
    d_results(23) = a / e;
  }
};

TEST(TEST_CATEGORY, complex_basic_math) {
  TestComplexBasicMath<TEST_EXECSPACE, float> f;
  f.testit();
  TestComplexBasicMath<TEST_EXECSPACE, double> d;
  d.testit();

#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  TestComplexBasicMath<TEST_EXECSPACE, Kokkos::Experimental::half_t> h;
  h.testit();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  TestComplexBasicMath<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t> bh;
  bh.testit();
#endif
}

// Compares a Kokkos::complex result with a std::complex<double> reference,
// using a relative tolerance scaled by the modulus of the reference.
// A macro rather than a function, so that gtest reports the line of each check.
#define CHECK_COMPLEX_NEAR(value, ref, eps)                              \
  do {                                                                   \
    const std::complex<double> ref_ = (ref);                             \
    const double tol_ = 10 * (eps) * std::max(1.0, std::abs(ref_));      \
    EXPECT_NEAR(static_cast<double>((value).real()), ref_.real(), tol_); \
    EXPECT_NEAR(static_cast<double>((value).imag()), ref_.imag(), tol_); \
  } while (false)

template <class ExecSpace, class RealType>
struct TestComplexSpecialFunctions {
  // |a|^9.3 ~ 2e4 with a phase of ~9.6 rad: too inaccurate in half precision.
  static constexpr double pow_exponent =
      std::is_floating_point_v<RealType> ? 9.3 : 1.2;

  Kokkos::View<Kokkos::complex<RealType> *, ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<RealType> *,
                        ExecSpace>::host_mirror_type h_results;

  void testit() {
    d_results = Kokkos::View<Kokkos::complex<RealType> *, ExecSpace>(
        "TestComplexSpecialFunctions", 20);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    // epsilon_v<half_t> is a bit pattern wrapper: convert through RealType.
    const double eps =
        static_cast<double>(static_cast<RealType>(Kokkos::epsilon_v<RealType>));

    std::complex<double> a(1.5, 2.5);
    double c = pow_exponent;

    CHECK_COMPLEX_NEAR(h_results(0), a, eps);
    CHECK_COMPLEX_NEAR(h_results(1), std::sqrt(a), eps);
    CHECK_COMPLEX_NEAR(h_results(2), std::pow(a, c), eps);
    CHECK_COMPLEX_NEAR(h_results(3), std::abs(a), eps);
    CHECK_COMPLEX_NEAR(h_results(4), std::exp(a), eps);
    CHECK_COMPLEX_NEAR(h_results(4), Kokkos::exp(a), eps);
    CHECK_COMPLEX_NEAR(h_results(5), std::log(a), eps);
    CHECK_COMPLEX_NEAR(h_results(6), std::sin(a), eps);
    CHECK_COMPLEX_NEAR(h_results(7), std::cos(a), eps);
    CHECK_COMPLEX_NEAR(h_results(8), std::tan(a), eps);
    CHECK_COMPLEX_NEAR(h_results(9), std::sinh(a), eps);
    CHECK_COMPLEX_NEAR(h_results(10), std::cosh(a), eps);
    CHECK_COMPLEX_NEAR(h_results(11), std::tanh(a), eps);
    CHECK_COMPLEX_NEAR(h_results(12), std::asinh(a), eps);
    CHECK_COMPLEX_NEAR(h_results(13), std::acosh(a), eps);
    // atanh
    CHECK_COMPLEX_NEAR(h_results(14), std::atanh(a), eps);
    CHECK_COMPLEX_NEAR(h_results(15), std::asin(a), eps);
    CHECK_COMPLEX_NEAR(h_results(16), std::acos(a), eps);
    // atan
    CHECK_COMPLEX_NEAR(h_results(17), std::atan(a), eps);
    // log10
    CHECK_COMPLEX_NEAR(h_results(18), std::log10(a), eps);
    // norm
    CHECK_COMPLEX_NEAR(h_results(19), std::norm(a), eps);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    Kokkos::complex<RealType> a(1.5, 2.5);
    RealType c = pow_exponent;

    d_results(0)  = Kokkos::complex<RealType>(Kokkos::real(a), Kokkos::imag(a));
    d_results(1)  = Kokkos::sqrt(a);
    d_results(2)  = Kokkos::pow(a, c);
    d_results(3)  = Kokkos::abs(a);
    d_results(4)  = Kokkos::exp(a);
    d_results(5)  = Kokkos::log(a);
    d_results(6)  = Kokkos::sin(a);
    d_results(7)  = Kokkos::cos(a);
    d_results(8)  = Kokkos::tan(a);
    d_results(9)  = Kokkos::sinh(a);
    d_results(10) = Kokkos::cosh(a);
    d_results(11) = Kokkos::tanh(a);
    d_results(12) = Kokkos::asinh(a);
    d_results(13) = Kokkos::acosh(a);
    d_results(14) = Kokkos::atanh(a);
    d_results(15) = Kokkos::asin(a);
    d_results(16) = Kokkos::acos(a);
    d_results(17) = Kokkos::atan(a);
    d_results(18) = Kokkos::log10(a);
    d_results(19) = Kokkos::norm(a);
  }
};

#undef CHECK_COMPLEX_NEAR

TEST(TEST_CATEGORY, complex_special_funtions) {
  TestComplexSpecialFunctions<TEST_EXECSPACE, float> f;
  f.testit();
  TestComplexSpecialFunctions<TEST_EXECSPACE, double> d;
  d.testit();

#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  TestComplexSpecialFunctions<TEST_EXECSPACE, Kokkos::Experimental::half_t> h;
  h.testit();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  TestComplexSpecialFunctions<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t> bh;
  bh.testit();
#endif
}

template <typename RealType>
void testComplexIO() {
  // 9/64 and 25/64 are exact down to bhalf_t and print the same with the
  // default stream format (float/double) and with half_t's fixed 6 decimals.
  Kokkos::complex<RealType> z = {0.140625, 0.390625};
  std::stringstream ss;
  ss << z;
  ASSERT_EQ(ss.str(), "(0.140625,0.390625)");

  // FIXME_HALF: half_t's operator>> reads a whole whitespace-delimited token,
  // so the compact form "(2) (3,4)" cannot be parsed. Separate every token.
  ss.str("1 ( 2 ) ( 3 , 4 )");
  ss.clear();
  ss >> z;
  ASSERT_EQ(z, (Kokkos::complex<RealType>{1, 0}));
  ss >> z;
  ASSERT_EQ(z, (Kokkos::complex<RealType>{2, 0}));
  ss >> z;
  ASSERT_EQ(z, (Kokkos::complex<RealType>{3, 4}));
}

TEST(TEST_CATEGORY, complex_io) {
  testComplexIO<float>();
  testComplexIO<double>();
#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  testComplexIO<Kokkos::Experimental::half_t>();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  testComplexIO<Kokkos::Experimental::bhalf_t>();
#endif
}

static_assert(std::is_trivially_copyable_v<Kokkos::complex<float>>);
static_assert(std::is_trivially_copyable_v<Kokkos::complex<double>>);
#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
static_assert(std::is_trivially_copyable_v<
              Kokkos::complex<Kokkos::Experimental::half_t>>);
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
static_assert(std::is_trivially_copyable_v<
              Kokkos::complex<Kokkos::Experimental::bhalf_t>>);
#endif
#ifndef KOKKOS_IMPL_32BIT  // FIXME_32BIT
// error: requested alignment '24' is not a positive power of 2
static_assert(std::is_trivially_copyable_v<Kokkos::complex<long double>>);
#endif

template <class ExecSpace, class RealType>
struct TestBugPowAndLogComplex {
  Kokkos::View<Kokkos::complex<RealType> *, ExecSpace> d_pow;
  Kokkos::View<Kokkos::complex<RealType> *, ExecSpace> d_log;
  TestBugPowAndLogComplex() : d_pow("pow", 2), d_log("log", 2) { test(); }
  void test() {
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    auto h_pow =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_pow);
    ASSERT_FLOAT_EQ(h_pow(0).real(), 18);
    ASSERT_FLOAT_EQ(h_pow(0).imag(), 26);
    ASSERT_FLOAT_EQ(h_pow(1).real(), -18);
    ASSERT_FLOAT_EQ(h_pow(1).imag(), 26);
    auto h_log =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_log);
    ASSERT_FLOAT_EQ(h_log(0).real(), 1.151292546497023);
    ASSERT_FLOAT_EQ(h_log(0).imag(), 0.3217505543966422);
    ASSERT_FLOAT_EQ(h_log(1).real(), 1.151292546497023);
    ASSERT_FLOAT_EQ(h_log(1).imag(), 2.819842099193151);
  }
  KOKKOS_FUNCTION void operator()(int) const {
    d_pow(0) = Kokkos::pow(Kokkos::complex<RealType>(+3., 1.), 3.);
    d_pow(1) = Kokkos::pow(Kokkos::complex<RealType>(-3., 1.), 3.);
    d_log(0) = Kokkos::log(Kokkos::complex<RealType>(+3., 1.));
    d_log(1) = Kokkos::log(Kokkos::complex<RealType>(-3., 1.));
  }
};

TEST(TEST_CATEGORY, complex_issue_3865) {
  TestBugPowAndLogComplex<TEST_EXECSPACE, double>();
}

TEST(TEST_CATEGORY, complex_operations_arithmetic_types_overloads) {
  static_assert(Kokkos::real(1) == 1.);
  static_assert(Kokkos::real(2.f) == 2.f);
  static_assert(Kokkos::real(3.) == 3.);
  static_assert(Kokkos::real(4.l) == 4.l);
  static_assert((std::is_same_v<decltype(Kokkos::real(1)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::real(2.f)), float>));
  static_assert((std::is_same_v<decltype(Kokkos::real(3.)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::real(4.l)), long double>));

  static_assert(Kokkos::imag(1) == 0.);
  static_assert(Kokkos::imag(2.f) == 0.f);
  static_assert(Kokkos::imag(3.) == 0.);
  static_assert(Kokkos::imag(4.l) == 0.l);
  static_assert((std::is_same_v<decltype(Kokkos::imag(1)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::imag(2.f)), float>));
  static_assert((std::is_same_v<decltype(Kokkos::imag(3.)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::real(4.l)), long double>));

  static_assert(Kokkos::conj(1) == Kokkos::complex<double>(1));
  static_assert(Kokkos::conj(2.f) == Kokkos::complex<float>(2.f));
  static_assert(Kokkos::conj(3.) == Kokkos::complex<double>(3.));
// long double has size 12 but Kokkos::complex requires 2*sizeof(T) to be a
// power of two.
#ifndef KOKKOS_IMPL_32BIT
  static_assert(Kokkos::conj(4.l) == Kokkos::complex<long double>(4.l));
  static_assert(
      (std::is_same_v<decltype(Kokkos::conj(1)), Kokkos::complex<double>>));
#endif
  static_assert(
      (std::is_same_v<decltype(Kokkos::conj(2.f)), Kokkos::complex<float>>));
  static_assert(
      (std::is_same_v<decltype(Kokkos::conj(3.)), Kokkos::complex<double>>));
  static_assert((std::is_same_v<decltype(Kokkos::conj(4.l)),
                                Kokkos::complex<long double>>));
}

template <class ExecSpace, class RealType>
struct TestComplexStructuredBindings {
  using exec_space       = ExecSpace;
  using value_type       = RealType;
  using complex_type     = Kokkos::complex<RealType>;
  using device_view_type = Kokkos::View<complex_type *, exec_space>;
  using host_view_type   = typename device_view_type::host_mirror_type;

  device_view_type d_results;
  host_view_type h_results;

  // tuple_size
  static_assert(std::is_same_v<typename std::tuple_size<complex_type>::type,
                               std::integral_constant<size_t, 2>>);

  // tuple_element
  static_assert(
      std::is_same_v<std::tuple_element_t<0, complex_type>, value_type>);
  static_assert(
      std::is_same_v<std::tuple_element_t<1, complex_type>, value_type>);

  static void testgetreturnreferencetypes() {
    complex_type m;
    const complex_type c;

    // get lvalue
    complex_type &ml = m;
    static_assert(std::is_same_v<decltype(Kokkos::get<0>(ml)), value_type &>);
    static_assert(std::is_same_v<decltype(Kokkos::get<1>(ml)), value_type &>);

    // get rvalue
    complex_type &&mr = std::move(m);
    static_assert(
        std::is_same_v<decltype(Kokkos::get<0>(std::move(mr))), value_type &&>);
    static_assert(
        std::is_same_v<decltype(Kokkos::get<1>(std::move(mr))), value_type &&>);

    // get const lvalue
    const complex_type &cl = c;
    static_assert(
        std::is_same_v<decltype(Kokkos::get<0>(cl)), value_type const &>);
    static_assert(
        std::is_same_v<decltype(Kokkos::get<1>(cl)), value_type const &>);

    // get const rvalue
    complex_type const &&cr = std::move(c);
    static_assert(std::is_same_v<decltype(Kokkos::get<0>(std::move(cr))),
                                 value_type const &&>);
    static_assert(std::is_same_v<decltype(Kokkos::get<1>(std::move(cr))),
                                 value_type const &&>);

    maybe_unused(m, c, ml, mr, cl, cr);
  }

  void testit() {
    testgetreturnreferencetypes();

    d_results = device_view_type("TestComplexStructuredBindings", 6);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    // get lvalue
    ASSERT_EQ(h_results[0].real(), 2.);
    ASSERT_EQ(h_results[0].imag(), 3.);

    // get rvalue
    ASSERT_EQ(h_results[1].real(), 2.);
    ASSERT_EQ(h_results[1].imag(), 3.);

    // get const lvalue
    ASSERT_EQ(h_results[2].real(), 5.);
    ASSERT_EQ(h_results[2].imag(), 7.);

    // get const rvalue
    ASSERT_EQ(h_results[3].real(), 5.);
    ASSERT_EQ(h_results[3].imag(), 7.);

    // swap real and imaginary
    ASSERT_EQ(h_results[4].real(), 11.);
    ASSERT_EQ(h_results[4].imag(), 13.);
    ASSERT_EQ(h_results[5].real(), 13.);
    ASSERT_EQ(h_results[5].imag(), 11.);
  }

  KOKKOS_FUNCTION
  void operator()(int) const {
    complex_type m(2., 3.);
    const complex_type c(5., 7.);

    // get lvalue
    {
      complex_type &ml = m;
      auto &[mlr, mli] = ml;
      d_results[0]     = complex_type(mlr, mli);
    }

    // get rvalue
    {
      complex_type &&mr = std::move(m);
      auto &&[mrr, mri] = std::move(mr);
      d_results[1]      = complex_type(mrr, mri);
    }

    // get const lvalue
    {
      const complex_type &cl = c;
      auto &[clr, cli]       = cl;
      d_results[2]           = complex_type(clr, cli);
    }

    // get const rvalue
    {
      complex_type const &&cr = std::move(c);
      auto &&[crr, cri]       = std::move(cr);
      d_results[3]            = complex_type(crr, cri);
    }

    // swap real and imaginary
    {
      complex_type z(11., 13.);
      d_results[4] = z;

      auto &[zr, zi] = z;
      Kokkos::kokkos_swap(zr, zi);
      d_results[5] = z;
    }
  }
};

TEST(TEST_CATEGORY, complex_structured_bindings) {
  TestComplexStructuredBindings<TEST_EXECSPACE, float> f;
  f.testit();

  TestComplexStructuredBindings<TEST_EXECSPACE, double> d;
  d.testit();

#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  TestComplexStructuredBindings<TEST_EXECSPACE, Kokkos::Experimental::half_t> h;
  h.testit();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  TestComplexStructuredBindings<TEST_EXECSPACE, Kokkos::Experimental::bhalf_t>
      bh;
  bh.testit();
#endif
}

#define CHECK_COMPLEX(_value_, _real_, _imag_) \
  (void)_value_;                               \
  if (_value_.real() != _real_) return false;  \
  if (_value_.imag() != _imag_) return false;

template <class RealType>
constexpr bool can_appear_in_constant_expressions() {
  const Kokkos::complex<RealType> from_single{1.2};
  const Kokkos::complex<RealType> from_both{1.2, 3.4};
  const Kokkos::complex<RealType> from_none{};

  CHECK_COMPLEX(from_single, RealType(1.2), RealType(0.));
  CHECK_COMPLEX(from_both, RealType(1.2), RealType(3.4));
  CHECK_COMPLEX(from_none, RealType(0.), RealType(0.));

  Kokkos::complex<RealType> from_copy_assign;
  from_copy_assign = from_both;
  const auto from_copy_constr(from_both);

  CHECK_COMPLEX(from_copy_assign, RealType(1.2), RealType(3.4));
  CHECK_COMPLEX(from_copy_constr, RealType(1.2), RealType(3.4));

  Kokkos::complex<RealType> from_move_assign;
  from_move_assign = std::move(from_both);
  const auto from_move_constr(std::move(from_copy_assign));

  CHECK_COMPLEX(from_move_assign, RealType(1.2), RealType(3.4));
  CHECK_COMPLEX(from_move_constr, RealType(1.2), RealType(3.4));

  Kokkos::complex<RealType> from_real;
  from_real = 4.;

  CHECK_COMPLEX(from_real, RealType(4.), RealType(0.));

  return true;
}

#undef CHECK_COMPLEX

static_assert(can_appear_in_constant_expressions<float>());
static_assert(can_appear_in_constant_expressions<double>());

template <class RealType>
constexpr bool comparison_in_constant_expression() {
  static_assert(Kokkos::complex<RealType>{42., 43.} ==
                Kokkos::complex<RealType>{42., 43.});
  static_assert(Kokkos::complex<RealType>{42., 43.} !=
                Kokkos::complex<RealType>{42., 42.});

  static_assert(Kokkos::complex<RealType>{42., 0.} == RealType{42.});
  static_assert(Kokkos::complex<RealType>{42., 43.} != RealType{42.});

  static_assert(RealType{42.} == Kokkos::complex<RealType>{42., 0.});
  static_assert(RealType{43.} != Kokkos::complex<RealType>{42., 0.});

  return true;
}

static_assert(comparison_in_constant_expression<float>());
static_assert(comparison_in_constant_expression<double>());

template <class RealType>
constexpr bool test_complex_norm() {
  return Kokkos::norm(Kokkos::complex<RealType>{4., 2.}) == RealType(20.);
}
static_assert(test_complex_norm<float>());
static_assert(test_complex_norm<double>());

constexpr bool test_overload_norm() {
  constexpr auto res_int = Kokkos::norm(int(100000));
  static_assert(std::same_as<decltype(res_int), const double>);
  static_assert(res_int == 1e10);

  constexpr auto res_float = Kokkos::norm(float(666.));
  static_assert(std::same_as<decltype(res_float), const float>);
  static_assert(res_float == 666. * 666.);

  return true;
}
static_assert(test_overload_norm());

template <class RealType>
constexpr bool test_complex_conj() {
  static_assert(Kokkos::conj(Kokkos::complex<RealType>{1., 1.}) ==
                Kokkos::complex<RealType>{1., -1.});
  static_assert(Kokkos::conj(Kokkos::complex<RealType>{1., -1.}) ==
                Kokkos::complex<RealType>{1., 1.});
  static_assert(Kokkos::conj(Kokkos::complex<RealType>{-1., 1.}) ==
                Kokkos::complex<RealType>{-1., -1.});
  static_assert(Kokkos::conj(Kokkos::complex<RealType>{-1., -1.}) ==
                Kokkos::complex<RealType>{-1., 1.});
  return true;
}
static_assert(test_complex_conj<float>());
static_assert(test_complex_conj<double>());

// half_t and bhalf_t cannot be used in constant expressions: their conversions,
// comparisons and arithmetic are not constexpr (e.g. CUDA __half intrinsics).
// Run the checks above at runtime instead.
template <class RealType>
bool comparison_at_runtime() {
  return Kokkos::complex<RealType>{42., 43.} ==
             Kokkos::complex<RealType>{42., 43.} &&
         Kokkos::complex<RealType>{42., 43.} !=
             Kokkos::complex<RealType>{42., 42.} &&
         Kokkos::complex<RealType>{42., 0.} == RealType{42.} &&
         Kokkos::complex<RealType>{42., 43.} != RealType{42.} &&
         RealType{42.} == Kokkos::complex<RealType>{42., 0.} &&
         RealType{43.} != Kokkos::complex<RealType>{42., 0.};
}

template <class RealType>
bool complex_conj_at_runtime() {
  return Kokkos::conj(Kokkos::complex<RealType>{1., 1.}) ==
             Kokkos::complex<RealType>{1., -1.} &&
         Kokkos::conj(Kokkos::complex<RealType>{1., -1.}) ==
             Kokkos::complex<RealType>{1., 1.} &&
         Kokkos::conj(Kokkos::complex<RealType>{-1., 1.}) ==
             Kokkos::complex<RealType>{-1., -1.} &&
         Kokkos::conj(Kokkos::complex<RealType>{-1., -1.}) ==
             Kokkos::complex<RealType>{-1., 1.};
}

template <class RealType>
void test_constexpr_checks_at_runtime() {
  ASSERT_TRUE(can_appear_in_constant_expressions<RealType>());
  ASSERT_TRUE(comparison_at_runtime<RealType>());
  ASSERT_TRUE(test_complex_norm<RealType>());
  ASSERT_TRUE(complex_conj_at_runtime<RealType>());
}

TEST(TEST_CATEGORY, complex_constexpr_checks_at_runtime) {
  test_constexpr_checks_at_runtime<float>();
  test_constexpr_checks_at_runtime<double>();
#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  test_constexpr_checks_at_runtime<Kokkos::Experimental::half_t>();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  test_constexpr_checks_at_runtime<Kokkos::Experimental::bhalf_t>();
#endif
}

}  // namespace Test

#ifdef KOKKOS_COMPILER_NVCC
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diagnostic pop
#else
#ifdef __CUDA_ARCH__
#pragma diagnostic pop
#endif
#endif
#endif
