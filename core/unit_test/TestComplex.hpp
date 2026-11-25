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
#include <functional>
#include <iomanip>
#include <iterator>
#include <random>
#include <sstream>
#include <string>

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
    // The interop with std::complex is tested in complex_std_interop_*.
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

// The reference of CHECK_COMPLEX_NEAR as a std::complex<double>: a real
// number, a std::complex<double>, or a Kokkos::complex (from its parts; its
// conversion operator to std::complex is deprecated, D11).
inline std::complex<double> complex_near_ref(double x) { return x; }
inline std::complex<double> complex_near_ref(std::complex<double> x) {
  return x;
}
template <class T>
std::complex<double> complex_near_ref(Kokkos::complex<T> x) {
  return {static_cast<double>(x.real()), static_cast<double>(x.imag())};
}

// Compares a Kokkos::complex result with a std::complex<double> reference,
// using a relative tolerance scaled by the modulus of the reference.
// A macro rather than a function, so that gtest reports the line of each check.
#define CHECK_COMPLEX_NEAR(value, ref, eps)                                     \
  do {                                                                          \
    const std::complex<double> ref_ = complex_near_ref(ref);                    \
    const double tol_               = 10 * (eps)*std::max(1.0, std::abs(ref_)); \
    EXPECT_NEAR(static_cast<double>((value).real()), ref_.real(), tol_);        \
    EXPECT_NEAR(static_cast<double>((value).imag()), ref_.imag(), tol_);        \
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
    CHECK_COMPLEX_NEAR(h_results(4), Kokkos::exp(Kokkos::complex<double>(a)),
                       eps);
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
  // 9/64 and 25/64 are exact down to bhalf_t, so every type prints them the
  // same way (half_t and bhalf_t are written as the float they equal).
  Kokkos::complex<RealType> z = {0.140625, 0.390625};
  std::stringstream ss;
  ss << z;
  ASSERT_EQ(ss.str(), "(0.140625,0.390625)");

  ss.str("1 (2) (3,4)");
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

// A failed extraction sets failbit and leaves the complex number unchanged.
TEST(TEST_CATEGORY, complex_io_failure) {
  const Kokkos::complex<double> orig{5., 6.};
  for (const char *input : {"x", "(x)", "(1,x)", "(1;2)", "(1,2"}) {
    std::stringstream ss(input);
    Kokkos::complex<double> z = orig;
    ss >> z;
    EXPECT_TRUE(ss.fail()) << input;
    EXPECT_EQ(z, orig) << input;
  }
}

// Insertion and extraction behave like those of std::complex<RealType>.  For
// half_t and bhalf_t (std::complex<half_t> is unspecified), like those of
// std::complex<float> holding the same values, the parts then converted to
// the half type: std::complex<std::float16_t> reads its parts through float
// (C++23 [istream.formatted.arithmetic]/6).  Parts beyond the half type's
// range are tested separately.
template <class RealType, class CharT>
void testComplexIOLikeStd() {
  using Out =
      std::conditional_t<std::is_floating_point_v<RealType>, RealType, float>;
  using In =
      std::conditional_t<std::is_floating_point_v<RealType>, RealType, float>;
  using Stream = std::basic_stringstream<CharT>;
  using String = std::basic_string<CharT>;
  auto widen   = [](const char *s) {
    std::string n(s);
    return String(n.begin(), n.end());
  };

  // Output: identical characters for every formatting state.
  const RealType values[][2] = {{RealType(3.14), RealType(1.41)},
                                {RealType(-0.5), RealType(1e-7)},
                                {RealType(12345.678), RealType(-0.)}};
  const std::function<void(Stream &)> formats[] = {
      [](Stream &) {},
      [](Stream &s) { s.precision(3); },
      [](Stream &s) { s << std::scientific << std::uppercase; },
      [](Stream &s) { s << std::fixed << std::setprecision(2); },
      [](Stream &s) { s << std::showpos << std::showpoint; },
      [](Stream &s) { s << std::hexfloat; },
      // The width applies to the whole "(re,im)"
      [](Stream &s) {
        s << std::left << std::setfill(CharT('*')) << std::setw(30);
      },
      [](Stream &s) { s << std::setw(30); },
  };
  for (const auto &v : values) {
    for (const auto &format : formats) {
      Stream sk;
      Stream ss;
      format(sk);
      format(ss);
      sk << Kokkos::complex<RealType>(v[0], v[1]) << CharT('|');
      ss << std::complex<Out>(static_cast<Out>(v[0]), static_cast<Out>(v[1]))
         << CharT('|');
      EXPECT_TRUE(sk.str() == ss.str());
    }
  }

  // Input: same value, same stream state and same unread characters.
  const char *inputs[] = {"1",      "(2)",          "(3,4)", " ( 5 , 6 ) z",
                          "7 8",    "(1e-2,-3.5)z", "",      "x",
                          "(x)",    "(1,x)",        "(1;2)", "(1,2",
                          "(1,2]z", "(1",           "("};
  for (const char *input : inputs) {
    SCOPED_TRACE(input);
    Stream sk(widen(input));
    Stream ss(widen(input));
    Kokkos::complex<RealType> zk(RealType(5), RealType(6));
    std::complex<In> zs(In(5), In(6));
    sk >> zk;
    ss >> zs;
    EXPECT_EQ(sk.fail(), ss.fail());
    EXPECT_EQ(zk.real(), static_cast<RealType>(zs.real()));
    EXPECT_EQ(zk.imag(), static_cast<RealType>(zs.imag()));
    EXPECT_EQ(sk.eof(), ss.eof());
    sk.clear();
    ss.clear();
    EXPECT_TRUE(String(std::istreambuf_iterator<CharT>(sk), {}) ==
                String(std::istreambuf_iterator<CharT>(ss), {}));
  }
}

// A part beyond the range of the half type fails as in
// std::complex<std::float16_t>: the part's extraction sets failbit, so the
// complex number is unchanged and the rest of the input is not read.
template <class RealType>
void testComplexIOHalfRange(const char *too_big) {
  using C = Kokkos::complex<RealType>;
  const std::string big(too_big);
  for (const std::string &input :
       {big, "(" + big + ")", "(" + big + ",2)z", "(1," + big + ")z"}) {
    SCOPED_TRACE(input);
    std::stringstream ss(input);
    C z(RealType(5), RealType(6));
    ss >> z;
    EXPECT_TRUE(ss.fail());
    EXPECT_EQ(z, C(RealType(5), RealType(6)));
    ss.clear();
    const std::string rest(std::istreambuf_iterator<char>(ss), {});
    const std::string expected_rest =
        input.substr(input.find(big) + big.size());
    EXPECT_EQ(rest, expected_rest);
  }
}

TEST(TEST_CATEGORY, complex_io_like_std) {
  testComplexIOLikeStd<float, char>();
  testComplexIOLikeStd<double, char>();
  testComplexIOLikeStd<double, wchar_t>();
#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  testComplexIOLikeStd<Kokkos::Experimental::half_t, char>();
  testComplexIOLikeStd<Kokkos::Experimental::half_t, wchar_t>();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  testComplexIOLikeStd<Kokkos::Experimental::bhalf_t, char>();
#endif
#if !KOKKOS_HALF_T_IS_FLOAT
  testComplexIOHalfRange<Kokkos::Experimental::half_t>("1e5");
#endif
#if !KOKKOS_BHALF_T_IS_FLOAT
  testComplexIOHalfRange<Kokkos::Experimental::bhalf_t>("3.4e38");
#endif
}

// Division assignment by a std::complex<RealType> converts it with the
// (host-only) converting constructor and divides like
// operator/(complex, complex) (D11: no own overload any more).
template <class RealType>
void testComplexDivideAssignStdComplex() {
  using C = Kokkos::complex<RealType>;
  C z(RealType(1.5), RealType(2.5));
  z /= std::complex<RealType>(RealType(3.25), RealType(5.75));
  EXPECT_EQ(
      z, C(RealType(1.5), RealType(2.5)) / C(RealType(3.25), RealType(5.75)));
}

TEST(TEST_CATEGORY, complex_divide_assign_std_complex) {
  testComplexDivideAssignStdComplex<float>();
  testComplexDivideAssignStdComplex<double>();
}

// Interop with std::complex (D11, A8). Only the converting constructor from
// std::complex<RealType> stays: it is implicit, so a std::complex<RealType>
// also works as a call argument, in assignment and in the compound
// assignments. Everything else is deprecated in Kokkos 5 and removed in
// Kokkos 6. float and double only: std::complex<half_t> is unspecified.
template <class RealType>
void testComplexStdInteropConvertingCtor() {
  using C = Kokkos::complex<RealType>;
  using S = std::complex<RealType>;
  const S s(RealType(1.5), RealType(-2.5));
  const C c(RealType(1.5), RealType(-2.5));
  const C a = s;  // copy-initialization
  const C b(s);   // direct-initialization
  EXPECT_EQ(a, c);
  EXPECT_EQ(b, c);
  auto take = [](C z) { return z; };
  EXPECT_EQ(take(s), c);  // call argument

  C z(RealType(1), RealType(2));
  z = s;
  EXPECT_EQ(z, c);
  z = C(RealType(1), RealType(2));
  z += s;
  EXPECT_EQ(z, C(RealType(2.5), RealType(-0.5)));
  z = C(RealType(1), RealType(2));
  z -= s;
  EXPECT_EQ(z, C(RealType(-0.5), RealType(4.5)));
  z = C(RealType(1), RealType(2));
  z *= s;
  EXPECT_EQ(z, C(RealType(1), RealType(2)) * c);
  z = C(RealType(1), RealType(2));
  z /= s;
  EXPECT_EQ(z, C(RealType(1), RealType(2)) / c);

  // Braced assignment: unambiguous since the std::complex overloads of
  // operator= and operator/= are gone (they made both ambiguous).
  z = {RealType(3), RealType(4)};
  EXPECT_EQ(z, C(RealType(3), RealType(4)));
  z /= {RealType(1), RealType(0)};
  EXPECT_EQ(z, C(RealType(3), RealType(4)));
}

TEST(TEST_CATEGORY, complex_std_interop_converting_ctor) {
  testComplexStdInteropConvertingCtor<float>();
  testComplexStdInteropConvertingCtor<double>();
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
// The deprecated interop still works until Kokkos 6, for the half types too
// (where std::complex<half_t> is generic libstdc++ code).
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
template <class RealType>
void testComplexStdInteropDeprecated() {
  using C = Kokkos::complex<RealType>;
  using S = std::complex<RealType>;
  // Compare through double: ASSERT_FLOAT_EQ does not accept half_t, and all
  // values below are exactly representable down to bhalf_t.
  auto same = [](const S &x, const C &y) {
    return static_cast<double>(x.real()) == static_cast<double>(y.real()) &&
           static_cast<double>(x.imag()) == static_cast<double>(y.imag());
  };
  const C a(RealType(1.5), RealType(2.5));
  const C b(RealType(3.25), RealType(5.25));

  // The conversion operator to std::complex (copy- and direct-init,
  // assignment).
  const S sa = a;
  const S sd(a);
  S r;
  r = a;
  EXPECT_TRUE(same(sa, a));
  EXPECT_TRUE(same(sd, a));
  EXPECT_TRUE(same(r, a));

  // std::complex * Kokkos::complex.
  const S sb(b.real(), b.imag());
  const C p = sb * a;
  EXPECT_EQ(p, b * a);
  EXPECT_EQ(static_cast<double>(p.real()), -8.25);
  EXPECT_EQ(static_cast<double>(p.imag()), 16.);

  // == and != with std::complex, both orders.
  EXPECT_TRUE(sa == a);
  EXPECT_TRUE(a == sa);
  EXPECT_FALSE(sa != a);
  EXPECT_TRUE(a != sb);

  if constexpr (std::is_floating_point_v<RealType>) {
    // exp of a std::complex: the textbook formula, so compare with a
    // tolerance with exp(complex), whose algorithm differs (A6).
    const C e  = Kokkos::exp(sa);
    const C ek = Kokkos::exp(a);
    const RealType tol =
        8 * Kokkos::Experimental::epsilon_v<RealType> * Kokkos::abs(ek);
    EXPECT_NEAR(e.real(), ek.real(), tol);
    EXPECT_NEAR(e.imag(), ek.imag(), tol);
  }
  if constexpr (std::is_same_v<RealType, double>) {
    // Assignment and division assignment from a std::complex of another
    // value type (std::complex<float> converts to std::complex<double>).
    const std::complex<float> f(1.5f, -2.5f);
    C z;
    z = f;
    EXPECT_EQ(z, C(1.5, -2.5));
    z = C(1, 2);
    z /= f;
    EXPECT_EQ(z, C(1, 2) / C(1.5, -2.5));
  }
}
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()

TEST(TEST_CATEGORY, complex_std_interop_deprecated) {
  testComplexStdInteropDeprecated<float>();
  testComplexStdInteropDeprecated<double>();
#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  testComplexStdInteropDeprecated<Kokkos::Experimental::half_t>();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  testComplexStdInteropDeprecated<Kokkos::Experimental::bhalf_t>();
#endif
}
#endif

// Complex division: special values required by C99 Annex G (G.5.2), operands
// near the ends of the range, and mixed precision.  Each case is computed with
// both operator/ and operator/=.
template <class RealType, class ExecSpace>
struct TestComplexDivision {
  using C                       = Kokkos::complex<RealType>;
  static constexpr int N        = 7;
  static constexpr RealType inf = Kokkos::Experimental::infinity_v<RealType>;
  static constexpr RealType big =
      Kokkos::Experimental::finite_max_v<RealType> / 2;
  static constexpr RealType tiny = Kokkos::Experimental::norm_min_v<RealType>;

  Kokkos::View<C *[2], ExecSpace> d_in;   // numerator, denominator
  Kokkos::View<C *[2], ExecSpace> d_out;  // x / y, x /= y
  Kokkos::View<Kokkos::complex<float>[1], ExecSpace> d_xf;
  Kokkos::View<Kokkos::complex<double>[1], ExecSpace> d_yd;
  Kokkos::View<Kokkos::complex<double>[2], ExecSpace> d_mixed;

  TestComplexDivision()
      : d_in("in", N),
        d_out("out", N),
        d_xf("xf"),
        d_yd("yd"),
        d_mixed("mixed") {}

  KOKKOS_FUNCTION void operator()(int i) const {
    d_out(i, 0) = d_in(i, 0) / d_in(i, 1);
    C z         = d_in(i, 0);
    z /= d_in(i, 1);
    d_out(i, 1) = z;
    if (i == 0) {
      d_mixed(0) = d_xf(0) / d_yd(0);  // complex<float> / complex<double>
      d_mixed(1) = Kokkos::complex<double>(d_xf(0)) / d_yd(0);
    }
  }

  void testit() {
    const C num[N] = {C(1, 1), C(1, 1),     C(inf, 0),
                      C(1, 1), C(big, big), C(RealType(1.5), RealType(2.5)),
                      C(1, 1)};
    const C den[N] = {C(0, 0),
                      C(-RealType(0), 0),
                      C(1, 1),
                      C(inf, 0),
                      C(big, big),
                      C(RealType(3.25), RealType(5.75)),
                      C(4 * tiny, 2 * tiny)};
    auto h_in      = Kokkos::create_mirror_view(d_in);
    for (int i = 0; i < N; ++i) {
      h_in(i, 0) = num[i];
      h_in(i, 1) = den[i];
    }
    Kokkos::deep_copy(d_in, h_in);
    Kokkos::deep_copy(d_xf, Kokkos::complex<float>(1.f / 3.f, 2.f / 7.f));
    Kokkos::deep_copy(d_yd, Kokkos::complex<double>(1.1, 2.3));

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, N), *this);
    auto h_out =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    auto h_mixed =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_mixed);

    // Reference for ordinary values: textbook formula in long double.
    auto ref = [](C x, C y) {
      using L       = long double;
      const L a     = x.real();
      const L b     = x.imag();
      const L c     = y.real();
      const L d     = y.imag();
      const L denom = c * c + d * d;
      return std::complex<L>((a * c + b * d) / denom, (b * c - a * d) / denom);
    };
    auto near = [](C r, std::complex<long double> e) {
      const long double err =
          std::abs(std::complex<long double>(r.real(), r.imag()) - e);
      return err <= 4 * Kokkos::Experimental::epsilon_v<RealType> * std::abs(e);
    };

    for (int j = 0; j < 2; ++j) {
      SCOPED_TRACE(j == 0 ? "operator/" : "operator/=");
      // nonzero / zero
      EXPECT_EQ(h_out(0, j).real(), inf);
      EXPECT_EQ(h_out(0, j).imag(), inf);
      // nonzero / zero, the sign of infinity is the sign of real(y)
      EXPECT_EQ(h_out(1, j).real(), -inf);
      EXPECT_EQ(h_out(1, j).imag(), -inf);
      // infinite / finite
      EXPECT_EQ(h_out(2, j).real(), inf);
      EXPECT_EQ(h_out(2, j).imag(), -inf);
      // finite / infinite
      EXPECT_EQ(h_out(3, j), C(0, 0));
      // |real(y)| + |imag(y)| overflows
      EXPECT_EQ(h_out(4, j), C(1, 0));
      // ordinary values
      EXPECT_TRUE(near(h_out(5, j), ref(num[5], den[5]))) << h_out(5, j);
      // smallest normal denominator: (1 + i) / (m (4 + 2i)) = (6 + 2i) / (20 m)
      EXPECT_TRUE(near(
          h_out(6, j),
          std::complex<long double>(6.L / (20.L * tiny), 2.L / (20.L * tiny))))
          << h_out(6, j);
    }

    // complex<float> / complex<double> is computed in double
    EXPECT_EQ(h_mixed(0), h_mixed(1));
    const auto e = std::complex<long double>(1.f / 3.f, 2.f / 7.f) /
                   std::complex<long double>(1.1, 2.3);
    const double tol = 4 * Kokkos::Experimental::epsilon_v<double> *
                       static_cast<double>(std::abs(e));
    EXPECT_NEAR(h_mixed(0).real(), static_cast<double>(e.real()), tol);
    EXPECT_NEAR(h_mixed(0).imag(), static_cast<double>(e.imag()), tol);
  }
};

TEST(TEST_CATEGORY, complex_division) {
#if __FINITE_MATH_ONLY__
  GTEST_SKIP() << "needs infinities, NaNs and signed zeros, which "
                  "-ffinite-math-only does not provide";
#endif
  TestComplexDivision<float, TEST_EXECSPACE>().testit();
  TestComplexDivision<double, TEST_EXECSPACE>().testit();
}

// Complex multiplication: C99 Annex G (G.5.1) recovery of infinities when
// both parts of the usual formula are NaN, compared with std::complex.  Each
// case is computed with both operator* and operator*=.
template <class RealType, class ExecSpace>
struct TestComplexMultiplication {
  using C                       = Kokkos::complex<RealType>;
  static constexpr int N        = 6;
  static constexpr RealType inf = Kokkos::Experimental::infinity_v<RealType>;
  static constexpr RealType nan = Kokkos::Experimental::quiet_NaN_v<RealType>;
  static constexpr RealType huge =
      Kokkos::Experimental::finite_max_v<RealType> / 4;

  Kokkos::View<C *[2], ExecSpace> d_in;
  Kokkos::View<C *[2], ExecSpace> d_out;    // x * y, x *= y
  Kokkos::View<C[1], ExecSpace> d_formula;  // textbook formula for case N-1

  TestComplexMultiplication()
      : d_in("in", N), d_out("out", N), d_formula("formula") {}

  KOKKOS_FUNCTION void operator()(int i) const {
    const C x   = d_in(i, 0);
    const C y   = d_in(i, 1);
    d_out(i, 0) = x * y;
    C z         = x;
    z *= y;
    d_out(i, 1) = z;
    if (i == N - 1) {
      d_formula(0) = C(x.real() * y.real() - x.imag() * y.imag(),
                       x.real() * y.imag() + x.imag() * y.real());
    }
  }

  void testit() {
    const C num[N] = {C(inf, nan),   // infinite left operand
                      C(inf, inf),   // infinite left operand
                      C(1, 0),       // infinite right operand
                      C(nan, 0),     // stays (nan, nan)
                      C(huge, nan),  // overflowing partial products
                      C(RealType(1.5), RealType(2.5))};  // ordinary values
    const C den[N] = {C(1, 0),   C(1, 0),    C(inf, nan),
                      C(inf, 0), C(huge, 1), C(RealType(3.25), RealType(5.75))};
    auto h_in      = Kokkos::create_mirror_view(d_in);
    for (int i = 0; i < N; ++i) {
      h_in(i, 0) = num[i];
      h_in(i, 1) = den[i];
    }
    Kokkos::deep_copy(d_in, h_in);
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, N), *this);
    auto h_out =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    auto h_formula =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_formula);

    // Same value and sign, or both NaN.
    auto same = [](RealType v, RealType e) {
      return (std::isnan(v) && std::isnan(e)) ||
             (v == e && std::signbit(v) == std::signbit(e));
    };
    for (int j = 0; j < 2; ++j) {
      SCOPED_TRACE(j == 0 ? "operator*" : "operator*=");
      for (int i = 0; i < N - 1; ++i) {
        const std::complex<RealType> e =
            std::complex<RealType>(num[i].real(), num[i].imag()) *
            std::complex<RealType>(den[i].real(), den[i].imag());
        SCOPED_TRACE(::testing::Message() << num[i] << " * " << den[i]);
        EXPECT_TRUE(same(h_out(i, j).real(), e.real())) << h_out(i, j);
        EXPECT_TRUE(same(h_out(i, j).imag(), e.imag())) << h_out(i, j);
      }
      // The recovered values themselves (not only agreement with std).
      EXPECT_EQ(h_out(0, j).real(), inf);
      EXPECT_TRUE(std::isnan(h_out(0, j).imag()));
      EXPECT_EQ(h_out(1, j), C(inf, inf));
      EXPECT_EQ(h_out(2, j).real(), inf);
      EXPECT_TRUE(std::isnan(h_out(3, j).real()) &&
                  std::isnan(h_out(3, j).imag()));
      EXPECT_EQ(h_out(4, j), C(inf, inf));
      // Ordinary values: bit-identical to the textbook formula.
      EXPECT_EQ(h_out(N - 1, j), h_formula(0));
    }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
    // std::complex * Kokkos::complex (host only, deprecated) recovers the
    // same way.
    KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
    const C h = std::complex<RealType>(inf, nan) * C(1, 0);
    KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
    EXPECT_EQ(h.real(), inf);
    EXPECT_TRUE(std::isnan(h.imag()));
#endif
  }
};

TEST(TEST_CATEGORY, complex_multiplication) {
#if __FINITE_MATH_ONLY__
  GTEST_SKIP() << "needs infinities, NaNs and signed zeros, which "
                  "-ffinite-math-only does not provide";
#endif
  TestComplexMultiplication<float, TEST_EXECSPACE>().testit();
  TestComplexMultiplication<double, TEST_EXECSPACE>().testit();
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

  static_assert((std::is_same_v<decltype(Kokkos::arg(1)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::arg(2.f)), float>));
  static_assert((std::is_same_v<decltype(Kokkos::arg(3.)), double>));
  static_assert((std::is_same_v<decltype(Kokkos::arg(4.l)), long double>));
  EXPECT_EQ(Kokkos::arg(1), 0.);
  EXPECT_DOUBLE_EQ(Kokkos::arg(-1), std::arg(-1.));  // pi

  static_assert(
      (std::is_same_v<decltype(Kokkos::proj(1)), Kokkos::complex<double>>));
  static_assert(
      (std::is_same_v<decltype(Kokkos::proj(2.f)), Kokkos::complex<float>>));
  static_assert(
      (std::is_same_v<decltype(Kokkos::proj(3.)), Kokkos::complex<double>>));
  EXPECT_EQ(Kokkos::proj(1), Kokkos::complex<double>(1));
}

// arg and proj, for complex numbers and for real numbers, compared with
// std::arg and std::proj.  The inputs include signed zeros and infinities.
template <class RealType, class ExecSpace>
struct TestComplexArgProj {
  using C                       = Kokkos::complex<RealType>;
  static constexpr int N        = 13;
  static constexpr RealType inf = Kokkos::Experimental::infinity_v<RealType>;

  Kokkos::View<C *, ExecSpace> d_in;
  Kokkos::View<RealType *, ExecSpace> d_arg;
  Kokkos::View<C *, ExecSpace> d_proj;
  Kokkos::View<RealType[3], ExecSpace> d_arg_real;  // arg(-2), arg(-0.), arg(3)
  Kokkos::View<C[2], ExecSpace> d_proj_real;        // proj(-inf), proj(2)

  TestComplexArgProj()
      : d_in("in", N),
        d_arg("arg", N),
        d_proj("proj", N),
        d_arg_real("arg_real"),
        d_proj_real("proj_real") {}

  KOKKOS_FUNCTION void operator()(int i) const {
    d_arg(i)  = Kokkos::arg(d_in(i));
    d_proj(i) = Kokkos::proj(d_in(i));
    if (i == 0) {
      d_arg_real(0)  = Kokkos::arg(-RealType(2));
      d_arg_real(1)  = Kokkos::arg(-RealType(0));
      d_arg_real(2)  = Kokkos::arg(RealType(3));
      d_proj_real(0) = Kokkos::proj(-inf);
      d_proj_real(1) = Kokkos::proj(RealType(2));
    }
  }

  void testit() {
    const RealType z = 0;
    const C in[N]    = {C(1, 1),    C(-1, z),   C(-1, -z),  C(z, z),
                        C(-z, z),   C(z, -z),   C(-z, -z),  C(3, -4),
                        C(inf, -1), C(1, -inf), C(-inf, 2), C(-inf, -inf),
                        C(-z, -3)};
    auto h_in        = Kokkos::create_mirror_view(d_in);
    for (int i = 0; i < N; ++i) h_in(i) = in[i];
    Kokkos::deep_copy(d_in, h_in);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, N), *this);
    auto h_arg =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_arg);
    auto h_proj =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_proj);
    auto h_arg_real =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_arg_real);
    auto h_proj_real =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_proj_real);

    // atan2 on the device may differ from the host libm in the last bit.
    const RealType tol = 2 * Kokkos::Experimental::epsilon_v<RealType> * 4;
    auto same = [](RealType x, RealType y) {  // equal, and same sign of zero
      return x == y && std::signbit(x) == std::signbit(y);
    };
    for (int i = 0; i < N; ++i) {
      const std::complex<RealType> s(in[i].real(), in[i].imag());
      SCOPED_TRACE(::testing::Message() << "z = " << in[i]);
      EXPECT_NEAR(h_arg(i), std::arg(s), tol);
      EXPECT_EQ(std::signbit(h_arg(i)), std::signbit(std::arg(s)));
      const std::complex<RealType> p = std::proj(s);
      EXPECT_TRUE(same(h_proj(i).real(), p.real())) << h_proj(i);
      EXPECT_TRUE(same(h_proj(i).imag(), p.imag())) << h_proj(i);
    }
    EXPECT_NEAR(h_arg_real(0), std::arg(-RealType(2)), tol);
    EXPECT_NEAR(h_arg_real(1), std::arg(-RealType(0)), tol);  // pi
    EXPECT_EQ(h_arg_real(2), RealType(0));
    EXPECT_TRUE(same(h_proj_real(0).real(), inf));
    EXPECT_TRUE(same(h_proj_real(0).imag(), RealType(0)));  // +0
    EXPECT_EQ(h_proj_real(1), C(2, 0));
  }
};

TEST(TEST_CATEGORY, complex_arg_proj) {
#if __FINITE_MATH_ONLY__
  GTEST_SKIP() << "needs infinities, NaNs and signed zeros, which "
                  "-ffinite-math-only does not provide";
#endif
  TestComplexArgProj<float, TEST_EXECSPACE>().testit();
  TestComplexArgProj<double, TEST_EXECSPACE>().testit();
}

// exp, sqrt and log on the grid {0, -0, 1, -1, 0.5, 2.5, inf, -inf, nan}^2,
// compared with std::complex (glibc's C99 Annex G functions with libstdc++):
// same class (NaN, infinite, zero, finite) and sign for every part, and
// finite parts within a few eps.  Also polar with an infinite magnitude, and
// sqrt on both sides of its branch cut.
// The reference of the special-value tests below: for every input of their
// 9 x 9 grid (0, -0, 1, -1, 0.5, 2.5, inf, -inf, NaN for each part; input
// a * 9 + b is (v[a], v[b])) and each function, the kind and sign of the real
// and the imaginary part of the result as GCC/libstdc++/glibc give it, the
// reference the CUDA and std backends were verified against. Fixed here, so
// that every backend is held to the same signs: under icpx the host's
// std::complex uses Intel's math library, which gives other signs where C99
// Annex G leaves them unspecified, and C99's (NaN, NaN) where C11 specifies a
// zero (tanh(+0 + i inf) = +0 + iNaN). std::complex still provides the finite
// values. Same for float and double. Generated by
// tmp/claude_builds/complex_a6/gen_complex_special_ref.cpp.
// Codes: N NaN (any sign), I/i +-inf, Z/z +-0, P/m finite positive/negative.
// Function order: exp sqrt log sinh cosh tanh sin cos tan asinh acosh atanh
// asin acos atan.
inline constexpr const char *complex_special_ref[15] = {
    "PZPzPPPmPPmPNNNNNNPZPzPPPmPPmPNNNNNNPZPzPPPmPPmPNNNNNNPZPzPPPmPPmPNNNNNNPZ"
    "PzPPPmPPmPNNNNNNPZPzPPPmPPmPNNNNNNIZIzIIIiIIiIINININZZZzZZZzZZzZZZZzZZNZNz"
    "NNNNNNNNNNNNNN",  // exp
    "ZZZzPPPmPPPPIIIiNNZZZzPPPmPPPPIIIiNNPZPzPPPmPPPPIIIiNNZPZmPPPmPPPPIIIiNNPZ"
    "PzPPPmPPPPIIIiNNPZPzPPPmPPPPIIIiNNIZIzIZIzIZIZIIIiINZIZiZIZiZIZIIIIiNINNNN"
    "NNNNNNNNIIIiNN",  // sqrt
    "iZizZPZmmPPPIPImNNiPimZPZmmPPPIPImNNZZZzPPPmPPPPIPImNNZPZmPPPmPPPPIPImNNmZ"
    "mzPPPmmPPPIPImNNPZPzPPPmPPPPIPImNNIZIzIZIzIZIZIPImINIPImIPImIPIPIPImINNNNN"
    "NNNNNNNNININNN",  // log
    "ZZZzZPZmZPzPZNZNZNzZzzzPzmzPZPzNzNzNPZPzPPPmPPmPNNNNNNmZmzmPmmmPPPNNNNNNPZ"
    "PzPPPmPPmPNNNNNNPZPzPPPmPPmPNNNNNNIZIzIIIiIIiIINININiZiziIiiiIIIINININNZNz"
    "NNNNNNNNNNNNNN",  // sinh
    "PZPzPZPzPZmZNZNZNZPzPZPzPZPzmzNZNZNZPZPzPPPmPPmPNNNNNNPzPZPmPPPmmmNNNNNNPZ"
    "PzPPPmPPmPNNNNNNPZPzPPPmPPmPNNNNNNIZIzIIIiIIiIINININIzIZIiIIIiiiINININNZNz"
    "NNNNNNNNNNNNNN",  // cosh
    "ZZZzZPZmZPZmZNZNZNzZzzzPzmzPzmzNzNzNPZPzPPPmPPPmNNNNNNmZmzmPmmmPmmNNNNNNPZ"
    "PzPPPmPPPmNNNNNNPZPzPPPmPPPmNNNNNNPZPzPZPzPZPzPZPzPZmZmzmZmzmZmzmZmzmZNZNz"
    "NNNNNNNNNNNNNN",  // tanh
    "ZZZzZPZmZPZPZIZiZNzZzzzPzmzPzPzIzizNPZPzPPPmPPPPIIIiNNmZmzmPmmmPmPiIiiNNPZ"
    "PzPPPmPPPPIIIiNNPzPZPmPPPmPmIiIINNNZNzNNNNNNNNNININNNZNzNNNNNNNNNININNNZNz"
    "NNNNNNNNNININN",  // sin
    "PzPZPzPZPzPzIzIZNZPZPzPZPzPZPZIZIzNzPzPZPmPPPmPmIiIINNPZPzPPPmPPPPIIIiNNPz"
    "PZPmPPPmPmIiIINNmzmZmmmPmmmmiiiINNNZNZNNNNNNNNININNNNZNZNNNNNNNNININNNNZNZ"
    "NNNNNNNNININNN",  // cos
    "ZZZzZPZmZPZPZPZmZNzZzzzPzmzPzPzPzmzNPZPzPPPmPPPPZPZmNNmZmzmPmmmPmPzPzmNNPZ"
    "PzPPPmPPPPZPZmNNmZmzmPmmmPmPzPzmNNNZNzNNNNNNNNZPZmNNNZNzNNNNNNNNzPzmNNNZNz"
    "NNNNNNNNZPZmNN",  // tan
    "ZZZzZPZmZPPPIPImNNzZzzzPzmzPmPiPimNNPZPzPPPmPPPPIPImNNmZmzmPmmmPmPiPimNNPZ"
    "PzPPPmPPPPIPImNNPZPzPPPmPPPPIPImNNIZIzIZIzIZIZIPImINiZiziZiziZiZiPimiNNZNz"
    "NNNNNNNNININNN",  // asinh
    "ZPZmPPPmPPPPIPImNPZPZmPPPmPPPPIPImNPZZZzPPPmPPPPIPImNNZPZmPPPmPPPPIPImNNZP"
    "ZmPPPmPPPPIPImNNPZPzPPPmPPPPIPImNNIZIzIZIzIZIZIPImINIPImIPImIPIPIPImINNNNN"
    "NNNNNNNNININNN",  // acosh
    "ZZZzZPZmZPZPZPZmZNzZzzzPzmzPzPzPzmzNIZIzPPPmPPPPZPZmNNiZizmPmmmPmPzPzmNNPZ"
    "PzPPPmPPPPZPZmNNPPPmPPPmPPPPZPZmNNZPZmZPZmZPZPZPZmZNzPzmzPzmzPzPzPzmzNNNNN"
    "NNNNNNNNZPZmNN",  // atanh
    "ZZZzZPZmZPZPZIZiZNzZzzzPzmzPzPzIzizNPZPzPPPmPPPPZIZiNNmZmzmPmmmPmPzIziNNPZ"
    "PzPPPmPPPPZIZiNNPPPmPPPmPPPPZIZiNNPIPiPIPiPIPIPIPiNImImimImimImImImiNINNNN"
    "NNNNNNNNNINiNN",  // asin
    "PzPZPmPPPmPmPiPIPNPzPZPmPPPmPmPiPIPNZzZZPmPPPmPmPiPINNPzPZPmPPPmPmPiPINNPz"
    "PZPmPPPmPmPiPINNZmZPPmPPPmPmPiPINNZiZIZiZIZiZiPiPINiPiPIPiPIPiPiPiPINiNNNN"
    "NNNNNNNNNiNINN",  // acos
    "ZZZzZIZiZPPPPZPzNNzZzzzIzizPmPmZmzNNPZPzPPPmPPPPPZPzNNmZmzmPmmmPmPmZmzNNPZ"
    "PzPPPmPPPPPZPzNNPZPzPPPmPPPPPZPzNNPZPzPZPzPZPZPZPzPZmZmzmZmzmZmZmZmzmZNZNz"
    "NNNNNNNNNZNzNN",  // atan
};

template <class T>
char complex_special_code(T x) {
  if (std::isnan(x)) return 'N';
  if (std::isinf(x)) return std::signbit(x) ? 'i' : 'I';
  if (x == 0) return std::signbit(x) ? 'z' : 'Z';
  return std::signbit(x) ? 'm' : 'P';
}

// Whether part x of a result has the reference kind (and sign, if with_sign;
// never for NaN) of code c, and, when finite, is within tol of std's value e
// (if e is finite too: Intel's library can give NaN where glibc does not).
template <class T>
bool complex_special_check(T x, T e, char c, bool with_sign, T tol) {
  const char k = complex_special_code(x);
  auto kind    = [](char q) {
    return q == 'N'                 ? 0
              : (q == 'I' || q == 'i') ? 1
              : (q == 'Z' || q == 'z') ? 2
                                       : 3;
  };
  if (kind(k) != kind(c)) return false;
  if (with_sign && k != 'N' && k != c) return false;
  if (kind(k) == 3 && std::isfinite(e))
    return std::abs(x - e) <= tol * std::abs(e);
  return true;
}

template <class RealType, class ExecSpace>
struct TestComplexElementarySpecialValues {
  using C                       = Kokkos::complex<RealType>;
  static constexpr int NV       = 9;
  static constexpr int N        = NV * NV;
  static constexpr RealType inf = Kokkos::Experimental::infinity_v<RealType>;
  static constexpr RealType nan = Kokkos::Experimental::quiet_NaN_v<RealType>;

  Kokkos::View<C *, ExecSpace> d_in;
  Kokkos::View<C *[3], ExecSpace> d_out;  // exp, sqrt, log
  Kokkos::View<C[4], ExecSpace> d_polar;

  TestComplexElementarySpecialValues()
      : d_in("in", N), d_out("out", N), d_polar("polar") {}

  KOKKOS_FUNCTION void operator()(int i) const {
    d_out(i, 0) = Kokkos::exp(d_in(i));
    d_out(i, 1) = Kokkos::sqrt(d_in(i));
    d_out(i, 2) = Kokkos::log(d_in(i));
    if (i == 0) {
      // polar takes const T&: pass a local copy, not the static member
      // (odr-use of a host variable in device code).
      const RealType inf = TestComplexElementarySpecialValues::inf;
      d_polar(0)         = Kokkos::polar(inf, RealType(0));
      d_polar(1)         = Kokkos::polar(inf, -RealType(0));
      d_polar(2)         = Kokkos::polar(inf, RealType(1));
      d_polar(3)         = Kokkos::polar(RealType(2), RealType(0.5));
    }
  }

  void testit() {
    const RealType v[NV] = {
        0, -RealType(0), 1, -1, RealType(0.5), RealType(2.5), inf, -inf, nan};
    auto h_in = Kokkos::create_mirror_view(d_in);
    for (int a = 0; a < NV; ++a)
      for (int b = 0; b < NV; ++b) h_in(a * NV + b) = C(v[a], v[b]);
    Kokkos::deep_copy(d_in, h_in);
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, N), *this);
    auto h_out =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    auto h_polar =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_polar);

    auto kind = [](RealType x) {
      return std::isnan(x) ? 0 : std::isinf(x) ? 1 : x == 0 ? 2 : 3;
    };
    // Device math functions may differ from the host libm in the last bits.
    const RealType tol   = 8 * Kokkos::Experimental::epsilon_v<RealType>;
    const char *names[3] = {"exp", "sqrt", "log"};
    for (int i = 0; i < N; ++i) {
      const std::complex<RealType> z(h_in(i).real(), h_in(i).imag());
      const std::complex<RealType> e[3] = {std::exp(z), std::sqrt(z),
                                           std::log(z)};
      for (int f = 0; f < 3; ++f) {
        const char *ref = complex_special_ref[f] + 2 * i;
        EXPECT_TRUE(complex_special_check(h_out(i, f).real(), e[f].real(),
                                          ref[0], true, tol) &&
                    complex_special_check(h_out(i, f).imag(), e[f].imag(),
                                          ref[1], true, tol))
            << names[f] << h_in(i) << " = " << h_out(i, f) << ", reference "
            << ref[0] << ref[1] << ", std gives " << e[f];
      }
    }

    // sqrt is continuous from either side of the branch cut: the sign of
    // the imaginary part follows that of the input, also for -0.
    auto at = [&](RealType a, RealType b) {
      for (int i = 0; i < N; ++i)
        if (h_in(i).real() == a && h_in(i).imag() == b &&
            std::signbit(h_in(i).real()) == std::signbit(a) &&
            std::signbit(h_in(i).imag()) == std::signbit(b))
          return i;
      return -1;
    };
    EXPECT_EQ(h_out(at(-1, -RealType(0)), 1), C(0, -1));
    EXPECT_EQ(h_out(at(-1, RealType(0)), 1), C(0, 1));

    // polar(inf, +-0) is (inf, +-0) as in libc++ (libstdc++ gives
    // (inf, nan) from inf * sin(0)).
    EXPECT_EQ(h_polar(0).real(), inf);
    EXPECT_TRUE(h_polar(0).imag() == 0 && !std::signbit(h_polar(0).imag()));
    EXPECT_EQ(h_polar(1).real(), inf);
    EXPECT_TRUE(h_polar(1).imag() == 0 && std::signbit(h_polar(1).imag()));
    EXPECT_EQ(h_polar(2), C(inf, inf));
    const std::complex<RealType> p = std::polar(RealType(2), RealType(0.5));
    EXPECT_NEAR(h_polar(3).real(), p.real(), tol * 2);
    EXPECT_NEAR(h_polar(3).imag(), p.imag(), tol * 2);
  }
};

TEST(TEST_CATEGORY, complex_exp_sqrt_log_special_values) {
#if __FINITE_MATH_ONLY__
  GTEST_SKIP() << "needs infinities, NaNs and signed zeros, which "
                  "-ffinite-math-only does not provide";
#endif
  TestComplexElementarySpecialValues<float, TEST_EXECSPACE>().testit();
  TestComplexElementarySpecialValues<double, TEST_EXECSPACE>().testit();
}

// sinh, cosh, tanh on the same grid, compared with std::complex for class,
// sign and value; sin, cos, tan for class and value, and bitwise against
// the identities sin z = -i sinh(iz), cos z = cosh(iz), tan z = -i tanh(iz)
// (C99 G.6), which leave the signs that Annex G does not specify to sinh,
// cosh and tanh (glibc's csin/ccos choose them independently).  Also tanh for
// large real parts, where sinh(z) / cosh(z) overflowed to NaN.
template <class RealType, class ExecSpace>
struct TestComplexHyperbolicSpecialValues {
  using C                       = Kokkos::complex<RealType>;
  static constexpr int NV       = 9;
  static constexpr int N        = NV * NV;
  static constexpr RealType inf = Kokkos::Experimental::infinity_v<RealType>;
  static constexpr RealType nan = Kokkos::Experimental::quiet_NaN_v<RealType>;

  Kokkos::View<C *, ExecSpace> d_in;
  Kokkos::View<C *[9], ExecSpace> d_out;  // sinh cosh tanh sin cos tan, and
                                          // the identities for sin cos tan
  TestComplexHyperbolicSpecialValues() : d_in("in", N), d_out("out", N) {}

  KOKKOS_FUNCTION void operator()(int k) const {
    const C z   = d_in(k);
    const C iz  = C(-z.imag(), z.real());  // i z
    d_out(k, 0) = Kokkos::sinh(z);
    d_out(k, 1) = Kokkos::cosh(z);
    d_out(k, 2) = Kokkos::tanh(z);
    d_out(k, 3) = Kokkos::sin(z);
    d_out(k, 4) = Kokkos::cos(z);
    d_out(k, 5) = Kokkos::tan(z);
    const C s   = Kokkos::sinh(iz);
    const C t   = Kokkos::tanh(iz);
    d_out(k, 6) = C(s.imag(), -s.real());  // -i sinh(iz)
    d_out(k, 7) = Kokkos::cosh(iz);
    d_out(k, 8) = C(t.imag(), -t.real());  // -i tanh(iz)
  }

  void testit() {
    const RealType v[NV] = {
        0, -RealType(0), 1, -1, RealType(0.5), RealType(2.5), inf, -inf, nan};
    auto h_in = Kokkos::create_mirror_view(d_in);
    for (int a = 0; a < NV; ++a)
      for (int b = 0; b < NV; ++b) h_in(a * NV + b) = C(v[a], v[b]);
    Kokkos::deep_copy(d_in, h_in);
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, N), *this);
    auto h_out =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);

    auto kind = [](RealType x) {
      return std::isnan(x) ? 0 : std::isinf(x) ? 1 : x == 0 ? 2 : 3;
    };
    const RealType tol = 8 * Kokkos::Experimental::epsilon_v<RealType>;
    auto check         = [&](RealType x, RealType e, bool with_sign) {
      if (kind(x) != kind(e)) return false;
      if (with_sign && kind(x) != 0 && std::signbit(x) != std::signbit(e))
        return false;
      return kind(x) != 3 || std::abs(x - e) <= tol * std::abs(e);
    };
    auto bitwise = [](C x, C y) {
      auto same = [](RealType a, RealType b) {
        return (std::isnan(a) && std::isnan(b)) ||
               (a == b && std::signbit(a) == std::signbit(b));
      };
      return same(x.real(), y.real()) && same(x.imag(), y.imag());
    };
    const char *names[6] = {"sinh", "cosh", "tanh", "sin", "cos", "tan"};
    for (int k = 0; k < N; ++k) {
      const std::complex<RealType> z(h_in(k).real(), h_in(k).imag());
      const std::complex<RealType> e[6] = {std::sinh(z), std::cosh(z),
                                           std::tanh(z), std::sin(z),
                                           std::cos(z),  std::tan(z)};
      for (int f = 0; f < 6; ++f) {
        const bool with_sign = f < 3;
        const char *ref      = complex_special_ref[3 + f] + 2 * k;
        EXPECT_TRUE(complex_special_check(h_out(k, f).real(), e[f].real(),
                                          ref[0], with_sign, tol) &&
                    complex_special_check(h_out(k, f).imag(), e[f].imag(),
                                          ref[1], with_sign, tol))
            << names[f] << h_in(k) << " = " << h_out(k, f) << ", reference "
            << ref[0] << ref[1] << ", std gives " << e[f];
      }
      for (int f = 3; f < 6; ++f) {
        EXPECT_TRUE(bitwise(h_out(k, f), h_out(k, f + 3)))
            << names[f] << h_in(k) << " = " << h_out(k, f)
            << " differs from its identity " << h_out(k, f + 3);
      }
    }
  }
};

TEST(TEST_CATEGORY, complex_hyperbolic_special_values) {
#if __FINITE_MATH_ONLY__
  GTEST_SKIP() << "needs infinities, NaNs and signed zeros, which "
                  "-ffinite-math-only does not provide";
#endif
  TestComplexHyperbolicSpecialValues<float, TEST_EXECSPACE>().testit();
  TestComplexHyperbolicSpecialValues<double, TEST_EXECSPACE>().testit();
}

// tanh for real parts beyond (max_exponent - 1) ln 2 / 2, where sinh^2
// overflows: previously NaN (sinh(z) / cosh(z) = inf / inf), now
// (+-1, 4 sin(y) cos(y) exp(-2|x|)), and tan symmetrically.  Below that,
// Kahan's formula is accurate componentwise: the imaginary part of
// tanh(20 + i) (~7.7e-18) was 7 times too large with sinh(z) / cosh(z), and
// near a pole the parts must stay accurate too.
template <class RealType, class ExecSpace>
struct TestComplexTanhLarge {
  using C = Kokkos::complex<RealType>;
  Kokkos::View<C[6], ExecSpace> d_out;
  TestComplexTanhLarge() : d_out("out") {}
  // 800 (double) and 100 (float) are beyond the threshold; 20 / 8 are below.
  static constexpr RealType big   = sizeof(RealType) == 4 ? 100 : 800;
  static constexpr RealType small = sizeof(RealType) == 4 ? 8 : 20;
  // Close to the pole i pi/2 of tanh.
  static constexpr RealType pole_x = RealType(1e-3);
  static constexpr RealType pole_y = RealType(1.5707963267948966 - 1e-3);

  KOKKOS_FUNCTION void operator()(int) const {
    d_out(0) = Kokkos::tanh(C(big, 1));
    d_out(1) = Kokkos::tanh(C(-big, -1));
    d_out(2) = Kokkos::tan(C(1, big));
    d_out(3) = Kokkos::tanh(C(small, 1));
    d_out(4) = Kokkos::tanh(C(big, 0));
    d_out(5) = Kokkos::tanh(C(pole_x, pole_y));
  }
  void testit() {
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    // The imaginary parts underflow to (+-)0 for these real parts.
    EXPECT_EQ(h(0), C(1, 0));
    EXPECT_EQ(h(1), C(-1, 0));
    EXPECT_TRUE(h(1).imag() == 0 && std::signbit(h(1).imag()));
    EXPECT_EQ(h(2), C(0, 1));
    const std::complex<RealType> e =
        std::tanh(std::complex<RealType>(small, 1));
    const RealType tol = 8 * Kokkos::Experimental::epsilon_v<RealType>;
    EXPECT_NEAR(h(3).real(), e.real(), tol);
    EXPECT_NEAR(h(3).imag(), e.imag(), tol * std::abs(e.imag()));
    EXPECT_EQ(h(4), C(1, 0));
    const std::complex<RealType> p =
        std::tanh(std::complex<RealType>(pole_x, pole_y));
    EXPECT_NEAR(h(5).real(), p.real(), tol * std::abs(p.real()));
    EXPECT_NEAR(h(5).imag(), p.imag(), tol * std::abs(p.imag()));
  }
};

TEST(TEST_CATEGORY, complex_tanh_large_argument) {
#if __FINITE_MATH_ONLY__
  GTEST_SKIP() << "needs infinities, NaNs and signed zeros, which "
                  "-ffinite-math-only does not provide";
#endif
  TestComplexTanhLarge<float, TEST_EXECSPACE>().testit();
  TestComplexTanhLarge<double, TEST_EXECSPACE>().testit();
}

// The inverse functions on the same grid, compared with std::complex for
// class, sign and value (asin, atan: class and value, and bitwise against
// -i asinh(iz) and -i atanh(iz), whose unspecified signs glibc's casin and
// catan choose differently).
// Also both sides of every branch cut, and a large negative real part, where
// z + sqrt(z^2 + 1) cancelled.  The finite grid values are within 3 eps of
// glibc on the host; the general accuracy of these functions is gap C25.
template <class RealType, class ExecSpace>
struct TestComplexInverseSpecialValues {
  using C                       = Kokkos::complex<RealType>;
  static constexpr int NV       = 9;
  static constexpr int N        = NV * NV;
  static constexpr int NC       = 10;  // branch-cut cases
  static constexpr RealType inf = Kokkos::Experimental::infinity_v<RealType>;
  static constexpr RealType nan = Kokkos::Experimental::quiet_NaN_v<RealType>;

  Kokkos::View<C *, ExecSpace> d_in;
  Kokkos::View<C *[8], ExecSpace> d_out;  // 6 functions, 2 identities
  Kokkos::View<C[NC], ExecSpace> d_cut;
  TestComplexInverseSpecialValues()
      : d_in("in", N), d_out("out", N), d_cut("cut") {}

  KOKKOS_FUNCTION void operator()(int k) const {
    const C z   = d_in(k);
    const C iz  = C(-z.imag(), z.real());
    d_out(k, 0) = Kokkos::asinh(z);
    d_out(k, 1) = Kokkos::acosh(z);
    d_out(k, 2) = Kokkos::atanh(z);
    d_out(k, 3) = Kokkos::asin(z);
    d_out(k, 4) = Kokkos::acos(z);
    d_out(k, 5) = Kokkos::atan(z);
    const C s   = Kokkos::asinh(iz);
    const C t   = Kokkos::atanh(iz);
    d_out(k, 6) = C(s.imag(), -s.real());  // -i asinh(iz)
    d_out(k, 7) = C(t.imag(), -t.real());  // -i atanh(iz)
    if (k == 0) {
      const RealType z0 = 0;
      d_cut(0)          = Kokkos::asin(C(RealType(2.5), z0));
      d_cut(1)          = Kokkos::asin(C(RealType(2.5), -z0));
      d_cut(2)          = Kokkos::acos(C(RealType(2.5), z0));
      d_cut(3)          = Kokkos::acos(C(RealType(2.5), -z0));
      d_cut(4)          = Kokkos::atanh(C(RealType(2), z0));
      d_cut(5)          = Kokkos::atanh(C(RealType(2), -z0));
      d_cut(6)          = Kokkos::acosh(C(RealType(-2), z0));
      d_cut(7)          = Kokkos::acosh(C(RealType(-2), -z0));
      d_cut(8)          = Kokkos::asinh(C(-z0, RealType(2.5)));
      d_cut(9)          = Kokkos::asinh(C(-RealType(1e8), RealType(0.5)));
    }
  }

  void testit() {
    const RealType v[NV] = {
        0, -RealType(0), 1, -1, RealType(0.5), RealType(2.5), inf, -inf, nan};
    auto h_in = Kokkos::create_mirror_view(d_in);
    for (int a = 0; a < NV; ++a)
      for (int b = 0; b < NV; ++b) h_in(a * NV + b) = C(v[a], v[b]);
    Kokkos::deep_copy(d_in, h_in);
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, N), *this);
    auto h_out =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    auto h_cut =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_cut);

    auto kind = [](RealType x) {
      return std::isnan(x) ? 0 : std::isinf(x) ? 1 : x == 0 ? 2 : 3;
    };
    const RealType tol = 16 * Kokkos::Experimental::epsilon_v<RealType>;
    auto check         = [&](RealType x, RealType e, bool with_sign) {
      if (kind(x) != kind(e)) return false;
      if (with_sign && kind(x) != 0 && std::signbit(x) != std::signbit(e))
        return false;
      return kind(x) != 3 || std::abs(x - e) <= tol * std::abs(e);
    };
    auto bitwise = [](C x, C y) {
      auto same = [](RealType a, RealType b) {
        return (std::isnan(a) && std::isnan(b)) ||
               (a == b && std::signbit(a) == std::signbit(b));
      };
      return same(x.real(), y.real()) && same(x.imag(), y.imag());
    };
    const char *names[6] = {"asinh", "acosh", "atanh", "asin", "acos", "atan"};
    for (int k = 0; k < N; ++k) {
      const std::complex<RealType> z(h_in(k).real(), h_in(k).imag());
      const std::complex<RealType> e[6] = {std::asinh(z), std::acosh(z),
                                           std::atanh(z), std::asin(z),
                                           std::acos(z),  std::atan(z)};
      for (int f = 0; f < 6; ++f) {
        const bool with_sign = f != 3 && f != 5;
        const char *ref      = complex_special_ref[9 + f] + 2 * k;
        EXPECT_TRUE(complex_special_check(h_out(k, f).real(), e[f].real(),
                                          ref[0], with_sign, tol) &&
                    complex_special_check(h_out(k, f).imag(), e[f].imag(),
                                          ref[1], with_sign, tol))
            << names[f] << h_in(k) << " = " << h_out(k, f) << ", reference "
            << ref[0] << ref[1] << ", std gives " << e[f];
      }
      EXPECT_TRUE(bitwise(h_out(k, 3), h_out(k, 6)))
          << "asin" << h_in(k) << " differs from -i asinh(iz)";
      EXPECT_TRUE(bitwise(h_out(k, 5), h_out(k, 7)))
          << "atan" << h_in(k) << " differs from -i atanh(iz)";
    }

    // Both sides of each branch cut: the sign of the zero selects the side
    // (continuity from that side, C99 G.6).
    const RealType z0                       = 0;
    const std::complex<RealType> cut_in[NC] = {
        {RealType(2.5), z0},  {RealType(2.5), -z0},
        {RealType(2.5), z0},  {RealType(2.5), -z0},
        {RealType(2), z0},    {RealType(2), -z0},
        {RealType(-2), z0},   {RealType(-2), -z0},
        {-z0, RealType(2.5)}, {-RealType(1e8), RealType(0.5)}};
    const std::complex<RealType> cut_ref[NC] = {
        std::asin(cut_in[0]),  std::asin(cut_in[1]),  std::acos(cut_in[2]),
        std::acos(cut_in[3]),  std::atanh(cut_in[4]), std::atanh(cut_in[5]),
        std::acosh(cut_in[6]), std::acosh(cut_in[7]), std::asinh(cut_in[8]),
        std::asinh(cut_in[9])};
    for (int c = 0; c < NC; ++c) {
      EXPECT_TRUE(check(h_cut(c).real(), cut_ref[c].real(), true) &&
                  check(h_cut(c).imag(), cut_ref[c].imag(), true))
          << "case " << c << ": " << h_cut(c) << ", std gives " << cut_ref[c];
    }
    // The imaginary parts on the two sides have opposite signs.
    EXPECT_TRUE(h_cut(0).imag() > 0 && h_cut(1).imag() < 0);  // asin
    EXPECT_TRUE(h_cut(2).imag() < 0 && h_cut(3).imag() > 0);  // acos
    EXPECT_TRUE(h_cut(4).imag() > 0 && h_cut(5).imag() < 0);  // atanh
    EXPECT_TRUE(h_cut(6).imag() > 0 && h_cut(7).imag() < 0);  // acosh
    EXPECT_TRUE(h_cut(8).real() < 0);                         // asinh
  }
};

TEST(TEST_CATEGORY, complex_inverse_special_values) {
#if __FINITE_MATH_ONLY__
  GTEST_SKIP() << "needs infinities, NaNs and signed zeros, which "
                  "-ffinite-math-only does not provide";
#endif
  TestComplexInverseSpecialValues<float, TEST_EXECSPACE>().testit();
  TestComplexInverseSpecialValues<double, TEST_EXECSPACE>().testit();
}

// Componentwise accuracy of the inverse functions for finite arguments in
// several regimes: ordinary, tiny, huge, and close to the branch points +-1
// and +-i.  The naive log formulas used before were off by up to 1e18 eps
// there (or not finite); the reference is std::complex<long double>.
template <class RealType, class ExecSpace>
struct TestComplexInverseAccuracy {
  using C                = Kokkos::complex<RealType>;
  static constexpr int R = 64;  // inputs per regime
  static constexpr int N = 5 * R + 1;

  Kokkos::View<C *, ExecSpace> d_in;
  Kokkos::View<C *[6], ExecSpace> d_out;
  TestComplexInverseAccuracy() : d_in("in", N), d_out("out", N) {}

  KOKKOS_FUNCTION void operator()(int k) const {
    const C z   = d_in(k);
    d_out(k, 0) = Kokkos::asinh(z);
    d_out(k, 1) = Kokkos::acosh(z);
    d_out(k, 2) = Kokkos::atanh(z);
    d_out(k, 3) = Kokkos::asin(z);
    d_out(k, 4) = Kokkos::acos(z);
    d_out(k, 5) = Kokkos::atan(z);
  }

  void testit() {
    const bool f32 = sizeof(RealType) == 4;
    std::mt19937_64 g(2026);
    std::uniform_real_distribution<double> u(-1, 1), off(-15, -1);
    auto h_in = Kokkos::create_mirror_view(d_in);
    for (int t = 0; t < R; ++t) {
      h_in(t)     = C(RealType(2 * u(g)), RealType(2 * u(g)));  // |x|, |y| < 2
      h_in(R + t) = C(RealType(u(g) * 1e-30), RealType(u(g) * 1e-30));  // tiny
      const RealType huge = f32 ? RealType(1e37) : RealType(1e300);
      h_in(2 * R + t)     = C(huge * RealType(u(g)), huge * RealType(u(g)));
      h_in(3 * R + t) = C(RealType(1e8 * u(g)), RealType(1e8 * u(g)));  // large
      // Close to a branch point: offsets 10^-15 .. 10^-1 (float: 10^-7 ..).
      const double bp[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      const double scale    = f32 ? 0.45 : 1;
      h_in(4 * R + t) =
          C(RealType(bp[t % 4][0] + u(g) * std::pow(10., off(g) * scale)),
            RealType(bp[t % 4][1] + u(g) * std::pow(10., off(g) * scale)));
    }
    // Where a too-early small-y approximation was off by 1e15 eps.
    h_in(N - 1) = f32 ? C(RealType(0.99999994), RealType(1e-7))
                      : C(0.99999999999999989, 1.9649419124143349e-16);
    Kokkos::deep_copy(d_in, h_in);
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, N), *this);
    auto h_out =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);

    using L = std::complex<long double>;
    // A few ulps on the host; device math functions may add an ulp or two.
    const long double tol = 8 * Kokkos::Experimental::epsilon_v<RealType>;
    auto rel_ok           = [&](RealType v, long double e) {
      const RealType et = static_cast<RealType>(e);
      if (et == 0 || !std::isfinite(et) ||
          std::abs(et) < Kokkos::Experimental::norm_min_v<RealType>)
        return true;  // the reference is not representable accurately
      return std::abs(v - e) <= tol * std::abs(e);
    };
    const char *names[6] = {"asinh", "acosh", "atanh", "asin", "acos", "atan"};
    for (int k = 0; k < N; ++k) {
      const L z(h_in(k).real(), h_in(k).imag());
      const L e[6] = {std::asinh(z), std::acosh(z), std::atanh(z),
                      std::asin(z),  std::acos(z),  std::atan(z)};
      for (int f = 0; f < 6; ++f) {
        EXPECT_TRUE(rel_ok(h_out(k, f).real(), e[f].real()) &&
                    rel_ok(h_out(k, f).imag(), e[f].imag()))
            << std::setprecision(17) << names[f] << h_in(k) << " = "
            << h_out(k, f) << ", reference ("
            << static_cast<double>(e[f].real()) << ","
            << static_cast<double>(e[f].imag()) << ")";
      }
    }
  }
};

TEST(TEST_CATEGORY, complex_inverse_accuracy) {
  TestComplexInverseAccuracy<float, TEST_EXECSPACE>().testit();
  TestComplexInverseAccuracy<double, TEST_EXECSPACE>().testit();
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

// half_t and bhalf_t cannot be used in constant expressions on the HIP and
// SYCL backends: their conversions, comparisons and arithmetic are not
// constexpr. Run the checks above at runtime there (and everywhere, as a
// run-time check); on the std backend and on CUDA they also run at compile
// time (CH5, below).
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

// The non-member arithmetic operators can appear in constant expressions.
// Division by a complex number is tested below.  All results are exact in
// float and double.
template <class T>
constexpr bool test_arithmetic_in_constant_expression() {
  using C = Kokkos::complex<T>;
  constexpr C a(T(1), T(2));
  constexpr C b(T(3), T(4));

  static_assert(+a == a);
  static_assert(-a == C(T(-1), T(-2)));

  static_assert(a + b == C(T(4), T(6)));
  static_assert(a + T(1) == C(T(2), T(2)));
  static_assert(T(1) + a == C(T(2), T(2)));

  static_assert(a - b == C(T(-2), T(-2)));
  static_assert(a - T(1) == C(T(0), T(2)));
  static_assert(T(1) - a == C(T(0), T(-2)));

  static_assert(a * b == C(T(-5), T(10)));
  static_assert(a * T(2) == C(T(2), T(4)));
  static_assert(T(2) * a == C(T(2), T(4)));

  static_assert(a / T(2) == C(T(0.5), T(1)));

  return true;
}
static_assert(test_arithmetic_in_constant_expression<float>());
static_assert(test_arithmetic_in_constant_expression<double>());

// Mixed precision and mixed operand types (a Kokkos extension).
static_assert(Kokkos::complex<float>(1, 2) * Kokkos::complex<double>(3, 4) ==
              Kokkos::complex<double>(-5, 10));
static_assert(std::is_same_v<decltype(Kokkos::complex<float>(1, 2) +
                                      Kokkos::complex<double>(3, 4)),
                             Kokkos::complex<double>>);
static_assert(2 * Kokkos::complex<double>(1, 2) ==
              Kokkos::complex<double>(2, 4));
static_assert(Kokkos::complex<float>(1, 2) - 1.0 ==
              Kokkos::complex<double>(0, 2));

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
// std::complex times Kokkos::complex (host only, deprecated).
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
static_assert(std::complex<double>(1, 2) * Kokkos::complex<double>(3, 4) ==
              Kokkos::complex<double>(-5, 10));
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif

// Division by a complex number in constant expressions.  Only finite operands
// and a nonzero divisor are constant expressions (see Impl::complex_divide).
template <class T>
constexpr bool test_division_in_constant_expression() {
  using C = Kokkos::complex<T>;
  // Both branches of Smith's algorithm; exact results.
  static_assert(C(T(-5), T(10)) / C(T(3), T(4)) == C(T(1), T(2)));  // |c|<|d|
  static_assert(C(T(-2), T(11)) / C(T(4), T(3)) == C(T(1), T(2)));  // |c|>=|d|
  static_assert(T(25) / C(T(3), T(4)) == C(T(3), T(-4)));

  C z(T(-2), T(11));
  z /= C(T(4), T(3));
  if (z != C(T(1), T(2))) return false;

  // Range guards: halving near the largest value, scaling a divisor at the
  // smallest normal value: (1 + i) / (m (4 + 2i)) = (0.3 + 0.1i) / m.
  constexpr T big = Kokkos::Experimental::finite_max_v<T> / 2;
  static_assert(C(big, big) / C(big, big) == C(T(1), T(0)));
  constexpr T m = Kokkos::Experimental::norm_min_v<T>;
  constexpr C r = C(T(1), T(1)) / C(4 * m, 2 * m);
  static_assert(r.real() * m > T(0.2999) && r.real() * m < T(0.3001));
  static_assert(r.imag() * m > T(0.0999) && r.imag() * m < T(0.1001));
  return true;
}
static_assert(test_division_in_constant_expression<float>());
static_assert(test_division_in_constant_expression<double>());
static_assert(Kokkos::complex<float>(-5, 10) / Kokkos::complex<double>(3, 4) ==
              Kokkos::complex<double>(1, 2));

// constexpr complex<half_t> and complex<bhalf_t> (CH5). On the std backend
// (C++23 std::float16_t / std::bfloat16_t; B4a) and on CUDA (B4b-CUDA) half
// values are constant expressions, so the checks above run for the half types
// too. The HIP and SYCL half types are not constexpr (future work): there only
// the run-time twins below run.
#ifdef KOKKOS_IMPL_HALF_IS_CONSTEXPR
// Division as above, with tolerances for the half types: the range checks of
// test_division_in_constant_expression are tighter than half precision.
template <class T>
constexpr bool test_half_division_in_constant_expression() {
  using C = Kokkos::complex<T>;
  static_assert(C(T(-5), T(10)) / C(T(3), T(4)) == C(T(1), T(2)));  // |c|<|d|
  static_assert(C(T(-2), T(11)) / C(T(4), T(3)) == C(T(1), T(2)));  // |c|>=|d|
  static_assert(T(25) / C(T(3), T(4)) == C(T(3), T(-4)));

  C z(T(-2), T(11));
  z /= C(T(4), T(3));
  if (z != C(T(1), T(2))) return false;

  // The largest and the smallest normal values: half division computes in
  // float (CH3), so neither overflows nor underflows.
  constexpr T big(Kokkos::Experimental::finite_max_v<T>);
  static_assert(C(big, big) / C(big, big) == C(T(1), T(0)));
  constexpr T m(Kokkos::Experimental::norm_min_v<T>);
  constexpr C r = C(T(1), T(1)) / C(T(4) * m, T(2) * m);  // (0.3 + 0.1i) / m
  constexpr float e =
      2 * static_cast<float>(T(Kokkos::Experimental::epsilon_v<T>));
  constexpr float re = static_cast<float>(r.real()) * static_cast<float>(m);
  constexpr float im = static_cast<float>(r.imag()) * static_cast<float>(m);
  static_assert(re > 0.3f * (1 - e) && re < 0.3f * (1 + e));
  static_assert(im > 0.1f * (1 - e) && im < 0.1f * (1 + e));
  return true;
}

template <class T>
constexpr bool test_half_complex_in_constant_expression() {
  static_assert(can_appear_in_constant_expressions<T>());
  static_assert(comparison_in_constant_expression<T>());
  static_assert(test_complex_norm<T>());
  static_assert(test_complex_conj<T>());
  static_assert(test_arithmetic_in_constant_expression<T>());
  static_assert(test_half_division_in_constant_expression<T>());
  // Conversions between the complex types (CH2) in constant expressions.
  using CH = Kokkos::complex<T>;
  static_assert(Kokkos::complex<float>(CH(T(1.5), T(-2))) ==
                Kokkos::complex<float>(1.5f, -2.f));
  static_assert(CH(Kokkos::complex<double>(0.5, 4)) == CH(T(0.5), T(4)));
  return true;
}
#if !KOKKOS_HALF_T_IS_FLOAT
static_assert(
    test_half_complex_in_constant_expression<Kokkos::Experimental::half_t>());
#endif
#if !KOKKOS_BHALF_T_IS_FLOAT
static_assert(
    test_half_complex_in_constant_expression<Kokkos::Experimental::bhalf_t>());
#endif
#endif  // KOKKOS_IMPL_HALF_IS_CONSTEXPR

// Compile-time and run-time division agree (up to contraction at run time).
TEST(TEST_CATEGORY, complex_division_constexpr_matches_runtime) {
  using C           = Kokkos::complex<double>;
  constexpr C q     = C(1.5, 2.5) / C(3.25, 5.75);
  volatile double a = 1.5, b = 2.5, c = 3.25, d = 5.75;
  // Read through volatile so that the division is computed at run time.
  const C r = C(double(a), double(b)) / C(double(c), double(d));
  const double tol =
      2 * Kokkos::Experimental::epsilon_v<double> * Kokkos::abs(q);
  EXPECT_NEAR(q.real(), r.real(), tol);
  EXPECT_NEAR(q.imag(), r.imag(), tol);
}

// half_t and bhalf_t promote to themselves in the scalar overloads, as
// std::float16_t and std::bfloat16_t do, and combine with other types through
// std::common_type: half + float is float, an integer counts as double, and
// half_t and bhalf_t do not combine.
template <class T, class U>
concept has_promote_2 = requires { typename Kokkos::Impl::promote_2_t<T, U>; };
template <class T>
concept has_scalar_conj = requires(T x) { Kokkos::conj(x); };
template <class T>
concept has_scalar_proj = requires(T x) { Kokkos::proj(x); };

template <class H>
void test_half_scalar_promotion() {
  using Kokkos::Impl::promote_2_t;
  using Kokkos::Impl::promote_3_t;
  using Kokkos::Impl::promote_t;
  using CF = Kokkos::complex<float>;
  using CD = Kokkos::complex<double>;
  static_assert(std::is_same_v<promote_t<H>, H>);
  static_assert(std::is_same_v<promote_2_t<H, H>, H>);
  static_assert(std::is_same_v<promote_2_t<H, float>, float>);
  static_assert(std::is_same_v<promote_2_t<float, H>, float>);
  static_assert(std::is_same_v<promote_2_t<H, double>, double>);
  static_assert(std::is_same_v<promote_2_t<H, int>, double>);
  static_assert(std::is_same_v<promote_2_t<long, H>, double>);
  static_assert(std::is_same_v<promote_3_t<H, H, float>, float>);
  // Unchanged for the arithmetic types.
  static_assert(std::is_same_v<promote_2_t<float, float>, float>);
  static_assert(std::is_same_v<promote_2_t<float, int>, double>);
  static_assert(std::is_same_v<promote_2_t<int, int>, double>);
  static_assert(std::is_same_v<promote_2_t<float, long double>, long double>);
  static_assert(!has_promote_2<CF, float>);
  static_assert(has_scalar_conj<float> && has_scalar_proj<int>);

  static_assert(std::is_same_v<decltype(Kokkos::real(H())), H>);
  static_assert(std::is_same_v<decltype(Kokkos::imag(H())), H>);
  static_assert(std::is_same_v<decltype(Kokkos::norm(H())), H>);
  static_assert(std::is_same_v<decltype(Kokkos::arg(H())), H>);
  static_assert(std::is_same_v<decltype(Kokkos::pow(CF(), H())), CF>);
  static_assert(std::is_same_v<decltype(Kokkos::pow(H(), CF())), CF>);
  static_assert(std::is_same_v<decltype(Kokkos::pow(CD(), H())), CD>);
  static_assert(std::is_same_v<decltype(Kokkos::pow(CF(), CD())), CD>);

  EXPECT_EQ(static_cast<float>(Kokkos::norm(H(3.f))), 9.f);
  EXPECT_EQ(static_cast<float>(Kokkos::arg(H(2.f))), 0.f);
  const float pi = Kokkos::numbers::pi_v<float>;
  const float tol =
      2 * static_cast<float>(H(Kokkos::Experimental::epsilon_v<H>));
  EXPECT_NEAR(static_cast<float>(Kokkos::arg(H(-1.f))), pi, tol * pi);
  EXPECT_NEAR(static_cast<float>(Kokkos::arg(H(-0.f))), pi, tol * pi);
  const CF p = Kokkos::pow(CF(0, 1), H(2.f));
  EXPECT_NEAR(p.real(), -1.f, 1e-6f);
  EXPECT_NEAR(p.imag(), 0.f, 1e-6f);
  const CF q = Kokkos::pow(H(2.f), CF(3, 0));
  EXPECT_NEAR(q.real(), 8.f, 1e-5f);
  EXPECT_NEAR(q.imag(), 0.f, 1e-5f);

  // The same in a kernel.
  int errors = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
      KOKKOS_LAMBDA(int, int &err) {
        if (!(static_cast<float>(Kokkos::norm(H(3.f))) == 9.f)) ++err;
        const float a = static_cast<float>(Kokkos::arg(H(-1.f)));
        if (!(Kokkos::fabs(a - pi) <= tol * pi)) ++err;
        const CF r = Kokkos::pow(CF(0, 1), H(2.f));
        if (!(Kokkos::fabs(r.real() + 1.f) <= 1e-6f &&
              Kokkos::fabs(r.imag()) <= 1e-6f))
          ++err;
      },
      errors);
  EXPECT_EQ(errors, 0);
}

TEST(TEST_CATEGORY, complex_half_scalar_promotion) {
  using Kokkos::Experimental::bhalf_t;
  using Kokkos::Experimental::half_t;
  test_half_scalar_promotion<half_t>();
  test_half_scalar_promotion<bhalf_t>();
  // complex<half_t> exists, so the scalar conj and proj return it.
  static_assert(std::is_same_v<decltype(Kokkos::conj(half_t())),
                               Kokkos::complex<half_t>>);
  static_assert(std::is_same_v<decltype(Kokkos::proj(bhalf_t())),
                               Kokkos::complex<bhalf_t>>);
  // ... and they stay absent for a type complex cannot hold.
  static_assert(!has_scalar_conj<const char *>);
  static_assert(!has_scalar_proj<const char *>);
#if !KOKKOS_HALF_T_IS_FLOAT && !KOKKOS_BHALF_T_IS_FLOAT
  static_assert(!has_promote_2<half_t, bhalf_t>);
#endif
}

// Conversions between complex types follow std::complex for the half types
// (P1467): explicit into half_t and bhalf_t, implicit out of them, none
// between half_t and bhalf_t.  Assignment works in every direction where a
// conversion exists, as for std::complex.
template <class H>
void test_complex_half_conversions() {
  using CH = Kokkos::complex<H>;
  using CF = Kokkos::complex<float>;
  using CD = Kokkos::complex<double>;
  static_assert(!std::is_convertible_v<CD, CH>);
  static_assert(!std::is_convertible_v<CF, CH>);
  static_assert(std::is_constructible_v<CH, CD>);
  static_assert(std::is_constructible_v<CH, CF>);
  static_assert(std::is_convertible_v<CH, CF>);
  static_assert(std::is_convertible_v<CH, CD>);
  static_assert(std::is_assignable_v<CH &, CD>);
  static_assert(std::is_assignable_v<CF &, CH>);

  // Into half: each part is rounded once.  Out of half: exact.
  const CH n(CD(0.1, -1.0 / 3));
  EXPECT_EQ(n.real(), H(0.1));
  EXPECT_EQ(n.imag(), H(-1.0 / 3));
  const CF w = n;
  EXPECT_EQ(w.real(), static_cast<float>(n.real()));
  EXPECT_EQ(w.imag(), static_cast<float>(n.imag()));
  CH a;
  a = CD(0.1, -1.0 / 3);
  EXPECT_EQ(a, n);
  CD b;
  b = n;
  EXPECT_EQ(b, CD(w));

  // Mixed pow converts the half operand to complex<float>.
  static_assert(std::is_same_v<decltype(Kokkos::pow(CH(), CF())), CF>);
  const CF p = Kokkos::pow(CH(H(0.f), H(1.f)), CF(2, 0));
  EXPECT_NEAR(p.real(), -1.f, 1e-6f);
  EXPECT_NEAR(p.imag(), 0.f, 1e-6f);

  // The same in a kernel.
  int errors = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
      KOKKOS_LAMBDA(int, int &err) {
        const CH m(CD(0.1, -1.0 / 3));
        if (!(m.real() == H(0.1) && m.imag() == H(-1.0 / 3))) ++err;
        const CF v = m;
        if (!(v.real() == static_cast<float>(m.real()))) ++err;
        CH c;
        c = CD(0.1, -1.0 / 3);
        if (!(c == m)) ++err;
      },
      errors);
  EXPECT_EQ(errors, 0);
}

// Between the standard types the conversion stays implicit in both
// directions (std::complex<float>(std::complex<double>) is explicit; a known
// divergence), and it is now constexpr.
static_assert(
    std::is_convertible_v<Kokkos::complex<double>, Kokkos::complex<float>>);
static_assert(
    std::is_convertible_v<Kokkos::complex<float>, Kokkos::complex<double>>);
static_assert(Kokkos::complex<double>(Kokkos::complex<float>(1.5f, 2.5f)) ==
              Kokkos::complex<double>(1.5, 2.5));
static_assert(Kokkos::complex<float>(Kokkos::complex<double>(0.5, -2)) ==
              Kokkos::complex<float>(0.5f, -2.f));
#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE) && \
    defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
static_assert(
    !std::is_constructible_v<Kokkos::complex<Kokkos::Experimental::half_t>,
                             Kokkos::complex<Kokkos::Experimental::bhalf_t>>);
static_assert(
    !std::is_constructible_v<Kokkos::complex<Kokkos::Experimental::bhalf_t>,
                             Kokkos::complex<Kokkos::Experimental::half_t>>);
static_assert(
    !std::is_assignable_v<Kokkos::complex<Kokkos::Experimental::half_t> &,
                          Kokkos::complex<Kokkos::Experimental::bhalf_t>>);
#endif

TEST(TEST_CATEGORY, complex_half_conversions) {
#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  test_complex_half_conversions<Kokkos::Experimental::half_t>();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  test_complex_half_conversions<Kokkos::Experimental::bhalf_t>();
#endif
}

// complex<half_t> and complex<bhalf_t> arithmetic and functions are computed
// in float and rounded once (D5): the result is exactly that of complex<float>
// on the same values, rounded to the half type.
template <class H>
KOKKOS_FUNCTION Kokkos::complex<H> half_compute_op(int k, Kokkos::complex<H> x,
                                                   Kokkos::complex<H> y) {
  switch (k) {
    case 0: return x * y;
    case 1: return x / y;
    case 2: {
      Kokkos::complex<H> z = x;
      z *= y;
      return z;
    }
    case 3: {
      Kokkos::complex<H> z = x;
      z /= y;
      return z;
    }
    case 4: return Kokkos::exp(x);
    case 5: return Kokkos::log(x);
    case 6: return Kokkos::log10(x);
    case 7: return Kokkos::sqrt(x);
    case 8: return Kokkos::sin(x);
    case 9: return Kokkos::cos(x);
    case 10: return Kokkos::tan(x);
    case 11: return Kokkos::sinh(x);
    case 12: return Kokkos::cosh(x);
    case 13: return Kokkos::tanh(x);
    case 14: return Kokkos::asin(x);
    case 15: return Kokkos::acos(x);
    case 16: return Kokkos::atan(x);
    case 17: return Kokkos::asinh(x);
    case 18: return Kokkos::acosh(x);
    case 19: return Kokkos::atanh(x);
    case 20: return Kokkos::pow(x, y);
    default: return Kokkos::pow(x, y.real());
  }
}

template <class H>
KOKKOS_FUNCTION Kokkos::complex<H> half_compute_op_via_float(
    int k, Kokkos::complex<H> x, Kokkos::complex<H> y) {
  using CF = Kokkos::complex<float>;
  const CF xf(x);
  const CF yf(y);
  CF r;
  switch (k) {
    case 0:
    case 2: r = xf * yf; break;
    case 1:
    case 3: r = xf / yf; break;
    case 4: r = Kokkos::exp(xf); break;
    case 5: r = Kokkos::log(xf); break;
    case 6: r = Kokkos::log10(xf); break;
    case 7: r = Kokkos::sqrt(xf); break;
    case 8: r = Kokkos::sin(xf); break;
    case 9: r = Kokkos::cos(xf); break;
    case 10: r = Kokkos::tan(xf); break;
    case 11: r = Kokkos::sinh(xf); break;
    case 12: r = Kokkos::cosh(xf); break;
    case 13: r = Kokkos::tanh(xf); break;
    case 14: r = Kokkos::asin(xf); break;
    case 15: r = Kokkos::acos(xf); break;
    case 16: r = Kokkos::atan(xf); break;
    case 17: r = Kokkos::asinh(xf); break;
    case 18: r = Kokkos::acosh(xf); break;
    case 19: r = Kokkos::atanh(xf); break;
    case 20: r = Kokkos::pow(xf, yf); break;
    default: r = Kokkos::pow(xf, yf.real()); break;
  }
  return Kokkos::complex<H>(r);
}

// Same bits, or both NaN.
template <class H>
KOKKOS_FUNCTION bool half_same(H a, H b) {
  return (a != a && b != b) || Kokkos::bit_cast<std::uint16_t>(a) ==
                                   Kokkos::bit_cast<std::uint16_t>(b);
}

template <class H>
void test_complex_half_compute_in_float() {
  using CH           = Kokkos::complex<H>;
  using CD           = Kokkos::complex<double>;
  constexpr int nops = 22;
  // Ordinary values, values whose products or squares overflow H, and values
  // near the branch cuts.
  const CH vals[]  = {CH(H(1.5f), H(-0.5f)),      CH(H(0.25f), H(2.f)),
                      CH(H(269.f), H(111.5f)),    CH(H(-3.f), H(0.125f)),
                      CH(H(16384.f), H(-8192.f)), CH(H(0.75f), H(-0.f)),
                      CH(H(-2.f), H(0.f)),        CH(H(1e-3f), H(-4.f))};
  constexpr int nv = sizeof(vals) / sizeof(vals[0]);
  Kokkos::View<CH *[2], TEST_EXECSPACE> in("in", nv * nv);
  Kokkos::View<int[1], TEST_EXECSPACE> errors("errors");
  auto h_in = Kokkos::create_mirror_view(in);
  for (int i = 0; i < nv; ++i) {
    for (int j = 0; j < nv; ++j) {
      h_in(i * nv + j, 0) = vals[i];
      h_in(i * nv + j, 1) = vals[j];
    }
  }
  Kokkos::deep_copy(in, h_in);
  int host_errors = 0;
  for (int i = 0; i < nv * nv; ++i) {
    for (int k = 0; k < nops; ++k) {
      const CH r = half_compute_op<H>(k, h_in(i, 0), h_in(i, 1));
      const CH e = half_compute_op_via_float<H>(k, h_in(i, 0), h_in(i, 1));
      if (!half_same(r.real(), e.real()) || !half_same(r.imag(), e.imag())) {
        ++host_errors;
        ADD_FAILURE() << "op " << k << " on " << h_in(i, 0) << ", "
                      << h_in(i, 1) << ": " << r << " vs " << e;
      }
    }
  }
  EXPECT_EQ(host_errors, 0);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, nv * nv), KOKKOS_LAMBDA(int i) {
        for (int k = 0; k < nops; ++k) {
          const CH r = half_compute_op<H>(k, in(i, 0), in(i, 1));
          const CH e = half_compute_op_via_float<H>(k, in(i, 0), in(i, 1));
          if (!half_same(r.real(), e.real()) || !half_same(r.imag(), e.imag()))
            Kokkos::atomic_inc(&errors(0));
        }
      });
  auto h_errors =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
  EXPECT_EQ(h_errors(0), 0);

  // Intermediate overflow: 269^2 = 72361 overflows half_t, but the product
  // (59928.75, 59987) does not; it is the exact result rounded once.
  const CH z(H(269.f), H(111.5f));
  EXPECT_EQ(z * z, CH(CD(z) * CD(z)));
  // a c + b d = 120000 overflows half_t, the quotient does not.
  EXPECT_EQ(CH(H(60000.f), H(60000.f)) / CH(H(1.f), H(1.f)),
            CH(H(60000.f), H(0.f)));
#if !__FINITE_MATH_ONLY__
  // Special values (C99 Annex G) are kept.
  const H inf = H(Kokkos::Experimental::infinity_v<float>);
  EXPECT_EQ(CH(H(1.f), H(1.f)) / CH(H(0.f), H(0.f)), CH(inf, inf));
  EXPECT_EQ(CH(H(1.f), H(1.f)) / CH(inf, H(0.f)), CH(H(0.f), H(0.f)));
  // (inf, NaN) * (1, 0) is (inf, NaN), not (NaN, NaN).
  const CH m =
      CH(inf, H(Kokkos::Experimental::quiet_NaN_v<float>)) * CH(H(1.f), H(0.f));
  EXPECT_EQ(m.real(), inf);
  EXPECT_TRUE(Kokkos::isnan(static_cast<float>(m.imag())));
#endif
  // asin of a large argument: no spurious overflow in the squares.
  const CH a = Kokkos::asin(CH(H(16384.f), H(16384.f)));
  EXPECT_TRUE(Kokkos::isfinite(static_cast<float>(a.real())) &&
              Kokkos::isfinite(static_cast<float>(a.imag())))
      << a;
}

TEST(TEST_CATEGORY, complex_half_compute_in_float) {
#if defined(KOKKOS_HAS_NATIVE_HALF_TYPE)
  test_complex_half_compute_in_float<Kokkos::Experimental::half_t>();
#endif
#if defined(KOKKOS_HAS_NATIVE_BHALF_TYPE)
  test_complex_half_compute_in_float<Kokkos::Experimental::bhalf_t>();
  // bhalf_t has float's range: the squares in float do not overflow either.
  using CB = Kokkos::complex<Kokkos::Experimental::bhalf_t>;
  using Kokkos::Experimental::bhalf_t;
  EXPECT_EQ(
      CB(bhalf_t(1e38f), bhalf_t(1e38f)) / CB(bhalf_t(1e38f), bhalf_t(1e38f)),
      CB(bhalf_t(1.f), bhalf_t(0.f)));
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
