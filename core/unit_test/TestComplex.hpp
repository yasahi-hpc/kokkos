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

template <class ExecSpace>
struct TestComplexConstruction {
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<double> *, ExecSpace>::host_mirror_type
      h_results;

  void testit() {
    d_results = Kokkos::View<Kokkos::complex<double> *, ExecSpace>(
        "TestComplexConstruction", 10);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    ASSERT_FLOAT_EQ(h_results(0).real(), 1.5);
    ASSERT_FLOAT_EQ(h_results(0).imag(), 2.5);
    ASSERT_FLOAT_EQ(h_results(1).real(), 1.5);
    ASSERT_FLOAT_EQ(h_results(1).imag(), 2.5);
    ASSERT_FLOAT_EQ(h_results(2).real(), 0.0);
    ASSERT_FLOAT_EQ(h_results(2).imag(), 0.0);
    ASSERT_FLOAT_EQ(h_results(3).real(), 3.5);
    ASSERT_FLOAT_EQ(h_results(3).imag(), 0.0);
    ASSERT_FLOAT_EQ(h_results(4).real(), 4.5);
    ASSERT_FLOAT_EQ(h_results(4).imag(), 5.5);
    ASSERT_FLOAT_EQ(h_results(5).real(), 1.5);
    ASSERT_FLOAT_EQ(h_results(5).imag(), 2.5);
    ASSERT_FLOAT_EQ(h_results(6).real(), 4.5);
    ASSERT_FLOAT_EQ(h_results(6).imag(), 5.5);
    ASSERT_FLOAT_EQ(h_results(7).real(), 7.5);
    ASSERT_FLOAT_EQ(h_results(7).imag(), 0.0);
    ASSERT_FLOAT_EQ(h_results(8).real(), double(8));
    ASSERT_FLOAT_EQ(h_results(8).imag(), 0.0);

    // Copy construction conversion between
    // Kokkos::complex and std::complex doesn't compile
    Kokkos::complex<double> a(1.5, 2.5), b(3.25, 5.25), r_kk;
    std::complex<double> sa(a), sb(3.25, 5.25), r;
    r    = a;
    r_kk = a;
    ASSERT_FLOAT_EQ(r.real(), r_kk.real());
    ASSERT_FLOAT_EQ(r.imag(), r_kk.imag());
    r    = sb * a;
    r_kk = b * a;
    ASSERT_FLOAT_EQ(r.real(), r_kk.real());
    ASSERT_FLOAT_EQ(r.imag(), r_kk.imag());
    r    = sa;
    r_kk = a;
    ASSERT_FLOAT_EQ(r.real(), r_kk.real());
    ASSERT_FLOAT_EQ(r.imag(), r_kk.imag());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    Kokkos::complex<double> a(1.5, 2.5);
    d_results(0) = a;
    Kokkos::complex<double> b(a);
    d_results(1)              = b;
    Kokkos::complex<double> c = Kokkos::complex<double>();
    d_results(2)              = c;
    Kokkos::complex<double> d(3.5);
    d_results(3) = d;
    Kokkos::complex<double> a_v(4.5, 5.5);
    d_results(4) = a_v;
    Kokkos::complex<double> b_v(a);
    d_results(5) = b_v;
    Kokkos::complex<double> e(a_v);
    d_results(6) = e;

    d_results(7) = double(7.5);
    d_results(8) = int(8);
  }
};

TEST(TEST_CATEGORY, complex_construction) {
  TestComplexConstruction<TEST_EXECSPACE> test;
  test.testit();
}

// Test Math FUnction

template <class ExecSpace>
struct TestComplexBasicMath {
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<double> *, ExecSpace>::host_mirror_type
      h_results;

  void testit() {
    d_results = Kokkos::View<Kokkos::complex<double> *, ExecSpace>(
        "TestComplexBasicMath", 24);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    std::complex<double> a(1.5, 2.5);
    std::complex<double> b(3.25, 5.75);
    std::complex<double> d(1.0, 2.0);
    double c = 9.3;
    int e    = 2;

    std::complex<double> r;
    r = a + b;
    ASSERT_FLOAT_EQ(h_results(0).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(0).imag(), r.imag());
    r = a - b;
    ASSERT_FLOAT_EQ(h_results(1).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(1).imag(), r.imag());
    r = a * b;
    ASSERT_FLOAT_EQ(h_results(2).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(2).imag(), r.imag());
    r = a / b;
    ASSERT_FLOAT_EQ(h_results(3).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(3).imag(), r.imag());
    r = d + a;
    ASSERT_FLOAT_EQ(h_results(4).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(4).imag(), r.imag());
    r = d - a;
    ASSERT_FLOAT_EQ(h_results(5).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(5).imag(), r.imag());
    r = d * a;
    ASSERT_FLOAT_EQ(h_results(6).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(6).imag(), r.imag());
    r = d / a;
    ASSERT_FLOAT_EQ(h_results(7).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(7).imag(), r.imag());
    r = a + c;
    ASSERT_FLOAT_EQ(h_results(8).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(8).imag(), r.imag());
    r = a - c;
    ASSERT_FLOAT_EQ(h_results(9).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(9).imag(), r.imag());
    r = a * c;
    ASSERT_FLOAT_EQ(h_results(10).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(10).imag(), r.imag());
    r = a / c;
    ASSERT_FLOAT_EQ(h_results(11).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(11).imag(), r.imag());
    r = d + c;
    ASSERT_FLOAT_EQ(h_results(12).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(12).imag(), r.imag());
    r = d - c;
    ASSERT_FLOAT_EQ(h_results(13).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(13).imag(), r.imag());
    r = d * c;
    ASSERT_FLOAT_EQ(h_results(14).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(14).imag(), r.imag());
    r = d / c;
    ASSERT_FLOAT_EQ(h_results(15).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(15).imag(), r.imag());
    r = c + a;
    ASSERT_FLOAT_EQ(h_results(16).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(16).imag(), r.imag());
    r = c - a;
    ASSERT_FLOAT_EQ(h_results(17).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(17).imag(), r.imag());
    r = c * a;
    ASSERT_FLOAT_EQ(h_results(18).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(18).imag(), r.imag());
    r = c / a;
    ASSERT_FLOAT_EQ(h_results(19).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(19).imag(), r.imag());
    r = a;
    /* r = a+e; */ ASSERT_FLOAT_EQ(h_results(20).real(), r.real() + e);
    ASSERT_FLOAT_EQ(h_results(20).imag(), r.imag());
    /* r = a-e; */ ASSERT_FLOAT_EQ(h_results(21).real(), r.real() - e);
    ASSERT_FLOAT_EQ(h_results(21).imag(), r.imag());
    /* r = a*e; */ ASSERT_FLOAT_EQ(h_results(22).real(), r.real() * e);
    ASSERT_FLOAT_EQ(h_results(22).imag(), r.imag() * e);
    /* r = a/e; */ ASSERT_FLOAT_EQ(h_results(23).real(), r.real() / 2);
    ASSERT_FLOAT_EQ(h_results(23).imag(), r.imag() / e);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    Kokkos::complex<double> a(1.5, 2.5);
    Kokkos::complex<double> b(3.25, 5.75);
    // Basic math complex / complex
    d_results(0) = a + b;
    d_results(1) = a - b;
    d_results(2) = a * b;
    d_results(3) = a / b;
    d_results(4).real(1.0);
    d_results(4).imag(2.0);
    d_results(4) += a;
    d_results(5) = Kokkos::complex<double>(1.0, 2.0);
    d_results(5) -= a;
    d_results(6) = Kokkos::complex<double>(1.0, 2.0);
    d_results(6) *= a;
    d_results(7) = Kokkos::complex<double>(1.0, 2.0);
    d_results(7) /= a;

    // Basic math complex / scalar
    double c      = 9.3;
    d_results(8)  = a + c;
    d_results(9)  = a - c;
    d_results(10) = a * c;
    d_results(11) = a / c;
    d_results(12).real(1.0);
    d_results(12).imag(2.0);
    d_results(12) += c;
    d_results(13) = Kokkos::complex<double>(1.0, 2.0);
    d_results(13) -= c;
    d_results(14) = Kokkos::complex<double>(1.0, 2.0);
    d_results(14) *= c;
    d_results(15) = Kokkos::complex<double>(1.0, 2.0);
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
  TestComplexBasicMath<TEST_EXECSPACE> test;
  test.testit();
}

template <class ExecSpace>
struct TestComplexSpecialFunctions {
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<double> *, ExecSpace>::host_mirror_type
      h_results;

  void testit() {
    d_results = Kokkos::View<Kokkos::complex<double> *, ExecSpace>(
        "TestComplexSpecialFunctions", 20);
    h_results = Kokkos::create_mirror_view(d_results);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results, d_results);

    std::complex<double> a(1.5, 2.5);
    double c = 9.3;

    std::complex<double> r;
    r = a;
    ASSERT_FLOAT_EQ(h_results(0).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(0).imag(), r.imag());
    r = std::sqrt(a);
    ASSERT_FLOAT_EQ(h_results(1).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(1).imag(), r.imag());
    r = std::pow(a, c);
    ASSERT_FLOAT_EQ(h_results(2).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(2).imag(), r.imag());
    r = std::abs(a);
    ASSERT_FLOAT_EQ(h_results(3).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(3).imag(), r.imag());
    r = std::exp(a);
    ASSERT_FLOAT_EQ(h_results(4).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(4).imag(), r.imag());
    r = Kokkos::exp(a);
    ASSERT_FLOAT_EQ(h_results(4).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(4).imag(), r.imag());
    r = std::log(a);
    ASSERT_FLOAT_EQ(h_results(5).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(5).imag(), r.imag());
    r = std::sin(a);
    ASSERT_FLOAT_EQ(h_results(6).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(6).imag(), r.imag());
    r = std::cos(a);
    ASSERT_FLOAT_EQ(h_results(7).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(7).imag(), r.imag());
    r = std::tan(a);
    ASSERT_FLOAT_EQ(h_results(8).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(8).imag(), r.imag());
    r = std::sinh(a);
    ASSERT_FLOAT_EQ(h_results(9).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(9).imag(), r.imag());
    r = std::cosh(a);
    ASSERT_FLOAT_EQ(h_results(10).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(10).imag(), r.imag());
    r = std::tanh(a);
    ASSERT_FLOAT_EQ(h_results(11).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(11).imag(), r.imag());
    r = std::asinh(a);
    ASSERT_FLOAT_EQ(h_results(12).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(12).imag(), r.imag());
    r = std::acosh(a);
    ASSERT_FLOAT_EQ(h_results(13).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(13).imag(), r.imag());
    // atanh
    r = std::atanh(a);
    ASSERT_FLOAT_EQ(h_results(14).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(14).imag(), r.imag());
    r = std::asin(a);
    ASSERT_FLOAT_EQ(h_results(15).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(15).imag(), r.imag());
    r = std::acos(a);
    ASSERT_FLOAT_EQ(h_results(16).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(16).imag(), r.imag());
    // atan
    r = std::atan(a);
    ASSERT_FLOAT_EQ(h_results(17).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(17).imag(), r.imag());
    // log10
    r = std::log10(a);
    ASSERT_FLOAT_EQ(h_results(18).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(18).imag(), r.imag());
    // norm
    r = std::norm(a);
    ASSERT_FLOAT_EQ(h_results(19).real(), r.real());
    ASSERT_FLOAT_EQ(h_results(19).imag(), r.imag());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    Kokkos::complex<double> a(1.5, 2.5);
    double c = 9.3;

    d_results(0)  = Kokkos::complex<double>(Kokkos::real(a), Kokkos::imag(a));
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

void testComplexIO() {
  Kokkos::complex<double> z = {3.14, 1.41};
  std::stringstream ss;
  ss << z;
  ASSERT_EQ(ss.str(), "(3.14,1.41)");

  ss.str("1 (2) (3,4)");
  ss.clear();
  ss >> z;
  ASSERT_EQ(z, (Kokkos::complex<double>{1, 0}));
  ss >> z;
  ASSERT_EQ(z, (Kokkos::complex<double>{2, 0}));
  ss >> z;
  ASSERT_EQ(z, (Kokkos::complex<double>{3, 4}));
}

TEST(TEST_CATEGORY, complex_special_funtions) {
  TestComplexSpecialFunctions<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, complex_io) { testComplexIO(); }

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

// Insertion and extraction behave like those of std::complex<RealType>.
template <class RealType, class CharT>
void testComplexIOLikeStd() {
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
      ss << std::complex<RealType>(v[0], v[1]) << CharT('|');
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
    std::complex<RealType> zs(RealType(5), RealType(6));
    sk >> zk;
    ss >> zs;
    EXPECT_EQ(sk.fail(), ss.fail());
    EXPECT_EQ(zk.real(), zs.real());
    EXPECT_EQ(zk.imag(), zs.imag());
    EXPECT_EQ(sk.eof(), ss.eof());
    sk.clear();
    ss.clear();
    EXPECT_TRUE(String(std::istreambuf_iterator<CharT>(sk), {}) ==
                String(std::istreambuf_iterator<CharT>(ss), {}));
  }
}

TEST(TEST_CATEGORY, complex_io_like_std) {
  testComplexIOLikeStd<float, char>();
  testComplexIOLikeStd<double, char>();
  testComplexIOLikeStd<double, wchar_t>();
}

// operator/=(std::complex<RealType>) is host-only, and divides like
// operator/(complex, complex).
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

    // std::complex * Kokkos::complex (host only) recovers the same way.
    const C h = std::complex<RealType>(inf, nan) * C(1, 0);
    EXPECT_EQ(h.real(), inf);
    EXPECT_TRUE(std::isnan(h.imag()));
  }
};

TEST(TEST_CATEGORY, complex_multiplication) {
  TestComplexMultiplication<float, TEST_EXECSPACE>().testit();
  TestComplexMultiplication<double, TEST_EXECSPACE>().testit();
}

static_assert(std::is_trivially_copyable_v<Kokkos::complex<float>>);
static_assert(std::is_trivially_copyable_v<Kokkos::complex<double>>);
#ifndef KOKKOS_IMPL_32BIT  // FIXME_32BIT
// error: requested alignment '24' is not a positive power of 2
static_assert(std::is_trivially_copyable_v<Kokkos::complex<long double>>);
#endif

template <class ExecSpace>
struct TestBugPowAndLogComplex {
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_pow;
  Kokkos::View<Kokkos::complex<double> *, ExecSpace> d_log;
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
    d_pow(0) = Kokkos::pow(Kokkos::complex<double>(+3., 1.), 3.);
    d_pow(1) = Kokkos::pow(Kokkos::complex<double>(-3., 1.), 3.);
    d_log(0) = Kokkos::log(Kokkos::complex<double>(+3., 1.));
    d_log(1) = Kokkos::log(Kokkos::complex<double>(-3., 1.));
  }
};

TEST(TEST_CATEGORY, complex_issue_3865) {
  TestBugPowAndLogComplex<TEST_EXECSPACE>();
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
  TestComplexArgProj<float, TEST_EXECSPACE>().testit();
  TestComplexArgProj<double, TEST_EXECSPACE>().testit();
}

// exp, sqrt and log on the grid {0, -0, 1, -1, 0.5, 2.5, inf, -inf, nan}^2,
// compared with std::complex (glibc's C99 Annex G functions with libstdc++):
// same class (NaN, infinite, zero, finite) and sign for every part, and
// finite parts within a few eps.  Also polar with an infinite magnitude, and
// sqrt on both sides of its branch cut.
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
    const RealType tol = 8 * Kokkos::Experimental::epsilon_v<RealType>;
    auto check         = [&](RealType x, RealType e) {
      if (kind(x) != kind(e)) return false;
      if (kind(x) != 0 && std::signbit(x) != std::signbit(e)) return false;
      return kind(x) != 3 || std::abs(x - e) <= tol * std::abs(e);
    };
    const char *names[3] = {"exp", "sqrt", "log"};
    for (int i = 0; i < N; ++i) {
      const std::complex<RealType> z(h_in(i).real(), h_in(i).imag());
      const std::complex<RealType> e[3] = {std::exp(z), std::sqrt(z),
                                           std::log(z)};
      for (int f = 0; f < 3; ++f) {
        EXPECT_TRUE(check(h_out(i, f).real(), e[f].real()) &&
                    check(h_out(i, f).imag(), e[f].imag()))
            << names[f] << h_in(i) << " = " << h_out(i, f) << ", std gives "
            << e[f];
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
        EXPECT_TRUE(check(h_out(k, f).real(), e[f].real(), with_sign) &&
                    check(h_out(k, f).imag(), e[f].imag(), with_sign))
            << names[f] << h_in(k) << " = " << h_out(k, f) << ", std gives "
            << e[f];
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
        EXPECT_TRUE(check(h_out(k, f).real(), e[f].real(), with_sign) &&
                    check(h_out(k, f).imag(), e[f].imag(), with_sign))
            << names[f] << h_in(k) << " = " << h_out(k, f) << ", std gives "
            << e[f];
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

template <class ExecSpace>
struct TestComplexStructuredBindings {
  using exec_space       = ExecSpace;
  using value_type       = double;
  using complex_type     = Kokkos::complex<double>;
  using device_view_type = Kokkos::View<complex_type *, exec_space>;
  using host_view_type   = typename device_view_type::host_mirror_type;

  device_view_type d_results;
  host_view_type h_results;

  // tuple_size
  static_assert(std::is_same_v<std::tuple_size<complex_type>::type,
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
    ASSERT_FLOAT_EQ(h_results[0].real(), 2.);
    ASSERT_FLOAT_EQ(h_results[0].imag(), 3.);

    // get rvalue
    ASSERT_FLOAT_EQ(h_results[1].real(), 2.);
    ASSERT_FLOAT_EQ(h_results[1].imag(), 3.);

    // get const lvalue
    ASSERT_FLOAT_EQ(h_results[2].real(), 5.);
    ASSERT_FLOAT_EQ(h_results[2].imag(), 7.);

    // get const rvalue
    ASSERT_FLOAT_EQ(h_results[3].real(), 5.);
    ASSERT_FLOAT_EQ(h_results[3].imag(), 7.);

    // swap real and imaginary
    ASSERT_FLOAT_EQ(h_results[4].real(), 11.);
    ASSERT_FLOAT_EQ(h_results[4].imag(), 13.);
    ASSERT_FLOAT_EQ(h_results[5].real(), 13.);
    ASSERT_FLOAT_EQ(h_results[5].imag(), 11.);
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
  TestComplexStructuredBindings<TEST_EXECSPACE> test;
  test.testit();
}

#define CHECK_COMPLEX(_value_, _real_, _imag_) \
  (void)_value_;                               \
  if (_value_.real() != _real_) return false;  \
  if (_value_.imag() != _imag_) return false;

constexpr bool can_appear_in_constant_expressions() {
  const Kokkos::complex<double> from_single{1.2};
  const Kokkos::complex<double> from_both{1.2, 3.4};
  const Kokkos::complex<double> from_none{};

  CHECK_COMPLEX(from_single, 1.2, 0.);
  CHECK_COMPLEX(from_both, 1.2, 3.4);
  CHECK_COMPLEX(from_none, 0., 0.);

  Kokkos::complex<double> from_copy_assign;
  from_copy_assign = from_both;
  const auto from_copy_constr(from_both);

  CHECK_COMPLEX(from_copy_assign, 1.2, 3.4);
  CHECK_COMPLEX(from_copy_constr, 1.2, 3.4);

  Kokkos::complex<double> from_move_assign;
  from_move_assign = std::move(from_both);
  const auto from_move_constr(std::move(from_copy_assign));

  CHECK_COMPLEX(from_move_assign, 1.2, 3.4);
  CHECK_COMPLEX(from_move_constr, 1.2, 3.4);

  Kokkos::complex<double> from_real;
  from_real = 4.;

  CHECK_COMPLEX(from_real, 4., 0.);

  return true;
}

#undef CHECK_COMPLEX

static_assert(can_appear_in_constant_expressions());

constexpr bool comparison_in_constant_expression() {
  static_assert(Kokkos::complex<double>{42., 43.} ==
                Kokkos::complex<double>{42., 43.});
  static_assert(Kokkos::complex<double>{42., 43.} !=
                Kokkos::complex<double>{42., 42.});

  static_assert(Kokkos::complex<double>{42., 0.} == double{42.});
  static_assert(Kokkos::complex<double>{42., 43.} != double{42.});

  static_assert(double{42.} == Kokkos::complex<double>{42., 0.});
  static_assert(double{43.} != Kokkos::complex<double>{42., 0.});

  return true;
}

static_assert(comparison_in_constant_expression());

constexpr bool test_complex_norm() {
  return Kokkos::norm(Kokkos::complex<double>{4., 2.}) == 20.;
}
static_assert(test_complex_norm());

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

constexpr bool test_complex_conj() {
  static_assert(Kokkos::conj(Kokkos::complex<double>{1., 1.}) ==
                Kokkos::complex<double>{1., -1.});
  static_assert(Kokkos::conj(Kokkos::complex<double>{1., -1.}) ==
                Kokkos::complex<double>{1., 1.});
  static_assert(Kokkos::conj(Kokkos::complex<double>{-1., 1.}) ==
                Kokkos::complex<double>{-1., -1.});
  static_assert(Kokkos::conj(Kokkos::complex<double>{-1., -1.}) ==
                Kokkos::complex<double>{-1., 1.});
  return true;
}
static_assert(test_complex_conj());

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

// std::complex times Kokkos::complex (host only).
static_assert(std::complex<double>(1, 2) * Kokkos::complex<double>(3, 4) ==
              Kokkos::complex<double>(-5, 10));

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
