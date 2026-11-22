// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef TESTHALFIO_HPP_
#define TESTHALFIO_HPP_

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iterator>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#if defined(KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT) || \
    defined(KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT)
#include <stdfloat>
#endif

namespace Test {

// Stream insertion and extraction of half_t and bhalf_t, as for
// std::float16_t and std::bfloat16_t (C++23 [ostream.inserters.arithmetic],
// [istream.formatted.arithmetic]/4-6).  Insertion writes the value as the
// float it is exactly equal to, so it follows the stream's flags, precision
// and locale.  Extraction reads a float, stops where a number ends, and
// reports errors with failbit (no exceptions): beyond +-max() it stores
// +-max(), on a parse failure 0.  The tests also hold where half_t or bhalf_t
// is float (no native half type): that is float's own extraction.

// A locale whose decimal point is ',', to check that the stream's own locale
// is used (and not the C global locale).  A named locale is used: a
// user-defined std::numpunct facet crashes in libstdc++ when compiled by nvcc
// (seen with CUDA 12.6 and GCC 12.2), independently of Kokkos.  Returns false
// if no such locale is installed.
inline bool comma_decimal_locale(std::locale &loc) {
  for (const char *name :
       {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8"}) {
    try {
      loc = std::locale(name);
      return true;
    } catch (const std::runtime_error &) {
    }
  }
  return false;
}

template <class H>
void test_half_io_output() {
  const H values[]                                    = {H(3.14f),
                                                         H(1.41f),
                                                         H(-0.5f),
                                                         H(-0.f),
                                                         H(Kokkos::Experimental::denorm_min_v<H>),
                                                         H(1e-5f),
                                                         H(100.f),
                                                         H(Kokkos::Experimental::finite_max_v<H>)};
  const std::function<void(std::ostream &)> formats[] = {
      [](std::ostream &) {},
      [](std::ostream &s) { s << std::setprecision(3); },
      [](std::ostream &s) { s << std::scientific << std::uppercase; },
      [](std::ostream &s) { s << std::fixed << std::setprecision(2); },
      [](std::ostream &s) { s << std::showpos << std::showpoint; },
      [](std::ostream &s) { s << std::hexfloat; },
      [](std::ostream &s) {
        s << std::left << std::setfill('*') << std::setw(16);
      },
  };
  for (const H &h : values) {
    for (const auto &format : formats) {
      std::ostringstream sh;
      std::ostringstream sf;
      format(sh);
      format(sf);
      sh << h << '|';
      sf << static_cast<float>(h) << '|';
      EXPECT_EQ(sh.str(), sf.str());
    }
  }

  // Small values are not printed as 0.000000 any more.
  std::ostringstream tiny;
  tiny << H(Kokkos::Experimental::denorm_min_v<H>);
  EXPECT_NE(tiny.str(), "0.000000");

  // The locale of the stream is used.
  std::locale comma;
  if (comma_decimal_locale(comma)) {
    std::ostringstream loc;
    loc.imbue(comma);
    loc << H(3.14f);
    std::ostringstream ref;
    ref.imbue(comma);
    ref << static_cast<float>(H(3.14f));
    EXPECT_EQ(loc.str(), ref.str());
    EXPECT_NE(loc.str().find(','), std::string::npos) << loc.str();
  }

  // Wide streams.
  std::wostringstream wh;
  std::wostringstream wf;
  wh << H(3.14f);
  wf << static_cast<float>(H(3.14f));
  EXPECT_TRUE(wh.str() == wf.str());
}

template <class H>
void test_half_io_input() {
  // A number ends where a double ends: ',' and ')' stay in the stream.
  {
    std::istringstream is("2,3");
    H a(0.f);
    H b(0.f);
    char c = 0;
    is >> a >> c >> b;
    EXPECT_FALSE(is.fail());
    EXPECT_EQ(static_cast<float>(a), 2.f);
    EXPECT_EQ(c, ',');
    EXPECT_EQ(static_cast<float>(b), 3.f);
  }
  {
    std::istringstream is("1.5)");
    H a(0.f);
    char c = 0;
    is >> a >> c;
    EXPECT_FALSE(is.fail());
    EXPECT_EQ(static_cast<float>(a), 1.5f);
    EXPECT_EQ(c, ')');
  }
  // Errors set failbit and do not throw.  A parse failure stores 0, a value
  // beyond the range +-max(); without input (the sentry fails) the value is
  // unchanged.
  const float max =
      static_cast<float>(H(Kokkos::Experimental::finite_max_v<H>));
  const struct {
    const char *input;
    float expected;
  } errors[] = {{"x", 0.f}, {"", 7.f}, {"1e999", max}, {"-1e999", -max}};
  for (const auto &e : errors) {
    SCOPED_TRACE(e.input);
    std::istringstream is(e.input);
    H h(7.f);
    EXPECT_NO_THROW(is >> h);
    EXPECT_TRUE(is.fail());
    EXPECT_EQ(static_cast<float>(h), e.expected);
  }
  // In range: exactly the float read by the stream, converted to H.
  for (const char *input :
       {"3.14159", "-2.5e-3", "65504", "1e-7", "  42", "-0"}) {
    SCOPED_TRACE(input);
    std::istringstream is(input);
    H h(7.f);
    is >> h;
    EXPECT_FALSE(is.fail());
    const H expected(std::strtof(input, nullptr));
    // The bits of the float values: also tells -0 from 0.
    EXPECT_EQ(Kokkos::bit_cast<std::uint32_t>(static_cast<float>(h)),
              Kokkos::bit_cast<std::uint32_t>(static_cast<float>(expected)));
  }
  // Reading until the end of the input stops normally.
  {
    std::istringstream is("1 2 3");
    H h(0.f);
    int n   = 0;
    float s = 0;
    while (is >> h) {
      ++n;
      s += static_cast<float>(h);
    }
    EXPECT_EQ(n, 3);
    EXPECT_EQ(s, 6.f);
    EXPECT_TRUE(is.eof());
  }
  // Wide streams.
  {
    std::wistringstream is(L"2.5");
    H h(0.f);
    is >> h;
    EXPECT_FALSE(is.fail());
    EXPECT_EQ(static_cast<float>(h), 2.5f);
  }
}

// Extraction reads a float and then converts, as the standard specifies; this
// can round twice (the standard's note accepts it): a decimal just above the
// midpoint between 1 and the next value reads as the midpoint, which rounds to
// even, i.e. to 1.
template <class H>
void test_half_io_rounding(const char *just_above_midpoint, float expected) {
  std::istringstream is(just_above_midpoint);
  H h(0.f);
  is >> h;
  EXPECT_FALSE(is.fail());
  EXPECT_EQ(static_cast<float>(h), expected) << just_above_midpoint;
}

// A value beyond the range of H but not of float sets failbit and stores
// +-max(), as for std::float16_t (and std::bfloat16_t, whose max() is below
// float's).
template <class H>
void test_half_io_range(const char *above_max, const char *below_min) {
  const H max(Kokkos::Experimental::finite_max_v<H>);
  for (const auto &[input, expected] :
       {std::pair{above_max, max}, std::pair{below_min, H(-max)}}) {
    SCOPED_TRACE(input);
    std::istringstream is(input);
    H h(7.f);
    is >> h;
    EXPECT_TRUE(is.fail());
    EXPECT_EQ(static_cast<float>(h), static_cast<float>(expected));
  }
}

// Every finite value survives insertion with the default precision (6, which
// is at least max_digits10) and extraction, bit for bit.
template <class H>
void test_half_io_round_trip() {
  static_assert(sizeof(H) == sizeof(std::uint16_t));
  int failures = 0;
  for (std::uint32_t bits = 0; bits <= 0xffff; ++bits) {
    const H h = Kokkos::bit_cast<H>(static_cast<std::uint16_t>(bits));
    if (!Kokkos::isfinite(static_cast<float>(h))) continue;
    std::stringstream ss;
    ss << h;
    H back(0.f);
    ss >> back;
    if (ss.fail() || Kokkos::bit_cast<std::uint16_t>(back) !=
                         static_cast<std::uint16_t>(bits)) {
      if (++failures <= 5) {
        ADD_FAILURE() << "bits 0x" << std::hex << bits << " printed as \""
                      << ss.str() << "\" did not read back";
      }
    }
  }
  EXPECT_EQ(failures, 0);
}

// On the std backend, compare with libstdc++'s own insertion and extraction of
// std::float16_t / std::bfloat16_t: the same characters in, the same value
// bits (NaN: both NaN), the same stream state and the same remaining input;
// and the same characters out, under several format flags.
//
// Not compared: input where the sentry fails (empty, or only whitespace).
// The standard does the extraction only if the sentry succeeds, so the value
// stays unchanged (half_io checks that). libstdc++ (GCC 13 to 15 and master)
// assigns static_cast<_Float16>(__flt) unconditionally, and __flt is then
// uninitialized, so its result is indeterminate there.
#if defined(KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT) || \
    defined(KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT)
template <class H, class F>
void test_half_io_vs_std(std::initializer_list<const char *> extra) {
  static_assert(sizeof(H) == 2 && sizeof(F) == 2);
  auto bits        = [](auto x) { return Kokkos::bit_cast<std::uint16_t>(x); };
  auto is_nan      = [](auto x) { return std::isnan(static_cast<float>(x)); };
  auto check_input = [&](const char *input) {
    SCOPED_TRACE(std::string("input \"") + input + "\"");
    std::istringstream ik(input);
    std::istringstream is(input);
    H h(7.f);
    F f = static_cast<F>(7.f);
    ik >> h;
    is >> f;
    EXPECT_EQ(ik.rdstate(), is.rdstate());
    if (is_nan(f)) {
      EXPECT_TRUE(is_nan(h)) << std::hex << bits(h);
    } else {
      EXPECT_EQ(bits(h), bits(f))
          << static_cast<float>(h) << " vs " << static_cast<float>(f);
    }
    ik.clear();
    is.clear();
    std::string rk, rs;
    std::getline(ik, rk);
    std::getline(is, rs);
    EXPECT_EQ(rk, rs);
  };
  for (const char *input :
       {"1",     "3.14159", "-2.5e-3", "1e-7",   "1e-30",    "-0",  "  42",
        "2,3",   "1.5)",    "x",       "-",      "+",        "e5",  ".",
        "1e",    "1e+",     "inf",     "-inf",   "infinity", "nan", "-nan",
        "1e999", "-1e999",  "1e-999",  "0x1p-3", "1.0e0x"}) {
    check_input(input);
  }
  for (const char *input : extra) check_input(input);

  // Output: every 1009th bit pattern (all classes appear), under several
  // format settings.
  using Setup          = void (*)(std::ostream &);
  const Setup setups[] = {
      [](std::ostream &) {},
      [](std::ostream &o) { o << std::setprecision(10); },
      [](std::ostream &o) { o << std::scientific << std::setprecision(3); },
      [](std::ostream &o) { o << std::fixed << std::setprecision(2); },
      [](std::ostream &o) { o << std::hexfloat; },
      [](std::ostream &o) { o << std::showpos << std::uppercase; },
      [](std::ostream &o) { o << std::setw(12) << std::left; },
  };
  int mismatches = 0;
  for (std::uint32_t b = 0; b <= 0xffff; b += 1009) {
    const std::uint16_t u = static_cast<std::uint16_t>(b);
    for (int k = 0; k < static_cast<int>(std::size(setups)); ++k) {
      std::ostringstream ok;
      std::ostringstream os;
      setups[k](ok);
      setups[k](os);
      ok << Kokkos::bit_cast<H>(u);
      os << Kokkos::bit_cast<F>(u);
      if (ok.str() != os.str() && ++mismatches <= 5) {
        ADD_FAILURE() << "bits 0x" << std::hex << b << ", setup " << k
                      << ": Kokkos \"" << ok.str() << "\", libstdc++ \""
                      << os.str() << "\"";
      }
    }
  }
  EXPECT_EQ(mismatches, 0);
}
#endif

TEST(TEST_CATEGORY, half_io_vs_std) {
#ifdef KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT
  // Around max() = 65504: 65519.99 would round to max() in half, but as a
  // float it is beyond max(), so both store max() with failbit; 5.96e-8 and
  // 2.98e-8 are near the smallest subnormal and half of it.
  test_half_io_vs_std<Kokkos::Experimental::half_t, std::float16_t>(
      {"65504", "65519.99", "65520", "1e5", "-1e5", "5.96e-8", "2.98e-8",
       "1.000488281250001"});
#else
  GTEST_SKIP() << "needs the std half backend (std::float16_t, C++23)";
#endif
}

TEST(TEST_CATEGORY, bhalf_io_vs_std) {
#ifdef KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT
  test_half_io_vs_std<Kokkos::Experimental::bhalf_t, std::bfloat16_t>(
      {"3.389e38", "3.4e38", "-3.4e38", "9.2e-41", "1.003906250000001"});
#else
  GTEST_SKIP() << "needs the std half backend (std::bfloat16_t, C++23)";
#endif
}

TEST(TEST_CATEGORY, half_io) {
  test_half_io_output<Kokkos::Experimental::half_t>();
  test_half_io_input<Kokkos::Experimental::half_t>();
#if !KOKKOS_HALF_T_IS_FLOAT
  // 1 + 2^-11 + 1e-15 reads as the float 1 + 2^-11, the half midpoint between
  // 1 and 1 + 2^-10, which rounds to even.
  test_half_io_rounding<Kokkos::Experimental::half_t>("1.000488281250001", 1.f);
  test_half_io_range<Kokkos::Experimental::half_t>("1e5", "-1e5");
  test_half_io_round_trip<Kokkos::Experimental::half_t>();
#endif
}

TEST(TEST_CATEGORY, bhalf_io) {
  test_half_io_output<Kokkos::Experimental::bhalf_t>();
  test_half_io_input<Kokkos::Experimental::bhalf_t>();
#if !KOKKOS_BHALF_T_IS_FLOAT
  // 1 + 2^-8 + 1e-15 reads as the float 1 + 2^-8, the bhalf midpoint between 1
  // and 1 + 2^-7, which rounds to even.
  test_half_io_rounding<Kokkos::Experimental::bhalf_t>("1.003906250000001",
                                                       1.f);
  // bhalf_t's max() is 0x1.fep127 < float's max().
  test_half_io_range<Kokkos::Experimental::bhalf_t>("3.4e38", "-3.4e38");
  test_half_io_round_trip<Kokkos::Experimental::bhalf_t>();
#endif
}

}  // namespace Test

#endif
