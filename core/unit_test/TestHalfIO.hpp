// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef TESTHALFIO_HPP_
#define TESTHALFIO_HPP_

#include <cstdint>
#include <functional>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace Test {

// Stream insertion and extraction of half_t and bhalf_t.  Insertion writes
// the value as the float it is exactly equal to, so it follows the stream's
// flags, precision and locale; extraction reads a double, stops where a
// number ends, and reports errors with failbit (no exceptions).

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
  // Errors set failbit, do not throw, and leave the value unchanged.
  for (const char *input : {"x", "", "1e999", "-1e999"}) {
    SCOPED_TRACE(input);
    std::istringstream is(input);
    H h(7.f);
    EXPECT_NO_THROW(is >> h);
    EXPECT_TRUE(is.fail());
    EXPECT_EQ(static_cast<float>(h), 7.f);
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

// Extraction rounds once (through double): a decimal just above the midpoint
// between 1 and the next value must round up.  Through float it would first
// round to the midpoint and then to even, i.e. to 1.
template <class H>
void test_half_io_rounding(const char *just_above_midpoint, float expected) {
  std::istringstream is(just_above_midpoint);
  H h(0.f);
  is >> h;
  EXPECT_FALSE(is.fail());
  EXPECT_EQ(static_cast<float>(h), expected) << just_above_midpoint;
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

TEST(TEST_CATEGORY, half_io) {
  test_half_io_output<Kokkos::Experimental::half_t>();
  test_half_io_input<Kokkos::Experimental::half_t>();
#if !KOKKOS_HALF_T_IS_FLOAT
  // 1 + 2^-11 + 1e-15: the half midpoint is 1 + 2^-11, the next value
  // 1 + 2^-10.
  test_half_io_rounding<Kokkos::Experimental::half_t>("1.000488281250001",
                                                      1.0009765625f);
  test_half_io_round_trip<Kokkos::Experimental::half_t>();
#endif
}

TEST(TEST_CATEGORY, bhalf_io) {
  test_half_io_output<Kokkos::Experimental::bhalf_t>();
  test_half_io_input<Kokkos::Experimental::bhalf_t>();
#if !KOKKOS_BHALF_T_IS_FLOAT
  // 1 + 2^-8 + 1e-15: the bhalf midpoint is 1 + 2^-8, the next value 1 + 2^-7.
  test_half_io_rounding<Kokkos::Experimental::bhalf_t>("1.003906250000001",
                                                       1.0078125f);
  test_half_io_round_trip<Kokkos::Experimental::bhalf_t>();
#endif
}

}  // namespace Test

#endif
