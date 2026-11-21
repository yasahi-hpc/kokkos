// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOS_COMPLEX_HPP
#define KOKKOS_COMPLEX_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_COMPLEX
#endif

#include <Kokkos_Atomic.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <Kokkos_ReductionIdentity.hpp>
#include <impl/Kokkos_Error.hpp>
#include <complex>
#include <type_traits>
#include <iosfwd>
#include <tuple>

namespace Kokkos {

namespace Impl {

// |x| for the comparisons in complex_divide_smith: Kokkos::fabs at run time,
// and a comparison during constant evaluation (Kokkos::fabs is not constexpr).
// The sign of a zero result does not matter for the comparisons.
struct ComplexRuntimeFabs {
  template <class T>
  KOKKOS_INLINE_FUNCTION static T apply(T x) {
    return Kokkos::fabs(x);
  }
};
struct ComplexConstexprFabs {
  template <class T>
  KOKKOS_INLINE_FUNCTION static constexpr T apply(T x) {
    return x < T(0) ? -x : x;
  }
};

/// \brief Smith's algorithm for (a + ib) / (c + id) = x + iy, with the range
///   guards of libgcc's __divdc3 (GCC >= 12), without any NaN recovery.
template <class Fabs, class T>
KOKKOS_INLINE_FUNCTION constexpr void complex_divide_smith(
    T a, T b, T c, T d, T& x, T& y) noexcept(noexcept(T{} / T{})) {
  // Thresholds of libgcc's RBIG, RMIN, RMIN2, RMINSCAL and RMAX2.
  constexpr T rbig     = Kokkos::finite_max_v<T> / 2;
  constexpr T rmin     = Kokkos::norm_min_v<T>;
  constexpr T rmin2    = Kokkos::epsilon_v<T>;
  constexpr T rminscal = T(1) / Kokkos::epsilon_v<T>;
  constexpr T rmax2    = rbig * rmin2;

  if (Fabs::apply(c) < Fabs::apply(d)) {
    // Prevent underflow when the denominator is near the largest value.
    if (Fabs::apply(d) >= rbig) {
      a /= 2;
      b /= 2;
      c /= 2;
      d /= 2;
    }
    // Scale up small operands to avoid underflow; no new overflow is possible.
    if (Fabs::apply(d) < rmin2 ||
        (Fabs::apply(a) < rmin && Fabs::apply(b) < rmax2 &&
         Fabs::apply(d) < rmax2) ||
        (Fabs::apply(b) < rmin && Fabs::apply(a) < rmax2 &&
         Fabs::apply(d) < rmax2)) {
      a *= rminscal;
      b *= rminscal;
      c *= rminscal;
      d *= rminscal;
    }
    const T ratio = c / d;
    const T denom = c * ratio + d;
    // Choose another order of computation if ratio is subnormal.
    if (Fabs::apply(ratio) > rmin) {
      x = (a * ratio + b) / denom;
      y = (b * ratio - a) / denom;
    } else {
      x = (c * (a / d) + b) / denom;
      y = (c * (b / d) - a) / denom;
    }
  } else {
    if (Fabs::apply(c) >= rbig) {
      a /= 2;
      b /= 2;
      c /= 2;
      d /= 2;
    }
    if (Fabs::apply(c) < rmin2 ||
        (Fabs::apply(a) < rmin && Fabs::apply(b) < rmax2 &&
         Fabs::apply(c) < rmax2) ||
        (Fabs::apply(b) < rmin && Fabs::apply(a) < rmax2 &&
         Fabs::apply(c) < rmax2)) {
      a *= rminscal;
      b *= rminscal;
      c *= rminscal;
      d *= rminscal;
    }
    const T ratio = d / c;
    const T denom = d * ratio + c;
    if (Fabs::apply(ratio) > rmin) {
      x = (b * ratio + a) / denom;
      y = (b - a * ratio) / denom;
    } else {
      x = (a + d * (b / c)) / denom;
      y = (b - d * (a / c)) / denom;
    }
  }
}

// Not constexpr: calling it during constant evaluation makes the expression
// not constant, and the name says why.
KOKKOS_INLINE_FUNCTION void
complex_division_by_zero_or_of_inf_or_nan_is_not_a_constant_expression() {}

/// \brief Complex division (a + ib) / (c + id) = re + i im.
///
/// Smith's algorithm, with the range guards of libgcc's __divdc3 (GCC >= 12),
/// followed by the C99 Annex G (G.5.2) recovery of the infinities and zeros
/// that the formula computes as NaN + iNaN: nonzero / zero, infinite / finite
/// and finite / infinite.  No test is made up front, so finite operands only
/// pay for the comparisons of the range guards.
///
/// During constant evaluation, the operands must be finite and the divisor
/// nonzero: compilers reject floating-point operations that produce a NaN or
/// divide by zero there.  For such operands Smith's algorithm never produces a
/// NaN, so the recovery is not needed, and the result is the same as at run
/// time (up to floating-point contraction at run time).
template <class T>
KOKKOS_INLINE_FUNCTION constexpr void complex_divide(
    T a, T b, T c, T d, T& re, T& im) noexcept(noexcept(T{} / T{})) {
#if defined(__cpp_if_consteval)  // since C++23
  if consteval {
#else
  if (__builtin_is_constant_evaluated()) {
#endif
    constexpr T inf = Kokkos::infinity_v<T>;
    auto finite     = [](T v) { return v == v && v != inf && v != -inf; };
    if (!(finite(a) && finite(b) && finite(c) && finite(d)) ||
        (c == T(0) && d == T(0))) {
      complex_division_by_zero_or_of_inf_or_nan_is_not_a_constant_expression();
    }
    complex_divide_smith<ComplexConstexprFabs>(a, b, c, d, re, im);
    return;
  }

  // The operands as given, for the NaN recovery below.
  const T a0 = a;
  const T b0 = b;
  const T c0 = c;
  const T d0 = d;

  T x;
  T y;
  complex_divide_smith<ComplexRuntimeFabs>(a, b, c, d, x, y);

  // Recover infinities and zeros that computed as NaN + iNaN.
  if (Kokkos::isnan(x) && Kokkos::isnan(y)) {
    constexpr T inf = Kokkos::infinity_v<T>;
    if (c0 == T(0) && d0 == T(0) &&
        (!Kokkos::isnan(a0) || !Kokkos::isnan(b0))) {
      x = Kokkos::copysign(inf, c0) * a0;
      y = Kokkos::copysign(inf, c0) * b0;
    } else if ((Kokkos::isinf(a0) || Kokkos::isinf(b0)) &&
               Kokkos::isfinite(c0) && Kokkos::isfinite(d0)) {
      const T ai = Kokkos::copysign(Kokkos::isinf(a0) ? T(1) : T(0), a0);
      const T bi = Kokkos::copysign(Kokkos::isinf(b0) ? T(1) : T(0), b0);
      x          = inf * (ai * c0 + bi * d0);
      y          = inf * (bi * c0 - ai * d0);
    } else if ((Kokkos::isinf(c0) || Kokkos::isinf(d0)) &&
               Kokkos::isfinite(a0) && Kokkos::isfinite(b0)) {
      const T ci = Kokkos::copysign(Kokkos::isinf(c0) ? T(1) : T(0), c0);
      const T di = Kokkos::copysign(Kokkos::isinf(d0) ? T(1) : T(0), d0);
      x          = T(0) * (a0 * ci + b0 * di);
      y          = T(0) * (b0 * ci - a0 * di);
    }
  }
  re = x;
  im = y;
}

// Not constexpr: calling it during constant evaluation makes the expression
// not constant, and the name says why.
KOKKOS_INLINE_FUNCTION void
complex_multiplication_of_inf_or_nan_is_not_a_constant_expression() {}

/// \brief Complex multiplication (a + ib) (c + id) = re + i im.
///
/// The usual formula (ac - bd) + i(ad + bc).  At run time, when both parts
/// are NaN, the infinities are recovered as specified by C99 Annex G (G.5.1)
/// and done by libgcc's and compiler-rt's __muldc3, libc++ and libcu++: e.g.
/// (inf, nan) * (1, 0) is (inf, nan), not (nan, nan).  Finite operands never
/// get there, so they only pay for the two isnan tests.
///
/// During constant evaluation the operands must be finite; for those the
/// result is the same as at run time (up to contraction at run time).
template <class T>
KOKKOS_INLINE_FUNCTION constexpr void complex_multiply(T a, T b, T c, T d,
                                                       T& re, T& im) noexcept {
#if defined(__cpp_if_consteval)  // since C++23
  if consteval {
#else
  if (__builtin_is_constant_evaluated()) {
#endif
    constexpr T inf = Kokkos::infinity_v<T>;
    auto finite     = [](T v) { return v == v && v != inf && v != -inf; };
    if (!(finite(a) && finite(b) && finite(c) && finite(d))) {
      complex_multiplication_of_inf_or_nan_is_not_a_constant_expression();
    }
    re = a * c - b * d;
    im = a * d + b * c;
    return;
  }

  // Same expressions as before the recovery was added, so that the rounding
  // (and floating-point contraction) of ordinary values is unchanged.
  T x = a * c - b * d;
  T y = a * d + b * c;
  if (Kokkos::isnan(x) && Kokkos::isnan(y)) {
    // Replace infinite operands by +-1 and NaN partners by +-0, then scale.
    bool recalc = false;
    if (Kokkos::isinf(a) || Kokkos::isinf(b)) {
      a = Kokkos::copysign(Kokkos::isinf(a) ? T(1) : T(0), a);
      b = Kokkos::copysign(Kokkos::isinf(b) ? T(1) : T(0), b);
      if (Kokkos::isnan(c)) c = Kokkos::copysign(T(0), c);
      if (Kokkos::isnan(d)) d = Kokkos::copysign(T(0), d);
      recalc = true;
    }
    if (Kokkos::isinf(c) || Kokkos::isinf(d)) {
      c = Kokkos::copysign(Kokkos::isinf(c) ? T(1) : T(0), c);
      d = Kokkos::copysign(Kokkos::isinf(d) ? T(1) : T(0), d);
      if (Kokkos::isnan(a)) a = Kokkos::copysign(T(0), a);
      if (Kokkos::isnan(b)) b = Kokkos::copysign(T(0), b);
      recalc = true;
    }
    if (!recalc && (Kokkos::isinf(a * c) || Kokkos::isinf(b * d) ||
                    Kokkos::isinf(a * d) || Kokkos::isinf(b * c))) {
      // Overflowing partial products: recover infinities from them.
      if (Kokkos::isnan(a)) a = Kokkos::copysign(T(0), a);
      if (Kokkos::isnan(b)) b = Kokkos::copysign(T(0), b);
      if (Kokkos::isnan(c)) c = Kokkos::copysign(T(0), c);
      if (Kokkos::isnan(d)) d = Kokkos::copysign(T(0), d);
      recalc = true;
    }
    if (recalc) {
      constexpr T inf = Kokkos::infinity_v<T>;
      x               = inf * (a * c - b * d);
      y               = inf * (a * d + b * c);
    }
  }
  re = x;
  im = y;
}

}  // namespace Impl

/// \class complex
/// \brief Partial reimplementation of std::complex that works as the
///   result of a Kokkos::parallel_reduce.
/// \tparam RealType The type of the real and imaginary parts of the
///   complex number.  As with std::complex, this is only defined for
///   \c float, \c double, and <tt>long double</tt>.  The latter is
///   currently forbidden in CUDA device kernels.
template <class RealType>
class
#ifdef KOKKOS_ENABLE_COMPLEX_ALIGN
    alignas(2 * sizeof(RealType))
#endif
        complex {
  static_assert(std::is_floating_point_v<RealType> &&
                    std::is_same_v<RealType, std::remove_cv_t<RealType>>,
                "Kokkos::complex can only be instantiated for a cv-unqualified "
                "floating point type");

 private:
  RealType re_{};
  RealType im_{};

 public:
  //! The type of the real or imaginary parts of this complex number.
  using value_type = RealType;

  //! Default constructor (initializes both real and imaginary parts to zero).
  KOKKOS_DEFAULTED_FUNCTION
  complex() = default;

  /// \brief Conversion constructor from compatible RType
  template <class RType,
            std::enable_if_t<std::is_convertible_v<RType, RealType>, int> = 0>
  KOKKOS_INLINE_FUNCTION complex(const complex<RType>& other) noexcept
      // Intentionally do the conversions implicitly here so that users don't
      // get any warnings about narrowing, etc., that they would expect to get
      // otherwise.
      : re_(other.real()), im_(other.imag()) {}

  /// \brief Conversion constructor from std::complex.
  ///
  /// This constructor cannot be called in a CUDA device function,
  /// because std::complex's methods and nonmember functions are not
  /// marked as CUDA device functions.
  KOKKOS_INLINE_FUNCTION
  complex(const std::complex<RealType>& src) noexcept
      // We can use this aspect of the standard to avoid calling
      // non-device-marked functions `std::real` and `std::imag`: "For any
      // object z of type complex<T>, reinterpret_cast<T(&)[2]>(z)[0] is the
      // real part of z and reinterpret_cast<T(&)[2]>(z)[1] is the imaginary
      // part of z." Now we don't have to provide a whole bunch of the overloads
      // of things taking either Kokkos::complex or std::complex
      : re_(reinterpret_cast<const RealType (&)[2]>(src)[0]),
        im_(reinterpret_cast<const RealType (&)[2]>(src)[1]) {}

  /// \brief Conversion operator to std::complex.
  ///
  /// This operator cannot be called in a CUDA device function,
  /// because std::complex's methods and nonmember functions are not
  /// marked as CUDA device functions.
  // TODO: make explicit.  DJS 2019-08-28
  operator std::complex<RealType>() const noexcept {
    return std::complex<RealType>(re_, im_);
  }

  /// \brief Constructor that takes just the real part, and sets the
  ///   imaginary part to zero.
  KOKKOS_INLINE_FUNCTION constexpr complex(const RealType& val) noexcept
      : re_(val), im_(static_cast<RealType>(0)) {}

  //! Constructor that takes the real and imaginary parts.
  KOKKOS_INLINE_FUNCTION
  constexpr complex(const RealType& re, const RealType& im) noexcept
      : re_(re), im_(im) {}

  //! Assignment operator (from a real number).
  KOKKOS_INLINE_FUNCTION constexpr complex& operator=(
      const RealType& val) noexcept {
    re_ = val;
    im_ = RealType(0);
    return *this;
  }

  /// \brief Assignment operator from std::complex.
  ///
  /// This constructor cannot be called in a CUDA device function,
  /// because std::complex's methods and nonmember functions are not
  /// marked as CUDA device functions.
  complex& operator=(const std::complex<RealType>& src) noexcept {
    *this = complex(src);
    return *this;
  }

  //! The imaginary part of this complex number.
  KOKKOS_INLINE_FUNCTION
  constexpr RealType& imag() noexcept { return im_; }

  //! The real part of this complex number.
  KOKKOS_INLINE_FUNCTION
  constexpr RealType& real() noexcept { return re_; }

  //! The imaginary part of this complex number.
  KOKKOS_INLINE_FUNCTION
  constexpr RealType imag() const noexcept { return im_; }

  //! The real part of this complex number.
  KOKKOS_INLINE_FUNCTION
  constexpr RealType real() const noexcept { return re_; }

  //! Set the imaginary part of this complex number.
  KOKKOS_INLINE_FUNCTION
  constexpr void imag(RealType v) noexcept { im_ = v; }

  //! Set the real part of this complex number.
  KOKKOS_INLINE_FUNCTION
  constexpr void real(RealType v) noexcept { re_ = v; }

  constexpr KOKKOS_INLINE_FUNCTION complex& operator+=(
      const complex<RealType>& src) noexcept {
    re_ += src.re_;
    im_ += src.im_;
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION complex& operator+=(
      const RealType& src) noexcept {
    re_ += src;
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION complex& operator-=(
      const complex<RealType>& src) noexcept {
    re_ -= src.re_;
    im_ -= src.im_;
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION complex& operator-=(
      const RealType& src) noexcept {
    re_ -= src;
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION complex& operator*=(
      const complex<RealType>& src) noexcept {
    Impl::complex_multiply(re_, im_, src.re_, src.im_, re_, im_);
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION complex& operator*=(
      const RealType& src) noexcept {
    re_ *= src;
    im_ *= src;
    return *this;
  }

  // Conditional noexcept, just in case RType throws on divide-by-zero
  constexpr KOKKOS_INLINE_FUNCTION complex& operator/=(
      const complex<RealType>& y) noexcept(noexcept(RealType{} / RealType{})) {
    Impl::complex_divide(re_, im_, y.re_, y.im_, re_, im_);
    return *this;
  }

  /// \brief Division assignment from std::complex.
  ///
  /// This operator cannot be called in a CUDA device function,
  /// because std::complex's methods and nonmember functions are not
  /// marked as CUDA device functions.
  constexpr complex& operator/=(const std::complex<RealType>& y) noexcept(
      noexcept(RealType{} / RealType{})) {
    return *this /= complex(y.real(), y.imag());
  }

  constexpr KOKKOS_INLINE_FUNCTION complex& operator/=(
      const RealType& src) noexcept(noexcept(RealType{} / RealType{})) {
    re_ /= src;
    im_ /= src;
    return *this;
  }

  template <size_t I, typename RT>
  friend constexpr const RT& get(const complex<RT>&) noexcept;

  template <size_t I, typename RT>
  friend constexpr const RT&& get(const complex<RT>&&) noexcept;
};

}  // namespace Kokkos

// Tuple protocol for complex based on https://wg21.link/P2819R2 (voted into
// the C++26 working draft on 2023-11)

template <typename RealType>
// NOLINTNEXTLINE(bugprone-std-namespace-modification)
struct std::tuple_size<Kokkos::complex<RealType>>
    : std::integral_constant<size_t, 2> {};

template <size_t I, typename RealType>
// NOLINTNEXTLINE(bugprone-std-namespace-modification)
struct std::tuple_element<I, Kokkos::complex<RealType>> {
  static_assert(I < 2);
  using type = RealType;
};

namespace Kokkos {

// get<...>(...) defined here so as not to be hidden friends, as per P2819R2

template <size_t I, typename RealType>
KOKKOS_FUNCTION constexpr RealType& get(complex<RealType>& z) noexcept {
  static_assert(I < 2);
  if constexpr (I == 0)
    return z.real();
  else
    return z.imag();
}

template <size_t I, typename RealType>
KOKKOS_FUNCTION constexpr RealType&& get(complex<RealType>&& z) noexcept {
  static_assert(I < 2);
  if constexpr (I == 0)
    return std::move(z.real());
  else
    return std::move(z.imag());
}

template <size_t I, typename RealType>
KOKKOS_FUNCTION constexpr const RealType& get(
    const complex<RealType>& z) noexcept {
  static_assert(I < 2);
  if constexpr (I == 0)
    return z.re_;
  else
    return z.im_;
}

template <size_t I, typename RealType>
KOKKOS_FUNCTION constexpr const RealType&& get(
    const complex<RealType>&& z) noexcept {
  static_assert(I < 2);
  if constexpr (I == 0)
    return std::move(z.re_);
  else
    return std::move(z.im_);
}

//==============================================================================
// <editor-fold desc="Equality and inequality"> {{{1

// Note that this is not the same behavior as std::complex, which doesn't allow
// implicit conversions, but since this is the way we had it before, we have
// to do it this way now.

//! Binary == operator for complex complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(
    complex<RealType1> const& x, complex<RealType2> const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) == common_type(y.real()) &&
         common_type(x.imag()) == common_type(y.imag());
}

// TODO (here and elsewhere) decide if we should convert to a Kokkos::complex
//      and do the comparison in a device-marked function
//! Binary == operator for std::complex complex.
template <class RealType1, class RealType2>
inline bool operator==(std::complex<RealType1> const& x,
                       complex<RealType2> const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) == common_type(y.real()) &&
         common_type(x.imag()) == common_type(y.imag());
}

//! Binary == operator for complex std::complex.
template <class RealType1, class RealType2>
inline bool operator==(complex<RealType1> const& x,
                       std::complex<RealType2> const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) == common_type(y.real()) &&
         common_type(x.imag()) == common_type(y.imag());
}

//! Binary == operator for complex real.
template <
    class RealType1, class RealType2,
    // Constraints to avoid participation in oparator==() for every possible RHS
    std::enable_if_t<std::is_convertible_v<RealType2, RealType1>, int> = 0>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(complex<RealType1> const& x,
                                                 RealType2 const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) == common_type(y) &&
         common_type(x.imag()) == common_type(0);
}

//! Binary == operator for real complex.
template <
    class RealType1, class RealType2,
    // Constraints to avoid participation in oparator==() for every possible RHS
    std::enable_if_t<std::is_convertible_v<RealType1, RealType2>, int> = 0>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(
    RealType1 const& x, complex<RealType2> const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x) == common_type(y.real()) &&
         common_type(0) == common_type(y.imag());
}

//! Binary != operator for complex complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(
    complex<RealType1> const& x, complex<RealType2> const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) != common_type(y.real()) ||
         common_type(x.imag()) != common_type(y.imag());
}

//! Binary != operator for std::complex complex.
template <class RealType1, class RealType2>
inline bool operator!=(std::complex<RealType1> const& x,
                       complex<RealType2> const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) != common_type(y.real()) ||
         common_type(x.imag()) != common_type(y.imag());
}

//! Binary != operator for complex std::complex.
template <class RealType1, class RealType2>
inline bool operator!=(complex<RealType1> const& x,
                       std::complex<RealType2> const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) != common_type(y.real()) ||
         common_type(x.imag()) != common_type(y.imag());
}

//! Binary != operator for complex real.
template <
    class RealType1, class RealType2,
    // Constraints to avoid participation in oparator==() for every possible RHS
    std::enable_if_t<std::is_convertible_v<RealType2, RealType1>, int> = 0>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(complex<RealType1> const& x,
                                                 RealType2 const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) != common_type(y) ||
         common_type(x.imag()) != common_type(0);
}

//! Binary != operator for real complex.
template <
    class RealType1, class RealType2,
    // Constraints to avoid participation in oparator==() for every possible RHS
    std::enable_if_t<std::is_convertible_v<RealType1, RealType2>, int> = 0>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(
    RealType1 const& x, complex<RealType2> const& y) noexcept {
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x) != common_type(y.real()) ||
         common_type(0) != common_type(y.imag());
}

// </editor-fold> end Equality and inequality }}}1
//==============================================================================

//! Binary + operator for complex complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator+(const complex<RealType1>& x, const complex<RealType2>& y) noexcept {
  return complex<std::common_type_t<RealType1, RealType2>>(x.real() + y.real(),
                                                           x.imag() + y.imag());
}

//! Binary + operator for complex scalar.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator+(const complex<RealType1>& x, const RealType2& y) noexcept {
  return complex<std::common_type_t<RealType1, RealType2>>(x.real() + y,
                                                           x.imag());
}

//! Binary + operator for scalar complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator+(const RealType1& x, const complex<RealType2>& y) noexcept {
  return complex<std::common_type_t<RealType1, RealType2>>(x + y.real(),
                                                           y.imag());
}

//! Unary + operator for complex.
template <class RealType>
KOKKOS_INLINE_FUNCTION constexpr complex<RealType> operator+(
    const complex<RealType>& x) noexcept {
  return complex<RealType>{+x.real(), +x.imag()};
}

//! Binary - operator for complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator-(const complex<RealType1>& x, const complex<RealType2>& y) noexcept {
  return complex<std::common_type_t<RealType1, RealType2>>(x.real() - y.real(),
                                                           x.imag() - y.imag());
}

//! Binary - operator for complex scalar.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator-(const complex<RealType1>& x, const RealType2& y) noexcept {
  return complex<std::common_type_t<RealType1, RealType2>>(x.real() - y,
                                                           x.imag());
}

//! Binary - operator for scalar complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator-(const RealType1& x, const complex<RealType2>& y) noexcept {
  return complex<std::common_type_t<RealType1, RealType2>>(x - y.real(),
                                                           -y.imag());
}

//! Unary - operator for complex.
template <class RealType>
KOKKOS_INLINE_FUNCTION constexpr complex<RealType> operator-(
    const complex<RealType>& x) noexcept {
  return complex<RealType>(-x.real(), -x.imag());
}

//! Binary * operator for complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator*(const complex<RealType1>& x, const complex<RealType2>& y) noexcept {
  using common_real_type = std::common_type_t<RealType1, RealType2>;
  common_real_type re{};
  common_real_type im{};
  Impl::complex_multiply<common_real_type>(x.real(), x.imag(), y.real(),
                                           y.imag(), re, im);
  return complex<common_real_type>(re, im);
}

/// \brief Binary * operator for std::complex and complex.
///
/// This needs to exist because template parameters can't be deduced when
/// conversions occur.  We could probably fix this using hidden friends patterns
///
/// This function cannot be called in a CUDA device function, because
/// std::complex's methods and nonmember functions are not marked as
/// CUDA device functions.
template <class RealType1, class RealType2>
constexpr complex<std::common_type_t<RealType1, RealType2>> operator*(
    const std::complex<RealType1>& x, const complex<RealType2>& y) {
  using common_real_type = std::common_type_t<RealType1, RealType2>;
  common_real_type re{};
  common_real_type im{};
  Impl::complex_multiply<common_real_type>(x.real(), x.imag(), y.real(),
                                           y.imag(), re, im);
  return complex<common_real_type>(re, im);
}

/// \brief Binary * operator for RealType times complex.
///
/// This function exists because the compiler doesn't know that
/// RealType and complex<RealType> commute with respect to operator*.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator*(const RealType1& x, const complex<RealType2>& y) noexcept {
  return complex<std::common_type_t<RealType1, RealType2>>(x * y.real(),
                                                           x * y.imag());
}

/// \brief Binary * operator for RealType times complex.
///
/// This function exists because the compiler doesn't know that
/// RealType and complex<RealType> commute with respect to operator*.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator*(const complex<RealType1>& y, const RealType2& x) noexcept {
  return complex<std::common_type_t<RealType1, RealType2>>(x * y.real(),
                                                           x * y.imag());
}

//! Imaginary part of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION constexpr RealType imag(
    const complex<RealType>& x) noexcept {
  return x.imag();
}

template <class ArithmeticType>
KOKKOS_INLINE_FUNCTION constexpr Impl::promote_t<ArithmeticType> imag(
    ArithmeticType) {
  return ArithmeticType();
}

//! Real part of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION constexpr RealType real(
    const complex<RealType>& x) noexcept {
  return x.real();
}

template <class ArithmeticType>
KOKKOS_INLINE_FUNCTION constexpr Impl::promote_t<ArithmeticType> real(
    ArithmeticType x) {
  return x;
}

//! Constructs a complex number from magnitude and phase angle
template <class T>
KOKKOS_INLINE_FUNCTION complex<T> polar(const T& r, const T& theta = T()) {
  // As for std::polar, r must be non-negative and non-NaN, and theta finite.
  KOKKOS_EXPECTS(r >= 0);
  T x = r * cos(theta);
  T y = r * sin(theta);
  // r = inf with cos(theta) or sin(theta) = 0 (theta = +-0): inf * 0 is NaN,
  // but the part is 0 (with the sign of the product), as in libc++.
  if (Kokkos::isnan(x)) x = T(0);
  if (Kokkos::isnan(y)) y = Kokkos::copysign(T(0), theta);
  return complex<T>(x, y);
}

//! Absolute value (magnitude) of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION RealType abs(const complex<RealType>& x) {
  return hypot(x.real(), x.imag());
}

//! Power of a complex number
template <class T>
KOKKOS_INLINE_FUNCTION complex<T> pow(const complex<T>& x, const T& y) {
  T r     = abs(x);
  T theta = atan2(x.imag(), x.real());
  return polar(pow(r, y), y * theta);
}

template <class T>
KOKKOS_INLINE_FUNCTION complex<T> pow(const T& x, const complex<T>& y) {
  return pow(complex<T>(x), y);
}

template <class T>
KOKKOS_INLINE_FUNCTION complex<T> pow(const complex<T>& x,
                                      const complex<T>& y) {
  return x == T() ? T() : exp(y * log(x));
}

template <class T, class U, class = std::enable_if_t<std::is_arithmetic_v<T>>>
KOKKOS_INLINE_FUNCTION complex<Impl::promote_2_t<T, U>> pow(
    const T& x, const complex<U>& y) {
  using type = Impl::promote_2_t<T, U>;
  return pow(type(x), complex<type>(y));
}

template <class T, class U, class = std::enable_if_t<std::is_arithmetic_v<U>>>
KOKKOS_INLINE_FUNCTION complex<Impl::promote_2_t<T, U>> pow(const complex<T>& x,
                                                            const U& y) {
  using type = Impl::promote_2_t<T, U>;
  return pow(complex<type>(x), type(y));
}

template <class T, class U>
KOKKOS_INLINE_FUNCTION complex<Impl::promote_2_t<T, U>> pow(
    const complex<T>& x, const complex<U>& y) {
  using type = Impl::promote_2_t<T, U>;
  return pow(complex<type>(x), complex<type>(y));
}

//! Square root of a complex number. This is intended to match the stdc++
//! implementation, which returns sqrt(z*z) = z; where z is complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> sqrt(
    const complex<RealType>& x) {
  RealType r = x.real();
  RealType i = x.imag();

  // Special values, as specified by C99 Annex G (G.6.4.2) and done by libc++.
  if (Kokkos::isinf(i)) {
    constexpr RealType inf = Kokkos::infinity_v<RealType>;
    return Kokkos::complex<RealType>(inf, i);  // also for a NaN real part
  }
  if (Kokkos::isinf(r)) {
    if (r > RealType()) {  // (+inf, y): (+inf, +-0), or (+inf, NaN)
      return Kokkos::complex<RealType>(
          r, Kokkos::isnan(i) ? i : Kokkos::copysign(RealType(), i));
    }
    // (-inf, y): (+0, +-inf), or (NaN, +-inf)
    return Kokkos::complex<RealType>(Kokkos::isnan(i) ? i : RealType(),
                                     Kokkos::copysign(r, i));
  }

  // The sign of the imaginary part follows that of i, also for i = -0, so
  // that sqrt is continuous from either side of the branch cut (C99 G.6.4.2).
  if (r == RealType()) {
    RealType t = sqrt(fabs(i) / 2);
    return Kokkos::complex<RealType>(t, Kokkos::copysign(t, i));
  } else {
    RealType t = sqrt(2 * (abs(x) + fabs(r)));
    RealType u = t / 2;
    return r > RealType()
               ? Kokkos::complex<RealType>(u, i / t)
               : Kokkos::complex<RealType>(fabs(i) / t, Kokkos::copysign(u, i));
  }
}

//! Conjugate of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION constexpr complex<RealType> conj(
    const complex<RealType>& x) noexcept {
  return {real(x), -imag(x)};
}

template <class ArithmeticType>
KOKKOS_INLINE_FUNCTION constexpr complex<Impl::promote_t<ArithmeticType>> conj(
    ArithmeticType x) {
  using type = Impl::promote_t<ArithmeticType>;
  return complex<type>(x, -type());
}

//! Norm of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION constexpr RealType norm(const complex<RealType>& x) {
  return x.real() * x.real() + x.imag() * x.imag();
}

template <class ArithmeticType>
KOKKOS_INLINE_FUNCTION constexpr Impl::promote_t<ArithmeticType> norm(
    ArithmeticType x) {
  return static_cast<Impl::promote_t<ArithmeticType>>(x) * x;
}

//! Phase angle of a complex number, in [-pi, pi].
template <class RealType>
KOKKOS_INLINE_FUNCTION RealType arg(const complex<RealType>& x) {
  return Kokkos::atan2(x.imag(), x.real());
}

//! Phase angle of a real number, as of complex(x): atan2(+0, x).
//! Integers are promoted to double; arg(-0.) is pi, as for std::arg.
template <class ArithmeticType>
KOKKOS_INLINE_FUNCTION Impl::promote_t<ArithmeticType> arg(ArithmeticType x) {
  using type = Impl::promote_t<ArithmeticType>;
  return Kokkos::atan2(type(0), static_cast<type>(x));
}

//! Projection onto the Riemann sphere: complex infinities map to
//! (inf, copysign(0, imag(x))), every other value to itself.
template <class RealType>
KOKKOS_INLINE_FUNCTION complex<RealType> proj(const complex<RealType>& x) {
  if (Kokkos::isinf(x.real()) || Kokkos::isinf(x.imag())) {
    // A local copy: binding infinity_v to the const& parameter would odr-use
    // a host-only variable in device code.
    constexpr RealType inf = Kokkos::infinity_v<RealType>;
    return complex<RealType>(inf, Kokkos::copysign(RealType(0), x.imag()));
  }
  return x;
}

//! Projection of a real number, as of complex(x): +inf for +-inf.
//! Integers are promoted to double.
template <class ArithmeticType>
KOKKOS_INLINE_FUNCTION complex<Impl::promote_t<ArithmeticType>> proj(
    ArithmeticType x) {
  return proj(complex<Impl::promote_t<ArithmeticType>>(x));
}

//! Exponential of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION complex<RealType> exp(const complex<RealType>& x) {
  const RealType r = x.real();
  RealType i       = x.imag();
  // Special values, as specified by C99 Annex G (G.6.3.1) and done by libc++.
  if (i == RealType()) {  // exp of a real number; keeps the sign of i
    return complex<RealType>(exp(r), i);
  }
  if (Kokkos::isinf(r)) {
    if (r < RealType()) {  // (-inf, y): (+-0, +-0) for y infinite or NaN
      // The signs are unspecified; copysign gives glibc's (0, -0) for -inf.
      if (!Kokkos::isfinite(i)) i = Kokkos::copysign(RealType(1), i);
    } else if (!Kokkos::isfinite(i)) {  // (+inf, inf or NaN): (+-inf, NaN)
      if (Kokkos::isinf(i)) {
        constexpr RealType nan = Kokkos::quiet_NaN_v<RealType>;
        i                      = nan;
      }
      return complex<RealType>(r, i);
    }
  }
  const RealType e = exp(r);
  return complex<RealType>(e * cos(i), e * sin(i));
}

//! natural log of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> log(
    const complex<RealType>& x) {
  RealType phi = atan2(x.imag(), x.real());
  return Kokkos::complex<RealType>(log(abs(x)), phi);
}

//! base 10 log of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> log10(
    const complex<RealType>& x) {
  return log(x) / log(RealType(10));
}

//! sine of a complex number: sin(z) = -i sinh(iz) (C99 G.6), so that it
//! inherits the special values of sinh.  The same products as
//! (sin(x) cosh(y), cos(x) sinh(y)) for finite arguments.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> sin(
    const complex<RealType>& x) {
  const Kokkos::complex<RealType> t =
      sinh(Kokkos::complex<RealType>(-x.imag(), x.real()));
  return Kokkos::complex<RealType>(t.imag(), -t.real());
}

//! cosine of a complex number: cos(z) = cosh(iz) (C99 G.6), so that it
//! inherits the special values of cosh.  The same products as
//! (cos(x) cosh(y), -sin(x) sinh(y)) for finite arguments.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> cos(
    const complex<RealType>& x) {
  return cosh(Kokkos::complex<RealType>(-x.imag(), x.real()));
}

//! tangent of a complex number: tan(z) = -i tanh(iz) (C99 G.6), so that it
//! inherits the special values and the large-argument branch of tanh.  For
//! finite arguments this is the same quotient as sin(z) / cos(z).
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> tan(
    const complex<RealType>& x) {
  const Kokkos::complex<RealType> t =
      tanh(Kokkos::complex<RealType>(-x.imag(), x.real()));
  return Kokkos::complex<RealType>(t.imag(), -t.real());
}

//! hyperbolic sine of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> sinh(
    const complex<RealType>& x) {
  const RealType r = x.real();
  const RealType i = x.imag();
  // Special values, as specified by C99 Annex G (G.6.2.5), as in libc++.
  if ((Kokkos::isinf(r) || r == RealType()) && !Kokkos::isfinite(i)) {
    // (+-inf, NaN) or (+-0, NaN).  The sign of an infinite real part is
    // unspecified; +inf as in glibc.
    constexpr RealType nan = Kokkos::quiet_NaN_v<RealType>;
    return Kokkos::complex<RealType>(Kokkos::isinf(r) ? Kokkos::fabs(r) : r,
                                     nan);
  }
  if (i == RealType() && !Kokkos::isfinite(r)) return x;  // (inf or NaN, +-0)
  return Kokkos::complex<RealType>(sinh(r) * cos(i), cosh(r) * sin(i));
}

//! hyperbolic cosine of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> cosh(
    const complex<RealType>& x) {
  const RealType r = x.real();
  const RealType i = x.imag();
  // Special values, as specified by C99 Annex G (G.6.2.4), as in libc++.
  if (Kokkos::isinf(r) && !Kokkos::isfinite(i)) {
    constexpr RealType nan = Kokkos::quiet_NaN_v<RealType>;
    return Kokkos::complex<RealType>(Kokkos::fabs(r), nan);  // (inf, NaN)
  }
  if (r == RealType() && !Kokkos::isfinite(i)) {
    // (NaN, +-0), the sign is unspecified; +0 as in glibc.
    constexpr RealType nan = Kokkos::quiet_NaN_v<RealType>;
    return Kokkos::complex<RealType>(nan, RealType());
  }
  if (i == RealType() && !Kokkos::isfinite(r)) {
    // (inf or NaN, +-0).  cosh is even and conj(cosh(z)) = cosh(conj(z)), so
    // the sign of the zero is the product of the signs of r and i.  (For
    // r = +-0 the formula below already gives (1, +-0) with that sign.)
    // For a NaN real part the sign is unspecified; i itself, not copysign:
    // the sign bit of a NaN is not portable (see atanh).
    return Kokkos::complex<RealType>(
        Kokkos::fabs(r),
        Kokkos::isnan(r) ? i : Kokkos::copysign(RealType(), r) * i);
  }
  return Kokkos::complex<RealType>(cosh(r) * cos(i), sinh(r) * sin(i));
}

//! hyperbolic tangent of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> tanh(
    const complex<RealType>& x) {
  const RealType r = x.real();
  const RealType i = x.imag();
  // Special values, as specified by C99 Annex G (G.6.2.6), as in libc++.
  if (Kokkos::isinf(r)) {
    if (!Kokkos::isfinite(i)) {
      // (+-1, +-0), the sign of the zero is unspecified; copysign as glibc.
      return Kokkos::complex<RealType>(Kokkos::copysign(RealType(1), r),
                                       Kokkos::copysign(RealType(), i));
    }
    return Kokkos::complex<RealType>(
        Kokkos::copysign(RealType(1), r),
        Kokkos::copysign(RealType(), sin(RealType(2) * i)));
  }
  if (r == RealType() && !Kokkos::isfinite(i)) {
    constexpr RealType nan = Kokkos::quiet_NaN_v<RealType>;
    return Kokkos::complex<RealType>(r, nan);  // (+-0, NaN), since C11
  }
  // A real argument (also NaN): (tanh(r), +-0) keeps the signs of zeros,
  // which the quotient below loses, and is more accurate.
  if (i == RealType()) return Kokkos::complex<RealType>(tanh(r), i);
  if (Kokkos::isnan(r)) {  // (NaN, y) for y != 0: (NaN, NaN)
    constexpr RealType nan = Kokkos::quiet_NaN_v<RealType>;
    return Kokkos::complex<RealType>(nan, nan);
  }
  const RealType si = sin(i);
  const RealType ci = cos(i);
  // Beyond (max_exponent - 1) ln 2 / 2, sinh(r)^2 overflows.  There tanh(z) is
  // (+-1, 4 sin(i) cos(i) exp(-2|r|)) to far below one ulp (the neglected
  // relative term is about 2 exp(-2|r|)), as in glibc.
  constexpr RealType r_max =
      RealType(Kokkos::Experimental::max_exponent_v<RealType> - 1) *
      Kokkos::numbers::ln2_v<RealType> / 2;
  if (Kokkos::fabs(r) > r_max) {
    return Kokkos::complex<RealType>(
        Kokkos::copysign(RealType(1), r),
        RealType(4) * si * ci * exp(-RealType(2) * Kokkos::fabs(r)));
  }
  // Kahan's formula, as in glibc:
  //   tanh(z) = (sinh(r) cosh(r) + i sin(i) cos(i)) / (sinh(r)^2 + cos(i)^2).
  // The denominator is a sum of squares, so both parts are accurate to a few
  // ulps also where sinh(z) / cosh(z) cancels (the imaginary part for
  // |r| >~ 5, or near the poles).
  RealType sh = r;
  RealType ch = RealType(1);
  if (Kokkos::fabs(r) > Kokkos::Experimental::norm_min_v<RealType>) {
    sh = sinh(r);
    ch = cosh(r);
  }
  const RealType den =
      Kokkos::fabs(sh) >
              Kokkos::fabs(ci) * Kokkos::Experimental::epsilon_v<RealType>
          ? sh * sh + ci * ci
          : ci * ci;
  return Kokkos::complex<RealType>(sh * ch / den, si * ci / den);
}

namespace Impl {

/// \brief asin and acos of x + iy for finite x >= 0, y >= 0, following
///   T. E. Hull, T. F. Fairgrieve, P. T. P. Tang, "Implementing the complex
///   arcsine and arccosine functions using exception handling", ACM TOMS 23
///   (1997) 299-335.
///
/// Gives the real parts of asin and acos and the common magnitude of their
/// imaginary parts: asin(z) = (re_asin, im), acos(z) = (re_acos, -im).  With
/// r = |z + 1|, s = |z - 1|, A = (r + s) / 2 >= 1 and B = x / A <= 1:
/// re_asin = asin(B), re_acos = acos(B), im = log(A + sqrt(A^2 - 1)), each
/// rewritten where it would lose accuracy (B near 1, A near 1) or overflow or
/// underflow (|z| huge, y tiny).  Accurate to a few ulps componentwise.
template <class T>
KOKKOS_INLINE_FUNCTION void complex_asin_acos(T x, T y, T& re_asin, T& re_acos,
                                              T& im) {
  constexpr T eps     = Kokkos::Experimental::epsilon_v<T>;
  constexpr T big     = Kokkos::Experimental::finite_max_v<T> / 4;
  constexpr T ln2     = Kokkos::numbers::ln2_v<T>;
  constexpr T b_cross = T(0.6417);  // crossover points of the paper
  constexpr T a_cross = T(1.5);
  if (x > big || y > big) {
    // r + s could overflow.  Here asin(z) = (atan2(x, y), log(2|z|)) to far
    // below one ulp; 2|z| = 4 |z / 2| avoids overflow.
    re_asin = Kokkos::atan2(x, y);
    re_acos = Kokkos::atan2(y, x);
    im      = Kokkos::log(Kokkos::hypot(x / 2, y / 2)) + 2 * ln2;
    return;
  }
  const T xp1 = x + T(1);
  const T r   = Kokkos::hypot(xp1, y);
  const T s   = Kokkos::hypot(x - T(1), y);
  const T a   = (r + s) / 2;
  const T b   = x / a;

  // Real parts.  Near B = 1, asin(B) and acos(B) are ill-conditioned; use
  // atan of a ratio whose terms are computed without cancellation.
  if (b <= b_cross) {
    re_asin = Kokkos::asin(b);
    re_acos = Kokkos::acos(b);
  } else if (x <= T(1)) {
    const T q =
        Kokkos::sqrt((a + x) / 2 * (y * y / (r + xp1) + (s + (T(1) - x))));
    re_asin = Kokkos::atan(x / q);
    re_acos = Kokkos::atan(q / x);
  } else {
    const T apx = a + x;
    const T q =
        y * Kokkos::sqrt((apx / (r + xp1) + apx / (s + (x - T(1)))) / 2);
    re_asin = Kokkos::atan(x / q);
    re_acos = Kokkos::atan(q / x);
  }

  // Imaginary part.  Near A = 1, A - 1 is computed without cancellation and
  // log1p is used.
  if (a <= a_cross) {
    if (x < T(1)) {
      // For y small compared with the distance 1 - x to the branch point,
      // im = y / sqrt(1 - x^2) with a relative error of order (y / (1 - x))^2.
      // This also avoids the underflow of y^2 for tiny y (1 - x >= eps / 2,
      // so the right-hand side does not underflow).
      if (y * y < eps * (T(1) - x) * (T(1) - x)) {
        im = y / Kokkos::sqrt((T(1) - x) * xp1);
        return;
      }
      const T am1 = (y * y / (r + xp1) + y * y / (s + (T(1) - x))) / 2;
      im          = Kokkos::log1p(am1 + Kokkos::sqrt(am1 * (a + T(1))));
    } else {
      const T am1 = (y * y / (r + xp1) + (s + (x - T(1)))) / 2;
      im          = Kokkos::log1p(am1 + Kokkos::sqrt(am1 * (a + T(1))));
    }
  } else if (a > T(1) / eps) {
    // sqrt(A^2 - 1) = A to within eps^2 (and A^2 could overflow).
    im = Kokkos::log(a) + ln2;
  } else {
    im = Kokkos::log(a + Kokkos::sqrt((a - T(1)) * (a + T(1))));
  }
}

}  // namespace Impl

//! inverse hyperbolic sine of a complex number.
//! Special values as specified by C99 Annex G (G.6.2.2), as in libc++.
//! asinh is odd and asinh(conj(z)) = conj(asinh(z)): the formula is evaluated
//! for (|x|, |y|) and the signs are restored, which makes the branch cuts
//! follow the sign of a zero real part and avoids the cancellation in
//! z + sqrt(z^2 + 1) for negative real parts.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> asinh(
    const complex<RealType>& x) {
  using C               = Kokkos::complex<RealType>;
  const RealType r      = x.real();
  const RealType i      = x.imag();
  constexpr RealType pi = Kokkos::numbers::pi_v<RealType>;
  if (Kokkos::isinf(r)) {
    if (Kokkos::isnan(i)) return x;  // (+-inf, NaN)
    if (Kokkos::isinf(i)) return C(r, Kokkos::copysign(pi / 4, i));
    return C(r, Kokkos::copysign(RealType(), i));  // (+-inf, +-0)
  }
  if (Kokkos::isnan(r)) {
    // (+-inf, NaN), the sign is unspecified; +inf as in glibc.
    if (Kokkos::isinf(i)) return C(Kokkos::fabs(i), r);
    if (i == RealType()) return x;  // (NaN, +-0)
    return C(r, r);
  }
  if (Kokkos::isinf(i)) {
    return C(Kokkos::copysign(i, r), Kokkos::copysign(pi / 2, i));
  }
  // On the axes the result is exact with the real functions: a real argument,
  // and the imaginary axis between -i and i, where the real part is +-0.
  if (i == RealType()) return C(asinh(r), i);
  if (r == RealType() && Kokkos::fabs(i) <= RealType(1)) return C(r, asin(i));
  // asinh(x + iy) = i asin(y - ix): for x, y >= 0 that is (im, re_asin) of
  // the Hull-Fairgrieve-Tang algorithm at (y, x).
  RealType re_asin;
  RealType re_acos;
  RealType im;
  Impl::complex_asin_acos(Kokkos::fabs(i), Kokkos::fabs(r), re_asin, re_acos,
                          im);
  return C(Kokkos::copysign(im, r), Kokkos::copysign(re_asin, i));
}

//! inverse hyperbolic cosine of a complex number.
//! Special values as specified by C99 Annex G (G.6.2.1), as in libc++.
//! acosh(conj(z)) = conj(acosh(z)) and its real part is >= 0: the formula is
//! evaluated for (x, |y|), so that the branch cut on the real axis below 1
//! follows the sign of a zero imaginary part.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> acosh(
    const complex<RealType>& x) {
  using C               = Kokkos::complex<RealType>;
  const RealType r      = x.real();
  const RealType i      = x.imag();
  constexpr RealType pi = Kokkos::numbers::pi_v<RealType>;
  if (Kokkos::isinf(r)) {
    if (Kokkos::isnan(i)) return C(Kokkos::fabs(r), i);  // (+inf, NaN)
    if (Kokkos::isinf(i)) {
      return C(Kokkos::fabs(r),
               Kokkos::copysign(r > RealType() ? pi / 4 : 3 * pi / 4, i));
    }
    return C(Kokkos::fabs(r),
             Kokkos::copysign(r < RealType() ? pi : RealType(), i));
  }
  if (Kokkos::isnan(r)) {
    if (Kokkos::isinf(i)) return C(Kokkos::fabs(i), r);  // (+inf, NaN)
    return C(r, r);
  }
  if (Kokkos::isinf(i)) return C(Kokkos::fabs(i), Kokkos::copysign(pi / 2, i));
  if (r == RealType() && Kokkos::isnan(i)) return C(i, pi / 2);  // as glibc
  // On the real axis the result is exact with the real functions: (+0, +-acos)
  // between -1 and 1, (acosh, +-0) beyond 1.
  if (i == RealType() && Kokkos::fabs(r) <= RealType(1)) {
    return C(RealType(), Kokkos::copysign(acos(r), i));
  }
  if (i == RealType() && r > RealType(1)) return C(acosh(r), i);
  // acosh(z) = (-imag(acos(z)), real(acos(z))) for y >= 0, and
  // acos(-z) = pi - acos(z): the Hull-Fairgrieve-Tang algorithm at (|x|, |y|).
  RealType re_asin;
  RealType re_acos;
  RealType im;
  Impl::complex_asin_acos(Kokkos::fabs(r), Kokkos::fabs(i), re_asin, re_acos,
                          im);
  return C(im, Kokkos::copysign(r < RealType() ? pi - re_acos : re_acos, i));
}

//! inverse hyperbolic tangent of a complex number.
//! Special values as specified by C99 Annex G (G.6.2.3), as in libc++.
//! atanh is odd and atanh(conj(z)) = conj(atanh(z)): the formula is evaluated
//! for (|x|, |y|) and the signs are restored, so that the branch cuts on the
//! real axis beyond +-1 follow the sign of a zero imaginary part.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> atanh(
    const complex<RealType>& x) {
  using C                = Kokkos::complex<RealType>;
  const RealType xr      = x.real();
  const RealType xi      = x.imag();
  constexpr RealType pi  = Kokkos::numbers::pi_v<RealType>;
  constexpr RealType inf = Kokkos::infinity_v<RealType>;
  if (Kokkos::isinf(xi)) {
    // (+-0, +-pi/2).  For a NaN real part the sign of the zero is unspecified;
    // +0, not copysign: the sign bit of a NaN is not portable (negating a NaN,
    // as the identity atan(z) = -i atanh(iz) does, keeps the sign on x86 but
    // may give a canonical NaN on GPUs).
    return C(Kokkos::isnan(xr) ? RealType() : Kokkos::copysign(RealType(), xr),
             Kokkos::copysign(pi / 2, xi));
  }
  if (Kokkos::isnan(xi)) {
    if (Kokkos::isinf(xr) || xr == RealType()) {
      return C(Kokkos::copysign(RealType(), xr), xi);  // (+-0, NaN)
    }
    return C(xi, xi);
  }
  if (Kokkos::isnan(xr)) return C(xr, xr);
  if (Kokkos::isinf(xr)) {
    return C(Kokkos::copysign(RealType(), xr), Kokkos::copysign(pi / 2, xi));
  }
  if (Kokkos::fabs(xr) == RealType(1) && xi == RealType()) {
    return C(Kokkos::copysign(inf, xr), Kokkos::copysign(RealType(), xi));
  }
  // W. Kahan, "Branch cuts for complex elementary functions" (1987), for
  // a, b >= 0:  real = log1p(4a / ((1 - a)^2 + b^2)) / 4,
  //             imag = atan2(2b, (1 - a)(1 + a) - b^2) / 2.
  // 1 - a is exact near a = 1, and no term cancels except where the result
  // is insensitive to it.
  const RealType a       = Kokkos::fabs(xr);
  const RealType b       = Kokkos::fabs(xi);
  constexpr RealType eps = Kokkos::Experimental::epsilon_v<RealType>;
  RealType re;
  RealType im;
  if (a > RealType(1) / eps || b > RealType(1) / eps) {
    // atanh(z) = (a / |z|^2, pi/2) to within eps; the squares could overflow.
    const RealType h = Kokkos::hypot(a, b);
    re               = (a / h) / h;
    im               = pi / 2;
  } else {
    const RealType am1 = RealType(1) - a;
    re = Kokkos::log1p(RealType(4) * a / (am1 * am1 + b * b)) / RealType(4);
    im = Kokkos::atan2(RealType(2) * b, am1 * (RealType(1) + a) - b * b) /
         RealType(2);
  }
  return C(Kokkos::copysign(re, xr), Kokkos::copysign(im, xi));
}

//! inverse sine of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> asin(
    const complex<RealType>& x) {
  Kokkos::complex<RealType> t =
      asinh(Kokkos::complex<RealType>(-x.imag(), x.real()));
  return Kokkos::complex<RealType>(t.imag(), -t.real());
}

//! inverse cosine of a complex number.
//! Special values as specified by C99 Annex G (G.6.1.1), as in libc++; the
//! other values as pi/2 - asin(z), which inherits the branch cuts of asin.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> acos(
    const complex<RealType>& x) {
  using C               = Kokkos::complex<RealType>;
  const RealType r      = x.real();
  const RealType i      = x.imag();
  constexpr RealType pi = Kokkos::numbers::pi_v<RealType>;
  if (Kokkos::isinf(r)) {
    // (NaN, +-inf), the sign is unspecified; -inf as in glibc.
    if (Kokkos::isnan(i)) return C(i, -Kokkos::fabs(r));
    if (Kokkos::isinf(i)) return C(r < RealType() ? 3 * pi / 4 : pi / 4, -i);
    if (r < RealType()) return C(pi, Kokkos::signbit(i) ? -r : r);
    return C(RealType(), Kokkos::signbit(i) ? r : -r);
  }
  if (Kokkos::isnan(r)) {
    if (Kokkos::isinf(i)) return C(r, -i);
    return C(r, r);
  }
  if (Kokkos::isinf(i)) return C(pi / 2, -i);
  if (r == RealType() && (i == RealType() || Kokkos::isnan(i))) {
    return C(pi / 2, -i);
  }
  // The Hull-Fairgrieve-Tang algorithm at (|x|, |y|), with
  // acos(-z) = pi - acos(z) and acos(conj(z)) = conj(acos(z)).
  RealType re_asin;
  RealType re_acos;
  RealType im;
  Impl::complex_asin_acos(Kokkos::fabs(r), Kokkos::fabs(i), re_asin, re_acos,
                          im);
  return C(r < RealType() ? pi - re_acos : re_acos, -Kokkos::copysign(im, i));
}

//! inverse tangent of a complex number: atan(z) = -i atanh(iz) (C99 G.6), so
//! that it inherits the special values and branch cuts of atanh.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> atan(
    const complex<RealType>& x) {
  const Kokkos::complex<RealType> t =
      atanh(Kokkos::complex<RealType>(-x.imag(), x.real()));
  return Kokkos::complex<RealType>(t.imag(), -t.real());
}

/// This function cannot be called in a CUDA device function,
/// because std::complex's methods and nonmember functions are not
/// marked as CUDA device functions.
template <class RealType>
inline complex<RealType> exp(const std::complex<RealType>& c) {
  return complex<RealType>(std::exp(c.real()) * std::cos(c.imag()),
                           std::exp(c.real()) * std::sin(c.imag()));
}

//! Binary operator / for complex and real numbers
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator/(const complex<RealType1>& x,
          const RealType2& y) noexcept(noexcept(RealType1{} / RealType2{})) {
  return complex<std::common_type_t<RealType1, RealType2>>(real(x) / y,
                                                           imag(x) / y);
}

//! Binary operator / for complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator/(const complex<RealType1>& x,
          const complex<RealType2>& y) noexcept(noexcept(RealType1{} /
                                                         RealType2{})) {
  // All of the computation is done in the common type.
  using common_real_type = std::common_type_t<RealType1, RealType2>;
  common_real_type re{};
  common_real_type im{};
  Impl::complex_divide<common_real_type>(x.real(), x.imag(), y.real(), y.imag(),
                                         re, im);
  return complex<common_real_type>(re, im);
}

//! Binary operator / for complex and real numbers
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION constexpr complex<
    std::common_type_t<RealType1, RealType2>>
operator/(const RealType1& x,
          const complex<RealType2>& y) noexcept(noexcept(RealType1{} /
                                                         RealType2{})) {
  return complex<std::common_type_t<RealType1, RealType2>>(x) / y;
}

// Stream I/O delegates to std::complex, so Kokkos::complex is written and read
// exactly as the standard library does for std::complex.  These overloads are
// still needed: std::complex's operators are templates on complex<T>, and T
// cannot be deduced from Kokkos::complex (implicit conversions are not
// considered during deduction), and operator>> needs an lvalue std::complex.

/// \brief Insertion operator: writes "(real,imag)" as std::complex does.
template <class RealType, class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os, const complex<RealType>& x) {
  return os << std::complex<RealType>(x.real(), x.imag());
}

/// \brief Extraction operator: reads "r", "(r)" or "(r,i)" as std::complex
///   does.  On failure, failbit is set and \c x is left unchanged.
template <class RealType, class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(
    std::basic_istream<CharT, Traits>& is, complex<RealType>& x) {
  std::complex<RealType> x_std;
  if (is >> x_std) x = complex<RealType>(x_std.real(), x_std.imag());
  return is;
}

template <class T>
struct reduction_identity<Kokkos::complex<T>> {
  KOKKOS_FUNCTION static Kokkos::complex<T> sum() noexcept { return 0; }
  KOKKOS_FUNCTION static Kokkos::complex<T> prod() noexcept { return 1; }
};

}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_COMPLEX
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_COMPLEX
#endif
#endif  // KOKKOS_COMPLEX_HPP
