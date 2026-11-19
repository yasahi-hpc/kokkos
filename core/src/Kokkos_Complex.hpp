// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOS_COMPLEX_HPP
#define KOKKOS_COMPLEX_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_COMPLEX
#endif

#include <Kokkos_Atomic.hpp>
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

/// \brief Complex division (a + ib) / (c + id) = re + i im.
///
/// Smith's algorithm, with the range guards of libgcc's __divdc3 (GCC >= 12),
/// followed by the C99 Annex G (G.5.2) recovery of the infinities and zeros
/// that the formula computes as NaN + iNaN: nonzero / zero, infinite / finite
/// and finite / infinite.  No test is made up front, so finite operands only
/// pay for the comparisons of the range guards.
template <class T>
KOKKOS_INLINE_FUNCTION void complex_divide(T a, T b, T c, T d, T& re,
                                           T& im) noexcept(noexcept(T{} /
                                                                    T{})) {
  // Thresholds of libgcc's RBIG, RMIN, RMIN2, RMINSCAL and RMAX2.
  constexpr T rbig     = Kokkos::finite_max_v<T> / 2;
  constexpr T rmin     = Kokkos::norm_min_v<T>;
  constexpr T rmin2    = Kokkos::epsilon_v<T>;
  constexpr T rminscal = T(1) / Kokkos::epsilon_v<T>;
  constexpr T rmax2    = rbig * rmin2;

  // The operands as given, for the NaN recovery below.
  const T a0 = a;
  const T b0 = b;
  const T c0 = c;
  const T d0 = d;

  T x;
  T y;
  if (Kokkos::fabs(c) < Kokkos::fabs(d)) {
    // Prevent underflow when the denominator is near the largest value.
    if (Kokkos::fabs(d) >= rbig) {
      a /= 2;
      b /= 2;
      c /= 2;
      d /= 2;
    }
    // Scale up small operands to avoid underflow; no new overflow is possible.
    if (Kokkos::fabs(d) < rmin2 ||
        (Kokkos::fabs(a) < rmin && Kokkos::fabs(b) < rmax2 &&
         Kokkos::fabs(d) < rmax2) ||
        (Kokkos::fabs(b) < rmin && Kokkos::fabs(a) < rmax2 &&
         Kokkos::fabs(d) < rmax2)) {
      a *= rminscal;
      b *= rminscal;
      c *= rminscal;
      d *= rminscal;
    }
    const T ratio = c / d;
    const T denom = c * ratio + d;
    // Choose another order of computation if ratio is subnormal.
    if (Kokkos::fabs(ratio) > rmin) {
      x = (a * ratio + b) / denom;
      y = (b * ratio - a) / denom;
    } else {
      x = (c * (a / d) + b) / denom;
      y = (c * (b / d) - a) / denom;
    }
  } else {
    if (Kokkos::fabs(c) >= rbig) {
      a /= 2;
      b /= 2;
      c /= 2;
      d /= 2;
    }
    if (Kokkos::fabs(c) < rmin2 ||
        (Kokkos::fabs(a) < rmin && Kokkos::fabs(b) < rmax2 &&
         Kokkos::fabs(c) < rmax2) ||
        (Kokkos::fabs(b) < rmin && Kokkos::fabs(a) < rmax2 &&
         Kokkos::fabs(c) < rmax2)) {
      a *= rminscal;
      b *= rminscal;
      c *= rminscal;
      d *= rminscal;
    }
    const T ratio = d / c;
    const T denom = d * ratio + c;
    if (Kokkos::fabs(ratio) > rmin) {
      x = (b * ratio + a) / denom;
      y = (b - a * ratio) / denom;
    } else {
      x = (a + d * (b / c)) / denom;
      y = (b - d * (a / c)) / denom;
    }
  }

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
    const RealType realPart = re_ * src.re_ - im_ * src.im_;
    const RealType imagPart = re_ * src.im_ + im_ * src.re_;
    re_                     = realPart;
    im_                     = imagPart;
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
  return complex<std::common_type_t<RealType1, RealType2>>(
      x.real() * y.real() - x.imag() * y.imag(),
      x.real() * y.imag() + x.imag() * y.real());
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
  return complex<std::common_type_t<RealType1, RealType2>>(
      x.real() * y.real() - x.imag() * y.imag(),
      x.real() * y.imag() + x.imag() * y.real());
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
  KOKKOS_EXPECTS(r >= 0);
  return complex<T>(r * cos(theta), r * sin(theta));
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

  if (r == RealType()) {
    RealType t = sqrt(fabs(i) / 2);
    return Kokkos::complex<RealType>(t, i < RealType() ? -t : t);
  } else {
    RealType t = sqrt(2 * (abs(x) + fabs(r)));
    RealType u = t / 2;
    return r > RealType() ? Kokkos::complex<RealType>(u, i / t)
                          : Kokkos::complex<RealType>(fabs(i) / t,
                                                      i < RealType() ? -u : u);
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

//! Exponential of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION complex<RealType> exp(const complex<RealType>& x) {
  return exp(x.real()) * complex<RealType>(cos(x.imag()), sin(x.imag()));
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

//! sine of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> sin(
    const complex<RealType>& x) {
  return Kokkos::complex<RealType>(sin(x.real()) * cosh(x.imag()),
                                   cos(x.real()) * sinh(x.imag()));
}

//! cosine of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> cos(
    const complex<RealType>& x) {
  return Kokkos::complex<RealType>(cos(x.real()) * cosh(x.imag()),
                                   -sin(x.real()) * sinh(x.imag()));
}

//! tangent of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> tan(
    const complex<RealType>& x) {
  return sin(x) / cos(x);
}

//! hyperbolic sine of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> sinh(
    const complex<RealType>& x) {
  return Kokkos::complex<RealType>(sinh(x.real()) * cos(x.imag()),
                                   cosh(x.real()) * sin(x.imag()));
}

//! hyperbolic cosine of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> cosh(
    const complex<RealType>& x) {
  return Kokkos::complex<RealType>(cosh(x.real()) * cos(x.imag()),
                                   sinh(x.real()) * sin(x.imag()));
}

//! hyperbolic tangent of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> tanh(
    const complex<RealType>& x) {
  return sinh(x) / cosh(x);
}

//! inverse hyperbolic sine of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> asinh(
    const complex<RealType>& x) {
  return log(x + sqrt(x * x + RealType(1.0)));
}

//! inverse hyperbolic cosine of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> acosh(
    const complex<RealType>& x) {
  return RealType(2.0) * log(sqrt(RealType(0.5) * (x + RealType(1.0))) +
                             sqrt(RealType(0.5) * (x - RealType(1.0))));
}

//! inverse hyperbolic tangent of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> atanh(
    const complex<RealType>& x) {
  const RealType i2 = x.imag() * x.imag();
  const RealType r  = RealType(1.0) - i2 - x.real() * x.real();

  RealType p = RealType(1.0) + x.real();
  RealType m = RealType(1.0) - x.real();

  p = i2 + p * p;
  m = i2 + m * m;

  RealType phi = atan2(RealType(2.0) * x.imag(), r);
  return Kokkos::complex<RealType>(RealType(0.25) * (log(p) - log(m)),
                                   RealType(0.5) * phi);
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
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> acos(
    const complex<RealType>& x) {
  Kokkos::complex<RealType> t = asin(x);
  RealType pi_2               = acos(RealType(0.0));
  return Kokkos::complex<RealType>(pi_2 - t.real(), -t.imag());
}

//! inverse tangent of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> atan(
    const complex<RealType>& x) {
  const RealType r2 = x.real() * x.real();
  const RealType i  = RealType(1.0) - r2 - x.imag() * x.imag();

  RealType p = x.imag() + RealType(1.0);
  RealType m = x.imag() - RealType(1.0);

  p = r2 + p * p;
  m = r2 + m * m;

  return Kokkos::complex<RealType>(
      RealType(0.5) * atan2(RealType(2.0) * x.real(), i),
      RealType(0.25) * log(p / m));
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
KOKKOS_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
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
KOKKOS_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
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
