// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HALF_FLOATING_POINT_WRAPPER_HPP_
#define KOKKOS_HALF_FLOATING_POINT_WRAPPER_HPP_

#include <Kokkos_Macros.hpp>
#include <Kokkos_BitManipulation.hpp>  // bit_cast

#include <type_traits>
#include <istream>  // extraction and insertion operators
#include <iterator>
#include <locale>
#include <ostream>
#include <cstdint>

#if defined(KOKKOS_ENABLE_CUDA)
#include <Cuda/Kokkos_Cuda_Half_Impl_Type.hpp>
#elif defined(KOKKOS_ENABLE_HIP)
#include <HIP/Kokkos_HIP_Half_Impl_Type.hpp>
#elif defined(KOKKOS_ENABLE_SYCL)
#include <SYCL/Kokkos_SYCL_Half_Impl_Type.hpp>
#else
#include <impl/Kokkos_STD_Half_Impl_Type.hpp>
#endif

// constexpr half_t and bhalf_t.
// - std backend (B4a): std::float16_t and std::bfloat16_t are arithmetic
//   types, so every conversion, operator and comparison of the wrapper is a
//   core-language constant expression.
// - CUDA (B4b-CUDA): __half and __nv_bfloat16 are not constexpr (only their
//   __half_raw constructors are). During constant evaluation the cast
//   functions use the software conversions of Impl::half_constexpr_* below
//   (round to nearest even, once, from the source type), and the operators
//   that use the CUDA types' own arithmetic (KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH,
//   device pass) compute in float, rounding once, which gives the same result
//   for + - * / and the conversions. At run time nothing changes.
// - SYCL (B4b-SYCL): as CUDA. sycl::half is constexpr but
//   sycl::ext::oneapi::bfloat16 is not, so the cast functions use the same
//   software conversions during constant evaluation (device pass), and always
//   in the host pass (below).
// - HIP: not constexpr (B4b-HIP: future work).
// - SYCL host: DPC++'s host conversion sycl::half(float) rounds wrongly below
//   2^-13 (it truncates, and ties do not go to even; seen with oneAPI 2024.0
//   and 2026.1), and sycl::half's host arithmetic converts back through it.
//   So in the SYCL host pass the cast functions round with the same software
//   conversions, and the operators compute in float. The device keeps the
//   hardware conversions.
// KOKKOS_IMPL_HALF_CONSTEXPR marks the functions (every declaration of the
// cast functions too, as C++ requires of constexpr functions);
// KOKKOS_IMPL_HALF_IS_CONSTEXPR tells tests (and CH5) that half values can
// appear in constant expressions; KOKKOS_IMPL_HALF_SOFTWARE_PATH() is true
// where the software conversions and the float path are used: during constant
// evaluation on CUDA and in the SYCL device pass, and always in the SYCL host
// pass.
#if defined(KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT) || \
    defined(KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT)
#define KOKKOS_IMPL_HALF_IS_CONSTEXPR
#define KOKKOS_IMPL_HALF_CONSTEXPR constexpr
#define KOKKOS_IMPL_HALF_SOFTWARE_PATH() false
#elif defined(KOKKOS_ENABLE_CUDA)
#define KOKKOS_IMPL_HALF_IS_CONSTEXPR
#define KOKKOS_IMPL_HALF_CONSTEXPR constexpr
#define KOKKOS_IMPL_HALF_SOFTWARE_PATH() __builtin_is_constant_evaluated()
#elif defined(KOKKOS_ENABLE_SYCL)
#define KOKKOS_IMPL_HALF_IS_CONSTEXPR
#define KOKKOS_IMPL_HALF_CONSTEXPR constexpr
#ifdef __SYCL_DEVICE_ONLY__
#define KOKKOS_IMPL_HALF_SOFTWARE_PATH() __builtin_is_constant_evaluated()
#else
#define KOKKOS_IMPL_HALF_SOFTWARE_PATH() true
#endif
#else
#define KOKKOS_IMPL_HALF_CONSTEXPR
#define KOKKOS_IMPL_HALF_SOFTWARE_PATH() false
#endif

namespace Kokkos::Experimental::Impl {
/// @brief templated struct for determining if half_t is an alias to float.
/// @tparam T The type to specialize on.
template <class T>
struct is_float16 : std::false_type {};

/// @brief templated struct for determining if bhalf_t is an alias to float.
/// @tparam T The type to specialize on.
template <class T>
struct is_bfloat16 : std::false_type {};
}  // namespace Kokkos::Experimental::Impl

#if !KOKKOS_HALF_T_IS_FLOAT

/************************* BEGIN forward declarations *************************/
namespace Kokkos {
namespace Experimental {
namespace Impl {
template <class FloatType>
class floating_point_wrapper;
}

// Declare half_t (binary16)
using half_t = Kokkos::Experimental::Impl::floating_point_wrapper<
    Kokkos::Impl::half_impl_t::type>;
namespace Impl {
template <>
struct is_float16<half_t> : std::true_type {};
}  // namespace Impl
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(float val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(bool val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(double val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(short val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(int val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(long val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(long long val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(unsigned short val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(unsigned int val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(unsigned long val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(unsigned long long val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(half_t);

template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, float>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, bool>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, double>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, short>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, int>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, long>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, long long>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned short>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned int>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned long>, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned long long>, T>
        cast_from_half(half_t);

// declare bhalf_t
#if !KOKKOS_BHALF_T_IS_FLOAT
using bhalf_t = Kokkos::Experimental::Impl::floating_point_wrapper<
    Kokkos::Impl ::bhalf_impl_t ::type>;
namespace Impl {
template <>
struct is_bfloat16<bhalf_t> : std::true_type {};
}  // namespace Impl
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(float val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(bool val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(double val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(short val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(int val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(long val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(long long val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(unsigned short val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(unsigned int val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(unsigned long val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(unsigned long long val);
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(bhalf_t val);

template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, float>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, bool>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, double>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, short>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, int>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, long>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, long long>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned short>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned int>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned long>, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned long long>, T>
        cast_from_bhalf(bhalf_t);
#endif

template <class T>
static KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    Kokkos::Experimental::half_t
    cast_to_wrapper(T x, const Kokkos::Impl::half_impl_t::type&);

#if !KOKKOS_BHALF_T_IS_FLOAT
template <class T>
static KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    Kokkos::Experimental::bhalf_t
    cast_to_wrapper(T x, const Kokkos::Impl::bhalf_impl_t::type&);
#endif

template <class T>
static KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR T
cast_from_wrapper(const Kokkos::Experimental::half_t& x);

#if !KOKKOS_BHALF_T_IS_FLOAT
template <class T>
static KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR T
cast_from_wrapper(const Kokkos::Experimental::bhalf_t& x);
#endif
/************************** END forward declarations **************************/

namespace Impl {

template <typename FloatType>
struct BitComparisonWrapper {
  std::uint16_t value;

  template <typename Number>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend bool operator==(
      BitComparisonWrapper a, Number b) {
    return static_cast<FloatType>(a) == b;
  }

  template <typename Number>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend bool operator==(
      Number b, BitComparisonWrapper a) {
    return static_cast<FloatType>(a) == b;
  }

  template <typename Number>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend bool operator!=(
      BitComparisonWrapper a, Number b) {
    return static_cast<FloatType>(a) != b;
  }

  template <typename Number>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend bool operator<(
      BitComparisonWrapper a, Number b) {
    return static_cast<FloatType>(a) < b;
  }

  template <typename Number>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend bool operator<=(
      BitComparisonWrapper a, Number b) {
    return static_cast<FloatType>(a) <= b;
  }

  template <typename Number>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend bool operator>(
      BitComparisonWrapper a, Number b) {
    return static_cast<FloatType>(a) > b;
  }

  template <typename Number>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend bool operator>=(
      BitComparisonWrapper a, Number b) {
    return static_cast<FloatType>(a) >= b;
  }
};

template <typename FloatType>
inline constexpr BitComparisonWrapper<FloatType> exponent_mask;
template <typename FloatType>
inline constexpr BitComparisonWrapper<FloatType> fraction_mask;

#if !KOKKOS_HALF_T_IS_FLOAT
template <>
inline constexpr BitComparisonWrapper<Kokkos::Experimental::half_t>
    exponent_mask<Kokkos::Experimental::half_t>{0b0'11111'0000000000};
template <>
inline constexpr BitComparisonWrapper<Kokkos::Experimental::half_t>
    fraction_mask<Kokkos::Experimental::half_t>{0b0'00000'1111111111};
#endif

#if !KOKKOS_BHALF_T_IS_FLOAT
template <>
inline constexpr BitComparisonWrapper<Kokkos::Experimental::bhalf_t>
    exponent_mask<Kokkos::Experimental::bhalf_t>{0b0'11111111'0000000};
template <>
inline constexpr BitComparisonWrapper<Kokkos::Experimental::bhalf_t>
    fraction_mask<Kokkos::Experimental::bhalf_t>{0b0'00000000'1111111};
#endif

template <class FloatType>
class alignas(FloatType) floating_point_wrapper {
 public:
  using impl_type           = FloatType;
  using bit_comparison_type = BitComparisonWrapper<floating_point_wrapper>;

 private:
  impl_type val;

 public:
  // In-class initialization and defaulted default constructors not used
  // since Cuda supports half precision initialization via the below constructor
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper() : val() {}

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
  floating_point_wrapper(bit_comparison_type rhs)
#ifdef KOKKOS_IMPL_HALF_IS_CONSTEXPR
      : val(__builtin_bit_cast(impl_type, rhs)){}
#else
  {
    val = Kokkos::bit_cast<impl_type>(rhs);
  }
#endif

        // Don't support implicit conversion back to impl_type.
        // impl_type is a storage only type on host.
        KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit
        operator impl_type() const {
    return val;
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator float() const {
    return cast_from_wrapper<float>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator bool() const {
    return cast_from_wrapper<bool>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator double() const {
    return cast_from_wrapper<double>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator short() const {
    return cast_from_wrapper<short>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator int() const {
    return cast_from_wrapper<int>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator long() const {
    return cast_from_wrapper<long>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator long long()
      const {
    return cast_from_wrapper<long long>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator unsigned short()
      const {
    return cast_from_wrapper<unsigned short>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator unsigned int()
      const {
    return cast_from_wrapper<unsigned int>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit operator unsigned long()
      const {
    return cast_from_wrapper<unsigned long>(*this);
  }
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit
  operator unsigned long long() const {
    return cast_from_wrapper<unsigned long long>(*this);
  }

  /**
   * Conversion constructors.
   *
   * Support implicit conversions from impl_type, float, double ->
   * floating_point_wrapper. Mixed precision expressions require upcasting which
   * is done in the
   * "// Binary Arithmetic" operator overloads below.
   *
   * Support implicit conversions from integral types -> floating_point_wrapper.
   * Expressions involving floating_point_wrapper with integral types require
   * downcasting the integral types to floating_point_wrapper. Existing operator
   * overloads can handle this with the addition of the below implicit
   * conversion constructors.
   */
  KOKKOS_FUNCTION
  constexpr floating_point_wrapper(impl_type rhs) : val(rhs) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper(float rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper(double rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR explicit floating_point_wrapper(
      bool rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper(short rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper(int rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper(long rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
  floating_point_wrapper(long long rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
  floating_point_wrapper(unsigned short rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
  floating_point_wrapper(unsigned int rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
  floating_point_wrapper(unsigned long rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
  floating_point_wrapper(unsigned long long rhs)
      : val(cast_to_wrapper(rhs, val).val) {}

  // Unary operators
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper
  operator+() const {
    floating_point_wrapper tmp = *this;
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      tmp.val = +tmp.val;
    } else
#endif
    {
      tmp.val = cast_to_wrapper(+cast_from_wrapper<float>(tmp), val).val;
    }
    return tmp;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper
  operator-() const {
    floating_point_wrapper tmp = *this;
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      tmp.val = -tmp.val;
    } else
#endif
    {
      tmp.val = cast_to_wrapper(-cast_from_wrapper<float>(tmp), val).val;
    }
    return tmp;
  }

  // Prefix operators
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper&
  operator++() {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = val + impl_type(1.0F);  // cuda has no operator++ for __nv_bfloat
    } else
#endif
    {
      float tmp = cast_from_wrapper<float>(*this);
      ++tmp;
      val = cast_to_wrapper(tmp, val).val;
    }
    return *this;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper&
  operator--() {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = val - impl_type(1.0F);  // cuda has no operator-- for __nv_bfloat
    } else
#endif
    {
      float tmp = cast_from_wrapper<float>(*this);
      --tmp;
      val = cast_to_wrapper(tmp, val).val;
    }
    return *this;
  }

  // Postfix operators
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper
  operator++(int) {
    floating_point_wrapper tmp = *this;
    operator++();
    return tmp;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper
  operator--(int) {
    floating_point_wrapper tmp = *this;
    operator--();
    return tmp;
  }

  // Binary operators
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator=(
      impl_type rhs) {
    val = rhs;
    return *this;
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator=(
      T rhs) {
    val = cast_to_wrapper(rhs, val).val;
    return *this;
  }

  // Compound operators
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator+=(
      floating_point_wrapper rhs) {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = val + rhs.val;  // cuda has no operator+= for __nv_bfloat
    } else
#endif
    {
      val = cast_to_wrapper(
                cast_from_wrapper<float>(*this) + cast_from_wrapper<float>(rhs),
                val)
                .val;
    }
    return *this;
  }

  // Compound operators: upcast overloads for +=
  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator+=(T& lhs, floating_point_wrapper rhs) {
    lhs += static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator+=(
      float rhs) {
    if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = cast_to_wrapper(cast_from_wrapper<float>(*this) + rhs, val).val;
      return *this;
    }
    float result = static_cast<float>(val) + rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator+=(
      double rhs) {
    if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = cast_to_wrapper(cast_from_wrapper<double>(*this) + rhs, val).val;
      return *this;
    }
    double result = static_cast<double>(val) + rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator-=(
      floating_point_wrapper rhs) {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = val - rhs.val;  // cuda has no operator-= for __nv_bfloat
    } else
#endif
    {
      val = cast_to_wrapper(
                cast_from_wrapper<float>(*this) - cast_from_wrapper<float>(rhs),
                val)
                .val;
    }
    return *this;
  }

  // Compund operators: upcast overloads for -=
  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator-=(T& lhs, floating_point_wrapper rhs) {
    lhs -= static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator-=(
      float rhs) {
    if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = cast_to_wrapper(cast_from_wrapper<float>(*this) - rhs, val).val;
      return *this;
    }
    float result = static_cast<float>(val) - rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator-=(
      double rhs) {
    if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = cast_to_wrapper(cast_from_wrapper<double>(*this) - rhs, val).val;
      return *this;
    }
    double result = static_cast<double>(val) - rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator*=(
      floating_point_wrapper rhs) {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = val * rhs.val;  // cuda has no operator*= for __nv_bfloat
    } else
#endif
    {
      val = cast_to_wrapper(
                cast_from_wrapper<float>(*this) * cast_from_wrapper<float>(rhs),
                val)
                .val;
    }
    return *this;
  }

  // Compund operators: upcast overloads for *=
  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator*=(T& lhs, floating_point_wrapper rhs) {
    lhs *= static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator*=(
      float rhs) {
    if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = cast_to_wrapper(cast_from_wrapper<float>(*this) * rhs, val).val;
      return *this;
    }
    float result = static_cast<float>(val) * rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator*=(
      double rhs) {
    if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = cast_to_wrapper(cast_from_wrapper<double>(*this) * rhs, val).val;
      return *this;
    }
    double result = static_cast<double>(val) * rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator/=(
      floating_point_wrapper rhs) {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = val / rhs.val;  // cuda has no operator/= for __nv_bfloat
    } else
#endif
    {
      val = cast_to_wrapper(
                cast_from_wrapper<float>(*this) / cast_from_wrapper<float>(rhs),
                val)
                .val;
    }
    return *this;
  }

  // Compund operators: upcast overloads for /=
  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator/=(T& lhs, floating_point_wrapper rhs) {
    lhs /= static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator/=(
      float rhs) {
    if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = cast_to_wrapper(cast_from_wrapper<float>(*this) / rhs, val).val;
      return *this;
    }
    float result = static_cast<float>(val) / rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR floating_point_wrapper& operator/=(
      double rhs) {
    if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      val = cast_to_wrapper(cast_from_wrapper<double>(*this) / rhs, val).val;
      return *this;
    }
    double result = static_cast<double>(val) / rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  // Binary Arithmetic
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend floating_point_wrapper
  operator+(floating_point_wrapper lhs, floating_point_wrapper rhs) {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      lhs += rhs;
    } else
#endif
    {
      lhs.val = cast_to_wrapper(cast_from_wrapper<float>(lhs) +
                                    cast_from_wrapper<float>(rhs),
                                lhs.val)
                    .val;
    }
    return lhs;
  }

  // Binary Arithmetic upcast operators for +
  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator+(floating_point_wrapper lhs, T rhs) {
    return T(lhs) + rhs;
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator+(T lhs, floating_point_wrapper rhs) {
    return lhs + T(rhs);
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend floating_point_wrapper
  operator-(floating_point_wrapper lhs, floating_point_wrapper rhs) {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      lhs -= rhs;
    } else
#endif
    {
      lhs.val = cast_to_wrapper(cast_from_wrapper<float>(lhs) -
                                    cast_from_wrapper<float>(rhs),
                                lhs.val)
                    .val;
    }
    return lhs;
  }

  // Binary Arithmetic upcast operators for -
  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator-(floating_point_wrapper lhs, T rhs) {
    return T(lhs) - rhs;
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator-(T lhs, floating_point_wrapper rhs) {
    return lhs - T(rhs);
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend floating_point_wrapper
  operator*(floating_point_wrapper lhs, floating_point_wrapper rhs) {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      lhs *= rhs;
    } else
#endif
    {
      lhs.val = cast_to_wrapper(cast_from_wrapper<float>(lhs) *
                                    cast_from_wrapper<float>(rhs),
                                lhs.val)
                    .val;
    }
    return lhs;
  }

  // Binary Arithmetic upcast operators for *
  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator*(floating_point_wrapper lhs, T rhs) {
    return T(lhs) * rhs;
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator*(T lhs, floating_point_wrapper rhs) {
    return lhs * T(rhs);
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend floating_point_wrapper
  operator/(floating_point_wrapper lhs, floating_point_wrapper rhs) {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      lhs /= rhs;
    } else
#endif
    {
      lhs.val = cast_to_wrapper(cast_from_wrapper<float>(lhs) /
                                    cast_from_wrapper<float>(rhs),
                                lhs.val)
                    .val;
    }
    return lhs;
  }

  // Binary Arithmetic upcast operators for /
  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator/(floating_point_wrapper lhs, T rhs) {
    return T(lhs) / rhs;
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_same_v<T, float> || std::is_same_v<T, double>, T>
  operator/(T lhs, floating_point_wrapper rhs) {
    return lhs / T(rhs);
  }

  // Logical operators
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bool operator!() const {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      return static_cast<bool>(!val);
    }
#endif
    return !cast_from_wrapper<float>(*this);
  }

  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bool operator&&(
      floating_point_wrapper rhs) const {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      return static_cast<bool>(val && rhs.val);
    }
#endif
    return cast_from_wrapper<float>(*this) && cast_from_wrapper<float>(rhs);
  }

  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bool operator||(
      floating_point_wrapper rhs) const {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      return static_cast<bool>(val || rhs.val);
    }
#endif
    return cast_from_wrapper<float>(*this) || cast_from_wrapper<float>(rhs);
  }

  // Comparison operators
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bool operator==(
      floating_point_wrapper rhs) const {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      return static_cast<bool>(val == rhs.val);
    }
#endif
    return cast_from_wrapper<float>(*this) == cast_from_wrapper<float>(rhs);
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bool operator!=(
      floating_point_wrapper rhs) const {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      return static_cast<bool>(val != rhs.val);
    }
#endif
    return cast_from_wrapper<float>(*this) != cast_from_wrapper<float>(rhs);
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bool operator<(
      floating_point_wrapper rhs) const {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      return static_cast<bool>(val < rhs.val);
    }
#endif
    return cast_from_wrapper<float>(*this) < cast_from_wrapper<float>(rhs);
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bool operator>(
      floating_point_wrapper rhs) const {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      return static_cast<bool>(val > rhs.val);
    }
#endif
    return cast_from_wrapper<float>(*this) > cast_from_wrapper<float>(rhs);
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bool operator<=(
      floating_point_wrapper rhs) const {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      return static_cast<bool>(val <= rhs.val);
    }
#endif
    return cast_from_wrapper<float>(*this) <= cast_from_wrapper<float>(rhs);
  }

  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bool operator>=(
      floating_point_wrapper rhs) const {
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
    if (!KOKKOS_IMPL_HALF_SOFTWARE_PATH()) {
      return static_cast<bool>(val >= rhs.val);
    }
#endif
    return cast_from_wrapper<float>(*this) >= cast_from_wrapper<float>(rhs);
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_convertible_v<T, float> &&
          (std::is_same_v<T, float> || std::is_same_v<T, double>),
      bool>
  operator<(floating_point_wrapper lhs, T rhs) {
    return static_cast<float>(lhs) < rhs;
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_convertible_v<T, float> &&
          (std::is_same_v<T, float> || std::is_same_v<T, double>),
      bool>
  operator<(T lhs, floating_point_wrapper rhs) {
    return lhs < static_cast<float>(rhs);
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_convertible_v<T, float> &&
          (std::is_same_v<T, float> || std::is_same_v<T, double>),
      bool>
  operator>(floating_point_wrapper lhs, T rhs) {
    return static_cast<float>(lhs) > rhs;
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_convertible_v<T, float> &&
          (std::is_same_v<T, float> || std::is_same_v<T, double>),
      bool>
  operator>(T lhs, floating_point_wrapper rhs) {
    return lhs > static_cast<float>(rhs);
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_convertible_v<T, float> &&
          (std::is_same_v<T, float> || std::is_same_v<T, double>),
      bool>
  operator<=(floating_point_wrapper lhs, T rhs) {
    return static_cast<float>(lhs) <= rhs;
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_convertible_v<T, float> &&
          (std::is_same_v<T, float> || std::is_same_v<T, double>),
      bool>
  operator<=(T lhs, floating_point_wrapper rhs) {
    return lhs <= static_cast<float>(rhs);
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_convertible_v<T, float> &&
          (std::is_same_v<T, float> || std::is_same_v<T, double>),
      bool>
  operator>=(floating_point_wrapper lhs, T rhs) {
    return static_cast<float>(lhs) >= rhs;
  }

  template <class T>
  KOKKOS_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR friend std::enable_if_t<
      std::is_convertible_v<T, float> &&
          (std::is_same_v<T, float> || std::is_same_v<T, double>),
      bool>
  operator>=(T lhs, floating_point_wrapper rhs) {
    return lhs >= static_cast<float>(rhs);
  }

  // Insertion and extraction operators
  // Insertion writes the value as a float: every half and bhalf value is
  // exactly a float, so the stream's flags, precision and locale apply (as for
  // std::float16_t, whose insertion goes through float).
  template <class CharT, class Traits>
  friend std::basic_ostream<CharT, Traits>& operator<<(
      std::basic_ostream<CharT, Traits>& os, const floating_point_wrapper& x) {
    return os << static_cast<float>(x);
  }

  // Extraction as for std::float16_t and std::bfloat16_t (C++23
  // [istream.formatted.arithmetic]/4-6): read a float with the locale's
  // num_get (so the number ends where a float ends, and ')' and ',' stay in
  // the stream), then convert.  A value beyond +-max() sets failbit and stores
  // +-max(); a parse failure sets failbit and stores 0 (num_get's result).
  // The conversion through float can round twice, as the standard accepts
  // (e.g. "1.000488281250001", just above the midpoint between 1 and
  // 1 + 2^-10, reads as the midpoint and rounds to 1).
  template <class CharT, class Traits>
  friend std::basic_istream<CharT, Traits>& operator>>(
      std::basic_istream<CharT, Traits>& is, floating_point_wrapper& x) {
    const typename std::basic_istream<CharT, Traits>::sentry ok(is);
    if (ok) {
      using num_get =
          std::num_get<CharT, std::istreambuf_iterator<CharT, Traits>>;
      std::ios_base::iostate state = std::ios_base::goodbit;
      float f                      = 0.f;
      std::use_facet<num_get>(is.getloc())
          .get(std::istreambuf_iterator<CharT, Traits>(is),
               std::istreambuf_iterator<CharT, Traits>(), is, state, f);
      const floating_point_wrapper max(
          Kokkos::Experimental::finite_max_v<floating_point_wrapper>);
      if (f < -static_cast<float>(max)) {
        state |= std::ios_base::failbit;
        x = -max;
      } else if (static_cast<float>(max) < f) {
        state |= std::ios_base::failbit;
        x = max;
      } else {
        x = f;
      }
      is.setstate(state);
    }
    return is;
  }
};
}  // namespace Impl

// Software conversions for constant evaluation (B4b-CUDA): round to nearest
// even, once, from the source value (converting a double or a 64-bit integer
// through float would round twice); every half and bfloat16 value is exactly a
// float. NaN results are the canonical NaN 0x7fff, as CUDA's conversions
// return. Usable at run time too (the tests compare them with the hardware).
namespace Impl {
KOKKOS_INLINE_FUNCTION constexpr int half_bit_width(std::uint64_t x) {
  int n = 0;
  while (x) {
    x >>= 1;
    ++n;
  }
  return n;
}

// sig * 2^exp (sig > 0) rounded to the format with EB exponent and MB fraction
// bits; the bits without the sign.
template <int EB, int MB>
KOKKOS_INLINE_FUNCTION constexpr std::uint16_t half_round_to_bits(
    std::uint64_t sig, int exp) {
  constexpr int bias = (1 << (EB - 1)) - 1;
  constexpr int emin = 1 - bias;
  const int e        = half_bit_width(sig) - 1 + exp;  // in [2^e, 2^(e+1))
  if (e > bias) return std::uint16_t(((1 << EB) - 1) << MB);  // inf
  const int q     = (e < emin ? emin : e) - MB;  // exponent of the quantum
  const int shift = q - exp;
  std::uint64_t n = 0;
  if (shift <= 0) {
    n = sig << -shift;  // exact
  } else if (shift <= 64) {
    const std::uint64_t hi = shift == 64 ? 0 : sig >> shift;
    const std::uint64_t rem =
        shift == 64 ? sig : sig & ((std::uint64_t(1) << shift) - 1);
    const std::uint64_t half = std::uint64_t(1) << (shift - 1);
    n = hi + ((rem > half || (rem == half && (hi & 1))) ? 1 : 0);
  }
  if (e < emin) return std::uint16_t(n);  // subnormal (or the smallest normal)
  // n in [2^MB, 2^(MB+1)]: a carry moves into the exponent, possibly to inf.
  return std::uint16_t((std::uint64_t(e + bias) << MB) +
                       (n - (std::uint64_t(1) << MB)));
}

template <int EB, int MB>
KOKKOS_INLINE_FUNCTION constexpr std::uint16_t half_double_to_bits(double d) {
  const std::uint64_t x       = __builtin_bit_cast(std::uint64_t, d);
  const std::uint16_t s       = std::uint16_t((x >> 48) & 0x8000);
  const int be                = int((x >> 52) & 0x7ff);
  const std::uint64_t m       = x & ((std::uint64_t(1) << 52) - 1);
  constexpr std::uint16_t inf = std::uint16_t(((1 << EB) - 1) << MB);
  if (be == 0x7ff) return m ? std::uint16_t(0x7fff) : std::uint16_t(s | inf);
  if (be == 0 && m == 0) return s;
  const std::uint64_t sig = be ? (m | (std::uint64_t(1) << 52)) : m;
  const int exp           = (be ? be : 1) - 1075;
  return std::uint16_t(s | half_round_to_bits<EB, MB>(sig, exp));
}

template <int EB, int MB, class I>
KOKKOS_INLINE_FUNCTION constexpr std::uint16_t half_int_to_bits(I v) {
  if (v == 0) return 0;
  bool neg = false;
  if constexpr (std::is_signed_v<I>) neg = v < 0;
  const std::uint64_t a =
      neg ? std::uint64_t(0) - std::uint64_t(v) : std::uint64_t(v);
  return std::uint16_t((neg ? 0x8000 : 0) | half_round_to_bits<EB, MB>(a, 0));
}

template <int EB, int MB>
KOKKOS_INLINE_FUNCTION constexpr float half_bits_to_float(std::uint16_t h) {
  if constexpr (EB == 8) {  // bfloat16: the upper half of a float
    return __builtin_bit_cast(float, std::uint32_t(h) << 16);
  } else {
    const std::uint32_t s = std::uint32_t(h & 0x8000) << 16;
    int be                = (h >> MB) & ((1 << EB) - 1);
    std::uint32_t m       = h & ((1u << MB) - 1);
    constexpr int bias    = (1 << (EB - 1)) - 1;
    if (be == (1 << EB) - 1)
      return __builtin_bit_cast(float, s | 0x7f800000u | (m << (23 - MB)));
    if (be == 0) {
      if (m == 0) return __builtin_bit_cast(float, s);
      be = 1;
      while (!(m & (1u << MB))) {
        m <<= 1;
        --be;
      }
      m &= (1u << MB) - 1;
    }
    return __builtin_bit_cast(
        float, s | (std::uint32_t(be - bias + 127) << 23) | (m << (23 - MB)));
  }
}

// The value v as the half type H (half_t or bhalf_t).
template <class H, class T>
KOKKOS_INLINE_FUNCTION constexpr H half_constexpr_from(T v) {
  constexpr bool bf16 = is_bfloat16<H>::value;
  constexpr int EB    = bf16 ? 8 : 5;
  constexpr int MB    = bf16 ? 7 : 10;
  std::uint16_t bits  = 0;
  if constexpr (std::is_same_v<T, bool>) {
    bits = half_double_to_bits<EB, MB>(v ? 1.0 : 0.0);
  } else if constexpr (std::is_floating_point_v<T>) {
    bits = half_double_to_bits<EB, MB>(static_cast<double>(v));
  } else {
    bits = half_int_to_bits<EB, MB>(v);
  }
  return H(__builtin_bit_cast(typename H::impl_type, bits));
}

// The half value h as T: exact for float and double; toward zero for the
// integers, as CUDA's _rz conversions; false only for +-0.
template <class T, class H>
KOKKOS_INLINE_FUNCTION constexpr T half_constexpr_to(H h) {
  constexpr bool bf16 = is_bfloat16<H>::value;
  const float f       = half_bits_to_float < bf16 ? 8 : 5,
              bf16 ? 7
                         : 10 > (__builtin_bit_cast(std::uint16_t,
                                              typename H::impl_type(h)));
  if constexpr (std::is_same_v<T, bool>) {
    return f != 0.0f;
  } else {
    return static_cast<T>(f);
  }
}
}  // namespace Impl

// Declare wrapper overloads now that floating_point_wrapper is declared
template <class T>
static KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    Kokkos::Experimental::half_t
    cast_to_wrapper(T x, const Kokkos::Impl::half_impl_t::type&) {
  return Kokkos::Experimental::cast_to_half(x);
}

#if !KOKKOS_BHALF_T_IS_FLOAT
template <class T>
static KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR
    Kokkos::Experimental::bhalf_t
    cast_to_wrapper(T x, const Kokkos::Impl::bhalf_impl_t::type&) {
  return Kokkos::Experimental::cast_to_bhalf(x);
}
#endif

template <class T>
static KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR T
cast_from_wrapper(const Kokkos::Experimental::half_t& x) {
  return Kokkos::Experimental::cast_from_half<T>(x);
}

#if !KOKKOS_BHALF_T_IS_FLOAT
template <class T>
static KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR T
cast_from_wrapper(const Kokkos::Experimental::bhalf_t& x) {
  return Kokkos::Experimental::cast_from_bhalf<T>(x);
}
#endif

}  // namespace Experimental
}  // namespace Kokkos

#endif

// If none of the above actually did anything and defined a half precision type
// define a fallback implementation here using float
#if KOKKOS_HALF_T_IS_FLOAT
namespace Kokkos {
namespace Impl {
struct half_impl_t {
  using type = float;
};
}  // namespace Impl
namespace Experimental {

using half_t = Kokkos::Impl::half_impl_t::type;

// cast_to_half
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(float val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(bool val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(double val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(short val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(unsigned short val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(int val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(unsigned int val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(long val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(unsigned long val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(long long val) {
  return half_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR half_t
cast_to_half(unsigned long long val) {
  return half_t(val);
}

// cast_from_half
// Using an explicit list here too, since the other ones are explicit and for
// example don't include char
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<
    std::is_same_v<T, float> || std::is_same_v<T, bool> ||
        std::is_same_v<T, double> || std::is_same_v<T, short> ||
        std::is_same_v<T, unsigned short> || std::is_same_v<T, int> ||
        std::is_same_v<T, unsigned int> || std::is_same_v<T, long> ||
        std::is_same_v<T, unsigned long> || std::is_same_v<T, long long> ||
        std::is_same_v<T, unsigned long long>,
    T>
cast_from_half(half_t val) {
  return T(val);
}

}  // namespace Experimental
}  // namespace Kokkos
#endif

#if KOKKOS_BHALF_T_IS_FLOAT
namespace Kokkos {
namespace Impl {
struct bhalf_impl_t {
  using type = float;
};
}  // namespace Impl

namespace Experimental {

using bhalf_t = Kokkos::Impl::bhalf_impl_t::type;

// cast_to_bhalf
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(float val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(bool val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(double val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(short val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(unsigned short val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(int val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(unsigned int val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(long val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(unsigned long val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(long long val) {
  return bhalf_t(val);
}
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(unsigned long long val) {
  return bhalf_t(val);
}

// cast_from_bhalf
template <class T>
KOKKOS_INLINE_FUNCTION KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<
    std::is_same_v<T, float> || std::is_same_v<T, bool> ||
        std::is_same_v<T, double> || std::is_same_v<T, short> ||
        std::is_same_v<T, unsigned short> || std::is_same_v<T, int> ||
        std::is_same_v<T, unsigned int> || std::is_same_v<T, long> ||
        std::is_same_v<T, unsigned long> || std::is_same_v<T, long long> ||
        std::is_same_v<T, unsigned long long>,
    T>
cast_from_bhalf(bhalf_t val) {
  return T(val);
}
}  // namespace Experimental
}  // namespace Kokkos
#endif

#endif  // KOKKOS_HALF_FLOATING_POINT_WRAPPER_HPP_
