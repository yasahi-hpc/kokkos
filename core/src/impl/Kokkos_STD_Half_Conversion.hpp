// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_STD_HALF_HPP_
#define KOKKOS_STD_HALF_HPP_

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT) || \
    defined(KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT)
#include <impl/Kokkos_STD_Half_Impl_Type.hpp>
#include <impl/Kokkos_Half_FloatingPointWrapper.hpp>
#endif

#ifdef KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT
namespace Kokkos {
namespace Experimental {
// cast_to_half
inline half_t cast_to_half(float val) { return half_t::impl_type(val); }
inline half_t cast_to_half(bool val) { return half_t::impl_type(val); }
inline half_t cast_to_half(double val) { return half_t::impl_type(val); }
inline half_t cast_to_half(short val) { return half_t::impl_type(val); }
inline half_t cast_to_half(unsigned short val) {
  return half_t::impl_type(val);
}
inline half_t cast_to_half(int val) { return half_t::impl_type(val); }
inline half_t cast_to_half(unsigned int val) { return half_t::impl_type(val); }
inline half_t cast_to_half(long val) { return half_t::impl_type(val); }
inline half_t cast_to_half(unsigned long val) { return half_t::impl_type(val); }
inline half_t cast_to_half(long long val) { return half_t::impl_type(val); }
inline half_t cast_to_half(unsigned long long val) {
  return half_t::impl_type(val);
}
// Declared in Kokkos_Half_FloatingPointWrapper.hpp and defined by every other
// backend; reached by floating_point_wrapper::operator=(T) for T convertible to
// half_t, e.g. a numeric-trait constant: h = Kokkos::infinity<half_t>::value.
inline half_t cast_to_half(half_t val) { return val; }

// cast_from_half
template <typename T>
inline std::enable_if_t<std::is_same_v<T, float>, T> cast_from_half(
    half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, bool>, T> cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, double>, T> cast_from_half(
    half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, short>, T> cast_from_half(
    half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, unsigned short>, T> cast_from_half(
    half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, int>, T> cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, unsigned int>, T> cast_from_half(
    half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, long>, T> cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, unsigned long>, T> cast_from_half(
    half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, long long>, T> cast_from_half(
    half_t val) {
  return static_cast<half_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, unsigned long long>, T>
cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}
}  // namespace Experimental
}  // namespace Kokkos
#endif

#ifdef KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT
namespace Kokkos {
namespace Experimental {
// The short and unsigned short overloads convert through float, which holds
// every value of both types and of std::bfloat16_t exactly, so the result is
// that of the direct conversion. Converting directly makes GCC >= 13 warn
// wherever the x86 BF16 intrinsics are visible (e.g. -march=sapphirerapids):
// "'__bfloat16' is redefined from typedef 'short' to real '__bf16'".

// cast_to_bhalf
inline bhalf_t cast_to_bhalf(float val) { return bhalf_t::impl_type(val); }
inline bhalf_t cast_to_bhalf(bool val) { return bhalf_t::impl_type(val); }
inline bhalf_t cast_to_bhalf(double val) { return bhalf_t::impl_type(val); }
inline bhalf_t cast_to_bhalf(short val) {
  return bhalf_t::impl_type(static_cast<float>(val));
}
inline bhalf_t cast_to_bhalf(unsigned short val) {
  return bhalf_t::impl_type(static_cast<float>(val));
}
inline bhalf_t cast_to_bhalf(int val) { return bhalf_t::impl_type(val); }
inline bhalf_t cast_to_bhalf(unsigned int val) {
  return bhalf_t::impl_type(val);
}
inline bhalf_t cast_to_bhalf(long val) { return bhalf_t::impl_type(val); }
inline bhalf_t cast_to_bhalf(unsigned long val) {
  return bhalf_t::impl_type(val);
}
inline bhalf_t cast_to_bhalf(long long val) { return bhalf_t::impl_type(val); }
inline bhalf_t cast_to_bhalf(unsigned long long val) {
  return bhalf_t::impl_type(val);
}
// See cast_to_half(half_t).
inline bhalf_t cast_to_bhalf(bhalf_t val) { return val; }

// cast_from_bhalf
template <typename T>
inline std::enable_if_t<std::is_same_v<T, float>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<bhalf_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, bool>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<bhalf_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, double>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<bhalf_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, short>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<T>(
      static_cast<float>(static_cast<bhalf_t::impl_type>(val)));
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, unsigned short>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<T>(
      static_cast<float>(static_cast<bhalf_t::impl_type>(val)));
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, int>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<bhalf_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, unsigned int>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<bhalf_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, long>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<bhalf_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, unsigned long>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<bhalf_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, long long>, T> cast_from_bhalf(
    bhalf_t val) {
  return static_cast<bhalf_t::impl_type>(val);
}
template <typename T>
inline std::enable_if_t<std::is_same_v<T, unsigned long long>, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<bhalf_t::impl_type>(val);
}
}  // namespace Experimental
}  // namespace Kokkos
#endif

#endif
