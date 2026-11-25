// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SYCL_HALF_HPP_
#define KOKKOS_SYCL_HALF_HPP_

#ifdef KOKKOS_IMPL_SYCL_HALF_TYPE_DEFINED

#include <SYCL/Kokkos_SYCL_Half_Impl_Type.hpp>
#include <impl/Kokkos_Half_FloatingPointWrapper.hpp>

namespace Kokkos::Experimental {

/************************** half conversions **********************************/
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(half_t val) {
  return val;
}

// On the host, round with Kokkos's software conversion: DPC++'s host
// sycl::half(float) rounds wrongly below 2^-13 (see
// Kokkos_Half_FloatingPointWrapper.hpp).
// sycl::half and bfloat16 construct only from float, so a double (and, for
// bfloat16, an integer beyond 2^24) would round twice, through float: e.g.
// 2^-25 + 2^-77 gives 0, not the smallest subnormal 2^-24 (Intel GPU Max,
// Aurora job 9127596). Those conversions round once, in software, on the
// device too. For half_t the integers are exact in float up to 2^24, beyond
// half's range.
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(float val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<half_t>(val);
  return half_t::impl_type(val);
}
// false and true convert to 0 and 1 (exact), as for any floating-point type.
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(bool val) {
  return cast_to_half(static_cast<float>(val));
}
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(double val) {
  return Impl::half_constexpr_from<half_t>(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(short val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<half_t>(val);
  return half_t::impl_type(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(unsigned short val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<half_t>(val);
  return half_t::impl_type(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(int val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<half_t>(val);
  return half_t::impl_type(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(unsigned int val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<half_t>(val);
  return half_t::impl_type(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(long long val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<half_t>(val);
  return half_t::impl_type(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(unsigned long long val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<half_t>(val);
  return half_t::impl_type(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(long val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<half_t>(val);
  return half_t::impl_type(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR half_t cast_to_half(unsigned long val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<half_t>(val);
  return half_t::impl_type(val);
}

template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, float>, T>
cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}
// The boolean conversion: false for +-0, true otherwise (also for NaN).
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, bool>, T>
cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<float>(half_t::impl_type(val)) != 0.0f;
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, double>, T>
cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, short>, T>
cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned short>, T>
    cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, int>, T>
cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, unsigned int>, T>
cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, long long>, T>
cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned long long>, T>
    cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, long>, T>
cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, unsigned long>, T>
cast_from_half(half_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(half_t::impl_type(val));
}

}  // namespace Kokkos::Experimental
#endif  // KOKKOS_IMPL_SYCL_HALF_TYPE_DEFINED

#ifdef KOKKOS_IMPL_SYCL_BHALF_TYPE_DEFINED

namespace Kokkos::Experimental {

/************************** bhalf conversions *********************************/
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(bhalf_t val) {
  return val;
}

inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(float val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<bhalf_t>(val);
  return bhalf_t::impl_type(val);
}
// false and true convert to 0 and 1 (exact), as for any floating-point type.
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(bool val) {
  return cast_to_bhalf(static_cast<float>(val));
}
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(double val) {
  return Impl::half_constexpr_from<bhalf_t>(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(short val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<bhalf_t>(val);
  return bhalf_t::impl_type(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(unsigned short val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH())
    return Impl::half_constexpr_from<bhalf_t>(val);
  return bhalf_t::impl_type(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(int val) {
  return Impl::half_constexpr_from<bhalf_t>(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(unsigned int val) {
  return Impl::half_constexpr_from<bhalf_t>(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(long long val) {
  return Impl::half_constexpr_from<bhalf_t>(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t
cast_to_bhalf(unsigned long long val) {
  return Impl::half_constexpr_from<bhalf_t>(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(long val) {
  return Impl::half_constexpr_from<bhalf_t>(val);
}
inline KOKKOS_IMPL_HALF_CONSTEXPR bhalf_t cast_to_bhalf(unsigned long val) {
  return Impl::half_constexpr_from<bhalf_t>(val);
}

template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, float>, T>
cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
// The boolean conversion: false for +-0, true otherwise (also for NaN).
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, bool>, T>
cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<float>(bhalf_t::impl_type(val)) != 0.0f;
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, double>, T>
cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, short>, T>
cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned short>, T>
    cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, int>, T>
cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, unsigned int>, T>
cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, long long>, T>
cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR
    std::enable_if_t<std::is_same_v<T, unsigned long long>, T>
    cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, long>, T>
cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
template <class T>
KOKKOS_IMPL_HALF_CONSTEXPR std::enable_if_t<std::is_same_v<T, unsigned long>, T>
cast_from_bhalf(bhalf_t val) {
  if (KOKKOS_IMPL_HALF_SOFTWARE_PATH()) return Impl::half_constexpr_to<T>(val);
  return static_cast<T>(bhalf_t::impl_type(val));
}
}  // namespace Kokkos::Experimental

#endif  // KOKKOS_IMPL_SYCL_BHALF_TYPE_DEFINED

#endif
