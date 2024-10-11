//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include "Kokkos_Core.hpp"
#include "Kokkos_HostSpace_deepcopy.hpp"

namespace Kokkos {

namespace Impl {

void hostspace_fence(const DefaultHostExecutionSpace& exec) {
  exec.fence("HostSpace fence");
}

void hostspace_parallel_deepcopy(void* dst, const void* src, ptrdiff_t n) {
  Kokkos::DefaultHostExecutionSpace exec;
  hostspace_parallel_deepcopy_async(exec, dst, src, n);
}

// DeepCopy called with an execution space that can't access HostSpace
void hostspace_parallel_deepcopy_async(void* dst, const void* src,
                                       ptrdiff_t n) {
  Kokkos::DefaultHostExecutionSpace exec;
  hostspace_parallel_deepcopy_async(exec, dst, src, n);
  exec.fence(
      "Kokkos::Impl::hostspace_parallel_deepcopy_async: fence after copy");
}

template <typename ExecutionSpace>
void hostspace_parallel_deepcopy_async(const ExecutionSpace& exec, void* dst,
                                       const void* src, ptrdiff_t n) {
  using policy_t = Kokkos::RangePolicy<ExecutionSpace>;

  // If the asynchronous HPX backend is enabled, do *not* copy anything
  // synchronously. The deep copy must be correctly sequenced with respect to
  // other kernels submitted to the same instance, so we only use the fallback
  // parallel_for version in this case.
#if !(defined(KOKKOS_ENABLE_HPX) && \
      defined(KOKKOS_ENABLE_IMPL_HPX_ASYNC_DISPATCH))
  constexpr int host_deep_copy_serial_limit = 10 * 8192;
  if ((n < host_deep_copy_serial_limit) || (exec.concurrency() == 1)) {
    if (0 < n) std::memcpy(dst, src, n);
    return;
  }

  // Both src and dst are aligned the same way with respect to 8 byte words
  if (reinterpret_cast<ptrdiff_t>(src) % 8 ==
      reinterpret_cast<ptrdiff_t>(dst) % 8) {
    char* dst_c       = reinterpret_cast<char*>(dst);
    const char* src_c = reinterpret_cast<const char*>(src);
    char* dst_end     = reinterpret_cast<char*>(dst) + n;
    int count         = 0;
    hostspace_bytes_deepcopy_head(exec, dst_c, src_c, 8, count);
    hostspace_deepcopy_bulk<ExecutionSpace, double>(exec, dst_c, src_c, n,
                                                    count);
    hostspace_bytes_deepcopy_tail(exec, dst_c, src_c, dst_end, n, 8, count);
    return;
  }

  // Both src and dst are aligned the same way with respect to 4 byte words
  if (reinterpret_cast<ptrdiff_t>(src) % 4 ==
      reinterpret_cast<ptrdiff_t>(dst) % 4) {
    char* dst_c       = reinterpret_cast<char*>(dst);
    const char* src_c = reinterpret_cast<const char*>(src);
    char* dst_end     = reinterpret_cast<char*>(dst) + n;
    int count         = 0;
    hostspace_bytes_deepcopy_head(exec, dst_c, src_c, 4, count);
    hostspace_deepcopy_bulk<ExecutionSpace, int32_t>(exec, dst_c, src_c, n,
                                                     count);
    hostspace_bytes_deepcopy_tail(exec, dst_c, src_c, dst_end, n, 4, count);
    return;
  }
#endif

  // Src and dst are not aligned the same way, we can only to byte wise copy.
  {
    char* dst_p       = reinterpret_cast<char*>(dst);
    const char* src_p = reinterpret_cast<const char*>(src);
    Kokkos::parallel_for("Kokkos::Impl::host_space_deepcopy_char",
                         policy_t(exec, 0, n),
                         [=](const ptrdiff_t i) { dst_p[i] = src_p[i]; });
  }
}

template <typename ExecutionSpace>
void hostspace_bytes_deepcopy_head(const ExecutionSpace& exec, char* dst_c,
                                   const char* src_c, ptrdiff_t byte,
                                   int& count) {
#if (defined(KOKKOS_ENABLE_HPX) || defined(KOKKOS_ENABLE_THREADS))
  if constexpr (!std::is_same_v<ExecutionSpace,
                                Kokkos::DefaultHostExecutionSpace>) {
    auto* internal_instance = exec.impl_instance();
    std::lock_guard<std::mutex> lock_instance(
        internal_instance->m_instance_mutex);
  }
#else
  auto* internal_instance = exec.impl_internal_space_instance();
  std::lock_guard<std::mutex> lock_instance(
      internal_instance->m_instance_mutex);
#endif
  count = 0;
  // get initial bytes copied
  while (reinterpret_cast<ptrdiff_t>(dst_c) % byte != 0) {
    *dst_c = *src_c;
    dst_c++;
    src_c++;
    count++;
  }
}

template <typename ExecutionSpace, typename DataType>
void hostspace_deepcopy_bulk(const ExecutionSpace& exec, char* dst_c,
                             const char* src_c, ptrdiff_t n, int count) {
  // copy the bulk of the data
  DataType* dst_p       = reinterpret_cast<DataType*>(dst_c);
  const DataType* src_p = reinterpret_cast<const DataType*>(src_c);

  std::string name = "Kokkos::Impl::host_space_deepcopy_";
  name += std::is_same_v<DataType, double> ? "double" : "int";

  using policy_t = Kokkos::RangePolicy<ExecutionSpace>;
  Kokkos::parallel_for(name, policy_t(exec, 0, (n - count) / sizeof(DataType)),
                       [=](const ptrdiff_t i) { dst_p[i] = src_p[i]; });
}

template <typename ExecutionSpace>
void hostspace_bytes_deepcopy_tail(const ExecutionSpace& exec, char* dst_c,
                                   const char* src_c, const char* dst_end,
                                   ptrdiff_t n, ptrdiff_t byte, int count) {
#if (defined(KOKKOS_ENABLE_HPX) || defined(KOKKOS_ENABLE_THREADS))
  if constexpr (!std::is_same_v<ExecutionSpace,
                                Kokkos::DefaultHostExecutionSpace>) {
    auto* internal_instance = exec.impl_instance();
    std::lock_guard<std::mutex> lock_instance(
        internal_instance->m_instance_mutex);
  }
#else
  auto* internal_instance = exec.impl_internal_space_instance();
  std::lock_guard<std::mutex> lock_instance(
      internal_instance->m_instance_mutex);
#endif
  // get final data copied
  dst_c += ((n - count) / byte) * byte;
  src_c += ((n - count) / byte) * byte;
  while (dst_c != dst_end) {
    *dst_c = *src_c;
    dst_c++;
    src_c++;
  }
}

// Explicit instantiation
template void hostspace_parallel_deepcopy_async<DefaultHostExecutionSpace>(
    const DefaultHostExecutionSpace&, void*, const void*, ptrdiff_t);

#if defined(KOKKOS_ENABLE_SERIAL) &&                                    \
    (defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS) || \
     defined(KOKKOS_ENABLE_HPX))
// Instantiate only if both the Serial backend and some other host parallel
// backend are enabled
template void hostspace_parallel_deepcopy_async<Kokkos::Serial>(
    const Kokkos::Serial&, void*, const void*, ptrdiff_t);
#endif
}  // namespace Impl

}  // namespace Kokkos
