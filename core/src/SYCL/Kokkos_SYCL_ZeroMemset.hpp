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

#ifndef KOKKOS_SYCL_ZEROMEMSET_HPP
#define KOKKOS_SYCL_ZEROMEMSET_HPP

#include <impl/Kokkos_ZeroMemset_fwd.hpp>
#include <SYCL/Kokkos_SYCL.hpp>

namespace Kokkos {
namespace Impl {

template <class T, class... P>
struct ZeroMemset<Kokkos::SYCL, View<T, P...>> {
  ZeroMemset(const Kokkos::SYCL& exec_space, const View<T, P...>& dst) {
    auto event = exec_space.impl_internal_space_instance()->m_queue->memset(
        dst.data(), 0, dst.size() * sizeof(typename View<T, P...>::value_type));
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
    exec_space.impl_internal_space_instance()
        ->m_queue->ext_oneapi_submit_barrier(std::vector<sycl::event>{event});
#endif
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // !defined(KOKKOS_SYCL_ZEROMEMSET_HPP)
