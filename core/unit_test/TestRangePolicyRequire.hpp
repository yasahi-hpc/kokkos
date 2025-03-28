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

#include <cstdio>

#include <Kokkos_Core.hpp>

// This file is largely duplicating TestRange.hpp but it applies
// Kokkos::Experimental require at every place where a parallel
// operation is executed.

namespace Test {

namespace {

template <class ExecSpace, class ScheduleType, class Property>
struct TestRangeRequire {
  using value_type = int;  ///< alias required for the parallel_reduce

  using view_type = Kokkos::View<int *, ExecSpace>;

  view_type m_flags;

  struct VerifyInitTag {};
  struct ResetTag {};
  struct VerifyResetTag {};
  struct OffsetTag {};
  struct VerifyOffsetTag {};

  int N;
  static const int offset = 13;
  TestRangeRequire(const size_t N_)
      : m_flags(Kokkos::view_alloc(Kokkos::WithoutInitializing, "flags"), N_),
        N(N_) {}

  void test_for() {
    typename view_type::HostMirror host_flags =
        Kokkos::create_mirror_view(m_flags);

    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecSpace, ScheduleType>(0, N), Property()),
        *this);

    {
      using ThisType = TestRangeRequire<ExecSpace, ScheduleType, Property>;
      std::string label("parallel_for");
      Kokkos::Impl::ParallelConstructName<ThisType, void> pcn(label);
      ASSERT_EQ(pcn.get(), label);
      std::string empty_label("");
      Kokkos::Impl::ParallelConstructName<ThisType, void> empty_pcn(
          empty_label);
      ASSERT_EQ(empty_pcn.get(), typeid(ThisType).name());
    }

    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecSpace, ScheduleType, VerifyInitTag>(0, N),
            Property()),
        *this);

    {
      using ThisType = TestRangeRequire<ExecSpace, ScheduleType, Property>;
      std::string label("parallel_for");
      Kokkos::Impl::ParallelConstructName<ThisType, VerifyInitTag> pcn(label);
      ASSERT_EQ(pcn.get(), label);
      std::string empty_label("");
      Kokkos::Impl::ParallelConstructName<ThisType, VerifyInitTag> empty_pcn(
          empty_label);
      ASSERT_EQ(empty_pcn.get(), std::string(typeid(ThisType).name()) + "/" +
                                     typeid(VerifyInitTag).name());
    }

    Kokkos::deep_copy(host_flags, m_flags);

    int error_count = 0;
    for (int i = 0; i < N; ++i) {
      if (int(i) != host_flags(i)) ++error_count;
    }
    ASSERT_EQ(error_count, int(0));

    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecSpace, ScheduleType, ResetTag>(0, N),
            Property()),
        *this);
    Kokkos::parallel_for(
        std::string("TestKernelFor"),
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecSpace, ScheduleType, VerifyResetTag>(0, N),
            Property()),
        *this);

    Kokkos::deep_copy(host_flags, m_flags);

    error_count = 0;
    for (int i = 0; i < N; ++i) {
      if (int(2 * i) != host_flags(i)) ++error_count;
    }
    ASSERT_EQ(error_count, int(0));

    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecSpace, ScheduleType, OffsetTag>(offset,
                                                                    N + offset),
            Property()),
        *this);
    Kokkos::parallel_for(
        std::string("TestKernelFor"),
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecSpace, ScheduleType,
                                Kokkos::IndexType<unsigned int>,
                                VerifyOffsetTag>(0, N),
            Property()),
        *this);

    Kokkos::deep_copy(host_flags, m_flags);

    error_count = 0;
    for (int i = 0; i < N; ++i) {
      if (i + offset != host_flags(i)) ++error_count;
    }
    ASSERT_EQ(error_count, int(0));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const { m_flags(i) = i; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const VerifyInitTag &, const int i) const {
    if (i != m_flags(i)) {
      Kokkos::printf("TestRangeRequire::test_for error at %d != %d\n", i,
                     m_flags(i));
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ResetTag &, const int i) const {
    m_flags(i) = 2 * m_flags(i);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const VerifyResetTag &, const int i) const {
    if (2 * i != m_flags(i)) {
      Kokkos::printf("TestRangeRequire::test_for error at %d != %d\n", i,
                     m_flags(i));
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const OffsetTag &, const int i) const {
    m_flags(i - offset) = i;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const VerifyOffsetTag &, const int i) const {
    if (i + offset != m_flags(i)) {
      Kokkos::printf("TestRangeRequire::test_for error at %d != %d\n",
                     i + offset, m_flags(i));
    }
  }

  //----------------------------------------

  void test_reduce() {
    int total = 0;

    Kokkos::parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecSpace, ScheduleType>(0, N), Property()),
        *this);

    Kokkos::parallel_reduce(
        "TestKernelReduce",
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecSpace, ScheduleType>(0, N), Property()),
        *this, total);
    // sum( 0 .. N-1 )
    ASSERT_EQ(size_t((N - 1) * (N) / 2), size_t(total));

    Kokkos::parallel_reduce(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<ExecSpace, ScheduleType, OffsetTag>(offset,
                                                                    N + offset),
            Property()),
        *this, total);
    // sum( 1 .. N )
    ASSERT_EQ(size_t((N) * (N + 1) / 2), size_t(total));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &update) const {
    update += m_flags(i);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const OffsetTag &, const int i, value_type &update) const {
    update += 1 + m_flags(i - offset);
  }

  //----------------------------------------

  void test_dynamic_policy() {
    auto const N_no_implicit_capture = N;
    using policy_t =
        Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic> >;
    int const concurrency = ExecSpace().concurrency();

    {
      Kokkos::View<size_t *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic> >
          count("Count", concurrency);
      Kokkos::View<int *, ExecSpace> a("A", N);

      Kokkos::parallel_for(
          policy_t(0, N), KOKKOS_LAMBDA(const int &i) {
            for (int k = 0; k < (i < N_no_implicit_capture / 2 ? 1 : 10000);
                 k++) {
              a(i)++;
            }
            count(ExecSpace::impl_hardware_thread_id())++;
          });

      int error = 0;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<ExecSpace>(0, N),
          KOKKOS_LAMBDA(const int &i, int &lsum) {
            lsum += (a(i) != (i < N_no_implicit_capture / 2 ? 1 : 10000));
          },
          error);
      ASSERT_EQ(error, 0);

      if ((concurrency > 1) && (N > 4 * concurrency)) {
        size_t min = N;
        size_t max = 0;
        for (int t = 0; t < concurrency; t++) {
          if (count(t) < min) min = count(t);
          if (count(t) > max) max = count(t);
        }
        ASSERT_LT(min, max);

        // if ( concurrency > 2 ) {
        //  ASSERT_LT( 2 * min, max );
        //}
      }
    }

    {
      Kokkos::View<size_t *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic> >
          count("Count", concurrency);
      Kokkos::View<int *, ExecSpace> a("A", N);

      int sum = 0;
      Kokkos::parallel_reduce(
          policy_t(0, N),
          KOKKOS_LAMBDA(const int &i, int &lsum) {
            for (int k = 0; k < (i < N_no_implicit_capture / 2 ? 1 : 10000);
                 k++) {
              a(i)++;
            }
            count(ExecSpace::impl_hardware_thread_id())++;
            lsum++;
          },
          sum);
      ASSERT_EQ(sum, N);

      int error = 0;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<ExecSpace>(0, N),
          KOKKOS_LAMBDA(const int &i, int &lsum) {
            lsum += (a(i) != (i < N_no_implicit_capture / 2 ? 1 : 10000));
          },
          error);
      ASSERT_EQ(error, 0);

      if ((concurrency > 1) && (N > 4 * concurrency)) {
        size_t min = N;
        size_t max = 0;
        for (int t = 0; t < concurrency; t++) {
          if (count(t) < min) min = count(t);
          if (count(t) > max) max = count(t);
        }
        ASSERT_LT(min, max);

        // if ( concurrency > 2 ) {
        //  ASSERT_LT( 2 * min, max );
        //}
      }
    }
  }
};

}  // namespace

TEST(TEST_CATEGORY, range_for_require) {
  using Property = Kokkos::Experimental::WorkItemProperty::HintLightWeight_t;
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static>, Property>
        f(0);
    f.test_for();
  }
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                     Property>
        f(0);
    f.test_for();
  }

  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static>, Property>
        f(2);
    f.test_for();
  }
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                     Property>
        f(3);
    f.test_for();
  }

  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static>, Property>
        f(1000);
    f.test_for();
  }
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                     Property>
        f(1001);
    f.test_for();
  }
}

TEST(TEST_CATEGORY, range_reduce_require) {
  using Property = Kokkos::Experimental::WorkItemProperty::HintLightWeight_t;
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static>, Property>
        f(0);
    f.test_reduce();
  }
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                     Property>
        f(0);
    f.test_reduce();
  }

  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static>, Property>
        f(2);
    f.test_reduce();
  }
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                     Property>
        f(3);
    f.test_reduce();
  }

  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static>, Property>
        f(1000);
    f.test_reduce();
  }
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                     Property>
        f(1001);
    f.test_reduce();
  }
}

#ifndef KOKKOS_ENABLE_OPENMPTARGET
TEST(TEST_CATEGORY, range_dynamic_policy_require) {
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && \
    !defined(KOKKOS_ENABLE_SYCL)
  using Property = Kokkos::Experimental::WorkItemProperty::HintLightWeight_t;
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                     Property>
        f(0);
    f.test_dynamic_policy();
  }
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                     Property>
        f(3);
    f.test_dynamic_policy();
  }
  {
    TestRangeRequire<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                     Property>
        f(1001);
    f.test_dynamic_policy();
  }
#endif
}
#endif

}  // namespace Test
