#!/bin/bash -e
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=0:20:00"
#PJM -s
#PJM -g jh220036a
#PJM --mpi proc=1

# CUDA check of the double/integer rounding fixes (sycl-tests.md, Open, 0.):
# - the wrapper's double compound assignments now end through cast_to_half /
#   cast_to_bhalf (__double2half / __double2bfloat16);
# - half_constexpr_path_matches_runtime also checks integers just above a tie
#   (against __ll2bfloat16_rn / __int2bfloat16_rn);
# - the pre-Ampere bf16 branch uses the correctly rounded intrinsics: v_sm70
#   (built with -arch=sm_70, so the A100 JITs its compute_70 PTX and runs the
#   pre-Ampere code path).
# The objects are compiled on the login node; link here (libcuda.so).

. /etc/profile.d/modules.sh
module purge
module load gcc/12.2.0 cuda/12.6 cmake/3.24.0

cd /work/04/jh220036a/i18048/tests/kokkos/build_gpu
for t in Default Serial_SmokeTest Cuda_SmokeTest; do
  make -j8 -f core/unit_test/CMakeFiles/Kokkos_CoreUnitTest_$t.dir/build.make \
       core/unit_test/Kokkos_CoreUnitTest_$t
done
cd core/unit_test
echo "===== Default (half)"
./Kokkos_CoreUnitTest_Default --gtest_filter='*half*' || echo "EXIT Default $?"
for n in Serial Cuda; do
  echo "===== $n complex"
  ./Kokkos_CoreUnitTest_${n}_SmokeTest --gtest_filter='*complex*:*atomic*' || echo "EXIT $n $?"
done
echo "===== pre-Ampere bf16 branch (sm_70 PTX on the A100)"
# (on /work: the compute nodes cannot see /home)
/work/04/jh220036a/i18048/tmp/claude_builds/complex_a6/v_sm70 || echo "EXIT v_sm70 $?"
