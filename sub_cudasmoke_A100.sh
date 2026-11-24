#!/bin/bash -e
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=0:20:00"
#PJM -s
#PJM -g jh220036a
#PJM --mpi proc=1

# Kokkos_CoreUnitTest_Cuda_SmokeTest without cuda_graph.sized_functor_launch,
# which segfaults in build_gpu since before the branch (ctest, 2026-10-22, 10-26,
# 10-27) and hides the tests after it. The executable is linked by
# sub_full_A100.sh.
set +e
. /etc/profile.d/modules.sh
module purge
module load gcc/12.2.0 cuda/12.6 cmake/3.24.0
cd "$(pwd -P)/build_gpu/core/unit_test" || exit 2
./Kokkos_CoreUnitTest_Cuda_SmokeTest --gtest_filter='-*sized_functor_launch*'
echo "EXIT $?"
