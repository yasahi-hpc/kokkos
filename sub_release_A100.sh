#!/bin/bash -e
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=0:30:00"
#PJM -s
#PJM -g jh220036a
#PJM --mpi proc=1

# The complex, half and numeric-trait tests in the optimized CUDA build
# build_gpu_release (Release, -O3, no Kokkos_ENABLE_DEBUG; cuda-tests.md, open
# task 1). The objects are compiled on the login node first, from the /work path:
#   cd /work/04/jh220036a/i18048/tests/kokkos/build_gpu_release
#   cmake --build . -j 8 --target <the targets below> -- -k
# This job links and runs. Exit status 0 if every run passed.
#
# Submit from the Kokkos checkout:   pjsub sub_release_A100.sh

set +e
. /etc/profile.d/modules.sh
module purge
module load gcc/12.2.0 cuda/12.6 cmake/3.24.0

ROOT=$(pwd -P)
B=$ROOT/build_gpu_release
echo "commit: $(git -C "$ROOT" rev-parse --short HEAD) on $(git -C "$ROOT" branch --show-current)"
echo "build:  $B ($(grep -m1 '^CMAKE_BUILD_TYPE' "$B/CMakeCache.txt"))"
cd "$B" || exit 2

status=0
run() {  # run <test executable suffix> <filter>
  local target=Kokkos_CoreUnitTest_$1
  make -j8 -f core/unit_test/CMakeFiles/$target.dir/build.make core/unit_test/$target \
    > link_$1.log 2>&1 || { echo "===== $target: FAILED to link"; tail -5 link_$1.log; status=1; return; }
  echo "===== $target $2"
  (cd core/unit_test && ./$target --gtest_filter="$2") > run_$1.log 2>&1
  local rc=$?
  grep -E '^\[ *(FAILED|PASSED|SKIPPED) *\]|Failure|Expected|Which is|actual' run_$1.log | sort -u | head -30
  [ $rc = 0 ] || { echo "EXIT $target $rc"; status=1; }
}
run Serial_SmokeTest '*complex*:*atomic*'
run Cuda_SmokeTest '*complex*:*atomic*'
run Serial1 '*numeric_traits*'
run Cuda1 '*numeric_traits*'
run Default '*half*'
echo "===== status $status"
exit $status
