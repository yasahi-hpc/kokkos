#!/bin/bash -e
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=2:00:00"
#PJM -s
#PJM -g jh220036a
#PJM --mpi proc=1

# The complete CUDA core test suite of build_gpu (cuda-tests.md, open task 1):
# every core test executable, without a filter. The objects are compiled on the
# login node first (from the /work path):
#   cd /work/04/jh220036a/i18048/tests/kokkos/build_gpu
#   cmake --build . -j 8 --target <the targets below> -- -k
# (linking fails there: libcuda.so exists only on the GPU nodes). This job links
# and runs. Exit status 0 if every executable passed.
#
# Submit from the Kokkos checkout:   pjsub sub_full_A100.sh

set +e
. /etc/profile.d/modules.sh
module purge
module load gcc/12.2.0 cuda/12.6 cmake/3.24.0

TESTS="Cuda_SmokeTest Cuda1 Cuda2 Cuda3 Cuda_ViewSupport Serial_SmokeTest Serial1 Serial2 Serial_ViewSupport Default"
ROOT=$(pwd -P)
LOG_DIR=$ROOT/full_A100_logs.${PJM_JOBID:-local}
mkdir -p "$LOG_DIR"
echo "commit: $(git -C "$ROOT" rev-parse --short HEAD) on $(git -C "$ROOT" branch --show-current)"
echo "host:   $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "logs:   $LOG_DIR"

cd "$ROOT/build_gpu" || exit 2
declare -a NAMES RESULTS
status=0
for t in $TESTS; do
  target=Kokkos_CoreUnitTest_$t
  echo "===== Link $target"
  if ! make -j8 -f core/unit_test/CMakeFiles/$target.dir/build.make \
       core/unit_test/$target > "$LOG_DIR/link_$t.log" 2>&1; then
    NAMES+=("$t"); RESULTS+=("FAILED to build (see $LOG_DIR/link_$t.log)"); status=1
    grep -m5 -E 'error|Error' "$LOG_DIR/link_$t.log"
    continue
  fi
  echo "===== Run $target"
  log=$LOG_DIR/run_$t.log
  (cd core/unit_test && ./$target) > "$log" 2>&1
  rc=$?
  grep -E '^\[ *(FAILED|PASSED) *\]|Failure' "$log" | sort -u | head -20
  nskip=$(grep -cE '^\[ *SKIPPED *\] [^ ]+ \(' "$log")
  if [ $rc = 0 ]; then
    RESULTS+=("$(grep -m1 PASSED "$log" | sed 's/.*PASSED *\] *//') ($nskip skipped by Kokkos)")
  else
    RESULTS+=("FAILED (exit $rc; see $log)"); status=1
  fi
  NAMES+=("$t")
done

echo "===== Summary"
for i in "${!NAMES[@]}"; do printf '  %-22s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}"; done
echo "logs: $LOG_DIR"
exit $status
