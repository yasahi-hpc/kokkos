#!/bin/bash -l
#PBS -N SYCLDEBUG
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=flare
#PBS -q debug
#PBS -A kokkos_math

# Minimal reproducer of the Debug-only failures of Kokkos's SYCL backend
# (sycl-tests.md, "Unrelated bugs"). They are not from the complex<half_t> work:
# the base a360acf37 fails identically (Aurora job 9128099), and Release passes.
#
#   1. Kokkos_CoreUnitTest_SYCL1A does not link: "relocation truncated to fit"
#   2. sycl_DeathTest.abort_from_device: the device message is missing
#   3. sycl.team_single_thread_int_ptr fails (TestTeamBasic.hpp:514)
#   4. sycl.team_for passes, but prints ~210,000 "TestTeamPolicy member(...)
#      error" lines (TestTeam.hpp:59 prints a mismatch and does not count it)
#   5. sycl.TeamThreadMDRangeParallelFor: GPU page fault, the process aborts
#   6. Kokkos_CoreUnitTest_SYCL2A: segmentation fault (the test is unknown:
#      earlier logs were lost; this run is line-buffered)
#
# Configures Kokkos (Debug, SYCL + Serial, Kokkos_ARCH_INTEL_PVC, tests on),
# builds only SYCL_SmokeTest, SYCL2A and SYCL1A, and runs each case on its own.
# Use any Kokkos checkout, e.g. develop, to see whether a bug is still there:
#
#   git clone https://github.com/kokkos/kokkos.git kokkos-develop
#   qsub -v KOKKOS_SRC=$PWD/kokkos-develop sycl_debug.sh
#
# Settings (qsub -v NAME=value):
#   KOKKOS_SRC  Kokkos source tree  (default: the submit directory)
#   BUILD_DIR   build directory     (default: $KOKKOS_SRC/build_sycl_debug)
#   BUILD_TYPE  CMake build type    (default: Debug; Release for comparison)
#   JOBS        build parallelism   (default: number of CPUs)
# Output: sycl_debug_logs.<job id>/ in the submit directory (job.out and logs).

cd "${PBS_O_WORKDIR:-$PWD}" || exit 2
set -uo pipefail

KOKKOS_SRC=${KOKKOS_SRC:-$PWD}
BUILD_TYPE=${BUILD_TYPE:-Debug}
BUILD_DIR=${BUILD_DIR:-$KOKKOS_SRC/build_sycl_${BUILD_TYPE,,}}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 8)}
JOBID=${PBS_JOBID:-local.$(date +%Y%m%d-%H%M%S)}
LOG_DIR=$PWD/sycl_debug_logs.${JOBID%%.*}
mkdir -p "$LOG_DIR" || exit 2
exec > >(tee "$LOG_DIR/job.out") 2>&1

set +u; module load cmake; set -u
CXX=$(command -v icpx) || { echo "icpx not found"; exit 2; }

declare -a RESULTS
note() { RESULTS+=("$(printf '  %-44s %s' "$1" "$2")"); }

echo "===== Environment"
echo "source:   $KOKKOS_SRC ($(git -C "$KOKKOS_SRC" rev-parse --short HEAD 2>/dev/null))"
echo "compiler: $("$CXX" --version | head -1)"
echo "build:    $BUILD_DIR ($BUILD_TYPE)"
echo "device:   $(sycl-ls 2>/dev/null | head -1)"
df -h "$PWD" | tail -1 | sed 's/^/space:    /'

echo "===== Configure"
cmake -S "$KOKKOS_SRC" -B "$BUILD_DIR" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_CXX_FLAGS=-fp-model=precise \
  -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_INTEL_PVC=ON \
  -DKokkos_ENABLE_TESTS=ON > "$LOG_DIR/configure.log" 2>&1 ||
  { tail -20 "$LOG_DIR/configure.log"; exit 2; }

for t in SYCL_SmokeTest SYCL2A SYCL1A; do
  echo "===== Build $t"
  if cmake --build "$BUILD_DIR" -j "$JOBS" --target Kokkos_CoreUnitTest_$t \
       > "$LOG_DIR/build_$t.log" 2>&1; then
    note "build $t" ok
  else
    n=$(grep -c 'relocation truncated to fit' "$LOG_DIR/build_$t.log")
    note "build $t" "FAILED ($n 'relocation truncated to fit')"
    grep -m3 -E 'relocation truncated|offset overflow|error:' "$LOG_DIR/build_$t.log" | cut -c1-200
  fi
done

UT=$BUILD_DIR/core/unit_test
# run <name> <executable> [filter]: line-buffered, 20 minutes at most. The
# "TestTeamPolicy member" lines are counted, and only the first 5 are kept.
run() {
  local name=$1 exe=$UT/Kokkos_CoreUnitTest_$2 log=$LOG_DIR/run_$1.log
  [ -x "$exe" ] || { note "$name" "NOT RUN (not built)"; return; }
  echo "===== Run $name"
  timeout -k 60 1200 stdbuf -oL -eL "$exe" ${3:+--gtest_filter="$3"} 2>&1 |
    awk '/TestTeamPolicy member/ { if (++n > 5) next } { print; fflush() }
         END { if (n) print "[ note ] " n " TestTeamPolicy member lines" }' > "$log"
  local rc=${PIPESTATUS[0]}
  grep -E 'Failure|Expected|Actual|Which is|Segmentation fault|Abort|TestTeamPolicy member lines|^\[ *(FAILED|PASSED|OK) *\]' "$log" |
    grep -v '^\[       OK \].*' | head -12
  local last; last=$(grep '^\[ RUN' "$log" | tail -1 | sed 's/.*\] *//')
  local res
  if [ "$rc" = 0 ] && grep -q '^\[  PASSED  \]' "$log" && ! grep -q '^\[  FAILED  \]' "$log"; then
    res="passed"
  else
    res="FAILED (exit $rc$([ "$rc" = 124 ] && echo ', timeout'), last started: ${last:-none})"
  fi
  grep -q 'TestTeamPolicy member lines' "$log" &&
    res="$res; $(grep -o '[0-9]* TestTeamPolicy member lines' "$log")"
  note "$name" "$res"
}
run abort_from_device          SYCL_SmokeTest 'sycl_DeathTest.abort_from_device'
run team_single_thread_int_ptr SYCL_SmokeTest 'sycl.team_single_thread_int_ptr'
run team_for                   SYCL_SmokeTest 'sycl.team_for'
run TeamThreadMDRangeParallelFor SYCL_SmokeTest 'sycl.TeamThreadMDRangeParallelFor'
# All of SYCL_SmokeTest in order, in case a failure needs the tests before it.
run SYCL_SmokeTest_all         SYCL_SmokeTest
run SYCL2A                     SYCL2A
run SYCL1A                     SYCL1A

echo "===== Summary ($BUILD_TYPE)"
printf '%s\n' "${RESULTS[@]}"
echo "expected in Debug (jobs 9127705, 9128099): SYCL1A link FAILED; abort_from_device,"
echo "  team_single_thread_int_ptr FAILED; team_for passed with ~210738 member lines;"
echo "  TeamThreadMDRangeParallelFor FAILED (GPU page fault, exit 134); SYCL2A FAILED (exit 139)"
echo "  (seen in the full SYCL_SmokeTest; a case that passes alone needs the tests before it)"
echo "logs: $LOG_DIR"
