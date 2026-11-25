#!/bin/bash -l
#PBS -N SYCLHALF
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=2:00:00
#PBS -l filesystems=flare
#PBS -q capacity
#PBS -A kokkos_math

# Validate the committed complex/half work on the SYCL backend (sycl-tests.md,
# part 1): half_t / bhalf_t wrap sycl::half / sycl::ext::oneapi::bfloat16, on
# Aurora's Intel Data Center GPU Max (Kokkos_ARCH_INTEL_PVC, with FP64).
#
# It configures Kokkos with icpx (available by default on Aurora; only cmake is
# loaded), SYCL + Serial, builds the test executables that hold the complex, half
# and numeric-trait tests, prints the half macros and the SYCL devices, runs the
# tests, and prints a summary. Runs over 1 hour must use the capacity queue; an
# AOT build for PVC takes long, so the header asks for 2 hours there.
#
# Submit from a checkout of refactor-complex:   qsub run_sycl_aurora.sh
#
# Settings (qsub -v NAME=value):
#   KOKKOS_SRC   Kokkos source tree    (default: the submit directory)
#   BUILD_DIR    build directory       (default: $KOKKOS_SRC/build_sycl_pvc[_<type>])
#   BUILD_TYPE   CMake build type      (default: Debug; other types get a suffix)
#   ARCH         Kokkos_ARCH_<ARCH>    (default: INTEL_PVC)
#   JOBS         build parallelism     (default: number of CPUs)
#   FULL         ON: also build and run every core test executable, without a
#                filter (needs more time: qsub -l walltime=6:00:00 -v FULL=ON ...)
# Output: also saved as sycl_logs.<job id>/job.out, with the build and test logs.

cd "${PBS_O_WORKDIR:-$PWD}" || exit 2
set -uo pipefail

KOKKOS_SRC=${KOKKOS_SRC:-$PWD}
BUILD_TYPE=${BUILD_TYPE:-Debug}
BUILD_DIR=${BUILD_DIR:-$KOKKOS_SRC/build_sycl_pvc$([ "$BUILD_TYPE" != Debug ] && echo "_${BUILD_TYPE,,}")}
ARCH=${ARCH:-INTEL_PVC}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 8)}
FULL=${FULL:-OFF}
JOBID=${PBS_JOBID:-local.$(date +%Y%m%d-%H%M%S)}
LOG_DIR=$KOKKOS_SRC/sycl_logs.${JOBID%%.*}

declare -a STEP_NAMES STEP_RESULTS
record() { STEP_NAMES+=("$1"); STEP_RESULTS+=("$2"); }
die() { echo "ERROR: $*" >&2; exit 2; }

mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
exec > >(tee "$LOG_DIR/job.out") 2>&1

export TERM=${TERM:-dumb}
set +u  # the module scripts may read unset variables
module load cmake || echo "WARNING: module load cmake failed; using $(command -v cmake || echo 'no cmake')"
set -u
command -v cmake >/dev/null 2>&1 || die "cmake not found"
CXX=$(command -v icpx) || die "icpx not found (Aurora provides it by default)"
[ -f "$KOKKOS_SRC/core/src/Kokkos_Complex.hpp" ] || die "KOKKOS_SRC=$KOKKOS_SRC is not a Kokkos source tree"

echo "===== Environment"
echo "date:     $(date -Is)"
echo "job:      $JOBID"
echo "host:     $(hostname)"
echo "source:   $KOKKOS_SRC ($(git -C "$KOKKOS_SRC" rev-parse --short HEAD 2>/dev/null) on $(git -C "$KOKKOS_SRC" branch --show-current 2>/dev/null))"
echo "compiler: $CXX: $("$CXX" --version | head -1)"
echo "cmake:    $(cmake --version | head -1)"
echo "build:    $BUILD_DIR ($BUILD_TYPE, Kokkos_ARCH_$ARCH)"
echo "logs:     $LOG_DIR"
echo "===== SYCL devices"
(sycl-ls 2>&1 || echo "sycl-ls not available") | head -20

echo "===== Configure"
cmake -S "$KOKKOS_SRC" -B "$BUILD_DIR" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DKokkos_ENABLE_SYCL=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_"$ARCH"=ON \
  -DKokkos_ENABLE_TESTS=ON \
  > "$LOG_DIR/configure.log" 2>&1
rc=$?
if [ $rc != 0 ]; then
  tail -30 "$LOG_DIR/configure.log"
  record configure "FAILED (see $LOG_DIR/configure.log)"
else
  record configure ok
fi

if [ $rc = 0 ]; then
  # The half macros as the tests see them.
  echo "===== Kokkos half configuration (as seen by the tests)"
  F=$BUILD_DIR/core/unit_test/CMakeFiles/Kokkos_CoreUnitTest_Default.dir/flags.make
  FLAGS="$(grep '^CXX_DEFINES' "$F" | cut -d= -f2-) $(grep '^CXX_INCLUDES' "$F" | cut -d= -f2-) $(grep '^CXX_FLAGS' "$F" | cut -d= -f2-)"
  cat > "$LOG_DIR/half_probe.cpp" <<'EOF'
#include <Kokkos_Core.hpp>
#ifdef KOKKOS_HAS_NATIVE_HALF_TYPE
PROBE_NATIVE_HALF=1
#else
PROBE_NATIVE_HALF=0
#endif
#ifdef KOKKOS_HAS_NATIVE_BHALF_TYPE
PROBE_NATIVE_BHALF=1
#else
PROBE_NATIVE_BHALF=0
#endif
#ifdef KOKKOS_IMPL_HALF_IS_CONSTEXPR
PROBE_HALF_IS_CONSTEXPR=1
#else
PROBE_HALF_IS_CONSTEXPR=0
#endif
#ifdef KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH
PROBE_HALF_FULL_TYPE=1
#else
PROBE_HALF_FULL_TYPE=0
#endif
PROBE_HALF_T_IS_FLOAT=KOKKOS_HALF_T_IS_FLOAT
PROBE_BHALF_T_IS_FLOAT=KOKKOS_BHALF_T_IS_FLOAT
EOF
  # shellcheck disable=SC2086
  (cd "$BUILD_DIR" && "$CXX" $FLAGS -E -P "$LOG_DIR/half_probe.cpp" 2>&1 | grep '^PROBE_' | sed 's/^PROBE_/  /')
  echo "  (expected: NATIVE_HALF=1, NATIVE_BHALF=1, HALF_IS_CONSTEXPR=0, HALF_T_IS_FLOAT=false, BHALF_T_IS_FLOAT=false)"
fi

# The executables: complex in <B>_SmokeTest, numeric traits in Serial1 / SYCL1B,
# half in Default (default execution space SYCL). FULL=ON: all core tests.
TESTS="Serial_SmokeTest Serial1 SYCL_SmokeTest SYCL1B Default"
[ "$FULL" = ON ] && TESTS="$TESTS Serial2 Serial_ViewSupport SYCL1A SYCL2A SYCL2B SYCL2C SYCL2D SYCL3 SYCL_ViewSupport"
if [ $rc = 0 ]; then
  for t in kokkoscore $(for t in $TESTS; do echo Kokkos_CoreUnitTest_$t; done); do
    echo "===== Build $t"
    cmake --build "$BUILD_DIR" -j "$JOBS" --target "$t" > "$LOG_DIR/build_$t.log" 2>&1
    brc=$?
    nwarn=$(grep -c 'warning:' "$LOG_DIR/build_$t.log")
    if [ $brc = 0 ]; then
      record "build $t" "ok ($nwarn warnings)"
    else
      record "build $t" "FAILED (see $LOG_DIR/build_$t.log)"
      grep -m10 -E 'error' "$LOG_DIR/build_$t.log"
    fi
    [ "$nwarn" -gt 0 ] && grep 'warning:' "$LOG_DIR/build_$t.log" | sort | uniq -c | sort -rn | head -10
  done
fi

run_test() {  # run_test <executable suffix> <filter>
  local exe=$BUILD_DIR/core/unit_test/Kokkos_CoreUnitTest_$1 name="run $1 $2"
  if [ ! -x "$exe" ]; then
    record "$name" "NOT RUN (not built)"
    return
  fi
  echo "===== Run $1 $2"
  local log="$LOG_DIR/run_$1_$(echo "$2" | tr -c 'A-Za-z0-9' _).log"
  "$exe" ${2:+--gtest_filter="$2"} > "$log" 2>&1
  local trc=$?
  grep -E '^\[ *(FAILED|PASSED) *\]|Failure|Expected|Which is|actual' "$log" | sort -u | head -30
  local nskip
  nskip=$(grep -cE '^\[ *SKIPPED *\] [^ ]+ \(' "$log")
  record "$name" "$( [ $trc = 0 ] && grep -m1 'PASSED' "$log" | sed 's/.*PASSED *\] *//' || echo "FAILED (exit $trc, see $log)")$([ "$nskip" -gt 0 ] && echo " ($nskip skipped)")"
}
run_test Serial_SmokeTest '*complex*:*atomic*'
run_test SYCL_SmokeTest '*complex*:*atomic*'
run_test Serial1 '*numeric_traits*'
run_test SYCL1B '*numeric_traits*'
run_test Default '*half*'
if [ "$FULL" = ON ]; then
  for t in $TESTS; do run_test "$t" ''; done
fi

echo "===== Summary"
status=0
for i in "${!STEP_NAMES[@]}"; do
  printf '  %-58s %s\n' "${STEP_NAMES[$i]}" "${STEP_RESULTS[$i]}"
  case "${STEP_RESULTS[$i]}" in ok*|*tests*) ;; *) status=1 ;; esac
done
echo "expected: complex/atomic 27 (Serial and SYCL); numeric_traits 14 + 1 skipped;"
echo "          half 16: 12 + 4 skipped (std-only *_io_vs_std; *_constexpr_path_* until B4b-SYCL)"
echo "logs: $LOG_DIR (job output: $LOG_DIR/job.out)"
exit $status
