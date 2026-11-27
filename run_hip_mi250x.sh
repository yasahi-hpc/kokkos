#!/bin/bash
#SBATCH --account=gen2224
#SBATCH --job-name=hip-half
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH --output=run_hip_mi250x.%j.out

# Task 1 of hip-tests.md: validate the committed complex/half work on the HIP
# backend, where half_t / bhalf_t wrap ROCm's __half / __hip_bfloat16, on an
# MI250X (Kokkos_ARCH_AMD_GFX90A, one GCD).
#
# It configures Kokkos with hipcc (Cray PE, PrgEnv-amd), HIP + Serial, builds
# the test executables that hold the complex, half, numeric-trait and
# mathematical-function tests, prints the ROCm version and the half macros of
# both compilation passes, runs the tests and prints a summary.
#
# Submit from a checkout of refactor-complex (on a file system the compute
# nodes see, e.g. /lus/scratch/...):   sbatch run_hip_mi250x.sh
#
# Settings (sbatch --export=ALL,NAME=value ...):
#   KOKKOS_SRC   Kokkos source tree   (default: the submit directory)
#   BUILD_TYPE   CMake build type     (default: Debug; other types get a suffix)
#   BUILD_DIR    build directory      (default: $KOKKOS_SRC/build_hip_gfx90a[_<type>])
#   ARCH         Kokkos_ARCH_<ARCH>   (default: AMD_GFX90A)
#   CXX_FLAGS    CMAKE_CXX_FLAGS      (default: none, HIP clang's defaults)
#   JOBS         build parallelism    (default: number of CPUs)
#   RUN_TIMEOUT  seconds per test run (default: 1800)
# Output: also saved as hip_logs.<job id>/job.out, with the build and test logs.

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 2
set -uo pipefail

KOKKOS_SRC=${KOKKOS_SRC:-$PWD}
BUILD_TYPE=${BUILD_TYPE:-Debug}
BUILD_DIR=${BUILD_DIR:-$KOKKOS_SRC/build_hip_gfx90a$([ "$BUILD_TYPE" != Debug ] && echo "_${BUILD_TYPE,,}")}
ARCH=${ARCH:-AMD_GFX90A}
CXX_FLAGS=${CXX_FLAGS:-}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 16)}
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
JOBID=${SLURM_JOB_ID:-local.$(date +%Y%m%d-%H%M%S)}
LOG_DIR=$KOKKOS_SRC/hip_logs.$JOBID

die() { echo "ERROR: $*" >&2; exit 2; }
mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
exec > >(tee "$LOG_DIR/job.out") 2>&1

set +u  # the module scripts may read unset variables
module purge
module load craype-x86-trento craype-accel-amd-gfx90a
module load PrgEnv-amd
module load cmake/3.27.9
set -u
CXX=$(command -v hipcc) || die "hipcc not found"
[ -f "$KOKKOS_SRC/core/src/Kokkos_Complex.hpp" ] || die "KOKKOS_SRC=$KOKKOS_SRC is not a Kokkos source tree"
ROCM=${ROCM_PATH:-/opt/rocm}

# Each result also goes to steps.txt at once, so a job that is killed keeps them.
declare -a STEP_NAMES STEP_RESULTS
record() {
  STEP_NAMES+=("$1"); STEP_RESULTS+=("$2")
  printf '  %-58s %s\n' "$1" "$2" >> "$LOG_DIR/steps.txt"
}
space() { df -h "$LOG_DIR" 2>/dev/null | tail -1 | sed 's/^/  /'; }

echo "===== Environment"
echo "date:     $(date -Is)"
echo "job:      $JOBID on $(hostname)"
echo "source:   $KOKKOS_SRC ($(git -C "$KOKKOS_SRC" rev-parse --short HEAD 2>/dev/null) on $(git -C "$KOKKOS_SRC" branch --show-current 2>/dev/null))"
echo "rocm:     $ROCM, $(cat "$ROCM/.info/version" 2>/dev/null || echo 'version file not found')"
echo "compiler: $CXX: $("$CXX" --version 2>&1 | grep -m1 -i 'version')"
echo "cmake:    $(cmake --version | head -1)"
echo "build:    $BUILD_DIR ($BUILD_TYPE, Kokkos_ARCH_$ARCH, CMAKE_CXX_FLAGS=${CXX_FLAGS:-<none>})"
echo "logs:     $LOG_DIR"
echo "space:"; space
echo "===== GPUs"
(rocm-smi --showproductname 2>&1 || rocminfo 2>&1 | grep -m4 -E 'Marketing|gfx') | grep -v '^$' | head -12
module list 2>&1 | sed 's/^/  /'

echo "===== Configure"
cmake -S "$KOKKOS_SRC" -B "$BUILD_DIR" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_"$ARCH"=ON \
  -DKokkos_ENABLE_TESTS=ON \
  ${CXX_FLAGS:+-DCMAKE_CXX_FLAGS="$CXX_FLAGS"} \
  > "$LOG_DIR/configure.log" 2>&1
rc=$?
if [ $rc != 0 ]; then
  tail -30 "$LOG_DIR/configure.log"
  record configure "FAILED (see $LOG_DIR/configure.log)"
else
  record configure ok
fi

if [ $rc = 0 ]; then
  # The half macros as the tests see them, in the host and the device pass.
  echo "===== Kokkos half configuration (as seen by the tests)"
  F=$BUILD_DIR/core/unit_test/CMakeFiles/Kokkos_CoreUnitTest_Default.dir/flags.make
  FLAGS="$(grep '^CXX_DEFINES' "$F" | cut -d= -f2-) $(grep '^CXX_INCLUDES' "$F" | cut -d= -f2-) $(grep '^CXX_FLAGS' "$F" | cut -d= -f2-)"
  cat > "$LOG_DIR/half_probe.cpp" <<'EOF'
#include <Kokkos_Core.hpp>
PROBE_HIP_VERSION=HIP_VERSION_MAJOR.HIP_VERSION_MINOR.HIP_VERSION_PATCH
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
  # Preprocess each pass on its own (-E with both passes at once fails). The
  # flags are written for a shell (make passes them to /bin/sh), so use eval.
  for pass in host device; do
    (cd "$BUILD_DIR" && eval "\"\$CXX\" $FLAGS --offload-$pass-only -E -P \"\$LOG_DIR/half_probe.cpp\"" \
      > "$LOG_DIR/half_probe_$pass.out" 2>&1)
    echo "  $pass pass:"
    grep '^PROBE_' "$LOG_DIR/half_probe_$pass.out" | sed 's/^PROBE_/    /'
    grep -q '^PROBE_' "$LOG_DIR/half_probe_$pass.out" ||
      { echo "    probe failed:"; grep -m5 error "$LOG_DIR/half_probe_$pass.out"; }
  done
  echo "  (expected: NATIVE_HALF=1, NATIVE_BHALF=1, HALF_IS_CONSTEXPR=0, HALF_T_IS_FLOAT=false,"
  echo "   BHALF_T_IS_FLOAT=false; HALF_FULL_TYPE=1 in the device pass, and in the host pass"
  echo "   only for HIP >= 6.4)"
fi

# The executables: complex in <B>_SmokeTest, numeric traits and mathematical
# functions in Serial1 and HIP (every HIP test), half in Default (default
# execution space HIP).
TESTS="Serial_SmokeTest Serial1 HIP_SmokeTest HIP Default"
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
    [ "$nwarn" -gt 0 ] && grep 'warning:' "$LOG_DIR/build_$t.log" | sed 's/.*warning: //' | sort | uniq -c | sort -rn | head -10
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
  # Line-buffered, so a crash keeps the output; killed after RUN_TIMEOUT.
  echo "  start $(date +%T)"
  timeout -k 60 "$RUN_TIMEOUT" stdbuf -oL -eL "$exe" ${2:+--gtest_filter="$2"} > "$log" 2>&1
  local trc=$?
  echo "  end   $(date +%T), exit $trc$([ $trc = 124 ] && echo ' (timeout)')"
  [ -s "$log" ] || { echo "  empty log: no output was written (file system full?)"; space; }
  grep -E '^\[ *(FAILED|PASSED) *\]|Failure|Expected|Which is|actual|first failing' "$log" | sort -u | head -30
  local nskip
  nskip=$(grep -cE '^\[ *SKIPPED *\] [^ ]+ \(' "$log")
  record "$name" "$( [ $trc = 0 ] && grep -m1 'PASSED' "$log" | sed 's/.*PASSED *\] *//' || echo "FAILED (exit $trc, last started: $(grep '^\[ RUN' "$log" | tail -1 | sed 's/.*\] *//'), see $log)")$([ "$nskip" -gt 0 ] && echo " ($nskip skipped)")"
}
run_test Serial_SmokeTest '*complex*:*atomic*'
run_test HIP_SmokeTest '*complex*:*atomic*'
run_test Serial1 '*numeric_traits*'
run_test HIP '*numeric_traits*'
run_test Default '*half*'
# The SYCL lesson (signed zeros of the device math library): the mathematical
# functions, on Serial and HIP.
run_test Serial1 '*mathematical_functions*'
run_test HIP '*mathematical_functions*'

echo "===== Space at the end"
space
echo "===== Summary"
status=0
for i in "${!STEP_NAMES[@]}"; do
  printf '  %-58s %s\n' "${STEP_NAMES[$i]}" "${STEP_RESULTS[$i]}"
  case "${STEP_RESULTS[$i]}" in ok*|*tests*) ;; *) status=1 ;; esac
done
echo "expected: complex/atomic 27 (Serial and HIP); numeric_traits 14 + 1 skipped;"
echo "          half 16: 12 + 4 skipped (the std-only *_io_vs_std; *_constexpr_path_*,"
echo "          not constexpr on HIP); mathematical_functions: all pass"
echo "logs: $LOG_DIR (job output: $LOG_DIR/job.out)"
exit $status
