#!/bin/bash -l
#PBS -N AFFINITY
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=flare
#PBS -q debug
#PBS -A kokkos_math

# The std half backend on a CPU with native half arithmetic (complex_half.md §8.2,
# D5 "CH3 vs. C++23 and libstdc++", and §8.5).
#
# half_t/bhalf_t wrap std::float16_t/std::bfloat16_t (GCC >= 13, C++23). The script builds
# and runs the complex/half tests twice, in separate build directories:
#   default  no Kokkos_ARCH_*: x86-64 baseline. GCC evaluates _Float16 in float
#            (__FLT_EVAL_METHOD_TS_18661_3__ = 0), as in job 4260499 on aarch64.
#   native   Kokkos_ARCH_SPR (-march=sapphirerapids, AVX512-FP16) on x86-64, or
#            Kokkos_ARCH_ARMV9_GRACE (-mcpu=neoverse-v2, Arm FP16) on aarch64. GCC evaluates
#            _Float16 in half (method 16), so the scalar half_t operators and
#            libstdc++'s std::complex<std::float16_t> multiplication change; Kokkos's
#            complex<half_t> * and / should not (CH3 computes in float).
#   gnu, gnu-native  the same two in Kokkos's default dialect for GCC, -std=gnu++23
#            (CMAKE_CXX_EXTENSIONS=ON; default and native use strict -std=c++23).
#            GNU mode changes GCC's -fexcess-precision default, which may change
#            how half expressions round.
# A build fails if GCC reports "used but never defined" (a cast function declared
# for the backend but not defined, as H18 was); the half tests include
# half_cast_overloads/bhalf_cast_overloads, which call every declared cast.
# For each build it records the evaluation method, the generated code for a half
# multiplication, and the results of a few expressions where the two methods differ.
#
# Submit from a checkout of refactor-complex:   qsub run_std_half_fp16.sh
# (or run it with bash; it then uses the current directory).
#
# Settings (environment variables, all optional; with PBS: qsub -v NAME=value):
#   KOKKOS_SRC   Kokkos source tree    (default: the submit directory)
#   BUILD_ROOT   parent of the builds  (default: $KOKKOS_SRC)
#   CXX          C++ compiler, GCC only (default: the first GCC >= 13 of g++-14, g++-13,
#                g++; a non-GCC CXX from the environment, e.g. oneAPI's icpx, is ignored)
#   ARCH         Kokkos_ARCH_<ARCH> of the native build (default: SPR on x86-64,
#                ARMV9_GRACE on aarch64)
#   CONFIGS      builds to run         (default: "default native gnu gnu-native")
#   BUILD_TYPE   CMake build type      (default: Debug)
#   JOBS         build parallelism     (default: number of CPUs)
#   ENABLE_OPENMP  ON: also enable the OpenMP backend; builds and runs
#                Kokkos_CoreUnitTest_OpenMP_SmokeTest and Kokkos_CoreUnitTest_OpenMP too,
#                in separate build directories build_std_half_<config>_omp (default:
#                OFF). OMP_NUM_THREADS defaults to 8 unless already set.
#   FULL         ON: also build Serial2, <backend>_ViewSupport and OpenMPInterOp, and run
#                every test executable without a filter (default: OFF). Tests that
#                Kokkos skips for missing features are counted, not failures. A FULL run needs more
#                than the 1 hour of the debug queue above; runs longer than 1 hour
#                must use the capacity queue:
#                qsub -q capacity -l walltime=3:00:00 \
#                     -v 'ENABLE_OPENMP=ON,FULL=ON,CONFIGS=default gnu-native' run_std_half_fp16.sh
#
# Output: the job's standard output, also saved as $LOG_DIR/job.out, where LOG_DIR is
# $BUILD_ROOT/std_half_fp16_logs.<job id>; build and test logs go to the same directory.
# Exit status: 0 if every step passed.

cd "${PBS_O_WORKDIR:-$PWD}" || exit 2

set -uo pipefail

KOKKOS_SRC=${KOKKOS_SRC:-$PWD}
BUILD_ROOT=${BUILD_ROOT:-$KOKKOS_SRC}
# CPU-specific settings: the native build's Kokkos_ARCH, the /proc/cpuinfo flag of
# native half arithmetic, and the instructions of a half multiplication, native or
# through float.
case $(uname -m) in
  aarch64)
    ARCH=${ARCH:-ARMV9_GRACE}
    CPU_FP16_FLAG=fphp
    NATIVE_MUL_RE='fmul[[:space:]]+h[0-9]+'
    FLOAT_MUL_RE='fcvt[[:space:]]+s[0-9]+, *h[0-9]+|__extendhfsf2'
    ;;
  *)
    ARCH=${ARCH:-SPR}
    CPU_FP16_FLAG=avx512_fp16
    NATIVE_MUL_RE='vmulsh'
    FLOAT_MUL_RE='vcvtph2ps|vcvtsh2ss|__extendhfsf2'
    ;;
esac
CONFIGS=${CONFIGS:-default native gnu gnu-native}
BUILD_TYPE=${BUILD_TYPE:-Debug}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 8)}
FULL=${FULL:-OFF}
ENABLE_OPENMP=${ENABLE_OPENMP:-OFF}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
BACKENDS=Serial
[ "$ENABLE_OPENMP" = ON ] && BACKENDS="Serial OpenMP"
JOBID=${PBS_JOBID:-local.$(date +%Y%m%d-%H%M%S)}
LOG_DIR=$BUILD_ROOT/std_half_fp16_logs.${JOBID%%.*}

declare -a STEP_NAMES STEP_RESULTS
record() { STEP_NAMES+=("$1"); STEP_RESULTS+=("$2"); }
die() { echo "ERROR: $*" >&2; exit 2; }

mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
# Everything below goes both to the job's standard output and to $LOG_DIR/job.out.
exec > >(tee "$LOG_DIR/job.out") 2>&1

# A batch job has no terminal: without TERM the module command prints tput errors.
export TERM=${TERM:-dumb}
set +u  # the module scripts may read unset variables
# A failure (e.g. a cmake module already loaded) only matters if no cmake is found.
module load cmake || echo "WARNING: module load cmake failed; using $(command -v cmake || echo 'no cmake')"
set -u
command -v cmake >/dev/null 2>&1 || die "cmake not found"

[ -f "$KOKKOS_SRC/core/src/Kokkos_Complex.hpp" ] ||
  die "KOKKOS_SRC=$KOKKOS_SRC is not a Kokkos source tree"

# The compiler: GCC >= 13 with std::float16_t in C++23. A CXX from the environment is
# used only if it is GCC: the oneAPI environment sets CXX=icpx (clang-based, which does
# not provide std::float16_t), and that would otherwise be picked up.
gcc_major() { "$1" -dumpversion 2>/dev/null | cut -d. -f1; }
is_gcc() {  # GCC proper: defines __GNUC__ but none of the clang-based compilers' macros
  local m
  m=$(echo | "$1" -dM -E -x c++ - 2>/dev/null) || return 1
  echo "$m" | grep -q '__GNUC__' && ! echo "$m" | grep -qE '__clang__|__INTEL_LLVM_COMPILER|__INTEL_COMPILER'
}
if [ -n "${CXX:-}" ] && ! is_gcc "$CXX"; then
  echo "NOTE: ignoring CXX=$CXX from the environment (not GCC); looking for GCC >= 13"
  CXX=
fi
if [ -z "${CXX:-}" ]; then
  for c in g++-14 g++-13 g++; do
    if command -v "$c" >/dev/null 2>&1 && [ "$(gcc_major "$c")" -ge 13 ] 2>/dev/null; then
      CXX=$(command -v "$c")
      break
    fi
  done
fi
[ -n "${CXX:-}" ] || die "no GCC >= 13 found (tried g++-14, g++-13, g++); set CXX"
command -v "$CXX" >/dev/null 2>&1 || die "compiler CXX=$CXX not found"
[ "$(gcc_major "$CXX")" -ge 13 ] 2>/dev/null ||
  die "$CXX is version $("$CXX" -dumpversion); GCC >= 13 is needed"
echo | "$CXX" -std=c++23 -dM -E -x c++ - | grep -q '__STDCPP_FLOAT16_T__' ||
  die "$CXX -std=c++23 does not define __STDCPP_FLOAT16_T__"

# Whether this CPU can run AVX512-FP16 code; without it the native build is built and
# probed, but not run (it would die with SIGILL).
if grep -qw "$CPU_FP16_FLAG" /proc/cpuinfo 2>/dev/null; then CPU_FP16=yes; else CPU_FP16=no; fi

echo "===== Environment"
echo "date:     $(date -Is)"
echo "job:      $JOBID"
echo "host:     $(hostname) ($(uname -m))"
echo "cpu:      $(lscpu 2>/dev/null | grep -m1 -E '^Model name' | cut -d: -f2- | sed 's/^ *//')"
echo "fp16 cpu: $CPU_FP16 ($CPU_FP16_FLAG in /proc/cpuinfo)"
echo "source:   $KOKKOS_SRC"
(cd "$KOKKOS_SRC" && echo "commit:   $(git rev-parse --short HEAD 2>/dev/null) on $(git branch --show-current 2>/dev/null)" &&
  git status --short --untracked-files=no 2>/dev/null | sed 's/^/          modified: /')
echo "compiler: $CXX: $("$CXX" --version | head -1)"
echo "cmake:    $(cmake --version | head -1)"
echo "configs:  $CONFIGS (native: Kokkos_ARCH_$ARCH)"
if [ "$FULL" = ON ] && [ "${PBS_QUEUE:-}" = debug ]; then
  echo "WARNING: FULL=ON in the debug queue (1 hour at most) will likely run out of time;"
  echo "         submit runs longer than 1 hour to the capacity queue: qsub -q capacity -l walltime=3:00:00 ..."
fi
echo "backends: $BACKENDS$([ "$ENABLE_OPENMP" = ON ] && echo " (OMP_NUM_THREADS=$OMP_NUM_THREADS)"); FULL=$FULL"
echo "logs:     $LOG_DIR"

# The probe program: expressions whose value depends on the evaluation method of
# _Float16, for std::float16_t and for Kokkos's half_t, scalar and complex.
PROBE_SRC=$LOG_DIR/fp16_probe.cpp
cat > "$PROBE_SRC" <<'EOF'
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdio>
#include <stdfloat>

using F16 = std::float16_t;
using H   = Kokkos::Experimental::half_t;
template <class T>
double d(T x) { return static_cast<double>(static_cast<float>(x)); }

// Inputs through volatile, so that nothing is folded at compile time.
volatile float v256 = 256.f, v65000 = 65000.f, v269 = 269.f, v111 = 111.5f,
               v60000 = 60000.f, v1 = 1.f;

int main() {
  std::printf("PROBE eval_method_ts=%d\n", __FLT_EVAL_METHOD_TS_18661_3__);
  // 256*256 = 65536 overflows half; in float the subtraction brings it back to
  // 65536 - 64992 = 544 (65000 is not representable in half; it rounds to 64992).
  const F16 a = static_cast<F16>(v256), c = static_cast<F16>(v65000);
  const F16 r = a * a - c;
  std::printf("PROBE std_float16_t   256*256-65000        = %g   (float: 544, half: inf)\n", d(r));
  const H ha(v256), hc(v65000);
  const H hr = ha * ha - hc;
  std::printf("PROBE kokkos_half_t   256*256-65000        = %g   (rounded per operation: inf)\n", d(hr));
  // complex: re = 269^2 - 111.5^2 = 59936.75, with 269^2 = 72361 > 65504.
  const std::complex<F16> z(static_cast<F16>(v269), static_cast<F16>(v111));
  const std::complex<F16> zz = z * z;
  std::printf("PROBE std_complex     (269,111.5)^2        = (%g, %g)   (float: (59936, 60000))\n",
              d(zz.real()), d(zz.imag()));
  const Kokkos::complex<H> k{H{v269}, H{v111}};  // braces: H(v269) would declare a parameter
  const Kokkos::complex<H> kk = k * k;
  std::printf("PROBE kokkos_complex  (269,111.5)^2        = (%g, %g)   (CH3: (59936, 60000))\n",
              d(kk.real()), d(kk.imag()));
  const std::complex<F16> q = std::complex<F16>(static_cast<F16>(v60000), static_cast<F16>(v60000)) /
                              std::complex<F16>(static_cast<F16>(v1), static_cast<F16>(v1));
  std::printf("PROBE std_complex     (60000,60000)/(1,1)  = (%g, %g)   (float: (60000, 0))\n",
              d(q.real()), d(q.imag()));
  const Kokkos::complex<H> kq =
      Kokkos::complex<H>(H(v60000), H(v60000)) / Kokkos::complex<H>(H(v1), H(v1));
  std::printf("PROBE kokkos_complex  (60000,60000)/(1,1)  = (%g, %g)   (CH3: (60000, 0))\n",
              d(kq.real()), d(kq.imag()));
}
EOF
# The code GCC generates for one multiplication, -O2.
ASM_SRC=$LOG_DIR/fp16_mul.cpp
cat > "$ASM_SRC" <<'EOF'
#include <Kokkos_Core.hpp>
#include <stdfloat>
std::float16_t mul_std(std::float16_t a, std::float16_t b) { return a * b; }
Kokkos::Experimental::half_t mul_kokkos(Kokkos::Experimental::half_t a,
                                        Kokkos::Experimental::half_t b) { return a * b; }
EOF

# Compile flags of a test target, for the probes.
test_flags() {  # test_flags <build dir>
  local f=$1/core/unit_test/CMakeFiles/Kokkos_CoreUnitTest_Serial_SmokeTest.dir/flags.make
  echo "$(grep '^CXX_DEFINES' "$f" | cut -d= -f2-) $(grep '^CXX_INCLUDES' "$f" | cut -d= -f2-) $(grep '^CXX_FLAGS' "$f" | cut -d= -f2-)"
}

run_test() {  # run_test <config> <build dir> <executable> <filter>
  local cfg=$1 exe=$2/core/unit_test/$3 name="$1: run $3 $4"
  if [ ! -x "$exe" ]; then
    record "$name" "NOT RUN (not built)"
    return
  fi
  if [ "$cfg" = native ] && [ "$CPU_FP16" != yes ]; then
    record "$name" "NOT RUN (CPU lacks $CPU_FP16_FLAG)"
    return
  fi
  echo "===== [$cfg] Run $3 $4"
  local log="$LOG_DIR/${cfg}_run_$3_$(echo "$4" | tr -c 'A-Za-z0-9' _).log"
  "$exe" ${4:+--gtest_filter="$4"} > "$log" 2>&1
  local rc=$?
  grep -E '^\[ *(OK|FAILED|PASSED|SKIPPED) *\]|Failure|Expected|Which is|actual' "$log" | tail -40
  # The std-backend tests (*_vs_std*) skip only without the std backend: a skip
  # there is a failure. Other tests skip for features this configuration lacks
  # (death tests, graphs, shared spaces): counted, not failures.
  local nstd nskip
  # gtest prints a skipped test twice: as it runs ("name (N ms)") and in the final
  # list; count the first.
  nstd=$(grep -cE '^\[ *SKIPPED *\] [^ ]+_vs_std[^ ]* \(' "$log")
  nskip=$(grep -E '^\[ *SKIPPED *\] [^ ]+ \(' "$log" | grep -cv '_vs_std')
  record "$name" "$( [ $rc = 0 ] && grep -m1 'PASSED' "$log" | sed 's/.*PASSED *\] *//' || echo "FAILED (exit $rc, see $log)")$([ "$nstd" -gt 0 ] && echo " ($nstd std-backend tests SKIPPED)")$([ "$nskip" -gt 0 ] && echo " ($nskip skipped by Kokkos)")"
}

# Test executables (core/unit_test/CMakeLists.txt): Serial is split into
# Serial1/Serial2, OpenMP is one executable, Kokkos_CoreUnitTest_OpenMP. MAIN_<B> is
# the one with the numeric traits; FULL=ON adds the rest. Plus Default (the default
# execution space).
MAIN_Serial=Kokkos_CoreUnitTest_Serial1
MAIN_OpenMP=Kokkos_CoreUnitTest_OpenMP
TESTS=""
for b in $BACKENDS; do
  main=MAIN_$b
  TESTS="$TESTS Kokkos_CoreUnitTest_${b}_SmokeTest ${!main}"
  if [ "$FULL" = ON ]; then
    TESTS="$TESTS Kokkos_CoreUnitTest_${b}_ViewSupport"
    [ "$b" = Serial ] && TESTS="$TESTS Kokkos_CoreUnitTest_Serial2"
    [ "$b" = OpenMP ] && TESTS="$TESTS Kokkos_CoreUnitTest_OpenMPInterOp"
  fi
done
TESTS="$TESTS Kokkos_CoreUnitTest_Default"
TARGETS="kokkoscore Kokkos_CoreTestCompileOnly$TESTS"

for cfg in $CONFIGS; do
  case $cfg in
    default) ARCH_ARGS=() EXT=OFF ;;
    native) ARCH_ARGS=("-DKokkos_ARCH_$ARCH=ON") EXT=OFF ;;
    gnu) ARCH_ARGS=() EXT=ON ;;
    gnu-native) ARCH_ARGS=("-DKokkos_ARCH_$ARCH=ON") EXT=ON ;;
    *) die "unknown config '$cfg' in CONFIGS (default, native, gnu, gnu-native)" ;;
  esac
  B=$BUILD_ROOT/build_std_half_$cfg$([ "$ENABLE_OPENMP" = ON ] && echo _omp)

  echo "===== [$cfg] Configure ($BUILD_TYPE, ${BACKENDS// / + }, ${ARCH_ARGS[*]:-no Kokkos_ARCH}, CMAKE_CXX_EXTENSIONS=$EXT)"
  cmake -S "$KOKKOS_SRC" -B "$B" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CXX_STANDARD=23 \
    -DCMAKE_CXX_EXTENSIONS="$EXT" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP="$ENABLE_OPENMP" \
    -DKokkos_ENABLE_TESTS=ON \
    -DKokkos_ENABLE_DEBUG=ON \
    "${ARCH_ARGS[@]}" \
    > "$LOG_DIR/${cfg}_configure.log" 2>&1
  rc=$?
  if [ $rc != 0 ]; then
    record "$cfg: configure" "FAILED (see $LOG_DIR/${cfg}_configure.log)"
    tail -30 "$LOG_DIR/${cfg}_configure.log"
    continue
  fi
  record "$cfg: configure" ok
  grep -E 'Architectures|^--  *[A-Z0-9_]+$|SPR|GRACE' "$LOG_DIR/${cfg}_configure.log" | head -5

  echo "===== [$cfg] Build"
  for t in $TARGETS; do
    cmake --build "$B" -j "$JOBS" --target "$t" > "$LOG_DIR/${cfg}_build_$t.log" 2>&1
    rc=$?
    nwarn=$(grep -c 'warning:' "$LOG_DIR/${cfg}_build_$t.log")
    nundef=$(grep -c 'used but never defined' "$LOG_DIR/${cfg}_build_$t.log")
    if [ $rc = 0 ] && [ "$nundef" = 0 ]; then
      record "$cfg: build $t" "ok ($nwarn warnings)"
    else
      record "$cfg: build $t" "FAILED (exit $rc, $nundef 'used but never defined'; see $LOG_DIR/${cfg}_build_$t.log)"
      grep -m10 -E 'error|used but never defined' "$LOG_DIR/${cfg}_build_$t.log"
    fi
    [ "$nwarn" -gt 0 ] && grep 'warning:' "$LOG_DIR/${cfg}_build_$t.log" | sort | uniq -c | sort -rn | head -10
  done

  # The probes, with the tests' own flags (so the arch flags are included).
  echo "===== [$cfg] Half evaluation (with the tests' flags)"
  FLAGS=$(test_flags "$B")
  echo "  dialect: $(echo "$FLAGS" | grep -oE -- '-std=[a-z+0-9]+' | sort -u | tr '\n' ' ')"
  # shellcheck disable=SC2086
  (cd "$B" && echo | "$CXX" $FLAGS -dM -E -x c++ - 2>&1 |
    grep -E '__FLT_EVAL_METHOD(_TS_18661_3)?__ |__AVX512FP16__ |__ARM_FEATURE_FP16_SCALAR_ARITHMETIC |__STDCPP_B?FLOAT16_T__ |__STRICT_ANSI__ ' |
    sed 's/^#define /  /')
  # shellcheck disable=SC2086
  (cd "$B" && echo '#include <Kokkos_Macros.hpp>
KOKKOS_HALF_T_IS_FLOAT KOKKOS_BHALF_T_IS_FLOAT' | "$CXX" $FLAGS -E -P -x c++ - 2>/dev/null | tail -1 |
    sed 's/^/  HALF_T_IS_FLOAT, BHALF_T_IS_FLOAT: /')
  # shellcheck disable=SC2086
  if (cd "$B" && "$CXX" $FLAGS -O2 -S -o "$LOG_DIR/${cfg}_fp16_mul.s" "$ASM_SRC") > "$LOG_DIR/${cfg}_fp16_mul.log" 2>&1; then
    echo "  half multiplication instructions (-O2): native half ($NATIVE_MUL_RE): $(grep -cE "$NATIVE_MUL_RE" "$LOG_DIR/${cfg}_fp16_mul.s"), through float ($FLOAT_MUL_RE): $(grep -cE "$FLOAT_MUL_RE" "$LOG_DIR/${cfg}_fp16_mul.s")"
    record "$cfg: codegen probe" ok
  else
    record "$cfg: codegen probe" "FAILED (see $LOG_DIR/${cfg}_fp16_mul.log)"
  fi
  # shellcheck disable=SC2086
  if (cd "$B" && "$CXX" $FLAGS -o "$LOG_DIR/${cfg}_fp16_probe" "$PROBE_SRC" core/src/libkokkoscore*.a -ldl -lpthread) > "$LOG_DIR/${cfg}_fp16_probe.log" 2>&1; then
    if [ "$cfg" = native ] && [ "$CPU_FP16" != yes ]; then
      record "$cfg: runtime probe" "NOT RUN (CPU lacks $CPU_FP16_FLAG)"
    else
      "$LOG_DIR/${cfg}_fp16_probe" | sed 's/^PROBE /  /'
      prc=${PIPESTATUS[0]}
      record "$cfg: runtime probe" "$( [ "$prc" = 0 ] && echo ok || echo "FAILED (exit $prc)")"
    fi
  else
    record "$cfg: runtime probe" "FAILED to build (see $LOG_DIR/${cfg}_fp16_probe.log)"
    tail -20 "$LOG_DIR/${cfg}_fp16_probe.log"
  fi

  for b in $BACKENDS; do
    run_test "$cfg" "$B" "Kokkos_CoreUnitTest_${b}_SmokeTest" '*complex*:*atomic*'
    main=MAIN_$b
    run_test "$cfg" "$B" "${!main}" '*numeric_traits*'
  done
  run_test "$cfg" "$B" Kokkos_CoreUnitTest_Default '*half*'
  if [ "$FULL" = ON ]; then
    for t in $TESTS; do
      run_test "$cfg" "$B" "$t" ''
    done
  fi
done

echo "===== Summary"
status=0
for i in "${!STEP_NAMES[@]}"; do
  printf '  %-70s %s\n' "${STEP_NAMES[$i]}" "${STEP_RESULTS[$i]}"
  case "${STEP_RESULTS[$i]}" in *SKIPPED*) status=1 ;; ok*|*tests*) ;; *) status=1 ;; esac
done
echo "logs: $LOG_DIR (job output: $LOG_DIR/job.out)"
exit $status
