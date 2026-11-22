#!/bin/bash -l
#PBS -N BF16WARN
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:15:00
#PBS -l filesystems=flare
#PBS -q debug
#PBS -A kokkos_math

# Why GCC >= 13 warns "'__bfloat16' is redefined from typedef 'short' to real '__bf16'
# since GCC 13.1 ..." at 7 test lines with -march=sapphirerapids (complex_half.md §8.2,
# run_std_half_fp16.sh job 9116933). Each of those lines initializes a bhalf_t from a
# numeric-trait constant, a BitComparisonWrapper (one std::uint16_t), which only copies
# bits.
#
# Hypothesis: the x86 target hook ix86_invalid_conversion warns, when BF16 is enabled,
# on any conversion whose source and destination have the modes BFmode and HImode, and
# a struct with a single member takes that member's mode. So the class-to-class
# conversion BitComparisonWrapper (HImode) -> floating_point_wrapper<__bf16> (BFmode)
# looks like a short <-> __bf16 conversion: a false positive.
#
# Each case is a separate compilation (-c, -O0) of the probe below with -DCASE=<name>,
# under four flag sets; the table counts the warnings. Compile-only, no execution.
#
# Submit from the Kokkos checkout:   qsub run_bf16_warning_probe.sh
# The probe needs Kokkos's generated headers (KokkosCore_config.h, ...) and include
# flags. The script configures Kokkos itself (configure only, Serial, C++23, no build;
# a few seconds) under the log directory, unless BUILD_DIR names an existing
# configured build.
#
# Settings: KOKKOS_SRC (default: the submit directory), BUILD_DIR (optional), CXX
# (GCC >= 13; a non-GCC CXX such as oneAPI's icpx is ignored).
# Output: also saved as bf16_warning_logs.<job id>/job.out, with one log per compile.

cd "${PBS_O_WORKDIR:-$PWD}" || exit 2
set -uo pipefail

KOKKOS_SRC=${KOKKOS_SRC:-$PWD}
BUILD_DIR=${BUILD_DIR:-}
JOBID=${PBS_JOBID:-local.$(date +%Y%m%d-%H%M%S)}
LOG_DIR=$KOKKOS_SRC/bf16_warning_logs.${JOBID%%.*}

die() { echo "ERROR: $*" >&2; exit 2; }
mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
exec > >(tee "$LOG_DIR/job.out") 2>&1

export TERM=${TERM:-dumb}
set +u  # the module scripts may read unset variables
module load cmake || echo "WARNING: module load cmake failed; using $(command -v cmake || echo 'no cmake')"
set -u
command -v cmake >/dev/null 2>&1 || die "cmake not found"

# The compiler: GCC >= 13 (same rule as run_std_half_fp16.sh).
gcc_major() { "$1" -dumpversion 2>/dev/null | cut -d. -f1; }
is_gcc() {
  local m
  m=$(echo | "$1" -dM -E -x c++ - 2>/dev/null) || return 1
  echo "$m" | grep -q '__GNUC__' && ! echo "$m" | grep -qE '__clang__|__INTEL_LLVM_COMPILER|__INTEL_COMPILER'
}
if [ -n "${CXX:-}" ] && ! is_gcc "$CXX"; then
  echo "NOTE: ignoring CXX=$CXX from the environment (not GCC)"
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
[ "$(gcc_major "$CXX")" -ge 13 ] 2>/dev/null || die "$CXX is not GCC >= 13"

# Include flags of the kokkoscore target; flags.make is written at configure time.
flags_make() { echo "$1/core/src/CMakeFiles/kokkoscore.dir/flags.make"; }
if [ -n "$BUILD_DIR" ]; then
  [ -f "$(flags_make "$BUILD_DIR")" ] || die "BUILD_DIR=$BUILD_DIR is not a configured Kokkos build"
else
  BUILD_DIR=$LOG_DIR/kokkos_config
  echo "===== Configure Kokkos (configure only) in $BUILD_DIR"
  cmake -S "$KOKKOS_SRC" -B "$BUILD_DIR" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CXX_STANDARD=23 \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DKokkos_ENABLE_SERIAL=ON \
    > "$LOG_DIR/configure.log" 2>&1 || { tail -30 "$LOG_DIR/configure.log"; die "configure failed"; }
  [ -f "$(flags_make "$BUILD_DIR")" ] || die "configure did not write $(flags_make "$BUILD_DIR")"
fi
INCLUDES=$(grep '^CXX_INCLUDES' "$(flags_make "$BUILD_DIR")" | cut -d= -f2-)

echo "===== Environment"
echo "date:     $(date -Is)"
echo "host:     $(hostname) ($(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//'))"
echo "compiler: $CXX: $("$CXX" --version | head -1)"
echo "source:   $KOKKOS_SRC ($(cd "$KOKKOS_SRC" && git rev-parse --short HEAD 2>/dev/null))"
echo "build:    $BUILD_DIR (for KokkosCore_config.h)"
echo "logs:     $LOG_DIR"

SRC=$LOG_DIR/bf16_warn_probe.cpp
cat > "$SRC" <<'EOF'
// One case per compilation, selected with -DCASE_<name>.
#include <cstdint>
#include <cstring>
#include <stdfloat>
#if defined(CASE_K1) || defined(CASE_K2) || defined(CASE_K3) || defined(CASE_K4) || \
    defined(CASE_K5)
#include <Kokkos_Core.hpp>
#endif

// Stand-ins for BitComparisonWrapper and floating_point_wrapper<__bf16>.
struct Bits {               // one std::uint16_t: mode HImode
  std::uint16_t value;
};
struct BitsPadded {         // 4 bytes: not HImode
  std::uint16_t value;
  std::uint16_t pad;
};
struct BF {                 // one __bf16: mode BFmode
  std::bfloat16_t val;
  BF() : val() {}
  BF(Bits b) { std::memcpy(&val, &b, sizeof val); }
  BF(BitsPadded b) { std::memcpy(&val, &b, sizeof val); }
};

#if defined(CASE_M1)  // class -> class, copy-init: a user-defined conversion
BF f() { BF x = Bits{0x7f80}; return x; }
#elif defined(CASE_M2)  // direct-init: the argument is already a Bits, no conversion
BF f() { BF x(Bits{0x7f80}); return x; }
#elif defined(CASE_M3)  // copy-init from a 4-byte source
BF f() { BF x = BitsPadded{0x7f80, 0}; return x; }
#elif defined(CASE_M4)  // a real value conversion short -> bf16 (the header case)
std::bfloat16_t f(short s) { return s; }
#elif defined(CASE_M5)  // float -> bf16
std::bfloat16_t f(float s) { return static_cast<std::bfloat16_t>(s); }
#elif defined(CASE_K1)  // the test lines: bhalf_t from a numeric-trait constant
using B = Kokkos::Experimental::bhalf_t;
B f() { B const inf = Kokkos::infinity<B>::value; return inf; }
#elif defined(CASE_K2)  // the same, direct-init
using B = Kokkos::Experimental::bhalf_t;
B f() { B const inf(Kokkos::infinity<B>::value); return inf; }
#elif defined(CASE_K3)  // the same, by assignment
using B = Kokkos::Experimental::bhalf_t;
B f() { B x; x = Kokkos::infinity<B>::value; return x; }
#elif defined(CASE_K4)  // half_t (_Float16, HFmode): expected silent
using H = Kokkos::Experimental::half_t;
H f() { H const inf = Kokkos::infinity<H>::value; return inf; }
#elif defined(CASE_K5)  // bhalf_t from short: the header path fixed in 3c70cfdca
using B = Kokkos::Experimental::bhalf_t;
B f() { return B(short(1)); }
#else
#error "define one CASE_<name>"
#endif
EOF

CASES="M1 M2 M3 M4 M5 K1 K2 K3 K4 K5"
declare -A PREDICT=(
  [M1]=warn [M2]=silent [M3]=silent [M4]=warn [M5]=silent
  [K1]=warn [K2]=silent [K3]=warn [K4]=silent [K5]=silent
)
FLAGSETS=("none:" "bf16:-mavx512bf16" "fp16:-mavx512fp16" "spr:-march=sapphirerapids")

echo "===== Warnings per compilation ('__bfloat16' is redefined ...); prediction applies to bf16 and spr"
printf '  %-5s' case
for fs in "${FLAGSETS[@]}"; do printf '%-8s' "${fs%%:*}"; done
printf '%-12s %s\n' prediction verdict
status=0
for c in $CASES; do
  printf '  %-5s' "$c"
  row=()
  for fs in "${FLAGSETS[@]}"; do
    name=${fs%%:*} flags=${fs#*:}
    log=$LOG_DIR/${c}_$name.log
    # shellcheck disable=SC2086
    "$CXX" -std=c++23 $INCLUDES $flags -O0 -c -o /dev/null -DCASE_$c "$SRC" > "$log" 2>&1
    rc=$?
    n=$(grep -c "redefined from typedef" "$log")
    if [ $rc != 0 ]; then
      printf '%-8s' "ERR"
      row+=(ERR)
    else
      printf '%-8s' "$n"
      row+=("$n")
    fi
  done
  # Verdict: none/fp16 must be 0; bf16/spr must match the prediction.
  v=ok
  [ "${row[0]}" = 0 ] && [ "${row[2]}" = 0 ] || v=differs
  for i in 1 3; do
    case ${PREDICT[$c]}:${row[$i]} in
      warn:0 | silent:[1-9]* | *:ERR) v=differs ;;
    esac
  done
  [ "$v" = ok ] || status=1
  printf '%-12s %s\n' "${PREDICT[$c]}" "$v"
done

echo "===== First warning of each case (spr), with GCC's context"
for c in $CASES; do
  log=$LOG_DIR/${c}_spr.log
  if grep -q "redefined from typedef" "$log"; then
    echo "--- $c"
    grep -m1 -B3 "redefined from typedef" "$log" | sed 's/^/    /'
  fi
done
for c in $CASES; do
  for fs in "${FLAGSETS[@]}"; do
    log=$LOG_DIR/${c}_${fs%%:*}.log
    if grep -q 'error' "$log"; then
      echo "--- compile error in $c (${fs%%:*}):"
      grep -m5 'error' "$log" | sed 's/^/    /'
    fi
  done
done

echo "===== Result: $([ $status = 0 ] && echo 'all cases as predicted' || echo 'some cases differ from the prediction')"
echo "logs: $LOG_DIR"
exit $status
