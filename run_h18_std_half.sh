#!/bin/bash -l
#PBS -N H18
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=flare
#PBS -q debug
#PBS -A kokkos_math

# Verify H18 on the std half backend (complex_half.md, H18 note and §8.5):
# cast_to_half(half_t) / cast_to_bhalf(bhalf_t) are now defined in
# Kokkos_STD_Half_Conversion.hpp, so assigning a numeric-trait constant,
#   h = Kokkos::infinity<half_t>::value;
# compiles without "used but never defined" and links.
#
# For each build (default: x86-64 baseline; native: Kokkos_ARCH_SPR) the script
#   1. configures Kokkos (Serial, C++23) and builds kokkoscore and
#      Kokkos_CoreUnitTest_Default; the build directories are those of
#      run_std_half_fp16.sh (build_std_half_<config>), so an earlier build is reused;
#   2. compiles and links a small program that assigns trait constants to half_t and
#      bhalf_t, checks the compiler output for "used but never defined" and the binary
#      for undefined cast_to_half/cast_to_bhalf symbols, and runs it;
#   3. runs Kokkos_CoreUnitTest_Default '*half*' (10 tests, including the new
#      half_assign_trait_constants and bhalf_assign_trait_constants).
#
# Submit from a checkout of refactor-complex at 1220b11de or later:
#     qsub run_h18_std_half.sh
# Settings (qsub -v NAME=value): KOKKOS_SRC, BUILD_ROOT, CXX (GCC >= 13; a non-GCC CXX
# such as oneAPI's icpx is ignored), CONFIGS (default "default native"), ARCH (SPR), JOBS.
# Output: also saved as h18_logs.<job id>/job.out. Exit status 0 if everything passed.

cd "${PBS_O_WORKDIR:-$PWD}" || exit 2
set -uo pipefail

KOKKOS_SRC=${KOKKOS_SRC:-$PWD}
BUILD_ROOT=${BUILD_ROOT:-$KOKKOS_SRC}
CONFIGS=${CONFIGS:-default native}
ARCH=${ARCH:-SPR}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 8)}
JOBID=${PBS_JOBID:-local.$(date +%Y%m%d-%H%M%S)}
LOG_DIR=$BUILD_ROOT/h18_logs.${JOBID%%.*}

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

[ -f "$KOKKOS_SRC/core/src/impl/Kokkos_STD_Half_Conversion.hpp" ] ||
  die "KOKKOS_SRC=$KOKKOS_SRC is not a Kokkos source tree"
grep -q 'cast_to_bhalf(bhalf_t val)' "$KOKKOS_SRC/core/src/impl/Kokkos_STD_Half_Conversion.hpp" ||
  die "the source tree does not have the H18 fix (commit 1220b11de)"

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
if grep -qw avx512_fp16 /proc/cpuinfo 2>/dev/null; then CPU_FP16=yes; else CPU_FP16=no; fi

echo "===== Environment"
echo "date:     $(date -Is)"
echo "job:      $JOBID"
echo "host:     $(hostname) ($(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//'))"
echo "source:   $KOKKOS_SRC ($(cd "$KOKKOS_SRC" && git rev-parse --short HEAD 2>/dev/null) on $(cd "$KOKKOS_SRC" && git branch --show-current 2>/dev/null))"
echo "compiler: $CXX: $("$CXX" --version | head -1)"
echo "configs:  $CONFIGS (native: Kokkos_ARCH_$ARCH; fp16 cpu: $CPU_FP16)"
echo "logs:     $LOG_DIR"

# The link check: assignment from trait constants, host only.
PROG_SRC=$LOG_DIR/h18_assign.cpp
cat > "$PROG_SRC" <<'EOF'
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstdio>

template <class H>
int check(const char* name) {
  int errors = 0;
  auto one = [&](const char* what, H assigned, H direct) {
    const auto a = Kokkos::bit_cast<std::uint16_t>(assigned);
    const auto d = Kokkos::bit_cast<std::uint16_t>(direct);
    std::printf("  %-7s %-11s assigned 0x%04x  direct 0x%04x  %s\n", name, what, a, d,
                a == d ? "ok" : "MISMATCH");
    errors += a != d;
  };
  H x;
  x = Kokkos::infinity<H>::value;   one("infinity", x, H(Kokkos::infinity<H>::value));
  x = Kokkos::epsilon<H>::value;    one("epsilon", x, H(Kokkos::epsilon<H>::value));
  x = Kokkos::finite_max<H>::value; one("finite_max", x, H(Kokkos::finite_max<H>::value));
  x = Kokkos::norm_min<H>::value;   one("norm_min", x, H(Kokkos::norm_min<H>::value));
  return errors;
}

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  static_assert(sizeof(Kokkos::Experimental::half_t) == 2 &&
                sizeof(Kokkos::Experimental::bhalf_t) == 2,
                "half_t/bhalf_t are float: not the std backend");
  const int errors = check<Kokkos::Experimental::half_t>("half_t") +
                     check<Kokkos::Experimental::bhalf_t>("bhalf_t");
  std::printf("  %s\n", errors ? "FAILED" : "all assignments match");
  return errors != 0;
}
EOF

for cfg in $CONFIGS; do
  case $cfg in
    default) ARCH_ARGS=() ;;
    native) ARCH_ARGS=("-DKokkos_ARCH_$ARCH=ON") ;;
    *) die "unknown config '$cfg' in CONFIGS (default, native)" ;;
  esac
  B=$BUILD_ROOT/build_std_half_$cfg
  runnable=yes
  [ "$cfg" = native ] && [ "$CPU_FP16" != yes ] && runnable=no

  echo "===== [$cfg] Configure and build ($B)"
  cmake -S "$KOKKOS_SRC" -B "$B" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CXX_STANDARD=23 \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_BUILD_TYPE=Debug \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_TESTS=ON \
    -DKokkos_ENABLE_DEBUG=ON \
    "${ARCH_ARGS[@]}" \
    > "$LOG_DIR/${cfg}_configure.log" 2>&1
  if [ $? != 0 ]; then
    record "$cfg: configure" "FAILED (see $LOG_DIR/${cfg}_configure.log)"
    tail -20 "$LOG_DIR/${cfg}_configure.log"
    continue
  fi
  record "$cfg: configure" ok
  for t in kokkoscore Kokkos_CoreUnitTest_Default; do
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
  done

  echo "===== [$cfg] Link check: assignment from trait constants"
  F=$B/core/unit_test/CMakeFiles/Kokkos_CoreUnitTest_Default.dir/flags.make
  FLAGS="$(grep '^CXX_DEFINES' "$F" | cut -d= -f2-) $(grep '^CXX_INCLUDES' "$F" | cut -d= -f2-) $(grep '^CXX_FLAGS' "$F" | cut -d= -f2-)"
  EXE=$LOG_DIR/${cfg}_h18_assign
  # shellcheck disable=SC2086
  (cd "$B" && "$CXX" $FLAGS -o "$EXE" "$PROG_SRC" core/src/libkokkoscore*.a -ldl -lpthread) \
    > "$LOG_DIR/${cfg}_h18_assign_build.log" 2>&1
  rc=$?
  nundef=$(grep -c 'used but never defined' "$LOG_DIR/${cfg}_h18_assign_build.log")
  if [ $rc != 0 ] || [ "$nundef" != 0 ]; then
    record "$cfg: link check" "FAILED (exit $rc, $nundef 'used but never defined'; see $LOG_DIR/${cfg}_h18_assign_build.log)"
    grep -m10 -E 'error|undefined|never defined' "$LOG_DIR/${cfg}_h18_assign_build.log"
  else
    undef=$(nm -C "$EXE" | grep -E ' U .*cast_to_b?half' | wc -l)
    echo "  compiled and linked; undefined cast_to_half/cast_to_bhalf symbols: $undef"
    if [ "$runnable" = yes ]; then
      "$EXE"
      rc=$?
      record "$cfg: link check" "$( [ $rc = 0 ] && [ "$undef" = 0 ] && echo ok || echo "FAILED (exit $rc, $undef undefined symbols)")"
    else
      record "$cfg: link check" "$( [ "$undef" = 0 ] && echo 'ok (linked; not run: CPU lacks avx512_fp16)' || echo "FAILED ($undef undefined symbols)")"
    fi
  fi

  if [ "$runnable" = yes ] && [ -x "$B/core/unit_test/Kokkos_CoreUnitTest_Default" ]; then
    echo "===== [$cfg] Run Kokkos_CoreUnitTest_Default *half*"
    log=$LOG_DIR/${cfg}_run_half.log
    "$B/core/unit_test/Kokkos_CoreUnitTest_Default" --gtest_filter='*half*' > "$log" 2>&1
    rc=$?
    grep -E '^\[ *(OK|FAILED|PASSED) *\]|Failure|Expected|Which is' "$log" | tail -30
    new=$(grep -cE '^\[ *OK *\] .*b?half_assign_trait_constants' "$log")
    record "$cfg: run *half*" "$( [ $rc = 0 ] && [ "$new" = 2 ] && grep -m1 PASSED "$log" | sed 's/.*PASSED *\] *//;s/$/ (new tests: 2 of 2)/' || echo "FAILED (exit $rc; new tests passed: $new of 2; see $log)")"
  else
    record "$cfg: run *half*" "NOT RUN"
  fi
done

echo "===== Summary"
status=0
for i in "${!STEP_NAMES[@]}"; do
  printf '  %-36s %s\n' "${STEP_NAMES[$i]}" "${STEP_RESULTS[$i]}"
  case "${STEP_RESULTS[$i]}" in ok*|*tests*) ;; *) status=1 ;; esac
done
echo "logs: $LOG_DIR (job output: $LOG_DIR/job.out)"
exit $status
