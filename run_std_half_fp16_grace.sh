#!/bin/bash -e
#PBS -l select=1
#PBS -q regular-g
#PBS -l walltime=1:00:00
#PBS -W group_list=jh220036
#PBS -j oe

# run_std_half_fp16.sh on the Grace (aarch64, GCC 15) machine: the std half backend
# with and without native half arithmetic. The native builds use
# Kokkos_ARCH_ARMV9_GRACE (-mcpu=neoverse-v2), whose Arm FP16 extension makes GCC
# evaluate _Float16 in half (__FLT_EVAL_METHOD_TS_18661_3__ = 16); job 4260499 used
# Kokkos's default target, where it is float (0). See complex_half.md §8.2, §8.5.
#
# Submit from the Kokkos checkout:   qsub run_std_half_fp16_grace.sh
# The settings of run_std_half_fp16.sh apply (qsub -v NAME=value), e.g. CONFIGS,
# ENABLE_OPENMP, FULL. This header gives 1 hour; FULL=ON needs more:
#   qsub -l walltime=3:00:00 -v 'FULL=ON' run_std_half_fp16_grace.sh
# (check this machine's queue limits for runs over 1 hour).
# The module's assembler is too old for -mcpu=neoverse-v2; the script then uses the
# binutils of a newer gcc-toolset (see below, and aarch64_tests.md).

set +e
cd "${PBS_O_WORKDIR:-$PWD}" || exit 2

# A batch job has no terminal: without TERM the module command prints tput errors.
export TERM=${TERM:-dumb}
type module >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; }
module purge
module load cmake/3.31.1 gcc/15.2.0 || { echo "ERROR: module load cmake/3.31.1 gcc/15.2.0 failed" >&2; exit 2; }
export CXX=g++

# The gcc/15.2.0 module uses the system binutils (2.35), whose assembler rejects
# -mcpu=neoverse-v2 ("unknown architecture armv9-a+..."), so every native build would
# fail. If the assembler on PATH cannot do it, put the binutils (only those: the
# toolset's bin also has its own, older g++) of the newest gcc-toolset that can first on
# PATH, through a directory of symlinks. BINUTILS_DIR=<dir> (qsub -v) names one instead.
can_assemble_armv9() {  # can_assemble_armv9 [dir to put first on PATH]
  echo 'int main() { return 0; }' |
    PATH=${1:+$1:}$PATH "$CXX" -mcpu=neoverse-v2 -x c++ -c -o /dev/null - >/dev/null 2>&1
}
if [ -n "${BINUTILS_DIR:-}" ]; then
  export PATH=$BINUTILS_DIR:$PATH
elif ! can_assemble_armv9; then
  BINUTILS_DIR=
  for t in 15 14 13; do
    src=/opt/rh/gcc-toolset-$t/root/usr/bin
    [ -x "$src/as" ] || continue
    d=$(mktemp -d "${TMPDIR:-/tmp}/binutils-gcc-toolset-$t.XXXXXX") || break
    for tool in as ld ld.bfd objdump nm ar ranlib strip readelf objcopy; do
      [ -e "$src/$tool" ] && ln -s "$src/$tool" "$d/$tool"
    done
    if can_assemble_armv9 "$d"; then BINUTILS_DIR=$d; break; fi
    rm -r "$d"
  done
  if [ -n "$BINUTILS_DIR" ]; then
    export PATH=$BINUTILS_DIR:$PATH
  else
    echo "WARNING: $(as --version | head -1) cannot assemble -mcpu=neoverse-v2 and no newer" \
         "gcc-toolset binutils was found; the native builds will fail" >&2
  fi
fi
echo "assembler: $(command -v as): $(as --version | head -1)"
can_assemble_armv9 && echo "assembler: -mcpu=neoverse-v2 ok"

exec bash "${PBS_O_WORKDIR:-$PWD}/run_std_half_fp16.sh"
