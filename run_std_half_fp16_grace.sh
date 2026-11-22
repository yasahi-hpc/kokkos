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

set +e
cd "${PBS_O_WORKDIR:-$PWD}" || exit 2

# A batch job has no terminal: without TERM the module command prints tput errors.
export TERM=${TERM:-dumb}
type module >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; }
module purge
module load cmake/3.31.1 gcc/15.2.0 || { echo "ERROR: module load cmake/3.31.1 gcc/15.2.0 failed" >&2; exit 2; }
export CXX=g++

exec bash "${PBS_O_WORKDIR:-$PWD}/run_std_half_fp16.sh"
