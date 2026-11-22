# Native half arithmetic on the Grace (aarch64) machine

How to check whether GCC computes `std::float16_t` natively on Grace, and how to run the
complex/half tests with it. See `complex_half.md` §8.2 and §8.5 (row "aarch64 with
native half arithmetic").

**Background.** Grace's cores (Arm Neoverse V2) have the Arm FP16 extension: scalar
and SIMD instructions on 16-bit floats. GCC uses them only when it compiles for such a
CPU, e.g. `-mcpu=neoverse-v2`, which is what `Kokkos_ARCH_ARMV9_GRACE` adds. Without
it, GCC computes `_Float16` in `float`. That is what job 4260499 measured: Kokkos's
default target, `__FLT_EVAL_METHOD_TS_18661_3__` = 0. With it the value should be 16,
as with `-march=sapphirerapids` on Aurora.

Steps 1-3 take a few minutes and need no Kokkos build. Step 4 is the full test job.

## Prerequisites

- **Kokkos checkout:** `refactor-complex` with `run_std_half_fp16.sh` (aarch64 support)
  and `run_std_half_fp16_grace.sh`. These are added after `e90d6c937`; check with
  `grep -c ARMV9_GRACE run_std_half_fp16.sh` (expect 2 or more). The examples use the
  checkout of job 4260499, `/home/i18048/jh220036/develop/complex/kokkos`.
- **Modules:** `cmake/3.31.1 gcc/15.2.0`.
- **Where to run:** steps 1 and 3 must run **on a compute node**. The login node may be a
  different CPU, and a program built for Neoverse V2 crashes with "Illegal
  instruction" on a CPU without FP16. Use an interactive job, for example
  `qsub -I -l select=1 -q <interactive queue> -W group_list=jh220036`, or put the
  commands in a small batch job.

## 1. Does the CPU have FP16 arithmetic?

On a compute node:

```bash
lscpu | grep -E 'Model name|Vendor'
grep -m1 -o -w -E 'fphp|asimdhp' /proc/cpuinfo | sort -u
```

- **Expected:** model name `Neoverse-V2`, and both `fphp` (scalar half) and `asimdhp`
  (SIMD half).
- **If `fphp` is missing:** the CPU cannot compute in half natively. Stop here and mark
  the task as future work in `complex_half.md` §8.5.

## 2. What does GCC 15 do for this CPU?

On the login node is fine; this only compiles:

```bash
module purge
module load cmake/3.31.1 gcc/15.2.0
for flags in "" "-mcpu=neoverse-v2"; do
  echo "== g++ -std=c++23 $flags"
  echo | g++ -std=c++23 $flags -dM -E -x c++ - |
    grep -E '__FLT_EVAL_METHOD_TS_18661_3__|__ARM_FEATURE_FP16_SCALAR_ARITHMETIC|__STDCPP_B?FLOAT16_T__'
done
```

Expected:

| | no flag | `-mcpu=neoverse-v2` |
|---|---|---|
| `__FLT_EVAL_METHOD_TS_18661_3__` | 0 (half computed in float) | **16** (half computed in half) |
| `__ARM_FEATURE_FP16_SCALAR_ARITHMETIC` | not defined | 1 |
| `__STDCPP_FLOAT16_T__`, `__STDCPP_BFLOAT16_T__` | 1 | 1 |

## 3. A small program: results and instructions

Save as `fp16_check.cpp`:

```cpp
#include <complex>
#include <cstdio>
#include <stdfloat>

// Inputs through volatile, so that nothing is computed at compile time.
volatile float v256 = 256.f, v65000 = 65000.f, v269 = 269.f, v111 = 111.5f;

std::float16_t mul(std::float16_t a, std::float16_t b) { return a * b; }

int main() {
  using F = std::float16_t;
  const F a = static_cast<F>(v256), c = static_cast<F>(v65000);
  const F r = a * a - c;  // 65536 overflows half; in float: 65536 - 64992 = 544
  const std::complex<F> z(static_cast<F>(v269), static_cast<F>(v111));
  const std::complex<F> zz = z * z;  // 269^2 = 72361 overflows half
  std::printf("eval method %d: 256*256-65000 = %g, (269,111.5)^2 = (%g, %g)\n",
              __FLT_EVAL_METHOD_TS_18661_3__, static_cast<double>(r),
              static_cast<double>(zz.real()), static_cast<double>(zz.imag()));
}
```

Build and run on a compute node:

```bash
g++ -std=c++23 -O2 -o fp16_default fp16_check.cpp
g++ -std=c++23 -O2 -mcpu=neoverse-v2 -o fp16_native fp16_check.cpp
./fp16_default
./fp16_native
# The instructions of mul(): native half (fmul h0, h0, h1) or through float (fcvt).
for e in fp16_default fp16_native; do
  echo "== $e"; objdump -d --no-show-raw-insn "$e" | sed -n '/<_Z3mulDF16_S_>:/,/ret/p'
done
```

Expected:

| | `fp16_default` | `fp16_native` |
|---|---|---|
| output | `eval method 0: 256*256-65000 = 544, (269,111.5)^2 = (59936, 60000)` | `eval method 16: 256*256-65000 = inf, (269,111.5)^2 = (inf, 60000)` |
| `mul()` | `fcvt s…, h…` twice, `fmul s…`, `fcvt h…, s…` | `fmul h0, h0, h1` |

What it shows:

- **`fp16_native` prints method 16 and `inf`:** GCC computes in half. Then step 4 is worth
  running: it checks that Kokkos's results don't change, as on Sapphire Rapids.
- **`fp16_native` prints method 0, or `mul()` uses `fcvt`:** this GCC does not compute in
  half for Grace. Record that in `complex_half.md` §8.5; step 4 then adds little.
- **"Illegal instruction":** the program ran on a CPU without FP16, probably the login
  node. Run it on a compute node.

## 4. The test job

From the Kokkos checkout on Grace:

```bash
cd /home/i18048/jh220036/develop/complex/kokkos
qsub run_std_half_fp16_grace.sh
```

- **What it runs:** four builds, each Serial, Debug, C++23:
  - `default`: no `Kokkos_ARCH`, strict `-std=c++23`;
  - `native`: `Kokkos_ARCH_ARMV9_GRACE`, strict;
  - `gnu`: no `Kokkos_ARCH`, `-std=gnu++23`;
  - `gnu-native`: `Kokkos_ARCH_ARMV9_GRACE`, `-std=gnu++23`.

  For each build it prints the half macros, the instructions of one half multiplication
  and the probe results. Then it runs `*complex*:*atomic*`, `*numeric_traits*` and
  `*half*`.
- **Time:** the header gives 1 hour. With `-v 'ENABLE_OPENMP=ON,FULL=ON'` it needs more;
  check this machine's queue limits for runs over 1 hour.
- **Output:** the job's output file, also saved as `std_half_fp16_logs.<job id>/job.out`,
  with the build and test logs.

What to look for in `job.out`:

| | `default`, `gnu` | `native`, `gnu-native` |
|---|---|---|
| `fp16 cpu:` (top) | `yes (fphp in /proc/cpuinfo)` | the same |
| `__FLT_EVAL_METHOD_TS_18661_3__` | 0 | **16** |
| `__ARM_FEATURE_FP16_SCALAR_ARITHMETIC` | not listed | 1 |
| half multiplication instructions | native 0, through float > 0 | **native > 0**, through float 0 |
| `std_float16_t 256*256-65000` | 544 | **inf** |
| `std_complex (269,111.5)^2` | (59936, 60000) | **(inf, 60000)** |
| `kokkos_complex (269,111.5)^2` | (59936, 60000) | (59936, 60000) |
| build warnings | 0 | 0 |
| tests | 25 / 15 / 14 pass | 25 / 15 / 14 pass |
| summary | no `FAILED`, no `NOT RUN`, no `SKIPPED` | the same |

Then send me the log directory, or copy it to the Aurora/A100 side as before, and I'll
record it in `complex_half.md`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `fp16 cpu: no` and `NOT RUN (CPU lacks fphp)` for the native builds | The compute node has no FP16; check `lscpu` there (step 1). If so, the task becomes future work. |
| `ERROR: no GCC >= 13 found` or `module load ... failed` | The module names differ; check `module avail gcc cmake`, then edit the `module load` line of `run_std_half_fp16_grace.sh`. |
| `Kokkos did not select the std half backend` | The compiler does not define `__STDCPP_FLOAT16_T__` with `-std=c++23`; check step 2. |
| The native builds fail to configure | Kokkos's compiler check for `-mcpu=neoverse-v2` failed; see `std_half_fp16_logs.<id>/native_configure.log`. `ARMV9_GRACE` is the only Kokkos option that enables FP16 here: `ARMV84`, `ARMV84_SVE` and `A64FX` (`-march=armv8.2-a+sve`) do not add `+fp16`. Send me the log. |
| `Illegal instruction` in a native test | The job ran on a node without FP16; the script checks `/proc/cpuinfo`, so this means the flag is listed but the instructions are missing. Send me the log. |
| `tput: No value for $TERM` lines at the top | From the job shell's startup; harmless. |
