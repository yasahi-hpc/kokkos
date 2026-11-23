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
- **Assembler (found 2026-11-23):** `gcc/15.2.0` uses the system binutils 2.35
  (`/usr/bin/as`), which rejects `-mcpu=neoverse-v2`: "Error: unknown architecture
  `armv9-a+rng+crc+i8mm+bf16+sve2-bitperm+memtag+profile'". Use binutils 2.41 from
  gcc-toolset-14, but **only its binutils**: prepending
  `/opt/rh/gcc-toolset-14/root/usr/bin` would also put its `g++` (GCC 14) ahead of
  GCC 15. A directory of symlinks does it (made once, on the shared file system):

  ```bash
  B=/home/i18048/jh220036/develop/complex/aarch64_checks/binutils241
  mkdir -p $B
  for t in as ld ld.bfd objdump nm ar ranlib strip readelf objcopy; do
    ln -sf /opt/rh/gcc-toolset-14/root/usr/bin/$t $B/$t
  done
  export PATH=$B:$PATH   # after `module load`; `as --version` should say 2.41
  ```
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

Build and run on a compute node, with the binutils 2.41 directory on `PATH`
(Prerequisites). `-ffp-contract=off` keeps `a * a - c` as two operations; without it
GCC fuses them at `-O2` (see the note below the table):

```bash
g++ -std=c++23 -O2 -ffp-contract=off -o fp16_default fp16_check.cpp
g++ -std=c++23 -O2 -ffp-contract=off -mcpu=neoverse-v2 -o fp16_native fp16_check.cpp
./fp16_default
./fp16_native
# The instructions of mul(): native half (fmul h0, h0, h1) or through float (fcvt).
# GCC 15 mangles mul(std::float16_t, std::float16_t) as _Z3mulDF16_DF16_.
for e in fp16_default fp16_native; do
  echo "== $e"; objdump -d --no-show-raw-insn "$e" | sed -n '/<_Z3mulDF16_DF16_>:/,/ret/p'
done
```

Expected:

| | `fp16_default` | `fp16_native` |
|---|---|---|
| output | `eval method 0: 256*256-65000 = 544, (269,111.5)^2 = (59936, 60000)` | `eval method 16: 256*256-65000 = inf, (269,111.5)^2 = (inf, 60000)` |
| `mul()` | `fcvt s…, h…` twice, `fmul s…`, `fcvt h…, s…` | `fmul h0, h0, h1` |

*Contraction (found 2026-11-23):* without `-ffp-contract=off`, `fp16_native` prints
`256*256-65000 = 544`. GCC's default for C++, also with `-std=c++23`, is
`-ffp-contract=fast`, and it compiles `a * a - c` to one half `fmadd`: the product
is exact, there is only one rounding, so nothing overflows. That is not float
evaluation: `mul()` is still `fmul h0, h0, h1`, and the complex square still overflows.
The Kokkos job (step 4) is not affected: its probes use the tests' Debug flags, which do
not contract.

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

The wrapper handles the assembler itself (added 2026-11-23, after job 4264666): if the
assembler on `PATH` cannot compile `-mcpu=neoverse-v2`, it makes a temporary directory
of symlinks to the binutils of the newest gcc-toolset (15, 14, 13) that can, and puts
it first on `PATH`. The top of the PBS output file (not `job.out`, which starts later)
shows the choice, e.g.
`assembler: …/binutils-gcc-toolset-14.XXXXXX/as: GNU assembler version 2.41-3.el9` and
`assembler: -mcpu=neoverse-v2 ok`. `qsub -v BINUTILS_DIR=<dir>` names a directory
instead. Job 4264666 predates this and was submitted with
`qsub -v "PATH=<binutils241 dir>:$PATH"`, which works too.

**Build directories are reused.** The builds go to `$BUILD_ROOT/build_std_half_<config>`
(default `BUILD_ROOT` = the checkout) and are built incrementally: a second job with
unchanged sources compiles nothing, so its "0 warnings" says nothing (job 4265289 did
this). For a clean build, pass an empty directory, e.g.
`qsub -v 'BUILD_ROOT=/home/i18048/jh220036/develop/complex/grace_fresh_build' …`;
the logs then go to `$BUILD_ROOT/std_half_fp16_logs.<job id>/`. Check the build logs:
a real build has `Building CXX object` lines, a reused one only `Built target`.

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

## Results (2026-11-23): all as expected

Recorded in `complex_half.md` §8.2 ("Grace run") and §8.5. Scripts and outputs of
steps 1-3: `/home/i18048/jh220036/develop/complex/aarch64_checks/`.

1. **CPU (job 4264402, `mg0837`):** Neoverse-V2, with `fphp` and `asimdhp`. The login
   node `miyabi-g3` is also a Neoverse-V2 with both flags.
2. **GCC 15.2.0 macros:** exactly as in the table: method 0 without a flag; 16 and
   `__ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1` with `-mcpu=neoverse-v2`.
3. **Small program:**
   - job 4264413, as first written: the native build failed in the assembler
     (binutils 2.35);
   - job 4264540, binutils 2.41, no `-ffp-contract=off`: default
     `eval method 0: 544, (59936, 60000)`, `mul()` = `fcvt`, `fcvt`, `fmul s0`, `fcvt`;
     native `eval method 16: 544, (inf, 60000)`, `mul()` = `fmul h0, h0, h1`. The 544
     is the half `fmadd`;
   - job 4264624, native with `-ffp-contract=off`:
     `eval method 16: inf, (inf, 60000)`, `fmul h0, h0, h1`.
4. **Test job 4264666** (`mg0639`, commit `48e69227b`, binutils 2.41 on `PATH`): every
   row of the table above holds:

   | | `default`, `gnu` | `native`, `gnu-native` |
   |---|---|---|
   | method | 0 | 16 (+ `__ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1`) |
   | half `*` instructions, native / through float | 0 / 4 | 2 / 0 |
   | `std_float16_t 256*256-65000` | 544 | inf |
   | `std_complex (269,111.5)^2` | (59936, 60000) | (inf, 60000) |
   | `kokkos_complex (269,111.5)^2` | (59936, 60000) | (59936, 60000) |
   | `(60000,60000)/(1,1)`, both | (60000, 0) | (60000, 0) |
   | warnings, 5 targets | 0 | 0 |
   | tests | 25 / 15 / 14 pass | 25 / 15 / 14 pass |

   Logs: `std_half_fp16_logs.4264666/` in the checkout. Every build compiled from
   scratch (157 files each).
5. **The wrapper's own binutils choice (job 4271100, commit `bfab679c6`, plain `qsub`,
   `CONFIGS=native`, empty `BUILD_ROOT`):** it chose gcc-toolset-14's binutils 2.41,
   compiled 157 files with 0 warnings, the probe values equal job 4264666's native
   column, and 25 / 15 / 14 tests pass. Logs:
   `/home/i18048/jh220036/develop/complex/grace_fresh_build/std_half_fp16_logs.4271100/`.
   (Job 4265289, the same without an empty `BUILD_ROOT`, reused job 4264666's build
   and compiled only the probes.)

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Error: unknown architecture 'armv9-a+…'` / `unrecognized option -march=armv9-a…` (native builds or `fp16_native`) | The system assembler (binutils 2.35) is too old for Neoverse V2. Steps 1-3: put binutils 2.41 first on `PATH` (Prerequisites). Step 4: the wrapper does it; if the job output has `WARNING: … cannot assemble -mcpu=neoverse-v2`, no gcc-toolset binutils was found: pass one with `qsub -v BINUTILS_DIR=<dir>`. |
| `objdump … sed` prints nothing for `mul()` | The symbol is `_Z3mulDF16_DF16_` with GCC 15; check with `nm fp16_default \| grep mul`. |
| `fp16_native` prints method 16 but `256*256-65000 = 544` | Floating-point contraction (`fmadd` in half); compile with `-ffp-contract=off` (step 3). |
| `fp16 cpu: no` and `NOT RUN (CPU lacks fphp)` for the native builds | The compute node has no FP16; check `lscpu` there (step 1). If so, the task becomes future work. |
| `ERROR: no GCC >= 13 found` or `module load ... failed` | The module names differ; check `module avail gcc cmake`, then edit the `module load` line of `run_std_half_fp16_grace.sh`. |
| `Kokkos did not select the std half backend` | The compiler does not define `__STDCPP_FLOAT16_T__` with `-std=c++23`; check step 2. |
| The native builds fail to configure | Kokkos's compiler check for `-mcpu=neoverse-v2` failed; see `std_half_fp16_logs.<id>/native_configure.log`. `ARMV9_GRACE` is the only Kokkos option that enables FP16 here: `ARMV84`, `ARMV84_SVE` and `A64FX` (`-march=armv8.2-a+sve`) do not add `+fp16`. Send me the log. |
| `Illegal instruction` in a native test | The job ran on a node without FP16; the script checks `/proc/cpuinfo`, so this means the flag is listed but the instructions are missing. Send me the log. |
| `tput: No value for $TERM` lines at the top | From the job shell's startup; harmless. |
