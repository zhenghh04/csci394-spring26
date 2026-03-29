# GPU Assignment: Intel XPU FLOPs on Aurora with AI-Generated Code

## Objective

Use ChatGPT or AskALCF ChatBot (https://ask.alcf.anl.gov) to generate a starting-point Intel XPU program for
Aurora, then inspect, correct, run, and analyze it yourself.

You will measure two kinds of matrix-multiplication performance on Intel GPUs:

1. standard FP32 throughput
2. systolic-array throughput using Intel XMX

The central questions are:

- how much FP32 GEMM performance can you measure on Aurora?
- how much higher is the effective throughput when the XMX matrix engine is used?
- what code corrections were still required after AI code generation?

## Brief Background: SYCL, oneAPI, and XMX

Before starting, you should know the following terms:

- **SYCL**
  - a modern C++ programming model for heterogeneous systems
  - lets you write host code and device code in a mostly standard C++ style
  - commonly uses objects such as queues, buffers, accessors, and kernels
  - on Aurora, SYCL is the main programming model for Intel GPUs
- **oneAPI**
  - Intel's software stack for CPUs, GPUs, and other accelerators
  - includes the SYCL compiler toolchain and performance libraries
  - on Aurora, this usually means tools such as `icpx`, SYCL runtime support,
    and libraries such as oneMKL
- **Intel XPU**
  - a general Intel term for accelerator devices such as Intel GPUs
  - in this assignment, it refers to the Intel GPU on Aurora
- **Intel XMX**
  - Intel's matrix engine hardware for fast matrix multiply-accumulate
  - conceptually similar to NVIDIA tensor cores
  - usually accessed in SYCL through the `joint_matrix` extension

Practical interpretation for this assignment:

1. use **SYCL / oneAPI** as the programming environment
2. use a normal FP32 GEMM path as the baseline
3. use an XMX / `joint_matrix` path as the matrix-engine measurement

You do not need to become a SYCL expert before starting, but you should be able
to recognize the following basic pattern in generated code:

```cpp
sycl::queue q{sycl::gpu_selector_v};
q.submit([&](sycl::handler& h) {
    h.parallel_for(..., [=](sycl::id<1> i) {
        // device work
    });
});
q.wait();
```

At a high level:

- the **queue** chooses where work runs
- `submit(...)` launches work to the device
- `parallel_for(...)` defines the device kernel
- `wait()` ensures the work finishes before timing or reading results

For GEMM, you may either:

1. call a library implementation such as oneMKL GEMM
2. write a SYCL kernel directly
3. use `joint_matrix` to access XMX for tiled matrix multiply

The point of this assignment is not just to make the code compile. The point is
to understand what toolchain the AI is using, what hardware path it is targeting,
and whether the timing and accuracy claims are actually valid.

## Technical Scope

Your code should target **Aurora** and **Intel XPU** using **SYCL / oneAPI**.

You must implement or generate two GPU paths:

1. **FP32 GEMM path**
   - multiply square matrices with FP32 inputs and FP32 output
   - acceptable approaches:
     - oneMKL GEMM
     - a SYCL kernel if it is correct and reasonably efficient
2. **Systolic-array GEMM path**
   - use Intel XMX through the SYCL `joint_matrix` extension
   - recommended mixed-precision choice:
     - BF16 inputs
     - FP32 accumulation/output

Recommended interpretation:

- treat the FP32 path as the conventional floating-point baseline
- treat the XMX path as the systolic-array measurement
- report the XMX result as **effective FLOP/s** using the usual GEMM operation
  count `2*n^3 / time`, while clearly stating the operand precision used

Important note:

- Intel documents `joint_matrix` as the interface for XMX programming, and notes
  that it requires XMX-capable hardware with no fallback emulation path.

## Required Program Features

Create your own code in this assignment folder.

Minimum requirement:

1. accept matrix size `n` from the command line
2. accept measured iterations `iters`
3. accept warmup iterations `warmup`
4. initialize matrices with a reproducible pattern
5. run one FP32 GEMM path on the Intel GPU
6. run one XMX / `joint_matrix` GEMM path on the Intel GPU
7. time the GEMM region for both paths
8. compute and print FLOP/s or GFLOP/s
9. perform a correctness check
10. print parseable output

Recommended output columns:

- `mode,n,iters,warmup,time_s,gflops,max_abs_err`

Recommended `mode` values:

- `fp32`
- `xmx_bf16_fp32`

## Correctness Requirement

You must compare against a trusted reference.

Acceptable choices:

1. CPU FP32 GEMM for small and medium sizes
2. oneMKL FP32 GEMM used as the reference when validating the XMX path

Minimum requirement:

- verify correctness for at least small and medium matrix sizes
- report `max_abs_err`
- clearly state whether the XMX path is mixed precision

Because the XMX path is typically BF16 input with FP32 accumulation, you should
expect a larger numerical error than the pure FP32 path. That is acceptable if
you explain it clearly.

## Performance Metric

For square GEMM `C = A * B` with `n x n` matrices, use:

- operations = `2 * n^3`

If average kernel time is `t` seconds, report:

- `FLOP/s = 2 * n^3 / t`
- `GFLOP/s = (2 * n^3) / (t * 1e9)`

Minimum requirement:

- report kernel-only timing for both paths

Recommended:

- also report end-to-end timing
- clearly separate setup, transfer, and GEMM timing if you measure them

## AI Requirement

Use **ChatGPT** or **AskALCF** to produce the initial code draft.

Minimum requirement:

1. save the prompt or prompts you used
2. save the first generated code version
3. describe what was wrong, missing, or misleading in that generated code
4. describe what you changed before the program built and ran correctly

Your report must answer:

1. Did the AI generate valid SYCL / oneAPI code immediately?
2. Did it use the correct GEMM API or `joint_matrix` API?
3. Did it handle timing correctly?
4. Did it use a valid Aurora build command?
5. What manual debugging was still required?

Do not treat AI-generated code as automatically correct.

## Suggested Prompt Ideas

You may adapt one of these prompts.

### Prompt A: FP32 baseline

```text
Write a SYCL C++ program for Aurora using Intel oneAPI that measures FP32 GEMM
performance on Intel XPU. The program should accept n, iters, and warmup from
the command line, allocate square matrices, initialize them reproducibly, run a
warmup phase, time the GEMM region only, compute GFLOP/s using 2*n^3/time,
check correctness against a CPU reference for small sizes, and print parseable
CSV-style output. Also provide a Makefile using icpx -fsycl.
```

### Prompt B: XMX / systolic array

```text
Write a SYCL C++ program for Aurora using Intel oneAPI and the joint_matrix
extension to run GEMM on Intel XMX. Use BF16 inputs and FP32 accumulation,
accept n, iters, and warmup from the command line, time the joint_matrix GEMM
region only, compute effective GFLOP/s using 2*n^3/time, compare against an
FP32 reference for correctness, and print parseable CSV-style output. Also
provide a Makefile using icpx -fsycl.
```

## Experimental Plan

Run both paths on Aurora on the same GPU type.

Recommended sweep:

- `n = 256, 512, 1024, 2048, 4096, 8192`

If memory or runtime is limiting, use a smaller sweep and state that clearly.

For each size:

1. run at least 3 measured repeats
2. record average kernel time
3. compute average GFLOP/s
4. record correctness error

## Aurora Build and Run Requirement

Create your own build and run instructions in this assignment folder.

Minimum requirement:

1. a `Makefile`
2. a build command using `icpx -fsycl`
3. a note explaining how you set up the Aurora environment
4. one Aurora batch script or one documented interactive launch procedure

Suggested Aurora reminders:

1. Aurora provides the Intel oneAPI programming environment
2. SYCL compilation uses `icpx -fsycl`
3. if you use `joint_matrix`, verify the device supports the required matrix
   extension before claiming XMX results

Because site configurations can change, document the exact commands that worked
for you on the date you ran them.

## Suggested Commands

These are examples and may need to be adapted on Aurora.

```bash
make
./app_fp32 1024 5 1
./app_xmx 1024 5 1
./app_fp32 4096 3 1
./app_xmx 4096 3 1
```

If you build one combined executable, that is also acceptable, for example:

```bash
./app fp32 4096 5 1
./app xmx 4096 5 1
```

## What to Analyze

Your report should answer:

1. How does FP32 GEMM throughput change with matrix size?
2. How does XMX throughput change with matrix size?
3. How large is the gap between the FP32 path and the XMX path?
4. Is the XMX result truly FP32 arithmetic, or mixed precision with FP32
   accumulation?
5. What numerical accuracy tradeoff did you observe?
6. What mistakes did the AI tool make in the initial code?
7. What did AskALCF or ChatGPT help with, and what did it not do reliably?

## Deliverables

Submit one folder containing:

1. Source code
   - FP32 program or mode
   - XMX / `joint_matrix` program or mode
   - Makefile
   - Aurora run script or documented launch commands
2. AI artifacts
   - prompt text
   - first generated code version
   - short notes on fixes you made
3. Results data
   - raw logs
   - one CSV file with at least these columns:
     - `mode,n,iters,warmup,time_s,gflops,max_abs_err`
4. Report (`report_intel_xpu_flops.pdf`, 1-3 pages) including:
   - hardware used
   - compiler and flags
   - whether the XMX path used BF16, FP16, or another input type
   - one plot of GFLOP/s vs matrix size
   - one short accuracy discussion
   - one short discussion of AI-generated code quality

## Grading Focus

1. Correct use of Intel XPU programming tools.
2. Correct FLOP/s calculation.
3. Clear distinction between FP32 baseline and XMX measurement.
4. Honest discussion of mixed precision and accuracy.
5. Honest and specific discussion of the AI-generated starting point.
6. Clear Aurora build and run documentation.

## Notes

1. Do not fabricate performance results.
2. Clearly state the exact Aurora environment and commands you used.
3. If the XMX path does not work, still submit the FP32 path, the attempted XMX
   code, the prompts, the errors you hit, and a short explanation.

## References

Use the official Aurora, SYCL, and Intel oneAPI documentation as the primary
reference for this assignment.

1. Aurora SYCL programming guide
   - https://docs.alcf.anl.gov/aurora/programming-models/sycl-aurora/
2. SYCL 2020 specification
   - https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
3. Intel oneMKL GEMM reference
   - https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/gemm.html
4. Intel article on the unified SYCL joint matrix extension
   - https://www.intel.com/content/www/us/en/developer/articles/technical/ixpug-sycl-joint-matrix.html
5. Intel oneAPI documentation portal
   - https://www.intel.com/content/www/us/en/developer/tools/oneapi/documentation.html
