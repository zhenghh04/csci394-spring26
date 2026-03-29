# GPU Assignments

This will be a **group project**.

- form groups of `3-4` students per group
- submit **one submission per group**
- choose **one** of the following three projects
- in your submission, clearly write down each group member's participation
- be specific about who implemented, debugged, ran experiments, analyzed results,
  and wrote the report

Participation requirement:

- include a short contribution statement for each group member
- do not write only "everyone did everything"
- list concrete responsibilities and contributions for each person

Project choices:

- `01_gpu_offload_comparison/` - compare dense matrix multiplication across CPU, OpenACC, OpenMP target offload, and CUDA, and analyze transfer overheads
- `02_cublas_gemm_flops/` - use cuBLAS to measure standard FP32 GEMM throughput and tensor core GEMM throughput, and document any AI-generated starting point carefully
- `03_intel_xpu/` - use ChatGPT or AskALCF to generate Intel XPU SYCL code for Aurora, then measure standard FP32 GEMM throughput and XMX systolic-array throughput
