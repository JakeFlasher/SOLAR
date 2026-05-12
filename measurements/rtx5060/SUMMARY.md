# RTX 5060 Laptop GEMM cross-check (run: 2026-05-12T17:07:00Z)

Result status: `blockscaled_only`

| Source | Precision triplet | BF16 best TFLOPS | FP8 best TFLOPS | BF16 median TFLOPS | FP8 median TFLOPS | Source artifact |
| --- | --- | --- | --- | --- | --- | --- |
| Spec peak | bf16/bf16 dense, fp8/fp8 dense | 71.50 | 143.00 | 71.50 | 143.00 | [configs/arch/5060.yaml](../../configs/arch/5060.yaml) |
| cuBLASLt achieved | bf16/bf16 dense, fp8/fp8 dense | 26.30 | 101.28 | 26.30 | 101.28 | [cublaslt_bf16](/home/jakeshea/SOLAR/measurements/rtx5060/cublaslt/cublaslt_bf16_8192.csv), [cublaslt_fp8](/home/jakeshea/SOLAR/measurements/rtx5060/cublaslt/cublaslt_fp8_8192.csv) |
| CUTLASS dense | bf16/bf16/fp32_acc, fp8_e4m3/fp8_e4m3/fp32_acc | N/A — see [bf16 log](/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/cutlass_profiler_bf16_8192.log) | N/A — see [fp8 log](/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/cutlass_profiler_fp8_8192.log) | N/A — see [bf16 log](/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/cutlass_profiler_bf16_8192.log) | N/A — see [fp8 log](/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/cutlass_profiler_fp8_8192.log) | [cutlass_profiler_bf16](/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/cutlass_profiler_bf16_8192.csv), [cutlass_profiler_fp8](/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/cutlass_profiler_fp8_8192.csv) |
| CUTLASS blockscaled | nvfp4_in/bf16_out/fp32_acc, mxfp8_in/bf16_out/fp32_acc (representative) | 55.85 | 57.21 | 49.44 | 49.38 | [cutlass_profiler_bf16](/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/cutlass_profiler_bf16_8192.csv), [cutlass_profiler_fp8](/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/cutlass_profiler_fp8_8192.csv) |

## Notes

- The CUTLASS blockscaled row groups kernels whose input precision differs from output precision (e.g. NVFP4-in / BF16-out). Per the CUTLASS v4.4.1 sm_120 profiler-library coverage, the dense BF16-in / BF16-out variant is not present, so the blockscaled row is *not directly comparable to dense BF16*.
- TFLOPS values are computed from the profiler `GFLOPs` column divided by 1000.
