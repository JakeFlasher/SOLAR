# RTX 3090 Ti GEMM cross-check (run: 2026-05-14T06:03:54Z)

Result status: `dense_found`

| Source | Precision triplet | BF16 best TFLOPS | TF32 best TFLOPS | INT8 best TOPS | BF16 median TFLOPS | TF32 median TFLOPS | INT8 median TOPS | Source artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spec peak | bf16/bf16 dense, tf32/tf32 dense, s8/s8 dense | 80.00 | 40.00 | 160.00 | 80.00 | 40.00 | 160.00 | [configs/arch/3090ti.yaml](../../configs/arch/3090ti.yaml) |
| cuBLASLt achieved | bf16/bf16 dense (TF32/INT8: not measured) | 38.82 | N/A (not measured) | N/A (not measured) | 38.82 | N/A (not measured) | N/A (not measured) | [cublaslt_bf16](/home/jakeshea/SOLAR/measurements/rtx3090/cublaslt/cublaslt_bf16_8192.csv) |
| CUTLASS dense | bf16/bf16/fp32_acc, tf32/tf32/fp32_acc, s8/s8/s32_acc | 84.40 | 38.70 | 243.53 | 66.49 | 32.20 | 64.33 | [cutlass_profiler_bf16](/home/jakeshea/SOLAR/measurements/rtx3090/cutlass/cutlass_profiler_bf16_8192.csv), [cutlass_profiler_tf32](/home/jakeshea/SOLAR/measurements/rtx3090/cutlass/cutlass_profiler_tf32_8192.csv), [cutlass_profiler_int8](/home/jakeshea/SOLAR/measurements/rtx3090/cutlass/cutlass_profiler_int8_8192.csv) |

## Notes

- The RTX 3090 Ti / GA102 (sm_86) has no FP8 tensor-core hardware (FP8 was introduced in Hopper sm_89/sm_90), no NVFP4 (Blackwell sm_100/sm_120), and therefore no blockscaled CUTLASS kernel families. The TF32 + INT8 columns substitute for the FP8 / NVFP4 columns that the 5060 SUMMARY.md reports.
- BF16 / TF32 are reported in TFLOPS (= GFLOPs / 1000); INT8 is reported in TOPS using the same formula (the CUTLASS profiler emits the integer-op count under the GFLOPs column for s8 GEMMs).
- TF32 storage on Ampere is FP32-shaped; the CUTLASS profiler still treats `Bytes` and `Flops` consistently, so the TFLOPS computation is comparable.
