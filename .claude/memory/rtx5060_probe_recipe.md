---
name: rtx5060-probe-recipe
description: Working recipe for probing GPU specs on CUDA 13.2 + Nsight Compute 2026; avoids common hallucinations
metadata: 
  node_type: memory
  type: reference
  originSessionId: e2dfbdb5-f953-4135-9aa1-2a2a6eeeebb8
---

To populate a SOLAR `configs/arch/<gpu>.yaml`, **do not** rely on these (hallucinated or broken on this env):
- `nvidia-smi --query-gpu=memory.bus_width` — not a valid NVML field; use a compiled CUDA probe instead.
- `cudaDeviceProp.clockRate` / `cudaDeviceProp.memoryClockRate` — **removed in CUDA 13.x headers**. Use `cudaDeviceGetAttribute(&val, cudaDevAttrClockRate, 0)` and `cudaDevAttrMemoryClockRate` / `cudaDevAttrGlobalMemoryBusWidth` / `cudaDevAttrL2CacheSize`.
- `torch.cuda.get_device_properties()` — torch is not in system Python here.
- `ncu --print-metric-name name` — only supported with the *details* page, fails for `--page raw`.

**Working ncu metric suffix chains** (verified on sm_120, ncu 2026.1.1):
- `dram__bytes.sum.peak_sustained` (byte/mem-cycle) — multiply by memory clock for peak GB/s
- `dram__bytes.sum.per_second` (byte/s achieved)
- `lts__t_bytes.sum.peak_sustained` (byte/SM-cycle) — peak L2 throughput
- `sm__inst_executed_pipe_tensor.sum.peak_sustained` (inst/cycle peak tensor pipe)
- `gpu__time_duration.sum`

A minimal `probe.cu` doing `cudaGetDeviceProperties` + `cudaDeviceGetAttribute` lives at `/tmp/solar-rtx5060/probe.cu` (ephemeral); recompile with `nvcc -O3 -std=c++17 -arch=sm_120`. NCU `--set full` is overkill (multi-pass, >60s); use hand-picked metrics with `--launch-skip` and `--launch-count` for ≤2 s captures.

NVIDIA RTX 50-series "AI TOPS" rating is **sparse FP4**, computed at boost clock. Divide by 2 for dense FP4, then halve per precision step for FP8 / FP16 / TF32. Source: https://www.nvidia.com/en-us/geforce/laptops/50-series/

**Empirical cross-check (cuBLASLt 13.4 + NCU on sm_120, May 2026):**
- FP8 e4m3 8192³ GEMM picks `nvjet_sm120_qqtsq_mma_*_tmaAB_*` (Blackwell-native, uses TMA) and hits **per_cycle_active 6.24 / peak 6.50 = 96% utilization → 106 TFLOPS achieved (74% of 143 spec).
- BF16 8192³ GEMM picks `cutlass_80_tensorop_bf16_s16816gemm_*` (Ampere-era fallback, no sm_120 BF16 kernel in cuBLAS 13.4 yet) → 50% utilization → only 28 TFLOPS achieved (39% of 71.5 spec). The yaml's 71.5 TFLOPS BF16 spec is still correct — the gap is a *library tuning* gap, not a hardware peak gap. This will likely close as cuBLAS adds sm_120-native BF16 kernels.
- FP16 wmma microbench confirms `sm__inst_executed_pipe_tensor.sum.peak_sustained = 6.50 inst/cycle` (identical to BF16) → 5th-gen TC has FP16 == BF16 throughput.

## CUTLASS v4.4.1 cross-check (run: 2026-05-12T17:07:00Z)

- Kernel filter BF16: `*sm120*bf16*` matched 98 kernel(s); profiled: best 55.85 TFLOPS, median 49.44 TFLOPS (n=48)
- Kernel filter FP8: `*sm120*e4m3*` matched 98 kernel(s); profiled: best 57.21 TFLOPS, median 49.38 TFLOPS (n=48)
- Result status: `blockscaled_only`
- Manifest: `/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/manifest.json`
