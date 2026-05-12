# SOLAR — Project context for Claude

This file consolidates the per-project memory that lives in `.claude/memory/`. Each section below mirrors one memory file; edit either source and re-sync when something changes.

---

## User's GPU (type: user)

Source: [.claude/memory/user_gpu.md](.claude/memory/user_gpu.md)

The user develops on a laptop with `NVIDIA GeForce RTX 5060 Laptop GPU` (Blackwell, sm_120, GB206, 26 SMs, 8 GB GDDR7, 384 GB/s, 32 MB L2, boost 1455-2497 MHz, 100 W cap; 115 W max-limit observed). System has driver 595.71.05, CUDA 13.2, nvcc at `/opt/cuda/bin/nvcc`, Nsight Compute 2026.1.1. PyTorch is NOT in system Python; the project uses `uv` (≥ 0.11) with `.venv` created by `install_uv.sh`.

When the user asks about benchmarking, profiling, or hardware specs, default to using this GPU. When citing peak TFLOPS, use **572 AI TOPS sparse FP4 = 286 dense FP4 = 143 dense FP8/INT8 = 71.5 dense FP16/BF16 = 35.75 dense TF32** (all at 2.497 GHz boost).

---

## SOLAR arch YAML convention (type: project)

Source: [.claude/memory/solar_arch_yaml_convention.md](.claude/memory/solar_arch_yaml_convention.md)

`configs/arch/<arch>.yaml` files encode whole-chip peak rates relative to a chosen `freq_GHz` reference clock. The roofline math in `solar/perf/perf_model.py:176-225` is:

- `compute_cycles = total_macs / MAC_per_cycle`
- `mem_cycles    = total_bytes / DRAM_byte_per_cycle`
- `runtime_ms    = max(compute_cycles, mem_cycles) / (freq_GHz * 1e6)`

So `MAC_per_cycle` is **MACs per chip-cycle** (whole-chip, not per-SM), `DRAM_byte_per_cycle` is bytes per chip-cycle at the same reference, and `freq_GHz` converts cycles to seconds. Conversion identity: `TFLOPS = MAC_per_cycle * 2 * freq_GHz / 1000`.

**Why:** All three quantities must share the same reference clock or the model returns wrong runtimes. `B200.yaml` and `H100_PCIe.yaml` both use a sub-boost reference (1.5 / 2.0 GHz) and dense throughputs at that clock. `5060.yaml` uses the NVIDIA-published boost of 2.497 GHz (because NVIDIA's 572 AI TOPS rating is computed at boost).

**How to apply:** For Blackwell-class GPUs, derive tensor MAC/cycle by halving NVIDIA's "AI TOPS" (sparse FP4 → dense FP4), then halve again for each precision step (FP4 → FP8 → FP16/BF16 → TF32), then `MAC = TFLOPS * 1000 / (2 * freq_GHz)`. `MAC_per_cycle_fp32_sm = SM_count * 128` (CUDA-core FP32 ALUs). All seven fields `MAC_per_cycle_{fp32_sm,int8_tc,tf32_tc,fp16_tc,bf16_tc,fp8_tc,nvfp4_tc}` should be present; on Blackwell consumer (RTX 50) `nvfp4_tc` is supported.

---

## RTX 5060 probe recipe (type: reference)

Source: [.claude/memory/rtx5060_probe_recipe.md](.claude/memory/rtx5060_probe_recipe.md)

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

A minimal `probe.cu` doing `cudaGetDeviceProperties` + `cudaDeviceGetAttribute` lives at `/tmp/solar-rtx5060/probe.cu` (ephemeral); recompile with `nvcc -O3 -std=c++17 -arch=sm_120`. NCU `--set full` is overkill (multi-pass, >60 s); use hand-picked metrics with `--launch-skip` and `--launch-count` for ≤2 s captures.

NVIDIA RTX 50-series "AI TOPS" rating is **sparse FP4**, computed at boost clock. Divide by 2 for dense FP4, then halve per precision step for FP8 / FP16 / TF32. Source: <https://www.nvidia.com/en-us/geforce/laptops/50-series/>

**Empirical cross-check (cuBLASLt 13.4 + NCU on sm_120, May 2026):**

- FP8 e4m3 8192³ GEMM picks `nvjet_sm120_qqtsq_mma_*_tmaAB_*` (Blackwell-native, uses TMA) and hits **per_cycle_active 6.24 / peak 6.50 = 96 % utilization → 106 TFLOPS achieved** (74 % of 143 spec).
- BF16 8192³ GEMM picks `cutlass_80_tensorop_bf16_s16816gemm_*` (Ampere-era fallback, no sm_120 BF16 kernel in cuBLAS 13.4 yet) → 50 % utilization → only **28 TFLOPS achieved** (39 % of 71.5 spec). The yaml's 71.5 TFLOPS BF16 spec is still correct — the gap is a *library tuning* gap, not a hardware peak gap. This will likely close as cuBLAS adds sm_120-native BF16 kernels.
- FP16 wmma microbench confirms `sm__inst_executed_pipe_tensor.sum.peak_sustained = 6.50 inst/cycle` (identical to BF16) → 5th-gen TC has FP16 == BF16 throughput.
