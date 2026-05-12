---
name: solar-arch-yaml-convention
description: "How configs/arch/*.yaml fields are defined and computed for SOLAR's roofline model"
metadata: 
  node_type: memory
  type: project
  originSessionId: e2dfbdb5-f953-4135-9aa1-2a2a6eeeebb8
---

`configs/arch/<arch>.yaml` files encode whole-chip peak rates relative to a chosen `freq_GHz` reference clock. The roofline math in `solar/perf/perf_model.py:176-225` is:

- `compute_cycles = total_macs / MAC_per_cycle`
- `mem_cycles    = total_bytes / DRAM_byte_per_cycle`
- `runtime_ms    = max(compute_cycles, mem_cycles) / (freq_GHz * 1e6)`

So `MAC_per_cycle` is **MACs per chip-cycle** (whole-chip, not per-SM), `DRAM_byte_per_cycle` is bytes per chip-cycle at the same reference, and `freq_GHz` converts cycles to seconds. Conversion identity: `TFLOPS = MAC_per_cycle * 2 * freq_GHz / 1000`.

**Why:** All three quantities must share the same reference clock or the model returns wrong runtimes. B200.yaml and H100_PCIe.yaml both use a sub-boost reference (1.5 / 2.0 GHz) and dense throughputs at that clock. 5060.yaml uses the NVIDIA-published boost of 2.497 GHz (because NVIDIA's 572 AI TOPS rating is computed at boost).

**How to apply:** For Blackwell-class GPUs, derive tensor MAC/cycle by halving NVIDIA's "AI TOPS" (sparse FP4 → dense FP4), then halve again for each precision step (FP4 → FP8 → FP16/BF16 → TF32), then `MAC = TFLOPS * 1000 / (2 * freq_GHz)`. `MAC_per_cycle_fp32_sm = SM_count * 128` (CUDA-core FP32 ALUs). All five fields `MAC_per_cycle_{fp32_sm,int8_tc,tf32_tc,fp16_tc,bf16_tc,fp8_tc,nvfp4_tc}` should be present; on Blackwell consumer (RTX 50) `nvfp4_tc` is supported. Related: [[rtx5060-probe-recipe]].
