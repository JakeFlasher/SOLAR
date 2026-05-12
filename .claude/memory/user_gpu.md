---
name: user-gpu
description: "User's local development GPU is an RTX 5060 Laptop (Blackwell sm_120), CUDA 13.2, ncu 2026.1.1"
metadata: 
  node_type: memory
  type: user
  originSessionId: e2dfbdb5-f953-4135-9aa1-2a2a6eeeebb8
---

The user develops on a laptop with `NVIDIA GeForce RTX 5060 Laptop GPU` (Blackwell, sm_120, GB206, 26 SMs, 8 GB GDDR7, 384 GB/s, 32 MB L2, boost 1455-2497 MHz, 100W cap; 115W max-limit observed). System has driver 595.71.05, CUDA 13.2, nvcc at `/opt/cuda/bin/nvcc`, Nsight Compute 2026.1.1. PyTorch is NOT in system Python; the project uses `uv` (≥ 0.11) with `.venv` created by `install_uv.sh`.

When the user asks about benchmarking, profiling, or hardware specs, default to using this GPU. When citing peak TFLOPS, use 572 AI TOPS sparse FP4 = 286 dense FP4 = 143 dense FP8/INT8 = 71.5 dense FP16/BF16 = 35.75 dense TF32 (all at 2.497 GHz boost). Related: [[solar-arch-yaml-convention]].
