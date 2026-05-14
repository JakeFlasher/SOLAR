#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build and run the SOLAR rtx-class probe binaries (probe.cu, probe_fp16.cu,
# lt_gemm.cu) for the local RTX 3090 Ti (sm_86 / GA102), then drive
# Nsight Compute (`ncu`) over them to harvest DRAM / L2 / tensor-pipe
# peak metrics that triangulate the SUMMARY.md cross-check rows.
#
# This script is the sm_86 analog of the ad-hoc probe workflow that produced
# the rtx5060 mirror under solbench/solar-rtx5060/. Differences vs. that
# workflow:
#   * no FP8 path  -- 3090 Ti / sm_86 has no FP8 tensor-core hardware
#     (FP8 e4m3/e5m2 was introduced in Hopper sm_89/sm_90), so the FP8
#     cuBLASLt path in lt_gemm.cu is intentionally NOT invoked.
#   * no NVFP4 / MXFP8 path -- those are Blackwell-only.
#
# Outputs (TOPDIR is /home/jakeshea/solar-rtx3090 by default):
#   ${TOPDIR}/probe                     ELF
#   ${TOPDIR}/probe_fp16                ELF
#   ${TOPDIR}/lt_gemm                   ELF
#   ${TOPDIR}/props.txt                 cudaDeviceProp dump
#   ${TOPDIR}/nvidia-smi.txt            single-line nvidia-smi snapshot
#   ${TOPDIR}/cublaslt_bf16_8192.csv    lt_gemm bf16 8192 stdout
#   ${TOPDIR}/fp16_probe_app.csv        probe_fp16 stdout
#   ${TOPDIR}/ncu_dram.{csv,err}
#   ${TOPDIR}/ncu_l2.{csv,err}
#   ${TOPDIR}/ncu_hmma.{csv,err}        BF16 wmma path
#   ${TOPDIR}/ncu_hmma_fp16.{csv,err}
#   ${TOPDIR}/ncu_hmma_fp16_app.csv     companion app stdout (matches fp16_probe_app.csv)
#   ${TOPDIR}/ncu_cublaslt_bf16.{csv,err}
#   ${TOPDIR}/ncu_cublaslt_bf16_app.csv companion app stdout (1 timed iter)
#
# Mirrored into measurements/rtx3090/ for SUMMARY.md consumption:
#   ${MEASUREMENTS}/cublaslt/cublaslt_bf16_8192.csv
#   ${MEASUREMENTS}/wmma/ncu_hmma_fp16.csv
#   ${MEASUREMENTS}/wmma/ncu_hmma_fp16_app.csv
#
# Exit codes:
#   0  success
#   1  preflight failure (missing tool / GPU / source files)
#   2  build failure
#   3  probe / NCU run failure

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ARCH="${ARCH:-86}"
DIM="${DIM:-8192}"
ITERS="${ITERS:-20}"
NCU_ITERS="${NCU_ITERS:-1}"            # iters to pass to lt_gemm under NCU
FP16_PROBE_ITERS="${FP16_PROBE_ITERS:-20000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TOPDIR="${SOLAR_RTX3090_TOPDIR:-/home/jakeshea/solar-rtx3090}"
MEASUREMENTS="${SOLAR_MEASUREMENTS:-${SOLAR_ROOT}/measurements/rtx3090}"

# Source-of-truth probe source files. Live in the read-only solbench tree;
# we copy them into TOPDIR if they are not already there so the build (and
# any future edit) happens in a writable location.
PROBE_SRC_BASE="${PROBE_SRC_BASE:-/home/jakeshea/solbench/solar-rtx5060}"

# ncu binary -- explicit path so we are not at the mercy of $PATH order
NCU="${NCU:-/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RED=$'\033[0;31m'; C_GREEN=$'\033[0;32m'; C_YELLOW=$'\033[1;33m'
  C_BLUE=$'\033[0;34m'; C_NC=$'\033[0m'
else
  C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_NC=""
fi
log_section() { printf '\n%s==> %s%s\n' "${C_BLUE}" "$*" "${C_NC}"; }
log_info()    { printf '    %s\n' "$*"; }
log_ok()      { printf '%sOK%s: %s\n' "${C_GREEN}" "${C_NC}" "$*"; }
log_warn()    { printf '%sWARN%s: %s\n' "${C_YELLOW}" "${C_NC}" "$*" >&2; }
log_err()     { printf '%sERR%s: %s\n' "${C_RED}" "${C_NC}" "$*" >&2; }

die() { log_err "${1:-error}"; exit "${2:-1}"; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
preflight() {
  log_section "preflight"
  local missing=()
  for t in nvcc nvidia-smi; do
    if ! command -v "$t" >/dev/null 2>&1; then
      missing+=("$t")
      log_err "${t} not found on PATH"
    else
      log_info "found ${t}: $(command -v "$t")"
    fi
  done
  if [[ ! -x "$NCU" ]]; then
    missing+=("ncu (${NCU})")
    log_err "ncu not executable at ${NCU}"
  else
    log_info "found ncu: ${NCU}"
  fi
  if [[ ${#missing[@]} -gt 0 ]]; then
    die "preflight failed: missing tool(s): ${missing[*]}" 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    die "no CUDA GPU detected" 1
  fi
  local cc
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  if [[ -n "$cc" && "$cc" != "8.6" ]]; then
    log_warn "GPU compute capability ${cc} is not 8.6 (RTX 3090 Ti); script will still attempt sm_${ARCH} build"
  fi
  for s in probe.cu probe_fp16.cu lt_gemm.cu; do
    if [[ ! -f "${PROBE_SRC_BASE}/${s}" ]]; then
      die "probe source not found: ${PROBE_SRC_BASE}/${s}" 1
    fi
  done
  mkdir -p "${TOPDIR}" "${MEASUREMENTS}/cublaslt" "${MEASUREMENTS}/wmma"
  log_ok "preflight ok"
}

# ---------------------------------------------------------------------------
# Stage source files into the writable working dir
# ---------------------------------------------------------------------------
stage_sources() {
  log_section "stage-sources"
  for s in probe.cu probe_fp16.cu lt_gemm.cu; do
    if [[ ! -f "${TOPDIR}/${s}" ]]; then
      cp "${PROBE_SRC_BASE}/${s}" "${TOPDIR}/${s}"
      log_info "copied ${s} -> ${TOPDIR}/"
    else
      log_info "${TOPDIR}/${s} already present; reusing"
    fi
  done
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
build_probes() {
  log_section "build"
  local nvcc_args=(-O2 -arch="sm_${ARCH}" -std=c++17 -Xcompiler=-Wno-unused-variable)
  log_info "nvcc ${nvcc_args[*]} -o probe probe.cu -lcudart"
  ( cd "${TOPDIR}" && nvcc "${nvcc_args[@]}" -o probe probe.cu -lcudart ) \
    || die "build probe.cu failed" 2
  log_info "nvcc ${nvcc_args[*]} -o probe_fp16 probe_fp16.cu -lcudart"
  ( cd "${TOPDIR}" && nvcc "${nvcc_args[@]}" -o probe_fp16 probe_fp16.cu -lcudart ) \
    || die "build probe_fp16.cu failed" 2
  # lt_gemm.cu also has an FP8 codepath. It compiles fine on sm_86 (the FP8
  # types are header-only); we simply never invoke `lt_gemm fp8` at runtime
  # because the 3090 Ti has no FP8 tensor-core hardware.
  log_info "nvcc ${nvcc_args[*]} -o lt_gemm lt_gemm.cu -lcudart -lcublas -lcublasLt"
  ( cd "${TOPDIR}" && nvcc "${nvcc_args[@]}" -o lt_gemm lt_gemm.cu -lcudart -lcublas -lcublasLt ) \
    || die "build lt_gemm.cu failed" 2
  log_ok "build complete"
}

# ---------------------------------------------------------------------------
# Standalone (no-NCU) runs
# ---------------------------------------------------------------------------
run_standalone() {
  log_section "standalone"

  log_info "probe props -> props.txt"
  ( cd "${TOPDIR}" && ./probe props ) > "${TOPDIR}/props.txt" 2>/dev/null

  log_info "nvidia-smi snapshot -> nvidia-smi.txt"
  nvidia-smi --query-gpu=name,memory.total,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory,temperature.gpu,power.draw \
    --format=csv,noheader > "${TOPDIR}/nvidia-smi.txt" 2>/dev/null || \
    log_warn "nvidia-smi snapshot failed (non-fatal)"

  log_info "lt_gemm bf16 ${DIM} ${ITERS} -> cublaslt_bf16_8192.csv"
  ( cd "${TOPDIR}" && ./lt_gemm bf16 "${DIM}" "${ITERS}" ) \
    > "${TOPDIR}/cublaslt_bf16_${DIM}.csv" 2> "${TOPDIR}/lt_gemm_bf16.err" \
    || die "lt_gemm bf16 standalone failed (see lt_gemm_bf16.err)" 3

  log_info "probe_fp16 ${FP16_PROBE_ITERS} -> fp16_probe_app.csv"
  ( cd "${TOPDIR}" && ./probe_fp16 "${FP16_PROBE_ITERS}" ) \
    > "${TOPDIR}/fp16_probe_app.csv" 2>/dev/null \
    || die "probe_fp16 standalone failed" 3

  log_ok "standalone runs complete"
}

# ---------------------------------------------------------------------------
# NCU helper
#
# Pattern: app stdout -> /dev/null OR an _app.csv companion. NCU's CSV +
# preamble (==PROF==) -> the named .csv via redirect of NCU's stdout. App
# stderr (cuBLASLt heuristic chatter from lt_gemm.cu) -> .err. This matches
# the file set in solbench/solar-rtx5060/ exactly.
# ---------------------------------------------------------------------------
run_ncu_kernel() {
  local label="$1" metric_csv="$2" cmd_string="$3"
  local out_csv="${TOPDIR}/ncu_${label}.csv"
  local out_err="${TOPDIR}/ncu_${label}.err"
  log_info "ncu_${label}: ${cmd_string}"
  # `--log-file` directs NCU's own CSV + ==PROF== messages to ${out_csv};
  # app stdout is discarded so the CSV stays clean (matches the rtx5060
  # mirror's ncu_*.csv format). App stderr -> ${out_err} for separate debug.
  # shellcheck disable=SC2086  # cmd_string is intentionally word-split
  if "${NCU}" --csv --log-file "${out_csv}" --target-processes all \
        --metrics "${metric_csv}" \
        ${cmd_string} >/dev/null 2>"${out_err}"; then
    log_ok "ncu_${label} -> ${out_csv}"
  else
    log_warn "ncu_${label} exit non-zero -- partial CSV may still be present (see ${out_err})"
  fi
}

run_ncu_runs() {
  log_section "ncu"

  # 1. DRAM throughput (probe.cu dram_copy_kernel)
  run_ncu_kernel "dram" \
    "dram__bytes.sum,dram__bytes.sum.peak_sustained,dram__bytes.sum.peak_sustained.per_second,dram__bytes.sum.per_second,gpu__time_duration.sum" \
    "${TOPDIR}/probe dram"

  # 2. L2 throughput (probe.cu l2_kernel)
  run_ncu_kernel "l2" \
    "lts__t_bytes.sum,lts__t_bytes.sum.peak_sustained,lts__t_bytes.sum.peak_sustained.per_second,lts__t_bytes.sum.per_second,gpu__time_duration.sum" \
    "${TOPDIR}/probe l2"

  # 3. HMMA BF16 (probe.cu hmma_kernel; uses __nv_bfloat16 + WMMA fragments)
  run_ncu_kernel "hmma" \
    "sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_tensor.sum.peak_sustained,sm__inst_executed_pipe_tensor.sum.peak_sustained_active,sm__inst_executed_pipe_tensor.sum.per_cycle_active,smsp__inst_executed_pipe_tensor.sum.peak_sustained_active,gpu__time_duration.sum" \
    "${TOPDIR}/probe hmma"

  # 4. HMMA FP16 (probe_fp16.cu hmma_fp16 kernel)
  cp "${TOPDIR}/fp16_probe_app.csv" "${TOPDIR}/ncu_hmma_fp16_app.csv"
  run_ncu_kernel "hmma_fp16" \
    "sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_tensor.sum.peak_sustained,sm__inst_executed_pipe_tensor.sum.peak_sustained_active,sm__inst_executed_pipe_tensor.sum.per_cycle_active,gpu__time_duration.sum" \
    "${TOPDIR}/probe_fp16 5000"

  # 5. cuBLASLt BF16 (lt_gemm bf16 8192 1 ; iters=1 keeps NCU overhead bounded)
  ( cd "${TOPDIR}" && ./lt_gemm bf16 "${DIM}" "${NCU_ITERS}" ) \
    > "${TOPDIR}/ncu_cublaslt_bf16_app.csv" 2>/dev/null \
    || log_warn "lt_gemm app-companion (BF16) exited non-zero; companion CSV may be partial"
  run_ncu_kernel "cublaslt_bf16" \
    "dram__bytes.sum.per_second,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_tensor.sum.peak_sustained,sm__inst_executed_pipe_tensor.sum.peak_sustained_active,sm__inst_executed_pipe_tensor.sum.per_cycle_active,gpu__time_duration.sum" \
    "${TOPDIR}/lt_gemm bf16 ${DIM} ${NCU_ITERS}"
}

# ---------------------------------------------------------------------------
# Mirror the cross-source CSVs into measurements/rtx3090/{cublaslt,wmma}/
# ---------------------------------------------------------------------------
mirror_into_measurements() {
  log_section "mirror"
  cp "${TOPDIR}/cublaslt_bf16_${DIM}.csv" "${MEASUREMENTS}/cublaslt/"
  log_info "cublaslt -> ${MEASUREMENTS}/cublaslt/cublaslt_bf16_${DIM}.csv"
  cp "${TOPDIR}/ncu_hmma_fp16.csv" "${TOPDIR}/ncu_hmma_fp16_app.csv" \
     "${MEASUREMENTS}/wmma/" 2>/dev/null || \
    log_warn "wmma mirror partial (some ncu_hmma_fp16 files missing)"
  log_ok "mirror complete"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
  preflight
  stage_sources
  build_probes
  run_standalone
  run_ncu_runs
  mirror_into_measurements
  log_section "done"
  log_info "outputs under: ${TOPDIR}"
  log_info "mirror under:  ${MEASUREMENTS}"
}

main "$@"
