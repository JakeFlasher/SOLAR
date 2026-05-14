#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the CUTLASS v4.4.1 cutlass_profiler with sm_80 (Ampere datacenter)
# kernel filters appropriate for sm_86 codegen, run BF16 / TF32 / INT8 GEMM
# benchmarks at 8192^3 on the local RTX 3090 Ti, and emit a triangulated
# SUMMARY.md against the spec peak in configs/arch/3090ti.yaml and the
# probe-driven cuBLASLt BF16 datapoint.
#
# This is the sm_86 / Ampere analog of bench_cutlass_5060.sh. Differences:
#
#   * three precisions instead of two: BF16, TF32, INT8 (not FP8). The
#     RTX 3090 Ti / GA102 has no FP8 tensor-core hardware (FP8 was
#     introduced in Hopper sm_89/sm_90), and likewise no NVFP4/MXFP8/MXFP4
#     (Blackwell-only). Substituting TF32 + INT8 keeps the precision sweep
#     meaningful on Ampere.
#   * sm_86 picks up the sm_80 CUTLASS kernel libraries (sm_86 does not have
#     its own CUTLASS instance lineage; tensor-core paths reuse sm_80).
#   * no blockscaled row in SUMMARY.md (no blockscaled kernels on sm_80).
#
# Outputs (default):
#   measurements/rtx3090/SUMMARY.md
#   measurements/rtx3090/cutlass/cutlass_profiler_bf16_8192.csv  + .log
#   measurements/rtx3090/cutlass/cutlass_profiler_tf32_8192.csv  + .log
#   measurements/rtx3090/cutlass/cutlass_profiler_int8_8192.csv  + .log
#   measurements/rtx3090/cutlass/dryrun_kernels_{bf16,tf32,int8}.txt
#   measurements/rtx3090/cutlass/manifest.json
#
# Exit codes: 0 success / 1 preflight / 2 zero-match (without --allow-empty-kernels)
# / 3 build / 4 profile / 5 invalid CLI argument

set -euo pipefail

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Prefer system python (avoid the conda interpreter that may be active).
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python3"

# ---------------------------------------------------------------------------
# Color helpers (mirrored from bench_cutlass_5060.sh)
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
# Defaults
# ---------------------------------------------------------------------------
ARCH="86"
M=8192; N=8192; K=8192
WARMUP_ITERATIONS=10
PROFILING_ITERATIONS=20
OUTPUT_DIR="${SOLAR_ROOT}/measurements/rtx3090/cutlass"
CUTLASS_DIR="${SOLAR_ROOT}/cutlass"
BUILD_DIR=""
# sm_80/sm_86 CUTLASS profiler library uses 2.x kernel naming
# `cutlass_tensorop_<input>_<inst>gemm_<output>_<tile>_<stages>_<layout>_align<N>`
# (no `sm{N}` token unlike 3.x kernels, which start at sm_90).
# These narrow filters yield ~138 kernels total: 60 BF16 + 56 TF32 + 22 INT8.
KERNEL_FILTER_BF16='cutlass_tensorop_bf16_s16816gemm_bf16_*_align8'
KERNEL_FILTER_TF32='cutlass_tensorop_s1688gemm_tf32_*_align4'
KERNEL_FILTER_INT8='cutlass_tensorop_i8816gemm_s8_*_align16'
JOBS=8
SKIP_BUILD=false
DO_CLEAN=false
ALLOW_EMPTY_KERNELS=false
DRY_RUN=false
SUMMARY_STATISTIC="both"
DO_SMOKE=false
NO_SUMMARY=false

# Internal state populated as the run progresses
SUMMARY_DIR=""
GENERATED_KERNELS_TXT=""
BUILD_GENERATOR="make"
BUILD_EXIT_CODE=-1
RESULT_STATUS="unknown"
RUN_TIMESTAMP=""

# Per-precision arrays (associative-ish: bf16/tf32/int8 are the keys via array index)
PRECISIONS=(bf16 tf32 int8)
declare -A FILTER N_MATCHED DRYRUN_NAMES PROFILE_EXIT PRE_STATE POST_STATE
FILTER[bf16]="${KERNEL_FILTER_BF16}"
FILTER[tf32]="${KERNEL_FILTER_TF32}"
FILTER[int8]="${KERNEL_FILTER_INT8}"
for p in "${PRECISIONS[@]}"; do
  N_MATCHED[$p]=0
  DRYRUN_NAMES[$p]=""
  PROFILE_EXIT[$p]=-1
  PRE_STATE[$p]='{"sm_clock_mhz":"unknown","mem_clock_mhz":"unknown","temp_c":"unknown","power_w":"unknown"}'
  POST_STATE[$p]='{"sm_clock_mhz":"unknown","mem_clock_mhz":"unknown","temp_c":"unknown","power_w":"unknown"}'
done

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
print_help() {
  cat <<EOF
Usage: bench_cutlass_3090.sh [OPTIONS]

Build the CUTLASS v4.4.1 cutlass_profiler with sm_86-targeting kernel filters
(sm_80 GEMM library) and run BF16 + TF32 + INT8 8192^3 GEMM benchmarks on the
local RTX 3090 Ti.

Options:
  --arch=<str>            CUTLASS_NVCC_ARCHS value [86]
  --m=<int>, --n=<int>, --k=<int>, --matrix-size=<int>
                          GEMM dimensions [8192]
  --warmup-iterations=<int>      Profiler warmup [10]
  --profiling-iterations=<int>   Profiler timed iterations [20]
  --output-dir=<path>     [<solar>/measurements/rtx3090/cutlass]
  --cutlass-dir=<path>    [<solar>/cutlass]
  --build-dir=<path>      [<cutlass-dir>/build-sm86-profiler]
  --kernel-filter-bf16=<glob>  ['*sm80*bf16*']
  --kernel-filter-tf32=<glob>  ['*sm80*tf32*']
  --kernel-filter-int8=<glob>  ['*sm80*s8*']
  --jobs=<int>            cmake parallel build jobs [8]
  --skip-build            Reuse existing cutlass_profiler binary
  --clean                 rm -rf build-dir before configure
  --allow-empty-kernels   Don't abort if any precision matches 0 kernels
  --dry-run               Print would-run commands; exit 0
  --summary-statistic=<best|median|both>  [both]
  --smoke                 Pre-run 1024^3 smoke before 8192^3 timed runs
  --no-summary            Skip SUMMARY.md emission
  -h, --help

EOF
}

require_positive_int() {
  if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    die "invalid value for ${1}: '${2}' (expected positive integer)" 5
  fi
}
require_nonneg_int() {
  if [[ ! "$2" =~ ^(0|[1-9][0-9]*)$ ]]; then
    die "invalid value for ${1}: '${2}' (expected non-negative integer)" 5
  fi
}

parse_flags() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) print_help; exit 0 ;;
      --arch=*) ARCH="${1#--arch=}"; shift ;;
      --arch) ARCH="$2"; shift 2 ;;
      --m=*) M="${1#--m=}"; require_positive_int --m "$M"; shift ;;
      --m) M="$2"; require_positive_int --m "$M"; shift 2 ;;
      --n=*) N="${1#--n=}"; require_positive_int --n "$N"; shift ;;
      --n) N="$2"; require_positive_int --n "$N"; shift 2 ;;
      --k=*) K="${1#--k=}"; require_positive_int --k "$K"; shift ;;
      --k) K="$2"; require_positive_int --k "$K"; shift 2 ;;
      --matrix-size=*) v="${1#--matrix-size=}"; require_positive_int --matrix-size "$v"; M="$v"; N="$v"; K="$v"; shift ;;
      --matrix-size) require_positive_int --matrix-size "$2"; M="$2"; N="$2"; K="$2"; shift 2 ;;
      --warmup-iterations=*) WARMUP_ITERATIONS="${1#--warmup-iterations=}"; require_nonneg_int --warmup-iterations "$WARMUP_ITERATIONS"; shift ;;
      --warmup-iterations) WARMUP_ITERATIONS="$2"; require_nonneg_int --warmup-iterations "$WARMUP_ITERATIONS"; shift 2 ;;
      --profiling-iterations=*) PROFILING_ITERATIONS="${1#--profiling-iterations=}"; require_positive_int --profiling-iterations "$PROFILING_ITERATIONS"; shift ;;
      --profiling-iterations) PROFILING_ITERATIONS="$2"; require_positive_int --profiling-iterations "$PROFILING_ITERATIONS"; shift 2 ;;
      --output-dir=*) OUTPUT_DIR="${1#--output-dir=}"; shift ;;
      --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
      --cutlass-dir=*) CUTLASS_DIR="${1#--cutlass-dir=}"; shift ;;
      --cutlass-dir) CUTLASS_DIR="$2"; shift 2 ;;
      --build-dir=*) BUILD_DIR="${1#--build-dir=}"; shift ;;
      --build-dir) BUILD_DIR="$2"; shift 2 ;;
      --kernel-filter-bf16=*) FILTER[bf16]="${1#--kernel-filter-bf16=}"; shift ;;
      --kernel-filter-bf16) FILTER[bf16]="$2"; shift 2 ;;
      --kernel-filter-tf32=*) FILTER[tf32]="${1#--kernel-filter-tf32=}"; shift ;;
      --kernel-filter-tf32) FILTER[tf32]="$2"; shift 2 ;;
      --kernel-filter-int8=*) FILTER[int8]="${1#--kernel-filter-int8=}"; shift ;;
      --kernel-filter-int8) FILTER[int8]="$2"; shift 2 ;;
      --jobs=*) JOBS="${1#--jobs=}"; require_positive_int --jobs "$JOBS"; shift ;;
      --jobs) JOBS="$2"; require_positive_int --jobs "$JOBS"; shift 2 ;;
      --skip-build) SKIP_BUILD=true; shift ;;
      --clean) DO_CLEAN=true; shift ;;
      --allow-empty-kernels) ALLOW_EMPTY_KERNELS=true; shift ;;
      --dry-run) DRY_RUN=true; shift ;;
      --smoke) DO_SMOKE=true; shift ;;
      --no-summary) NO_SUMMARY=true; shift ;;
      --summary-statistic=*) SUMMARY_STATISTIC="${1#--summary-statistic=}"; case "$SUMMARY_STATISTIC" in best|median|both) ;; *) die "invalid --summary-statistic: '${SUMMARY_STATISTIC}'" 5 ;; esac; shift ;;
      --summary-statistic) SUMMARY_STATISTIC="$2"; case "$SUMMARY_STATISTIC" in best|median|both) ;; *) die "invalid --summary-statistic: '${SUMMARY_STATISTIC}'" 5 ;; esac; shift 2 ;;
      --) shift; break ;;
      -*) die "unknown option: $1 (--help)" 5 ;;
      *)  die "unexpected positional: $1 (--help)" 5 ;;
    esac
  done
}

validate_flags() {
  if [[ "$SKIP_BUILD" == "true" && "$DO_CLEAN" == "true" ]]; then
    die "--skip-build and --clean are mutually exclusive" 5
  fi
  if [[ ! -d "$CUTLASS_DIR" ]]; then
    die "cutlass dir not found: ${CUTLASS_DIR}" 5
  fi
  : "${BUILD_DIR:=${CUTLASS_DIR}/build-sm86-profiler}"
  SUMMARY_DIR="$(dirname "${OUTPUT_DIR}")"
  RUN_TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
preflight() {
  log_section "preflight"
  local required=(cmake nvcc nvidia-smi jq) missing=()
  for t in "${required[@]}"; do
    if ! command -v "$t" >/dev/null 2>&1; then
      missing+=("$t")
      log_err "${t} not found on PATH"
    else
      log_info "found ${t}: $(command -v "$t")"
    fi
  done
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    missing+=("python3 (PYTHON_BIN=${PYTHON_BIN})")
  else
    log_info "python3: ${PYTHON_BIN}"
  fi
  if command -v ninja >/dev/null 2>&1; then
    BUILD_GENERATOR="ninja"; log_info "found ninja"
  elif command -v make >/dev/null 2>&1; then
    BUILD_GENERATOR="make"; log_info "found make (ninja absent)"
  else
    missing+=("make-or-ninja")
  fi
  if [[ ${#missing[@]} -gt 0 ]]; then
    RESULT_STATUS="preflight_failed"
    die "preflight failed: missing tool(s): ${missing[*]}" 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    RESULT_STATUS="preflight_failed"
    die "no CUDA GPU detected" 1
  fi
  local cc cc_expected
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  cc_expected="$(printf '%s' "${ARCH}" | sed -E 's/^([0-9])([0-9])$/\1.\2/')"
  if [[ -n "$cc" && -n "$cc_expected" && "$cc" != "$cc_expected" ]]; then
    log_warn "GPU compute capability ${cc} != --arch=${ARCH} (expected ${cc_expected}); continuing"
  fi
  if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "${OUTPUT_DIR}" || die "cannot mkdir ${OUTPUT_DIR}" 1
    if [[ "$SKIP_BUILD" != "true" ]]; then
      mkdir -p "$(dirname "${BUILD_DIR}")" || die "cannot mkdir parent of ${BUILD_DIR}" 1
    fi
  fi
  log_ok "preflight ok"
}

# ---------------------------------------------------------------------------
# CMake / build
# ---------------------------------------------------------------------------
clean_build_dir() {
  log_section "clean"
  [[ -d "$BUILD_DIR" ]] && { log_info "rm -rf ${BUILD_DIR}"; rm -rf -- "$BUILD_DIR"; } || log_info "no build dir; nothing to clean"
}

cmake_configure_argv() {
  local g="Unix Makefiles"
  [[ "$BUILD_GENERATOR" == "ninja" ]] && g="Ninja"
  local combined="${FILTER[bf16]},${FILTER[tf32]},${FILTER[int8]}"
  printf '%s\n' \
    cmake -S "${CUTLASS_DIR}" -B "${BUILD_DIR}" -G "${g}" \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCUTLASS_NVCC_ARCHS=${ARCH}" \
    "-DCUTLASS_LIBRARY_OPERATIONS=gemm" \
    "-DCUTLASS_LIBRARY_KERNELS=${combined}"
}

configure_cmake() {
  log_section "configure"
  mkdir -p "${BUILD_DIR}"
  local -a argv
  mapfile -t argv < <(cmake_configure_argv)
  log_info "${argv[*]}"
  "${argv[@]}"
}

discover_generated_kernels_txt() {
  local primary="${BUILD_DIR}/tools/library/generated_kernels.txt"
  if [[ -f "$primary" ]]; then GENERATED_KERNELS_TXT="$primary"; return 0; fi
  local found
  found="$(find "${BUILD_DIR}" -type f -name 'generated_kernels.txt' -print 2>/dev/null | head -n1)"
  if [[ -n "$found" ]]; then GENERATED_KERNELS_TXT="$found"; return 0; fi
  return 1
}

count_kernels_matching() {
  local pattern="$1"
  if [[ -z "$GENERATED_KERNELS_TXT" || ! -f "$GENERATED_KERNELS_TXT" ]]; then printf '0\n'; return 0; fi
  local re; re="$(printf '%s' "$pattern" | sed -e 's/[][\.|\^\$+(){}]/\\&/g' -e 's/\*/.*/g' -e 's/?/./g')"
  grep -Ec -- "$re" "$GENERATED_KERNELS_TXT" || true
}

handle_empty_kernels() {
  RESULT_STATUS="zero_match"
  if [[ "$ALLOW_EMPTY_KERNELS" == "true" ]]; then
    log_warn "0 kernels matched in at least one precision; continuing under --allow-empty-kernels"
    write_manifest
    final_summary_echo
    exit 0
  fi
  log_err "0 kernels matched in at least one precision (use --allow-empty-kernels to downgrade)"
  write_manifest
  exit 2
}

verify_kernel_filter() {
  log_section "verify"
  if ! discover_generated_kernels_txt; then
    log_warn "kernel manifest not found under ${BUILD_DIR}; treating as zero-match"
    handle_empty_kernels
  fi
  log_info "kernel manifest: ${GENERATED_KERNELS_TXT}"
  local any_zero=false
  for p in "${PRECISIONS[@]}"; do
    N_MATCHED[$p]="$(count_kernels_matching "${FILTER[$p]}")"
    log_info "${p} kernels matched (${FILTER[$p]}): ${N_MATCHED[$p]}"
    (( N_MATCHED[$p] == 0 )) && any_zero=true
  done
  $any_zero && handle_empty_kernels
  log_ok "kernel filter verification passed"
}

build_profiler() {
  log_section "build"
  local -a argv=(cmake --build "${BUILD_DIR}" --target cutlass_profiler --parallel "${JOBS}")
  log_info "${argv[*]}"
  if "${argv[@]}"; then
    BUILD_EXIT_CODE=0; log_ok "build complete"
  else
    BUILD_EXIT_CODE=$?
    RESULT_STATUS="build_failed"
    write_manifest
    die "cutlass_profiler build failed (exit ${BUILD_EXIT_CODE})" 3
  fi
}

cutlass_profiler_path() { printf '%s/tools/profiler/cutlass_profiler\n' "${BUILD_DIR}"; }

require_profiler_binary() {
  local p; p="$(cutlass_profiler_path)"
  [[ -x "$p" ]] || die "cannot --skip-build: cutlass_profiler not found at ${p}" 3
}

# ---------------------------------------------------------------------------
# Enumerate concrete kernel names
# ---------------------------------------------------------------------------
enumerate_kernels() {
  local p="$1" outfile="${OUTPUT_DIR}/dryrun_kernels_${p}.txt"
  log_section "enumerate-${p}"
  if [[ -z "$GENERATED_KERNELS_TXT" || ! -f "$GENERATED_KERNELS_TXT" ]]; then
    discover_generated_kernels_txt || true
  fi
  if [[ -z "$GENERATED_KERNELS_TXT" || ! -f "$GENERATED_KERNELS_TXT" ]]; then
    : > "${outfile}"; DRYRUN_NAMES[$p]=""
    return 0
  fi
  local re; re="$(printf '%s' "${FILTER[$p]}" | sed -e 's/[][\.|\^\$+(){}]/\\&/g' -e 's/\*/.*/g' -e 's/?/./g')"
  log_info "matching ${FILTER[$p]} against ${GENERATED_KERNELS_TXT}"
  local names; names="$(grep -E -- "${re}" "${GENERATED_KERNELS_TXT}" || true)"
  printf '%s\n' "${names}" > "${outfile}"
  DRYRUN_NAMES[$p]="${names}"
  local count; count="$(printf '%s\n' "${names}" | grep -cE '.' || true)"
  log_info "${p}: enumerated ${count} concrete kernel name(s) -> ${outfile}"
}

# ---------------------------------------------------------------------------
# nvidia-smi snapshot (returns JSON)
# ---------------------------------------------------------------------------
capture_gpu_state() {
  local q
  q="$(nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
  if [[ -z "$q" ]]; then
    printf '%s' '{"sm_clock_mhz":"unknown","mem_clock_mhz":"unknown","temp_c":"unknown","power_w":"unknown"}'
    return 0
  fi
  local sm mc tc pw
  IFS=',' read -r sm mc tc pw <<<"$q"
  jq -n --arg sm "$sm" --arg mc "$mc" --arg tc "$tc" --arg pw "$pw" \
    '{sm_clock_mhz:$sm, mem_clock_mhz:$mc, temp_c:$tc, power_w:$pw}'
}

# ---------------------------------------------------------------------------
# Run the profiler for one precision
# ---------------------------------------------------------------------------
run_profiler() {
  local p="$1" dim="$2" warmup="$3" iters="$4"
  local csv="${OUTPUT_DIR}/cutlass_profiler_${p}_${dim}.csv"
  local log="${OUTPUT_DIR}/cutlass_profiler_${p}_${dim}.log"
  local csv_base="${csv%.csv}"
  local names_file="${OUTPUT_DIR}/dryrun_kernels_${p}.txt"

  log_section "profile-${p}-${dim}"

  if [[ -z "${DRYRUN_NAMES[$p]}" ]]; then
    log_warn "0 matching kernels; nothing to profile for ${p}"
    {
      printf '0 matching kernels; nothing to profile for %s\n' "$p"
      printf '\n--- enumerated kernel list ---\n'
      cat "${names_file}" 2>/dev/null || true
    } > "$log"
    PROFILE_EXIT[$p]=0
    return 0
  fi

  PRE_STATE[$p]="$(capture_gpu_state)"

  local profiler exit_code=0
  profiler="$(cutlass_profiler_path)"

  while IFS= read -r f; do rm -f -- "$f"; done < <(find "$(dirname "${csv_base}")" -maxdepth 1 -type f -name "$(basename "${csv_base}").*.csv" -print 2>/dev/null)
  rm -f -- "${csv}"

  local -a argv=(
    "${profiler}"
    "--m=${dim}" "--n=${dim}" "--k=${dim}"
    "--warmup-iterations=${warmup}"
    "--profiling-iterations=${iters}"
    "--verification-enabled=false"
    "--kernels-file=${names_file}"
    "--output=${csv_base}"
  )
  log_info "${argv[*]}"
  if "${argv[@]}" >"${log}" 2>&1; then
    exit_code=0
  else
    exit_code=$?
    log_warn "profiler ${p} exited ${exit_code}; see ${log}"
  fi

  POST_STATE[$p]="$(capture_gpu_state)"
  PROFILE_EXIT[$p]=$exit_code

  local -a per_op=()
  while IFS= read -r f; do per_op+=("$f"); done < <(find "$(dirname "${csv_base}")" -maxdepth 1 -type f -name "$(basename "${csv_base}").*.csv" -print 2>/dev/null | sort)
  local primary="" primary_rows=0 op_name=""
  for f in "${per_op[@]}"; do
    local rows; rows=$(($(wc -l < "$f") - 1))
    if (( rows > primary_rows )); then primary="$f"; primary_rows=$rows; fi
  done
  if [[ -n "$primary" ]]; then
    op_name="$(basename "$primary")"
    op_name="${op_name#$(basename "${csv_base}").}"
    op_name="${op_name%.csv}"
    mv -- "$primary" "$csv"
    log_ok "${p} ${dim}^3 csv -> ${csv} (${primary_rows} data row(s) from operation '${op_name}')"
  fi
  for f in "${per_op[@]}"; do
    [[ -f "$f" ]] || continue
    if (( $(wc -l < "$f") <= 1 )); then rm -f -- "$f"; fi
  done
  if [[ ! -f "$csv" ]]; then log_warn "${p} ${dim}^3 csv missing — see ${log}"; fi
}

run_profiler_all_full() {
  for p in "${PRECISIONS[@]}"; do
    run_profiler "$p" "$M" "$WARMUP_ITERATIONS" "$PROFILING_ITERATIONS"
  done
}

run_profiler_all_smoke() {
  log_section "smoke"
  local -A prev
  for p in "${PRECISIONS[@]}"; do prev[$p]=${PROFILE_EXIT[$p]}; done
  local any_fail=false
  for p in "${PRECISIONS[@]}"; do
    run_profiler "$p" 1024 2 3
    (( PROFILE_EXIT[$p] != 0 )) && any_fail=true
  done
  for p in "${PRECISIONS[@]}"; do PROFILE_EXIT[$p]=${prev[$p]}; done
  $any_fail && die "smoke run failed" 4
  log_ok "smoke run passed"
}

# ---------------------------------------------------------------------------
# Classify result_status from concrete kernel names
# ---------------------------------------------------------------------------
classify_result_status() {
  if [[ "$RESULT_STATUS" == "preflight_failed" || "$RESULT_STATUS" == "build_failed" || "$RESULT_STATUS" == "zero_match" ]]; then return; fi
  # On sm_80/sm_86 there are NO blockscaled / NVFP4 / MXFP8 kernel families.
  # Any kernel matched here is dense by definition. If at least one precision
  # matched > 0 kernels, we report dense_found.
  local any_dense=false
  for p in "${PRECISIONS[@]}"; do (( N_MATCHED[$p] > 0 )) && any_dense=true; done
  $any_dense && RESULT_STATUS="dense_found" || RESULT_STATUS="zero_match"
}

# ---------------------------------------------------------------------------
# Manifest emission
# ---------------------------------------------------------------------------
write_manifest() {
  log_section "manifest"
  local manifest="${OUTPUT_DIR}/manifest.json"
  mkdir -p "${OUTPUT_DIR}"

  local host nvcc_v cmake_v driver_v cuda_v gpu_name gpu_pci cc cutlass_commit
  host="$(hostname 2>/dev/null || printf 'unknown')"
  nvcc_v="$(nvcc --version 2>/dev/null | tail -n1 || printf 'unknown')"
  cmake_v="$(cmake --version 2>/dev/null | head -n1 || printf 'unknown')"
  driver_v="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || printf 'unknown')"
  cuda_v="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || printf 'unknown')"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || printf 'unknown')"
  gpu_pci="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || printf 'unknown')"
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || printf 'unknown')"
  if [[ -d "${CUTLASS_DIR}/.git" ]]; then
    cutlass_commit="$(git -C "${CUTLASS_DIR}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
  else
    cutlass_commit="unknown"
  fi

  # Build per-precision blocks via jq
  local matched_json filters_json names_json pre_json post_json csv_paths_json log_paths_json exit_codes_json
  local jq_args=(-n)
  jq_args+=(--arg timestamp "$RUN_TIMESTAMP")
  jq_args+=(--arg host "$host")
  jq_args+=(--arg gpu_name "$gpu_name")
  jq_args+=(--arg gpu_pci_id "$gpu_pci")
  jq_args+=(--arg compute_capability "$cc")
  jq_args+=(--arg driver_version "$driver_v")
  jq_args+=(--arg cuda_version "$cuda_v")
  jq_args+=(--arg nvcc_version "$nvcc_v")
  jq_args+=(--arg cmake_version "$cmake_v")
  jq_args+=(--arg build_generator "$BUILD_GENERATOR")
  jq_args+=(--arg cutlass_commit "$cutlass_commit")
  jq_args+=(--arg cutlass_dir "$CUTLASS_DIR")
  jq_args+=(--arg build_dir "$BUILD_DIR")
  jq_args+=(--argjson m "$M" --argjson n "$N" --argjson k "$K")
  jq_args+=(--argjson warmup_iterations "$WARMUP_ITERATIONS")
  jq_args+=(--argjson profiling_iterations "$PROFILING_ITERATIONS")
  jq_args+=(--argjson jobs "$JOBS")
  jq_args+=(--arg result_status "$RESULT_STATUS")
  jq_args+=(--argjson build_exit_code "$BUILD_EXIT_CODE")

  local matched_obj='{}' filters_obj='{}' names_obj='{}' pre_obj='{}' post_obj='{}' csvs_obj='{}' logs_obj='{}' exits_obj='{}'
  for p in "${PRECISIONS[@]}"; do
    matched_obj="$(jq --argjson v "${N_MATCHED[$p]}" --arg k "$p" '. + {($k): $v}' <<<"$matched_obj")"
    filters_obj="$(jq --arg v "${FILTER[$p]}" --arg k "$p" '. + {($k): $v}' <<<"$filters_obj")"
    local names_arr
    names_arr="$(printf '%s' "${DRYRUN_NAMES[$p]}" | jq -R -s 'split("\n") | map(select(length > 0))')"
    names_obj="$(jq --argjson v "$names_arr" --arg k "$p" '. + {($k): $v}' <<<"$names_obj")"
    pre_obj="$(jq --argjson v "${PRE_STATE[$p]}" --arg k "$p" '. + {($k): $v}' <<<"$pre_obj")"
    post_obj="$(jq --argjson v "${POST_STATE[$p]}" --arg k "$p" '. + {($k): $v}' <<<"$post_obj")"
    csvs_obj="$(jq --arg v "${OUTPUT_DIR}/cutlass_profiler_${p}_${M}.csv" --arg k "$p" '. + {($k): $v}' <<<"$csvs_obj")"
    logs_obj="$(jq --arg v "${OUTPUT_DIR}/cutlass_profiler_${p}_${M}.log" --arg k "$p" '. + {($k): $v}' <<<"$logs_obj")"
    exits_obj="$(jq --argjson v "${PROFILE_EXIT[$p]}" --arg k "$p" '. + {($k): $v}' <<<"$exits_obj")"
  done
  jq_args+=(--argjson n_kernels_matched "$matched_obj")
  jq_args+=(--argjson kernel_filters "$filters_obj")
  jq_args+=(--argjson dryrun_kernel_names "$names_obj")
  jq_args+=(--argjson pre_state "$pre_obj")
  jq_args+=(--argjson post_state "$post_obj")
  jq_args+=(--argjson csv_paths "$csvs_obj")
  jq_args+=(--argjson log_paths "$logs_obj")
  jq_args+=(--argjson profile_exit_codes "$exits_obj")
  jq_args+=(--arg summary_path "${SUMMARY_DIR}/SUMMARY.md")

  # Source CSVs from the probe workflow (cuBLASLt + WMMA)
  local cublaslt_dir="${SOLAR_ROOT}/measurements/rtx3090/cublaslt"
  local wmma_dir="${SOLAR_ROOT}/measurements/rtx3090/wmma"
  local cublaslt_json='[]' wmma_json='[]'
  if [[ -d "$cublaslt_dir" ]]; then
    cublaslt_json="$(find "$cublaslt_dir" -type f -name '*.csv' -print 2>/dev/null | jq -R -s 'split("\n") | map(select(length > 0))')"
  fi
  if [[ -d "$wmma_dir" ]]; then
    wmma_json="$(find "$wmma_dir" -type f -name '*.csv' -print 2>/dev/null | jq -R -s 'split("\n") | map(select(length > 0))')"
  fi
  jq_args+=(--argjson cublaslt_source_csvs "$cublaslt_json")
  jq_args+=(--argjson wmma_source_csvs "$wmma_json")

  jq "${jq_args[@]}" '{
    timestamp: $timestamp, host: $host, gpu_name: $gpu_name, gpu_pci_id: $gpu_pci_id,
    compute_capability: $compute_capability, driver_version: $driver_version,
    cuda_version: $cuda_version, nvcc_version: $nvcc_version, cmake_version: $cmake_version,
    build_generator: $build_generator, cutlass_commit: $cutlass_commit,
    cutlass_dir: $cutlass_dir, build_dir: $build_dir,
    matrix_size: { m: $m, n: $n, k: $k },
    warmup_iterations: $warmup_iterations, profiling_iterations: $profiling_iterations,
    jobs: $jobs, result_status: $result_status, build_exit_code: $build_exit_code,
    kernel_filters: $kernel_filters,
    n_kernels_matched: $n_kernels_matched,
    dryrun_kernel_names: $dryrun_kernel_names,
    pre_state: $pre_state, post_state: $post_state,
    profile_exit_codes: $profile_exit_codes,
    csv_paths: $csv_paths, log_paths: $log_paths,
    summary_path: $summary_path,
    cublaslt_source_csvs: $cublaslt_source_csvs,
    wmma_source_csvs: $wmma_source_csvs
  }' > "${manifest}"

  if ! jq -e . "${manifest}" >/dev/null 2>&1; then
    die "manifest write produced invalid JSON: ${manifest}" 1
  fi
  log_ok "manifest -> ${manifest}"
}

# ---------------------------------------------------------------------------
# SUMMARY.md
# ---------------------------------------------------------------------------
emit_summary() {
  if [[ "$NO_SUMMARY" == "true" ]]; then log_info "skipping SUMMARY.md (--no-summary)"; return 0; fi
  log_section "tabulate"
  mkdir -p "${SUMMARY_DIR}"
  local summary_path="${SUMMARY_DIR}/SUMMARY.md"
  local cublaslt_dir="${SOLAR_ROOT}/measurements/rtx3090/cublaslt"
  local cublaslt_bf16=""
  [[ -f "${cublaslt_dir}/cublaslt_bf16_${M}.csv" ]] && cublaslt_bf16="${cublaslt_dir}/cublaslt_bf16_${M}.csv"
  local arch_yaml="${SOLAR_ROOT}/configs/arch/3090ti.yaml"
  local bf16_csv="${OUTPUT_DIR}/cutlass_profiler_bf16_${M}.csv"
  local tf32_csv="${OUTPUT_DIR}/cutlass_profiler_tf32_${M}.csv"
  local int8_csv="${OUTPUT_DIR}/cutlass_profiler_int8_${M}.csv"
  local bf16_log="${OUTPUT_DIR}/cutlass_profiler_bf16_${M}.log"
  local tf32_log="${OUTPUT_DIR}/cutlass_profiler_tf32_${M}.log"
  local int8_log="${OUTPUT_DIR}/cutlass_profiler_int8_${M}.log"

  "${PYTHON_BIN}" - "$summary_path" "$arch_yaml" "$bf16_csv" "$tf32_csv" "$int8_csv" \
                   "$cublaslt_bf16" "$SUMMARY_STATISTIC" "$RESULT_STATUS" "$RUN_TIMESTAMP" \
                   "$bf16_log" "$tf32_log" "$int8_log" <<'PYEOF'
import csv, statistics, sys
from pathlib import Path

(summary_path, arch_yaml, bf16_csv, tf32_csv, int8_csv, cublaslt_bf16,
 stat_mode, result_status, run_ts, bf16_log, tf32_log, int8_log) = sys.argv[1:13]

def load_arch_peaks(p):
    out = {'bf16': None, 'tf32': None, 'int8': None}
    if not Path(p).is_file():
        return out
    freq = None
    macs = {}
    for line in Path(p).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line or ':' not in line:
            continue
        k, v = (x.strip() for x in line.split(':', 1))
        try:
            num = float(v)
        except ValueError:
            continue
        if k == 'freq_GHz':
            freq = num
        elif k == 'MAC_per_cycle_bf16_tc':
            macs['bf16'] = num
        elif k == 'MAC_per_cycle_tf32_tc':
            macs['tf32'] = num
        elif k == 'MAC_per_cycle_int8_tc':
            macs['int8'] = num
    if freq is not None:
        for k, m in macs.items():
            out[k] = m * 2 * freq / 1000.0
    return out

TFLOPS_KEYS = (
    ('GFLOPs', 1000.0),
    ('GFLOPS', 1000.0),
    ('tflops', 1.0),
    ('TFLOPS', 1.0),
    ('Flops/Sec', 1e12),
    ('Flop/s', 1e12),
)

def extract_tflops(row):
    for key, divisor in TFLOPS_KEYS:
        v = row.get(key)
        if v is None or v == '':
            continue
        try:
            return float(v) / divisor
        except (TypeError, ValueError):
            continue
    return None

def parse_csv_tflops(path):
    if not path or not Path(path).is_file():
        return None
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t = extract_tflops(row)
            if t is not None:
                rows.append(t)
    if not rows:
        return None
    rows.sort(reverse=True)
    return {'best': rows[0], 'median': statistics.median(rows), 'count': len(rows)}

peaks = load_arch_peaks(arch_yaml)
cublaslt_bf16_stats = parse_csv_tflops(cublaslt_bf16)
cutlass_stats = {
    'bf16': parse_csv_tflops(bf16_csv),
    'tf32': parse_csv_tflops(tf32_csv),
    'int8': parse_csv_tflops(int8_csv),
}

show_best = stat_mode in ('best', 'both')
show_median = stat_mode in ('median', 'both')

def fmt(stats, key, missing_link):
    if stats is None or stats.get(key) is None:
        # Suppress the "see <link>" suffix when the missing_link itself is a
        # not-measured / not-applicable explanation (no actual artifact link).
        if missing_link.startswith('N/A'):
            return missing_link
        return f"N/A — see {missing_link}"
    return f"{stats[key]:.2f}"

def existing(p):
    return p and Path(p).is_file()

def link(label, path):
    return f'[{label}]({path})' if existing(path) else None

def joined(*parts):
    parts = [p for p in parts if p]
    return ', '.join(parts) if parts else 'N/A'

header_cells = ['Source', 'Precision triplet']
if show_best:
    header_cells += ['BF16 best TFLOPS', 'TF32 best TFLOPS', 'INT8 best TOPS']
if show_median:
    header_cells += ['BF16 median TFLOPS', 'TF32 median TFLOPS', 'INT8 median TOPS']
header_cells += ['Source artifact']
sep_cells = ['---'] * len(header_cells)

lines = []
lines.append(f"# RTX 3090 Ti GEMM cross-check (run: {run_ts})")
lines.append("")
lines.append(f"Result status: `{result_status}`")
lines.append("")
lines.append('| ' + ' | '.join(header_cells) + ' |')
lines.append('| ' + ' | '.join(sep_cells) + ' |')

spec_link = '[configs/arch/3090ti.yaml](../../configs/arch/3090ti.yaml)'
cublaslt_missing_bf16 = link('cublaslt_bf16_csv', cublaslt_bf16) or 'N/A (no source)'
cutlass_missing = {
    'bf16': link('bf16 log', bf16_log) or '(no log)',
    'tf32': link('tf32 log', tf32_log) or '(no log)',
    'int8': link('int8 log', int8_log) or '(no log)',
}

def row(name, prec_str, bf16_stats, tf32_stats, int8_stats, source_link,
        bf16_miss, tf32_miss, int8_miss):
    cells = [name, prec_str]
    stats = [bf16_stats, tf32_stats, int8_stats]
    misses = [bf16_miss, tf32_miss, int8_miss]
    if show_best:
        for s, m in zip(stats, misses):
            cells.append(fmt(s, 'best', m))
    if show_median:
        for s, m in zip(stats, misses):
            cells.append(fmt(s, 'median', m))
    cells.append(source_link)
    return '| ' + ' | '.join(cells) + ' |'

# Spec peak (no INT8/TF32 from cuBLASLt — just BF16)
peak_stats = {k: ({'best': v, 'median': v} if v is not None else None) for k, v in peaks.items()}
lines.append(row(
    'Spec peak',
    'bf16/bf16 dense, tf32/tf32 dense, s8/s8 dense',
    peak_stats['bf16'], peak_stats['tf32'], peak_stats['int8'],
    spec_link,
    spec_link, spec_link, spec_link,
))

# cuBLASLt (BF16 only on this rig — lt_gemm.cu has no TF32/INT8 paths)
cublaslt_link = joined(link('cublaslt_bf16', cublaslt_bf16))
lines.append(row(
    'cuBLASLt achieved',
    'bf16/bf16 dense (TF32/INT8: not measured)',
    cublaslt_bf16_stats, None, None,
    cublaslt_link,
    cublaslt_missing_bf16, 'N/A (not measured)', 'N/A (not measured)',
))

# CUTLASS dense (sm_80 has no blockscaled path — every match here is dense)
cutlass_link = joined(
    link('cutlass_profiler_bf16', bf16_csv),
    link('cutlass_profiler_tf32', tf32_csv),
    link('cutlass_profiler_int8', int8_csv),
)
lines.append(row(
    'CUTLASS dense',
    'bf16/bf16/fp32_acc, tf32/tf32/fp32_acc, s8/s8/s32_acc',
    cutlass_stats['bf16'], cutlass_stats['tf32'], cutlass_stats['int8'],
    cutlass_link,
    cutlass_missing['bf16'], cutlass_missing['tf32'], cutlass_missing['int8'],
))

lines.append('')
lines.append('## Notes')
lines.append('')
lines.append('- The RTX 3090 Ti / GA102 (sm_86) has no FP8 tensor-core hardware (FP8 was introduced in Hopper sm_89/sm_90), no NVFP4 (Blackwell sm_100/sm_120), and therefore no blockscaled CUTLASS kernel families. The TF32 + INT8 columns substitute for the FP8 / NVFP4 columns that the 5060 SUMMARY.md reports.')
lines.append('- BF16 / TF32 are reported in TFLOPS (= GFLOPs / 1000); INT8 is reported in TOPS using the same formula (the CUTLASS profiler emits the integer-op count under the GFLOPs column for s8 GEMMs).')
lines.append('- TF32 storage on Ampere is FP32-shaped; the CUTLASS profiler still treats `Bytes` and `Flops` consistently, so the TFLOPS computation is comparable.')
lines.append('')

Path(summary_path).write_text('\n'.join(lines))
PYEOF
  log_ok "summary -> ${summary_path}"
}

# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------
print_dry_run() {
  log_section "dry-run"
  log_info "  arch=${ARCH}"
  log_info "  matrix=${M}x${N}x${K}"
  log_info "  warmup=${WARMUP_ITERATIONS} iters=${PROFILING_ITERATIONS}"
  log_info "  output_dir=${OUTPUT_DIR}"
  log_info "  build_dir=${BUILD_DIR}"
  for p in "${PRECISIONS[@]}"; do
    log_info "  filter[${p}]=${FILTER[$p]}"
  done
  log_info "  jobs=${JOBS} build_generator=${BUILD_GENERATOR}"
  log_info "  skip_build=${SKIP_BUILD} clean=${DO_CLEAN} smoke=${DO_SMOKE} no_summary=${NO_SUMMARY}"

  $DO_CLEAN && log_info "would: rm -rf ${BUILD_DIR}"
  if [[ "$SKIP_BUILD" != "true" ]]; then
    local -a cargv; mapfile -t cargv < <(cmake_configure_argv)
    log_info "would: ${cargv[*]}"
    log_info "would: cmake --build ${BUILD_DIR} --target cutlass_profiler --parallel ${JOBS}"
  fi
  local profiler; profiler="$(cutlass_profiler_path)"
  for p in "${PRECISIONS[@]}"; do
    log_info "would: enumerate ${p} kernels by grep -E '${FILTER[$p]}' against generated_kernels.txt"
    log_info "would: ${profiler} --m=${M} --n=${N} --k=${K} --warmup-iterations=${WARMUP_ITERATIONS} --profiling-iterations=${PROFILING_ITERATIONS} --verification-enabled=false --kernels-file=${OUTPUT_DIR}/dryrun_kernels_${p}.txt --output=${OUTPUT_DIR}/cutlass_profiler_${p}_${M}"
  done
  $NO_SUMMARY || log_info "would: write ${SUMMARY_DIR}/SUMMARY.md"
  log_info "would: write ${OUTPUT_DIR}/manifest.json"
  log_ok "dry-run complete (no commands executed)"
}

# ---------------------------------------------------------------------------
# Final stdout summary
# ---------------------------------------------------------------------------
final_summary_echo() {
  printf '\n%s===============================================%s\n' "${C_GREEN}" "${C_NC}"
  printf '%s   bench_cutlass_3090.sh — final summary       %s\n' "${C_GREEN}" "${C_NC}"
  printf '%s===============================================%s\n' "${C_GREEN}" "${C_NC}"
  printf 'Result status:        %s\n' "${RESULT_STATUS}"
  printf 'Build exit code:      %s\n' "${BUILD_EXIT_CODE}"
  for p in "${PRECISIONS[@]}"; do
    printf '%s profile exit:    %s (matched=%s)\n' "$p" "${PROFILE_EXIT[$p]}" "${N_MATCHED[$p]}"
  done
  printf 'Output dir:           %s\n' "${OUTPUT_DIR}"
  printf 'Summary:              %s\n' "${SUMMARY_DIR}/SUMMARY.md"
  printf 'Manifest:             %s\n' "${OUTPUT_DIR}/manifest.json"
}

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
main() {
  parse_flags "$@"
  validate_flags
  preflight

  if [[ "$DRY_RUN" == "true" ]]; then print_dry_run; exit 0; fi

  if [[ "$DO_CLEAN" == "true" ]]; then clean_build_dir; fi

  if [[ "$SKIP_BUILD" == "true" ]]; then
    require_profiler_binary
    log_info "skip-build: reusing cutlass_profiler at $(cutlass_profiler_path)"
    if discover_generated_kernels_txt; then
      local any_zero=false
      for p in "${PRECISIONS[@]}"; do
        N_MATCHED[$p]="$(count_kernels_matching "${FILTER[$p]}")"
        log_info "skip-build: kernels matched ${p}=${N_MATCHED[$p]}"
        (( N_MATCHED[$p] == 0 )) && any_zero=true
      done
      $any_zero && handle_empty_kernels
    fi
  else
    configure_cmake
    verify_kernel_filter
    build_profiler
  fi

  for p in "${PRECISIONS[@]}"; do enumerate_kernels "$p"; done

  if [[ "$DO_SMOKE" == "true" ]]; then run_profiler_all_smoke; fi

  run_profiler_all_full
  classify_result_status
  emit_summary
  write_manifest

  local fail_reason=""
  for p in "${PRECISIONS[@]}"; do
    local csv="${OUTPUT_DIR}/cutlass_profiler_${p}_${M}.csv"
    if (( N_MATCHED[$p] > 0 )); then
      if (( PROFILE_EXIT[$p] != 0 )); then
        fail_reason="${p} profile exited ${PROFILE_EXIT[$p]}; see ${OUTPUT_DIR}/cutlass_profiler_${p}_${M}.log"
        break
      elif [[ ! -f "$csv" ]]; then
        fail_reason="required ${p} CSV not produced: ${csv}"
        break
      fi
    fi
  done
  if [[ -n "$fail_reason" ]]; then
    log_err "$fail_reason"
    final_summary_echo
    exit 4
  fi

  final_summary_echo
}

main "$@"
