#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the CUTLASS v4.4.1 cutlass_profiler with sm_120 (Blackwell consumer)
# kernel filters, run BF16 + FP8 e4m3 GEMM benchmarks at 8192^3 on the local
# RTX 5060 Laptop GPU, and emit a triangulated SUMMARY.md against the spec
# peak in configs/arch/5060.yaml and the prior cuBLASLt 13.4 datapoints.
#
# Outputs:
#   measurements/rtx5060/SUMMARY.md
#   measurements/rtx5060/cutlass/cutlass_profiler_bf16_8192.csv
#   measurements/rtx5060/cutlass/cutlass_profiler_bf16_8192.log
#   measurements/rtx5060/cutlass/cutlass_profiler_fp8_8192.csv
#   measurements/rtx5060/cutlass/cutlass_profiler_fp8_8192.log
#   measurements/rtx5060/cutlass/dryrun_kernels_bf16.txt
#   measurements/rtx5060/cutlass/dryrun_kernels_fp8.txt
#   measurements/rtx5060/cutlass/manifest.json
#
# Exit codes:
#   0  success (or zero-match under --allow-empty-kernels, or --dry-run)
#   1  preflight / environment failure
#   2  zero kernels matched (without --allow-empty-kernels)
#   3  build failure
#   4  profile failure
#   5  invalid CLI argument

set -euo pipefail

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Color helpers (mirrored from scripts/run_tests.sh)
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RED=$'\033[0;31m'
  C_GREEN=$'\033[0;32m'
  C_YELLOW=$'\033[1;33m'
  C_BLUE=$'\033[0;34m'
  C_NC=$'\033[0m'
else
  C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_NC=""
fi

log_section() { printf '\n%s==> %s%s\n' "${C_BLUE}" "$*" "${C_NC}"; }
log_info()    { printf '    %s\n' "$*"; }
log_ok()      { printf '%sOK%s: %s\n' "${C_GREEN}" "${C_NC}" "$*"; }
log_warn()    { printf '%sWARN%s: %s\n' "${C_YELLOW}" "${C_NC}" "$*" >&2; }
log_err()     { printf '%sERR%s: %s\n' "${C_RED}" "${C_NC}" "$*" >&2; }
die() {
  local msg="${1:-error}"
  local code="${2:-1}"
  log_err "$msg"
  exit "$code"
}

# ---------------------------------------------------------------------------
# Defaults (every flag documented in --help)
# ---------------------------------------------------------------------------
ARCH="120a"
M=8192
N=8192
K=8192
WARMUP_ITERATIONS=10
PROFILING_ITERATIONS=20
OUTPUT_DIR="${SOLAR_ROOT}/measurements/rtx5060/cutlass"
CUTLASS_DIR="${SOLAR_ROOT}/cutlass"
BUILD_DIR=""
KERNEL_FILTER_BF16='*sm120*bf16*'
KERNEL_FILTER_FP8='*sm120*fp8*'
JOBS=8
SKIP_BUILD=false
DO_CLEAN=false
ALLOW_EMPTY_KERNELS=false
DRY_RUN=false
UPDATE_MEMORY=false
MIGRATE_TMP=false
SUMMARY_STATISTIC="both"
DO_SMOKE=false
NO_SUMMARY=false

# ---------------------------------------------------------------------------
# Internal state populated as the run progresses
# ---------------------------------------------------------------------------
SUMMARY_DIR=""
GENERATED_KERNELS_TXT=""
N_KERNELS_MATCHED_BF16=0
N_KERNELS_MATCHED_FP8=0
DRYRUN_KERNEL_NAMES_BF16=""
DRYRUN_KERNEL_NAMES_FP8=""
BUILD_GENERATOR="make"
BUILD_EXIT_CODE=-1
PROFILE_BF16_EXIT_CODE=-1
PROFILE_FP8_EXIT_CODE=-1
RESULT_STATUS="unknown"
BF16_PRE_STATE='{"sm_clock_mhz":"unknown","mem_clock_mhz":"unknown","temp_c":"unknown","power_w":"unknown"}'
BF16_POST_STATE="${BF16_PRE_STATE}"
FP8_PRE_STATE="${BF16_PRE_STATE}"
FP8_POST_STATE="${BF16_PRE_STATE}"
RUN_TIMESTAMP=""

# ---------------------------------------------------------------------------
# --help body
# ---------------------------------------------------------------------------
print_help() {
  cat <<'EOF'
Usage: bench_cutlass_5060.sh [OPTIONS]

Build the CUTLASS v4.4.1 cutlass_profiler with sm_120 kernel filters and run
BF16 + FP8 8192^3 GEMM benchmarks against the local RTX 5060 Laptop GPU.

Options (all have defaults; defaults shown in brackets):

  --arch=<str>                    Target arch passed to CUTLASS_NVCC_ARCHS.
                                  Use 120a (GeForce Blackwell with extensions)
                                  or 120f (forward-compat) — plain 120 is
                                  rejected by CUTLASS's sm_120 generator path.
                                  [120a]

  --m=<int>                       GEMM M dimension. [8192]
  --n=<int>                       GEMM N dimension. [8192]
  --k=<int>                       GEMM K dimension. [8192]
  --matrix-size=<int>             Shorthand: sets --m, --n, --k all to <int>. [8192]

  --warmup-iterations=<int>       Profiler warmup iterations. [10]
  --profiling-iterations=<int>    Profiler timed iterations. [20]

  --output-dir=<path>             Output directory for CSV / log / manifest /
                                  dry-run kernel lists.
                                  [<solar>/measurements/rtx5060/cutlass]

  --cutlass-dir=<path>            CUTLASS source checkout.
                                  [<solar>/cutlass]

  --build-dir=<path>              Out-of-source build directory.
                                  [<cutlass-dir>/build-sm120-profiler]

  --kernel-filter-bf16=<glob>     Wildcard passed to CUTLASS_LIBRARY_KERNELS
                                  for BF16; expanded by the dry-run pass to
                                  concrete names before the timed run.
                                  ['*sm120*bf16*']

  --kernel-filter-fp8=<glob>      Same as above for FP8. ['*sm120*fp8*']

  --jobs=<int>                    Parallel build jobs (cmake --build --parallel).
                                  Default is conservative for a laptop. [8]

  --skip-build                    Reuse an existing cutlass_profiler binary at
                                  <build-dir>/tools/profiler/cutlass_profiler.
                                  Mutually exclusive with --clean.

  --clean                         Remove <build-dir> before configure.
                                  Mutually exclusive with --skip-build.

  --allow-empty-kernels           Downgrade the zero-match abort (exit 2) to a
                                  warning so the script writes the manifest
                                  with result_status=zero_match and exits 0.

  --dry-run                       Print every cmake / cmake-build / profiler
                                  invocation that would be executed (with
                                  resolved flag values) and exit 0 without
                                  running anything.

  --update-memory                 After a run, append a dated CUTLASS v4.4.1
                                  cross-check subsection to
                                  .claude/memory/rtx5060_probe_recipe.md and
                                  re-sync CLAUDE.md from the per-memory files.

  --migrate-tmp                   Copy /tmp/solar-rtx5060/cublaslt_*.csv into
                                  measurements/rtx5060/cublaslt/ and the
                                  ncu_hmma_fp16*.csv files into
                                  measurements/rtx5060/wmma/.

  --summary-statistic=<mode>      One of best | median | both. Controls which
                                  TFLOPS columns SUMMARY.md emits. [both]

  --smoke                         Run an extra 1024^3 / warmup=2 / iters=3
                                  pre-pass per precision before the 8192^3
                                  timed runs; abort early on smoke failure.

  --no-summary                    Skip SUMMARY.md emission entirely; the
                                  per-CUTLASS CSV / log / manifest are still
                                  produced.

  -h, --help                      Print this help and exit 0.

Each flag also accepts the space-separated form: --arch 120a is equivalent to
--arch=120a.

Exit codes:
  0  success (or --allow-empty-kernels zero-match, or --dry-run)
  1  preflight / environment failure
  2  zero kernels matched (without --allow-empty-kernels)
  3  build failure
  4  profile failure
  5  invalid CLI argument
EOF
}

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
require_positive_int() {
  local name="$1" val="$2"
  if [[ ! "$val" =~ ^[1-9][0-9]*$ ]]; then
    die "invalid value for ${name}: '${val}' (expected positive integer)" 5
  fi
}

require_nonneg_int() {
  local name="$1" val="$2"
  if [[ ! "$val" =~ ^(0|[1-9][0-9]*)$ ]]; then
    die "invalid value for ${name}: '${val}' (expected non-negative integer)" 5
  fi
}

# ---------------------------------------------------------------------------
# Flag parser
# ---------------------------------------------------------------------------
parse_flags() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        print_help; exit 0 ;;

      --arch=*)        ARCH="${1#--arch=}"; shift ;;
      --arch)          [[ $# -ge 2 ]] || die "--arch requires a value" 5
                       ARCH="$2"; shift 2 ;;

      --m=*)           M="${1#--m=}"; require_positive_int --m "$M"; shift ;;
      --m)             [[ $# -ge 2 ]] || die "--m requires a value" 5
                       M="$2"; require_positive_int --m "$M"; shift 2 ;;

      --n=*)           N="${1#--n=}"; require_positive_int --n "$N"; shift ;;
      --n)             [[ $# -ge 2 ]] || die "--n requires a value" 5
                       N="$2"; require_positive_int --n "$N"; shift 2 ;;

      --k=*)           K="${1#--k=}"; require_positive_int --k "$K"; shift ;;
      --k)             [[ $# -ge 2 ]] || die "--k requires a value" 5
                       K="$2"; require_positive_int --k "$K"; shift 2 ;;

      --matrix-size=*)
        local v="${1#--matrix-size=}"
        require_positive_int --matrix-size "$v"
        M="$v"; N="$v"; K="$v"
        shift ;;
      --matrix-size)
        [[ $# -ge 2 ]] || die "--matrix-size requires a value" 5
        require_positive_int --matrix-size "$2"
        M="$2"; N="$2"; K="$2"
        shift 2 ;;

      --warmup-iterations=*)
        WARMUP_ITERATIONS="${1#--warmup-iterations=}"
        require_nonneg_int --warmup-iterations "$WARMUP_ITERATIONS"
        shift ;;
      --warmup-iterations)
        [[ $# -ge 2 ]] || die "--warmup-iterations requires a value" 5
        WARMUP_ITERATIONS="$2"
        require_nonneg_int --warmup-iterations "$WARMUP_ITERATIONS"
        shift 2 ;;

      --profiling-iterations=*)
        PROFILING_ITERATIONS="${1#--profiling-iterations=}"
        require_positive_int --profiling-iterations "$PROFILING_ITERATIONS"
        shift ;;
      --profiling-iterations)
        [[ $# -ge 2 ]] || die "--profiling-iterations requires a value" 5
        PROFILING_ITERATIONS="$2"
        require_positive_int --profiling-iterations "$PROFILING_ITERATIONS"
        shift 2 ;;

      --output-dir=*)  OUTPUT_DIR="${1#--output-dir=}"; shift ;;
      --output-dir)    [[ $# -ge 2 ]] || die "--output-dir requires a value" 5
                       OUTPUT_DIR="$2"; shift 2 ;;

      --cutlass-dir=*) CUTLASS_DIR="${1#--cutlass-dir=}"; shift ;;
      --cutlass-dir)   [[ $# -ge 2 ]] || die "--cutlass-dir requires a value" 5
                       CUTLASS_DIR="$2"; shift 2 ;;

      --build-dir=*)   BUILD_DIR="${1#--build-dir=}"; shift ;;
      --build-dir)     [[ $# -ge 2 ]] || die "--build-dir requires a value" 5
                       BUILD_DIR="$2"; shift 2 ;;

      --kernel-filter-bf16=*) KERNEL_FILTER_BF16="${1#--kernel-filter-bf16=}"; shift ;;
      --kernel-filter-bf16)   [[ $# -ge 2 ]] || die "--kernel-filter-bf16 requires a value" 5
                              KERNEL_FILTER_BF16="$2"; shift 2 ;;

      --kernel-filter-fp8=*)  KERNEL_FILTER_FP8="${1#--kernel-filter-fp8=}"; shift ;;
      --kernel-filter-fp8)    [[ $# -ge 2 ]] || die "--kernel-filter-fp8 requires a value" 5
                              KERNEL_FILTER_FP8="$2"; shift 2 ;;

      --jobs=*)
        JOBS="${1#--jobs=}"
        require_positive_int --jobs "$JOBS"
        shift ;;
      --jobs)
        [[ $# -ge 2 ]] || die "--jobs requires a value" 5
        JOBS="$2"
        require_positive_int --jobs "$JOBS"
        shift 2 ;;

      --skip-build)            SKIP_BUILD=true; shift ;;
      --clean)                 DO_CLEAN=true; shift ;;
      --allow-empty-kernels)   ALLOW_EMPTY_KERNELS=true; shift ;;
      --dry-run)               DRY_RUN=true; shift ;;
      --update-memory)         UPDATE_MEMORY=true; shift ;;
      --migrate-tmp)           MIGRATE_TMP=true; shift ;;
      --smoke)                 DO_SMOKE=true; shift ;;
      --no-summary)            NO_SUMMARY=true; shift ;;

      --summary-statistic=*)
        SUMMARY_STATISTIC="${1#--summary-statistic=}"
        case "$SUMMARY_STATISTIC" in
          best|median|both) ;;
          *) die "invalid value for --summary-statistic: '${SUMMARY_STATISTIC}' (expected best|median|both)" 5 ;;
        esac
        shift ;;
      --summary-statistic)
        [[ $# -ge 2 ]] || die "--summary-statistic requires a value" 5
        SUMMARY_STATISTIC="$2"
        case "$SUMMARY_STATISTIC" in
          best|median|both) ;;
          *) die "invalid value for --summary-statistic: '${SUMMARY_STATISTIC}' (expected best|median|both)" 5 ;;
        esac
        shift 2 ;;

      --) shift; break ;;
      -*) die "unknown option: $1 (run with --help for usage)" 5 ;;
      *)  die "unexpected positional argument: $1 (run with --help for usage)" 5 ;;
    esac
  done
}

# ---------------------------------------------------------------------------
# Static validation that does not require any external tool to be installed.
# Runs after parse_flags, before preflight, so misconfigurations short-circuit
# before any cmake / nvcc invocation.
# ---------------------------------------------------------------------------
validate_flags() {
  if [[ "$SKIP_BUILD" == "true" && "$DO_CLEAN" == "true" ]]; then
    die "--skip-build and --clean are mutually exclusive" 5
  fi

  if [[ ! -d "$CUTLASS_DIR" ]]; then
    die "cutlass dir not found: ${CUTLASS_DIR}" 5
  fi

  : "${BUILD_DIR:=${CUTLASS_DIR}/build-sm120-profiler}"

  SUMMARY_DIR="$(dirname "${OUTPUT_DIR}")"
  RUN_TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

# ---------------------------------------------------------------------------
# Preflight: hard-required tools, GPU presence, GPU compute-capability match
# ---------------------------------------------------------------------------
preflight() {
  log_section "preflight"

  local required_tools=(cmake nvcc nvidia-smi python3 jq shellcheck)
  local missing=()

  for tool in "${required_tools[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      missing+=("$tool")
      log_err "${tool} not found on PATH"
    else
      log_info "found ${tool}: $(command -v "$tool")"
    fi
  done

  if command -v ninja >/dev/null 2>&1; then
    BUILD_GENERATOR="ninja"
    log_info "found ninja: $(command -v ninja) (preferred over make)"
  elif command -v make >/dev/null 2>&1; then
    BUILD_GENERATOR="make"
    log_info "found make: $(command -v make) (ninja unavailable; falling back)"
  else
    missing+=("make-or-ninja")
    log_err "neither ninja nor make found on PATH"
  fi

  if (( ${#missing[@]} > 0 )); then
    RESULT_STATUS="preflight_failed"
    die "preflight failed: missing tool(s): ${missing[*]}" 1
  fi

  if ! nvidia-smi -L >/dev/null 2>&1; then
    RESULT_STATUS="preflight_failed"
    die "no CUDA GPU detected (nvidia-smi -L failed or returned empty)" 1
  fi

  local gpu_list cc_observed cc_expected
  gpu_list="$(nvidia-smi -L)"
  if [[ -z "$gpu_list" ]]; then
    RESULT_STATUS="preflight_failed"
    die "no CUDA GPU detected (nvidia-smi -L returned empty)" 1
  fi
  log_info "GPU list: ${gpu_list}"

  cc_observed="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  cc_expected="$(printf '%s' "$ARCH" | sed -E 's/^([0-9]+)([afz]?)$/\1/' | sed -E 's/^([0-9]{2})([0-9])$/\1.\2/; s/^([0-9])([0-9])$/\1.\2/')"

  if [[ -n "$cc_observed" && -n "$cc_expected" && "$cc_observed" != "$cc_expected" ]]; then
    log_warn "GPU compute capability ${cc_observed} does not match --arch=${ARCH} (expected ${cc_expected}); continuing — pass --arch to override"
  fi

  if [[ "$DRY_RUN" != "true" ]]; then
    if ! mkdir -p "${OUTPUT_DIR}" 2>/dev/null; then
      die "output dir not writable or cannot be created: ${OUTPUT_DIR}" 1
    fi
    if [[ "$SKIP_BUILD" != "true" ]] && ! mkdir -p "$(dirname "${BUILD_DIR}")" 2>/dev/null; then
      die "build dir parent not writable: $(dirname "${BUILD_DIR}")" 1
    fi
  fi

  log_ok "preflight ok"
}

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
clean_build_dir() {
  log_section "clean"
  if [[ -d "$BUILD_DIR" ]]; then
    log_info "removing ${BUILD_DIR}"
    rm -rf -- "$BUILD_DIR"
  else
    log_info "build dir does not exist; nothing to clean"
  fi
}

cmake_configure_argv() {
  local generator_name
  if [[ "$BUILD_GENERATOR" == "ninja" ]]; then
    generator_name="Ninja"
  else
    generator_name="Unix Makefiles"
  fi
  printf '%s\n' \
    cmake \
    -S "${CUTLASS_DIR}" \
    -B "${BUILD_DIR}" \
    -G "${generator_name}" \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCUTLASS_NVCC_ARCHS=${ARCH}" \
    "-DCUTLASS_LIBRARY_OPERATIONS=gemm" \
    "-DCUTLASS_LIBRARY_KERNELS=${KERNEL_FILTER_BF16},${KERNEL_FILTER_FP8}"
}

configure_cmake() {
  log_section "configure"
  mkdir -p "${BUILD_DIR}"
  local -a argv
  mapfile -t argv < <(cmake_configure_argv)
  log_info "${argv[*]}"
  "${argv[@]}"
}

# ---------------------------------------------------------------------------
# Auto-discover the kernel manifest produced by cmake configure
# ---------------------------------------------------------------------------
discover_generated_kernels_txt() {
  local primary="${BUILD_DIR}/tools/library/generated_kernels.txt"
  if [[ -f "$primary" ]]; then
    GENERATED_KERNELS_TXT="$primary"
    return 0
  fi

  local found
  found="$(find "${BUILD_DIR}" -type f -name 'generated_kernels.txt' -print 2>/dev/null | head -n1)"
  if [[ -n "$found" ]]; then
    GENERATED_KERNELS_TXT="$found"
    return 0
  fi
  return 1
}

count_kernels_matching() {
  local pattern="$1"
  if [[ -z "$GENERATED_KERNELS_TXT" || ! -f "$GENERATED_KERNELS_TXT" ]]; then
    printf '0\n'
    return 0
  fi
  local glob_to_regex
  glob_to_regex="$(printf '%s' "$pattern" | sed -e 's/[][\.|\^\$+(){}]/\\&/g' -e 's/\*/.*/g' -e 's/?/./g')"
  grep -Ec -- "${glob_to_regex}" "$GENERATED_KERNELS_TXT" || true
}

verify_kernel_filter() {
  log_section "verify"

  if ! discover_generated_kernels_txt; then
    if [[ "$ALLOW_EMPTY_KERNELS" == "true" ]]; then
      log_warn "kernel manifest not found under ${BUILD_DIR}; treating as zero-match under --allow-empty-kernels"
      N_KERNELS_MATCHED_BF16=0
      N_KERNELS_MATCHED_FP8=0
      handle_empty_kernels
      return 0
    fi
    RESULT_STATUS="zero_match"
    handle_empty_kernels
  fi

  log_info "kernel manifest: ${GENERATED_KERNELS_TXT}"
  N_KERNELS_MATCHED_BF16="$(count_kernels_matching "${KERNEL_FILTER_BF16}")"
  N_KERNELS_MATCHED_FP8="$(count_kernels_matching "${KERNEL_FILTER_FP8}")"

  log_info "kernels matched bf16 (${KERNEL_FILTER_BF16}): ${N_KERNELS_MATCHED_BF16}"
  log_info "kernels matched fp8  (${KERNEL_FILTER_FP8}): ${N_KERNELS_MATCHED_FP8}"

  if (( N_KERNELS_MATCHED_BF16 == 0 && N_KERNELS_MATCHED_FP8 == 0 )); then
    handle_empty_kernels
  fi

  log_ok "kernel filter verification passed"
}

handle_empty_kernels() {
  RESULT_STATUS="zero_match"
  if [[ "$ALLOW_EMPTY_KERNELS" == "true" ]]; then
    log_warn "0 kernels matched; continuing under --allow-empty-kernels"
    write_manifest
    final_summary_echo
    exit 0
  fi
  log_err "0 kernels matched; aborting before build (use --allow-empty-kernels to downgrade)"
  write_manifest
  exit 2
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
build_profiler() {
  log_section "build"
  local -a argv=(cmake --build "${BUILD_DIR}" --target cutlass_profiler --parallel "${JOBS}")
  log_info "${argv[*]}"
  if "${argv[@]}"; then
    BUILD_EXIT_CODE=0
    log_ok "build complete"
  else
    BUILD_EXIT_CODE=$?
    RESULT_STATUS="build_failed"
    write_manifest
    die "cutlass_profiler build failed (exit ${BUILD_EXIT_CODE})" 3
  fi
}

cutlass_profiler_path() {
  printf '%s/tools/profiler/cutlass_profiler\n' "${BUILD_DIR}"
}

require_profiler_binary() {
  local p
  p="$(cutlass_profiler_path)"
  if [[ ! -x "$p" ]]; then
    die "cannot --skip-build: cutlass_profiler binary not found at ${p}" 3
  fi
}

# ---------------------------------------------------------------------------
# Dry-run kernel enumeration
# ---------------------------------------------------------------------------
enumerate_kernels() {
  local precision="$1"
  local pattern="" outfile="${OUTPUT_DIR}/dryrun_kernels_${precision}.txt"
  case "$precision" in
    bf16) pattern="$KERNEL_FILTER_BF16" ;;
    fp8)  pattern="$KERNEL_FILTER_FP8" ;;
    *) die "internal: enumerate_kernels called with unknown precision '${precision}'" ;;
  esac

  log_section "enumerate-${precision}"

  if [[ -z "$GENERATED_KERNELS_TXT" || ! -f "$GENERATED_KERNELS_TXT" ]]; then
    discover_generated_kernels_txt || true
  fi
  if [[ -z "$GENERATED_KERNELS_TXT" || ! -f "$GENERATED_KERNELS_TXT" ]]; then
    log_warn "${precision}: kernel manifest not found under ${BUILD_DIR}; cannot enumerate"
    : > "${outfile}"
    case "$precision" in
      bf16) DRYRUN_KERNEL_NAMES_BF16="" ;;
      fp8)  DRYRUN_KERNEL_NAMES_FP8=""  ;;
    esac
    return 0
  fi

  # cutlass_profiler --mode=dry_run / --mode=enumerate emit no output for Gemm
  # in v4.4.1, so source concrete names from the cmake-emitted manifest the
  # configure step already produced. Same data, different source.
  local glob_to_regex
  glob_to_regex="$(printf '%s' "$pattern" | sed -e 's/[][\.|\^\$+(){}]/\\&/g' -e 's/\*/.*/g' -e 's/?/./g')"
  log_info "matching ${pattern} against ${GENERATED_KERNELS_TXT}"

  local names
  names="$(grep -E -- "${glob_to_regex}" "${GENERATED_KERNELS_TXT}" || true)"
  printf '%s\n' "${names}" > "${outfile}"

  case "$precision" in
    bf16) DRYRUN_KERNEL_NAMES_BF16="$names" ;;
    fp8)  DRYRUN_KERNEL_NAMES_FP8="$names"  ;;
  esac

  local count
  count="$(printf '%s\n' "${names}" | grep -cE '.' || true)"
  log_info "${precision}: enumerated ${count} concrete kernel name(s) into ${outfile}"
}

# ---------------------------------------------------------------------------
# nvidia-smi snapshot for the manifest
# ---------------------------------------------------------------------------
capture_gpu_state() {
  local q
  q="$(nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
  if [[ -z "$q" ]]; then
    printf '%s' '{"sm_clock_mhz":"unknown","mem_clock_mhz":"unknown","temp_c":"unknown","power_w":"unknown"}'
    return 0
  fi
  local sm_clk mem_clk temp pwr
  IFS=',' read -r sm_clk mem_clk temp pwr <<<"$q"
  jq -n \
    --arg sm "$sm_clk" \
    --arg mc "$mem_clk" \
    --arg tc "$temp" \
    --arg pw "$pwr" \
    '{sm_clock_mhz:$sm, mem_clock_mhz:$mc, temp_c:$tc, power_w:$pw}'
}

# ---------------------------------------------------------------------------
# Timed profile for one precision
# ---------------------------------------------------------------------------
run_profiler() {
  local precision="$1" dim="$2" warmup="$3" iters="$4"
  local csv="${OUTPUT_DIR}/cutlass_profiler_${precision}_${dim}.csv"
  local log="${OUTPUT_DIR}/cutlass_profiler_${precision}_${dim}.log"
  # cutlass_profiler appends `.<operation>.csv` to whatever --output= path is
  # passed, so we hand it the bare prefix and rename the resulting file
  # afterwards to match the AC-4 expected `.csv` filename.
  local csv_base="${csv%.csv}"
  local csv_actual="${csv_base}.gemm.csv"
  local names_file="${OUTPUT_DIR}/dryrun_kernels_${precision}.txt"

  log_section "profile-${precision}-${dim}"

  local names
  case "$precision" in
    bf16) names="$DRYRUN_KERNEL_NAMES_BF16" ;;
    fp8)  names="$DRYRUN_KERNEL_NAMES_FP8"  ;;
    *) die "internal: run_profiler called with unknown precision '${precision}'" ;;
  esac

  if [[ -z "$names" ]]; then
    log_warn "0 matching kernels; nothing to profile for ${precision}"
    {
      printf '0 matching kernels; nothing to profile for %s\n' "$precision"
      printf '\n--- enumerated kernel list ---\n'
      cat "${names_file}" 2>/dev/null || true
    } > "$log"
    case "$precision" in
      bf16) PROFILE_BF16_EXIT_CODE=0 ;;
      fp8)  PROFILE_FP8_EXIT_CODE=0  ;;
    esac
    return 0
  fi

  local pre post
  pre="$(capture_gpu_state)"
  case "$precision" in
    bf16) BF16_PRE_STATE="$pre" ;;
    fp8)  FP8_PRE_STATE="$pre"  ;;
  esac

  local profiler exit_code=0
  profiler="$(cutlass_profiler_path)"

  # Clean any prior per-operation CSVs from a previous run with this prefix.
  while IFS= read -r f; do rm -f -- "$f"; done < <(find "$(dirname "${csv_base}")" -maxdepth 1 -type f -name "$(basename "${csv_base}").*.csv" -print 2>/dev/null)
  rm -f -- "${csv}"

  # No --operation: v4.4.1 sm_120 BF16 kernels are blockwise_gemm operations,
  # not Gemm. Letting the profiler match across all operation kinds writes one
  # CSV per kind; we then pick the populated one as the canonical output.
  #
  # --verification-enabled=false: the default verification path runs a cuBLAS
  # reference matmul per kernel. On RTX 5060 cuBLAS BF16 falls back to an
  # sm_80 kernel at ~28 TFLOPS, so verifying 98 kernels at 8192^3 takes
  # >1 hour. We're cross-checking achieved TFLOPS, not correctness, so the
  # profiler's own warmup+timing loop is sufficient.
  local -a argv=(
    "${profiler}"
    "--m=${dim}"
    "--n=${dim}"
    "--k=${dim}"
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
    log_warn "profiler ${precision} exited ${exit_code}; see ${log}"
  fi

  post="$(capture_gpu_state)"
  case "$precision" in
    bf16) BF16_POST_STATE="$post"; PROFILE_BF16_EXIT_CODE=$exit_code ;;
    fp8)  FP8_POST_STATE="$post";  PROFILE_FP8_EXIT_CODE=$exit_code  ;;
  esac

  local -a per_op=()
  while IFS= read -r f; do per_op+=("$f"); done < <(find "$(dirname "${csv_base}")" -maxdepth 1 -type f -name "$(basename "${csv_base}").*.csv" -print 2>/dev/null | sort)

  local primary="" primary_rows=0 op_name=""
  for f in "${per_op[@]}"; do
    local rows
    rows=$(($(wc -l < "$f") - 1))
    if (( rows > primary_rows )); then
      primary="$f"
      primary_rows=$rows
    fi
  done

  if [[ -n "$primary" ]]; then
    op_name="$(basename "$primary")"
    op_name="${op_name#$(basename "${csv_base}").}"
    op_name="${op_name%.csv}"
    mv -- "$primary" "$csv"
    log_ok "${precision} ${dim}^3 csv -> ${csv} (${primary_rows} data row(s) from operation '${op_name}')"
  fi

  # Drop empty per-op leftovers; keep any that have data alongside the primary
  for f in "${per_op[@]}"; do
    [[ -f "$f" ]] || continue
    if (( $(wc -l < "$f") <= 1 )); then
      rm -f -- "$f"
    fi
  done

  if [[ ! -f "$csv" ]]; then
    log_warn "${precision} ${dim}^3 csv missing — see ${log}"
  fi
}

run_profiler_pair_full() {
  run_profiler bf16 "$M" "$WARMUP_ITERATIONS" "$PROFILING_ITERATIONS"
  run_profiler fp8  "$M" "$WARMUP_ITERATIONS" "$PROFILING_ITERATIONS"
}

run_profiler_pair_smoke() {
  log_section "smoke"
  local prev_bf16=$PROFILE_BF16_EXIT_CODE
  local prev_fp8=$PROFILE_FP8_EXIT_CODE
  run_profiler bf16 1024 2 3
  run_profiler fp8  1024 2 3
  local smoke_bf16=$PROFILE_BF16_EXIT_CODE
  local smoke_fp8=$PROFILE_FP8_EXIT_CODE
  PROFILE_BF16_EXIT_CODE=$prev_bf16
  PROFILE_FP8_EXIT_CODE=$prev_fp8
  if (( smoke_bf16 != 0 )) || (( smoke_fp8 != 0 )); then
    die "smoke run failed (bf16=${smoke_bf16}, fp8=${smoke_fp8}); aborting before ${M}^3 runs" 4
  fi
  log_ok "smoke run passed"
}

# ---------------------------------------------------------------------------
# Heuristic classification of result_status from concrete kernel names
# ---------------------------------------------------------------------------
classify_result_status() {
  if [[ "$RESULT_STATUS" == "preflight_failed" || "$RESULT_STATUS" == "build_failed" || "$RESULT_STATUS" == "zero_match" ]]; then
    return
  fi

  local has_dense=false has_blockscaled=false combined
  combined="$(printf '%s\n%s\n' "$DRYRUN_KERNEL_NAMES_BF16" "$DRYRUN_KERNEL_NAMES_FP8")"

  while IFS= read -r kname; do
    [[ -z "$kname" ]] && continue
    local low_prec=false out_bf16=false
    case "$kname" in
      *blockscaled*|*blockwise*|*nvf4*|*nvfp4*|*mxf8*|*mxfp8*|*mxf6*|*mxf4*|*e4m3*|*e5m2*|*e2m1*|*e3m2*)
        low_prec=true ;;
    esac
    case "$kname" in
      *bf16*|*fp16*|*_f16_*|*_f32_*)
        out_bf16=true ;;
    esac
    if [[ "$low_prec" == "true" && "$out_bf16" == "true" ]]; then
      has_blockscaled=true
    elif [[ "$kname" == *blockscaled* || "$kname" == *blockwise* || "$kname" == *nvfp4* || "$kname" == *nvf4* ]]; then
      has_blockscaled=true
    else
      has_dense=true
    fi
  done <<<"$combined"

  if [[ "$has_dense" == "true" ]]; then
    RESULT_STATUS="dense_found"
  elif [[ "$has_blockscaled" == "true" ]]; then
    RESULT_STATUS="blockscaled_only"
  else
    RESULT_STATUS="zero_match"
  fi
}

# ---------------------------------------------------------------------------
# Manifest emission (mandatory)
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

  local bf16_csv="${OUTPUT_DIR}/cutlass_profiler_bf16_${M}.csv"
  local fp8_csv="${OUTPUT_DIR}/cutlass_profiler_fp8_${M}.csv"
  local bf16_log="${OUTPUT_DIR}/cutlass_profiler_bf16_${M}.log"
  local fp8_log="${OUTPUT_DIR}/cutlass_profiler_fp8_${M}.log"
  local summary_path="${SUMMARY_DIR}/SUMMARY.md"

  local bf16_names_json fp8_names_json
  bf16_names_json="$(printf '%s' "$DRYRUN_KERNEL_NAMES_BF16" | jq -R -s 'split("\n") | map(select(length > 0))')"
  fp8_names_json="$(printf '%s' "$DRYRUN_KERNEL_NAMES_FP8" | jq -R -s 'split("\n") | map(select(length > 0))')"

  local cublaslt_default="${SOLAR_ROOT}/measurements/rtx5060/cublaslt"
  local wmma_default="${SOLAR_ROOT}/measurements/rtx5060/wmma"
  local cublaslt_json='[]' wmma_json='[]'
  if [[ -d "$cublaslt_default" ]]; then
    cublaslt_json="$(find "$cublaslt_default" -type f -name '*.csv' -print 2>/dev/null | jq -R -s 'split("\n") | map(select(length > 0))')"
  elif [[ -d /tmp/solar-rtx5060 ]]; then
    cublaslt_json="$(find /tmp/solar-rtx5060 -maxdepth 1 -type f -name 'cublaslt_*.csv' -print 2>/dev/null | jq -R -s 'split("\n") | map(select(length > 0))')"
  fi
  if [[ -d "$wmma_default" ]]; then
    wmma_json="$(find "$wmma_default" -type f -name '*.csv' -print 2>/dev/null | jq -R -s 'split("\n") | map(select(length > 0))')"
  elif [[ -d /tmp/solar-rtx5060 ]]; then
    wmma_json="$(find /tmp/solar-rtx5060 -maxdepth 1 -type f -name 'ncu_hmma_fp16*.csv' -print 2>/dev/null | jq -R -s 'split("\n") | map(select(length > 0))')"
  fi

  jq -n \
    --arg timestamp "$RUN_TIMESTAMP" \
    --arg host "$host" \
    --arg gpu_name "$gpu_name" \
    --arg gpu_pci_id "$gpu_pci" \
    --arg compute_capability "$cc" \
    --arg driver_version "$driver_v" \
    --arg cuda_version "$cuda_v" \
    --arg nvcc_version "$nvcc_v" \
    --arg cmake_version "$cmake_v" \
    --arg build_generator "$BUILD_GENERATOR" \
    --arg cutlass_commit "$cutlass_commit" \
    --arg cutlass_dir "$CUTLASS_DIR" \
    --arg build_dir "$BUILD_DIR" \
    --arg kernel_filter_bf16 "$KERNEL_FILTER_BF16" \
    --arg kernel_filter_fp8 "$KERNEL_FILTER_FP8" \
    --argjson n_kernels_matched_bf16 "$N_KERNELS_MATCHED_BF16" \
    --argjson n_kernels_matched_fp8 "$N_KERNELS_MATCHED_FP8" \
    --argjson dryrun_kernel_names_bf16 "$bf16_names_json" \
    --argjson dryrun_kernel_names_fp8 "$fp8_names_json" \
    --argjson m "$M" --argjson n "$N" --argjson k "$K" \
    --argjson warmup_iterations "$WARMUP_ITERATIONS" \
    --argjson profiling_iterations "$PROFILING_ITERATIONS" \
    --argjson jobs "$JOBS" \
    --arg result_status "$RESULT_STATUS" \
    --argjson bf16_pre_state "$BF16_PRE_STATE" \
    --argjson bf16_post_state "$BF16_POST_STATE" \
    --argjson fp8_pre_state "$FP8_PRE_STATE" \
    --argjson fp8_post_state "$FP8_POST_STATE" \
    --argjson build_exit_code "$BUILD_EXIT_CODE" \
    --argjson profile_bf16_exit_code "$PROFILE_BF16_EXIT_CODE" \
    --argjson profile_fp8_exit_code "$PROFILE_FP8_EXIT_CODE" \
    --arg bf16_csv "$bf16_csv" \
    --arg fp8_csv "$fp8_csv" \
    --arg bf16_log "$bf16_log" \
    --arg fp8_log "$fp8_log" \
    --arg summary_path "$summary_path" \
    --argjson cublaslt_source_csvs "$cublaslt_json" \
    --argjson wmma_source_csvs "$wmma_json" \
    '{
       timestamp: $timestamp,
       host: $host,
       gpu_name: $gpu_name,
       gpu_pci_id: $gpu_pci_id,
       compute_capability: $compute_capability,
       driver_version: $driver_version,
       cuda_version: $cuda_version,
       nvcc_version: $nvcc_version,
       cmake_version: $cmake_version,
       build_generator: $build_generator,
       cutlass_commit: $cutlass_commit,
       cutlass_dir: $cutlass_dir,
       build_dir: $build_dir,
       kernel_filter_bf16: $kernel_filter_bf16,
       kernel_filter_fp8: $kernel_filter_fp8,
       n_kernels_matched_bf16: $n_kernels_matched_bf16,
       n_kernels_matched_fp8: $n_kernels_matched_fp8,
       dryrun_kernel_names_bf16: $dryrun_kernel_names_bf16,
       dryrun_kernel_names_fp8: $dryrun_kernel_names_fp8,
       matrix_size: { m: $m, n: $n, k: $k },
       warmup_iterations: $warmup_iterations,
       profiling_iterations: $profiling_iterations,
       jobs: $jobs,
       result_status: $result_status,
       bf16_pre_state: $bf16_pre_state,
       bf16_post_state: $bf16_post_state,
       fp8_pre_state: $fp8_pre_state,
       fp8_post_state: $fp8_post_state,
       build_exit_code: $build_exit_code,
       profile_bf16_exit_code: $profile_bf16_exit_code,
       profile_fp8_exit_code: $profile_fp8_exit_code,
       csv_paths: { bf16: $bf16_csv, fp8: $fp8_csv },
       log_paths: { bf16: $bf16_log, fp8: $fp8_log },
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
# Summary emission
# ---------------------------------------------------------------------------
emit_summary() {
  if [[ "$NO_SUMMARY" == "true" ]]; then
    log_info "skipping SUMMARY.md emission (--no-summary)"
    return 0
  fi
  log_section "tabulate"
  mkdir -p "${SUMMARY_DIR}"
  local summary_path="${SUMMARY_DIR}/SUMMARY.md"

  local cublaslt_dir="${SOLAR_ROOT}/measurements/rtx5060/cublaslt"
  local cublaslt_bf16="" cublaslt_fp8="" tmp_warning=""
  if [[ -f "${cublaslt_dir}/cublaslt_bf16_${M}.csv" ]]; then
    cublaslt_bf16="${cublaslt_dir}/cublaslt_bf16_${M}.csv"
  elif [[ -f "/tmp/solar-rtx5060/cublaslt_bf16_${M}.csv" ]]; then
    cublaslt_bf16="/tmp/solar-rtx5060/cublaslt_bf16_${M}.csv"
    tmp_warning="cuBLASLt BF16 source taken from /tmp; pass --migrate-tmp to copy into measurements/"
  fi
  if [[ -f "${cublaslt_dir}/cublaslt_fp8_${M}.csv" ]]; then
    cublaslt_fp8="${cublaslt_dir}/cublaslt_fp8_${M}.csv"
  elif [[ -f "/tmp/solar-rtx5060/cublaslt_fp8_${M}.csv" ]]; then
    cublaslt_fp8="/tmp/solar-rtx5060/cublaslt_fp8_${M}.csv"
    tmp_warning="cuBLASLt sources taken from /tmp; pass --migrate-tmp to copy into measurements/"
  fi

  local bf16_csv="${OUTPUT_DIR}/cutlass_profiler_bf16_${M}.csv"
  local fp8_csv="${OUTPUT_DIR}/cutlass_profiler_fp8_${M}.csv"
  local arch_yaml="${SOLAR_ROOT}/configs/arch/5060.yaml"

  python3 - "$summary_path" "$arch_yaml" "$bf16_csv" "$fp8_csv" "$cublaslt_bf16" "$cublaslt_fp8" "$SUMMARY_STATISTIC" "$RESULT_STATUS" "$RUN_TIMESTAMP" "${OUTPUT_DIR}/cutlass_profiler_bf16_${M}.log" "${OUTPUT_DIR}/cutlass_profiler_fp8_${M}.log" "$tmp_warning" <<'PYEOF'
import csv, os, statistics, sys
from pathlib import Path

(summary_path, arch_yaml, bf16_csv, fp8_csv, cublaslt_bf16, cublaslt_fp8,
 stat_mode, result_status, run_ts, bf16_log, fp8_log, tmp_warning) = sys.argv[1:13]

def load_arch_peaks(p):
    bf16_tflops = fp8_tflops = freq = None
    bf16_mac = fp8_mac = None
    if not Path(p).is_file():
        return bf16_tflops, fp8_tflops
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
            bf16_mac = num
        elif k == 'MAC_per_cycle_fp8_tc':
            fp8_mac = num
    if freq is not None:
        if bf16_mac is not None:
            bf16_tflops = bf16_mac * 2 * freq / 1000.0
        if fp8_mac is not None:
            fp8_tflops = fp8_mac * 2 * freq / 1000.0
    return bf16_tflops, fp8_tflops

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

def split_dense_blockscaled(path):
    if not path or not Path(path).is_file():
        return None, None
    dense_rows, blocks_rows = [], []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tflops = extract_tflops(row)
            if tflops is None:
                continue
            kname = row.get('Operation') or row.get('Kernel') or ''
            kname_l = kname.lower()
            low_prec_tokens = ('blockscaled', 'blockwise', 'nvfp4', 'nvf4',
                               'mxfp8', 'mxf8', 'mxfp6', 'mxf6', 'mxfp4', 'mxf4',
                               'e4m3', 'e5m2', 'e2m1', 'e3m2')
            mixed_out_tokens = ('bf16', 'fp16', 'f16')
            has_low = any(tok in kname_l for tok in low_prec_tokens)
            has_mixed = any(tok in kname_l for tok in mixed_out_tokens)
            explicit_block = any(tok in kname_l for tok in
                                 ('blockscaled', 'blockwise', 'nvfp4', 'nvf4'))
            is_blocked = explicit_block or (has_low and has_mixed)
            (blocks_rows if is_blocked else dense_rows).append(tflops)
    def summarize(xs):
        if not xs:
            return None
        xs.sort(reverse=True)
        return {'best': xs[0], 'median': statistics.median(xs), 'count': len(xs)}
    return summarize(dense_rows), summarize(blocks_rows)

bf16_peak, fp8_peak = load_arch_peaks(arch_yaml)
cublaslt_bf16_stats = parse_csv_tflops(cublaslt_bf16)
cublaslt_fp8_stats = parse_csv_tflops(cublaslt_fp8)
cutlass_bf16_dense, cutlass_bf16_blocks = split_dense_blockscaled(bf16_csv)
cutlass_fp8_dense, cutlass_fp8_blocks = split_dense_blockscaled(fp8_csv)

def fmt(stats, key, missing_link):
    if stats is None or stats.get(key) is None:
        return f"N/A — see [log]({missing_link})"
    return f"{stats[key]:.2f}"

show_best = stat_mode in ('best', 'both')
show_median = stat_mode in ('median', 'both')

def row(name, precision_triplet, bf16_stats, fp8_stats, source_link, missing_link):
    cells = [name, precision_triplet]
    if show_best:
        cells.append(fmt(bf16_stats, 'best', missing_link) if bf16_stats else "N/A")
        cells.append(fmt(fp8_stats, 'best', missing_link) if fp8_stats else "N/A")
    if show_median:
        cells.append(fmt(bf16_stats, 'median', missing_link) if bf16_stats else "N/A")
        cells.append(fmt(fp8_stats, 'median', missing_link) if fp8_stats else "N/A")
    cells.append(source_link)
    return '| ' + ' | '.join(cells) + ' |'

header_cells = ['Source', 'Precision triplet']
if show_best:
    header_cells += ['BF16 best TFLOPS', 'FP8 best TFLOPS']
if show_median:
    header_cells += ['BF16 median TFLOPS', 'FP8 median TFLOPS']
header_cells += ['Source artifact']

separator_cells = ['---'] * len(header_cells)

lines = []
lines.append(f"# RTX 5060 Laptop GEMM cross-check (run: {run_ts})")
lines.append("")
lines.append(f"Result status: `{result_status}`")
lines.append("")
lines.append('| ' + ' | '.join(header_cells) + ' |')
lines.append('| ' + ' | '.join(separator_cells) + ' |')

# Spec peak
spec_link = '[configs/arch/5060.yaml](../../configs/arch/5060.yaml)'
lines.append(row(
    'Spec peak',
    'bf16/bf16 dense, fp8/fp8 dense',
    {'best': bf16_peak, 'median': bf16_peak} if bf16_peak else None,
    {'best': fp8_peak, 'median': fp8_peak} if fp8_peak else None,
    spec_link,
    spec_link,
))

# cuBLASLt achieved
cublaslt_link_bf16 = f'[{cublaslt_bf16}]({cublaslt_bf16})' if cublaslt_bf16 else 'N/A'
cublaslt_link_fp8  = f'[{cublaslt_fp8}]({cublaslt_fp8})'  if cublaslt_fp8 else 'N/A'
cublaslt_link = ', '.join(x for x in (cublaslt_link_bf16, cublaslt_link_fp8) if x != 'N/A') or 'N/A'
lines.append(row(
    'cuBLASLt achieved',
    'bf16/bf16 dense, fp8/fp8 dense',
    cublaslt_bf16_stats,
    cublaslt_fp8_stats,
    cublaslt_link,
    cublaslt_link,
))

# CUTLASS dense
cutlass_link_bf16 = f'[cutlass_profiler_bf16]({bf16_csv})'
cutlass_link_fp8  = f'[cutlass_profiler_fp8]({fp8_csv})'
cutlass_link = f'{cutlass_link_bf16}, {cutlass_link_fp8}'
lines.append(row(
    'CUTLASS dense',
    'bf16/bf16/fp32_acc, fp8_e4m3/fp8_e4m3/fp32_acc',
    cutlass_bf16_dense,
    cutlass_fp8_dense,
    cutlass_link,
    f'[bf16 log]({bf16_log})',
))

# CUTLASS blockscaled (separate row per DEC-1)
lines.append(row(
    'CUTLASS blockscaled',
    'nvfp4_in/bf16_out/fp32_acc, mxfp8_in/bf16_out/fp32_acc (representative)',
    cutlass_bf16_blocks,
    cutlass_fp8_blocks,
    cutlass_link,
    f'[bf16 log]({bf16_log})',
))

lines.append('')
lines.append('## Notes')
lines.append('')
lines.append('- The CUTLASS blockscaled row groups kernels whose input precision differs from output precision (e.g. NVFP4-in / BF16-out). Per the CUTLASS v4.4.1 sm_120 profiler-library coverage, the dense BF16-in / BF16-out variant is not present, so the blockscaled row is *not directly comparable to dense BF16*.')
lines.append('- TFLOPS values are computed from the profiler `GFLOPs` column divided by 1000.')
if tmp_warning:
    lines.append(f'- {tmp_warning}')
lines.append('')

Path(summary_path).write_text('\n'.join(lines))
PYEOF

  log_ok "summary -> ${summary_path}"
}

# ---------------------------------------------------------------------------
# Memory updates (gated by --update-memory)
# ---------------------------------------------------------------------------
update_memory() {
  log_section "update-memory"
  local memory_file="${SOLAR_ROOT}/.claude/memory/rtx5060_probe_recipe.md"
  if [[ ! -f "$memory_file" ]]; then
    die "memory file not found: ${memory_file}" 1
  fi

  local heading="## CUTLASS v4.4.1 cross-check (run: ${RUN_TIMESTAMP})"
  if grep -F -q -- "$heading" "$memory_file"; then
    log_warn "subsection with timestamp ${RUN_TIMESTAMP} already exists — replacing in place"
    python3 - "$memory_file" "$heading" "$RESULT_STATUS" "$N_KERNELS_MATCHED_BF16" "$N_KERNELS_MATCHED_FP8" "$KERNEL_FILTER_BF16" "$KERNEL_FILTER_FP8" "${OUTPUT_DIR}/manifest.json" <<'PYEOF'
import re, sys
from pathlib import Path
mem, heading, status, n_bf16, n_fp8, filt_bf16, filt_fp8, manifest = sys.argv[1:9]
text = Path(mem).read_text()
new_block = (
    f"{heading}\n\n"
    f"- Kernel filter BF16: `{filt_bf16}` matched {n_bf16} kernel(s)\n"
    f"- Kernel filter FP8: `{filt_fp8}` matched {n_fp8} kernel(s)\n"
    f"- Result status: `{status}`\n"
    f"- Manifest: `{manifest}`\n"
)
pattern = re.compile(rf"({re.escape(heading)})\n.*?(?=\n## |\Z)", re.DOTALL)
text = pattern.sub(new_block.rstrip() + "\n", text)
Path(mem).write_text(text)
PYEOF
  else
    {
      printf '\n%s\n\n' "$heading"
      printf -- '- Kernel filter BF16: `%s` matched %s kernel(s)\n' "$KERNEL_FILTER_BF16" "$N_KERNELS_MATCHED_BF16"
      printf -- '- Kernel filter FP8: `%s` matched %s kernel(s)\n'  "$KERNEL_FILTER_FP8"  "$N_KERNELS_MATCHED_FP8"
      printf -- '- Result status: `%s`\n' "$RESULT_STATUS"
      printf -- '- Manifest: `%s`\n' "${OUTPUT_DIR}/manifest.json"
    } >> "$memory_file"
  fi
  log_ok "appended/updated CUTLASS subsection in ${memory_file}"

  resync_claude_md
}

resync_claude_md() {
  local claude_md="${SOLAR_ROOT}/CLAUDE.md"
  local memory_dir="${SOLAR_ROOT}/.claude/memory"
  if [[ ! -f "$claude_md" ]]; then
    log_warn "CLAUDE.md not found at ${claude_md}; skipping re-sync"
    return 0
  fi
  python3 - "$claude_md" "$memory_dir" <<'PYEOF'
import re, sys
from pathlib import Path

claude_md, memory_dir = sys.argv[1:3]
md = Path(claude_md)
mem = Path(memory_dir)
text = md.read_text()

source_re = re.compile(r"Source: \[\.claude/memory/([^\]]+)\]\([^)]+\)")
sections = re.split(r"\n---\n", text)

new_sections = []
for sec in sections:
    m = source_re.search(sec)
    if not m:
        new_sections.append(sec)
        continue
    fname = m.group(1)
    fpath = mem / fname
    if not fpath.is_file():
        new_sections.append(sec)
        continue
    body = fpath.read_text()
    # Strip YAML frontmatter (`---\n…\n---\n`) so its delimiters don't get
    # interpreted as CLAUDE.md section separators on subsequent re-syncs.
    if body.startswith('---\n'):
        end = body.find('\n---\n', 4)
        if end > 0:
            body = body[end + 5:]
    body = body.lstrip('\n').rstrip()
    head_match = re.search(r"(.*?Source: \[[^\]]+\]\([^)]+\))", sec, re.DOTALL)
    if head_match:
        head = head_match.group(1)
        new_sections.append(f"{head}\n\n{body}")
    else:
        new_sections.append(sec)

md.write_text("\n---\n".join(new_sections))
PYEOF
  log_ok "re-synced CLAUDE.md from ${memory_dir}"
}

# ---------------------------------------------------------------------------
# Migrate /tmp scratch artifacts
# ---------------------------------------------------------------------------
migrate_tmp_artifacts() {
  log_section "migrate-tmp"
  local tmp="/tmp/solar-rtx5060"
  local cublaslt_dest="${SOLAR_ROOT}/measurements/rtx5060/cublaslt"
  local wmma_dest="${SOLAR_ROOT}/measurements/rtx5060/wmma"

  if [[ ! -d "$tmp" ]]; then
    log_warn "no source dir at ${tmp}; nothing to migrate"
    return 0
  fi

  mkdir -p "${cublaslt_dest}" "${wmma_dest}"

  local copied=0
  while IFS= read -r -d '' f; do
    cp -n -- "$f" "${cublaslt_dest}/"
    copied=$((copied + 1))
  done < <(find "$tmp" -maxdepth 1 -type f \( -name 'cublaslt_*.csv' -o -name 'cublaslt_*.log' -o -name 'cublaslt_*.json' \) -print0 2>/dev/null)

  while IFS= read -r -d '' f; do
    cp -n -- "$f" "${wmma_dest}/"
    copied=$((copied + 1))
  done < <(find "$tmp" -maxdepth 1 -type f -name 'ncu_hmma_fp16*.csv' -print0 2>/dev/null)

  log_ok "migrated ${copied} file(s) from ${tmp}"
}

# ---------------------------------------------------------------------------
# --dry-run printer
# ---------------------------------------------------------------------------
print_dry_run() {
  log_section "dry-run"

  log_info "resolved arguments:"
  log_info "  arch=${ARCH}"
  log_info "  matrix=${M}x${N}x${K}"
  log_info "  warmup_iterations=${WARMUP_ITERATIONS}"
  log_info "  profiling_iterations=${PROFILING_ITERATIONS}"
  log_info "  output_dir=${OUTPUT_DIR}"
  log_info "  cutlass_dir=${CUTLASS_DIR}"
  log_info "  build_dir=${BUILD_DIR}"
  log_info "  kernel_filter_bf16=${KERNEL_FILTER_BF16}"
  log_info "  kernel_filter_fp8=${KERNEL_FILTER_FP8}"
  log_info "  jobs=${JOBS}"
  log_info "  build_generator=${BUILD_GENERATOR}"
  log_info "  skip_build=${SKIP_BUILD}, clean=${DO_CLEAN}"
  log_info "  smoke=${DO_SMOKE}, no_summary=${NO_SUMMARY}"
  log_info "  update_memory=${UPDATE_MEMORY}, migrate_tmp=${MIGRATE_TMP}"
  log_info "  summary_statistic=${SUMMARY_STATISTIC}, allow_empty_kernels=${ALLOW_EMPTY_KERNELS}"
  log_info ""

  if [[ "$DO_CLEAN" == "true" ]]; then
    log_info "would: rm -rf ${BUILD_DIR}"
  fi
  if [[ "$SKIP_BUILD" != "true" ]]; then
    local -a cargv
    mapfile -t cargv < <(cmake_configure_argv)
    log_info "would: ${cargv[*]}"
    log_info "would: cmake --build ${BUILD_DIR} --target cutlass_profiler --parallel ${JOBS}"
  else
    log_info "would: skip configure / build (--skip-build)"
  fi

  local profiler
  profiler="$(cutlass_profiler_path)"
  log_info "would: ${profiler} --mode=dry_run --operation=Gemm --kernels='${KERNEL_FILTER_BF16}'"
  log_info "would: ${profiler} --mode=dry_run --operation=Gemm --kernels='${KERNEL_FILTER_FP8}'"

  if [[ "$DO_SMOKE" == "true" ]]; then
    log_info "would: ${profiler} --operation=Gemm --m=1024 --n=1024 --k=1024 --warmup-iterations=2 --profiling-iterations=3 --kernels=<concrete-bf16-names>"
    log_info "would: ${profiler} --operation=Gemm --m=1024 --n=1024 --k=1024 --warmup-iterations=2 --profiling-iterations=3 --kernels=<concrete-fp8-names>"
  fi

  log_info "would: ${profiler} --operation=Gemm --m=${M} --n=${N} --k=${K} --warmup-iterations=${WARMUP_ITERATIONS} --profiling-iterations=${PROFILING_ITERATIONS} --kernels=<concrete-bf16-names> --output=${OUTPUT_DIR}/cutlass_profiler_bf16_${M}.csv"
  log_info "would: ${profiler} --operation=Gemm --m=${M} --n=${N} --k=${K} --warmup-iterations=${WARMUP_ITERATIONS} --profiling-iterations=${PROFILING_ITERATIONS} --kernels=<concrete-fp8-names> --output=${OUTPUT_DIR}/cutlass_profiler_fp8_${M}.csv"

  if [[ "$NO_SUMMARY" != "true" ]]; then
    log_info "would: write ${SUMMARY_DIR}/SUMMARY.md"
  fi
  log_info "would: write ${OUTPUT_DIR}/manifest.json"

  if [[ "$UPDATE_MEMORY" == "true" ]]; then
    log_info "would: append CUTLASS subsection to ${SOLAR_ROOT}/.claude/memory/rtx5060_probe_recipe.md and re-sync ${SOLAR_ROOT}/CLAUDE.md"
  fi
  if [[ "$MIGRATE_TMP" == "true" ]]; then
    log_info "would: copy /tmp/solar-rtx5060/cublaslt_*.csv into ${SOLAR_ROOT}/measurements/rtx5060/cublaslt/ and ncu_hmma_fp16*.csv into ${SOLAR_ROOT}/measurements/rtx5060/wmma/"
  fi
  log_ok "dry-run complete (no commands executed)"
}

# ---------------------------------------------------------------------------
# Final stdout summary box
# ---------------------------------------------------------------------------
final_summary_echo() {
  printf '\n%s===============================================%s\n' "${C_GREEN}" "${C_NC}"
  printf '%s   bench_cutlass_5060.sh — final summary       %s\n' "${C_GREEN}" "${C_NC}"
  printf '%s===============================================%s\n' "${C_GREEN}" "${C_NC}"
  printf 'Result status:        %s\n' "${RESULT_STATUS}"
  printf 'Build exit code:      %s\n' "${BUILD_EXIT_CODE}"
  printf 'BF16 profile exit:    %s (matched=%s)\n' "${PROFILE_BF16_EXIT_CODE}" "${N_KERNELS_MATCHED_BF16}"
  printf 'FP8  profile exit:    %s (matched=%s)\n' "${PROFILE_FP8_EXIT_CODE}"  "${N_KERNELS_MATCHED_FP8}"
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

  if [[ "$DRY_RUN" == "true" ]]; then
    print_dry_run
    exit 0
  fi

  if [[ "$MIGRATE_TMP" == "true" ]]; then
    migrate_tmp_artifacts
  fi

  if [[ "$DO_CLEAN" == "true" ]]; then
    clean_build_dir
  fi

  if [[ "$SKIP_BUILD" == "true" ]]; then
    require_profiler_binary
    log_info "skip-build: reusing cutlass_profiler at $(cutlass_profiler_path)"
    # Populate kernel counts from the existing manifest so the regenerated
    # manifest.json doesn't claim 0 matches when a prior build's kernels are
    # what we're profiling.
    if discover_generated_kernels_txt; then
      N_KERNELS_MATCHED_BF16="$(count_kernels_matching "${KERNEL_FILTER_BF16}")"
      N_KERNELS_MATCHED_FP8="$(count_kernels_matching "${KERNEL_FILTER_FP8}")"
      log_info "skip-build: kernels matched bf16=${N_KERNELS_MATCHED_BF16}, fp8=${N_KERNELS_MATCHED_FP8} (from ${GENERATED_KERNELS_TXT})"
    fi
  else
    configure_cmake
    verify_kernel_filter
    build_profiler
  fi

  enumerate_kernels bf16
  enumerate_kernels fp8

  if [[ "$DO_SMOKE" == "true" ]]; then
    run_profiler_pair_smoke
  fi

  run_profiler_pair_full
  classify_result_status
  emit_summary
  write_manifest

  if [[ "$UPDATE_MEMORY" == "true" ]]; then
    update_memory
  fi

  final_summary_echo
}

main "$@"
