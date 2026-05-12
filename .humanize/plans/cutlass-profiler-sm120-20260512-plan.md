# Tracked CUTLASS sm_120 Profiler Build And Benchmark Script

## Goal Description

Implement a tracked, reproducible Bash orchestrator at `/home/jakeshea/SOLAR/scripts/bench_cutlass_5060.sh` that drives the full CUTLASS v4.4.1 cutlass_profiler build, kernel-filter verification, BF16 / FP8 e4m3 GEMM benchmarks at 8192³, and result tabulation on the local RTX 5060 Laptop GPU (Blackwell `sm_120a`, GeForce variant). Outputs land under `/home/jakeshea/SOLAR/measurements/rtx5060/`: CSV + log + `manifest.json` per CUTLASS run, plus a top-level `SUMMARY.md` that compares the spec peak in `configs/arch/5060.yaml` against the prior cuBLASLt-13.4 datapoints and the new CUTLASS-4.4.1 datapoints.

Critical incorporated finding: CUTLASS v4.4.1's profiler-library `sm_120` GEMM coverage is **blockscaled-only** (e.g. NVFP4-in / BF16-out, MXFP8-in / BF16-out). There is no dense BF16-in / BF16-out `sm_120` GEMM in v4.4.1's profiler library. The script must produce useful artifacts in all three observed cases: dense kernels found, blockscaled-only found, or zero matches. Per user decision DEC-1, blockscaled kernels enter a separate row in `SUMMARY.md` with a precision-triplet label and a "not directly comparable to cuBLASLt dense BF16" caveat.

Implementation must be executed through `/humanize:start-rlcr-loop --skip-quiz --track-plan-file --privacy <plan-path>` (RLCR loop) as the user's hard requirement.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: `scripts/bench_cutlass_5060.sh` exists at `/home/jakeshea/SOLAR/scripts/bench_cutlass_5060.sh`, has executable mode, runs `set -euo pipefail`, and supports a fully documented `--help` flag listing every CLI option with its default.
  - Positive Tests (expected to PASS):
    - File `/home/jakeshea/SOLAR/scripts/bench_cutlass_5060.sh` exists with mode `0755`.
    - `bash scripts/bench_cutlass_5060.sh --help` exits 0 and prints usage with each documented flag (`--arch`, `--m`, `--n`, `--k`, `--matrix-size`, `--warmup-iterations`, `--profiling-iterations`, `--output-dir`, `--cutlass-dir`, `--build-dir`, `--kernel-filter-bf16`, `--kernel-filter-fp8`, `--jobs`, `--skip-build`, `--clean`, `--allow-empty-kernels`, `--dry-run`, `--update-memory`, `--migrate-tmp`, `--summary-statistic`, `--smoke`, `--no-summary`).
    - `shellcheck scripts/bench_cutlass_5060.sh` reports no severity-`error` findings.
  - Negative Tests (expected to FAIL):
    - `bash scripts/bench_cutlass_5060.sh --matrix-size=abc` exits non-zero with a clear `invalid value for --matrix-size` message.
    - `bash scripts/bench_cutlass_5060.sh --cutlass-dir=/nonexistent` exits non-zero with `cutlass dir not found` and does NOT invoke cmake.

- AC-2: Preflight phase validates required host tools and the local RTX 5060 GPU, and exits with actionable errors when any check fails.
  - Positive Tests (expected to PASS):
    - When `cmake`, `make` or `ninja`, `nvcc`, `nvidia-smi`, `python3`, `jq`, and `shellcheck` are on `PATH` and `nvidia-smi` reports a Blackwell-class GPU with compute capability matching the `--arch` value (default `120a`), the preflight phase logs `preflight ok` and continues to configure.
  - Negative Tests (expected to FAIL):
    - With `nvcc` removed from `PATH`, the script exits with `nvcc not found on PATH` and does NOT invoke cmake.
    - When `nvidia-smi -L` returns no CUDA device, the script exits with `no CUDA GPU detected`.

- AC-3: Kernel-filter verification phase runs **after** `cmake` configure but **before** the slow `cmake --build`, parses the generator-emitted manifest (`$BUILD_DIR/tools/library/generated_kernels.txt`, auto-discovered if path varies), counts kernels matching each filter, and short-circuits on zero matches unless `--allow-empty-kernels` is passed.
  - Positive Tests (expected to PASS):
    - With `--kernel-filter-bf16='*sm120*bf16*'` against CUTLASS v4.4.1, the manifest contains at least one matching kernel (blockscaled-output BF16 variants per the Phase 4 exploration), and the script logs `kernels matched: N >= 1` then proceeds to the build phase.
    - With `--kernel-filter-bf16='*sm999*'` AND `--allow-empty-kernels`, the script logs `0 kernels matched; continuing under --allow-empty-kernels`, writes `manifest.json` with `result_status: "zero_match"`, and skips the build phase cleanly with exit 0.
  - Negative Tests (expected to FAIL):
    - With `--kernel-filter-bf16='*sm999*'` and NO `--allow-empty-kernels`, the script aborts before `cmake --build` is invoked, exits with code 2, and writes a `manifest.json` whose `result_status` field is `"zero_match"`.

- AC-4: The CUTLASS profiler is invoked twice (once per precision filter) at the configured `(M, N, K)` (default 8192³), `warmup-iterations=10`, `profiling-iterations=20`, using **concrete kernel names** enumerated from a prior `cutlass_profiler --mode=dry_run` pass rather than raw wildcards, and the output CSV + log files are written to `$OUTPUT_DIR/`.
  - Positive Tests (expected to PASS):
    - The script invokes `cutlass_profiler --mode=dry_run --kernels=<wildcard>` once per precision, captures the concrete kernel names into `$OUTPUT_DIR/dryrun_kernels_{bf16,fp8}.txt`, then passes those exact names to the timed profile runs.
    - `$OUTPUT_DIR/cutlass_profiler_bf16_8192.csv` contains at least one row whose `Status` column reads `Passed` and whose `Provider` column reads `CUTLASS`.
    - `$OUTPUT_DIR/cutlass_profiler_fp8_8192.csv` contains at least one row matching an `e4m3` input data-type element.
    - Both log files capture stdout + stderr of the dry-run + timed phases, including any per-kernel error output.
  - Negative Tests (expected to FAIL):
    - If a profile run crashes (segfault, CUDA launch error), the log records the crash signature and either the CSV is absent OR contains a row with `Status` in `{Failed, Error}`; the script does NOT silently exit 0.
    - When zero kernels were matched (and `--allow-empty-kernels` is set), the CSV is absent and the log carries an explicit `0 matching kernels; nothing to profile` marker plus the dry-run output (no fake CSV row is injected).

- AC-5: `$SOLAR/measurements/rtx5060/SUMMARY.md` is emitted with a comparison table that spans BF16 and FP8 across spec, cuBLASLt, and CUTLASS data sources, and includes a separate row for any blockscaled CUTLASS kernels per user decision DEC-1.
  - Positive Tests (expected to PASS):
    - `SUMMARY.md` contains a markdown table with rows `{Spec peak, cuBLASLt achieved, CUTLASS dense, CUTLASS blockscaled}` and columns `{Precision triplet, Best TFLOPS, Median TFLOPS, Source artifact}` per user decision `--summary-statistic=both`.
    - Each cell's `Source artifact` column links to the underlying file (e.g., `configs/arch/5060.yaml`, `measurements/rtx5060/cublaslt/cublaslt_bf16_8192.csv`, `measurements/rtx5060/cutlass/cutlass_profiler_bf16_8192.csv`).
    - The `Notes` section below the table flags any row whose input precision differs from output precision as `blockscaled / not directly comparable to dense BF16` (e.g., `nvfp4_in / bf16_out / fp32_acc`).
  - Negative Tests (expected to FAIL):
    - If a row's data is missing (e.g., CUTLASS produced zero matching kernels under `--allow-empty-kernels`), the corresponding `Best TFLOPS` cell shows `N/A — see log` with a clickable link to the log file. The script does NOT inject a fake numeric value.
    - Re-running with `--no-summary` skips `SUMMARY.md` emission without raising an error and exits 0.

- AC-6: Every script run writes a mandatory `manifest.json` at `$OUTPUT_DIR/manifest.json` capturing host info, tool versions, kernel filter, match counts, exit codes per phase, and per-precision pre/post GPU state. The schema is valid JSON parseable by `jq .`.
  - Positive Tests (expected to PASS):
    - `manifest.json` exists and `jq .` parses it without error.
    - `manifest.json` contains keys: `timestamp` (ISO 8601), `host`, `gpu_name`, `gpu_pci_id`, `compute_capability`, `driver_version`, `cuda_version`, `nvcc_version`, `cmake_version`, `build_generator` (one of `ninja`, `make`), `cutlass_commit`, `cutlass_dir`, `build_dir`, `kernel_filter_bf16`, `kernel_filter_fp8`, `n_kernels_matched_bf16`, `n_kernels_matched_fp8`, `dryrun_kernel_names_bf16` (array), `dryrun_kernel_names_fp8` (array), `matrix_size` (object with `m`, `n`, `k`), `warmup_iterations`, `profiling_iterations`, `jobs`, `result_status` (one of `dense_found`, `blockscaled_only`, `zero_match`, `build_failed`, `preflight_failed`), `bf16_pre_state` and `bf16_post_state` and `fp8_pre_state` and `fp8_post_state` (each: object with `sm_clock_mhz`, `mem_clock_mhz`, `temp_c`, `power_w`), `build_exit_code`, `profile_bf16_exit_code`, `profile_fp8_exit_code`, `csv_paths` (object), `log_paths` (object), `summary_path`, `cublaslt_source_csvs`, `wmma_source_csvs`.
  - Negative Tests (expected to FAIL):
    - Removing `nvidia-smi` mid-run causes the corresponding `*_state` keys to record `unknown` rather than be omitted; `jq .` still parses.
    - Truncating the JSON output mid-write triggers a non-zero exit at the manifest-write step.

- AC-7: When `--update-memory` is passed, the script appends a `CUTLASS v4.4.1 cross-check` subsection to `.claude/memory/rtx5060_probe_recipe.md` and re-syncs `CLAUDE.md` from the per-memory files. Prior cuBLASLt content is preserved verbatim.
  - Positive Tests (expected to PASS):
    - After a run with `--update-memory`, `.claude/memory/rtx5060_probe_recipe.md` contains a new heading `## CUTLASS v4.4.1 cross-check (run: <timestamp>)` with bullet summaries of the BF16 and FP8 results, kernel filter, dry-run kernel count, and the `result_status` value.
    - `CLAUDE.md`'s `RTX 5060 probe recipe` section reflects the same new subsection (mirrored from `.claude/memory/rtx5060_probe_recipe.md`).
    - The pre-existing `Empirical cross-check (cuBLASLt 13.4 + NCU on sm_120, May 2026)` section remains present and unmodified.
  - Negative Tests (expected to FAIL):
    - Re-running the script with `--update-memory` and the same timestamp does NOT silently duplicate the subsection; instead it either updates in place or appends a dated sibling subsection (the implementation chooses one consistent behavior).
    - Without `--update-memory`, `.claude/memory/rtx5060_probe_recipe.md` is not modified.

- AC-8: `.gitignore` is updated to exclude build artifacts and binary blobs while keeping human-readable measurement artifacts tracked per user decision DEC-2.
  - Positive Tests (expected to PASS):
    - After a script run, `git status` (run from `/home/jakeshea/SOLAR`) shows the following as TRACKED-but-uncommitted: `measurements/rtx5060/SUMMARY.md`, `measurements/rtx5060/cutlass/cutlass_profiler_bf16_8192.csv`, `measurements/rtx5060/cutlass/cutlass_profiler_bf16_8192.log`, `measurements/rtx5060/cutlass/cutlass_profiler_fp8_8192.csv`, `measurements/rtx5060/cutlass/cutlass_profiler_fp8_8192.log`, `measurements/rtx5060/cutlass/manifest.json`, `measurements/rtx5060/cutlass/dryrun_kernels_bf16.txt`, `measurements/rtx5060/cutlass/dryrun_kernels_fp8.txt`.
    - `git status` does NOT show any file under `cutlass/build-sm120-profiler/`, any `*.bin` / `*.so` / `*.ncu-rep` / `*.qdrep` / `*.nsys-rep` under `measurements/`, or the `cutlass_profiler` binary itself.
  - Negative Tests (expected to FAIL):
    - Temporarily removing the `cutlass/build*/` gitignore entry causes `git status` to list thousands of compiled object files, confirming the entry is necessary.
    - The `!` re-include rules for `*.csv` / `*.log` / `*.json` survive across globbing precedence (positive test on `git check-ignore -v measurements/rtx5060/cutlass/manifest.json` returns no match).

- AC-9: The script supports `--skip-build` to reuse an existing `cutlass_profiler` binary and `--clean` to wipe `$BUILD_DIR` before configure.
  - Positive Tests (expected to PASS):
    - With `--skip-build` and an existing executable at `$BUILD_DIR/tools/profiler/cutlass_profiler`, the configure / build phases are skipped and the script jumps straight to the dry-run + profile phases.
    - With `--clean`, the script removes `$BUILD_DIR` before invoking `cmake -S ... -B ...`.
  - Negative Tests (expected to FAIL):
    - With `--skip-build` but no binary present at the expected path, the script exits non-zero with `cannot --skip-build: cutlass_profiler binary not found at <path>`.
    - Combining `--skip-build` and `--clean` exits non-zero with `--skip-build and --clean are mutually exclusive`.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

A complete `scripts/bench_cutlass_5060.sh` plus a small companion `scripts/parse_cutlass_csv.py` (or inline awk/jq pipeline) implementing all nine acceptance criteria; supporting every documented CLI flag (`--help`, `--arch`, `--m`, `--n`, `--k`, `--matrix-size`, `--warmup-iterations`, `--profiling-iterations`, `--output-dir`, `--cutlass-dir`, `--build-dir`, `--kernel-filter-bf16`, `--kernel-filter-fp8`, `--jobs`, `--skip-build`, `--clean`, `--allow-empty-kernels`, `--dry-run`, `--update-memory`, `--migrate-tmp`, `--summary-statistic`, `--smoke`, `--no-summary`); writing per-run outputs into both a timestamped subdirectory `measurements/rtx5060/cutlass/<ISO8601>/` and a stable `measurements/rtx5060/cutlass/latest/` symlink; capturing per-precision pre/post nvidia-smi state into `manifest.json`; emitting a 4-row `SUMMARY.md` table with linked sources and caveat notes; updating `.claude/memory/rtx5060_probe_recipe.md` and re-syncing `CLAUDE.md`; updating `.gitignore` with the precise inclusions and exclusions per DEC-2; preferring `ninja` over `make` when available and recording the chosen generator in the manifest; running a `--smoke` 1024³ pass before the 8192³ runs when requested; passing `shellcheck` cleanly.

### Lower Bound (Minimum Acceptable Scope)

A working `scripts/bench_cutlass_5060.sh` that satisfies AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, and AC-8: the script exists with shellcheck-clean shell; preflight runs and refuses to proceed without the required tools and a `sm_120a`-class GPU; kernel-filter verification short-circuits on zero matches; the profiler runs once each for BF16 and FP8 at the default 8192³ using enumerated concrete kernel names; the two CSV + two log files are produced under `measurements/rtx5060/cutlass/`; `manifest.json` is mandatory and captures the schema fields enumerated in AC-6; `SUMMARY.md` is emitted at `measurements/rtx5060/SUMMARY.md` with the comparison table; `.gitignore` is updated to exclude `cutlass/build*/` while keeping `*.csv` / `*.log` / `*.json` / `SUMMARY.md` tracked. AC-7 (`--update-memory`) and AC-9 (`--skip-build`/`--clean`) are deferred to later RLCR iterations; `--smoke`, `--migrate-tmp`, and the timestamped + `latest/` layout are optional refinements.

### Allowed Choices

- Can use: GNU Bash 4.x or later as the orchestrator language; `python3` (system-installed, NOT the SOLAR `.venv`, to keep the script invokable without `source .venv/bin/activate`) for CSV/YAML parsing; `jq` (already installed locally) for JSON parsing and manifest emission; `awk`, `grep`, `sed` for log scraping; `cmake` ≥ 3.20 with either `Ninja` (preferred) or `Unix Makefiles`; `nvcc` from `/opt/cuda/bin/nvcc` (CUDA 13.2); `nvidia-smi` for state capture; the canonical SOLAR script idioms from `install.sh`, `install_uv.sh`, `scripts/run_tests.sh`, `examples/Matmul/run_solar.sh`. Output schema is fixed: CSV (CUTLASS profiler native format) + JSON (`manifest.json`) + Markdown (`SUMMARY.md`).
- Cannot use: Docker (the draft is explicit about native build on the host); `torch` or other heavy ML frameworks (not installed in the system Python); any online service or `git` operation that mutates the CUTLASS clone (no `git fetch`, `git checkout`, `git pull` against `./cutlass`); automatic CUTLASS version-fallback ladder (the v4.4.2 / v4.5.0 climb is intentionally out of scope and deferred to a follow-up plan if needed); `--no-verify` / `--no-gpg-sign` / `--force-push` style escape hatches; commenting plan-document terminology (`AC-`, `Milestone`, `Step`, `Phase`) into the script source.

> **Note on Deterministic Designs**: The user specified the matrix size `m=n=k=8192`, warmup `10`, profiling iterations `20`, the two filter wildcards `'*sm120*bf16*'` / `'*sm120*fp8*'` (as starting wildcards before dry-run resolves them to concrete names), and the build flag set `-DCUTLASS_NVCC_ARCHS=120a -DCUTLASS_LIBRARY_OPERATIONS=gemm -DCUTLASS_LIBRARY_KERNELS=...`. These are deterministic per the user's commands. The path boundaries above are narrower than typical because the input is highly deterministic; the upper-lower gap is in *scaffolding richness* (timestamped layouts, memory sync, smoke mode), not in *measurement methodology*.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

A linear orchestration script driven by named functions, one per logical phase:

```bash
#!/usr/bin/env bash
set -euo pipefail

main() {
  parse_flags "$@"             # sets ARCH, M/N/K, OUTPUT_DIR, CUTLASS_DIR, ...
  log_section "preflight"
  preflight                    # command -v, nvidia-smi -L, cutlass-dir, GPU CC
  if [[ "$DO_CLEAN" == "true" ]]; then clean_build_dir; fi
  if [[ "$SKIP_BUILD" != "true" ]]; then
    log_section "configure"
    configure_cmake            # cmake -S "$CUTLASS_DIR" -B "$BUILD_DIR" -D...
    log_section "verify"
    verify_kernel_filter || handle_empty_kernels
    log_section "build"
    build_profiler             # cmake --build "$BUILD_DIR" --target cutlass_profiler --parallel "$JOBS"
  fi
  log_section "enumerate-bf16"; enumerate_kernels bf16
  log_section "enumerate-fp8";  enumerate_kernels fp8
  if [[ "$DO_SMOKE" == "true" ]]; then
    log_section "smoke-1024"
    run_profiler bf16 1024 2 3
    run_profiler fp8  1024 2 3
  fi
  log_section "profile-bf16"; run_profiler bf16 "$M" "$WARMUP" "$ITERS"
  log_section "profile-fp8";  run_profiler fp8  "$M" "$WARMUP" "$ITERS"
  log_section "tabulate";     emit_summary       # write SUMMARY.md (unless --no-summary)
  log_section "manifest";     write_manifest     # mandatory
  if [[ "$UPDATE_MEMORY" == "true" ]]; then update_memory; fi
  final_summary_echo
}
```

CMake invocation (default `--arch=120a` per Codex round-2 finding):

```bash
GENERATOR="Unix Makefiles"
command -v ninja >/dev/null 2>&1 && GENERATOR="Ninja"

cmake -S "$CUTLASS_DIR" -B "$BUILD_DIR" \
  -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS="$ARCH" \
  -DCUTLASS_LIBRARY_OPERATIONS="gemm" \
  -DCUTLASS_LIBRARY_KERNELS="$KERNEL_FILTER_BF16,$KERNEL_FILTER_FP8"

cmake --build "$BUILD_DIR" --target cutlass_profiler --parallel "$JOBS"
```

Kernel-filter verification reads `$BUILD_DIR/tools/library/generated_kernels.txt` (auto-discovered) after configure; counts lines matching each filter pattern; aborts with exit 2 if any count is zero and `--allow-empty-kernels` is not set.

Concrete kernel names are gathered by `cutlass_profiler --mode=dry_run --operation=Gemm --kernels='<wildcard>'`; the dry-run output is captured into `dryrun_kernels_{bf16,fp8}.txt`; the timed profile then uses the comma-joined concrete names as the `--kernels` argument so the timed run isn't subject to wildcard surprises.

### Relevant References

- `/home/jakeshea/SOLAR/install.sh` — flag-parser pattern (`while [[ $# -gt 0 ]]; case $1 in ... esac`), step-labeled `echo "==> Step N: ..."` logging.
- `/home/jakeshea/SOLAR/install_uv.sh` — `command -v` preflight detection, conditional logic, env-var resolution.
- `/home/jakeshea/SOLAR/scripts/run_tests.sh` — color helpers (`RED`, `GREEN`, `BLUE`, `NC`), summary box pattern, named phase functions.
- `/home/jakeshea/SOLAR/examples/Matmul/run_solar.sh` — `SCRIPT_DIR=$(cd ... && pwd)` resolution, env-var fallback `${SOLAR_*_OUTPUT_DIR:-…}`, final-summary echo.
- `/home/jakeshea/SOLAR/docker/Dockerfile` — canonical CUTLASS v4.4.1 build pattern (`git clone --depth 1 -b v4.4.1 ...`), package set (`cmake`, `ninja-build`, `ccache`).
- `/home/jakeshea/SOLAR/cutlass/tools/profiler/src/options.cu` — CLI flag definitions: `--warmup-iterations` (default 10), `--profiling-iterations` (default 100, we override to 20), `--output`, `--kernels`, `--mode` (accepts `dry_run`).
- `/home/jakeshea/SOLAR/cutlass/python/cutlass_library/generator.py` — procedural kernel naming template `cutlass3x_sm{ar}_{op}_{ex}{ct}{cs}_{l}_align{al}{t}{k}{e}`; confirms `*sm120*` wildcards match real kernel names.
- `/home/jakeshea/SOLAR/cutlass/python/cutlass_library/emit_kernel_listing.py` — sm120 BF16 entries `s1688gemm_bf16` and FP8 entries `gemm_e4m3_e4m3_f32_bf16_bf16`; confirms `120a` and `120f` are the supported arch strings (plain `120` may not produce kernels).
- `/home/jakeshea/SOLAR/cutlass/tools/library/CMakeLists.txt` — `generated_kernels.txt` is the standard kernel-list artifact; emitted during cmake configure (NOT make).
- `/home/jakeshea/SOLAR/cutlass/customConfigs.cmake` — `cutlass_generate_kernel_filter_and_testlist_files()` produces FK files only when `CUTLASS_BUILD_FOR_PROFILER_REGRESSIONS=ON`; we deliberately do NOT enable that flag and instead parse `generated_kernels.txt`.
- `/home/jakeshea/SOLAR/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu` — confirms `cutlass::arch::Sm120` is a real arch tag with NVFP4-in / BF16-out blockscaled GEMM in v4.4.1; the kind of kernel that `*sm120*bf16*` filter will actually match.
- `/home/jakeshea/SOLAR/configs/arch/5060.yaml` — spec peaks: `MAC_per_cycle_bf16_tc=14317` → 71.5 TFLOPS dense, `MAC_per_cycle_fp8_tc=28634` → 143 TFLOPS dense, both at `freq_GHz: 2.497`. Source for `SUMMARY.md`'s `Spec peak` row.
- `/home/jakeshea/SOLAR/.claude/memory/rtx5060_probe_recipe.md` — existing memory section format; the `--update-memory` flag appends a new dated subsection here.
- `/tmp/solar-rtx5060/cublaslt_bf16_8192.csv`, `/tmp/solar-rtx5060/cublaslt_fp8_8192.csv` — prior cuBLASLt datapoints; `--migrate-tmp` copies these into `measurements/rtx5060/cublaslt/`.

## Dependencies and Sequence

### Milestones

1. Milestone 1 — Scaffolding and Preflight
   - Phase A: Create `scripts/bench_cutlass_5060.sh` skeleton: shebang, `set -euo pipefail`, `SCRIPT_DIR` and `SOLAR_ROOT` resolution via `$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)`, flag-parser loop, log helpers, named function stubs, `--help` body.
   - Phase B: Implement `preflight()` covering `command -v` for `cmake`, `make`/`ninja`, `nvcc`, `nvidia-smi`, `python3`, `jq`, `shellcheck`; GPU presence via `nvidia-smi -L`; cutlass-dir + build-dir existence and writability.

2. Milestone 2 — CMake Configure and Pre-Compile Verification
   - Phase A: Implement `configure_cmake()` invoking the exact `cmake -S … -B …` with `-DCUTLASS_NVCC_ARCHS=120a`, `-DCUTLASS_LIBRARY_OPERATIONS=gemm`, `-DCUTLASS_LIBRARY_KERNELS="$KERNEL_FILTER_BF16,$KERNEL_FILTER_FP8"`, generator detection, `Release` build type.
   - Phase B: Implement `verify_kernel_filter()` that auto-discovers `$BUILD_DIR/tools/library/generated_kernels.txt` (or the closest analogue), greps for matches against each filter, populates `n_kernels_matched_bf16` / `n_kernels_matched_fp8` for the manifest, short-circuits to `handle_empty_kernels()` on zero unless `--allow-empty-kernels`.
   - Phase C: Implement `build_profiler()` via `cmake --build "$BUILD_DIR" --target cutlass_profiler --parallel "$JOBS"` with conservative default `JOBS=8`.

3. Milestone 3 — Dry-Run Enumeration and Timed Profiling
   - Phase A: Implement `enumerate_kernels(precision)` invoking `$BUILD_DIR/tools/profiler/cutlass_profiler --mode=dry_run --operation=Gemm --kernels='<wildcard>'`, capturing concrete kernel names into `$OUTPUT_DIR/dryrun_kernels_<precision>.txt`.
   - Phase B: Implement `run_profiler(precision, dim, warmup, iters)` invoking the profiler with `--m=<dim> --n=<dim> --k=<dim> --warmup-iterations=<warmup> --profiling-iterations=<iters> --kernels=<comma-joined-concrete-names> --output=<csv>`; capture pre/post `nvidia-smi` snapshots into manifest fields.
   - Phase C: Wire `--smoke` to invoke `run_profiler` at 1024³ with warmup=2, iters=3 before the 8192³ runs; abort early on smoke failure.

4. Milestone 4 — Reporting and Manifest
   - Phase A: Implement `emit_summary()` that reads the two CUTLASS CSVs, the cuBLASLt baseline CSVs (default `measurements/rtx5060/cublaslt/`; fallback `/tmp/solar-rtx5060/` with warning), and `configs/arch/5060.yaml`; computes best and median TFLOPS per source; writes `measurements/rtx5060/SUMMARY.md` with the 4-row table and the precision-triplet caveat note per DEC-1.
   - Phase B: Implement `write_manifest()` emitting `manifest.json` per AC-6 schema; mandatory in both upper and lower bounds.

5. Milestone 5 — Side Effects and Tracking
   - Phase A: Implement `update_memory()` (gated by `--update-memory`): append a dated `## CUTLASS v4.4.1 cross-check (run: <ISO8601>)` subsection to `.claude/memory/rtx5060_probe_recipe.md`; re-sync `CLAUDE.md` from the per-memory files.
   - Phase B: Update `.gitignore` with the precise exclusions for `cutlass/build*/`, binary artifacts under `measurements/rtx5060/cutlass/`, and the `!`-re-include rules for `*.csv` / `*.log` / `*.json` / `*.md` / `*.txt` under `measurements/rtx5060/`.
   - Phase C: Implement `--migrate-tmp` branch: `cp /tmp/solar-rtx5060/cublaslt_*.csv measurements/rtx5060/cublaslt/`, `cp /tmp/solar-rtx5060/ncu_hmma_fp16*.csv measurements/rtx5060/wmma/`. Default off.

Dependency graph: M1 → M2 → M3 → M4 → M5 (linear). Within M3, Phase A precedes B (enumeration before timed run). Within M4, Phase B (manifest) can run in parallel with Phase A (summary), but both must complete before M5. M5 phases (A, B, C) are mutually independent and may run in any order.

## Task Breakdown

Each task carries exactly one routing tag: `coding` (Claude implements directly) or `analyze` (Codex investigates via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Scaffold `scripts/bench_cutlass_5060.sh`: shebang, `set -euo pipefail`, `SCRIPT_DIR`/`SOLAR_ROOT` resolution, flag parser covering all 22 documented CLI flags with defaults, log helpers (colors from `scripts/run_tests.sh`), named function stubs, `--help` body. | AC-1 | coding | - |
| task2 | Implement `preflight()`: `command -v` for `cmake`, `make`/`ninja`, `nvcc`, `nvidia-smi`, `python3`, `jq`, `shellcheck`; `nvidia-smi -L` GPU presence + compute-capability match against `$ARCH`; cutlass-dir + build-dir validity and writability. | AC-2 | coding | task1 |
| task3 | Implement `configure_cmake()`: build the exact `cmake -S … -B …` invocation with `-DCUTLASS_NVCC_ARCHS="$ARCH"` (default `120a`), `-DCUTLASS_LIBRARY_OPERATIONS=gemm`, `-DCUTLASS_LIBRARY_KERNELS="$KERNEL_FILTER_BF16,$KERNEL_FILTER_FP8"`, `-DCMAKE_BUILD_TYPE=Release`, generator selection (`Ninja` if available else `Unix Makefiles`). Out-of-source build dir defaults to `$CUTLASS_DIR/build-sm120-profiler`. | AC-1, AC-3 | coding | task2 |
| task4 | Investigate the exact path / filename of the CUTLASS-emitted kernel manifest after configure (likely `$BUILD_DIR/tools/library/generated_kernels.txt`), confirm its presence with a real configure run, and document its schema for `verify_kernel_filter()` to parse. | AC-3 | analyze | task3 |
| task5 | Implement `verify_kernel_filter()`: read the auto-discovered manifest, grep against each filter pattern, populate kernel-count counters, short-circuit on zero matches unless `--allow-empty-kernels`. Implement `handle_empty_kernels()` which writes a `manifest.json` with `result_status: "zero_match"` and exits 2 (or 0 with `--allow-empty-kernels`). | AC-3, AC-6 | coding | task4 |
| task6 | Implement `build_profiler()`: `cmake --build "$BUILD_DIR" --target cutlass_profiler --parallel "$JOBS"`; default `JOBS=8`; respect `--jobs` override. | AC-1, AC-9 | coding | task5 |
| task7 | Implement `enumerate_kernels(precision)`: invoke `cutlass_profiler --mode=dry_run --operation=Gemm --kernels='<wildcard>'`; write concrete kernel names to `$OUTPUT_DIR/dryrun_kernels_<precision>.txt`. | AC-4 | coding | task6 |
| task8 | Implement `run_profiler(precision, dim, warmup, iters)`: invoke `cutlass_profiler --m=<dim> --n=<dim> --k=<dim> --warmup-iterations=<warmup> --profiling-iterations=<iters> --kernels=<comma-joined-from-task7> --output=<csv>`; tee stdout/stderr to log; record pre/post `nvidia-smi -q` snapshots into per-precision manifest state fields. | AC-4, AC-6 | coding | task7 |
| task9 | Wire two `run_profiler` calls (BF16 then FP8) at 8192³; expose per-precision exit codes individually for the manifest. | AC-4 | coding | task8 |
| task10 | Wire `--smoke` to run an extra `run_profiler bf16 1024 2 3` + `run_profiler fp8 1024 2 3` pre-pass; abort the 8192³ runs on smoke failure. | AC-4 | coding | task8 |
| task11 | Implement `emit_summary()`: parse `cutlass_profiler_{bf16,fp8}_8192.csv`, the cuBLASLt CSVs (default `measurements/rtx5060/cublaslt/*.csv`, fallback `/tmp/solar-rtx5060/cublaslt_*.csv` with manifest warning), and `configs/arch/5060.yaml`; compute best and median TFLOPS; emit `measurements/rtx5060/SUMMARY.md` with 4-row table (`Spec peak`, `cuBLASLt achieved`, `CUTLASS dense`, `CUTLASS blockscaled`) per DEC-1; include caveat notes; respect `--no-summary`. | AC-5 | coding | task9 |
| task12 | Implement `write_manifest()`: collect every AC-6 schema field; emit `$OUTPUT_DIR/manifest.json` via `jq -n` for safe JSON generation. Mandatory in lower bound. | AC-6 | coding | task9 |
| task13 | Implement `update_memory()` (gated by `--update-memory`): append `## CUTLASS v4.4.1 cross-check (run: <ISO8601>)` to `.claude/memory/rtx5060_probe_recipe.md`; re-sync `CLAUDE.md` by reading every `.claude/memory/*.md` and re-generating the consolidated body (matches the pattern from the existing CLAUDE.md). | AC-7 | coding | task11 |
| task14 | Update `.gitignore` precisely: add `/cutlass/build*/`, `/measurements/rtx5060/cutlass/**/*.bin`, `/measurements/rtx5060/cutlass/**/*.so`, `/measurements/rtx5060/cutlass/**/*.ncu-rep`, `/measurements/rtx5060/cutlass/**/*.qdrep`, `/measurements/rtx5060/cutlass/**/*.nsys-rep`, plus the `!` re-include rules for `*.csv`, `*.log`, `*.json`, `*.txt`, `*.md` under `measurements/rtx5060/`. Verify with `git check-ignore -v` on representative paths. | AC-8 | coding | - |
| task15 | Implement `--skip-build` branch (skip configure + build, jump to enumerate) and `--clean` branch (wipe `$BUILD_DIR` before configure); enforce mutual exclusion. | AC-9 | coding | task6 |
| task16 | Implement `--dry-run`: print every cmake / make / cutlass_profiler invocation that WOULD be executed, with their resolved flag values, but exit 0 without executing anything. | AC-1 | coding | task1 |
| task17 | Implement `--migrate-tmp`: `cp /tmp/solar-rtx5060/cublaslt_*.csv` into `measurements/rtx5060/cublaslt/`; `cp /tmp/solar-rtx5060/ncu_hmma_fp16*.csv` into `measurements/rtx5060/wmma/`; default off. | AC-5 | coding | task1 |
| task18 | Verify the script's CUTLASS v4.4.1 behavior on the local RTX 5060 via the RLCR loop's normal execution: confirm that `*sm120*bf16*` matches only blockscaled-output kernels (per the Phase 4 exploration finding), and document the exact kernel names produced by `dry_run` in the SUMMARY.md `Notes` section. | AC-5, AC-3 | analyze | task7, task11 |
| task19 | Sanity-run the script end-to-end on the local 5060 GPU via the RLCR loop, confirming AC-1 through AC-8 are observable in the produced artifacts and `git status`. | AC-1, AC-4, AC-5, AC-6, AC-8 | coding | task1-task17 |
| task20 | Cross-check that the `--arch=120a` default is correct for the local RTX 5060 Laptop GPU (NVIDIA reports it as GeForce Blackwell, compute capability `sm_120`; CUTLASS's generator gates on `120a`/`120f`). Adjust the default if the local probe disagrees. | AC-1, AC-2 | analyze | task2 |

## Claude-Codex Deliberation

### Agreements

- Pre-compile kernel-filter verification is essential; zero-match must abort BEFORE the slow nvcc compile to avoid wasting ~15 minutes.
- No fake CSV rows: zero-match produces an explicit log marker + `manifest.json` with `result_status: "zero_match"`, NOT a stub CSV.
- Manifest is mandatory (NOT a lower-bound deferral); it captures the host, GPU, tool versions, kernel filter, match counts, exit codes, and pre/post GPU state.
- Out-of-source build at `$CUTLASS_DIR/build-sm120-profiler` avoids colliding with any pre-existing `build/`.
- Per-precision pre/post `nvidia-smi` state samples in the manifest, NOT a single global pre/post pair (Codex round-2 OPTIONAL_IMPROVEMENT, adopted).
- Conservative default `--jobs=8` instead of `$(nproc)=32` to prevent OOM and thermal throttling on a laptop.
- Power/clock policy is observe-only; the script does NOT call `nvidia-smi -lgc` / `-lmc` / `-pl`.
- Timed runs use enumerated concrete kernel names from `--mode=dry_run`, not raw wildcards.
- Optional `--smoke` 1024³ pre-pass catches launch/filter failures before the 8192³ runs.
- `SUMMARY.md` row structure includes a separate `CUTLASS blockscaled` entry per the precision triplet (DEC-1, recommended).
- Human-readable artifacts (SUMMARY.md, CSVs, manifest.json, logs) are git-tracked under `measurements/rtx5060/`; binary blobs (build/, *.bin, *.ncu-rep) are gitignored (DEC-2, recommended).
- RLCR loop is the mandatory execution vehicle; the plan does not self-execute.

### Resolved Disagreements

- Default `--arch` value: Claude v1 proposed `--arch=120`; Codex round-2 noted CUTLASS's sm_120 generator path is keyed on `120a`/`120f` for GeForce Blackwell and rejects plain `120`. Resolution: default is `--arch=120a`, with `120f` / `121a` as user overrides. Manifest records the exact arch string used.
- Pre-compile manifest path: Claude v1/v2 proposed parsing `FK_functional_L0_testlist_SM*_*.csv` produced by `cutlass_generate_kernel_filter_and_testlist_files()`; Codex round-2 pointed out those FK files require `-DCUTLASS_BUILD_FOR_PROFILER_REGRESSIONS=ON` (not in the user's command). Resolution: parse `$BUILD_DIR/tools/library/generated_kernels.txt` (always produced during configure under a normal `-DCUTLASS_LIBRARY_KERNELS=...` flow). Avoids enabling extra regression-test machinery.
- `make -j$(nproc)` aggressiveness: Codex round-1 flagged laptop OOM/thermal risk on a 32-core machine. Resolution: default `--jobs=8`; user can override with `--jobs=$(nproc)` if confident.
- Artifact layout (single dir vs nested): Codex round-1 flagged a conflict between `measurements/rtx5060/SUMMARY.md` (top-level summary) and CUTLASS files under `measurements/rtx5060/cutlass/`. Resolution: SUMMARY.md is at `measurements/rtx5060/SUMMARY.md` (top-level because it spans all three sources: spec, cuBLASLt, CUTLASS); CUTLASS-specific files (CSV, log, manifest, dry-run kernel lists) live under `measurements/rtx5060/cutlass/`; cuBLASLt-specific files under `measurements/rtx5060/cublaslt/`; wmma-specific files under `measurements/rtx5060/wmma/`. The layout is consistent (top-level SUMMARY.md aggregates the three subdirs).
- Stub CSV vs manifest: Codex round-1 flagged that a zero-kernel "stub CSV" could confuse downstream parsers. Resolution: never inject a fake CSV row; always emit `manifest.json` with `result_status` and let the absence of the CSV be the signal.
- Statistic for `SUMMARY.md`: Codex round-1 asked best vs median vs both. Resolution: `--summary-statistic=both` is the default; SUMMARY.md shows both `Best TFLOPS` and `Median TFLOPS` columns.
- /tmp migration policy: Codex round-1 asked whether the script should auto-migrate `/tmp/solar-rtx5060/*`. Resolution: opt-in via `--migrate-tmp` (default off); the summary tabulator transparently falls back to `/tmp` with a manifest warning if `measurements/rtx5060/cublaslt/` is empty.

### Convergence Status

- Final Status: `converged`
- Rounds executed: 2 Codex review rounds (round 1 produced 9 REQUIRED_CHANGES, all incorporated; round 2 produced 2 REQUIRED_CHANGES — arch default and manifest-file path — both incorporated). Round 2 closing line: "After the two required changes above, I see no further required changes. With only the optional manifest granularity improvement remaining, we can declare convergence."

## Pending User Decisions

All previously-pending decisions have been resolved during Phase 6 user interaction. Recorded here for traceability:

- DEC-1: SUMMARY.md row policy for v4.4.1 sm_120 blockscaled-only BF16 kernels.
  - Claude Position: Emit a separate row `CUTLASS blockscaled` with precision triplet (e.g., `nvfp4_in / bf16_out / fp32_acc`) and "not directly comparable to cuBLASLt dense BF16" caveat.
  - Codex Position: Codex did not propose a competing layout in rounds 1/2; flagged comparability as an UNRESOLVED concern and recommended Claude's caveated-row approach.
  - Tradeoff Summary: Including the row preserves visibility into what v4.4.1 actually offers; omitting would simplify SUMMARY.md but lose data. The caveated-row approach is faithful to the underlying precision mismatch.
  - Decision Status: User selected "Separate row, caveated".

- DEC-2: Tracking policy for `measurements/rtx5060/` artifacts.
  - Claude Position: Track SUMMARY.md + CSV + JSON + logs; gitignore binary blobs (build/, *.bin, *.ncu-rep).
  - Codex Position: Codex flagged "ensure .gitignore rules don't accidentally ignore tracked measurement artifacts" — agnostic on which to track, but emphasized rule precision.
  - Tradeoff Summary: Tracking human-readable artifacts enables sharing/reproducing results across hosts via git history. Tracking nothing simplifies .gitignore but loses repo-tracked record. Tracking only SUMMARY.md is a middle ground.
  - Decision Status: User selected "Track SUMMARY.md + CSV + JSON + logs".

## Implementation Notes

### Code Style Requirements

- Implementation code and inline comments in `scripts/bench_cutlass_5060.sh` (and any helper Python parser) must NOT contain plan-specific workflow terminology such as `AC-`, `Milestone`, `Step N`, `Phase A`, or similar markers. Those identifiers belong in this plan document, not in the resulting codebase.
- Use descriptive, domain-appropriate names: `verify_kernel_filter`, `enumerate_kernels`, `run_profiler`, `emit_summary`, `write_manifest`, `update_memory`. NOT `phase_a`, `step3`, `ac4_check`.
- Bash idioms required: `set -euo pipefail`, `local` for function-scoped variables, `"${VAR:-default}"` for env-var fallbacks, color codes copied from `scripts/run_tests.sh` for log output.
- Comments only when the WHY is non-obvious (e.g., "use `120a` not `120` because CUTLASS generator filters reject plain `120` for GeForce Blackwell"). Don't restate WHAT the bash does.
- JSON emission uses `jq -n --arg key value '$ARGS.named'` patterns to avoid manual quoting bugs.
- All file paths in the script use absolute paths derived from `SOLAR_ROOT` (resolved via `SCRIPT_DIR/..`); no relative `..` traversal beyond the resolution step.
- The script must be invocable from any CWD, not only from `$SOLAR_ROOT`. All paths anchor on the resolved `SOLAR_ROOT`.

--- Original Design Draft Start ---

# Tracked CUTLASS sm_120 Profiler Build And Benchmark Script (gen-idea draft)

## Original Idea

Build CUTLASS v4.4.1 cutlass_profiler with sm_120 (Blackwell consumer) support and run BF16 + FP8 e4m3 8192³ GEMM benchmarks on the local RTX 5060 Laptop GPU, to provide an end-to-end cross-check against the values already in configs/arch/5060.yaml.

Concrete requirements from the user:
1. CUTLASS v4.4.1 has been freshly cloned at ./cutlass (HEAD = 4370102 v4.4.1 update, 200 MB). The sol-execbench Dockerfile (which we now have a local copy of under ./docker/) shows the canonical build pattern: ubuntu24.04 + nvidia/cuda:13.1.1-cudnn-devel + git clone -b v4.4.1.
2. We are NOT building inside Docker. We build natively with CUDA 13.2 + Nsight Compute 2026.1.1 already installed at /opt/cuda. nvcc supports -arch=sm_120.
3. Build flags supplied by the asker:
     cmake .. -DCUTLASS_NVCC_ARCHS="120" \
              -DCUTLASS_LIBRARY_OPERATIONS="gemm" \
              -DCUTLASS_LIBRARY_KERNELS="*sm120*bf16*,*sm120*fp8*"
     make -j$(nproc) cutlass_profiler
4. Two profile commands to run after build:
     ./tools/profiler/cutlass_profiler --kernels='cutlass*sm120*bf16*gemm*' --m=8192 --n=8192 --k=8192 --warmup-iterations=10 --profiling-iterations=20 --output=/workspace/solar-rtx5060/cutlass_profiler_bf16_8192.csv
     ./tools/profiler/cutlass_profiler --kernels='cutlass*sm120*fp8*gemm*'  --m=8192 --n=8192 --k=8192 --warmup-iterations=10 --profiling-iterations=20 --output=/workspace/solar-rtx5060/cutlass_profiler_fp8_8192.csv
   (Local equivalent path will be /tmp/solar-rtx5060/ — the docker /workspace path won't apply.)
5. Failure-handling requirement: if CUTLASS v4.4.1 has no sm_120 kernels (cmake skips them or the kernel filter matches 0), capture that exact failure in the .log files — both paths (success and "no sm_120 kernels in v4.4.1") are acceptable outcomes.

Known unknowns the idea must call out:
- CUTLASS v4.4.1 was tagged on 2026-04 (before v4.5.0). sm_120 (Blackwell consumer GeForce) was first added in CUTLASS 3.7 → 4.x. Whether v4.4.1 has sm_120 GEMM kernels at all needs verification — `find cutlass -name '*sm120*'` should return non-zero, and the CMake `library_kernels` filter should match >0 templates. If v4.4.1 silently skips sm_120, the task degrades to "produce a log file documenting that" (and we may need to escalate to v4.5.x or upstream main).
- The matching kernel name patterns (`cutlass*sm120*bf16*gemm*`, `cutlass*sm120*fp8*gemm*`) are speculative — actual CUTLASS profiler kernel names follow a strict scheme (e.g., `cutlass3x_sm120_tensorop_*bf16_*gemm_*`). Verify with `./cutlass_profiler --operation=Gemm --kernels='cutlass*sm120*' --list` *after* the build but *before* the timed runs.
- Build time: with `-DCUTLASS_LIBRARY_KERNELS="*sm120*bf16*,*sm120*fp8*"` the kernel-instantiation count is small (maybe 10-50 templates), so build should be <15 min on a 32-core box. Full library build would be 60-120 min.
- Output paths: user wrote `/workspace/solar-rtx5060/*` which is the docker mount. The local equivalent is `/tmp/solar-rtx5060/` (where the earlier cuBLASLt outputs live).
- We have already gathered: BF16 cuBLASLt 8192³ = ~28 TFLOPS (sm_80 fallback kernel), FP8 cuBLASLt 8192³ = ~106 TFLOPS (sm_120-native nvjet kernel, 96% peak). The CUTLASS profiler result is the *third* datapoint to triangulate whether the BF16 28 TFLOPS gap is library-specific or hardware. Expectation: if CUTLASS v4.4.1 has a properly-tuned sm_120 BF16 kernel, BF16 should land in the 45-65 TFLOPS range (much closer to the 71.5 TFLOPS spec peak).

Constraints:
- We have already populated /home/jakeshea/SOLAR/configs/arch/5060.yaml. This task does not modify the yaml; it only produces measurement artifacts in /tmp/solar-rtx5060/ (or a chosen permanent path).
- Memory file `.claude/memory/rtx5060_probe_recipe.md` already records the BF16-cuBLAS-sm_80-fallback finding. The CUTLASS measurement will either confirm or refute that finding and should be appended to the same memory after the run.
- Build artifacts will be ~1-3 GB under ./cutlass/build/. The user did not ask us to clean them up afterwards; we should not gitignore the cutlass/ tree if the user committed it, but should add cutlass/build/ to .gitignore.

Deliverables expected:
1. A working `./cutlass/build/tools/profiler/cutlass_profiler` binary that compiles + runs.
2. Two CSV outputs at /tmp/solar-rtx5060/cutlass_profiler_{bf16,fp8}_8192.csv with the *full* CUTLASS profiler columns (Runtime, GFLOPs, Status, Provider, Kernel, etc.) for at least one matching kernel each.
3. Two log files at /tmp/solar-rtx5060/cutlass_profiler_{bf16,fp8}_8192.log capturing stdout/stderr, kernel-list output, exit status, and (if applicable) the exact "no kernels matched" / "build skipped sm_120" message.
4. A short comparison table in the run notes: spec peak vs cuBLASLt achieved vs CUTLASS profiler best-achieved, for both BF16 and FP8.
5. Updated memory: `.claude/memory/rtx5060_probe_recipe.md` and the synced `CLAUDE.md` should record the final triangulation between cuBLASLt (sm_80 fallback) and CUTLASS profiler.

Repo context: this is the SOLAR repo at /home/jakeshea/SOLAR. Currently on `master`, no PR workflow set up. The work belongs in a build directory under ./cutlass and result files under a permanent /home/jakeshea/SOLAR/measurements/rtx5060/ (better than ephemeral /tmp).

Please draft the idea, then it will be turned into a plan via /humanize:gen-plan and executed via /humanize:start-rlcr-loop in a subsequent session.

## Primary Direction: Reproducible host-build script in `scripts/`

### Rationale

Rather than running ad-hoc commands once, encode the entire build+verify+benchmark+tabulate flow in a tracked shell script (`scripts/bench_cutlass_5060.sh`) that lives in the repo, is version-controlled, can be re-run on other Blackwell consumer hosts, and has CLI flags for arch and matrix size. The deliverable becomes a *reproducible artifact* — a tracked piece of code suitable for RLCR review — rather than an ephemeral set of CSVs from one-off shell commands.

### Approach Summary

Author `scripts/bench_cutlass_5060.sh`, a self-contained Bash orchestrator that drives the full pipeline:

1. **Build phase**: Run CMake from `cutlass/build/` with the user-supplied flags `-DCUTLASS_NVCC_ARCHS="120" -DCUTLASS_LIBRARY_OPERATIONS="gemm" -DCUTLASS_LIBRARY_KERNELS="*sm120*bf16*,*sm120*fp8*"`, then `make -j$(nproc) cutlass_profiler`. Tee all output to a tracked log.
2. **Verification phase** (folded in from Alt-2): After cmake configure but before make, parse the generated `tools/library/generated_kernels.txt` (or equivalent CMake-emitted manifest) to extract the kernel count for the bf16 and fp8 filters. If the count is zero, log the failure mode (with the exact "no kernels matched filter" message) and short-circuit — skip the build entirely and emit a stub CSV/log pair so downstream tabulation still has something to consume.
3. **Kernel-listing phase**: After a successful build, invoke `./cutlass_profiler --operation=Gemm --kernels='cutlass*sm120*' --mode=dry_run` (or the equivalent `--list` flag — to be confirmed from `options.cu` lines around 842-845) to enumerate concrete kernel names. Record the names in the log file so the user can disambiguate the speculative `cutlass*sm120*bf16*gemm*` wildcard from the real schema (`cutlass3x_sm120_tensorop_*` per `generator.py:1047`).
4. **Profiling phase**: Run the two timed profiler invocations with `--m=8192 --n=8192 --k=8192 --warmup-iterations=10 --profiling-iterations=20 --output=<dir>/cutlass_profiler_{bf16,fp8}_8192.csv`. Tee stdout/stderr to matching `.log` files.
5. **Tabulation phase** (folded in from Alt-3): Parse both CSVs, extract `Runtime`, `GFLOPs`, `Status`, `Kernel` columns, identify best-performing kernel per precision, and emit `SUMMARY.md` with a 3×3 comparison table (rows = spec peak / cuBLASLt achieved / CUTLASS achieved; cols = BF16 / FP8).
6. **Output layout**: Default `--output-dir` is `/home/jakeshea/SOLAR/measurements/rtx5060/cutlass/`. The same output dir also receives a `manifest.json` capturing host info (nvidia-smi, nvcc version, cmake version, git HEAD of cutlass clone, script invocation timestamp).
7. **CLI flags**: `--arch=120` (default), `--m=8192 --n=8192 --k=8192` (defaults), `--matrix-size=8192` (sets all three), `--warmup-iterations=10`, `--profiling-iterations=20`, `--output-dir=...`, `--skip-build` (reuse existing binary), `--cutlass-dir=./cutlass`, `--allow-empty-kernels` (downgrade zero-match error to warning).
8. **Error semantics**: Follow `install.sh` / `run_tests.sh` conventions — `set -euo pipefail`, named functions per phase (`build_cutlass()`, `verify_kernels()`, `run_profiler()`, `tabulate()`), explicit `step N:` log markers, exit codes 0 (success), 1 (env/dep error), 2 (kernel-match-zero — only fatal without `--allow-empty-kernels`), 3 (build failure), 4 (profile failure).

The .gitignore receives an addition for `cutlass/build/` and `measurements/rtx5060/cutlass/` (binary artifacts), but the SUMMARY.md and CSV files under `measurements/rtx5060/` should remain tracked.

### Objective Evidence

- `/home/jakeshea/SOLAR/scripts/run_tests.sh` (850+ lines) — uses `set -euo pipefail`, color-coded output helpers, function-based phase organization, exit-code propagation, modular test selection. This is the canonical pattern.
- `/home/jakeshea/SOLAR/scripts/README.md` documents that scripts live under `scripts/` with `.py` (e.g. `collect_perf_results.py`) and `.sh` orchestrators coexisting.
- `/home/jakeshea/SOLAR/install.sh` (130 lines) — uses `set -euo pipefail`, `while [[ $# -gt 0 ]]` argument parsing, step-labeled `echo "==> Step N: ..."` messages, directory existence checks, git commit validation.
- `/home/jakeshea/SOLAR/install_uv.sh` (141 lines) — same idioms, plus `command -v` binary detection, `mkdir -p`, conditional logic (`if [[ "$SKIP_TORCHVIEW" != "true" ]]`).
- `/home/jakeshea/SOLAR/examples/Matmul/run_solar.sh` (86 lines) — `SCRIPT_DIR=$(cd ... && pwd)` path resolution, env var fallback (`${SOLAR_MATMUL_OUTPUT_DIR:-${SCRIPT_DIR}/output}`), pipeline stage echoes, final summary of output paths.
- `/home/jakeshea/SOLAR/docker/Dockerfile` — canonical CUTLASS v4.4.1 build pattern: `git clone --depth 1 -b v4.4.1 https://github.com/NVIDIA/cutlass.git`, installs `cmake`, `ninja-build`, `ccache`.
- `/home/jakeshea/SOLAR/docker/entrypoint.sh` — `trap 'cleanup' EXIT` pattern, explicit error detection and logging, clock-locking wrapper.
- `/home/jakeshea/solbench_leaderboard/sol-execbench/scripts/run_docker.sh` (91 lines) — sibling-repo precedent: env-var configuration (`IMAGE_NAME`, `IMAGE_TAG`), flag-based build triggering, separator-based arg passing (`-- <extra args>`).
- `/home/jakeshea/SOLAR/cutlass/tools/profiler/src/options.cu` lines 493 (`--warmup-iterations` default 10), 494 (`--profiling-iterations` default 100), 702 (`--output`), 843 (`--kernels`) — CLI flags verified.
- `/home/jakeshea/SOLAR/cutlass/tools/profiler/src/performance_report.cpp:335-390` — CSV columns: `Problem,Provider,OperationKind,Operation,Disposition,Status,<arg_names>,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs` — schema known for tabulation step.
- `/home/jakeshea/SOLAR/cutlass/include/cute/arch/mma_sm120.hpp`, `mma_traits_sm120.hpp`, 7 sm_120 collective headers under `include/cutlass/gemm/collective/` — sm_120 GEMM exists in v4.4.1 source.
- `/home/jakeshea/SOLAR/cutlass/python/cutlass_library/emit_kernel_listing.py` lines 421-433 and 505-506 — BF16 (`s1688gemm_bf16`, `gemm_bf16_bf16_f32_bf16_bf16`) and FP8 (`gemm_e4m3_e4m3_f32_bf16_bf16`, `gemm_e4m3_e5m2_f32_bf16_e4m3`) explicitly listed in sm_120 filters.
- `/home/jakeshea/SOLAR/cutlass/python/cutlass_library/generator.py` line 1047 — actual kernel naming scheme: `cutlass3x_sm{ar}_{op}_{ex}{ct}{cs}_{l}_align{al}{t}{k}{e}` — confirms the speculative `cutlass*sm120*` pattern needs verification.
- `/home/jakeshea/SOLAR/cutlass/CMakeLists.txt` line 188 — `sm_120` listed in `CUTLASS_NVCC_ARCHS_SUPPORTED`. Lines 351-353 — `CUTLASS_LIBRARY_KERNELS` / `CUTLASS_LIBRARY_IGNORE_KERNELS` are comma-delimited regex filters.
- `/home/jakeshea/SOLAR/cutlass/.gitignore` includes `/build*`; SOLAR's `.gitignore` line 11 includes `build/`. Pattern coverage for build artifacts is established.
- `/home/jakeshea/SOLAR/configs/arch/5060.yaml` — provides `MAC_per_cycle_bf16_tc=14317` (71.5 TFLOPS at 2.497 GHz boost) and `MAC_per_cycle_fp8_tc=28634` (143 TFLOPS) as the spec baseline for the comparison table.
- `/home/jakeshea/SOLAR/.claude/memory/rtx5060_probe_recipe.md` — records cuBLASLt BF16 = 28 TFLOPS (sm_80 fallback) and FP8 = 106 TFLOPS (sm_120-native) findings to be triangulated against the CUTLASS measurement.
- `/tmp/solar-rtx5060/` — existing ephemeral cuBLASLt + wmma artifacts, ready to migrate into `measurements/rtx5060/cublaslt/` and `measurements/rtx5060/wmma/`.

### Known Risks

- **sm_120 kernel availability in v4.4.1**: The library may instantiate zero kernels for the `*sm120*bf16*` filter even if the source supports sm_120 — Alt-5's evidence quotes the CUTLASS v4.4.2 changelog: *"Enable Blackwell SM120f compilation of examples and exposes NVFP4/MX Grouped GEMM in the CUTLASS Profiler"*, implying v4.4.1's profiler integration may have been incomplete. The verification phase must catch this.
- **Kernel-name mismatch**: The speculative wildcards `cutlass*sm120*bf16*gemm*` won't match the real `cutlass3x_sm120_tensorop_*` scheme. The kernel-listing phase explicitly re-derives the correct pattern from the dry-run output before the timed runs.
- **Build time variance**: With ccache primed the second build is fast; from a cold cache the filtered kernel set may still take 10-20 min. Script must emit progress markers and a final timing summary so RLCR review sees elapsed time per phase.
- **CSV schema fragility**: CUTLASS profiler column order may differ across versions. The tabulator must parse by column name, not index, and emit a warning if expected columns (Runtime, GFLOPs, Status, Kernel) are absent.
- **Dependency assumptions**: Script assumes `cmake`, `make`, `nvcc`, `nvidia-smi` on PATH. Add an explicit preflight that runs `command -v` for each and prints actionable error messages — pattern reused from `install_uv.sh`'s uv-detection block.
- **GeForce-only constraints**: TMA multicast is unavailable on GeForce RTX 5060 (per the Alt-6 evidence in `examples/79_blackwell_geforce_gemm/`); kernels that demand cluster_shape > 1×1×1 will fail to launch. The kernel-listing phase should flag this if any returned kernels are cluster-multicast-only.
- **Concurrent access to `/home/jakeshea/SOLAR/cutlass/build/`**: If the user re-runs the script in parallel sessions, CMake cache pollution is possible. Add a `--clean` flag that wipes `build/` first; default behavior is to reuse.

### Confidence

high

## Alternative Directions Considered

### Alt-1: Direct one-shot build
- Gist: Execute the supplied `cmake` / `make` / `cutlass_profiler` commands as a linear minimal-scaffolding shell pipeline with no intermediate verification — just chain them together, redirect output to `/tmp/solar-rtx5060/`, and document the outcome (success or "no kernels matched"). Bias toward doing exactly what the asker wrote, with the smallest possible code surface (~30 lines of bash).
- Objective Evidence:
  - `/home/jakeshea/SOLAR/cutlass/tools/profiler/CMakeLists.txt` defines `cutlass_profiler` and links `cutlass_lib`.
  - `/home/jakeshea/SOLAR/cutlass/CMakeLists.txt` line 188 — `sm_120` in `CUTLASS_NVCC_ARCHS_SUPPORTED`.
  - `/home/jakeshea/SOLAR/cutlass/include/cute/arch/mma_sm120.hpp` + sparse + blockscaled test units present.
  - 22+ sm_120-specific kernel schedules in `/home/jakeshea/SOLAR/cutlass/python/cutlass_library/library.py` (`Mxf8f6f4TmaWarpSpecializedCooperativeSm120`, `Nvf4TmaWarpSpecializedPingpongSm120`, `BlockwiseTmaWarpSpecializedCooperativeSm120`, etc.).
  - CLI flags verified in `tools/profiler/src/options.cu`: lines 493, 494, 702, 843.
  - Both `bf16` and `e4m3` fully supported in `library.py` with C++ type mappings.
  - Dockerfile precedent: `/home/jakeshea/SOLAR/docker/Dockerfile` shows the canonical clone+build pattern.
- Why not primary: Has no verification stage, so a 15-minute build that produces zero matching kernels would only be discovered at the profiler-invocation step — wasting compute and leaving the cause ambiguous. The primary (D3) wraps this exact command sequence in a tracked, reproducible script with pre-flight verification.

### Alt-2: Verify-first kernel-name reconnaissance
- Gist: Three-stage workflow that does ALL feasible verification BEFORE the build: (Stage 1) source-code recon of v4.4.1's sm_120 kernel templates, (Stage 2) `cmake -DCUTLASS_LIBRARY_KERNELS=...` dry-run that parses the generated kernels manifest to count matches, (Stage 3) post-build `cutlass_profiler --mode=dry_run --kernels=...` for a no-cost kernel enumeration. Short-circuits with a clean "v4.4.1 lacks sm_120 GEMM" verdict if Stage 2 finds zero matches.
- Objective Evidence:
  - `/home/jakeshea/SOLAR/cutlass/python/cutlass_library/emit_kernel_listing.py` lines 505-506 list `s1688gemm_bf16` in sm120_mma_instruction_shapes; lines 421-422 include `gemm_bf16_bf16_f32_bf16_bf16`, `gemm_bf16_bf16_f32_f32_f32`; lines 429-433 include FP8 (`gemm_e4m3_e4m3_f32_bf16_bf16`, etc.); lines 525-526 set the sm120 kernel_filter regex.
  - `/home/jakeshea/SOLAR/cutlass/include/cutlass/gemm/collective/sm120_mma_tma.hpp`, `sm120_blockscaled_mma_tma.hpp` — 7 sm120 collective headers.
  - `/home/jakeshea/SOLAR/cutlass/tools/library/CMakeLists.txt` lines 345-356 — `generator.py --kernels=...` outputs `generated_kernels.txt`, which the verification phase can parse.
  - `/home/jakeshea/SOLAR/cutlass/tools/profiler/src/options.cu` lines 842-843 — `--kernels` flag accepts comma-separated globs; lines 930-945 — `--mode=dry_run` performs enumeration without timing.
  - 4 sm_120 test directories: `test/unit/gemm/device/sm120_tensorop_gemm/`, `sm120_blockscaled_tensorop_gemm/`, `sm120_sparse_tensorop_gemm/`, `sm120_blockscaled_sparse_tensorop_gemm/`.
  - `/home/jakeshea/SOLAR/cutlass/python/cutlass_library/generator.py` line 1047 — naming scheme `cutlass3x_sm{ar}_{op}_{ex}{ct}{cs}_{l}_align{al}{t}{k}{e}` shows real kernel-name pattern is `cutlass3x_sm120_*`, not the speculative `cutlass*sm120*`.
- Why not primary: This is a *prelude*, not a complete plan — the user's idea body explicitly demands the BF16 and FP8 timed runs as deliverables, which Alt-2 doesn't produce on its own. The primary (D3) absorbs Alt-2's three verification stages as the script's pre-build step and continues to the timed runs and tabulation.

### Alt-3: Triangulated `measurements/rtx5060/` bundle
- Gist: Treat the CUTLASS profiler runs as one of three independent measurement methodologies (alongside the already-collected cuBLASLt 13.4 GEMMs and the wmma microbench). Produce a permanent `measurements/rtx5060/` directory in the repo with `cutlass/`, `cublaslt/`, `wmma/`, and a top-level `SUMMARY.md` comparison table. The deliverable centers on *triangulating the BF16 28 TFLOPS-vs-spec gap* across all three sources at once.
- Objective Evidence:
  - `/tmp/solar-rtx5060/` contains existing ephemeral cuBLASLt + wmma artifacts (`cublaslt_bf16_8192.csv`, `cublaslt_fp8_8192.csv`, `ncu_hmma_fp16_*.csv`, `ncu_cublaslt_*.csv`).
  - `/home/jakeshea/solbench_leaderboard/solar-rtx5060/` directory in solbench's `.gitignore` line 42 — sibling precedent for an arch-named local measurement directory.
  - `/home/jakeshea/SOLAR/.claude/memory/rtx5060_probe_recipe.md` already records cuBLASLt findings — needs CUTLASS extension.
  - `/home/jakeshea/SOLAR/configs/arch/5060.yaml` provides spec peaks for the SUMMARY table rows.
  - `/home/jakeshea/SOLAR/cutlass/tools/profiler/src/performance_report.cpp:335-390` — CSV schema known: `Problem,Provider,OperationKind,Operation,Disposition,Status,...,Runtime,GB/s,GFLOPs`.
  - solbench_leaderboard's measurement structure: `data/benchmark/{FlashInfer-Bench,L1,L2,Quant}/NNN_problem_name/{definition.json,reference.py,workload.jsonl}` shows convention.
- Why not primary: This direction is a *presentation framework* — it doesn't address the actual "how to build and run cutlass_profiler" problem. The primary (D3) writes its outputs to exactly the `measurements/rtx5060/cutlass/` layout this alternative proposes, making D3 the *producer* and Alt-3's framework the *target schema*.

### Alt-4: CUTLASS-version fallback ladder
- Gist: Design the workflow as a version ladder: try v4.4.1; if its `*sm120*bf16*` / `*sm120*fp8*` kernel set is empty after configure, automatically retry against v4.4.2, v4.5.0, and finally upstream `main` — recording each attempt. Converts the "v4.4.1 might lack sm_120 GEMM" failure path into a structured search and produces a definitive answer of *which* CUTLASS version first introduces working sm_120 BF16/FP8 GEMM profiling.
- Objective Evidence:
  - `/home/jakeshea/solbench_leaderboard/cutlass/CHANGELOG.md` — v4.4.2 entry: *"Enable Blackwell SM120f compilation of examples and exposes NVFP4/MX Grouped GEMM in the CUTLASS Profiler"* → strong signal that v4.4.1 may lack profiler exposure.
  - v4.5.0 entry: *"Block Scaled MMA for SM120 now works on Spark"* + 128×32/128×64 tile variants — further maturity in 4.5.
  - 42 sm120 `.cu` test files in v4.4.1 — silicon support is present, but profiler integration may not be.
  - `/home/jakeshea/SOLAR/cutlass/customConfigs.cmake` — `cutlass_generate_kernel_filter_and_testlist_files()` invokes `generator.py --architectures=<list> --kernels=<pattern>` → enables version-independent kernel enumeration.
  - `/home/jakeshea/SOLAR/cutlass/CMakeLists.txt` — `CUTLASS_NVCC_ARCHS_SUPPORTED` includes `"120 120a 121 121a"`.
  - Bandwidth: each fallback re-clone (depth 1) is ~150-200 MB; 3-4 fallback steps ≈ ~600-800 MB.
- Why not primary: The primary (D3) keeps v4.4.1 as the singular target the user requested and treats the version ladder as an OPTIONAL `--fallback-versions` flag for future use. The fallback ladder doubles wall-clock time (~30 min vs ~15 min) for a contingency we don't yet need — Alt-2's verification phase already catches the zero-kernel case cheaply, so the version climb only kicks in when an actual fallback is wanted. The script's `--allow-empty-kernels` flag is the simpler escape hatch.

### Alt-5: CuTe DSL microbench complement
- Gist: Rather than relying solely on the cutlass_profiler binary, author standalone CuTe DSL example kernels in `cutlass/examples/cute/tutorial/sm120/{sm120_gemm_bf16.cu, sm120_gemm_fp8.cu}` that directly use `SM120_16x8x32_TN` MMA atoms from `include/cute/arch/mma_sm120.hpp`. Provides an independent measurement path: confirms peak rates regardless of whether the profiler's kernel-name filter matches anything, and can stand alone if the profiler-build path fails.
- Objective Evidence:
  - `/home/jakeshea/SOLAR/cutlass/include/cute/arch/mma_sm120.hpp` — 3278 lines, 80+ `SM120_16x8x32_TN<...>` template specializations: `<float_e4m3_t, float_e4m3_t, float>`, `<float_e4m3_t, float_e3m2_t, float>`, `<float_e2m1_t, float_e4m3_t, float>`, etc.
  - `/home/jakeshea/SOLAR/cutlass/include/cute/atom/mma_traits_sm120.hpp` — block-scaled variants.
  - `/home/jakeshea/SOLAR/cutlass/examples/cute/tutorial/blackwell/01_mma_sm100.cu` (614 lines) — directly portable tutorial pattern.
  - `/home/jakeshea/SOLAR/cutlass/examples/cute/tutorial/hopper/wgmma_sm90.cu` — older reference for the same pattern.
  - `/home/jakeshea/SOLAR/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu` (549 lines) — full `CollectiveBuilder<arch::Sm120, ...>` for NVFP4→BF16 GEMM.
  - `/home/jakeshea/SOLAR/cutlass/include/cutlass/gemm/collective/collective_builder.hpp` line 55 + `sm120_mma_builder.inl` — `CollectiveBuilder<arch::Sm120, arch::OpClassTensorOp, ...>` is wired up in v4.4.1.
  - No existing `examples/cute/tutorial/sm120/` directory — clear greenfield, follow the blackwell/ subdir pattern.
- Why not primary: Authoring ~600 LOC C++ per kernel is excessive for a *cross-check* whose primary purpose is validating numbers from cuBLASLt and the cutlass_profiler. The primary (D3) emphasizes leveraging the existing cutlass_profiler binary. Alt-5 is genuinely orthogonal — it could supplement D3 as a future enhancement if the profiler's BF16 kernel set turns out to be weakly tuned, but it shouldn't be the first attempt.

## Synthesis Notes

The primary direction (D3 — tracked script in `scripts/bench_cutlass_5060.sh`) is designed to *subsume* the strongest mechanisms from each alternative rather than treating them as exclusive. Specifically: **Alt-2's three-stage verification** folds in as the script's pre-build phase, parsing the CMake-emitted kernels manifest to fail fast on a zero-match filter and saving 10-15 minutes of pointless compile time. **Alt-3's output layout** (`measurements/rtx5060/{cutlass,cublaslt,wmma}/SUMMARY.md`) becomes the script's default `--output-dir`, with the tabulation phase emitting the SUMMARY.md table directly. **Alt-1's command sequence** is preserved verbatim inside the script's `build_cutlass()` and `run_profiler()` functions — D3 is essentially Alt-1 wrapped in error handling and CLI flags. **Alt-4's version ladder** is left as an optional `--fallback-versions` flag for users who want the automatic climb to v4.4.2 / v4.5.0; the simpler `--allow-empty-kernels` flag handles the more common "downgrade error to warning" case. **Alt-5's CuTe microbench** is intentionally NOT folded in — it's a separate measurement methodology that doesn't depend on the profiler binary, and authoring 600+ LOC of C++ before knowing whether the cutlass_profiler succeeds is the wrong order. If after the D3 run the BF16 profiler results are still ambiguous, Alt-5 becomes a natural follow-up RLCR iteration.

--- Original Design Draft End ---
