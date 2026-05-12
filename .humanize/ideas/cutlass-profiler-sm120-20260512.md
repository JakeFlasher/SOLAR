# Tracked CUTLASS sm_120 Profiler Build And Benchmark Script

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
