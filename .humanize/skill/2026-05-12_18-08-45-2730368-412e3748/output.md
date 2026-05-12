Use `cudaDeviceProp` for static properties, not NVML and not torch. It directly exposes `l2CacheSize`, `totalGlobalMem`, `memoryClockRate`, `memoryBusWidth`, SM count, and compute capability. I would run `bash install_uv.sh --skip-torchview` first only because you need SOLAR usable anyway; the profiling recipe below does not depend on torch.

Key conventions I confirmed in [perf_model.py](/home/jakeshea/SOLAR/solar/perf/perf_model.py:176): SOLAR uses whole-device `MAC_per_cycle_*`, whole-device `DRAM_byte_per_cycle`, and `freq_GHz` only to convert cycles to runtime. So bandwidth, MAC/cycle, and frequency must be paired.

**Important corrections**
- Current NVIDIA RTX 50 Laptop specs list RTX 5060 Laptop as 3328 CUDA cores, 1455-2497 MHz boost, 8 GB GDDR7, 384 GB/s bandwidth, and 572 AI TOPS, not 448 GB/s. If your local `memoryClockRate` reports 28 Gbps-equivalent, use the local 448 GB/s value. NVIDIA source: https://www.nvidia.com/en-us/geforce/laptops/50-series/
- `B200.yaml` is using dense Blackwell tensor ratios for tensor fields. `H100_PCIe.yaml` appears to use sparse Hopper tensor numbers, so do not copy that convention for 5060.
- Do not drop `MAC_per_cycle_nvfp4_tc`; SOLAR maps `fp4`/`nvfp4` metadata to that key, and RTX 50 Blackwell Tensor Cores support FP4. Source: https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/
- No public NVIDIA spec gives RTX 5060 Laptop L2 bandwidth. Measure `lts__t_bytes` with NCU or use NCU’s peak-sustained metric if exposed.

**NCU Metrics**
Use hand-picked metrics, not `--set full`. `--set roofline` is also too large for this task.

Validate suffixes first:

```bash
ncu --query-metrics-mode suffix --metrics dram__bytes_read,dram__bytes_write,lts__t_bytes,sm__inst_executed_pipe_tensor
```

Use these first:
- DRAM achieved: `dram__bytes_read.sum.per_second`, `dram__bytes_write.sum.per_second`
- L2 achieved: `lts__t_bytes.sum.per_second`
- L2 peak if available: `lts__t_bytes.sum.peak_sustained_active.per_second`
- Tensor instruction rate: `sm__inst_executed_pipe_tensor.sum.per_cycle_active`
- Tensor peak if available: `sm__inst_executed_pipe_tensor.sum.peak_sustained_active`

Nsight Compute metric suffix rules are documented here: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

**Run This**
From `/home/jakeshea/SOLAR`, paste this into `bash`:

```bash
bash install_uv.sh --skip-torchview

bash -s <<'BASH'
set -euo pipefail
OUT=/tmp/solar-rtx5060
mkdir -p "$OUT"

cat > "$OUT/probe.cu" <<'CU'
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(1);} } while(0)

__global__ void dram_copy_kernel(const float4* __restrict__ a, float4* __restrict__ b, size_t n, int iters) {
  for (int r=0; r<iters; ++r)
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x) b[i]=a[i];
}
__global__ void l2_kernel(float4* a, unsigned long long* sink, size_t n, int iters) {
  unsigned long long s=0;
  for (int r=0; r<iters; ++r)
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x) {
      float4 v=a[i]; v.x+=1.0f; a[i]=v; s += (unsigned long long)v.x;
    }
  atomicAdd(sink, s);
}
int main(int argc, char** argv) {
  cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p,0));
  if (argc < 2 || !strcmp(argv[1],"props")) {
    printf("name=%s\n", p.name);
    printf("totalGlobalMem=%zu\n", p.totalGlobalMem);
    printf("l2CacheSize=%d\n", p.l2CacheSize);
    printf("multiProcessorCount=%d\n", p.multiProcessorCount);
    printf("major=%d\nminor=%d\n", p.major, p.minor);
    printf("clockRateKHz=%d\n", p.clockRate);
    printf("memoryClockRateKHz=%d\n", p.memoryClockRate);
    printf("memoryBusWidthBits=%d\n", p.memoryBusWidth);
    return 0;
  }
  size_t bytes = !strcmp(argv[1],"l2") ? (16ull<<20) : (256ull<<20);
  int iters = !strcmp(argv[1],"l2") ? 256 : 8;
  size_t n = bytes / sizeof(float4);
  float4 *a, *b; unsigned long long *sink;
  CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMalloc(&sink, 8));
  CK(cudaMemset(a, 1, bytes)); CK(cudaMemset(b, 0, bytes)); CK(cudaMemset(sink, 0, 8));
  dim3 block(256), grid(p.multiProcessorCount * 8);
  for (int i=0; i<7; ++i) {
    if (!strcmp(argv[1],"l2")) l2_kernel<<<grid,block>>>(a,sink,n,iters);
    else dram_copy_kernel<<<grid,block>>>(a,b,n,iters);
  }
  CK(cudaDeviceSynchronize());
}
CU

/opt/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 "$OUT/probe.cu" -o "$OUT/probe"

"$OUT/probe" props | tee "$OUT/props.txt"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.memory,power.limit --format=csv,noheader,nounits | tee "$OUT/nvidia-smi.txt"

ncu --csv --page raw --print-units base --print-metric-name name \
  --kernel-name-base function -k regex:dram_copy_kernel --launch-skip 5 --launch-count 1 \
  --metrics dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,dram__bytes_read.sum,dram__bytes_write.sum \
  "$OUT/probe" dram > "$OUT/ncu_dram.csv" || true

ncu --csv --page raw --print-units base --print-metric-name name \
  --kernel-name-base function -k regex:l2_kernel --launch-skip 5 --launch-count 1 \
  --metrics lts__t_bytes.sum.per_second,lts__t_bytes.sum,lts__t_bytes.sum.peak_sustained_active.per_second \
  "$OUT/probe" l2 > "$OUT/ncu_l2.csv" || true

python3 - "$OUT" <<'PY'
import csv, re, sys, pathlib
out = pathlib.Path(sys.argv[1])
props = dict(line.strip().split("=",1) for line in (out/"props.txt").read_text().splitlines() if "=" in line)
smi = (out/"nvidia-smi.txt").read_text().splitlines()[0].split(",")
freq = float(smi[2].strip()) / 1000.0 if len(smi) >= 3 else int(props["clockRateKHz"]) / 1e6
dram_bps = 2 * int(props["memoryClockRateKHz"]) * 1000 * (int(props["memoryBusWidthBits"]) / 8)
dram_bpc = dram_bps / (freq * 1e9)

def metric(path, name):
    if not path.exists(): return None
    for row in csv.reader(path.open()):
        if len(row) >= 2 and row[-2] == name:
            try: return float(row[-1].replace(",",""))
            except: pass
    return None

l2_peak = metric(out/"ncu_l2.csv", "lts__t_bytes.sum.peak_sustained_active.per_second")
l2_obs = metric(out/"ncu_l2.csv", "lts__t_bytes.sum.per_second")
sram_bpc = (l2_peak or l2_obs or (8 * dram_bps)) / (freq * 1e9)

# NVIDIA RTX 5060 Laptop: 572 AI TOPS. Treat that as sparse FP4; SOLAR/B200 convention wants dense.
ref_boost = 2.497
scale = freq / ref_boost
tf32 = (572/16) * scale
fp16 = (572/8) * scale
fp8  = (572/4) * scale
fp4  = (572/2) * scale
mac = lambda tflops: round(tflops / (2 * freq) * 1000)

print(f'''name: "RTX_5060_Laptop"
SRAM_capacity: {props["l2CacheSize"]}
SRAM_byte_per_cycle: {sram_bpc:.1f}
DRAM_capacity: {props["totalGlobalMem"]}
DRAM_byte_per_cycle: {dram_bpc:.1f}
freq_GHz: {freq:.3f}
MAC_per_cycle_fp32_sm: {int(props["multiProcessorCount"]) * 128}
MAC_per_cycle_int8_tc: {mac(fp8)}
MAC_per_cycle_tf32_tc: {mac(tf32)}
MAC_per_cycle_fp16_tc: {mac(fp16)}
MAC_per_cycle_bf16_tc: {mac(fp16)}
MAC_per_cycle_fp8_tc: {mac(fp8)}
MAC_per_cycle_nvfp4_tc: {mac(fp4)}
''')
print(f"raw outputs: {out}", file=sys.stderr)
PY
BASH
```

If NCU cannot collect the L2 peak metric, the script falls back to observed L2 throughput, then to `8 * DRAM` as a last-resort placeholder. Paste back `/tmp/solar-rtx5060/props.txt`, `nvidia-smi.txt`, and the printed YAML; the only field I would expect to tune is `SRAM_byte_per_cycle`.
