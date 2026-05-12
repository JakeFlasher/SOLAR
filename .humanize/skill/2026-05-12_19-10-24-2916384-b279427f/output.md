**Recommendation**

Run FP16 once, but treat it only as a consistency check. On 5th-gen Tensor Cores, dense FP16 and BF16 peak should match. For NCU, the key consistency signal is that `sm__inst_executed_pipe_tensor.sum.peak_sustained` and `.peak_sustained_active` match your BF16 value, about `6.50 inst/cycle` aggregate on 26 SMs. If FP16 reports materially different peak, assume probe/profiling issue first.

For FP8, do not spend time on WMMA or hand inline PTX unless you specifically want an instruction-fragment experiment. PTX supports FP8 `mma.sync` shapes, but the operand packing is low-level and easy to get wrong. NVIDIA’s PTX ISA documents `.e4m3/.e5m2` `mma.sync` support and the packed `.b32` fragment layout; it also says FP8 MMA requires `sm_89+`, which includes `sm_120` [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). The pragmatic 5-15 minute path is cuBLASLt FP8 GEMM with NCU attached.

Do not build `cutlass_profiler` for this. It is the wrong time/benefit tradeoff here, and the proposed kernel name is probably not valid. Use cuBLASLt BF16 8192³ as the end-to-end check. For RTX 5060 Laptop, with 71.5 TFLOPS dense spec peak, I would expect roughly `50-64 TFLOPS` if power/thermals are favorable. `55+` is a good result; `<40` suggests clocks, power limit, thermals, or a poor heuristic.

NVIDIA cuBLASLt docs require FP8 tensor kernels to use `CUBLAS_COMPUTE_32F`, `CUDA_R_32F` scale type, 16-byte-compatible alignment, and TN layout on Blackwell GeForce, with supported `CUDA_R_8F_E4M3` operand combinations listed in the FP8 table [cuBLAS 13.2 docs](https://docs.nvidia.com/cuda/cublas/index.html). NVIDIA’s cuBLASLt FP8 sample uses the same scale-pointer attributes [CUDALibrarySamples LtFp8Matmul](https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtFp8Matmul/sample_cublasLt_LtFp8Matmul.cu).

Run this with `bash`:

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT=/tmp/solar-rtx5060
mkdir -p "$OUT"
cd "$OUT"

export PATH=/opt/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}

METRICS="gpu__time_duration.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_tensor.sum.peak_sustained,sm__inst_executed_pipe_tensor.sum.peak_sustained_active,sm__inst_executed_pipe_tensor.sum.per_cycle_active"

cat > probe_fp16.cu <<'CU'
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace nvcuda;

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__global__ void hmma_fp16(float *out, int iters) {
  int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

  wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16,16,16, float> c0, c1, c2, c3;

  wmma::fill_fragment(a, __float2half(1.0f));
  wmma::fill_fragment(b, __float2half(1.0f));
  wmma::fill_fragment(c0, 0.0f);
  wmma::fill_fragment(c1, 0.0f);
  wmma::fill_fragment(c2, 0.0f);
  wmma::fill_fragment(c3, 0.0f);

  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    wmma::mma_sync(c0, a, b, c0);
    wmma::mma_sync(c1, a, b, c1);
    wmma::mma_sync(c2, a, b, c2);
    wmma::mma_sync(c3, a, b, c3);
  }

  wmma::store_matrix_sync(out + warp_global * 256, c0, 16, wmma::mem_row_major);
}

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 20000;
  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  int threads = 128;
  int blocks = sms * 16;
  size_t warps = (size_t)blocks * threads / 32;
  float *out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, warps * 256 * sizeof(float)));

  hmma_fp16<<<blocks, threads>>>(out, 10);
  CUDA_CHECK(cudaDeviceSynchronize());

  hmma_fp16<<<blocks, threads>>>(out, iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("fp16_hmma,sm_count,blocks,threads,warps,iters\n");
  printf("fp16_hmma,%d,%d,%d,%zu,%d\n", sms, blocks, threads, warps, iters);
  CUDA_CHECK(cudaFree(out));
  return 0;
}
CU

cat > lt_gemm.cu <<'CU'
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS %s:%d: status %d\n",__FILE__,__LINE__,(int)s); exit(2);} } while(0)

__global__ void fill_bf16(__nv_bfloat16 *p, size_t n, float v) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2bfloat16(v);
}
__global__ void fill_fp8(__nv_fp8_e4m3 *p, size_t n, float v) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __nv_fp8_e4m3(v);
}

static void fill_all_bf16(__nv_bfloat16 *p, size_t n, float v) {
  int t = 256; int b = (int)((n + t - 1) / t);
  fill_bf16<<<b,t>>>(p,n,v); CUDA_CHECK(cudaGetLastError());
}
static void fill_all_fp8(__nv_fp8_e4m3 *p, size_t n, float v) {
  int t = 256; int b = (int)((n + t - 1) / t);
  fill_fp8<<<b,t>>>(p,n,v); CUDA_CHECK(cudaGetLastError());
}

static cublasLtMatmulHeuristicResult_t get_algo(
    cublasLtHandle_t lt, cublasLtMatmulDesc_t op,
    cublasLtMatrixLayout_t A, cublasLtMatrixLayout_t B,
    cublasLtMatrixLayout_t C, cublasLtMatrixLayout_t D,
    void *workspace, size_t workspace_bytes) {
  cublasLtMatmulPreference_t pref;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes)));
  cublasLtMatmulHeuristicResult_t h = {};
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, op, A, B, C, D, pref, 1, &h, &returned));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  if (returned == 0 || h.state != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "No cuBLASLt heuristic found\n");
    exit(3);
  }
  return h;
}

static void run_bf16(int dim, int iters) {
  int m=dim,n=dim,k=dim, warmup=5;
  size_t szA=(size_t)m*k, szB=(size_t)k*n, szC=(size_t)m*n;
  __nv_bfloat16 *A,*B,*C,*D;
  CUDA_CHECK(cudaMalloc(&A, szA*sizeof(*A)));
  CUDA_CHECK(cudaMalloc(&B, szB*sizeof(*B)));
  CUDA_CHECK(cudaMalloc(&C, szC*sizeof(*C)));
  CUDA_CHECK(cudaMalloc(&D, szC*sizeof(*D)));
  fill_all_bf16(A,szA,1.0f); fill_all_bf16(B,szB,1.0f);
  fill_all_bf16(C,szC,0.0f); fill_all_bf16(D,szC,0.0f);
  CUDA_CHECK(cudaDeviceSynchronize());

  cublasLtHandle_t lt; CUBLAS_CHECK(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op; CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t nt = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &nt, sizeof(nt)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nt, sizeof(nt)));

  cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16BF, m, k, m));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16BF, k, n, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, m, n, m));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_16BF, m, n, m));

  size_t ws_bytes = 64ull << 20; void *ws;
  CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
  auto h = get_algo(lt,op,Ad,Bd,Cd,Dd,ws,ws_bytes);
  float alpha = 1.0f / (float)k, beta = 0.0f;

  auto matmul = [&](){
    CUBLAS_CHECK(cublasLtMatmul(lt,op,&alpha,A,Ad,B,Bd,&beta,C,Cd,D,Dd,&h.algo,ws,ws_bytes,0));
  };

  for(int i=0;i<warmup;i++) matmul();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t st, en; CUDA_CHECK(cudaEventCreate(&st)); CUDA_CHECK(cudaEventCreate(&en));
  CUDA_CHECK(cudaEventRecord(st));
  for(int i=0;i<iters;i++) matmul();
  CUDA_CHECK(cudaEventRecord(en)); CUDA_CHECK(cudaEventSynchronize(en));
  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,st,en));
  double avg_ms = ms / iters;
  double tflops = (2.0 * m * n * k) / (avg_ms * 1e9);

  printf("mode,m,n,k,warmup,iters,avg_ms,tflops,heur_waves,workspace_bytes\n");
  printf("bf16,%d,%d,%d,%d,%d,%.6f,%.3f,%.3f,%zu\n",m,n,k,warmup,iters,avg_ms,tflops,h.wavesCount,ws_bytes);

  if (getenv("SOLAR_PROFILE_ONE")) { cudaProfilerStart(); matmul(); cudaProfilerStop(); CUDA_CHECK(cudaDeviceSynchronize()); }

  cudaFree(ws); cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D);
  cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd); cublasLtMatrixLayoutDestroy(Dd);
  cublasLtMatmulDescDestroy(op); cublasLtDestroy(lt);
}

static void run_fp8(int dim, int iters) {
  int m=dim,n=dim,k=dim, warmup=5;
  size_t szA=(size_t)k*m, szB=(size_t)k*n, szC=(size_t)m*n;
  __nv_fp8_e4m3 *A,*B,*D; __nv_bfloat16 *C;
  CUDA_CHECK(cudaMalloc(&A, szA*sizeof(*A)));
  CUDA_CHECK(cudaMalloc(&B, szB*sizeof(*B)));
  CUDA_CHECK(cudaMalloc(&C, szC*sizeof(*C)));
  CUDA_CHECK(cudaMalloc(&D, szC*sizeof(*D)));
  fill_all_fp8(A,szA,1.0f); fill_all_fp8(B,szB,1.0f);
  fill_all_bf16(C,szC,0.0f); fill_all_fp8(D,szC,0.0f);
  CUDA_CHECK(cudaDeviceSynchronize());

  float one_h=1.0f, *one_d, *amax_d;
  CUDA_CHECK(cudaMalloc(&one_d,sizeof(float)));
  CUDA_CHECK(cudaMalloc(&amax_d,sizeof(float)));
  CUDA_CHECK(cudaMemcpy(one_d,&one_h,sizeof(float),cudaMemcpyHostToDevice));

  cublasLtHandle_t lt; CUBLAS_CHECK(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op; CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, nb = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nb, sizeof(nb)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &one_d, sizeof(one_d)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &one_d, sizeof(one_d)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &one_d, sizeof(one_d)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &one_d, sizeof(one_d)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d)));

  cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, k, m, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, k, n, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, m, n, m));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_8F_E4M3, m, n, m));

  size_t ws_bytes = 64ull << 20; void *ws;
  CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
  auto h = get_algo(lt,op,Ad,Bd,Cd,Dd,ws,ws_bytes);
  float alpha = 1.0f / (float)k, beta = 0.0f;

  auto matmul = [&](){
    CUBLAS_CHECK(cublasLtMatmul(lt,op,&alpha,A,Ad,B,Bd,&beta,C,Cd,D,Dd,&h.algo,ws,ws_bytes,0));
  };

  for(int i=0;i<warmup;i++) matmul();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t st, en; CUDA_CHECK(cudaEventCreate(&st)); CUDA_CHECK(cudaEventCreate(&en));
  CUDA_CHECK(cudaEventRecord(st));
  for(int i=0;i<iters;i++) matmul();
  CUDA_CHECK(cudaEventRecord(en)); CUDA_CHECK(cudaEventSynchronize(en));
  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,st,en));
  double avg_ms = ms / iters;
  double tflops = (2.0 * m * n * k) / (avg_ms * 1e9);

  printf("mode,m,n,k,warmup,iters,avg_ms,tflops,heur_waves,workspace_bytes\n");
  printf("fp8_e4m3,%d,%d,%d,%d,%d,%.6f,%.3f,%.3f,%zu\n",m,n,k,warmup,iters,avg_ms,tflops,h.wavesCount,ws_bytes);

  if (getenv("SOLAR_PROFILE_ONE")) { cudaProfilerStart(); matmul(); cudaProfilerStop(); CUDA_CHECK(cudaDeviceSynchronize()); }

  cudaFree(ws); cudaFree(one_d); cudaFree(amax_d); cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D);
  cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd); cublasLtMatrixLayoutDestroy(Dd);
  cublasLtMatmulDescDestroy(op); cublasLtDestroy(lt);
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr,"usage: %s bf16|fp8 [dim] [iters]\n", argv[0]); return 1; }
  int dim = argc > 2 ? atoi(argv[2]) : 8192;
  int iters = argc > 3 ? atoi(argv[3]) : 20;
  if (!strcmp(argv[1],"bf16")) run_bf16(dim,iters);
  else if (!strcmp(argv[1],"fp8")) run_fp8(dim,iters);
  else { fprintf(stderr,"unknown mode\n"); return 1; }
  return 0;
}
CU

nvcc -O3 -lineinfo -arch=sm_120 -o probe_fp16 probe_fp16.cu
nvcc -O3 -lineinfo -std=c++17 -arch=sm_120 -o lt_gemm lt_gemm.cu -lcublasLt -lcublas

./probe_fp16 20000 > fp16_probe_app.csv

ncu --force-overwrite --csv --log-file ncu_hmma_fp16.csv \
  --launch-skip 1 --launch-count 1 \
  --metrics "$METRICS" \
  ./probe_fp16 20000 > ncu_hmma_fp16_app.csv 2> ncu_hmma_fp16.err

./lt_gemm fp8 8192 20 > cublaslt_fp8_8192.csv
SOLAR_PROFILE_ONE=1 ncu --force-overwrite --csv --log-file ncu_cublaslt_fp8.csv \
  --profile-from-start off \
  --metrics "$METRICS" \
  ./lt_gemm fp8 8192 1 > ncu_cublaslt_fp8_app.csv 2> ncu_cublaslt_fp8.err

./lt_gemm bf16 8192 20 > cublaslt_bf16_8192.csv

ls -lh "$OUT"/*.csv "$OUT"/*.err
```

Main critique of the other worker: FP8 via WMMA will not compile, `cutlass_profiler` is not available and the kernel name is hand-wavy, `--target-processes all` is unnecessary, and the NCU metric list should include `.peak_sustained_active`. For FP8 NCU, expect the tensor instruction peak metric to look similar in inst/cycle to FP16/BF16; FP8’s doubled dense FLOPS comes from larger work per MMA instruction, so the cuBLASLt event TFLOPS is the cleaner end-to-end FP8 number.
