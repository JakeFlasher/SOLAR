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

#define MAX_ALGOS 32
static int g_num_algos = 0;
static cublasLtMatmulHeuristicResult_t g_hs[MAX_ALGOS];

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
  g_num_algos = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, op, A, B, C, D, pref, MAX_ALGOS, g_hs, &g_num_algos));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  if (g_num_algos == 0) {
    fprintf(stderr, "No cuBLASLt heuristic found\n");
    exit(3);
  }
  fprintf(stderr, "cuBLASLt: heuristic returned %d algos\n", g_num_algos);
  return g_hs[0];
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
  cublasOperation_t ta = CUBLAS_OP_T, nb = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nb, sizeof(nb)));

  cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16BF, k, m, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16BF, k, n, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, m, n, m));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_16BF, m, n, m));

  size_t ws_bytes = 256ull << 20; void *ws;
  CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
  auto h = get_algo(lt,op,Ad,Bd,Cd,Dd,ws,ws_bytes);
  float alpha = 1.0f / (float)k, beta = 0.0f;

  // Try every returned algo; record fastest
  int best_idx = -1; float best_ms = 1e30f;
  for (int a = 0; a < g_num_algos; ++a) {
    auto algo = g_hs[a].algo;
    // skip if reported workspace > our allocation
    if (g_hs[a].workspaceSize > ws_bytes) { fprintf(stderr,"  algo[%d] needs %zuB workspace; skipped\n", a, g_hs[a].workspaceSize); continue; }
    // try-run with small warmup; ignore failures
    cublasStatus_t s = cublasLtMatmul(lt,op,&alpha,A,Ad,B,Bd,&beta,C,Cd,D,Dd,&algo,ws,ws_bytes,0);
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"  algo[%d] launch failed; skipped\n", a); continue; }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s1,e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
    cudaEventRecord(s1);
    for (int j=0;j<3;j++) cublasLtMatmul(lt,op,&alpha,A,Ad,B,Bd,&beta,C,Cd,D,Dd,&algo,ws,ws_bytes,0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_a=0; cudaEventElapsedTime(&ms_a,s1,e1);
    fprintf(stderr,"  algo[%d] waves=%.2f ws=%zu avg_ms=%.3f tflops=%.2f\n", a, g_hs[a].wavesCount, g_hs[a].workspaceSize, ms_a/3, (2.0*m*n*k)/((ms_a/3)*1e9));
    if (ms_a < best_ms) { best_ms = ms_a; best_idx = a; }
    cudaEventDestroy(s1); cudaEventDestroy(e1);
  }
  if (best_idx < 0) { fprintf(stderr,"All algos failed\n"); exit(4); }
  h = g_hs[best_idx];
  fprintf(stderr,"BEST: algo[%d] waves=%.2f\n", best_idx, h.wavesCount);

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
