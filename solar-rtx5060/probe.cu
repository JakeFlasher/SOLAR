#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA err: %s\n",cudaGetErrorString(e)); exit(1);} } while(0)

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

// HMMA tile loop -- exercises bf16 tensor pipe so ncu can read tensor-pipe peak
#include <mma.h>
using namespace nvcuda::wmma;
__global__ void hmma_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int reps) {
  fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
  fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b;
  fragment<accumulator, 16, 16, 16, float> c;
  fill_fragment(c, 0.0f);
  load_matrix_sync(a, A, 16);
  load_matrix_sync(b, B, 16);
  for (int r=0; r<reps; ++r) mma_sync(c, a, b, c);
  if (threadIdx.x==0 && blockIdx.x==0) store_matrix_sync(C, c, 16, mem_row_major);
}

int main(int argc, char** argv) {
  cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p,0));
  int clockRateKHz=0, memClockKHz=0, busWidthBits=0, l2BytesAttr=0;
  cudaDeviceGetAttribute(&clockRateKHz,    cudaDevAttrClockRate,             0);
  cudaDeviceGetAttribute(&memClockKHz,     cudaDevAttrMemoryClockRate,       0);
  cudaDeviceGetAttribute(&busWidthBits,    cudaDevAttrGlobalMemoryBusWidth,  0);
  cudaDeviceGetAttribute(&l2BytesAttr,     cudaDevAttrL2CacheSize,           0);
  if (argc < 2 || !strcmp(argv[1],"props")) {
    printf("name=%s\n", p.name);
    printf("totalGlobalMem=%zu\n", p.totalGlobalMem);
    printf("l2CacheSize=%d\n", p.l2CacheSize);
    printf("l2CacheSize_attr=%d\n", l2BytesAttr);
    printf("multiProcessorCount=%d\n", p.multiProcessorCount);
    printf("major=%d\n", p.major);
    printf("minor=%d\n", p.minor);
    printf("clockRateKHz=%d\n", clockRateKHz);
    printf("memoryClockRateKHz=%d\n", memClockKHz);
    printf("memoryBusWidthBits=%d\n", busWidthBits);
    printf("memoryBusWidth_prop=%d\n", p.memoryBusWidth);
    printf("sharedMemPerBlock=%zu\n", p.sharedMemPerBlock);
    printf("sharedMemPerMultiprocessor=%zu\n", p.sharedMemPerMultiprocessor);
    printf("warpSize=%d\n", p.warpSize);
    return 0;
  }
  size_t bytes = !strcmp(argv[1],"l2") ? (16ull<<20) : (256ull<<20);
  int iters = !strcmp(argv[1],"l2") ? 256 : 8;
  size_t n = bytes / sizeof(float4);
  float4 *a=nullptr, *b=nullptr; unsigned long long *sink=nullptr;
  CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMalloc(&sink, 8));
  CK(cudaMemset(a, 1, bytes)); CK(cudaMemset(b, 0, bytes)); CK(cudaMemset(sink, 0, 8));
  dim3 block(256), grid(p.multiProcessorCount * 8);

  if (!strcmp(argv[1],"hmma")) {
    __nv_bfloat16 *A=nullptr,*B=nullptr; float *C=nullptr;
    CK(cudaMalloc(&A, 16*16*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&B, 16*16*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&C, 16*16*sizeof(float)));
    CK(cudaMemset(A, 0, 16*16*sizeof(__nv_bfloat16)));
    CK(cudaMemset(B, 0, 16*16*sizeof(__nv_bfloat16)));
    dim3 hblock(32), hgrid(p.multiProcessorCount * 4);
    for (int i=0; i<7; ++i) hmma_kernel<<<hgrid,hblock>>>(A,B,C,4096);
    CK(cudaDeviceSynchronize());
    return 0;
  }
  for (int i=0; i<7; ++i) {
    if (!strcmp(argv[1],"l2")) l2_kernel<<<grid,block>>>(a,sink,n,iters);
    else dram_copy_kernel<<<grid,block>>>(a,b,n,iters);
  }
  CK(cudaDeviceSynchronize());
  return 0;
}
