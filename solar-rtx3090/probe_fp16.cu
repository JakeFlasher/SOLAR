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
