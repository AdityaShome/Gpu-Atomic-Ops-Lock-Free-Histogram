/*
 * three atomic strategies for histogram
 * nvcc -O3 -arch=native -allow-unsupported-compiler main.cu -o hist_bench
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define CUDA_CHECK(x) do {                                              \
    cudaError_t rc = (x);                                               \
    if (rc != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA %s:%d  %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(rc));            \
        exit(1);                                                        \
    }                                                                   \
} while (0)

constexpr int BINS = 256;
constexpr int BLK  = 256;

// global only atomics
__global__ void hist_global(const unsigned char *img, int *hist, size_t n)
{
    size_t idx = (size_t)blockIdx.x * BLK + threadIdx.x;
    if (idx < n)
        atomicAdd(&hist[img[idx]], 1);
}

// shared mem atomics, then merge
__global__ void hist_shared(const unsigned char *img, int *hist, size_t n)
{
    __shared__ int local[BINS];
    if (threadIdx.x < BINS) local[threadIdx.x] = 0;
    __syncthreads();
    size_t idx = (size_t)blockIdx.x * BLK + threadIdx.x;
    if (idx < n)
        atomicAdd(&local[img[idx]], 1);
    __syncthreads();
    if (threadIdx.x < BINS)
        atomicAdd(&hist[threadIdx.x], local[threadIdx.x]);
}

// warp aggregate, then shared, then global
__global__ void hist_warp(const unsigned char *img, int *hist, size_t n)
{
    __shared__ int local[BINS];
    if (threadIdx.x < BINS) local[threadIdx.x] = 0;
    __syncthreads();
    size_t idx = (size_t)blockIdx.x * BLK + threadIdx.x;
    if (idx < n) {
        unsigned char v = img[idx];
        unsigned mask = __match_any_sync(0xFFFFFFFF, v);
        int leader = __ffs(mask) - 1;
        if ((int)(threadIdx.x & 31) == leader)
            atomicAdd(&local[v], __popc(mask));
    }
    __syncthreads();
    if (threadIdx.x < BINS)
        atomicAdd(&hist[threadIdx.x], local[threadIdx.x]);
}

static bool check(const int *ref, const int *got, const char *tag)
{
    for (int i = 0; i < BINS; i++) {
        if (ref[i] != got[i]) {
            fprintf(stderr, "%s FAIL bin %d: expected %d got %d\n",
                    tag, i, ref[i], got[i]);
            return false;
        }
    }
    printf("  %-14s  PASS\n", tag);
    return true;
}

typedef void (*kernel_t)(const unsigned char*, int*, size_t);

static double bench(kernel_t kern, const unsigned char *d_img,
                    int *d_hist, size_t n, int nblk,
                    const int *cpu_ref, const char *tag)
{
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    kern<<<nblk, BLK>>>(d_img, d_hist, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int h[BINS];
    CUDA_CHECK(cudaMemcpy(h, d_hist, sizeof h, cudaMemcpyDeviceToHost));
    check(cpu_ref, h, tag);
    printf("  %-14s  %.2f ms\n\n", tag, ms);
    return ms;
}

int main()
{
    const size_t N = 1ULL << 28;
    printf("pixels: %zu  (~%zu MiB)\n\n", N, N >> 20);
    unsigned char *h_img = (unsigned char *)malloc(N);
    for (size_t i = 0; i < N; i++)
        h_img[i] = (unsigned char)((i / 16) % BINS);
    int cpu[BINS] = {};
    for (size_t i = 0; i < N; i++)
        cpu[h_img[i]]++;
    unsigned char *d_img;
    int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_img, N));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, N, cudaMemcpyHostToDevice));
    int nblk = (int)((N + BLK - 1) / BLK);
    bench(hist_global, d_img, d_hist, N, nblk, cpu, "global");
    bench(hist_shared, d_img, d_hist, N, nblk, cpu, "shared");
    bench(hist_warp,   d_img, d_hist, N, nblk, cpu, "warp");
    cudaFree(d_img);
    cudaFree(d_hist);
    free(h_img);
}
