#include "symplectic.cuh"

// Compute angular momentum L = sum(r_i × p_i)
__global__ void compute_angular_momentum_kernel(
    const float3* __restrict__ positions,
    const float3* __restrict__ momenta,
    float3* __restrict__ L_total,
    int N)
{
    extern __shared__ float shared_buf[];
    float* lx = shared_buf;
    float* ly = shared_buf + blockDim.x;
    float* lz = shared_buf + 2 * blockDim.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float my_lx = 0.0f, my_ly = 0.0f, my_lz = 0.0f;
    if (i < N) {
        float3 r = positions[i];
        float3 p = momenta[i];
        my_lx = r.y * p.z - r.z * p.y;
        my_ly = r.z * p.x - r.x * p.z;
        my_lz = r.x * p.y - r.y * p.x;
    }
    lx[tid] = my_lx;
    ly[tid] = my_ly;
    lz[tid] = my_lz;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            lx[tid] += lx[tid + s];
            ly[tid] += ly[tid + s];
            lz[tid] += lz[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&L_total[0].x, lx[0]);
        atomicAdd(&L_total[0].y, ly[0]);
        atomicAdd(&L_total[0].z, lz[0]);
    }
}

void launch_compute_angular_momentum(
    const float3* d_positions, const float3* d_momenta,
    float3* d_L_total, int N, cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    int sharedSize = 3 * blockSize * sizeof(float);
    compute_angular_momentum_kernel<<<gridSize, blockSize, sharedSize, stream>>>(
        d_positions, d_momenta, d_L_total, N);
}
