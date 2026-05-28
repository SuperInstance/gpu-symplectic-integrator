#include "symplectic.cuh"

// Compute total energy: KE + PE
__global__ void compute_energy_kernel(
    const float3* __restrict__ positions,
    const float3* __restrict__ momenta,
    const float* __restrict__ masses,
    float* __restrict__ energy,  // output: one scalar per system
    int N, float softening)
{
    // Use shared memory for partial sums
    extern __shared__ float shared_buf[];
    float* ke_parts = shared_buf;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float my_ke = 0.0f;
    if (i < N) {
        float inv_m = 1.0f / masses[i];
        my_ke = 0.5f * (momenta[i].x*momenta[i].x + momenta[i].y*momenta[i].y + momenta[i].z*momenta[i].z) * inv_m;
    }
    ke_parts[tid] = my_ke;
    __syncthreads();

    // Reduction for KE
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ke_parts[tid] += ke_parts[tid + s];
        __syncthreads();
    }

    // Thread 0 computes PE and writes total energy
    if (tid == 0) {
        float ke_total = ke_parts[0];
        float pe = 0.0f;
        for (int ii = 0; ii < N; ii++) {
            for (int jj = ii + 1; jj < N; jj++) {
                float dx = positions[jj].x - positions[ii].x;
                float dy = positions[jj].y - positions[ii].y;
                float dz = positions[jj].z - positions[ii].z;
                float r = sqrtf(dx*dx + dy*dy + dz*dz + softening*softening);
                pe -= G_CONST * masses[ii] * masses[jj] / r;
            }
        }
        energy[blockIdx.x] = ke_total + pe;
    }
}

void launch_compute_energy(
    const float3* d_positions, const float3* d_momenta,
    const float* d_masses, float* d_energy, int N,
    cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = 1;  // one energy output
    int sharedSize = blockSize * sizeof(float);
    compute_energy_kernel<<<gridSize, blockSize, sharedSize, stream>>>(
        d_positions, d_momenta, d_masses, d_energy, N, 1e-6f);
}
