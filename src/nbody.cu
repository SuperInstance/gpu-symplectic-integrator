#include "symplectic.cuh"

// Each thread handles one body; computes gravitational force from all others
__global__ void compute_gravitational_forces_kernel(
    const float3* __restrict__ positions,
    const float* __restrict__ masses,
    float3* __restrict__ forces,
    int N_bodies, float softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_bodies) return;

    float3 acc = {0.0f, 0.0f, 0.0f};
    float3 pi = positions[i];

    for (int j = 0; j < N_bodies; j++) {
        if (i == j) continue;
        float3 pj = positions[j];
        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        float r2 = dx*dx + dy*dy + dz*dz + softening*softening;
        float inv_r = rsqrtf(r2);
        float inv_r3 = inv_r * inv_r * inv_r;
        float f = G_CONST * masses[j] * inv_r3;
        acc.x += f * dx;
        acc.y += f * dy;
        acc.z += f * dz;
    }

    forces[i] = acc;
}

void launch_compute_gravitational_forces(
    const float3* d_positions, const float* d_masses,
    float3* d_forces, int N_bodies, float softening,
    cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (N_bodies + blockSize - 1) / blockSize;
    compute_gravitational_forces_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_positions, d_masses, d_forces, N_bodies, softening);
}
