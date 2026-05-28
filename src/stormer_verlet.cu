#include "symplectic.cuh"

// Force computation inline
__device__ float3 compute_force_on_body(
    const float3* __restrict__ positions,
    const float* __restrict__ masses,
    int i, int N, float softening)
{
    float3 acc = {0.0f, 0.0f, 0.0f};
    float3 pi = positions[i];
    for (int j = 0; j < N; j++) {
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
    return acc;
}

__global__ void stormer_verlet_step_kernel(
    float3* __restrict__ positions,
    float3* __restrict__ momenta,
    const float* __restrict__ masses,
    float dt, int N, float softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float m = masses[i];
    float inv_m = 1.0f / m;
    float dt_half = 0.5f * dt;

    // Compute force at current position
    float3 f = compute_force_on_body(positions, masses, i, N, softening);

    // Half kick: dp = m * a * dt/2
    momenta[i].x += m * f.x * dt_half;
    momenta[i].y += m * f.y * dt_half;
    momenta[i].z += m * f.z * dt_half;

    // Full drift: dq = p/m * dt
    positions[i].x += momenta[i].x * inv_m * dt;
    positions[i].y += momenta[i].y * inv_m * dt;
    positions[i].z += momenta[i].z * inv_m * dt;

    __syncthreads();

    // Recompute force at new position
    f = compute_force_on_body(positions, masses, i, N, softening);

    // Half kick
    momenta[i].x += m * f.x * dt_half;
    momenta[i].y += m * f.y * dt_half;
    momenta[i].z += m * f.z * dt_half;
}

void launch_stormer_verlet_step(
    float3* d_positions, float3* d_momenta,
    float3* d_forces, const float* d_masses,
    float dt, int N, cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    float softening = 1e-6f;
    stormer_verlet_step_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_positions, d_momenta, d_masses, dt, N, softening);
}
