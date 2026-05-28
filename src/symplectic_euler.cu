#include "symplectic.cuh"

// Symplectic Euler: update momenta then positions
// p += F * dt, q += p/m * dt
__global__ void symplectic_euler_step_kernel(
    float3* __restrict__ positions,
    float3* __restrict__ momenta,
    const float3* __restrict__ forces,
    const float* __restrict__ masses,
    float dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Update momentum: dp = m * a * dt (forces = acceleration)
    float m = masses[i];
    momenta[i].x += m * forces[i].x * dt;
    momenta[i].y += m * forces[i].y * dt;
    momenta[i].z += m * forces[i].z * dt;

    // Update position: dq = p/m * dt
    float inv_m = 1.0f / m;
    positions[i].x += momenta[i].x * inv_m * dt;
    positions[i].y += momenta[i].y * inv_m * dt;
    positions[i].z += momenta[i].z * inv_m * dt;
}

void launch_symplectic_euler_step(
    float3* d_positions, float3* d_momenta,
    const float3* d_forces, const float* d_masses,
    float dt, int N, cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    symplectic_euler_step_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_positions, d_momenta, d_forces, d_masses, dt, N);
}
