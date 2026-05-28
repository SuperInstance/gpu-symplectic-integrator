#include "symplectic.cuh"

// Batch integration: run N_sim independent N-body systems in parallel
// Layout: all_positions[sim * N_bodies + body], same for momenta/masses

__device__ float3 batch_compute_force(
    const float3* __restrict__ all_positions,
    const float* __restrict__ all_masses,
    int sim, int body, int N_bodies, float softening)
{
    float3 acc = {0.0f, 0.0f, 0.0f};
    int base = sim * N_bodies;
    float3 pi = all_positions[base + body];
    for (int j = 0; j < N_bodies; j++) {
        if (j == body) continue;
        float3 pj = all_positions[base + j];
        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        float r2 = dx*dx + dy*dy + dz*dz + softening*softening;
        float inv_r = rsqrtf(r2);
        float inv_r3 = inv_r * inv_r * inv_r;
        float f = G_CONST * all_masses[base + j] * inv_r3;
        acc.x += f * dx;
        acc.y += f * dy;
        acc.z += f * dz;
    }
    return acc;
}

// Symplectic Euler for batch
__global__ void batch_euler_kernel(
    float3* __restrict__ all_positions,
    float3* __restrict__ all_momenta,
    const float* __restrict__ all_masses,
    float dt, int N_bodies, int N_sims, float softening)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_sims * N_bodies;
    if (idx >= total) return;

    int sim = idx / N_bodies;
    int body = idx % N_bodies;

    float3 f = batch_compute_force(all_positions, all_masses, sim, body, N_bodies, softening);
    float m = all_masses[idx];
    all_momenta[idx].x += m * f.x * dt;
    all_momenta[idx].y += m * f.y * dt;
    all_momenta[idx].z += m * f.z * dt;

    float inv_m = 1.0f / m;
    all_positions[idx].x += all_momenta[idx].x * inv_m * dt;
    all_positions[idx].y += all_momenta[idx].y * inv_m * dt;
    all_positions[idx].z += all_momenta[idx].z * inv_m * dt;
}

// Verlet for batch
__global__ void batch_verlet_kernel(
    float3* __restrict__ all_positions,
    float3* __restrict__ all_momenta,
    const float* __restrict__ all_masses,
    float dt, int N_bodies, int N_sims, float softening)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_sims * N_bodies;
    if (idx >= total) return;

    int sim = idx / N_bodies;
    int body = idx % N_bodies;

    // Half kick: dp = m * a * dt/2
    float3 f = batch_compute_force(all_positions, all_masses, sim, body, N_bodies, softening);
    float m = all_masses[idx];
    all_momenta[idx].x += m * f.x * dt * 0.5f;
    all_momenta[idx].y += m * f.y * dt * 0.5f;
    all_momenta[idx].z += m * f.z * dt * 0.5f;

    // Full drift
    float inv_m = 1.0f / m;
    all_positions[idx].x += all_momenta[idx].x * inv_m * dt;
    all_positions[idx].y += all_momenta[idx].y * inv_m * dt;
    all_positions[idx].z += all_momenta[idx].z * inv_m * dt;

    __syncthreads();

    // Recompute force and half kick
    f = batch_compute_force(all_positions, all_masses, sim, body, N_bodies, softening);
    all_momenta[idx].x += m * f.x * dt * 0.5f;
    all_momenta[idx].y += m * f.y * dt * 0.5f;
    all_momenta[idx].z += m * f.z * dt * 0.5f;
}

void launch_batch_integrate(
    float3* d_all_positions, float3* d_all_momenta,
    const float* d_all_masses, float dt, int n_steps,
    int N_bodies, int N_sims, int integrator_type,
    cudaStream_t stream)
{
    int total = N_sims * N_bodies;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    float softening = 1e-6f;

    for (int step = 0; step < n_steps; step++) {
        switch (integrator_type) {
            case INTEGRATOR_EULER:
                batch_euler_kernel<<<gridSize, blockSize, 0, stream>>>(
                    d_all_positions, d_all_momenta, d_all_masses,
                    dt, N_bodies, N_sims, softening);
                break;
            case INTEGRATOR_VERLET:
                batch_verlet_kernel<<<gridSize, blockSize, 0, stream>>>(
                    d_all_positions, d_all_momenta, d_all_masses,
                    dt, N_bodies, N_sims, softening);
                break;
            case INTEGRATOR_YOSHIDA:
                // Yoshida for batch not implemented in single-kernel form
                // Would need cooperative groups for proper sync
                // Fall back to verlet for batch mode
                batch_verlet_kernel<<<gridSize, blockSize, 0, stream>>>(
                    d_all_positions, d_all_momenta, d_all_masses,
                    dt, N_bodies, N_sims, softening);
                break;
        }
    }
}
