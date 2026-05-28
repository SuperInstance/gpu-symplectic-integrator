#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Gravitational constant (normalized units)
#define G_CONST 1.0f

// Integrator types
#define INTEGRATOR_EULER  0
#define INTEGRATOR_VERLET 1
#define INTEGRATOR_YOSHIDA 2

// Yoshida 4th-order coefficients
#define YOSHIDA_C1 ( 1.0f / (2.0f * (2.0f - powf(2.0f, 1.0f/3.0f))))
#define YOSHIDA_C2 ((1.0f - powf(2.0f, 1.0f/3.0f)) / (2.0f * (2.0f - powf(2.0f, 1.0f/3.0f))))
#define YOSHIDA_D1 (1.0f / (2.0f - powf(2.0f, 1.0f/3.0f)))
#define YOSHIDA_D2 (-powf(2.0f, 1.0f/3.0f) / (2.0f - powf(2.0f, 1.0f/3.0f)))

#ifdef __cplusplus
extern "C" {
#endif

// N-body gravitational force computation
void launch_compute_gravitational_forces(
    const float3* d_positions, const float* d_masses,
    float3* d_forces, int N_bodies, float softening,
    cudaStream_t stream = 0);

// Symplectic Euler step
void launch_symplectic_euler_step(
    float3* d_positions, float3* d_momenta,
    const float3* d_forces, const float* d_masses,
    float dt, int N, cudaStream_t stream = 0);

// Störmer-Verlet step
void launch_stormer_verlet_step(
    float3* d_positions, float3* d_momenta,
    float3* d_forces, const float* d_masses,
    float dt, int N, cudaStream_t stream = 0);

// Yoshida 4th order step
void launch_yoshida4_step(
    float3* d_positions, float3* d_momenta,
    float3* d_forces, const float* d_masses,
    float dt, int N, cudaStream_t stream = 0);

// Energy computation
void launch_compute_energy(
    const float3* d_positions, const float3* d_momenta,
    const float* d_masses, float* d_energy, int N,
    cudaStream_t stream = 0);

// Angular momentum computation
void launch_compute_angular_momentum(
    const float3* d_positions, const float3* d_momenta,
    float3* d_L_total, int N, cudaStream_t stream = 0);

// Batch integration
void launch_batch_integrate(
    float3* d_all_positions, float3* d_all_momenta,
    const float* d_all_masses, float dt, int n_steps,
    int N_bodies, int N_sims, int integrator_type,
    cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
