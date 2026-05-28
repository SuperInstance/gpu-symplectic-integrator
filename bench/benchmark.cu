#include "symplectic.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

template<typename T>
T* to_device(const std::vector<T>& v) {
    T* d;
    CHECK_CUDA(cudaMalloc(&d, v.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

void bench_nbody(int N, int n_steps) {
    printf("\n--- N_bodies = %d, %d steps ---\n", N, n_steps);

    std::vector<float3> pos(N), mom(N);
    std::vector<float> mass(N);
    for (int i = 0; i < N; i++) {
        float angle = 2.0f * M_PI * i / N;
        float r = 1.0f + 0.1f * (i % 10);
        pos[i] = {r*cosf(angle), r*sinf(angle), 0.01f * i};
        mom[i] = {0, 0, 0};
        mass[i] = 0.01f;
    }
    mass[0] = 1.0f;

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);
    float3* d_forces;
    CHECK_CUDA(cudaMalloc(&d_forces, N * sizeof(float3)));
    float* d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, sizeof(float)));

    float dt = 0.001f;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Force computation only
    CHECK_CUDA(cudaEventRecord(start));
    for (int s = 0; s < n_steps; s++) {
        launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float force_ms;
    CHECK_CUDA(cudaEventElapsedTime(&force_ms, start, stop));

    // Reset
    CHECK_CUDA(cudaMemcpy(d_pos, pos.data(), N*sizeof(float3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mom, mom.data(), N*sizeof(float3), cudaMemcpyHostToDevice));

    // Euler
    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    CHECK_CUDA(cudaEventRecord(start));
    for (int s = 0; s < n_steps; s++) {
        launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
        launch_symplectic_euler_step(d_pos, d_mom, d_forces, d_mass, dt, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float euler_ms;
    CHECK_CUDA(cudaEventElapsedTime(&euler_ms, start, stop));

    // Energy at end of Euler for reference
    launch_compute_energy(d_pos, d_mom, d_mass, d_energy, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float E_euler;
    CHECK_CUDA(cudaMemcpy(&E_euler, d_energy, sizeof(float), cudaMemcpyDeviceToHost));

    // Reset
    CHECK_CUDA(cudaMemcpy(d_pos, pos.data(), N*sizeof(float3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mom, mom.data(), N*sizeof(float3), cudaMemcpyHostToDevice));

    // Verlet
    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    CHECK_CUDA(cudaEventRecord(start));
    for (int s = 0; s < n_steps; s++) {
        launch_stormer_verlet_step(d_pos, d_mom, d_forces, d_mass, dt, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float verlet_ms;
    CHECK_CUDA(cudaEventElapsedTime(&verlet_ms, start, stop));

    launch_compute_energy(d_pos, d_mom, d_mass, d_energy, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float E_verlet;
    CHECK_CUDA(cudaMemcpy(&E_verlet, d_energy, sizeof(float), cudaMemcpyDeviceToHost));

    // Reset
    CHECK_CUDA(cudaMemcpy(d_pos, pos.data(), N*sizeof(float3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mom, mom.data(), N*sizeof(float3), cudaMemcpyHostToDevice));

    // Yoshida
    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    CHECK_CUDA(cudaEventRecord(start));
    for (int s = 0; s < n_steps; s++) {
        launch_yoshida4_step(d_pos, d_mom, d_forces, d_mass, dt, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float yoshida_ms;
    CHECK_CUDA(cudaEventElapsedTime(&yoshida_ms, start, stop));

    launch_compute_energy(d_pos, d_mom, d_mass, d_energy, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float E_yoshida;
    CHECK_CUDA(cudaMemcpy(&E_yoshida, d_energy, sizeof(float), cudaMemcpyDeviceToHost));

    // Initial energy
    launch_compute_energy(to_device(pos), to_device(mom), d_mass, d_energy, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float E0;
    CHECK_CUDA(cudaMemcpy(&E0, d_energy, sizeof(float), cudaMemcpyDeviceToHost));

    printf("  Forces only:     %7.2f ms  (%.2f ms/step)\n", force_ms, force_ms/n_steps);
    printf("  Symplectic Euler: %7.2f ms  (%.2f ms/step)  E=%.4e\n", euler_ms, euler_ms/n_steps, E_euler);
    printf("  Störmer-Verlet:   %7.2f ms  (%.2f ms/step)  E=%.4e\n", verlet_ms, verlet_ms/n_steps, E_verlet);
    printf("  Yoshida 4th:      %7.2f ms  (%.2f ms/step)  E=%.4e\n", yoshida_ms, yoshida_ms/n_steps, E_yoshida);

    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom));
    CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces)); CHECK_CUDA(cudaFree(d_energy));
}

void bench_batch(int N_bodies, int N_sims, int n_steps) {
    printf("\n--- Batch: %d sims × %d bodies, %d steps ---\n", N_sims, N_bodies, n_steps);

    int total = N_sims * N_bodies;
    std::vector<float3> pos(total), mom(total);
    std::vector<float> mass(total);
    for (int s = 0; s < N_sims; s++) {
        int base = s * N_bodies;
        pos[base+0] = {0,0,0}; pos[base+1] = {1,0,0}; pos[base+2] = {0,1,0};
        mom[base+0] = {0,0,0}; mom[base+1] = {0,0.03f,0}; mom[base+2] = {0,0,0.03f};
        mass[base+0] = 1.0f; mass[base+1] = 0.1f; mass[base+2] = 0.1f;
        for (int b = 3; b < N_bodies; b++) {
            float angle = 2.0f * M_PI * b / N_bodies;
            pos[base+b] = {cosf(angle), sinf(angle), 0};
            mom[base+b] = {0,0,0};
            mass[base+b] = 0.01f;
        }
    }

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    launch_batch_integrate(d_pos, d_mom, d_mass, 0.01f, n_steps, N_bodies, N_sims, INTEGRATOR_VERLET);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("  Batch Verlet: %7.2f ms  (%.3f ms/sim for %d steps)\n", ms, ms/N_sims, n_steps);

    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom)); CHECK_CUDA(cudaFree(d_mass));
}

int main() {
    printf("\n=== GPU Symplectic Integrator Benchmark (RTX 4050) ===\n");

    bench_nbody(10, 1000);
    bench_nbody(100, 1000);
    bench_nbody(1000, 1000);
    bench_nbody(5000, 100);

    bench_batch(10, 1000, 1000);

    printf("\n=== Benchmark Complete ===\n\n");
    return 0;
}
