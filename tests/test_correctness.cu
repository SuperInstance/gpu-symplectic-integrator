#include "symplectic.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { \
    test_count++; \
    printf("  TEST %d: %-55s ", test_count, name); \
} while(0)

#define PASS() do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } else { PASS(); } } while(0)

// Helper: allocate and copy to device
template<typename T>
T* to_device(const std::vector<T>& v) {
    T* d;
    CHECK_CUDA(cudaMalloc(&d, v.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// CPU reference: compute forces
void cpu_forces(const std::vector<float3>& pos, const std::vector<float>& mass,
                std::vector<float3>& forces, int N, float soft) {
    for (int i = 0; i < N; i++) {
        forces[i] = {0,0,0};
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            float dx = pos[j].x - pos[i].x;
            float dy = pos[j].y - pos[i].y;
            float dz = pos[j].z - pos[i].z;
            float r2 = dx*dx + dy*dy + dz*dz + soft*soft;
            float r = sqrtf(r2);
            float f = G_CONST * mass[j] / (r2 * r);
            forces[i].x += f * dx;
            forces[i].y += f * dy;
            forces[i].z += f * dz;
        }
    }
}

// Setup 2-body Kepler orbit
void setup_kepler(std::vector<float3>& pos, std::vector<float3>& mom,
                  std::vector<float>& mass, float& dt) {
    // Two bodies: M=1 at center, m=0.001 orbiting at r=1
    // Circular orbit: v = sqrt(G*M/r) for test particle
    int N = 2;
    pos.resize(N); mom.resize(N); mass.resize(N);
    mass[0] = {1.0f}; pos[0] = {0,0,0}; mom[0] = {0,0,0};
    mass[1] = {0.001f}; pos[1] = {1.0f,0,0}; 
    float v = sqrtf(G_CONST * mass[0] / 1.0f);
    mom[1] = {0, mass[1]*v, 0};
    dt = 0.01f;
}

void test_1_kepler_elliptical_verlet() {
    TEST("2-body Kepler: stays elliptical over 1000 steps (Verlet)");
    std::vector<float3> pos, mom;
    std::vector<float> mass;
    float dt;
    setup_kepler(pos, mom, mass, dt);
    int N = 2;

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);
    float3* d_forces;
    CHECK_CUDA(cudaMalloc(&d_forces, N * sizeof(float3)));

    // Initial force
    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    CHECK_CUDA(cudaDeviceSynchronize());

    float init_r = sqrtf(pos[1].x*pos[1].x + pos[1].y*pos[1].y + pos[1].z*pos[1].z);

    for (int step = 0; step < 1000; step++) {
        launch_stormer_verlet_step(d_pos, d_mom, d_forces, d_mass, dt, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::vector<float3> h_pos(N);
    CHECK_CUDA(cudaMemcpy(h_pos.data(), d_pos, N*sizeof(float3), cudaMemcpyDeviceToHost));

    float final_r = sqrtf(h_pos[1].x*h_pos[1].x + h_pos[1].y*h_pos[1].y + h_pos[1].z*h_pos[1].z);
    
    // Should stay roughly circular (within 10% of initial radius)
    bool ok = fabsf(final_r - init_r) < 0.1f * init_r;
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom));
    CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces));
    CHECK(ok, "orbit diverged");
}

void test_2_kepler_energy_drift_verlet() {
    TEST("2-body Kepler: energy drift < 1e-4 over 10000 steps (Verlet)");
    std::vector<float3> pos, mom;
    std::vector<float> mass;
    float dt;
    setup_kepler(pos, mom, mass, dt);
    int N = 2;

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);
    float3* d_forces; float* d_energy;
    CHECK_CUDA(cudaMalloc(&d_forces, N*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_energy, sizeof(float)));

    // Initial energy
    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    launch_compute_energy(d_pos, d_mom, d_mass, d_energy, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float E0;
    CHECK_CUDA(cudaMemcpy(&E0, d_energy, sizeof(float), cudaMemcpyDeviceToHost));

    for (int step = 0; step < 10000; step++) {
        launch_stormer_verlet_step(d_pos, d_mom, d_forces, d_mass, dt, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    launch_compute_energy(d_pos, d_mom, d_mass, d_energy, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float E1;
    CHECK_CUDA(cudaMemcpy(&E1, d_energy, sizeof(float), cudaMemcpyDeviceToHost));

    float drift = fabsf((E1 - E0) / E0);
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom));
    CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces)); CHECK_CUDA(cudaFree(d_energy));
    CHECK(drift < 1e-4f, "energy drift too large");
}

void test_3_kepler_energy_drift_yoshida() {
    TEST("2-body Kepler: energy drift < 1e-5 over 10000 steps (Yoshida, float32)");
    // Use double precision for this test to see Yoshida's accuracy
    // But our library is float, so we use a smaller dt to compensate
    std::vector<float3> pos, mom;
    std::vector<float> mass;
    float dt_unused;
    setup_kepler(pos, mom, mass, dt_unused);
    int N = 2;
    float dt = 0.001f;  // smaller timestep for Yoshida

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);
    float3* d_forces; float* d_energy;
    CHECK_CUDA(cudaMalloc(&d_forces, N*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_energy, sizeof(float)));

    // Initial forces
    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    launch_compute_energy(d_pos, d_mom, d_mass, d_energy, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float E0;
    CHECK_CUDA(cudaMemcpy(&E0, d_energy, sizeof(float), cudaMemcpyDeviceToHost));

    for (int step = 0; step < 10000; step++) {
        launch_yoshida4_step(d_pos, d_mom, d_forces, d_mass, dt, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    launch_compute_energy(d_pos, d_mom, d_mass, d_energy, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float E1;
    CHECK_CUDA(cudaMemcpy(&E1, d_energy, sizeof(float), cudaMemcpyDeviceToHost));

    float drift = fabsf((E1 - E0) / E0);
    printf("[drift=%.2e] ", drift);
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom));
    CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces)); CHECK_CUDA(cudaFree(d_energy));
    CHECK(drift < 1e-5f, "energy drift too large for Yoshida");
}

void test_4_three_body_no_diverge() {
    TEST("3-body: doesn't diverge over 1000 steps");
    int N = 3;
    std::vector<float3> pos = {{0,0,0},{1,0,0},{0,1,0}};
    std::vector<float3> mom = {{0,0,0},{0,0.03f,0},{0,0,0.03f}};
    std::vector<float> mass = {1.0f, 0.1f, 0.1f};
    float dt = 0.01f;

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);
    float3* d_forces;
    CHECK_CUDA(cudaMalloc(&d_forces, N*sizeof(float3)));

    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-3f);
    for (int step = 0; step < 1000; step++) {
        launch_stormer_verlet_step(d_pos, d_mom, d_forces, d_mass, dt, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::vector<float3> h_pos(N);
    CHECK_CUDA(cudaMemcpy(h_pos.data(), d_pos, N*sizeof(float3), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float r = sqrtf(h_pos[i].x*h_pos[i].x + h_pos[i].y*h_pos[i].y + h_pos[i].z*h_pos[i].z);
        if (r > 100.0f || isnan(r)) ok = false;
    }
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom));
    CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces));
    CHECK(ok, "3-body system diverged");
}

void test_5_angular_momentum() {
    TEST("Angular momentum conservation < 1e-5");
    std::vector<float3> pos, mom;
    std::vector<float> mass;
    float dt;
    setup_kepler(pos, mom, mass, dt);
    int N = 2;

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);
    float3* d_forces; float3* d_L;
    CHECK_CUDA(cudaMalloc(&d_forces, N*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_L, sizeof(float3)));

    // Initial L
    float3 L0 = {0,0,0};
    CHECK_CUDA(cudaMemcpy(d_L, &L0, sizeof(float3), cudaMemcpyHostToDevice));
    launch_compute_angular_momentum(d_pos, d_mom, d_L, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&L0, d_L, sizeof(float3), cudaMemcpyDeviceToHost));

    // Integrate
    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    for (int step = 0; step < 1000; step++) {
        launch_stormer_verlet_step(d_pos, d_mom, d_forces, d_mass, dt, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    float3 L1 = {0,0,0};
    CHECK_CUDA(cudaMemcpy(d_L, &L1, sizeof(float3), cudaMemcpyHostToDevice));
    launch_compute_angular_momentum(d_pos, d_mom, d_L, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&L1, d_L, sizeof(float3), cudaMemcpyDeviceToHost));

    float dL = sqrtf((L1.x-L0.x)*(L1.x-L0.x)+(L1.y-L0.y)*(L1.y-L0.y)+(L1.z-L0.z)*(L1.z-L0.z));
    float L_mag = sqrtf(L0.x*L0.x+L0.y*L0.y+L0.z*L0.z);
    float rel = (L_mag > 0) ? dL / L_mag : dL;
    printf("[rel=%.2e] ", rel);
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom));
    CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces)); CHECK_CUDA(cudaFree(d_L));
    CHECK(rel < 1e-5f, "angular momentum not conserved");
}

void test_6_reversibility() {
    TEST("Reversibility: Verlet returns to initial conditions");
    std::vector<float3> pos, mom;
    std::vector<float> mass;
    float dt;
    setup_kepler(pos, mom, mass, dt);
    int N = 2;

    auto init_pos = pos;
    auto init_mom = mom;

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);
    float3* d_forces;
    CHECK_CUDA(cudaMalloc(&d_forces, N*sizeof(float3)));

    // Forward
    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    for (int step = 0; step < 500; step++) {
        launch_stormer_verlet_step(d_pos, d_mom, d_forces, d_mass, dt, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // Backward
    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    for (int step = 0; step < 500; step++) {
        launch_stormer_verlet_step(d_pos, d_mom, d_forces, d_mass, -dt, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::vector<float3> h_pos(N), h_mom(N);
    CHECK_CUDA(cudaMemcpy(h_pos.data(), d_pos, N*sizeof(float3), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mom.data(), d_mom, N*sizeof(float3), cudaMemcpyDeviceToHost));

    float err = 0;
    for (int i = 0; i < N; i++) {
        float dx = h_pos[i].x - init_pos[i].x;
        float dy = h_pos[i].y - init_pos[i].y;
        float dz = h_pos[i].z - init_pos[i].z;
        err += dx*dx + dy*dy + dz*dz;
    }
    err = sqrtf(err);
    printf("[err=%.2e] ", err);
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom));
    CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces));
    CHECK(err < 1e-3f, "reversibility failed");
}

void test_7_batch() {
    TEST("Batch: 100 independent 3-body systems in parallel");
    int N_bodies = 3, N_sims = 100;
    int total = N_sims * N_bodies;
    std::vector<float3> pos(total), mom(total);
    std::vector<float> mass(total);
    for (int s = 0; s < N_sims; s++) {
        int base = s * N_bodies;
        pos[base+0] = {0,0,0}; pos[base+1] = {1,0,0}; pos[base+2] = {0,1,0};
        mom[base+0] = {0,0,0}; mom[base+1] = {0,0.03f,0}; mom[base+2] = {0,0,0.03f};
        mass[base+0] = 1.0f; mass[base+1] = 0.1f; mass[base+2] = 0.1f;
    }

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);

    launch_batch_integrate(d_pos, d_mom, d_mass, 0.01f, 100, N_bodies, N_sims, INTEGRATOR_VERLET);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float3> h_pos(total);
    CHECK_CUDA(cudaMemcpy(h_pos.data(), d_pos, total*sizeof(float3), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < total; i++) {
        float r = sqrtf(h_pos[i].x*h_pos[i].x+h_pos[i].y*h_pos[i].y+h_pos[i].z*h_pos[i].z);
        if (isnan(r) || r > 100) ok = false;
    }
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom)); CHECK_CUDA(cudaFree(d_mass));
    CHECK(ok, "batch integration produced NaN or diverged");
}

void test_8_energy_physical() {
    TEST("Conservation tracking: energy values are physical");
    std::vector<float3> pos, mom;
    std::vector<float> mass;
    float dt;
    setup_kepler(pos, mom, mass, dt);
    int N = 2;

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);
    float3* d_forces; float* d_energy;
    CHECK_CUDA(cudaMalloc(&d_forces, N*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_energy, sizeof(float)));

    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    launch_compute_energy(d_pos, d_mom, d_mass, d_energy, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float E;
    CHECK_CUDA(cudaMemcpy(&E, d_energy, sizeof(float), cudaMemcpyDeviceToHost));

    printf("[E=%.4e] ", E);
    bool ok = !isnan(E) && !isinf(E) && E < 0;  // bound system should have negative total energy
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom));
    CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces)); CHECK_CUDA(cudaFree(d_energy));
    CHECK(ok, "energy not physical");
}

void test_9_performance() {
    TEST("Performance: 1000 bodies, 100 steps in < 100ms");
    int N = 1000;
    std::vector<float3> pos(N), mom(N);
    std::vector<float> mass(N);
    for (int i = 0; i < N; i++) {
        float angle = 2.0f * 3.14159f * i / N;
        float r = 1.0f + 0.1f * (i % 10);
        pos[i] = {r*cosf(angle), r*sinf(angle), 0};
        mom[i] = {0, 0, 0};
        mass[i] = 0.01f;
    }
    mass[0] = 1.0f;
    float dt = 0.001f;

    float3* d_pos = to_device(pos);
    float3* d_mom = to_device(mom);
    float* d_mass = to_device(mass);
    float3* d_forces;
    CHECK_CUDA(cudaMalloc(&d_forces, N*sizeof(float3)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int step = 0; step < 100; step++) {
        launch_stormer_verlet_step(d_pos, d_mom, d_forces, d_mass, dt, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("[%.1fms] ", ms);

    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mom));
    CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces));
    CHECK(ms < 100.0f, "too slow");
}

void test_10_force_matches_cpu() {
    TEST("Force computation matches CPU reference");
    int N = 5;
    std::vector<float3> pos = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}};
    std::vector<float> mass = {1.0f, 0.5f, 0.3f, 0.2f, 0.1f};
    std::vector<float3> cpu_f(N);
    float soft = 1e-6f;
    cpu_forces(pos, mass, cpu_f, N, soft);

    float3* d_pos = to_device(pos);
    float* d_mass = to_device(mass);
    float3* d_forces;
    CHECK_CUDA(cudaMalloc(&d_forces, N*sizeof(float3)));

    launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, soft);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float3> gpu_f(N);
    CHECK_CUDA(cudaMemcpy(gpu_f.data(), d_forces, N*sizeof(float3), cudaMemcpyDeviceToHost));

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float ex = gpu_f[i].x - cpu_f[i].x;
        float ey = gpu_f[i].y - cpu_f[i].y;
        float ez = gpu_f[i].z - cpu_f[i].z;
        max_err = fmaxf(max_err, sqrtf(ex*ex + ey*ey + ez*ez));
    }
    printf("[max_err=%.2e] ", max_err);
    CHECK_CUDA(cudaFree(d_pos)); CHECK_CUDA(cudaFree(d_mass)); CHECK_CUDA(cudaFree(d_forces));
    CHECK(max_err < 1e-4f, "GPU forces don't match CPU");
}

int main() {
    printf("\n=== GPU Symplectic Integrator Correctness Tests ===\n\n");

    test_1_kepler_elliptical_verlet();
    test_2_kepler_energy_drift_verlet();
    test_3_kepler_energy_drift_yoshida();
    test_4_three_body_no_diverge();
    test_5_angular_momentum();
    test_6_reversibility();
    test_7_batch();
    test_8_energy_physical();
    test_9_performance();
    test_10_force_matches_cpu();

    printf("\n=== Results: %d/%d passed ===\n\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
