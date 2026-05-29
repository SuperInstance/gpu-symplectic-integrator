# gpu-symplectic-integrator

**CUDA symplectic integrators for N-body gravitational systems — Euler, Störmer-Verlet, and 4th-order Yoshida with energy and angular momentum conservation tracking.**

GPU-accelerated symplectic integrators that preserve the geometric structure of Hamiltonian systems. Batch integration runs thousands of independent N-body systems in parallel. Tracks energy drift, angular momentum conservation, and phase space structure.

## What This Gives You

- **Symplectic Euler** — 1st order, preserves phase space volume
- **Störmer-Verlet** — 2nd order, time-reversible, excellent energy conservation
- **Yoshida 4th order** — composed integrator with minimal energy drift
- **Batch integration** — N_sim independent N-body systems in parallel
- **Energy & angular momentum tracking** — GPU parallel reduction
- **Benchmarking suite** — throughput comparison across integrators and system sizes

## Quick Start

```cuda
#include "symplectic.cuh"

// Single N-body step
launch_stormer_verlet_step(d_positions, d_momenta, d_forces, d_masses, dt, N, softening, stream);

// Batch: 1000 independent 16-body systems
launch_batch_verlet_step(d_all_pos, d_all_mom, d_all_mass, dt, 16, 1000, softening, stream);
```

## Build

```bash
nvcc -O3 -o test_correctness tests/test_correctness.cu src/*.cu -lcurand
nvcc -O3 -o benchmark bench/benchmark.cu src/*.cu -lcurand
./test_correctness
./benchmark
```

## API Reference

| Function | Description |
|----------|-------------|
| `launch_compute_gravitational_forces` | N-body gravity kernel |
| `launch_symplectic_euler_step` | 1st order symplectic |
| `launch_stormer_verlet_step` | 2nd order time-reversible |
| `launch_yoshida4_step` | 4th order composed |
| `launch_batch_*` | Run N_sim systems in parallel |
| `launch_compute_energy` | Total energy (KE + PE) |
| `launch_compute_angular_momentum` | L = Σ r×p |

## How It Fits

Part of the SuperInstance ecosystem:

- **[gpu-sheaf-laplacian](https://github.com/SuperInstance/gpu-sheaf-laplacian)** — CUDA sheaf Laplacian
- **gpu-symplectic-integrator** — CUDA symplectic N-body (this repo)

## License

MIT
