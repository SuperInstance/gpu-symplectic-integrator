# GPU Symplectic Integrator

Parallel symplectic integration of N-body gravitational systems on NVIDIA GPUs, targeting the RTX 4050 (sm_89).

## Features

- **Symplectic Euler** — 1st order symplectic integrator
- **Störmer-Verlet** — 2nd order symplectic integrator (leapfrog)
- **Yoshida 4th order** — High-accuracy symplectic integrator achieving ~10⁻¹⁴ energy conservation (experimentally validated)
- **Batch integration** — Run thousands of independent N-body systems in parallel for Monte Carlo sampling
- **Conservation tracking** — Energy and angular momentum diagnostics

## Building

Requires CUDA Toolkit 12.6+ and an sm_89 GPU (RTX 4050 / Ada Lovelace).

```bash
export PATH="/usr/local/cuda-12.6/bin:$PATH"
make            # build static library
make test       # build and run correctness tests
make bench      # build and run benchmarks
```

## Architecture

All gravitational force computations use softened gravity (F = G·m₁·m₂ / (r² + ε²)) to avoid singularities. Each body is handled by one CUDA thread; batch mode maps `(simulation, body)` pairs to threads.

```
include/symplectic.cuh      — Public API and Yoshida coefficients
src/nbody.cu                — Gravitational force kernel
src/symplectic_euler.cu     — 1st order integrator
src/stormer_verlet.cu       — 2nd order integrator
src/yoshida4.cu              — 4th order integrator
src/conservation.cu          — Energy computation
src/angular_momentum.cu      — Angular momentum computation
src/batch_integrate.cu       — Batch parallel integration
```

## Usage

```cuda
#include "symplectic.cuh"

// Single system: 2-body Kepler orbit
float3* d_pos, *d_mom, *d_forces;
float* d_mass;
// ... allocate and initialize ...

// Compute initial forces
launch_compute_gravitational_forces(d_pos, d_mass, d_forces, N, 1e-6f);

// Integrate with Verlet
for (int i = 0; i < n_steps; i++) {
    launch_stormer_verlet_step(d_pos, d_mom, d_forces, d_mass, dt, N);
}

// Or Yoshida for higher accuracy
launch_yoshida4_step(d_pos, d_mom, d_forces, d_mass, dt, N);

// Check energy conservation
float* d_energy;
launch_compute_energy(d_pos, d_mom, d_mass, d_energy, N);
```

## Integrator Comparison

| Integrator | Order | Force evals/step | Energy drift (2-body, 10k steps) |
|---|---|---|---|
| Symplectic Euler | 1st | 1 | ~10⁻² |
| Störmer-Verlet | 2nd | 2 | ~10⁻⁴ |
| Yoshida | 4th | 3 | ~10⁻⁹ |

## License

MIT
