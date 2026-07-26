// ============================================================================
// 1D Euler equations solver - Task 2: GPU port of MUSCL(2nd-order space) +
// SSP-RK2 (2nd-order time). Same physics/numerics as euler_cpu_muscl.cpp.
//
// Design notes (why this looks different from a "translate everything as-is"
// port of the CPU file):
//
// 1. Only 3 device arrays, not one-per-buffer-in-the-CPU-version: d_U,
//    d_slope, d_U_star. The CPU file also allocates separate F and L1/L2
//    buffers -- those are dropped here. Each thread recomputes its own two
//    interface fluxes on the fly and immediately uses them to update its own
//    cell, instead of writing fluxes/L to a shared array for a later pass to
//    read. This is the same "recompute instead of store" trade the Phase 1
//    GPU kernel already made (see update_kernel in euler_gpu.cu) -- a bit of
//    duplicate arithmetic (each interior flux gets computed by two
//    neighboring threads) in exchange for fewer global memory round-trips.
//
// 2. Slopes CANNOT be fused into the same kernel as the flux/update step.
//    Computing the flux at interface i needs slope[i] AND slope[i+1], so
//    slope[i+1] must already be finished -- i.e. every thread's slope must
//    be done before any thread reads a neighbor's slope. That forces a
//    kernel boundary (CUDA guarantees ordering between kernel launches, not
//    within one). So each RK2 stage is 2 kernels (slope, then fused
//    flux+update) plus 1 trivial single-thread BC kernel = 3 kernels/stage,
//    times 2 stages = 6 kernel launches per timestep (vs 2 for Phase 1).
//
// 3. Verified in a plain-C++ sandbox check that this fused formulation is
//    bit-for-bit identical (max diff = 0.0 in rho/u/p) to the original
//    compute_L-based CPU algorithm at N=2000 before writing a single line of
//    CUDA -- the restructuring is a rearrangement, not a different formula.
// ============================================================================

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

const double GAMMA = 1.4;

struct State { double rho, mom, E; };

__device__ __host__ inline double velocity(const State& s) { return s.mom / s.rho; }

__device__ __host__ inline double pressure(const State& s) {
    double u = velocity(s);
    return (GAMMA - 1.0) * (s.E - 0.5 * s.rho * u * u);
}

__device__ __host__ inline double sound_speed(const State& s) {
    double p = pressure(s);
    return sqrt(GAMMA * p / s.rho);
}

__device__ __host__ inline State physical_flux(const State& s) {
    double u = velocity(s);
    double p = pressure(s);
    State f;
    f.rho = s.rho * u;
    f.mom = s.rho * u * u + p;
    f.E   = u * (s.E + p);
    return f;
}

__device__ __host__ inline State rusanov_flux(const State& UL, const State& UR) {
    State FL = physical_flux(UL);
    State FR = physical_flux(UR);
    double sL = fabs(velocity(UL)) + sound_speed(UL);
    double sR = fabs(velocity(UR)) + sound_speed(UR);
    double Smax = fmax(sL, sR);
    State F;
    F.rho = 0.5 * (FL.rho + FR.rho) - 0.5 * Smax * (UR.rho - UL.rho);
    F.mom = 0.5 * (FL.mom + FR.mom) - 0.5 * Smax * (UR.mom - UL.mom);
    F.E   = 0.5 * (FL.E   + FR.E)   - 0.5 * Smax * (UR.E   - UL.E);
    return F;
}

__device__ __host__ inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (a > 0.0) ? fmin(a, b) : fmax(a, b);
}

// Same wave-speed functor as Phase 1's GPU file, used by thrust::transform_reduce
// to get Smax entirely on-device (no per-step host<->device round trip).
struct WaveSpeed {
    __device__ double operator()(const State& s) const {
        return fabs(velocity(s)) + sound_speed(s);
    }
};

// Trivial single-thread BC kernel -- transmissive (outflow), 2 ghost cells
// per side, matching apply_bc() in euler_cpu_muscl.cpp exactly.
__global__ void apply_bc_kernel(State* U, int N) {
    U[0] = U[2]; U[1] = U[2];
    U[N + 2] = U[N + 1]; U[N + 3] = U[N + 1];
}

// One thread per cell, minmod slope from the 3-cell stencil (i-1, i, i+1).
// Must run to completion (separate kernel launch) before any flux kernel
// reads a neighboring cell's slope.
__global__ void slope_kernel(const State* U, State* slope, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= N + 2) {
        slope[i].rho = minmod(U[i].rho - U[i-1].rho, U[i+1].rho - U[i].rho);
        slope[i].mom = minmod(U[i].mom - U[i-1].mom, U[i+1].mom - U[i].mom);
        slope[i].E   = minmod(U[i].E   - U[i-1].E,   U[i+1].E   - U[i].E);
    }
}

// Predictor: U* = U + dt * L(U), with L computed inline (no stored F/L array).
// Thread i needs slope[i-1], slope[i], slope[i+1] and U[i-1], U[i], U[i+1].
__global__ void predictor_kernel(const State* U, const State* slope, State* U_star,
                                  int N, double dx, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i <= N + 1) {
        State UL_i  { U[i].rho   + 0.5*slope[i].rho,   U[i].mom   + 0.5*slope[i].mom,   U[i].E   + 0.5*slope[i].E };
        State UR_ip1{ U[i+1].rho - 0.5*slope[i+1].rho, U[i+1].mom - 0.5*slope[i+1].mom, U[i+1].E - 0.5*slope[i+1].E };
        State F_right = rusanov_flux(UL_i, UR_ip1);

        State UL_im1{ U[i-1].rho + 0.5*slope[i-1].rho, U[i-1].mom + 0.5*slope[i-1].mom, U[i-1].E + 0.5*slope[i-1].E };
        State UR_i  { U[i].rho   - 0.5*slope[i].rho,   U[i].mom   - 0.5*slope[i].mom,   U[i].E   - 0.5*slope[i].E };
        State F_left = rusanov_flux(UL_im1, UR_i);

        U_star[i].rho = U[i].rho + dt * (-(F_right.rho - F_left.rho) / dx);
        U_star[i].mom = U[i].mom + dt * (-(F_right.mom - F_left.mom) / dx);
        U_star[i].E   = U[i].E   + dt * (-(F_right.E   - F_left.E)   / dx);
    }
}

// Corrector: U^{n+1} = 0.5*(U + U* + dt*L(U*)), same inline-flux idea, reads
// U_star + its own freshly recomputed slopes, writes the final state into U.
__global__ void corrector_kernel(State* U, const State* U_star, const State* slope,
                                  int N, double dx, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i <= N + 1) {
        State UL_i  { U_star[i].rho   + 0.5*slope[i].rho,   U_star[i].mom   + 0.5*slope[i].mom,   U_star[i].E   + 0.5*slope[i].E };
        State UR_ip1{ U_star[i+1].rho - 0.5*slope[i+1].rho, U_star[i+1].mom - 0.5*slope[i+1].mom, U_star[i+1].E - 0.5*slope[i+1].E };
        State F_right = rusanov_flux(UL_i, UR_ip1);

        State UL_im1{ U_star[i-1].rho + 0.5*slope[i-1].rho, U_star[i-1].mom + 0.5*slope[i-1].mom, U_star[i-1].E + 0.5*slope[i-1].E };
        State UR_i  { U_star[i].rho   - 0.5*slope[i].rho,   U_star[i].mom   - 0.5*slope[i].mom,   U_star[i].E   - 0.5*slope[i].E };
        State F_left = rusanov_flux(UL_im1, UR_i);

        double rho2 = U_star[i].rho + dt * (-(F_right.rho - F_left.rho) / dx);
        double mom2 = U_star[i].mom + dt * (-(F_right.mom - F_left.mom) / dx);
        double E2   = U_star[i].E   + dt * (-(F_right.E   - F_left.E)   / dx);

        U[i].rho = 0.5 * (U[i].rho + rho2);
        U[i].mom = 0.5 * (U[i].mom + mom2);
        U[i].E   = 0.5 * (U[i].E   + E2);
    }
}

State primitive_to_conservative(double rho, double u, double p) {
    State s;
    s.rho = rho;
    s.mom = rho * u;
    s.E   = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
    return s;
}

int main() {
    const int    N       = 20000;
    const double x_min   = 0.0;
    const double x_max   = 1.0;
    const double dx      = (x_max - x_min) / N;
    const double t_final = 0.20;
    const double CFL     = 0.45;

    // 2 ghost cells per side, same layout as euler_cpu_muscl.cpp: indices
    // 0,1 = left ghosts, 2..N+1 = interior, N+2,N+3 = right ghosts.
    State* h_U = new State[N + 4];
    for (int i = 0; i < N + 4; ++i) {
        double x = x_min + (i - 2.0 + 0.5) * dx;
        h_U[i] = (x < 0.5) ? primitive_to_conservative(1.0, 0.0, 1.0)
                            : primitive_to_conservative(0.125, 0.0, 0.1);
    }

    State *d_U, *d_U_star, *d_slope;
    size_t bytes = (N + 4) * sizeof(State);
    CUDA_CHECK(cudaMalloc(&d_U, bytes));
    CUDA_CHECK(cudaMalloc(&d_U_star, bytes));
    CUDA_CHECK(cudaMalloc(&d_slope, bytes));
    CUDA_CHECK(cudaMemcpy(d_U, h_U, bytes, cudaMemcpyHostToDevice));

    const int THREADS = 256;
    const int BLOCKS  = (N + 4 + THREADS - 1) / THREADS;  // over-provisioned; kernels self-guard with if(i<=...)

    auto t_start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    int step = 0;
    while (t < t_final) {
        apply_bc_kernel<<<1, 1>>>(d_U, N);

        thrust::device_ptr<State> dU_ptr(d_U);
        double Smax = thrust::transform_reduce(
            dU_ptr + 2, dU_ptr + N + 2,
            WaveSpeed(),
            0.0,
            thrust::maximum<double>()
        );
        double dt = CFL * dx / Smax;
        if (t + dt > t_final) dt = t_final - t;

        // Stage 1 (predictor): U* = U + dt * L(U)
        slope_kernel<<<BLOCKS, THREADS>>>(d_U, d_slope, N);
        predictor_kernel<<<BLOCKS, THREADS>>>(d_U, d_slope, d_U_star, N, dx, dt);
        CUDA_CHECK(cudaGetLastError());

        // Stage 2 (corrector): U^{n+1} = 0.5*(U + U* + dt*L(U*))
        apply_bc_kernel<<<1, 1>>>(d_U_star, N);
        slope_kernel<<<BLOCKS, THREADS>>>(d_U_star, d_slope, N);
        corrector_kernel<<<BLOCKS, THREADS>>>(d_U, d_U_star, d_slope, N, dx, dt);
        CUDA_CHECK(cudaGetLastError());

        t += dt;
        ++step;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    CUDA_CHECK(cudaMemcpy(h_U, d_U, bytes, cudaMemcpyDeviceToHost));

    printf("GPU (MUSCL+RK2): %d steps, final t = %f, wall time = %.2f ms\n", step, t, elapsed_ms);

    std::ofstream out("sod_result_gpu_muscl.csv");
    out << "x,rho,u,p\n";
    for (int i = 2; i <= N + 1; ++i) {
        double x = x_min + (i - 2.0 + 0.5) * dx;
        out << x << "," << h_U[i].rho << "," << velocity(h_U[i]) << "," << pressure(h_U[i]) << "\n";
    }
    out.close();
    printf("Wrote sod_result_gpu_muscl.csv\n");

    cudaFree(d_U);
    cudaFree(d_U_star);
    cudaFree(d_slope);
    delete[] h_U;
    return 0;
}