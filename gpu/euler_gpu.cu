// ============================================================================
// 1D Euler equations solver - Phase 1: CUDA GPU port (initial version)
// Same physics/numerics as the CPU version (euler_cpu.cpp):
// Finite Volume Method, Rusanov flux, Sod shock tube validation case.
//
// Parallelization strategy: one CUDA thread per grid cell.
// Each thread reads its own cell and its two neighbors (a 3-point stencil),
// computes the two interface fluxes, and writes the updated state.
//
// Known limitation of this version: the CFL timestep reduction is done by
// copying the full array back to the host every step (see main()) -- fixed
// in the next revision by moving the reduction onto the GPU with Thrust.
// ============================================================================

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

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

__global__ void apply_bc_kernel(State* U, int N) {
    U[0]     = U[1];
    U[N + 1] = U[N];
}

__global__ void update_kernel(const State* U, State* U_new, int N, double dt, double dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= N) {
        State F_left  = rusanov_flux(U[i - 1], U[i]);
        State F_right = rusanov_flux(U[i], U[i + 1]);
        U_new[i].rho = U[i].rho - (dt / dx) * (F_right.rho - F_left.rho);
        U_new[i].mom = U[i].mom - (dt / dx) * (F_right.mom - F_left.mom);
        U_new[i].E   = U[i].E   - (dt / dx) * (F_right.E   - F_left.E);
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
    const int    N       = 400;
    const double x_min   = 0.0;
    const double x_max   = 1.0;
    const double dx      = (x_max - x_min) / N;
    const double t_final = 0.20;
    const double CFL     = 0.45;

    State* h_U = new State[N + 2];
    for (int i = 0; i < N + 2; ++i) {
        double x = x_min + (i - 0.5) * dx;
        h_U[i] = (x < 0.5) ? primitive_to_conservative(1.0, 0.0, 1.0)
                            : primitive_to_conservative(0.125, 0.0, 0.1);
    }

    State *d_U, *d_U_new;
    size_t bytes = (N + 2) * sizeof(State);
    CUDA_CHECK(cudaMalloc(&d_U, bytes));
    CUDA_CHECK(cudaMalloc(&d_U_new, bytes));
    CUDA_CHECK(cudaMemcpy(d_U, h_U, bytes, cudaMemcpyHostToDevice));

    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;

    auto t_start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    int step = 0;
    while (t < t_final) {
        apply_bc_kernel<<<1, 1>>>(d_U, N);

        // Known bottleneck: full array copied back to host every step just
        // to compute the CFL-limited timestep. Fixed in the next revision.
        CUDA_CHECK(cudaMemcpy(h_U, d_U, bytes, cudaMemcpyDeviceToHost));
        double Smax = 0.0;
        for (int i = 1; i <= N; ++i)
            Smax = fmax(Smax, fabs(velocity(h_U[i])) + sound_speed(h_U[i]));
        double dt = CFL * dx / Smax;
        if (t + dt > t_final) dt = t_final - t;

        update_kernel<<<BLOCKS, THREADS>>>(d_U, d_U_new, N, dt, dx);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_U, d_U_new);

        t += dt;
        ++step;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    CUDA_CHECK(cudaMemcpy(h_U, d_U, bytes, cudaMemcpyDeviceToHost));

    printf("GPU: %d steps, final t = %f, wall time = %.2f ms\n", step, t, elapsed_ms);

    std::ofstream out("sod_result_gpu.csv");
    out << "x,rho,u,p\n";
    for (int i = 1; i <= N; ++i) {
        double x = x_min + (i - 0.5) * dx;
        out << x << "," << h_U[i].rho << "," << velocity(h_U[i]) << "," << pressure(h_U[i]) << "\n";
    }
    out.close();
    printf("Wrote sod_result_gpu.csv\n");

    cudaFree(d_U);
    cudaFree(d_U_new);
    delete[] h_U;
    return 0;
}