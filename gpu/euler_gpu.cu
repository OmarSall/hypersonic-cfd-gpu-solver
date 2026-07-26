// ============================================================================
// 1D Euler equations solver - Phase 1.6: Struct-of-Arrays (SoA) layout for
// memory coalescing. rho/mom/E now live in three separate flat arrays
// instead of one array of {rho,mom,E} structs, so that threads in the same
// warp reading the same field of consecutive cells hit contiguous memory.
// ============================================================================

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

const double GAMMA = 1.4;

// Only used as a temporary, per-thread return value -- never stored in an
// array, so it has no coalescing implications of its own.
struct Flux { double rho, mom, E; };

__device__ __host__ inline double velocity(double rho, double mom) { return mom / rho; }

__device__ __host__ inline double pressure(double rho, double mom, double E) {
    double u = velocity(rho, mom);
    return (GAMMA - 1.0) * (E - 0.5 * rho * u * u);
}

__device__ __host__ inline double sound_speed(double rho, double mom, double E) {
    double p = pressure(rho, mom, E);
    return sqrt(GAMMA * p / rho);
}

__device__ __host__ inline Flux physical_flux(double rho, double mom, double E) {
    double u = velocity(rho, mom);
    double p = pressure(rho, mom, E);
    Flux f;
    f.rho = rho * u;
    f.mom = rho * u * u + p;
    f.E   = u * (E + p);
    return f;
}

__device__ __host__ inline Flux rusanov_flux(double rhoL, double momL, double EL,
                                               double rhoR, double momR, double ER) {
    Flux FL = physical_flux(rhoL, momL, EL);
    Flux FR = physical_flux(rhoR, momR, ER);
    double sL = fabs(velocity(rhoL, momL)) + sound_speed(rhoL, momL, EL);
    double sR = fabs(velocity(rhoR, momR)) + sound_speed(rhoR, momR, ER);
    double Smax = fmax(sL, sR);
    Flux F;
    F.rho = 0.5 * (FL.rho + FR.rho) - 0.5 * Smax * (rhoR - rhoL);
    F.mom = 0.5 * (FL.mom + FR.mom) - 0.5 * Smax * (momR - momL);
    F.E   = 0.5 * (FL.E   + FR.E)   - 0.5 * Smax * (ER   - EL);
    return F;
}

struct WaveSpeed {
    __device__ double operator()(const thrust::tuple<double,double,double>& t) const {
        double rho = thrust::get<0>(t);
        double mom = thrust::get<1>(t);
        double E   = thrust::get<2>(t);
        return fabs(velocity(rho, mom)) + sound_speed(rho, mom, E);
    }
};

__global__ void apply_bc_kernel(double* rho, double* mom, double* E, int N) {
    rho[0] = rho[1]; mom[0] = mom[1]; E[0] = E[1];
    rho[N+1] = rho[N]; mom[N+1] = mom[N]; E[N+1] = E[N];
}

__global__ void update_kernel(const double* rho, const double* mom, const double* E,
                               double* rho_new, double* mom_new, double* E_new,
                               int N, double dt, double dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= N) {
        Flux F_left  = rusanov_flux(rho[i-1], mom[i-1], E[i-1], rho[i], mom[i], E[i]);
        Flux F_right = rusanov_flux(rho[i], mom[i], E[i], rho[i+1], mom[i+1], E[i+1]);
        rho_new[i] = rho[i] - (dt / dx) * (F_right.rho - F_left.rho);
        mom_new[i] = mom[i] - (dt / dx) * (F_right.mom - F_left.mom);
        E_new[i]   = E[i]   - (dt / dx) * (F_right.E   - F_left.E);
    }
}

int main() {
    const int    N       = 20000;
    const double x_min   = 0.0;
    const double x_max   = 1.0;
    const double dx      = (x_max - x_min) / N;
    const double t_final = 0.20;
    const double CFL     = 0.45;

    std::vector<double> h_rho(N+2), h_mom(N+2), h_E(N+2);
    for (int i = 0; i < N + 2; ++i) {
        double x = x_min + (i - 0.5) * dx;
        double r = (x < 0.5) ? 1.0 : 0.125;
        double u = 0.0;
        double p = (x < 0.5) ? 1.0 : 0.1;
        h_rho[i] = r;
        h_mom[i] = r * u;
        h_E[i]   = p / (GAMMA - 1.0) + 0.5 * r * u * u;
    }

    double *d_rho, *d_mom, *d_E, *d_rho_new, *d_mom_new, *d_E_new;
    size_t bytes = (N + 2) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_rho, bytes));
    CUDA_CHECK(cudaMalloc(&d_mom, bytes));
    CUDA_CHECK(cudaMalloc(&d_E, bytes));
    CUDA_CHECK(cudaMalloc(&d_rho_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_mom_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_E_new, bytes));
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mom, h_mom.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E.data(), bytes, cudaMemcpyHostToDevice));

    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;

    auto t_start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    int step = 0;
    while (t < t_final) {
        apply_bc_kernel<<<1, 1>>>(d_rho, d_mom, d_E, N);

        thrust::device_ptr<double> rho_ptr(d_rho), mom_ptr(d_mom), E_ptr(d_E);
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(rho_ptr+1, mom_ptr+1, E_ptr+1));
        auto zip_end   = thrust::make_zip_iterator(thrust::make_tuple(rho_ptr+N+1, mom_ptr+N+1, E_ptr+N+1));
        double Smax = thrust::transform_reduce(zip_begin, zip_end, WaveSpeed(), 0.0, thrust::maximum<double>());

        double dt = CFL * dx / Smax;
        if (t + dt > t_final) dt = t_final - t;

        update_kernel<<<BLOCKS, THREADS>>>(d_rho, d_mom, d_E, d_rho_new, d_mom_new, d_E_new, N, dt, dx);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_rho, d_rho_new);
        std::swap(d_mom, d_mom_new);
        std::swap(d_E, d_E_new);

        t += dt;
        ++step;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mom.data(), d_mom, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_E.data(), d_E, bytes, cudaMemcpyDeviceToHost));

    printf("GPU (v3, SoA): %d steps, final t = %f, wall time = %.2f ms\n", step, t, elapsed_ms);

    std::ofstream out("sod_result_gpu.csv");
    out << "x,rho,u,p\n";
    for (int i = 1; i <= N; ++i) {
        double x = x_min + (i - 0.5) * dx;
        out << x << "," << h_rho[i] << "," << velocity(h_rho[i],h_mom[i]) << "," << pressure(h_rho[i],h_mom[i],h_E[i]) << "\n";
    }
    out.close();
    printf("Wrote sod_result_gpu.csv\n");

    cudaFree(d_rho); cudaFree(d_mom); cudaFree(d_E);
    cudaFree(d_rho_new); cudaFree(d_mom_new); cudaFree(d_E_new);
    return 0;
}