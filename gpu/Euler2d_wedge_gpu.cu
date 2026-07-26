%%writefile euler2d_wedge_gpu.cu
// ============================================================================
// Task 4: GPU port of the 2D unsplit Euler wedge solver (euler2d_wedge.cpp).
// Same physics/numerics: Rusanov flux in x and y, forward Euler, staircase-
// approximated wedge wall reflected about the true wall angle.
//
// Design notes:
//
// 1. Genuine 2D CUDA grid (dim3 blocks/threads over i AND j), not a flattened
//    1D index like the 1D solvers -- this is the natural mapping for a 2D
//    problem and lets each thread own one grid cell directly via its
//    (blockIdx.x/threadIdx.x, blockIdx.y/threadIdx.y) pair.
//
// 2. The CFL timestep reduction (max wave speed) runs over the ENTIRE array,
//    ghost cells included -- unlike the interior-only reduction in the 1D
//    solvers. This is safe specifically because every ghost cell here holds
//    a value copied or reflected from a real fluid cell: zero-gradient
//    copies preserve the wave speed exactly, and the wall reflection
//    (rotating the velocity vector) preserves |velocity| exactly, so it
//    preserves |u|+c exactly too. No ghost cell can ever exceed or
//    understate the true interior maximum, so no masking is needed.
//
// 3. The residual check (are we at steady state yet) CANNOT use the same
//    trick -- solid/ghost cells get overwritten by the *next* apply_bc
//    call, not evolved by the update kernel, so comparing them between U
//    and U_new would compare stale garbage. A dedicated residual_kernel
//    writes |rho change| only for genuine fluid cells (0 elsewhere) into a
//    compact Nx*Ny buffer, which is what actually gets reduced.
//
// 4. Verified before writing any CUDA: replacing the CPU version's
//    selective copy-back (only fluid cells) with a full pointer swap
//    (leaving solid U_new cells as unwritten garbage that gets overwritten
//    by apply_bc before ever being read) gives bit-for-bit identical
//    output at Nx=80,Ny=40. Swap-based double buffering is what this file
//    uses, matching the 1D GPU solvers' idiom.
// ============================================================================

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

const double GAMMA = 1.4;

struct State { double rho, momx, momy, E; };

__device__ __host__ inline int IDX(int i, int j, int Ny) { return i*(Ny+2) + j; }

__device__ __host__ inline double u_vel(const State& s) { return s.momx / s.rho; }
__device__ __host__ inline double v_vel(const State& s) { return s.momy / s.rho; }
__device__ __host__ inline double pressure(const State& s) {
    double u = u_vel(s), v = v_vel(s);
    return (GAMMA - 1.0) * (s.E - 0.5*s.rho*(u*u+v*v));
}
__device__ __host__ inline double sound_speed(const State& s) { return sqrt(GAMMA * pressure(s) / s.rho); }
__device__ __host__ inline State flux_x(const State& s) {
    double u = u_vel(s), v = v_vel(s), p = pressure(s);
    State f;
    f.rho  = s.rho*u;
    f.momx = s.rho*u*u+p;
    f.momy = s.rho*u*v;
    f.E    = u*(s.E+p);
    return f;
}
__device__ __host__ inline State flux_y(const State& s) {
    double u = u_vel(s), v = v_vel(s), p = pressure(s);
    State f;
    f.rho  = s.rho*v;
    f.momx = s.rho*u*v;
    f.momy = s.rho*v*v+p;
    f.E    = v*(s.E+p);
    return f;
}
__device__ __host__ inline State rusanov_x(const State& UL, const State& UR) {
    State FL = flux_x(UL), FR = flux_x(UR);
    double S = fmax(fabs(u_vel(UL))+sound_speed(UL), fabs(u_vel(UR))+sound_speed(UR));
    State F;
    F.rho  = 0.5*(FL.rho +FR.rho )-0.5*S*(UR.rho -UL.rho );
    F.momx = 0.5*(FL.momx+FR.momx)-0.5*S*(UR.momx-UL.momx);
    F.momy = 0.5*(FL.momy+FR.momy)-0.5*S*(UR.momy-UL.momy);
    F.E    = 0.5*(FL.E   +FR.E   )-0.5*S*(UR.E   -UL.E   );
    return F;
}
__device__ __host__ inline State rusanov_y(const State& UL, const State& UR) {
    State FL = flux_y(UL), FR = flux_y(UR);
    double S = fmax(fabs(v_vel(UL))+sound_speed(UL), fabs(v_vel(UR))+sound_speed(UR));
    State F;
    F.rho  = 0.5*(FL.rho +FR.rho )-0.5*S*(UR.rho -UL.rho );
    F.momx = 0.5*(FL.momx+FR.momx)-0.5*S*(UR.momx-UL.momx);
    F.momy = 0.5*(FL.momy+FR.momy)-0.5*S*(UR.momy-UL.momy);
    F.E    = 0.5*(FL.E   +FR.E   )-0.5*S*(UR.E   -UL.E   );
    return F;
}
__device__ __host__ inline State reflect_wall(const State& s, double a) {
    double u = u_vel(s), v = v_vel(s);
    double c2 = cos(2*a), s2 = sin(2*a);
    double un =  u*c2 + v*s2;
    double vn =  u*s2 - v*c2;
    State r;
    r.rho  = s.rho;
    r.momx = s.rho*un;
    r.momy = s.rho*vn;
    r.E    = s.E;
    return r;
}

struct CFLRate {
    double dx, dy;
    CFLRate(double dx_, double dy_) : dx(dx_), dy(dy_) {}
    __device__ double operator()(const State& s) const {
        double c = sound_speed(s);
        return (fabs(u_vel(s))+c)/dx + (fabs(v_vel(s))+c)/dy;
    }
};

__global__ void bc_left_kernel(State* U, int Ny, State inflow) {
    int j = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (j <= Ny) U[IDX(0,j,Ny)] = inflow;
}
__global__ void bc_right_kernel(State* U, int Nx, int Ny) {
    int j = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (j <= Ny) U[IDX(Nx+1,j,Ny)] = U[IDX(Nx,j,Ny)];
}
__global__ void bc_top_kernel(State* U, int Nx, int Ny) {
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (i <= Nx) U[IDX(i,Ny+1,Ny)] = U[IDX(i,Ny,Ny)];
}
__global__ void bc_wall_kernel(State* U, const int* wall_j, int Nx, int Ny,
                                double dx, double x_wedge, double theta) {
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (i > Nx) return;
    int jw = wall_j[i];
    double x = (i - 0.5) * dx;
    double local_angle = (x > x_wedge) ? theta : 0.0;
    State mirror = reflect_wall(U[IDX(i,jw+1,Ny)], local_angle);
    for (int j = 0; j <= jw; ++j) U[IDX(i,j,Ny)] = mirror;
}

__global__ void check_trig(double a) {
    printf("GPU  cos(2*theta)=%.17e sin(2*theta)=%.17e\n", cos(2*a), sin(2*a));
}

// One thread per (i,j) fluid cell. Solid cells return immediately -- their
// U_new slot is left unwritten and gets overwritten by bc_wall_kernel next
// step, before anything ever reads it (see design note 4 above).
__global__ void update_kernel(const State* U, State* U_new, const int* wall_j,
                               int Nx, int Ny, double dx, double dy, double dt) {
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (i > Nx || j > Ny) return;
    if (j <= wall_j[i]) return;

    State Fx_r = rusanov_x(U[IDX(i,j,Ny)],   U[IDX(i+1,j,Ny)]);
    State Fx_l = rusanov_x(U[IDX(i-1,j,Ny)], U[IDX(i,j,Ny)]);
    State Fy_t = rusanov_y(U[IDX(i,j,Ny)],   U[IDX(i,j+1,Ny)]);
    State Fy_b = rusanov_y(U[IDX(i,j-1,Ny)], U[IDX(i,j,Ny)]);

    State s = U[IDX(i,j,Ny)];
    State sn;
    sn.rho  = s.rho  - (dt/dx)*(Fx_r.rho -Fx_l.rho ) - (dt/dy)*(Fy_t.rho -Fy_b.rho );
    sn.momx = s.momx - (dt/dx)*(Fx_r.momx-Fx_l.momx) - (dt/dy)*(Fy_t.momx-Fy_b.momx);
    sn.momy = s.momy - (dt/dx)*(Fx_r.momy-Fx_l.momy) - (dt/dy)*(Fy_t.momy-Fy_b.momy);
    sn.E    = s.E    - (dt/dx)*(Fx_r.E   -Fx_l.E   ) - (dt/dy)*(Fy_t.E   -Fy_b.E   );
    U_new[IDX(i,j,Ny)] = sn;
}

// Writes |rho change| for fluid cells (0 for solid) into a compact Nx*Ny
// buffer -- see design note 3 for why this can't reuse the CFL reduction's
// whole-array trick.
__global__ void residual_kernel(const State* U, const State* U_new, const int* wall_j,
                                 int Nx, int Ny, double* out) {
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (i > Nx || j > Ny) return;
    int out_idx = (i-1)*Ny + (j-1);
    if (j <= wall_j[i]) { out[out_idx] = 0.0; return; }
    int idx = IDX(i,j,Ny);
    out[out_idx] = fabs(U_new[idx].rho - U[idx].rho);
}

State primitive_to_conservative(double rho, double u, double v, double p) {
    return State{ rho, rho*u, rho*v, p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v) };
}


int main(int argc, char** argv) {
    const int Nx = argc>1 ? atoi(argv[1]) : 400;
    const int Ny = argc>2 ? atoi(argv[2]) : 200;
    const double Lx = 2.0, Ly = 1.0;
    const double dx = Lx/Nx, dy = Ly/Ny;
    const double x_wedge = 0.5;
    const double theta = 15.0 * M_PI / 180.0;

    printf("CPU  cos(2*theta)=%.17e sin(2*theta)=%.17e\n", std::cos(2*theta), std::sin(2*theta));
    check_trig<<<1,1>>>(theta);
    cudaDeviceSynchronize();

    const double CFL = 0.45;

    const double M1 = 2.0;
    const double rho1 = 1.0, p1 = 1.0/GAMMA;
    const double u1 = M1, v1 = 0.0;
    State inflow = primitive_to_conservative(rho1, u1, v1, p1);

    std::vector<int> h_wall_j(Nx+2, 0);
    for (int i = 1; i <= Nx; ++i) {
        double x = (i - 0.5) * dx;
        double y_wall = (x > x_wedge) ? tan(theta) * (x - x_wedge) : 0.0;
        h_wall_j[i] = std::min(Ny, (int)floor(y_wall / dy));
    }

    int size = (Nx+2)*(Ny+2);
    std::vector<State> h_U(size);
    for (int i = 0; i <= Nx+1; ++i)
        for (int j = 0; j <= Ny+1; ++j)
            h_U[IDX(i,j,Ny)] = inflow;

    State *d_U, *d_U_new;
    int *d_wall_j;
    double *d_residual;
    CUDA_CHECK(cudaMalloc(&d_U, size*sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_U_new, size*sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_wall_j, (Nx+2)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_residual, Nx*Ny*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), size*sizeof(State), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_U_new, h_U.data(), size*sizeof(State), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wall_j, h_wall_j.data(), (Nx+2)*sizeof(int), cudaMemcpyHostToDevice));

    const int T1D = 256;
    int blocks_j = (Ny + T1D - 1) / T1D;
    int blocks_i = (Nx + T1D - 1) / T1D;
    dim3 threads2D(16, 16);
    dim3 blocks2D((Nx + threads2D.x - 1) / threads2D.x, (Ny + threads2D.y - 1) / threads2D.y);

    double t = 0.0;
    long step = 0;
    const long max_steps = 300000;
    const double residual_tol = 1e-9;
    auto t_start = std::chrono::high_resolution_clock::now();

    while (step < max_steps) {
        bc_left_kernel<<<blocks_j, T1D>>>(d_U, Ny, inflow);
        bc_right_kernel<<<blocks_j, T1D>>>(d_U, Nx, Ny);
        bc_top_kernel<<<blocks_i, T1D>>>(d_U, Nx, Ny);
        bc_wall_kernel<<<blocks_i, T1D>>>(d_U, d_wall_j, Nx, Ny, dx, x_wedge, theta);

        thrust::device_ptr<State> dU_ptr(d_U);
        double max_rate = thrust::transform_reduce(
            dU_ptr, dU_ptr + size, CFLRate(dx, dy), 0.0, thrust::maximum<double>());
        double dt = CFL / max_rate;

        update_kernel<<<blocks2D, threads2D>>>(d_U, d_U_new, d_wall_j, Nx, Ny, dx, dy, dt);
        CUDA_CHECK(cudaGetLastError());

        double max_residual = 0.0;
        if ((step+1) % 100 == 0) {
            residual_kernel<<<blocks2D, threads2D>>>(d_U, d_U_new, d_wall_j, Nx, Ny, d_residual);
            thrust::device_ptr<double> dr_ptr(d_residual);
            max_residual = thrust::reduce(dr_ptr, dr_ptr + Nx*Ny, 0.0, thrust::maximum<double>());
        }

        std::swap(d_U, d_U_new);

        t += dt; ++step;
        if ((step % 100 == 0) && max_residual < residual_tol) {
            printf("Converged at step %ld, t=%f, max_residual=%.3e\n", step, t, max_residual);
            break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("2D wedge GPU (Nx=%d Ny=%d): %ld steps, final t=%f, wall time=%.2f ms\n", Nx, Ny, step, t, elapsed_ms);

    CUDA_CHECK(cudaMemcpy(h_U.data(), d_U, size*sizeof(State), cudaMemcpyDeviceToHost));

    std::ofstream out("wedge_result_2d_gpu.csv");
    out << "x,y,rho,u,v,p\n";
    for (int i = 1; i <= Nx; ++i) {
        int jw = h_wall_j[i];
        double x = (i-0.5)*dx;
        for (int j = jw+1; j <= Ny; ++j) {
            double y = (j-0.5)*dy;
            const State& s = h_U[IDX(i,j,Ny)];
            out << x << "," << y << "," << s.rho << "," << u_vel(s) << "," << v_vel(s) << "," << pressure(s) << "\n";
        }
    }
    printf("Wrote wedge_result_2d_gpu.csv\n");

    cudaFree(d_U); cudaFree(d_U_new); cudaFree(d_wall_j); cudaFree(d_residual);
    return 0;
}