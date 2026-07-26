// ============================================================================
// 1D Euler equations solver - Finite Volume Method, Rusanov flux
// Validation case: Sod shock tube
//
// Governing equations (conservative form):
//   dU/dt + dF(U)/dx = 0
//   U = [rho, rho*u, E]^T
//   F = [rho*u, rho*u^2 + p, u*(E+p)]^T
//   p = (gamma - 1) * (E - 0.5*rho*u^2)
//
// This is Phase 0: pure CPU, single-threaded, correctness-first.
// Phase 1 will port the flux/update loop below to a CUDA kernel.
// ============================================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>

const double GAMMA = 1.4;

// Conserved state at one cell
struct State {
    double rho;   // density
    double mom;   // momentum = rho * u
    double E;     // total energy per volume
};

// --- Helper conversions -----------------------------------------------------

double velocity(const State& s) { return s.mom / s.rho; }

double pressure(const State& s) {
    double u = velocity(s);
    return (GAMMA - 1.0) * (s.E - 0.5 * s.rho * u * u);
}

double sound_speed(const State& s) {
    double p = pressure(s);
    return std::sqrt(GAMMA * p / s.rho);
}

// Physical flux F(U) for the Euler equations
State physical_flux(const State& s) {
    double u = velocity(s);
    double p = pressure(s);
    State f;
    f.rho = s.rho * u;
    f.mom = s.rho * u * u + p;
    f.E   = u * (s.E + p);
    return f;
}

// --- Numerical flux: Rusanov (local Lax-Friedrichs) -------------------------
// F_{i+1/2} = 0.5*(F(UL)+F(UR)) - 0.5*Smax*(UR-UL)
// Smax = max(|u|+c) evaluated on both sides

State rusanov_flux(const State& UL, const State& UR) {
    State FL = physical_flux(UL);
    State FR = physical_flux(UR);

    double sL = std::fabs(velocity(UL)) + sound_speed(UL);
    double sR = std::fabs(velocity(UR)) + sound_speed(UR);
    double Smax = std::max(sL, sR);

    State F;
    F.rho = 0.5 * (FL.rho + FR.rho) - 0.5 * Smax * (UR.rho - UL.rho);
    F.mom = 0.5 * (FL.mom + FR.mom) - 0.5 * Smax * (UR.mom - UL.mom);
    F.E   = 0.5 * (FL.E   + FR.E)   - 0.5 * Smax * (UR.E   - UL.E);
    return F;
}

// --- Initial condition: Sod shock tube --------------------------------------
// Left state  (x < 0.5): rho=1.0,   u=0, p=1.0
// Right state (x >= 0.5): rho=0.125, u=0, p=0.1
// Domain: x in [0, 1], run until t_final = 0.2 (standard benchmark values)

State primitive_to_conservative(double rho, double u, double p) {
    State s;
    s.rho = rho;
    s.mom = rho * u;
    s.E   = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
    return s;
}

int main() {
    // --- Grid setup ---
    const int    N        = 400;      // number of interior cells
    const double x_min    = 0.0;
    const double x_max    = 1.0;
    const double dx       = (x_max - x_min) / N;
    const double t_final  = 0.20;
    const double CFL      = 0.45;     // Rusanov + forward Euler: keep <= 0.5

    // 2 ghost cells (one each side) for transmissive boundary conditions
    std::vector<State> U(N + 2);

    // --- Initialize Sod shock tube ---
    for (int i = 0; i < N + 2; ++i) {
        double x = x_min + (i - 0.5) * dx;  // cell center (i=0 is left ghost)
        if (x < 0.5)
            U[i] = primitive_to_conservative(1.0, 0.0, 1.0);
        else
            U[i] = primitive_to_conservative(0.125, 0.0, 0.1);
    }

    // --- Time marching loop ---
    double t = 0.0;
    int step = 0;
    while (t < t_final) {
        // Transmissive (outflow) boundary conditions: copy nearest interior cell
        U[0]     = U[1];
        U[N + 1] = U[N];

        // CFL-limited timestep: dt = CFL * dx / max(|u|+c) over the domain
        double Smax = 0.0;
        for (int i = 1; i <= N; ++i)
            Smax = std::max(Smax, std::fabs(velocity(U[i])) + sound_speed(U[i]));
        double dt = CFL * dx / Smax;
        if (t + dt > t_final) dt = t_final - t;

        // Compute fluxes at all interior interfaces (i+1/2 for i = 0..N)
        std::vector<State> F(N + 1);
        for (int i = 0; i <= N; ++i)
            F[i] = rusanov_flux(U[i], U[i + 1]);

        // Finite volume update: U_i^{n+1} = U_i^n - (dt/dx)*(F_{i+1/2} - F_{i-1/2})
        std::vector<State> U_new = U;
        for (int i = 1; i <= N; ++i) {
            U_new[i].rho = U[i].rho - (dt / dx) * (F[i].rho - F[i - 1].rho);
            U_new[i].mom = U[i].mom - (dt / dx) * (F[i].mom - F[i - 1].mom);
            U_new[i].E   = U[i].E   - (dt / dx) * (F[i].E   - F[i - 1].E);
        }
        U = U_new;

        t += dt;
        ++step;
    }

    printf("Done. %d steps, final t = %f\n", step, t);

    // --- Write result to CSV for plotting / comparison against analytical solution ---
    std::ofstream out("sod_result.csv");
    out << "x,rho,u,p\n";
    for (int i = 1; i <= N; ++i) {
        double x = x_min + (i - 0.5) * dx;
        out << x << "," << U[i].rho << "," << velocity(U[i]) << "," << pressure(U[i]) << "\n";
    }
    out.close();
    printf("Wrote sod_result.csv\n");

    return 0;
}