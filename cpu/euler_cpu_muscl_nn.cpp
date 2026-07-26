// ============================================================================
// Sod-shock-tube ROBUSTNESS CHECK for the NN-hybrid limiter (Task 5). This is
// a direct copy of euler_cpu_muscl.cpp with ONE change: compute_L now uses
// nn_is_troubled (troubled_cell_weights.h) instead of pure minmod to decide,
// per cell, whether to limit (troubled -> minmod, identical to before) or use
// the raw unlimited central-difference slope (smooth -> no limiting).
//
// Purpose: smooth_advection_accuracy_test_nn.cpp showed a striking order
// result (2.02 vs minmod's 1.61) but that test never contains a genuine
// shock -- it can't tell us whether the network correctly flags the Sod
// shock/contact as "troubled" and falls back to minmod there. If it doesn't,
// this run will show oscillations, overshoots, or negative rho/p near x=0.5.
// This is the FIRST of two checks flagged before trusting the NN result
// (the second is generalization to a non-sine smooth shape, e.g. a Gaussian).
//
// Everything else (Rusanov flux, SSP-RK2, Sod ICs, N=20000, CFL=0.45) is
// unchanged from euler_cpu_muscl.cpp so the L2 error is directly comparable
// to the established minmod baseline (rho=2.27129e-03, u=3.10181e-03,
// p=6.76008e-04 at N=20000).
// ============================================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include "troubled_cell_weights.h"

const double GAMMA = 1.4;
struct State { double rho, mom, E; };

double velocity(const State& s) { return s.mom / s.rho; }
double pressure(const State& s) {
    double u = velocity(s);
    return (GAMMA - 1.0) * (s.E - 0.5 * s.rho * u * u);
}
double sound_speed(const State& s) {
    double p = pressure(s);
    return std::sqrt(GAMMA * p / s.rho);
}
State physical_flux(const State& s) {
    double u = velocity(s), p = pressure(s);
    return State{ s.rho*u, s.rho*u*u + p, u*(s.E+p) };
}
State rusanov_flux(const State& UL, const State& UR) {
    State FL = physical_flux(UL), FR = physical_flux(UR);
    double sL = std::fabs(velocity(UL)) + sound_speed(UL);
    double sR = std::fabs(velocity(UR)) + sound_speed(UR);
    double Smax = std::max(sL, sR);
    State F;
    F.rho = 0.5*(FL.rho+FR.rho) - 0.5*Smax*(UR.rho-UL.rho);
    F.mom = 0.5*(FL.mom+FR.mom) - 0.5*Smax*(UR.mom-UL.mom);
    F.E   = 0.5*(FL.E  +FR.E)   - 0.5*Smax*(UR.E  -UL.E);
    return F;
}
State primitive_to_conservative(double rho, double u, double p) {
    return State{ rho, rho*u, p/(GAMMA-1.0) + 0.5*rho*u*u };
}

double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (a > 0.0) ? std::min(a, b) : std::max(a, b);
}

// Three-argument minmod: 0 unless x,y,z all share a sign, in which case the
// smallest-magnitude one. Building block of the MC (monotonized-central,
// Van Leer 1977) limiter below.
double minmod3(double x, double y, double z) {
    if (x > 0.0 && y > 0.0 && z > 0.0) return std::min({x, y, z});
    if (x < 0.0 && y < 0.0 && z < 0.0) return std::max({x, y, z});
    return 0.0;
}

// MC limiter: minmod3(2a, 2b, (a+b)/2). Provably TVD (it's a member of
// Sweby's TVD limiter family, same as minmod, just less diffusive) AND, in
// any region where a and b are close in magnitude (a true smooth interior,
// no curvature-asymmetry), it reduces almost exactly to the raw central
// average (a+b)/2 -- so it recovers most of the accuracy the NN's raw
// unlimited slope was going for, but WITHOUT giving up the TVD guarantee.
// Used below as the fallback when the NN says "smooth" instead of the fully
// unlimited central difference, so the scheme is TVD by construction no
// matter what the classifier gets wrong (see project notes: the fully
// unlimited fallback produced a genuine 9% TV(rho) violation at the Sod
// contact discontinuity, because the real contact is smeared over ~80
// cells with asymmetric curvature at its edges -- exactly the case a raw
// central difference can overshoot on, and no amount of retraining the
// classifier changes that; only bounding the limiter itself does).
double mc_limiter(double a, double b) {
    return minmod3(2.0*a, 2.0*b, 0.5*(a+b));
}

void apply_bc(std::vector<State>& U, int N) {
    U[1] = U[0] = U[2];
    U[N+2] = U[N+3] = U[N+1];
}

// The only change vs euler_cpu_muscl.cpp's compute_L: the slope for cell i is
// minmod's limited slope if the NN flags cell i as troubled (identical
// behavior to the pure-minmod baseline), or the raw unlimited central
// difference if the NN calls it smooth.
int n_troubled_calls = 0, n_troubled_flagged = 0;   // instrumentation for this check only
void compute_L_nn(const std::vector<State>& U, int N, double dx,
               std::vector<State>& slope, std::vector<State>& F, std::vector<State>& L) {
    for (int i = 1; i <= N+2; ++i) {
        bool troubled = nn_is_troubled(U[i-1].rho, U[i].rho, U[i+1].rho,
                                        pressure(U[i-1]), pressure(U[i]), pressure(U[i+1]));
        ++n_troubled_calls;
        if (troubled) {
            ++n_troubled_flagged;
            slope[i].rho = minmod(U[i].rho-U[i-1].rho, U[i+1].rho-U[i].rho);
            slope[i].mom = minmod(U[i].mom-U[i-1].mom, U[i+1].mom-U[i].mom);
            slope[i].E   = minmod(U[i].E  -U[i-1].E,   U[i+1].E  -U[i].E);
        } else {
            slope[i].rho = mc_limiter(U[i].rho-U[i-1].rho, U[i+1].rho-U[i].rho);
            slope[i].mom = mc_limiter(U[i].mom-U[i-1].mom, U[i+1].mom-U[i].mom);
            slope[i].E   = mc_limiter(U[i].E  -U[i-1].E,   U[i+1].E  -U[i].E);
        }
    }
    for (int i = 1; i <= N+1; ++i) {
        State UL{ U[i].rho   + 0.5*slope[i].rho,   U[i].mom   + 0.5*slope[i].mom,   U[i].E   + 0.5*slope[i].E };
        State UR{ U[i+1].rho - 0.5*slope[i+1].rho, U[i+1].mom - 0.5*slope[i+1].mom, U[i+1].E - 0.5*slope[i+1].E };
        F[i] = rusanov_flux(UL, UR);
    }
    for (int i = 2; i <= N+1; ++i) {
        L[i].rho = -(F[i].rho - F[i-1].rho) / dx;
        L[i].mom = -(F[i].mom - F[i-1].mom) / dx;
        L[i].E   = -(F[i].E   - F[i-1].E)   / dx;
    }
}

double compute_dt(const std::vector<State>& U, int N, double dx, double CFL) {
    double Smax = 0.0;
    for (int i = 2; i <= N+1; ++i)
        Smax = std::max(Smax, std::fabs(velocity(U[i])) + sound_speed(U[i]));
    return CFL * dx / Smax;
}

int main(int argc, char** argv) {
    const int    N       = argc > 1 ? std::atoi(argv[1]) : 20000;
    const double x_min = 0.0, x_max = 1.0;
    const double dx = (x_max - x_min) / N;
    const double t_final = 0.20, CFL = 0.45;

    std::vector<State> U(N+4);
    for (int i = 0; i < N+4; ++i) {
        double x = x_min + (i - 2.0 + 0.5) * dx;
        U[i] = (x < 0.5) ? primitive_to_conservative(1.0,0.0,1.0)
                          : primitive_to_conservative(0.125,0.0,0.1);
    }

    std::vector<State> slope(N+4), F(N+2), L1(N+4), L2(N+4), U_star(N+4);

    double rho_min_seen = 1e300, rho_max_seen = -1e300;
    double p_min_seen = 1e300, p_max_seen = -1e300;

    auto t_start = std::chrono::high_resolution_clock::now();
    double t = 0.0; int step = 0;
    while (t < t_final) {
        apply_bc(U, N);
        double dt = compute_dt(U, N, dx, CFL);
        if (t + dt > t_final) dt = t_final - t;

        compute_L_nn(U, N, dx, slope, F, L1);
        U_star = U;
        for (int i = 2; i <= N+1; ++i) {
            U_star[i].rho = U[i].rho + dt*L1[i].rho;
            U_star[i].mom = U[i].mom + dt*L1[i].mom;
            U_star[i].E   = U[i].E   + dt*L1[i].E;
        }

        apply_bc(U_star, N);
        compute_L_nn(U_star, N, dx, slope, F, L2);
        for (int i = 2; i <= N+1; ++i) {
            double rho2 = U_star[i].rho + dt*L2[i].rho;
            double mom2 = U_star[i].mom + dt*L2[i].mom;
            double E2   = U_star[i].E   + dt*L2[i].E;
            U[i].rho = 0.5*(U[i].rho + rho2);
            U[i].mom = 0.5*(U[i].mom + mom2);
            U[i].E   = 0.5*(U[i].E   + E2);
        }

        for (int i = 2; i <= N+1; ++i) {
            rho_min_seen = std::min(rho_min_seen, U[i].rho);
            rho_max_seen = std::max(rho_max_seen, U[i].rho);
            double p = pressure(U[i]);
            p_min_seen = std::min(p_min_seen, p);
            p_max_seen = std::max(p_max_seen, p);
        }

        t += dt; ++step;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("CPU (MUSCL+RK2+NN): %d steps, final t=%f, wall time=%.2f ms\n", step, t, elapsed_ms);
    printf("NN flagged %d/%d slope evaluations as troubled (%.2f%%)\n",
           n_troubled_flagged, n_troubled_calls, 100.0*n_troubled_flagged/n_troubled_calls);
    printf("rho range over run: [%.6f, %.6f]  (IC bounds: [0.125, 1.0])\n", rho_min_seen, rho_max_seen);
    printf("p   range over run: [%.6f, %.6f]  (IC bounds: [0.1, 1.0])\n", p_min_seen, p_max_seen);

    std::ofstream out("sod_result_cpu_muscl_nn_N" + std::to_string(N) + ".csv");
    out << "x,rho,u,p\n";
    for (int i = 2; i <= N+1; ++i) {
        double x = x_min + (i - 2.0 + 0.5) * dx;
        out << x << "," << U[i].rho << "," << velocity(U[i]) << "," << pressure(U[i]) << "\n";
    }
    printf("Wrote sod_result_cpu_muscl_nn_N%d.csv\n", N);
    return 0;
}