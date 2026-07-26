#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>

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
    double u = velocity(s);
    double p = pressure(s);
    State f;
    f.rho = s.rho * u;
    f.mom = s.rho * u * u + p;
    f.E   = u * (s.E + p);
    return f;
}
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
State primitive_to_conservative(double rho, double u, double p) {
    State s;
    s.rho = rho;
    s.mom = rho * u;
    s.E   = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
    return s;
}

int main() {
    const int    N        = 20000;
    const double x_min    = 0.0;
    const double x_max    = 1.0;
    const double dx       = (x_max - x_min) / N;
    const double t_final  = 0.20;
    const double CFL      = 0.45;

    std::vector<State> U(N + 2);
    for (int i = 0; i < N + 2; ++i) {
        double x = x_min + (i - 0.5) * dx;
        U[i] = (x < 0.5) ? primitive_to_conservative(1.0, 0.0, 1.0)
                          : primitive_to_conservative(0.125, 0.0, 0.1);
    }

    // CHANGED: F and U_new are allocated ONCE, before the loop, instead of
    // being freshly heap-allocated every single iteration. Their contents
    // get overwritten each step, but the underlying memory is reused --
    // exactly the "don't allocate inside the hot loop" rule from before.
    std::vector<State> F(N + 1);
    std::vector<State> U_new(N + 2);

    auto t_start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    int step = 0;
    while (t < t_final) {
        U[0]     = U[1];
        U[N + 1] = U[N];

        double Smax = 0.0;
        for (int i = 1; i <= N; ++i)
            Smax = std::max(Smax, std::fabs(velocity(U[i])) + sound_speed(U[i]));
        double dt = CFL * dx / Smax;
        if (t + dt > t_final) dt = t_final - t;

        for (int i = 0; i <= N; ++i)
            F[i] = rusanov_flux(U[i], U[i + 1]);

        for (int i = 1; i <= N; ++i) {
            U_new[i].rho = U[i].rho - (dt / dx) * (F[i].rho - F[i - 1].rho);
            U_new[i].mom = U[i].mom - (dt / dx) * (F[i].mom - F[i - 1].mom);
            U_new[i].E   = U[i].E   - (dt / dx) * (F[i].E   - F[i - 1].E);
        }

        // CHANGED: swap instead of U = U_new (which was a full O(N) copy).
        // This is the exact same double-buffering trick the GPU version
        // uses with std::swap(d_U, d_U_new) -- both versions now share the
        // same structural idiom, not just the same physics.
        std::swap(U, U_new);

        t += dt;
        ++step;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    printf("CPU (v2, no per-step allocation): %d steps, final t = %f, wall time = %.2f ms\n", step, t, elapsed_ms);

    std::ofstream out("sod_result_cpu.csv");
    out << "x,rho,u,p\n";
    for (int i = 1; i <= N; ++i) {
        double x = x_min + (i - 0.5) * dx;
        out << x << "," << U[i].rho << "," << velocity(U[i]) << "," << pressure(U[i]) << "\n";
    }
    out.close();
    printf("Wrote sod_result_cpu.csv\n");
    return 0;
}