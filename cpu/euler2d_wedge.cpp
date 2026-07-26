// First-order unsplit 2D Euler solver: Rusanov flux in x and y, forward Euler.
// Validation case: Mach 2 flow over a 15-degree wedge, staircase-approximated
// on a Cartesian grid. Ground truth: oblique_shock_relation.py predicts a
// weak-shock angle beta = 45.344 deg for M1=2, theta=15deg, gamma=1.4.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>

const double GAMMA = 1.4;

struct State { double rho, momx, momy, E; };

inline double u_vel(const State& s) { return s.momx / s.rho; }
inline double v_vel(const State& s) { return s.momy / s.rho; }
inline double pressure(const State& s) {
    double u = u_vel(s), v = v_vel(s);
    return (GAMMA - 1.0) * (s.E - 0.5*s.rho*(u*u+v*v));
}
inline double sound_speed(const State& s) { return std::sqrt(GAMMA * pressure(s) / s.rho); }
inline State flux_x(const State& s) {
    double u = u_vel(s), v = v_vel(s), p = pressure(s);
    return State{ s.rho*u, s.rho*u*u+p, s.rho*u*v, u*(s.E+p) };
}
inline State flux_y(const State& s) {
    double u = u_vel(s), v = v_vel(s), p = pressure(s);
    return State{ s.rho*v, s.rho*u*v, s.rho*v*v+p, v*(s.E+p) };
}
inline State rusanov_x(const State& UL, const State& UR) {
    State FL=flux_x(UL), FR=flux_x(UR);
    double S = std::max(std::fabs(u_vel(UL))+sound_speed(UL), std::fabs(u_vel(UR))+sound_speed(UR));
    return State{ 0.5*(FL.rho+FR.rho)-0.5*S*(UR.rho-UL.rho),
                  0.5*(FL.momx+FR.momx)-0.5*S*(UR.momx-UL.momx),
                  0.5*(FL.momy+FR.momy)-0.5*S*(UR.momy-UL.momy),
                  0.5*(FL.E+FR.E)-0.5*S*(UR.E-UL.E) };
}
inline State rusanov_y(const State& UL, const State& UR) {
    State FL=flux_y(UL), FR=flux_y(UR);
    double S = std::max(std::fabs(v_vel(UL))+sound_speed(UL), std::fabs(v_vel(UR))+sound_speed(UR));
    return State{ 0.5*(FL.rho+FR.rho)-0.5*S*(UR.rho-UL.rho),
                  0.5*(FL.momx+FR.momx)-0.5*S*(UR.momx-UL.momx),
                  0.5*(FL.momy+FR.momy)-0.5*S*(UR.momy-UL.momy),
                  0.5*(FL.E+FR.E)-0.5*S*(UR.E-UL.E) };
}
inline State reflect_y(const State& s) { return State{ s.rho, s.momx, -s.momy, s.E }; }
// Reflect velocity about a wall inclined at angle a from horizontal (the
// KNOWN true wedge angle, not something inferred from the staircase shape).
// At a=0 this reduces exactly to reflect_y -- same formula covers the flat
// plate and the wedge, no separate tread/riser cases needed.
inline State reflect_wall(const State& s, double a) {
    double u = u_vel(s), v = v_vel(s);
    double c2 = std::cos(2*a), s2 = std::sin(2*a);
    double un =  u*c2 + v*s2;
    double vn =  u*s2 - v*c2;
    return State{ s.rho, s.rho*un, s.rho*vn, s.E };
}
State primitive_to_conservative(double rho, double u, double v, double p) {
    return State{ rho, rho*u, rho*v, p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v) };
}
inline int IDX(int i, int j, int Ny) { return i*(Ny+2) + j; }

int main(int argc, char** argv) {
    const int Nx = argc>1 ? std::atoi(argv[1]) : 400;
    const int Ny = argc>2 ? std::atoi(argv[2]) : 200;
    const double Lx = 2.0, Ly = 1.0;
    const double dx = Lx/Nx, dy = Ly/Ny;
    const double x_wedge = 0.5;
    const double theta = 15.0 * M_PI / 180.0;
    const double CFL = 0.45;

    const double M1 = 2.0;
    const double rho1 = 1.0, p1 = 1.0/GAMMA;   // normalized so c1 = 1
    const double u1 = M1, v1 = 0.0;
    State inflow = primitive_to_conservative(rho1, u1, v1, p1);

    std::vector<int> wall_j(Nx+2, 0);
    for (int i = 1; i <= Nx; ++i) {
        double x = (i - 0.5) * dx;
        double y_wall = (x > x_wedge) ? std::tan(theta) * (x - x_wedge) : 0.0;
        wall_j[i] = std::min(Ny, (int)std::floor(y_wall / dy));
    }

    std::vector<State> U((Nx+2)*(Ny+2)), U_new((Nx+2)*(Ny+2));
    for (int i = 0; i <= Nx+1; ++i)
        for (int j = 0; j <= Ny+1; ++j)
            U[IDX(i,j,Ny)] = inflow;

    auto apply_bc = [&](std::vector<State>& U) {
        for (int j = 1; j <= Ny; ++j) U[IDX(0,j,Ny)] = inflow;
        for (int j = 1; j <= Ny; ++j) U[IDX(Nx+1,j,Ny)] = U[IDX(Nx,j,Ny)];
        for (int i = 1; i <= Nx; ++i) U[IDX(i,Ny+1,Ny)] = U[IDX(i,Ny,Ny)];
        for (int i = 1; i <= Nx; ++i) {
            int jw = wall_j[i];
            double x = (i - 0.5) * dx;
            double local_angle = (x > x_wedge) ? theta : 0.0;
            State mirror = reflect_wall(U[IDX(i,jw+1,Ny)], local_angle);
            for (int j = 0; j <= jw; ++j) U[IDX(i,j,Ny)] = mirror;
        }
    };

    double t = 0.0;
    long step = 0;
    const long max_steps = 300000;
    const double residual_tol = 1e-9;
    auto t_start = std::chrono::high_resolution_clock::now();

    while (step < max_steps) {
        apply_bc(U);

        double max_rate = 0.0;
        for (int i = 1; i <= Nx; ++i) {
            int jw = wall_j[i];
            for (int j = jw+1; j <= Ny; ++j) {
                const State& s = U[IDX(i,j,Ny)];
                double c = sound_speed(s);
                max_rate = std::max(max_rate, (std::fabs(u_vel(s))+c)/dx + (std::fabs(v_vel(s))+c)/dy);
            }
        }
        double dt = CFL / max_rate;

        double max_residual = 0.0;
        for (int i = 1; i <= Nx; ++i) {
            int jw = wall_j[i];
            for (int j = jw+1; j <= Ny; ++j) {
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
                max_residual = std::max(max_residual, std::fabs(sn.rho - s.rho));
            }
        }
        for (int i = 1; i <= Nx; ++i) {
            int jw = wall_j[i];
            for (int j = jw+1; j <= Ny; ++j) U[IDX(i,j,Ny)] = U_new[IDX(i,j,Ny)];
        }

        t += dt; ++step;
        if (step % 100 == 0 && max_residual < residual_tol) {
            printf("Converged at step %ld, t=%f, max_residual=%.3e\n", step, t, max_residual);
            break;
        }
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("2D wedge (first-order, Nx=%d Ny=%d): %ld steps, final t=%f, wall time=%.2f ms\n", Nx, Ny, step, t, elapsed_ms);

    std::ofstream out("wedge_result_2d.csv");
    out << "x,y,rho,u,v,p\n";
    for (int i = 1; i <= Nx; ++i) {
        int jw = wall_j[i];
        double x = (i-0.5)*dx;
        for (int j = jw+1; j <= Ny; ++j) {
            double y = (j-0.5)*dy;
            const State& s = U[IDX(i,j,Ny)];
            out << x << "," << y << "," << s.rho << "," << u_vel(s) << "," << v_vel(s) << "," << pressure(s) << "\n";
        }
    }
    printf("Wrote wedge_result_2d.csv\n");
    return 0;
}