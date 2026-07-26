// ============================================================================
// Order-of-accuracy test, extended with a third scheme: MUSCL+RK2 using the
// trained troubled-cell neural network (troubled_cell_weights.h) instead of
// minmod to decide, per cell, whether to apply limiting (troubled -> minmod,
// same as before) or the raw unlimited central-difference slope (smooth ->
// no limiting at all). Same smooth density-pulse-at-constant-velocity setup
// as smooth_advection_accuracy_test.cpp -- see that file's header comment
// for why this test (not Sod) is the right way to measure formal order of
// accuracy. First-order and minmod-MUSCL columns are kept unchanged from
// the original file so all three numbers come from one run, directly
// comparable to the already-recorded baseline (first-order order ~0.872,
// minmod-MUSCL order ~1.615).
// ============================================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "troubled_cell_weights.h"

const double GAMMA = 1.4;
struct State { double rho, mom, E; };

double velocity(const State& s) { return s.mom / s.rho; }
double pressure(const State& s) { double u=velocity(s); return (GAMMA-1.0)*(s.E-0.5*s.rho*u*u); }
double sound_speed(const State& s) { return std::sqrt(GAMMA*pressure(s)/s.rho); }
State physical_flux(const State& s) { double u=velocity(s),p=pressure(s); return State{s.rho*u,s.rho*u*u+p,u*(s.E+p)}; }
State rusanov_flux(const State& UL, const State& UR) {
    State FL=physical_flux(UL), FR=physical_flux(UR);
    double Smax = std::max(std::fabs(velocity(UL))+sound_speed(UL), std::fabs(velocity(UR))+sound_speed(UR));
    return State{ 0.5*(FL.rho+FR.rho)-0.5*Smax*(UR.rho-UL.rho),
                  0.5*(FL.mom+FR.mom)-0.5*Smax*(UR.mom-UL.mom),
                  0.5*(FL.E+FR.E)    -0.5*Smax*(UR.E-UL.E) };
}
double minmod(double a,double b){ if(a*b<=0.0) return 0.0; return (a>0.0)?std::min(a,b):std::max(a,b); }
double minmod3(double x,double y,double z){
    if (x>0.0 && y>0.0 && z>0.0) return std::min({x,y,z});
    if (x<0.0 && y<0.0 && z<0.0) return std::max({x,y,z});
    return 0.0;
}
// MC (monotonized-central) limiter, TVD by construction -- see
// euler_cpu_muscl_nn.cpp for the full explanation of why this replaced the
// fully unlimited central difference in the "smooth" branch below.
double mc_limiter(double a, double b){ return minmod3(2.0*a, 2.0*b, 0.5*(a+b)); }

State ic(double x, double u0, double p0) {
    double rho = 1.0 + 0.5*std::sin(2*M_PI*x);
    return State{ rho, rho*u0, p0/(GAMMA-1.0) + 0.5*rho*u0*u0 };
}

// ---- First order: Rusanov + forward Euler, 1 ghost cell each side ----
double run_first_order(int N) {
    double u0=1.0, p0=1.0, dx=1.0/N, CFL=0.45, t_final=1.0;
    std::vector<State> U(N+2), U0(N+2);
    for (int i=0;i<N+2;++i){ double x=(i-0.5)*dx; U[i]=ic(x,u0,p0); U0[i]=U[i]; }
    double t=0.0;
    std::vector<State> F(N+1), U_new(N+2);
    while (t<t_final) {
        U[0]=U[N]; U[N+1]=U[1];
        double Smax=0.0;
        for (int i=1;i<=N;++i) Smax=std::max(Smax, std::fabs(velocity(U[i]))+sound_speed(U[i]));
        double dt=CFL*dx/Smax; if (t+dt>t_final) dt=t_final-t;
        for (int i=0;i<=N;++i) F[i]=rusanov_flux(U[i],U[i+1]);
        for (int i=1;i<=N;++i) {
            U_new[i].rho=U[i].rho-(dt/dx)*(F[i].rho-F[i-1].rho);
            U_new[i].mom=U[i].mom-(dt/dx)*(F[i].mom-F[i-1].mom);
            U_new[i].E  =U[i].E  -(dt/dx)*(F[i].E  -F[i-1].E);
        }
        for (int i=1;i<=N;++i) U[i]=U_new[i];
        t+=dt;
    }
    double err2=0.0;
    for (int i=1;i<=N;++i) err2 += (U[i].rho-U0[i].rho)*(U[i].rho-U0[i].rho);
    return std::sqrt(err2/N);
}

// ---- MUSCL + SSP-RK2 with minmod (unchanged baseline), 2 ghost cells each side ----
void compute_L(std::vector<State>& U,int N,double dx,std::vector<State>& slope,std::vector<State>& F,std::vector<State>& L){
    for (int i=1;i<=N+2;++i){
        slope[i].rho=minmod(U[i].rho-U[i-1].rho,U[i+1].rho-U[i].rho);
        slope[i].mom=minmod(U[i].mom-U[i-1].mom,U[i+1].mom-U[i].mom);
        slope[i].E  =minmod(U[i].E  -U[i-1].E,  U[i+1].E  -U[i].E);
    }
    for (int i=1;i<=N+1;++i){
        State UL{U[i].rho+0.5*slope[i].rho,U[i].mom+0.5*slope[i].mom,U[i].E+0.5*slope[i].E};
        State UR{U[i+1].rho-0.5*slope[i+1].rho,U[i+1].mom-0.5*slope[i+1].mom,U[i+1].E-0.5*slope[i+1].E};
        F[i]=rusanov_flux(UL,UR);
    }
    for (int i=2;i<=N+1;++i){
        L[i].rho=-(F[i].rho-F[i-1].rho)/dx;
        L[i].mom=-(F[i].mom-F[i-1].mom)/dx;
        L[i].E  =-(F[i].E  -F[i-1].E)  /dx;
    }
}
double run_muscl_rk2(int N) {
    double u0=1.0, p0=1.0, dx=1.0/N, CFL=0.45, t_final=1.0;
    std::vector<State> U(N+4), U0(N+4);
    for (int i=0;i<N+4;++i){ double x=(i-2.0+0.5)*dx; U[i]=ic(x,u0,p0); U0[i]=U[i]; }
    std::vector<State> slope(N+4), F(N+2), L1(N+4), L2(N+4), U_star(N+4);
    double t=0.0;
    while (t<t_final) {
        U[1]=U[N+1]; U[0]=U[N];  U[N+2]=U[2]; U[N+3]=U[3];
        double Smax=0.0;
        for (int i=2;i<=N+1;++i) Smax=std::max(Smax, std::fabs(velocity(U[i]))+sound_speed(U[i]));
        double dt=CFL*dx/Smax; if (t+dt>t_final) dt=t_final-t;

        compute_L(U,N,dx,slope,F,L1);
        U_star=U;
        for (int i=2;i<=N+1;++i){ U_star[i].rho=U[i].rho+dt*L1[i].rho; U_star[i].mom=U[i].mom+dt*L1[i].mom; U_star[i].E=U[i].E+dt*L1[i].E; }

        U_star[1]=U_star[N+1]; U_star[0]=U_star[N]; U_star[N+2]=U_star[2]; U_star[N+3]=U_star[3];
        compute_L(U_star,N,dx,slope,F,L2);
        for (int i=2;i<=N+1;++i){
            double rho2=U_star[i].rho+dt*L2[i].rho, mom2=U_star[i].mom+dt*L2[i].mom, E2=U_star[i].E+dt*L2[i].E;
            U[i].rho=0.5*(U[i].rho+rho2); U[i].mom=0.5*(U[i].mom+mom2); U[i].E=0.5*(U[i].E+E2);
        }
        t+=dt;
    }
    double err2=0.0;
    for (int i=2;i<=N+1;++i) err2 += (U[i].rho-U0[i].rho)*(U[i].rho-U0[i].rho);
    return std::sqrt(err2/N);
}

// ---- MUSCL + SSP-RK2 with the NN troubled-cell indicator instead of
// minmod's own trigger: troubled cells still use minmod (same magnitude
// limiting as before); smooth cells (per the network) use the raw,
// unlimited central-difference slope instead of minmod's limited one. ----
void compute_L_nn(std::vector<State>& U,int N,double dx,std::vector<State>& slope,std::vector<State>& F,std::vector<State>& L){
    for (int i=1;i<=N+2;++i){
        bool troubled = nn_is_troubled(U[i-1].rho, U[i].rho, U[i+1].rho,
                                        pressure(U[i-1]), pressure(U[i]), pressure(U[i+1]));
        if (troubled) {
            slope[i].rho=minmod(U[i].rho-U[i-1].rho,U[i+1].rho-U[i].rho);
            slope[i].mom=minmod(U[i].mom-U[i-1].mom,U[i+1].mom-U[i].mom);
            slope[i].E  =minmod(U[i].E  -U[i-1].E,  U[i+1].E  -U[i].E);
        } else {
            slope[i].rho = mc_limiter(U[i].rho-U[i-1].rho, U[i+1].rho-U[i].rho);
            slope[i].mom = mc_limiter(U[i].mom-U[i-1].mom, U[i+1].mom-U[i].mom);
            slope[i].E   = mc_limiter(U[i].E  -U[i-1].E,   U[i+1].E  -U[i].E);
        }
    }
    for (int i=1;i<=N+1;++i){
        State UL{U[i].rho+0.5*slope[i].rho,U[i].mom+0.5*slope[i].mom,U[i].E+0.5*slope[i].E};
        State UR{U[i+1].rho-0.5*slope[i+1].rho,U[i+1].mom-0.5*slope[i+1].mom,U[i+1].E-0.5*slope[i+1].E};
        F[i]=rusanov_flux(UL,UR);
    }
    for (int i=2;i<=N+1;++i){
        L[i].rho=-(F[i].rho-F[i-1].rho)/dx;
        L[i].mom=-(F[i].mom-F[i-1].mom)/dx;
        L[i].E  =-(F[i].E  -F[i-1].E)  /dx;
    }
}
double run_muscl_rk2_nn(int N) {
    double u0=1.0, p0=1.0, dx=1.0/N, CFL=0.45, t_final=1.0;
    std::vector<State> U(N+4), U0(N+4);
    for (int i=0;i<N+4;++i){ double x=(i-2.0+0.5)*dx; U[i]=ic(x,u0,p0); U0[i]=U[i]; }
    std::vector<State> slope(N+4), F(N+2), L1(N+4), L2(N+4), U_star(N+4);
    double t=0.0;
    while (t<t_final) {
        U[1]=U[N+1]; U[0]=U[N];  U[N+2]=U[2]; U[N+3]=U[3];
        double Smax=0.0;
        for (int i=2;i<=N+1;++i) Smax=std::max(Smax, std::fabs(velocity(U[i]))+sound_speed(U[i]));
        double dt=CFL*dx/Smax; if (t+dt>t_final) dt=t_final-t;

        compute_L_nn(U,N,dx,slope,F,L1);
        U_star=U;
        for (int i=2;i<=N+1;++i){ U_star[i].rho=U[i].rho+dt*L1[i].rho; U_star[i].mom=U[i].mom+dt*L1[i].mom; U_star[i].E=U[i].E+dt*L1[i].E; }

        U_star[1]=U_star[N+1]; U_star[0]=U_star[N]; U_star[N+2]=U_star[2]; U_star[N+3]=U_star[3];
        compute_L_nn(U_star,N,dx,slope,F,L2);
        for (int i=2;i<=N+1;++i){
            double rho2=U_star[i].rho+dt*L2[i].rho, mom2=U_star[i].mom+dt*L2[i].mom, E2=U_star[i].E+dt*L2[i].E;
            U[i].rho=0.5*(U[i].rho+rho2); U[i].mom=0.5*(U[i].mom+mom2); U[i].E=0.5*(U[i].E+E2);
        }
        t+=dt;
    }
    double err2=0.0;
    for (int i=2;i<=N+1;++i) err2 += (U[i].rho-U0[i].rho)*(U[i].rho-U0[i].rho);
    return std::sqrt(err2/N);
}

int main() {
    int Ns[] = {50,100,200,400,800};
    printf("N\tL2_first_order\tL2_MUSCL_minmod\tL2_MUSCL_NN\n");
    for (int N : Ns) printf("%d\t%.6e\t%.6e\t%.6e\n", N, run_first_order(N), run_muscl_rk2(N), run_muscl_rk2_nn(N));
    printf("\nCompute empirical order via: order = polyfit(log(1/N), log(L2), 1)[0]\n");
    printf("Baseline (already established): first-order ~0.872, minmod-MUSCL ~1.615\n");
    return 0;
}