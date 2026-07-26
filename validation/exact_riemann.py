import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GAMMA = 1.4

def exact_sod_solution(x_array, t, x0=0.5,
                        rhoL=1.0, uL=0.0, pL=1.0,
                        rhoR=0.125, uR=0.0, pR=0.1,
                        gamma=GAMMA, tol=1e-10, maxiter=100):
    """
    Exact Riemann solver for the Euler equations (Toro, 'Riemann Solvers and
    Numerical Methods for Fluid Dynamics', Ch. 4), specialized to the Sod
    shock tube. Returns rho, u, p sampled at x_array, at time t.
    """
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    A_L = 2.0 / ((gamma + 1.0) * rhoL)
    B_L = (gamma - 1.0) / (gamma + 1.0) * pL
    A_R = 2.0 / ((gamma + 1.0) * rhoR)
    B_R = (gamma - 1.0) / (gamma + 1.0) * pR

    def f_K(p, p_K, c_K, rho_K, A_K, B_K):
        if p > p_K:
            return (p - p_K) * np.sqrt(A_K / (p + B_K))
        else:
            return (2.0 * c_K / (gamma - 1.0)) * ((p / p_K) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)

    def f_K_prime(p, p_K, c_K, rho_K, A_K, B_K):
        if p > p_K:
            return np.sqrt(A_K / (B_K + p)) * (1.0 - (p - p_K) / (2.0 * (B_K + p)))
        else:
            return (1.0 / (rho_K * c_K)) * (p / p_K) ** (-(gamma + 1.0) / (2.0 * gamma))

    def f(p):
        return f_K(p, pL, cL, rhoL, A_L, B_L) + f_K(p, pR, cR, rhoR, A_R, B_R) + (uR - uL)

    def fprime(p):
        return f_K_prime(p, pL, cL, rhoL, A_L, B_L) + f_K_prime(p, pR, cR, rhoR, A_R, B_R)

    p_star = max(0.5 * (pL + pR), tol)
    for _ in range(maxiter):
        p_new = max(p_star - f(p_star) / fprime(p_star), tol)
        if abs(p_new - p_star) < tol:
            p_star = p_new
            break
        p_star = p_new

    u_star = 0.5 * (uL + uR) + 0.5 * (f_K(p_star, pR, cR, rhoR, A_R, B_R) - f_K(p_star, pL, cL, rhoL, A_L, B_L))

    if p_star > pL:
        rho_star_L = rhoL * ((p_star/pL) + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1)*(p_star/pL) + 1)
    else:
        rho_star_L = rhoL * (p_star/pL) ** (1.0/gamma)

    if p_star > pR:
        rho_star_R = rhoR * ((p_star/pR) + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1)*(p_star/pR) + 1)
    else:
        rho_star_R = rhoR * (p_star/pR) ** (1.0/gamma)

    c_star_L = np.sqrt(gamma * p_star / rho_star_L)
    c_star_R = np.sqrt(gamma * p_star / rho_star_R)

    rho_out = np.zeros_like(x_array, dtype=float)
    u_out = np.zeros_like(x_array, dtype=float)
    p_out = np.zeros_like(x_array, dtype=float)

    for idx, x in enumerate(x_array):
        xi = (x - x0) / t if t > 0 else 0.0
        if xi <= u_star:
            if p_star > pL:
                S_L = uL - cL * np.sqrt((gamma+1)/(2*gamma)*(p_star/pL) + (gamma-1)/(2*gamma))
                rho, u, p = (rhoL, uL, pL) if xi < S_L else (rho_star_L, u_star, p_star)
            else:
                S_HL = uL - cL
                S_TL = u_star - c_star_L
                if xi < S_HL:
                    rho, u, p = rhoL, uL, pL
                elif xi > S_TL:
                    rho, u, p = rho_star_L, u_star, p_star
                else:
                    c_fan = (2.0/(gamma+1)) * (cL + (gamma-1)/2*(uL - xi))
                    rho = rhoL * (c_fan/cL) ** (2.0/(gamma-1))
                    u = (2.0/(gamma+1)) * (cL + (gamma-1)/2*uL + xi)
                    p = pL * (c_fan/cL) ** (2.0*gamma/(gamma-1))
        else:
            if p_star > pR:
                S_R = uR + cR * np.sqrt((gamma+1)/(2*gamma)*(p_star/pR) + (gamma-1)/(2*gamma))
                rho, u, p = (rhoR, uR, pR) if xi > S_R else (rho_star_R, u_star, p_star)
            else:
                S_HR = uR + cR
                S_TR = u_star + c_star_R
                if xi > S_HR:
                    rho, u, p = rhoR, uR, pR
                elif xi < S_TR:
                    rho, u, p = rho_star_R, u_star, p_star
                else:
                    c_fan = (2.0/(gamma+1)) * (cR - (gamma-1)/2*(uR - xi))
                    rho = rhoR * (c_fan/cR) ** (2.0/(gamma-1))
                    u = (2.0/(gamma+1)) * (-cR + (gamma-1)/2*uR + xi)
                    p = pR * (c_fan/cR) ** (2.0*gamma/(gamma-1))
        rho_out[idx], u_out[idx], p_out[idx] = rho, u, p

    return rho_out, u_out, p_out


# ---- Load your numerical result and compare against the exact solution ----
num = pd.read_csv("sod_result_cpu.csv")  # or sod_result_gpu.csv -- both should match
t_final = 0.20

rho_ex, u_ex, p_ex = exact_sod_solution(num["x"].values, t_final)

l2_rho = np.sqrt(np.mean((num["rho"].values - rho_ex) ** 2))
l2_u   = np.sqrt(np.mean((num["u"].values   - u_ex)   ** 2))
l2_p   = np.sqrt(np.mean((num["p"].values   - p_ex)   ** 2))
print(f"L2 error vs exact solution:  rho={l2_rho:.5e}   u={l2_u:.5e}   p={l2_p:.5e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, ex, label in zip(axes, ["rho", "u", "p"], [rho_ex, u_ex, p_ex], ["Density", "Velocity", "Pressure"]):
    ax.plot(num["x"], num[col], label="Numerical (Rusanov)", linewidth=2)
    ax.plot(num["x"], ex, "--", label="Exact", linewidth=2)
    ax.set_xlabel("x"); ax.set_ylabel(label); ax.legend()
plt.tight_layout()
plt.show()