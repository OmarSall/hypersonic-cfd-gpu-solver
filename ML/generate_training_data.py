# Training data for the ML troubled-cell indicator, v3: combines three sources.
#
# 1. Random Riemann problems (genuine discontinuities): exact solution from
#    exact_riemann.py, cells labeled troubled=1 if the exact VALUE jumps
#    meaningfully across that cell, 0 elsewhere.
# 2. Random smooth periodic functions (NO discontinuities anywhere): every
#    cell labeled troubled=0 by construction, including cells sitting right
#    at a smooth local max/min -- this is the specific scenario minmod gets
#    wrong (see smooth_advection_accuracy_test.cpp / the Osher-Chakravarthy
#    accuracy-barrier discussion), and v1 of this dataset (Riemann problems
#    only) never actually contained it, since an exact Riemann solution has
#    no true local extrema at all -- it's monotonic between waves.
# 3. Smeared jumps (v3, added after the Sod robustness check found a real
#    TVD violation at the contact discontinuity in euler_cpu_muscl_nn.cpp:
#    TV(rho) grew 9.2% over the initial value, vs minmod's near-exact match).
#    v2's Riemann examples are ALWAYS perfectly sharp single-cell jumps
#    (exact solution has zero numerical diffusion), and v2's smooth examples
#    are always zero-gradient-consistent sine waves -- neither one looks
#    like what a real contact discontinuity looks like a few hundred steps
#    into a MUSCL/RK2 run: a jump that's been smeared by numerical diffusion
#    over a handful of cells. That exact regime was entirely absent from
#    training, and is where the network failed. Fixed here by adding tanh
#    profiles at a range of widths (0.5dx, nearly sharp, up to 12dx, heavily
#    diffused) so the network sees the full continuum between "sharp shock"
#    and "smooth function" instead of only the two extremes.
import numpy as np
import sys
sys.path.insert(0, ".")
from exact_riemann import exact_sod_solution

GAMMA = 1.4
np.random.seed(42)

def riemann_examples(N=200, n_problems=300):
    X_list, y_list = [], []
    n_ok = 0
    for _ in range(n_problems):
        rhoL = np.random.uniform(0.2, 2.0); rhoR = np.random.uniform(0.2, 2.0)
        uL   = np.random.uniform(-0.5, 0.5); uR   = np.random.uniform(-0.5, 0.5)
        pL   = np.random.uniform(0.2, 2.0);  pR   = np.random.uniform(0.2, 2.0)
        t    = np.random.uniform(0.05, 0.3)
        dx = 1.0 / N
        x_centers = (np.arange(N) + 0.5) * dx
        try:
            rho_c, u_c, p_c = exact_sod_solution(x_centers, t, x0=0.5, rhoL=rhoL, uL=uL, pL=pL,
                                                  rhoR=rhoR, uR=uR, pR=pR, gamma=GAMMA)
            x_edges = np.arange(N+1) * dx
            rho_e, u_e, p_e = exact_sod_solution(x_edges, t, x0=0.5, rhoL=rhoL, uL=uL, pL=pL,
                                                  rhoR=rhoR, uR=uR, pR=pR, gamma=GAMMA)
        except Exception:
            continue
        if np.any(np.isnan(rho_c)) or np.any(rho_c <= 0) or np.any(p_c <= 0):
            continue
        n_ok += 1
        rel_drho = np.abs(rho_e[1:] - rho_e[:-1]) / (0.5*(rho_e[1:]+rho_e[:-1]) + 1e-8)
        rel_dp   = np.abs(p_e[1:]   - p_e[:-1])   / (0.5*(p_e[1:]+p_e[:-1])     + 1e-8)
        troubled = ((rel_drho > 0.02) | (rel_dp > 0.02)).astype(np.float64)
        for i in range(1, N-1):
            d_rho_l = (rho_c[i]   - rho_c[i-1]) / (abs(rho_c[i]) + 1e-8)
            d_rho_r = (rho_c[i+1] - rho_c[i])   / (abs(rho_c[i]) + 1e-8)
            d_p_l   = (p_c[i]     - p_c[i-1])   / (abs(p_c[i]) + 1e-8)
            d_p_r   = (p_c[i+1]   - p_c[i])     / (abs(p_c[i]) + 1e-8)
            X_list.append([d_rho_l, d_rho_r, d_p_l, d_p_r])
            y_list.append(troubled[i])
    print(f"Riemann problems ok: {n_ok}/{n_problems}, examples: {len(y_list)}")
    return X_list, y_list

def smooth_examples(N=200, n_problems=300):
    # random smooth periodic density/pressure fields, no discontinuities
    # anywhere -- every cell is genuinely smooth (troubled=0), INCLUDING
    # cells sitting exactly at a local max or min.
    X_list, y_list = [], []
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    for _ in range(n_problems):
        A_rho = np.random.uniform(0.5, 1.5); B_rho = np.random.uniform(0.1, 0.6)
        k_rho = np.random.uniform(1, 4); phase_rho = np.random.uniform(0, 2*np.pi)
        A_p   = np.random.uniform(0.5, 1.5); B_p   = np.random.uniform(0.05, 0.3)
        k_p   = np.random.uniform(1, 4); phase_p   = np.random.uniform(0, 2*np.pi)

        rho_c = A_rho + B_rho*np.sin(2*np.pi*k_rho*x + phase_rho)
        p_c   = A_p   + B_p  *np.sin(2*np.pi*k_p*x   + phase_p)
        if np.any(rho_c <= 0) or np.any(p_c <= 0):
            continue
        for i in range(1, N-1):
            d_rho_l = (rho_c[i]   - rho_c[i-1]) / (abs(rho_c[i]) + 1e-8)
            d_rho_r = (rho_c[i+1] - rho_c[i])   / (abs(rho_c[i]) + 1e-8)
            d_p_l   = (p_c[i]     - p_c[i-1])   / (abs(p_c[i]) + 1e-8)
            d_p_r   = (p_c[i+1]   - p_c[i])     / (abs(p_c[i]) + 1e-8)
            X_list.append([d_rho_l, d_rho_r, d_p_l, d_p_r])
            y_list.append(0.0)   # smooth by construction, always
    print(f"smooth-function examples: {len(y_list)}")
    return X_list, y_list

def smeared_jump_examples(N=200, n_problems=400):
    # Diffused two-plateau jumps: rho (and sometimes p) transition smoothly
    # from a left plateau to a right plateau via a tanh profile of width w.
    # w=0.5*dx looks almost like a sharp Riemann jump; w=12*dx looks almost
    # smooth. This is the regime a real contact discontinuity sits in after
    # a MUSCL/RK2 solver has numerically diffused it over several cells --
    # entirely missing from v2's training data (see header comment above).
    X_list, y_list = [], []
    dx = 1.0 / N
    x_centers = (np.arange(N) + 0.5) * dx
    x_edges = np.arange(N+1) * dx
    widths_dx = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    for _ in range(n_problems):
        x0 = np.random.uniform(0.3, 0.7)
        w = np.random.choice(widths_dx) * dx
        rhoL = np.random.uniform(0.2, 2.0); rhoR = np.random.uniform(0.2, 2.0)
        pL   = np.random.uniform(0.2, 2.0); pR   = np.random.uniform(0.2, 2.0)
        if np.random.rand() < 0.5:
            pR = pL   # contact-like: only rho jumps, p (and implicitly u) continuous

        def profile(x, L, R):
            return L + (R - L) * 0.5 * (1.0 + np.tanh((x - x0) / w))

        rho_c, p_c = profile(x_centers, rhoL, rhoR), profile(x_centers, pL, pR)
        rho_e, p_e = profile(x_edges, rhoL, rhoR), profile(x_edges, pL, pR)
        if np.any(rho_c <= 0) or np.any(p_c <= 0):
            continue
        rel_drho = np.abs(rho_e[1:] - rho_e[:-1]) / (0.5*(rho_e[1:]+rho_e[:-1]) + 1e-8)
        rel_dp   = np.abs(p_e[1:]   - p_e[:-1])   / (0.5*(p_e[1:]+p_e[:-1])     + 1e-8)
        troubled = ((rel_drho > 0.02) | (rel_dp > 0.02)).astype(np.float64)
        for i in range(1, N-1):
            d_rho_l = (rho_c[i]   - rho_c[i-1]) / (abs(rho_c[i]) + 1e-8)
            d_rho_r = (rho_c[i+1] - rho_c[i])   / (abs(rho_c[i]) + 1e-8)
            d_p_l   = (p_c[i]     - p_c[i-1])   / (abs(p_c[i]) + 1e-8)
            d_p_r   = (p_c[i+1]   - p_c[i])     / (abs(p_c[i]) + 1e-8)
            X_list.append([d_rho_l, d_rho_r, d_p_l, d_p_r])
            y_list.append(troubled[i])
    print(f"smeared-jump examples: {len(y_list)}")
    return X_list, y_list

if __name__ == "__main__":
    X1, y1 = riemann_examples(N=200, n_problems=300)
    X2, y2 = smooth_examples(N=200, n_problems=300)
    X3, y3 = smeared_jump_examples(N=200, n_problems=400)
    X = np.array(X1 + X2 + X3)
    y = np.array(y1 + y2 + y3)
    source = np.array([0]*len(y1) + [1]*len(y2) + [2]*len(y3))  # 0=riemann, 1=smooth, 2=smeared_jump
    print("combined feature matrix:", X.shape, "fraction troubled:", y.mean())
    np.savez("troubled_cell_data.npz", X=X, y=y, source=source)
    print("saved troubled_cell_data.npz")