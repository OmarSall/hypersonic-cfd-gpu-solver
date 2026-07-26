# Visualize the 2D wedge solver's output (pressure and Mach number fields)
# and overlay the theoretical oblique shock line + the numerically detected
# shock points, as a visual cross-check against oblique_shock_relation.py.
#
# Two plotting bugs were caught and fixed while building this (see project
# history): (1) an erroneous "+0.5" y-offset that displaced the theoretical
# shock line so it no longer started at the wedge corner, and (2) clipping
# the line's y-values to the domain height without recomputing the matching
# x-endpoint, which silently changed the line's apparent slope. The fixed
# version below computes the exact (x, y) where the theoretical shock exits
# the domain and stops the line there.
#
# Known (real, not a bug) artifact visible in the plot: a secondary, weaker
# compression wave radiates from the point where the primary shock crosses
# the top boundary (~x=1.488 for the default domain). This comes from the
# zero-gradient outflow BC not being a true non-reflecting condition -- it
# does not contaminate the shock-angle validation, which only samples data
# near the wall, well below where this secondary wave reaches.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

CSV_PATH = "wedge_result_2d.csv"
NX = 320          # must match the Nx used to generate CSV_PATH
LX, X_WEDGE = 2.0, 0.5
BETA_DEG = 45.344  # from oblique_shock_relation.py for M1=2, theta=15deg

df = pd.read_csv(CSV_PATH)
gamma = 1.4
mach = np.sqrt(df["u"]**2 + df["v"]**2) / np.sqrt(gamma*df["p"]/df["rho"])

# --- detect shock points: last point (ascending y) still above the
# midpoint between freestream and the column's local max pressure ---
shock_pts = []
for x, g in df.groupby("x"):
    g = g.sort_values("y")
    p, y = g["p"].values, g["y"].values
    if p.max() < 0.85:
        continue
    mid = 0.5*(0.714286 + p.max())
    above = p > mid
    if not above.any() or above.all():
        continue
    idx = np.where(above)[0][-1]
    shock_pts.append((x, 0.5*(y[idx]+y[idx+1])))
shock_pts = np.array(shock_pts)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

cell_size = LX / NX
triang = tri.Triangulation(df["x"], df["y"])
mask = np.zeros(triang.triangles.shape[0], dtype=bool)
xs, ys = df["x"].values, df["y"].values
for k, tr in enumerate(triang.triangles):
    xk, yk = xs[tr], ys[tr]
    if xk.max()-xk.min() > 2*cell_size or yk.max()-yk.min() > 2*cell_size:
        mask[k] = True
triang.set_mask(mask)

beta = np.radians(BETA_DEG)
x_top = X_WEDGE + 1.0/np.tan(beta)   # exact point where the theoretical line exits at y=1

for ax, field, label, cmap in [(axes[0], df["p"], "Pressure", "viridis"),
                                 (axes[1], mach, "Mach number", "plasma")]:
    tp = ax.tricontourf(triang, field, levels=30, cmap=cmap)
    fig.colorbar(tp, ax=ax, label=label, shrink=0.85)
    ax.plot([X_WEDGE, x_top], [0.0, 1.0], "w--", linewidth=1.3, label="theoretical shock")
    ax.plot(shock_pts[:,0], shock_pts[:,1], "r.", markersize=3, label="detected shock points")
    xw = np.linspace(X_WEDGE, LX, 50)
    yw = np.tan(np.radians(15)) * (xw - X_WEDGE)
    ax.fill_between(xw, 0, yw, color="0.3", zorder=5)
    ax.set_xlim(0, LX); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{label} field")
    ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.show()