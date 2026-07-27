# Compressible Flow Solver: 1D → 2D, CPU → GPU, with an ML-Augmented Limiter

A from-scratch compressible Euler solver built up in five stages: a validated 1D shock-tube solver (first-order, then 2nd-order MUSCL), a CPU→GPU port of each, a 2D extension to supersonic flow over a wedge (oblique shock), a GPU port of that, and a neural-network-augmented slope limiter benchmarked and stress-tested against the standard TVD limiter it replaces.

Every result below was validated against either an exact analytical solution (Riemann solver, θ-β-M oblique shock relation) or a formal order-of-accuracy convergence study - nothing here is "it looks right," everything is checked against a number.

## Repository structure

```
cpu/           1D and 2D CPU solvers (C++)
gpu/           CUDA ports of the same solvers
validation/    Exact-solution comparisons, order-of-accuracy tests, plotting
benchmarks/    CPU vs GPU timing sweeps
ml/            Training data generation, network training, C++ weight export
notebooks/     End-to-end reproducible notebook (Colab/Kaggle GPU runtime)
```

## Results at a glance

| Task                                    | Metric | Result |
|-----------------------------------------|---|---|
| 1 - 1D first-order                      | L2 error vs exact (ρ, u, p) | 5.28e-3, 4.73e-3, 1.43e-3 |
| 1 - CPU vs GPU                          | max \|Δρ\| | 0.000e+00 (bit-exact) |
| 2 - MUSCL+RK2                           | Empirical order | 1.615 (vs 0.872 first-order) |
| 3 - 2D wedge                            | Shock angle error at Nx=320 | +0.099° (theory: β=45.344°) |
| 4 - 2D GPU vs CPU                       | max \|Δρ\|, \|Δp\| | 0.0, 1e-5 (43,602 cells) |
| 4 - GPU speedup                         | Nx=400,Ny=200 | 15.7x |
| 5 - ML limiter, Sod shock tube          | TV(ρ) excess over exact | 0.27% (minmod: 0.003%) |
| 5 - ML limiter, Sod shock tube          | L2 error vs minmod (ρ) | 28% lower |
| 5 - ML limiter, smooth flow             | Empirical order | 2.125 (vs minmod 1.615) |
| 5 - ML limiter, unseen shape (Gaussian) | Empirical order | 1.923 (vs minmod 1.390) |

## Task 1 - 1D Sod shock tube, first-order (CPU + GPU)

**Method.** Rusanov (local Lax-Friedrichs) flux, forward Euler in time, N=20000, CFL=0.45. Rusanov was chosen over sharper Riemann solvers (Roe, HLLC) deliberately: it needs no eigenvalue decomposition or entropy fix, so it's close to impossible to get subtly wrong, at the cost of being more diffusive - the right trade-off for a first correctness baseline.

**Validation.** Compared against an exact Riemann solver (Toro's iterative star-region solution) at t=0.2: L2 error ρ=5.27856e-03, u=4.72632e-03, p=1.43151e-03.

**GPU port.** Genuine per-cell parallelism (each cell's update depends only on its own two neighboring fluxes) plus a `thrust::transform_reduce` max-reduction for the CFL timestep (max is exact regardless of reduction order, unlike a sum). Result: **bit-for-bit identical output to the CPU version** (`max|Δρ| = 0.000e+00`) after ~19,500 steps - a strong correctness signal, since any per-step floating-point divergence would very likely have amplified over that many iterations of a nonlinear, shock-containing solution rather than cancel to exactly zero.

## Task 2 - MUSCL + SSP-RK2 (2nd-order)

**Method.** Linear-in-cell reconstruction with a minmod slope limiter, fed into the same Rusanov flux; SSP-RK2 (Heun's method) in time to match the 2nd-order spatial reconstruction (a 1st-order time integrator would otherwise mask the accuracy gain).

**Validation vs Sod (N=20000).** L2 error ρ=2.27129e-03, u=3.10181e-03, p=6.76008e-04 - 2.3x/1.5x/2.1x lower than first-order at the same resolution.

**Formal order-of-accuracy.** Measured on a smooth periodic density pulse advecting at constant velocity/pressure (an exact solution of the full nonlinear Euler system, chosen specifically because Sod's shock/contact/rarefaction-fan kinks contaminate a direct convergence measurement). Result: **empirical order 1.615**, not the nominal 2.0 - this is the well-documented Osher–Chakravarthy accuracy barrier: minmod (like all standard TVD limiters) provably drops to 1st order at smooth extrema, because a smooth peak or trough is locally indistinguishable from a discontinuity to a limiter that only looks at the sign of two neighboring differences. This limitation is the direct motivation for Task 5.

## Task 3 - 2D supersonic flow over a wedge (oblique shock)

**Ground truth.** The θ-β-M relation for a straight oblique shock, solved via bisection for the weak-shock branch (the physically realized branch for external flow over a wedge). For M1=2, θ=15°, γ=1.4: **β = 45.344°** (θ_max = 22.974° at β=64.669°, confirming the shock stays attached).

**Method.** Genuinely unsplit 2D Rusanov flux (x and y), forward Euler, Mach-2 inflow over a 15° wedge staircase-approximated on a Cartesian grid. The one design choice that matters: wall reflection uses the **true** wedge angle (15°), not the local jagged staircase-step angle - reflecting about the local step angle generates spurious shocks off the staircase artifacts themselves, which was a real bug caught and fixed during development.

**Validation via grid refinement** - detecting the shock's numerical position near the wedge corner and fitting its angle:

| Nx | Detected β | Error |
|---|---|---|
| 80 | 45.000° | −0.344° |
| 160 | 45.629° | +0.285° |
| 320 | 45.442° | +0.099° |

Error shrinking with resolution is the actual proof the solver is solving the right physics - a wrong implementation can still render a plausible-looking diagonal shock.

## Task 4 - GPU port of the 2D solver

**Method.** Genuine 2D CUDA grid (`dim3` threads over i and j, one thread per cell), CFL reduction over the whole array (safe because every ghost cell holds a copied or reflected value that can't overstate the true interior max wave speed), and a dedicated residual kernel for the steady-state check (solid/ghost cells can't reuse the CFL trick since they aren't evolved by the update kernel).

**Correctness.** `max|Δρ| = 0.0`, `max|Δp| = 1e-5` across 43,602 matched cells. The 1e-5 discrepancy was tracked to a specific handful of cells at the exact overlap of the outflow edge and a separately-documented secondary compression wave artifact (from the top boundary's zero-gradient condition not being truly non-reflecting) - FMA contraction and sin/cos precision were both explicitly tested and ruled out as causes (`-fmad=false` gave an identical diff; CPU/GPU trig values were bit-identical). Accepted as negligible floating-point noise unrelated to the validated core physics.

**Benchmark sweep** (CPU vs GPU wall time, converged steady-state solve):

| Nx | Ny | Cells | CPU | GPU | Speedup |
|---|---|---|---|---|---|
| 80 | 40 | 3,200 | 883 ms | 602 ms | 1.47x |
| 160 | 80 | 12,800 | 5,013 ms | 1,069 ms | 4.69x |
| 320 | 160 | 51,200 | 32,576 ms | 2,537 ms | 12.84x |
| 400 | 200 | 80,000 | 60,248 ms | 3,847 ms | 15.66x |

Speedup grows with problem size because GPU kernel-launch overhead is fixed and only pays off once there's enough parallel work to amortize it across.

## Task 5 - ML-augmented troubled-cell indicator

**Motivation.** Task 2 established a specific, textbook weakness: minmod's binary troubled/smooth decision is a crude local sign-check that is provably wrong at smooth extrema. Task 5 asks whether a small learned classifier can make that same decision better, without changing anything else about the scheme.

**Architecture and training.** A 4→16→1 MLP (ReLU hidden layer, sigmoid output), hand-implemented forward/backward pass in plain NumPy - no ML framework - trained with Adam and class-weighted binary cross-entropy (troubled cells are ~2.8% of the data). Inputs are the same four relative density/pressure differences minmod itself looks at. Deployed via a hand-exported C++ header (`ml/troubled_cell_weights.h`) with a ~15-line forward pass - zero runtime ML dependency, deterministic, negligible inference cost.

Trained on three categories: exact Riemann-problem discontinuities (300 random problems, cells labeled troubled if the exact solution's relative jump exceeds 2%), random smooth sine waves (all cells labeled smooth by construction, including at true local extrema - the specific case minmod gets wrong), and diffused two-plateau jumps at varying widths (0.5–12 grid cells, added to represent a numerically-smeared contact discontinuity, which appears in neither of the first two categories).

**Robustness check - a genuine failure, found and fixed.** The first version replaced minmod, when the network said "smooth," with a fully unlimited central-difference slope. Total-variation analysis on the actual Sod shock tube (N=20000) revealed a real TVD violation: `TV(ρ)` grew 9.17% above the exact value of 0.875, versus minmod's own 0.003% (essentially floating-point noise). Critically, this was **invisible to a naive bounds check** (density/pressure staying within their initial [0.125,1.0]/[0.1,1.0] range) - the oscillation was a dip and overshoot entirely between two interior plateau values, not past the domain extremes.

Retraining with wider smeared-jump examples barely helped (9.17% → 9.01%): the real contact discontinuity turned out to be smeared over ~81 grid cells at this resolution, far past the 12-cell training width, and more fundamentally, an 81-cell-wide ramp has a per-cell relative jump (~0.2%) well below the 2% labeling threshold - the actual failure was concentrated at a few cells of asymmetric curvature at the ramp's edges, a case a magnitude-threshold label was never built to catch.

**The fix.** Rather than trying to make the classifier correct at every such edge case, the "smooth" fallback was changed to the MC (monotonized-central, Van Leer 1977) limiter instead of a fully unlimited slope - a different, standard member of the same classical TVD limiter family as minmod, just less restrictive. This does **not** constitute a formal proof that the resulting spatially-adaptive hybrid is globally TVD (the classical Sweby-region proofs assume one consistent limiter choice across the domain, not a per-cell switch driven by an external classifier) - but it bounds what a wrong classification can do to something both candidate limiters already individually keep close to monotone. Empirically, this dropped the TV violation to background-noise level:

| | TV(ρ) excess | TV(p) excess | L2 vs exact (ρ, u, p) |
|---|---|---|---|
| minmod | +0.003% | +0.018% | 2.271e-3, 3.102e-3, 6.760e-4 |
| NN + MC-safe | +0.269% | +0.147% | 1.637e-3, 3.082e-3, 6.748e-4 |

A 33x reduction in the TV violation (9.01% → 0.269%), and now a clean accuracy win across all three fields on the real shock tube, not just a smooth test.

**Smooth-flow accuracy** (same test as Task 2): empirical order 2.125 vs minmod's 1.615, error reduction factor growing from 2.2x to 8.1x across N=50→800.

**Generalization check.** The smooth-flow test problem and the network's "smooth" training category are both sine waves - a legitimate overfitting concern. Retested on a periodic Gaussian bump, a shape absent from training entirely: empirical order 1.923 vs minmod's 1.390, error reduction 1.6x–6.6x. Smaller margin than the sine result, as expected for an unfamiliar shape, but clearly nonzero - evidence against pure memorization, not proof of universal generalization.

## How to reproduce

The full pipeline (all five tasks, in order) is in `notebooks/euler_project_notebook.ipynb`, designed to run top to bottom in a single Colab/Kaggle GPU runtime session. Each `%%writefile` cell writes a source file to the ephemeral session disk; later cells depend on files and Python-session variables defined earlier, so **the notebook must be run in order in a fresh runtime** - restarting partway through and resuming from a later section will fail (missing files, undefined functions) without visibly explaining why.

To build and run any individual solver from the repo directly:

```bash
g++ -O3 -o euler_cpu_muscl cpu/euler_cpu_muscl.cpp && ./euler_cpu_muscl
nvcc -O3 -arch=sm_75 -o euler2d_wedge_gpu gpu/euler2d_wedge_gpu.cu && ./euler2d_wedge_gpu 320 160
```

The ML component needs `ml/troubled_cell_weights.h` generated first (`python3 ml/generate_training_data.py && python3 ml/train_troubled_cell_nn.py && python3 ml/export_weights_to_header.py`) before `cpu/euler_cpu_muscl_nn.cpp` will compile.

## Limitations and future work

- The staircase wall approximation and fully explicit time-stepping are appropriate for a first-order/early-stage solver, not for a production high-fidelity code - a body-fitted or immersed-boundary mesh and implicit or multi-stage time integration would be needed for stiffer, higher-Mach regimes.
- The ML training data is synthetic and 1D (random Riemann problems, sine waves, tanh jumps); it has not been tested on real multi-wave interactions, strong shocks at high Mach number, or extended to 2D, where the troubled-cell decision would need to be made per direction.
- The 2% relative-jump labeling threshold is a chosen hyperparameter, not derived from theory, and is directly implicated in why the first retraining attempt failed to fix the robustness issue.
- The MC-limiter fix is empirically validated, not formally proven TVD for the spatially-adaptive switching case - closing that gap analytically is the natural next step for anyone wanting to trust this beyond the tested cases.
- Rusanov flux is deliberately diffusive; a sharper Riemann solver (Roe, HLLC) would improve baseline accuracy at the cost of implementation complexity and robustness margin.

## References

- Toro, E. F., *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer.
- Van Leer, B., "Towards the Ultimate Conservative Difference Scheme. IV. A New Approach to Numerical Convection," *J. Comput. Phys.*, 1977 (MC limiter).
- Osher, S. and Chakravarthy, S., "High Resolution Schemes and the Entropy Condition," *SIAM J. Numer. Anal.*, 1984.
- Sweby, P. K., "High Resolution Schemes Using Flux Limiters for Hyperbolic Conservation Laws," *SIAM J. Numer. Anal.*, 1984.
- Ray, D. and Hesthaven, J. S., "An Artificial Neural Network as a Troubled-Cell Indicator," *J. Comput. Phys.*, 2018.