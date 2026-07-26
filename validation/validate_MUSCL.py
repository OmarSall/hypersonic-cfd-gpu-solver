num_m = pd.read_csv("sod_result_cpu_muscl.csv")
t_final = 0.20

rho_ex_m, u_ex_m, p_ex_m = exact_sod_solution(num_m["x"].values, t_final)

l2_rho_m = np.sqrt(np.mean((num_m["rho"].values - rho_ex_m) ** 2))
l2_u_m   = np.sqrt(np.mean((num_m["u"].values   - u_ex_m)   ** 2))
l2_p_m   = np.sqrt(np.mean((num_m["p"].values   - p_ex_m)   ** 2))
print(f"MUSCL+RK2 L2 error vs exact:  rho={l2_rho_m:.5e}   u={l2_u_m:.5e}   p={l2_p_m:.5e}")
print(f"Improvement over first-order: rho {l2_rho/l2_rho_m:.2f}x, u {l2_u/l2_u_m:.2f}x, p {l2_p/l2_p_m:.2f}x lower error")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, ex, label in zip(axes, ["rho", "u", "p"], [rho_ex_m, u_ex_m, p_ex_m], ["Density", "Velocity", "Pressure"]):
    ax.plot(num_m["x"], num_m[col], label="Numerical (MUSCL+RK2)", linewidth=2)
    ax.plot(num_m["x"], ex, "--", label="Exact", linewidth=2)
    ax.set_xlabel("x"); ax.set_ylabel(label); ax.legend()
plt.suptitle("MUSCL+RK2 scheme vs exact solution")
plt.tight_layout()
plt.show()
