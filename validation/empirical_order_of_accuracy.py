import subprocess

result = subprocess.run(["./smooth_test"], capture_output=True, text=True)
lines = result.stdout.strip().split("\n")

Ns, l2_1st, l2_muscl_list = [], [], []
for line in lines[1:]:
    parts = line.split("\t")
    if len(parts) == 3 and parts[0].isdigit():
        Ns.append(int(parts[0]))
        l2_1st.append(float(parts[1]))
        l2_muscl_list.append(float(parts[2]))

Ns = np.array(Ns); l2_1st = np.array(l2_1st); l2_muscl_list = np.array(l2_muscl_list)
order_1st = np.polyfit(np.log(1.0 / Ns), np.log(l2_1st), 1)[0]
order_muscl = np.polyfit(np.log(1.0 / Ns), np.log(l2_muscl_list), 1)[0]

print(f"Empirical order (first-order Rusanov):  {order_1st:.3f}  (expected -> ~1.0)")
print(f"Empirical order (MUSCL + SSP-RK2):      {order_muscl:.3f}  (expected -> ~1.6-1.7, not 2.0 -- Osher-Chakravarthy accuracy barrier at smooth extrema)")

plt.figure(figsize=(6,5))
plt.loglog(1.0/Ns, l2_1st, 'o-', label=f"1st order (slope={order_1st:.2f})")
plt.loglog(1.0/Ns, l2_muscl_list, 's-', label=f"MUSCL+RK2 (slope={order_muscl:.2f})")
plt.xlabel("dx"); plt.ylabel("L2 error"); plt.legend(); plt.title("Order-of-accuracy convergence")
plt.grid(True, which="both", alpha=0.3)
plt.show()
