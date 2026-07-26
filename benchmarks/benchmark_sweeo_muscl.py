import subprocess, re
import pandas as pd
import matplotlib.pyplot as plt

Ns = [400, 1000, 2000, 4000, 10000, 20000, 50000]
rows = []

cpu_src = open("euler_cpu_muscl.cpp").read()
gpu_src = open("euler_gpu_muscl.cu").read()

for N in Ns:
    cpu_N = re.sub(r"const int\s+N\s*=\s*\d+;", f"const int N = {N};", cpu_src)
    gpu_N = re.sub(r"const int\s+N\s*=\s*\d+;", f"const int N = {N};", gpu_src)
    open("_sweep_cpu.cpp", "w").write(cpu_N)
    open("_sweep_gpu.cu", "w").write(gpu_N)

    subprocess.run(["g++", "-O3", "-o", "_sweep_cpu", "_sweep_cpu.cpp"], check=True)
    subprocess.run(["nvcc", "-O3", "-arch=sm_75", "-o", "_sweep_gpu", "_sweep_gpu.cu"], check=True)

    cpu_out = subprocess.run(["./_sweep_cpu"], capture_output=True, text=True).stdout
    gpu_out = subprocess.run(["./_sweep_gpu"], capture_output=True, text=True).stdout

    cpu_ms = float(re.search(r"wall time=([\d.]+) ms", cpu_out).group(1))
    gpu_ms = float(re.search(r"wall time = ([\d.]+) ms", gpu_out).group(1))

    rows.append({"N": N, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms, "speedup": cpu_ms / gpu_ms})
    print(f"N={N}: CPU={cpu_ms:.1f}ms  GPU={gpu_ms:.1f}ms  speedup={cpu_ms/gpu_ms:.2f}x")

df = pd.DataFrame(rows)
plt.figure(figsize=(6,5))
plt.plot(df["N"], df["speedup"], "o-")
plt.xscale("log"); plt.xlabel("N"); plt.ylabel("Speedup (CPU/GPU)")
plt.title("MUSCL+RK2: GPU speedup vs problem size")
plt.grid(True, alpha=0.3)
plt.show()
df