# CPU vs GPU benchmark sweep for the 2D wedge solver (euler2d_wedge.cpp vs
# euler2d_wedge_gpu.cu). Both binaries already accept Nx, Ny as command-line
# arguments, so this just calls each one directly at increasing resolution --
# no source-substitution needed, unlike the 1D MUSCL sweep.
#
# Resolution list stops at Nx=400 by default: CPU cost for this first-order
# 2D solver scales steeply with resolution (see the refinement-study data --
# Nx=640,Ny=320 alone took ~240s on CPU), so a full automated loop through
# that size would add several extra minutes. Uncomment it below if you want
# that data point too; the GPU side is fast regardless.
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt

resolutions = [(80, 40), (160, 80), (320, 160), (400, 200)]
# resolutions.append((640, 320))  # uncomment for the extra data point (~4 extra min on CPU)

rows = []
for Nx, Ny in resolutions:
    cpu_out = subprocess.run(["./euler2d_wedge", str(Nx), str(Ny)], capture_output=True, text=True).stdout
    gpu_out = subprocess.run(["./euler2d_wedge_gpu", str(Nx), str(Ny)], capture_output=True, text=True).stdout

    cpu_ms = float(re.search(r"wall time=([\d.]+) ms", cpu_out).group(1))
    gpu_ms = float(re.search(r"wall time=([\d.]+) ms", gpu_out).group(1))

    rows.append({"Nx": Nx, "Ny": Ny, "cells": Nx * Ny,
                 "cpu_ms": cpu_ms, "gpu_ms": gpu_ms, "speedup": cpu_ms / gpu_ms})
    print(f"Nx={Nx:4d} Ny={Ny:4d} ({Nx*Ny:6d} cells): "
          f"CPU={cpu_ms:9.1f}ms  GPU={gpu_ms:8.1f}ms  speedup={cpu_ms/gpu_ms:5.2f}x")

df = pd.DataFrame(rows)

plt.figure(figsize=(6, 5))
plt.plot(df["cells"], df["speedup"], "o-")
plt.xscale("log")
plt.xlabel("Grid cells (Nx * Ny)")
plt.ylabel("Speedup (CPU / GPU)")
plt.title("2D Wedge Solver: GPU speedup vs problem size")
plt.grid(True, alpha=0.3)
plt.show()

df