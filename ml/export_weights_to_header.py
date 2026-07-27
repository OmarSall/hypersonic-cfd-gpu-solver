# Exports troubled_cell_weights.npz (trained by train_troubled_cell_nn.py)
# to troubled_cell_weights.h -- hand-coded C++ forward pass, no ML framework
# needed at inference time. Re-run this any time the network is retrained.
import numpy as np

d = np.load("troubled_cell_weights.npz")
W1, b1, W2, b2, mu, sigma = d["W1"], d["b1"], d["W2"], d["b2"], d["mu"], d["sigma"]

def carr(name, arr, shape_comment=""):
    flat = arr.flatten()
    body = ", ".join(f"{v:.10e}" for v in flat)
    return f"const double {name}[{len(flat)}] = {{ {body} }};{shape_comment}\n"

with open("troubled_cell_weights.h", "w") as f:
    f.write("// Auto-exported weights for the troubled-cell indicator MLP (4 -> 16 -> 1).\n")
    f.write("// Trained in train_troubled_cell_nn.py; DO NOT hand-edit.\n")
    f.write("// Regenerate via: python3 export_weights_to_header.py\n")
    f.write(carr("NN_W1", W1.T, "  // [4][16] row-major"))   # W1 is (4,16); flatten row-major k*16+h matches C++ indexing
    f.write(carr("NN_B1", b1))
    f.write(carr("NN_W2", W2, "  // [16][1]"))
    f.write(carr("NN_B2", b2))
    f.write(carr("NN_MU", mu))
    f.write(carr("NN_SIGMA", sigma))
    f.write("""
#include <cmath>
// Forward pass of the trained troubled-cell classifier. Takes density and
// pressure at cells i-1, i, i+1 and returns true if cell i should be
// treated as "troubled" (apply minmod, as before) vs "smooth" (use the raw
// unlimited central-difference slope instead). Mirrors the exact feature
// definition and normalization used in train_troubled_cell_nn.py.
inline bool nn_is_troubled(double rho_im1, double rho_i, double rho_ip1,
                            double p_im1, double p_i, double p_ip1) {
    double d_rho_l = (rho_i - rho_im1) / (std::fabs(rho_i) + 1e-8);
    double d_rho_r = (rho_ip1 - rho_i) / (std::fabs(rho_i) + 1e-8);
    double d_p_l   = (p_i - p_im1) / (std::fabs(p_i) + 1e-8);
    double d_p_r   = (p_ip1 - p_i) / (std::fabs(p_i) + 1e-8);

    double x[4] = { d_rho_l, d_rho_r, d_p_l, d_p_r };
    double xn[4];
    for (int k = 0; k < 4; ++k) xn[k] = (x[k] - NN_MU[k]) / NN_SIGMA[k];

    double hidden[16];
    for (int h = 0; h < 16; ++h) {
        double z = NN_B1[h];
        for (int k = 0; k < 4; ++k) z += xn[k] * NN_W1[k*16 + h];
        hidden[h] = z > 0.0 ? z : 0.0;   // ReLU
    }
    double z2 = NN_B2[0];
    for (int h = 0; h < 16; ++h) z2 += hidden[h] * NN_W2[h];
    // sigmoid(z2) > 0.5  <=>  z2 > 0 (sigmoid is monotonic, threshold at 0.5
    // is exactly z2=0) -- skips exp() entirely, pure speed optimization,
    // bit-for-bit identical decision to the sigmoid-then-threshold version.
    return z2 > 0.0;
}
""")
print("wrote troubled_cell_weights.h")