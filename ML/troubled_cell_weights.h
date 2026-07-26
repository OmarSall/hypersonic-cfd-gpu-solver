// Auto-exported weights for the troubled-cell indicator MLP (4 -> 16 -> 1).
// Trained in train_troubled_cell_nn.py; DO NOT hand-edit.
// Regenerate via: python3 export_weights_to_header.py
const double NN_W1[64] = { 1.6951214385e+00, 3.2201088243e-02, -2.9620218805e-01, 8.2259628573e-01, 5.6829510256e-01, -3.4785210845e-02, 1.9651198046e+00, 5.4829712305e-02, -6.3714125202e-01, -1.6513372375e+00, 1.2734758082e+00, 7.4410292372e-01, -2.0240962949e+00, -1.6839356444e+00, -7.4155349702e-03, -1.1919112673e+00, 2.3460417905e+00, 1.1605686471e+00, -1.5480816618e+00, 4.1567803537e-01, 2.8932352683e+00, 9.7117824187e-01, -1.4009475389e+00, -1.9572063176e+00, 1.5939523675e+00, 1.4318821429e+00, -3.8043799636e-01, 5.7925730195e-01, 2.6021900595e+00, 1.7244068196e+00, 6.5977270354e-01, -1.9029819569e+00, -4.6041896440e-01, -3.1314170980e+00, -1.0952393560e+00, -1.4456971623e-01, -5.7548925006e-01, -2.9296190376e+00, -1.1102260328e+00, 4.0709635038e-01, -7.2945216091e-01, -1.8330232266e+00, -4.8912372973e-01, -1.8474304420e+00, -4.3619936751e-01, -2.0663561315e+00, -1.1467048627e+00, -2.0534539076e-01, -1.7233350118e+00, -3.3876908351e+00, -6.4755924958e-01, -8.4564139219e-01, -9.2673196178e-01, 2.7238334968e-01, -9.2082391226e-02, 5.7039754494e-01, 1.7497231212e+00, 2.2566006796e+00, 3.7506209869e-01, -5.6549563002e-01, -8.6896534574e-01, -2.9333561611e+00, -5.7512834764e-01, -3.2263863753e-01 };  // [4][16] row-major
const double NN_B1[16] = { -1.2323707096e-02, 1.1105581390e+00, 3.8993607254e-03, 5.3136622785e-04, -1.1244001595e+00, -4.5019125431e-02, 2.2847547554e-02, 1.1672898394e+00, 1.6108565569e+00, 3.2119581067e-03, -2.5438246627e-02, -7.9577193822e-03, -1.9462713714e+00, -2.6608825770e+00, 5.8490696244e-03, 1.3528638474e+00 };
const double NN_W2[16] = { 5.6398847757e-01, 4.1592136124e-01, 1.2388817936e+00, 1.5327485565e+00, -1.7950577755e+00, 1.7049844000e+00, 8.3997778312e-01, -1.7727303516e+00, -1.2195893293e+00, 1.8541849959e+00, 1.5785773099e+00, 1.3019910281e+00, -1.9325114569e+00, -6.7718023240e-01, 1.0580373370e+00, -1.6795810493e+00 };  // [16][1]
const double NN_B2[1] = { 4.3183296219e-01 };
const double NN_MU[4] = { -1.5550952284e-03, 1.0800138251e-03, -3.2367762986e-04, 4.4030998693e-04 };
const double NN_SIGMA[4] = { 9.5928219482e-02, 6.7083578043e-02, 2.8395173968e-02, 3.9192071956e-02 };

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