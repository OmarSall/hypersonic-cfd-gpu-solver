# Trains a tiny MLP (4 -> 16 -> 1, hand-implemented forward/backward pass in
# plain NumPy, no framework) to classify "troubled" cells, and compares it
# against minmod's own implicit criterion (sign change in neighbor
# differences) evaluated against the SAME ground truth.
import numpy as np

data = np.load("troubled_cell_data.npz")
X, y, source = data["X"], data["y"], data["source"]
n = len(y)
print(f"total examples: {n}, fraction troubled: {y.mean():.4f}")

# ---- train/test split ----
rng = np.random.default_rng(0)
idx = rng.permutation(n)
split = int(0.8*n)
train_idx, test_idx = idx[:split], idx[split:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
source_test = source[test_idx]

# ---- feature normalization (store mean/std for use in C++ inference) ----
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-8
X_train_n = (X_train - mu) / sigma
X_test_n = (X_test - mu) / sigma

# ---- tiny MLP: 4 -> 16 (ReLU) -> 1 (sigmoid), hand-rolled ----
rng2 = np.random.default_rng(1)
n_in, n_hid = 4, 16
W1 = rng2.normal(0, np.sqrt(2.0/n_in), (n_in, n_hid))
b1 = np.zeros(n_hid)
W2 = rng2.normal(0, np.sqrt(2.0/n_hid), (n_hid, 1))
b2 = np.zeros(1)

def forward(X):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)          # ReLU
    z2 = a1 @ W2 + b2
    a2 = 1.0/(1.0+np.exp(-z2))      # sigmoid
    return z1, a1, z2, a2

# class-weighted BCE to handle the ~96/4 imbalance -- otherwise the network
# can get ~96% "accuracy" by always predicting "smooth", which is useless
pos_weight = (1 - y_train.mean()) / y_train.mean()

# Adam optimizer, hand-rolled
params = [W1, b1, W2, b2]
m_ = [np.zeros_like(p) for p in params]
v_ = [np.zeros_like(p) for p in params]
lr, beta1, beta2, eps = 0.01, 0.9, 0.999, 1e-8

n_epochs = 60
batch_size = 512
y_train_col = y_train.reshape(-1,1)

for epoch in range(n_epochs):
    perm = rng.permutation(len(X_train_n))
    total_loss = 0.0
    for start in range(0, len(perm), batch_size):
        b_idx = perm[start:start+batch_size]
        Xb, yb = X_train_n[b_idx], y_train_col[b_idx]

        z1, a1, z2, a2 = forward(Xb)
        w = 1 + (pos_weight-1)*yb           # per-sample weight
        eps_c = 1e-9
        loss = -np.mean(w*(yb*np.log(a2+eps_c) + (1-yb)*np.log(1-a2+eps_c)))
        total_loss += loss * len(b_idx)

        # backward pass
        dz2 = w*(a2 - yb) / len(Xb)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = Xb.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = [dW1, db1, dW2, db2]
        for k, (p, g) in enumerate(zip(params, grads)):
            m_[k] = beta1*m_[k] + (1-beta1)*g
            v_[k] = beta2*v_[k] + (1-beta2)*(g*g)
            m_hat = m_[k] / (1-beta1**(epoch+1))
            v_hat = v_[k] / (1-beta2**(epoch+1))
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    if (epoch+1) % 10 == 0:
        print(f"epoch {epoch+1}: loss={total_loss/len(X_train_n):.4f}")

# ---- evaluate on held-out test set ----
_, _, _, pred_test = forward(X_test_n)
pred_label = (pred_test.flatten() > 0.5).astype(int)
y_test_int = y_test.astype(int)

tp = np.sum((pred_label==1) & (y_test_int==1))
fp = np.sum((pred_label==1) & (y_test_int==0))
fn = np.sum((pred_label==0) & (y_test_int==1))
tn = np.sum((pred_label==0) & (y_test_int==0))
precision = tp/(tp+fp+1e-9)
recall = tp/(tp+fn+1e-9)
f1 = 2*precision*recall/(precision+recall+1e-9)
acc = (tp+tn)/len(y_test_int)
print(f"\nNN  test: accuracy={acc:.4f} precision={precision:.4f} recall={recall:.4f} F1={f1:.4f}")

# ---- minmod's own implicit criterion on the SAME test set, for comparison:
# "troubled" if either rho's or p's neighbor differences have opposite signs
# (the exact condition under which minmod(a,b) returns 0, i.e. full limiting) ----
d_rho_l, d_rho_r, d_p_l, d_p_r = X_test[:,0], X_test[:,1], X_test[:,2], X_test[:,3]
minmod_troubled = ((d_rho_l*d_rho_r <= 0) | (d_p_l*d_p_r <= 0)).astype(int)
tp_m = np.sum((minmod_troubled==1) & (y_test_int==1))
fp_m = np.sum((minmod_troubled==1) & (y_test_int==0))
fn_m = np.sum((minmod_troubled==0) & (y_test_int==1))
tn_m = np.sum((minmod_troubled==0) & (y_test_int==0))
precision_m = tp_m/(tp_m+fp_m+1e-9)
recall_m = tp_m/(tp_m+fn_m+1e-9)
f1_m = 2*precision_m*recall_m/(precision_m+recall_m+1e-9)
acc_m = (tp_m+tn_m)/len(y_test_int)
print(f"minmod test: accuracy={acc_m:.4f} precision={precision_m:.4f} recall={recall_m:.4f} F1={f1_m:.4f}")

print(f"\nminmod flags {minmod_troubled.mean()*100:.1f}% of cells as troubled (true rate: {y_test.mean()*100:.1f}%)")
print(f"NN     flags {pred_label.mean()*100:.1f}% of cells as troubled")

# ---- save weights for C++ export ----
np.savez("troubled_cell_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, mu=mu, sigma=sigma)
print("\nsaved troubled_cell_weights.npz")

# ---- stratified evaluation: Riemann-problem cells vs smooth-function cells,
# separately -- this is the comparison that actually matters, since minmod's
# known weakness is specifically about smooth extrema, not genuine shocks ----
def evaluate(mask, label):
    yt = y_test_int[mask]
    pred_nn = pred_label[mask]
    d_rho_l, d_rho_r, d_p_l, d_p_r = X_test[mask,0], X_test[mask,1], X_test[mask,2], X_test[mask,3]
    pred_mm = ((d_rho_l*d_rho_r < 0) | (d_p_l*d_p_r < 0)).astype(int)

    for name, pred in [("NN    ", pred_nn), ("minmod", pred_mm)]:
        fp = np.sum((pred==1)&(yt==0))
        tp = np.sum((pred==1)&(yt==1))
        fn = np.sum((pred==0)&(yt==1))
        false_trigger_rate = fp / max(1, np.sum(yt==0))
        recall = tp / max(1, np.sum(yt==1))
        print(f"  {label:8s} {name}: false-trigger-rate={false_trigger_rate*100:5.2f}%  recall={recall*100:5.1f}%  (n={mask.sum()}, n_troubled={yt.sum()})")

print("\n--- stratified by data source ---")
evaluate(source_test == 0, "Riemann")
evaluate(source_test == 1, "smooth")
if (source_test == 2).any():
    evaluate(source_test == 2, "smeared")