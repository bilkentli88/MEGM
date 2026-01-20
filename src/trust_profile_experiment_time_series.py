import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier


# --- REPRODUCIBILITY SETUP ---
def set_seed(seed=79):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(88)  # <--- This locks the results


# ==========================================
# 1. SETUP: Synthetic Time Series Data
# ==========================================
def generate_real_data(n_samples=1000, seq_len=50):
    """Generates synthetic sine waves with class-dependent frequencies."""
    X = []
    y = []
    for _ in range(n_samples):
        label = np.random.randint(0, 2)
        freq = 1.0 if label == 0 else 2.0
        noise = np.random.normal(0, 0.1, seq_len)
        time_steps = np.linspace(0, 4 * np.pi, seq_len)

        signal = np.sin(freq * time_steps) + noise
        X.append(signal)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


# ==========================================
# 2. DEFINE GENERATIVE MODELS
# ==========================================
class SimpleGenerator(nn.Module):
    def __init__(self, input_dim=10, seq_len=50, robustness_level='low'):
        super().__init__()
        self.seq_len = seq_len
        self.robustness_level = robustness_level
        self.fc = nn.Linear(input_dim, 128)
        self.rnn = nn.LSTM(128, 64, batch_first=True)
        self.head = nn.Linear(64, seq_len)

    def forward(self, z, labels):
        # Conditioning: Concatenate z with labels
        z_cond = torch.cat([z, labels.unsqueeze(1).float()], dim=1)
        x = torch.relu(self.fc(z_cond))
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.rnn(x)
        out = self.head(out[:, -1, :])

        # --- SIMULATING BEHAVIOR ---
        if self.robustness_level == 'low':
            # Add random noise to simulate instability
            # Note: We use torch.randn_like which respects the torch seed
            out = out + torch.randn_like(out) * 0.2

        return out


# ==========================================
# 3. METRICS
# ==========================================
def get_fidelity_score(real_data, syn_data):
    X = np.concatenate([real_data, syn_data])
    y = np.concatenate([np.ones(len(real_data)), np.zeros(len(syn_data))])
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    clf.fit(X, y)
    acc = clf.score(X, y)
    return abs(acc - 0.5) * 2


def get_diversity_score(syn_data):
    subset = syn_data[:100]
    dists = []
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            d = np.linalg.norm(subset[i] - subset[j])
            dists.append(d)
    return np.mean(dists)


def get_robustness_score(model, z_dim=9):
    model.eval()
    # Fixed seed guarantees these Z vectors are the same for both runs
    z = torch.randn(100, z_dim)
    labels = torch.randint(0, 2, (100,))
    epsilon = 0.1

    with torch.no_grad():
        orig_out = model(z, labels).numpy()
        pert_out = model(z + epsilon, labels).numpy()

    mse = np.mean((orig_out - pert_out) ** 2)
    return mse


def get_utility_consistency(real_X, real_y, generator, n_runs=5):
    accuracies = []
    real_X_test = real_X[:200]
    real_y_test = real_y[:200]

    for seed in range(n_runs):
        # Re-seed inside loop if needed, but 'seed' variable handles RF randomness
        torch.manual_seed(seed)  # Ensure Z generation is reproducible per run
        z = torch.randn(500, 9)
        gen_labels = torch.randint(0, 2, (500,))
        with torch.no_grad():
            syn_X = generator(z, gen_labels).numpy()

        clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=seed)
        clf.fit(syn_X, gen_labels.numpy())
        acc = clf.score(real_X_test, real_y_test)
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)


# ==========================================
# 4. RUN
# ==========================================
if __name__ == "__main__":
    print("--- Running Time Series Experiment ---")

    X_real, y_real = generate_real_data()

    model_base = SimpleGenerator(input_dim=10, robustness_level='low')
    model_robust = SimpleGenerator(input_dim=10, robustness_level='high')

    # Generate Samples
    torch.manual_seed(100)  # Specific seed for generation batch
    z = torch.randn(1000, 9)
    labels = torch.randint(0, 2, (1000,))

    with torch.no_grad():
        X_base = model_base(z, labels).numpy()
        X_robust = model_robust(z, labels).numpy()

    # Calculate Metrics
    fid_base = get_fidelity_score(X_real, X_base)
    fid_robust = get_fidelity_score(X_real, X_robust)

    div_base = get_diversity_score(X_base)
    div_robust = get_diversity_score(X_robust)

    rob_base = get_robustness_score(model_base)
    rob_robust = get_robustness_score(model_robust)

    util_mean_base, util_std_base = get_utility_consistency(X_real, y_real, model_base)
    util_mean_robust, util_std_robust = get_utility_consistency(X_real, y_real, model_robust)

    print("\n" + "=" * 40)
    print("FINAL RESULTS (TIME SERIES)")
    print("=" * 40)
    print(f"{'Metric':<20} | {'Baseline':<20} | {'Robust':<20}")
    print("-" * 65)
    print(f"{'Fidelity':<20} | {fid_base:.4f}               | {fid_robust:.4f}")
    print(f"{'Diversity':<20} | {div_base:.4f}               | {div_robust:.4f}")
    print(f"{'Robustness (MSE)':<20} | {rob_base:.4f}               | {rob_robust:.4f}")
    print(f"{'Utility (Mean)':<20} | {util_mean_base:.4f}               | {util_mean_robust:.4f}")
    print(f"{'Utility (Std)':<20} | {util_std_base:.4f}               | {util_std_robust:.4f}")
    print("=" * 40)