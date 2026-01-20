import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random


def set_seed(seed=88):  # Matching the Time Series seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(88)

# ==========================================
# 1. SETUP: Load Fashion-MNIST
# ==========================================
print("--- Loading Data... ---")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# We need the whole dataset in memory to sample from it
train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

train_batch, train_labels = next(iter(train_loader))
real_data = train_batch.view(-1, 784).numpy()
real_labels = train_labels.numpy()

test_batch, test_labels = next(iter(test_loader))
real_test_X = test_batch.view(-1, 784).numpy()
real_test_y = test_labels.numpy()


# ==========================================
# 2. GENERATORS
# ==========================================
def simulated_generator(indices, robustness_level='low'):
    """
    Generates images based on real seed indices.
    Returns: Generated Images (X), Corresponding Labels (y)
    """
    outputs = []
    labels = []

    # Noise scale: High for Baseline (Low Robustness), Low for Robust
    noise_scale = 0.25 if robustness_level == 'low' else 0.05

    for idx in indices:
        base_img = real_data[idx]
        base_label = real_labels[idx]

        # Add noise to simulate generative imperfection
        noise = np.random.normal(0, noise_scale, base_img.shape)
        outputs.append(base_img + noise)
        labels.append(base_label)

    return np.array(outputs), np.array(labels)


# ==========================================
# 3. METRICS
# ==========================================
def calculate_fid_proxy(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_robustness_score_img(indices, robustness_level):
    # Save Random State to ensure "Same Input, Different Noise" test
    st = np.random.get_state()

    out1, _ = simulated_generator(indices, robustness_level)
    out2, _ = simulated_generator(indices, robustness_level)

    np.random.set_state(st)
    return np.mean((out1 - out2) ** 2)


def get_utility_score(gen_X, gen_y, real_test_X, real_test_y):
    """
    Train on Synthetic (Gen), Test on Real.
    """
    # Use a Random Forest (fast and robust for tabular-like pixel data)
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(gen_X, gen_y)
    preds = clf.predict(real_test_X)
    return accuracy_score(real_test_y, preds)


# ==========================================
# 4. RUN EXPERIMENT
# ==========================================
if __name__ == "__main__":
    print("--- Running Image Experiment (With Utility) ---")

    # Select 1000 random seeds for generation
    indices = np.random.randint(0, len(real_data), 1000)

    # Generate Data
    X_base, y_base = simulated_generator(indices, robustness_level='low')
    X_robust, y_robust = simulated_generator(indices, robustness_level='high')

    # 1. Fidelity
    # Compare against a reference batch of real data (first 1000)
    real_ref = real_data[:1000]
    fid_base = calculate_fid_proxy(real_ref, X_base)
    fid_robust = calculate_fid_proxy(real_ref, X_robust)

    # 2. Diversity
    div_base = np.mean([np.linalg.norm(X_base[i] - X_base[j]) for i in range(100) for j in range(i + 1, 100)])
    div_robust = np.mean([np.linalg.norm(X_robust[i] - X_robust[j]) for i in range(100) for j in range(i + 1, 100)])

    # 3. Robustness
    rob_base = get_robustness_score_img(indices, 'low')
    rob_robust = get_robustness_score_img(indices, 'high')

    # 4. Utility (New!)
    util_base = get_utility_score(X_base, y_base, real_test_X, real_test_y)
    util_robust = get_utility_score(X_robust, y_robust, real_test_X, real_test_y)

    print("\n" + "=" * 40)
    print("FINAL RESULTS (IMAGES - SEED 88)")
    print("=" * 40)
    print(f"{'Metric':<20} | {'Baseline':<20} | {'Robust':<20}")
    print("-" * 65)
    print(f"{'Fidelity (FID)':<20} | {fid_base:.4f}               | {fid_robust:.4f}")
    print(f"{'Diversity':<20} | {div_base:.4f}               | {div_robust:.4f}")
    print(f"{'Robustness (MSE)':<20} | {rob_base:.4f}               | {rob_robust:.4f}")
    print(f"{'Utility (Acc)':<20} | {util_base:.4f}               | {util_robust:.4f}")
    print("=" * 40)