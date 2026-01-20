# Beyond Fidelity: A Trust-Oriented Framework for Evaluating Generative Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper **"Beyond Fidelity: A Trust-Oriented Framework for Evaluating Generative Models"**, submitted to *Engineering Applications of Artificial Intelligence (EAAI)*.

It introduces a comprehensive framework for evaluating generative models beyond scalar metrics (like FID), focusing on **Reliability**, **Safety**, and **Trustworthiness**. The repository includes code to reproduce the "Illusion of Fidelity" experiments comparing Adversarial (TimeGAN) and Variational (LSTM-VAE) architectures.

![Trust Profile Radar Chart](Figure4_TrustProfile.png)
> **Figure 4:** The Trust Profile reveals that while TimeGAN and LSTM-VAE have identical Fidelity, they exhibit opposite Safety and Robustness characteristics.

## 🚀 Key Features
* **Trust Profile Visualization:** A standardized radar chart to visualize trade-offs between Fidelity, Diversity, Robustness, Fairness, Utility, and Safety.
* **Dual-Track Protocol:** Includes code for both **Controlled Proxies** (Synthetic Sine Waves, Fashion-MNIST) and **Deep Generative Benchmarks** (TimeGAN vs. LSTM-VAE).
* **Metric Implementations:** Ready-to-use Python functions for:
    * Fidelity (Discriminative Score, FID Proxy)
    * Robustness (Latent Perturbation Sensitivity)
    * Safety (Physiological Constraint Checking for ECG)
    * Fairness (Recall Consistency)

---

## 📂 Repository Structure

```text
MEGM/
├── src/
│   ├── mitbih_multi_class_colabNew.py       # MAIN SCRIPT: Experiment III (TimeGAN vs LSTM-VAE)
│   ├── trust_profile_experiment_time_series.py  # Experiment I: Synthetic Sine Waves
│   ├── trust_profile_experiments_image.py       # Experiment II: Fashion-MNIST
│   ├── drawFigure1.py        # Framework Overview Diagram
│   ├── drawFigure2.py        # Metric Taxonomy Diagram
│   ├── drawFigure3.py        # Protocol Pipeline Diagram
│   └── drawFigure4.py        # Trust Profile Radar Chart (Figure 4)
├── data_mitbih/              # Folder for MIT-BIH dataset (auto-downloaded)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone [https://github.com/bilkentli88/MEGM.git](https://github.com/bilkentli88/MEGM.git)
cd MEGM
pip install -r requirements.txt
```

### 2. Run the Main Clinical Experiment (Experiment III)
This script reproduces the core results of the paper (Table 4). It downloads the MIT-BIH dataset, trains TimeGAN and LSTM-VAE models, computes the six trust metrics, and prints the comparison table.
```bash
python src/mitbih_multi_class_colabNew.py
```
* **Runtime:** Approx. about 30-45 minutes (depending on GPU).
* **Output:** Training logs and the final "Trust Profile" comparison table.

### 3. Run Controlled Proxy Experiments
To verify metric sensitivity using synthetic data (Experiment I) or images (Experiment II):
```bash
python src/trust_profile_experiment_time_series.py   # Experiment I (Sine Waves)
python src/trust_profile_experiments_image.py        # Experiment II (Fashion-MNIST)
```

### 4. Generate Figures
To reproduce the high-quality figures used in the manuscript:
```bash
python src/drawFigure4.py   # Generates the Radar Chart (Figure 4)
python src/drawFigure1.py   # Generates the Framework Overview
python src/drawFigure2.py   # Generates the Taxonomy Diagram
python src/drawFigure3.py   # Generates the Protocol Pipeline
```

---

## 📊 Main Results (The "Illusion of Fidelity")

The framework reveals that while Adversarial and Variational models may achieve identical **Fidelity**, they exhibit fundamentally different risk profiles.

| Metric | TimeGAN (Adversarial) | LSTM-VAE (Variational) |
| :--- | :---: | :---: |
| **Fidelity** (Lower is better) | **1.396** | **1.383** (Identical) |
| **Safety** (% Valid Signals) | 32.0% | **88.5%** |
| **Robustness** (MSE) | 5.9e-2 | **3.0e-3** |
| **Diversity** | **17.17** | 10.33 |

> **Key Insight:** TimeGAN acts as a "Risk-Seeker" (High Diversity, Low Safety), while LSTM-VAE acts as a "Risk-Avoider" (High Safety, Low Diversity). Scalar metrics like FID hide this trade-off.

---

## 📜 Citation
If you use this code or framework, please cite:

```bibtex
@article{Altay2026BeyondFidelity,
  title={Beyond Fidelity: A Trust-Oriented Framework for Evaluating Generative Models},
  author={Altay, Aykut T.},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026},
  note={Under Review}
}
```

## 📧 Contact
For questions or issues, please open an issue or contact the corresponding author:
**Aykut Altan** - [aykuttaltay@gmail.com](mailto:aykuttaltay@gmail.com)
