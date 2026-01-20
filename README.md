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
