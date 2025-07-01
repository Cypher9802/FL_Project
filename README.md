# Federated Learning for Personalized Fitness Tracking on Mobile Devices

## Overview

This repository implements a **federated learning (FL) framework** for privacy-preserving, mobile-optimized human activity recognition (HAR) using the UCI HAR dataset.  
It supports secure aggregation, differential privacy, and is ready for deployment and academic benchmarking.

---

## Features

- **Full UCI HAR pipeline:** normalization, subject split, augmentation, feature extraction
- **Mobile-optimized CNN:** <5MB, GroupNorm (DP-compatible), dropout, L2 regularization
- **Federated Learning:** FedAvg, 10/30 clients per round, 30 rounds
- **Differential Privacy:** Opacus, ε=8.0, δ=1e-5, per-client DP accounting
- **Secure Aggregation:** Noise-injected aggregation for privacy
- **Cross-platform:** Works on Mac (Intel/Apple Silicon), Windows, Linux, CUDA, CPU, and MPS
- **Evaluation scripts:** Centralized/Federated training, privacy analysis, cross-validation, ablation

---

## Usage

### 1. **Clone and Set Up**

git clone https://github.com/YourUsername/FL_Project.git
cd FL_Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
text

### 2. **Download and Place Dataset**

- Download the UCI HAR Dataset from:  
  [UCI HAR Dataset Link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- Extract the folder as `UCI HAR Dataset/` in your project root.

### 3. **Run Experiments**

**Centralized Pretraining (optional, for baseline):**
python -m scripts.pretrain_model
text

**Federated Training:**
python -m scripts.train_federated
text

**Model Evaluation:**
python -m scripts.evaluate_model
text

**Privacy Analysis:**
python -m scripts.privacy_analysis
text

**Cross-validation:**
python -m scripts.cross_validate
text

---

## Project Structure

FL_Project/
├── config/ # All configuration (see config/init.py)
├── data/ # Data loader and preprocessing
├── federated/ # FL client/server/aggregation logic
├── models/ # Mobile-optimized CNN model
├── scripts/ # Training, evaluation, privacy, and utility scripts
├── UCI HAR Dataset/ # Place the raw dataset here
├── requirements.txt
└── README.md
text

---

## Research Plan & Paper Alignment

- **Dataset:** UCI HAR, 30 subjects, 6 activities, 9 features (acc+gyro+total_acc)
- **Model:** Mobile-optimized CNN, GroupNorm (DP-safe), <5MB
- **FL:** 10/30 clients per round, 30 rounds, FedAvg, secure aggregation
- **Privacy:** Differential privacy (ε=8.0, δ=1e-5), Opacus, per-client accounting
- **Evaluation:** Accuracy, F1, recall, privacy/communication overhead, ablation
- **Benchmarks:** Centralized, local-only, FL with/without privacy

For full methodology and results, see [`FL_Planning_Steps.pdf`](./FL_Planning_Steps.pdf).

---

## Citation

If you use this codebase or its results, please cite:

@inproceedings{your2025flhar,
title={Federated Learning on Mobile Devices for Personalized Fitness Tracking: Privacy-Preserving Model Optimization and Performance Analysis},
author={Your Name},
year={2025},
note={\url{https://github.com/YourUsername/FL_Project}}
}
text

---

## License

MIT License. See [LICENSE](./LICENSE) for details.

---

## Contact

For questions or collaboration, open an issue or contact [your-email@example.com](mailto:your-email@example.com).
