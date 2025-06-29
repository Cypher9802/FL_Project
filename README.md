# FL_Project
Federated Learning Project that needs 90%> accuracy and privacy analsysis with "Differetiated Privacy"


Here is a comprehensive `README.md` for your project, summarizing its purpose, technical status, major issues encountered, and key accomplishments. This follows best practices from [modern ML/data science README templates][7][6][3][5], and is tailored to your federated learning, privacy, and mobile deployment context.

# Federated Learning for Personalized Fitness Tracking on Mobile Devices

[![Python](https://img.shields.io/badge/python-3.11%](https://img.shields.io/badge/License-MIT-yellow.svg of Contents

- [Project Overview](#project-overview)
- [Features & Accomplishments](#features--accomplishments)
- [Technical Architecture](#technical-architecture)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Known Issues & Lessons Learned](#known-issues--lessons-learned)
- [Results & Insights](#results--insights)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project implements a **federated learning (FL) framework for personalized fitness tracking on mobile devices**, with a strong focus on privacy (differential privacy, secure aggregation), model efficiency (quantization, pruning), and real-world deployability.  
The system uses mobile sensor data (accelerometer and gyroscope) from the UCI HAR Dataset to recognize human activities, training a neural network collaboratively across simulated clients (one per user/device) without sharing raw data[1].

## Features & Accomplishments

- **End-to-end Federated Learning Pipeline**
  - Client-server FL with FedAvg aggregation, subject-split data, and robust communication.
- **Differential Privacy (DP)**
  - Per-round DP accounting, adaptive noise/clipping schedules, and privacy budget tracking.
- **Secure Aggregation**
  - Masked model updates to protect client gradients from the server.
- **Model Optimization for Mobile**
  - Quantization (QNNPACK backend), pruning, and TorchScript export for efficient on-device inference.
- **Comprehensive Evaluation**
  - Accuracy, F1, confusion matrix, privacy/utility tradeoff plots, and ablation studies.
- **Mobile Inference Benchmark**
  - Latency measurement and deployment readiness analysis.
- **Privacy Analysis & Visualization**
  - Automated privacy budget plots, round-by-round ε tracking, and markdown reports.
- **Hyperparameter Tuning & Cross-Validation**
  - Scripts for DP/utility sweeps and k-fold validation.
- **Conference-Ready Paper Structure**
  - All code and results aligned with top-tier ML conference standards.

## Technical Architecture

- **Dataset**: [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) (561 features, 6 activity classes, 30 subjects)
- **Model**: Feed-Forward Neural Network (561 → 128 → 64 → 6, ReLU, Softmax, Dropout)
- **FL Setup**: 30 clients (one per subject), 20 rounds, 8 clients per round, FedAvg aggregation
- **Privacy**: Differential Privacy (ε=10, δ=1e-5, noise decays from 0.5 to 0.1), per-round accounting
- **Optimization**: Quantization, pruning, adaptive clipping, data augmentation (noise, rotation)
- **Evaluation**: Centralized pretrain, FL with/without DP, ablation, mobile inference, privacy plots

## Installation & Setup

### Prerequisites

- Python 3.11+
- pip, virtualenv (recommended)
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) extracted to project root

### Install

```bash
git clone  fl_project
cd fl_project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Edit `config/config.yaml` to adjust FL, DP, and model parameters.

## Usage

### 1. **Centralized Pre-training (optional but recommended)**
```bash
python scripts/pretrain_model.py
```

### 2. **Federated Training**
```bash
python scripts/train_federated.py
```

### 3. **Evaluation**
```bash
python scripts/evaluate_model.py
```

### 4. **Privacy Analysis**
```bash
python scripts/privacy_analysis.py
```

### 5. **Mobile Inference Benchmark**
```bash
python scripts/mobile_infer.py
```

### 6. **Hyperparameter Tuning & Cross-validation**
```bash
python scripts/tune_hparams.py
python scripts/cross_validate.py
```

## Known Issues & Lessons Learned

- **Client Connection Failures**: Early versions saw "Client X fail connect" due to connection storms. Fixed by increasing server backlog, adding client backoff, and staggering launches.
- **DP Utility Loss**: Strong DP (low ε, high noise) caused accuracy to drop to 10–17%. Solved by pre-training, using moderate ε=10, and decaying noise/clipping.
- **Privacy Budget Not Saved**: Initial runs missed `server_privacy_analysis.json`. Fixed by saving privacy metrics after each round.
- **UndefinedMetricWarnings**: Occurred when some activity classes weren't present in test splits. Fixed by filtering zero-support labels and using `zero_division=0`.
- **Config Key Errors**: Passing `pretrain_path` to model constructors caused errors. Fixed by removing extraneous keys before instantiation.
- **Missing Plots/Reports**: Privacy and training plots now generated after every run, with robust error handling for missing files.

## Results & Insights

- **Final FL Model Accuracy**: **91.7%** (with ε ≈ 10, DP enabled, pretraining, and full pipeline)
- **Privacy Budget**: Tracked and visualized per round; no budget exhaustion with moderate ε.
- **Mobile Inference**: Quantized model achieves low latency and small size, suitable for on-device deployment.
- **Ablation Study**: Removing DP yields 95%+ accuracy; strong DP (ε **Status:**  
> - All core functionality implemented and tested  
> - Major issues (connection, DP utility, privacy tracking) resolved  
> - Ready for research paper submission and mobile deployment  
> - See `scripts/privacy_analysis.py` and generated plots for privacy/utility tradeoff  
> - See `scripts/evaluate_model.py` for final accuracy and confusion matrix

**For any issues, suggestions, or contributions, please open an issue or pull request.**

[1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e6811296-5397-4cce-8ef4-f293b9544881/4c93b0fe-d52c-4754-8471-bfb136ab9ca5/FL_Planning_Steps.pdf
[2]: https://github.com/rochacbruno/python-project-template
[3]: https://github.com/othneildrew/Best-README-Template
[5]: https://github.com/azavea/python-project-template/blob/master/README.md
[6]: https://faun.pub/how-to-write-a-proper-readme-md-for-your-project-on-github-e8d51ac32e73?gi=9fc572ee7457
[7]: https://github.com/catiaspsilva/README-template

**This README summarizes your project's journey, technical solutions, and research outcomes, and is ready for publication or sharing with collaborators.**

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e6811296-5397-4cce-8ef4-f293b9544881/4c93b0fe-d52c-4754-8471-bfb136ab9ca5/FL_Planning_Steps.pdf
[2] https://github.com/rochacbruno/python-project-template/blob/main/README.md
[3] https://github.com/othneildrew/Best-README-Template
[4] https://github.com/rochacbruno/python-project-template
[5] https://github.com/azavea/python-project-template/blob/master/README.md
[6] https://faun.pub/how-to-write-a-proper-readme-md-for-your-project-on-github-e8d51ac32e73?gi=9fc572ee7457
[7] https://github.com/catiaspsilva/README-template
[8] https://coderefinery.github.io/documentation/writing-readme-files/
[9] https://github.com/microsoft/python-package-template/blob/main/README.md
[10] https://github.com/pamelafox/python-project-template/blob/main/README.md
[11] https://realpython.com/readme-python-project/
[12] https://github.com/python-project-templates
[13] https://github.com/sfbrigade/data-science-wg/blob/master/dswg_project_resources/Project-README-template.md
[14] https://www.makeareadme.com
[15] https://worldbank.github.io/template/README.html
[16] https://gist.github.com/ramantehlan/602ad8525699486e097092e4158c5bf1
[17] https://coding-boot-camp.github.io/full-stack/github/professional-readme-guide/
