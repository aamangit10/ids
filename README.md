# Explainable Self-Supervised Transformer-Based Intrusion Detection System for Unseen Network Attacks

> **Defensive Cybersecurity Research System**  
> Inspired by the research paper: *"Self-Supervised Transformer-based Contrastive Learning for Intrusion Detection Systems"* (Ippokratis Koukoulis, Ilias Syrigos, Thanasis Korakis, 2025).

[![Live Frontend](https://img.shields.io/badge/Live%20Demo-ids--frontend.onrender.com-00C7B7?style=for-the-badge&logo=render&logoColor=white)](https://ids-frontend-clzl.onrender.com)
[![Backend API](https://img.shields.io/badge/Backend%20API-ids--backend.onrender.com-46E3B7?style=for-the-badge&logo=fastapi&logoColor=black)](https://ids-backend-3jtg.onrender.com)
[![Swagger Docs](https://img.shields.io/badge/API%20Docs-Swagger%20UI-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)](https://ids-backend-3jtg.onrender.com/docs)

---

## 🌐 Live Deployed Application

The system is deployed in production on **Render** and is publicly accessible from any laptop, mobile device, or computer:

| Component | Live URL | Description |
| :--- | :--- | :--- |
| 🖥️ **Web Dashboard UI** | **[https://ids-frontend-clzl.onrender.com](https://ids-frontend-clzl.onrender.com)** | Interactive React 18 dashboard for live flow detection, unseen attack benchmarks, and SHAP explainability. |
| ⚙️ **FastAPI REST Backend** | **[https://ids-backend-3jtg.onrender.com](https://ids-backend-3jtg.onrender.com)** | PyTorch Transformer & classical ML inference engine. |
| 📑 **Interactive API Documentation** | **[https://ids-backend-3jtg.onrender.com/docs](https://ids-backend-3jtg.onrender.com/docs)** | Interactive Swagger UI to execute live model predictions and tests. |

---

## 🛡️ Safety & Scope Notice
**This software is exclusively defensive.** It only analyzes existing network traffic logs and packet sequence features to identify anomalous or suspicious patterns. It contains **NO** attack generation tools, exploit payloads, port scanning engines, credential crackers, or offensive network functionality.

---

## 1. Research Background & Problem Statement

Machine-Learning Intrusion Detection Systems (IDS) face two major operational challenges in production network security:
1. **Label Dependency**: Deep learning and classical models often require vast quantities of labeled network traffic data, which is expensive and time-consuming to obtain.
2. **Out-of-Distribution (OOD) Generalization**: Standard models struggle to detect novel or unseen network attack patterns that were not present during training.

### Research Inspiration
The 2025 research paper by Koukoulis et al. addresses these challenges by applying self-supervised contrastive learning (InfoNCE loss) on 5-tuple packet sequences using Transformer encoders.

---

## 2. Main Research Contributions

This project reproduces the core principles of Koukoulis et al. (2025) at an educational/research-project scale and introduces **TWO primary research contributions**:

### Contribution 1: Held-Out / Unseen Attack Class Evaluation
- **Zero-Sample Training Isolation**: During training, a target attack class (e.g. `Web Attack`, `PortScan`, `DDoS`, `Bot`) is **100% removed** from the training pool.
- **Evaluation**: The model is trained strictly on `BENIGN` + other known attacks, and then tested on a test set containing the held-out attack class.
- **Out-of-Distribution Metric**: Calculates **Held-Out Attack Recall** to evaluate zero-shot detection capability without claiming true zero-day attack detection.
- **Terminology**: Uses precise academic terms (`simulated unseen-attack scenario`, `held-out attack class`, `out-of-distribution attack class`).

### Contribution 2: Multi-Level Explainable AI (XAI)
- **Classical Models (Random Forest, XGBoost, Logistic Regression)**: Uses **SHAP** (SHapley Additive exPlanations) to identify top contributing numerical flow features (e.g., unusual packet length, short inter-arrival times, abnormal TCP flags).
- **Transformer Encoder**: Uses **Captum Integrated Gradients** to attribute predictions at both the **packet sequence timeline** and **packet feature level**.

---

## 3. Dataset & Dual-Mode Pipeline

### Primary Dataset: CICIDS2017

The system supports two operating modes:

#### A. CSV Mode (Tabular Traffic Features)
- Uses CICIDS2017 `MachineLearningCSV` files.
- Used for: Baseline models (Logistic Regression, Random Forest, XGBoost), SHAP explanations, binary & multiclass classification, and tabular held-out attack experiments.

#### B. Packet Mode (Packet Sequence Flows)
- Uses CICIDS2017 PCAP files or 5-tuple grouped packet sequence flows.
- Used for: PyTorch Transformer Encoder, Self-Supervised Contrastive Pretraining (InfoNCE), fine-tuning, and Captum Integrated Gradients.

#### DEMO Mode
If the full CICIDS2017 raw dataset is absent, the system automatically runs in **DEMO MODE** using synthetic flow data and labels results clearly as:
`“DEMO DATA — NOT RESEARCH RESULTS”`.

To use the real dataset, place raw CICIDS2017 CSV files into `data/raw/`.

---

## 4. Preprocessing & Data Leakage Prevention

To prevent models from trivially identifying flows via socket metadata rather than attack patterns, the following features are strictly excluded:
- `Flow ID`
- `Source IP` / `Src IP`
- `Destination IP` / `Dst IP`
- `Source Port` / `Destination Port`
- `Timestamp`
- `Unnamed: 0`

### Preprocessing Protocol
1. Clean headers and handle missing values without dropping valid rows.
2. Replace infinite values (`Infinity`, `inf`) with `NaN`.
3. Fit `StandardScaler` **strictly on training data** (no test set leakage).
4. Encode targets into binary (`BENIGN = 0`, `ATTACK = 1`) and multiclass label encodings.

---

## 5. System Architecture

```
                                  +---------------------------------------------------+
                                  |         CICIDS2017 Dataset / Demo Simulator       |
                                  +------------------------+--------------------------+
                                                           |
                                  +------------------------+--------------------------+
                                  |                                                   |
                         [CSV Mode Pipeline]                                [Packet Mode Pipeline]
                                  |                                                   |
                        Features Preprocessing                                Flow Reconstruction
                    (NaN removal, IP/ID strip,                                5-Tuple Flow Grouping
                      StandardScaler fit)                                 (32-packet max, IAT/Flags)
                                  |                                                   |
                     +------------+------------+                             +--------+--------+
                     |                         |                             |                 |
             Classical Baselines      Held-Out Experiment            Transformer Encoder   SSL Contrastive
           (Logistic, RF, XGBoost)   (Zero Train Samples)             (PyTorch 4-head)    (InfoNCE / NT-Xent)
                     |                         |                             |                 |
                  SHAP XAI             Unseen Attack Metrics                 Fine-Tuning      Captum Integrated Gradients
                     |                         |                             |                 |
                     +-------------------------+-----------------------------+-----------------+
                                                           |
                                              +------------v------------+
                                              |    FastAPI REST Backend |
                                              +------------+------------+
                                                           |
                                              +------------v------------+
                                              | React Dashboard UI (Vite)|
                                              +-------------------------+
```

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ (for React frontend)

### Installation
```bash
# Clone workspace and install Python dependencies
pip install -r requirements.txt
```

### Running CLI Experiments

```bash
# 1. Run Classical Baselines
python -m research.experiments.run_baselines

# 2. Run Held-Out Attack Experiment (e.g. Web Attack)
python -m research.experiments.run_holdout --attack "Web Attack" --model "random_forest"

# 3. Run Self-Supervised Contrastive Pretraining
python -m research.ssl.pretrain

# 4. Fine-Tune Transformer on Labeled Data
python -m research.transformer.finetune
```

### Running Automated Test Suite
```bash
python -m pytest tests/
```

### Running Backend API & Dashboard

```bash
# Start FastAPI backend server (Port 8000)
uvicorn backend.app.main:app --reload --port 8000

# In a separate terminal, start React Dashboard (Port 3000)
cd frontend
npm run dev
```

---

## 7. Research Limitations

- **CICIDS2017 Benchmark**: CICIDS2017 is an older benchmark dataset with known label noise and distribution artifacts.
- **Held-Out Simulation**: Holding out attack classes simulates out-of-distribution traffic, but does not guarantee performance against real-world zero-day exploits.
- **Explainability Scope**: SHAP and Integrated Gradients reflect internal model feature attributions, not causal ground-truth proofs.

---

## 8. Future Work

- Integration with **CICIDS2018** and **UNSW-NB15** datasets.
- Cross-dataset out-of-distribution evaluation.
- Concept drift detection and streaming online inference.
