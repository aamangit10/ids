# Implementation Plan - Explainable Self-Supervised Transformer-Based IDS for Unseen Network Attacks

## Executive Summary
This project implements a state-of-the-art defensive cybersecurity research system inspired by Koukoulis et al. (2025): *"Self-Supervised Transformer-based Contrastive Learning for Intrusion Detection Systems"*.

The system provides dual-mode traffic analysis (CICIDS2017 CSV tabular mode & PCAP packet sequence mode), zero-shot out-of-distribution held-out attack evaluation, self-supervised contrastive pretraining for data-scarce scenarios, and multi-level explainability (SHAP for tabular baselines & Captum Integrated Gradients for Transformer sequence representations).

> **Safety Notice**: This software is strictly defensive. It only analyzes network traffic patterns and detects anomalies. It contains no attack payloads, exploit modules, port scanners, or offensive features.

---

## System Architecture

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

## Detailed Milestone Roadmap

### Phase 1: Core Foundation & Baselines (Milestones 1–4)

- [x] **Milestone 1: Project Structure, YAML Configuration & Skeleton Setup**
  - Verify directory layout: `backend/`, `frontend/`, `research/`, `configs/`, `models/`, `results/`, `tests/`, `notebooks/`, `scripts/`.
  - Config files: `configs/model.yaml` and `configs/experiments.yaml`.
  - Initial `FastAPI` app skeleton & `React` app skeleton setup.

- [ ] **Milestone 2: CICIDS2017 CSV Preprocessing & Synthetic Dataset Simulator**
  - Fix `clean_dataframe` whitespace stripping for target columns and handle missing values without dropping valid rows.
  - Implement `strip_leakage_features` to exclude `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, etc.
  - Implement `TabularPreprocessor` with strict train-fit scaler and reproducible feature transformations.
  - Generate/load synthetic demo dataset in `demo_data/` with clear UI labeling: `"DEMO DATA — NOT RESEARCH RESULTS"`.

- [ ] **Milestone 3: Baseline Machine Learning Models**
  - Abstract base interface `BaseModel` (`train`, `predict`, `predict_proba`, `evaluate`, `save`, `load`).
  - Implement `LogisticRegressionModel`, `RandomForestModel`, and `XGBoostModel` (with graceful fallback if xgboost is absent).
  - Compute full metrics: Accuracy, Precision, Recall, F1, ROC-AUC, FPR, Confusion Matrix.
  - CLI script `research/experiments/run_baselines.py`.

- [ ] **Milestone 4: Held-Out / Unseen Attack Experiment Framework & Critical Verification**
  - Implement `research/experiments/holdout_experiment.py` (`run_holdout_experiment(attack_name)`).
  - Support holding out classes (`Web Attack`, `DDoS`, `PortScan`, `Bot`, `Brute Force`, `DoS Slowloris`).
  - **CRITICAL TEST**: Assert exactly ZERO samples of the held-out attack class in the training partition.
  - CLI script `research/experiments/run_holdout.py`.

---

### Phase 2: Explainability & Dashboard Integration (Milestones 5–6)

- [ ] **Milestone 5: Tabular Model Explainability (SHAP)**
  - Implement `research/explainability/shap_explainer.py`.
  - Calculate TreeSHAP / LinearSHAP values for single flow predictions & global feature rankings.

- [ ] **Milestone 6: Backend API & Modern React Dashboard Integration**
  - FastAPI endpoints (`/api/health`, `/api/models`, `/api/predict/csv`, `/api/predict/flow`, `/api/explain`, `/api/metrics`, `/api/experiments/holdout`).
  - React (Vite) frontend with Tailwind/CSS, cybersecurity dark mode, interactive navigation, prediction uploads, held-out experiment runner, and SHAP/Integrated Gradients charts.

---

### Phase 3: Transformer & Self-Supervised Learning Pipeline (Milestones 7–12)

- [ ] **Milestone 7: Flow Reconstruction & Packet Sequence Pipeline**
  - Group flows by 5-tuple (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`). Exclude IP addresses from inputs.
  - Packet features: `packet_length`, `protocol`, `flags`, `inter_arrival_time`, `direction` (max sequence length 32).
  - PyTorch `Dataset` and `DataLoader` with padding and attention masks.

- [ ] **Milestone 8: PyTorch Transformer Encoder Architecture**
  - PyTorch Transformer Encoder (`max_seq_len=32`, `embed_dim=256`, `nhead=4`, `num_layers=4`).
  - Positional encoding, sequence pooling, and configurable classification head.

- [ ] **Milestone 9: Self-Supervised Contrastive Learning (SSL)**
  - Packet-level augmentations: feature masking, jittering, token dropping, contiguous replacement.
  - InfoNCE / NT-Xent contrastive loss and SSL pretraining pipeline (`research/ssl/pretrain.py`).
  - Fine-tuning pipeline (`research/transformer/finetune.py`).

- [ ] **Milestone 10: Transformer Explainability (Captum Integrated Gradients)**
  - Implement `research/explainability/captum_explainer.py`.
  - Packet and feature attribution matrix visualization.

- [ ] **Milestone 11: Limited-Label Efficiency Experiments**
  - Label fraction evaluation (100%, 50%, 20%, 10%, 5%) comparing Supervised Transformer vs. SSL Pretrained + Fine-tuned Transformer.

- [ ] **Milestone 12: Documentation, Automated Verification & Reproducibility**
  - Comprehensive unit test suite (`pytest`).
  - Detailed `README.md` with research gap, architecture, dataset setup instructions, limitations, and future work.

---

## Verification & Safety Protocols

1. **Safety Check**: No offensive code or attack execution.
2. **Zero-Leakage Assertion**: `assert count(held_out_attack in train_set) == 0`.
3. **No Scaler Fitting on Test Data**: Scalers fitted only on `X_train`.
4. **Execution Verification**: Run pytest and CLI commands to verify all milestones.
