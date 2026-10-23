# Federated Agile Effort Estimation — Story Point Prediction

> **A research project exploring Split-Federation with FedProx for privacy-preserving, cross-project story point estimation using the Choetkiertikul et al. IEEE TSE 2018 benchmark.**

---

## Overview

This repository implements a **Federated Learning (FL) pipeline** for Agile story point estimation across 16 open-source JIRA projects. The goal is to train a shared global model without centralizing sensitive project data, while still achieving — and surpassing — the accuracy of a centralized baseline.

The project is structured across five progressive phases, each building upon the last: from a centralized stacking ensemble baseline to a personalized Split-Federation architecture with outlier-aware target transforms and hyperparameter tuning.

**Key result:** The final federated model (**Phase 5b**) achieves a **Macro MAE of 3.099**, representing a **~17.9% improvement** over the centralized baseline (MAE 3.774) — across all 16 clients — while never sharing raw project data.

---

## Architecture: Split-Federation (Phase 4+)

The core innovation is the **Split-Federation** design:

```
Server (FedProx, mu=0.1)
  |-- Aggregates: Keras MLP weights + Keras LSTM weights
  `-- Does NOT see: StackingRegressor, RF, LinearSVR, Ridge

Client (per JIRA project)
  |-- Global DL (received from server each round):
  |     |-- Keras MLP (128 -> Dropout -> 64 -> 1)  -> 64-dim embedding
  |     `-- Keras LSTM (64 -> 32 -> 1)             -> 32-dim embedding
  |
  |-- Embedding: concat(LSTM_emb, MLP_emb) = 96-dim vector
  |
  `-- Local ML (private, never aggregated):
        `-- StackingRegressor
              |-- RandomForest (100 trees)
              |-- LinearSVR (C=1.0)
              `-- Ridge (alpha=1.0) meta-learner
```

Global deep models act as **shared feature extractors**; local stacking ensembles adapt to each project's unique estimation dynamics — all without exposing private data to the server.

---

## Cross-Phase Results Summary

| Phase | Architecture | Macro MAE | Macro RMSE | vs. Phase 1 |
|-------|-------------|-----------|------------|-------------|
| 1 | Centralized Stacking Ensemble | 3.774 | 8.084 | — (baseline) |
| 2 | Vanilla FedAvg (DL only) | 4.209 | 10.263 | +11.5% worse |
| 3 | FedProx (DL only) | 4.282 | 10.189 | +13.5% worse |
| 4 | Split-Fed + True FedProx | 2.157* | 3.155* | -42.8% better* |
| 5 | Split-Fed + FedProx + log1p target | 3.166 | 5.507 | -16.1% better |
| **5b** | **Split-Fed + FedProx (frac=1.0) + log1p** | **3.099** | **5.485** | **-17.9% better** |

\* Phase 4 metrics are client-side local evaluation (best-round subset); Phases 5/5b use full 16-client comprehensive evaluation.

---

## Dataset

- **Source:** 16 open-source JIRA projects from [Choetkiertikul et al. (IEEE TSE 2018)](https://ieeexplore.ieee.org/document/8048213), hosted at `morakotch/datasets`.
- **Projects:** `appceleratorstudio`, `aptanastudio`, `bamboo`, `clover`, `datamanagement`, `duracloud`, `jirasoftware`, `mesos`, `moodle`, `mule`, `mulestudio`, `springxd`, `talenddataquality`, `talendesb`, `titanium`, `usergrid`
- **Scale:** ~23,313 issues total; 18,650 train / 4,663 test (80/20 stratified split)
- **Target:** Full continuous story point spectrum **[1, 100]** — *not* filtered to Fibonacci classes

**Features:** TF-IDF on `title + description`
- `TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True, stop_words="english")`
- Output: sparse matrices of shape `(n_samples, 5000)`

---

## Project Structure

```
.
|-- config.py                    # Centralized config: all paths, hyperparameters, constants
|-- requirements.txt             # Python dependencies
|-- run_colab.py                 # Entry point for Google Colab (Phase 3 simulation)
|-- run_phase1.ps1               # PowerShell script: run centralized baseline
|-- run_phase2.ps1               # PowerShell script: run Vanilla FedAvg
|-- run_phase3.ps1               # PowerShell script: run FedProx simulation
|-- setup_env.ps1                # PowerShell script: create virtual environment
|
|-- src/
|   |-- download_data.py         # Fetch raw CSVs from GitHub
|   |-- clean_data.py            # Data cleaning & preprocessing pipeline
|   |-- tfidf_pipeline.py        # TF-IDF feature extraction + train/test split
|   |-- CentralizedKhattab_phase1.py  # Phase 1: centralized MLP, SVR, RF, LSTM, stacking
|   |-- client.py                # FL client: FedProx training, Split-Federation logic
|   |-- server_phase2.py         # Phase 2: Flower server (FedAvg)
|   |-- server_phase3.py         # Phase 3: Flower server (FedProx)
|   |-- simulate_phase3.py       # Phase 3: single-process Flower simulation
|   |-- simulate_phase4.py       # Phase 4/5: Split-Federation simulation (CLI)
|   |-- generate_initial_weights.py   # Serialize initial model weights -> .npz
|   |-- partition_data.py        # Partition cleaned data per client (federated dir)
|   |-- retrain_local_ensembles_log1p.py  # Phase 5 validation: retrain ensembles only
|   |-- evaluate_all_phase4.py   # Full 16-client comprehensive evaluation
|   |-- evaluate_phase4_clipped.py    # Sanity check: clip out-of-range predictions
|   |-- check_distributions.py   # EDA: per-project SP distribution analysis
|   `-- metric_validation.py     # Metric sanity checks
|
|-- data/
|   |-- raw/                     # Per-project CSVs (downloaded)
|   |-- processed/               # cleaned_dataset.csv
|   |-- features/                # TF-IDF artifacts (vectorizer, X/y train/test)
|   `-- federated/               # Per-client data partitions
|
|-- models/
|   |-- initial_weights.npz      # Shared initial model weights
|   |-- phase1_khattab/          # Centralized baseline models
|   |-- phase2_basic_fl/         # Phase 2 global models
|   |-- phase3_fedprox/          # Phase 3 global models
|   |-- phase4_personalized/     # Phase 5b canonical models (Split-Fed, frac=1.0)
|   |-- phase4_personalized_phase5_mu01_frac05/  # Phase 5 reference checkpoint
|   `-- phase4_sweep_*/          # Hyperparameter sweep artifacts
|
|-- reports/                     # EDA outputs, distribution summaries, plots
|-- tests/                       # pytest test suite
`-- worktracker.md               # Empirical ledger: all metrics, decisions, caveats
```

---

## Phases at a Glance

### Phase 1 — Centralized Baseline
Trains MLP, LinearSVR, Random Forest, and LSTM regressors on a single centralized dataset, then combines them with a Ridge meta-learner stacking ensemble.
- **Best result:** MAE = 3.774, RMSE = 8.084

### Phase 2 — Vanilla FedAvg
Federates MLP and LSTM across 16 clients using the Flower framework with `FedAvg`. LinearSVR/RF are kept local (trees cannot be meaningfully averaged).
- **Observation:** Client drift causes 39.9% MAE degradation vs. Phase 1.

### Phase 3 — FedProx Simulation (Colab T4)
Migrates to Flower's single-process simulation engine (Ray actor pool) to solve RAM constraints. Applies FedProx (mu=0.1) to stabilize training over 10 rounds.
- **Observation:** FedProx stabilizes but does not solve the fundamental Non-IID drift problem.

### Phase 4 — Split-Federation + True FedProx
The architectural breakthrough: only deep model weights are aggregated; each client keeps a private local `StackingRegressor` trained on 96-dim deep embeddings.
- **Best result (local eval):** MAE = 2.157, RMSE = 3.155 — first phase to beat centralized baseline.

### Phase 5 — Outlier-Aware Target Transform (log1p)
Applies `np.log1p` to story points before the per-client `StandardScaler` to stabilize variance for heavy-tailed clients (notably `moodle`, `datamanagement`).
- **Impact:** Weighted MAE drops 15.9% (4.372 -> 3.677); MAE improvements on 13/16 clients.

### Phase 5b — FedProx Hyperparameter Sweep
Sweeps `proximal_mu` in {0.01, 0.1, 0.5} and `fraction_fit` in {0.5, 1.0} with 10 and 20 rounds.
- **Finding:** `fraction_fit=1.0` (full client participation per round) is the best configuration; mu=0.1 was already optimal.
- **Best result (comprehensive):** Macro MAE = 3.099, Weighted MAE = 3.614

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Windows (PowerShell) or Linux/macOS
- NVIDIA GPU recommended for Phase 4+ (tested on RTX 3050, 16 GB RAM)
- Google Colab T4 for Phase 3 (RAM-intensive multi-process simulation)

### Environment Setup

```powershell
# Windows (PowerShell)
.\setup_env.ps1
```

Or manually:

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Running the Pipeline

```powershell
# 1. Download & clean data
python src/download_data.py
python src/clean_data.py
python src/tfidf_pipeline.py
python src/partition_data.py

# 2. Phase 1: Centralized baseline
.\run_phase1.ps1

# 3. Phase 2: Vanilla FedAvg
.\run_phase2.ps1

# 4. Phase 3: FedProx simulation
.\run_phase3.ps1

# 5. Phase 4/5/5b: Split-Federation (canonical config)
python src/simulate_phase4.py --mu 0.1 --fraction_fit 1.0 --num_rounds 10

# 6. Evaluate all 16 clients
python src/evaluate_all_phase4.py
```

**Google Colab (Phase 3):**
```python
# Upload the repo, then:
!python run_colab.py
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Full-spectrum regression [1, 100]** | Real-world Agile tasks include rare high-effort items; filtering to {1,2,3,5,8} discards 18.4% of data and makes MAE values non-comparable |
| **LinearSVR over RBF SVR** | Eliminates O(n^3) compute on 5000-dim sparse TF-IDF features; better out-of-sample generalization |
| **LinearSVR/RF decoupled from FL loop** | Trees cannot be weight-averaged; SVR weights are stateless and overwrite injected globals, making FL aggregation meaningless |
| **Keras MLP over sklearn MLPRegressor** | Weight serialization compatible with Keras LSTM for uniform Flower aggregation |
| **Sequential execution (no Ray) on Windows** | TensorFlow CUDA DLLs crash inside Ray worker processes on Windows |
| **log1p target transform (Phase 5)** | Stabilizes variance for heavy right-tailed SP distributions without discarding any data |
| **fraction_fit=1.0 (Phase 5b)** | Full client participation reduces aggregation variance and smooths convergence |

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `pandas` | >=2.2 | Data manipulation |
| `numpy` | >=1.26 | Numerical operations |
| `scikit-learn` | >=1.5 | ML models (SVR, RF, stacking) |
| `tensorflow` | >=2.16 | Keras MLP & LSTM |
| `flwr[simulation]` | >=1.8 | Federated Learning framework |
| `joblib` | >=1.4 | Model persistence |
| `matplotlib` / `seaborn` | >=3.9 / >=0.13 | Visualization & EDA |
| `pytest` | >=8.2 | Testing |

---

## Citation

Dataset from:
> Choetkiertikul, M., Dam, H. K., Tran, T., Treude, C., & Ghose, A. (2018). **Predicting the Delay of Issues with Due Dates in Software Projects**. *IEEE Transactions on Software Engineering*. DOI: [10.1109/TSE.2018.2868772](https://doi.org/10.1109/TSE.2018.2868772)

---

## Further Reading

See [`worktracker.md`](worktracker.md) for the complete empirical ledger: per-round metrics for every phase, architectural decisions, root-cause analyses, and all documented caveats.
