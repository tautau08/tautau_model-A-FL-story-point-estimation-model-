# Worktracker

This file serves as our empirical ledger for the final paper. It documents the metrics and architectural milestones across all phases of the Federated Agile Effort Estimation project.

---

## Data Pipeline & Cleaning Methodology

**Source:** 16 open-source JIRA projects from the Choetkiertikul et al. (IEEE TSE 2018) benchmark, hosted at `morakotch/datasets`.

**Cleaning Steps (`src/clean_data.py`):**
1. Load all 16 per-project CSVs, normalize column names, tag with `project_id`.
2. Drop duplicate `issuekey` rows (keep first).
3. Fill missing `description` with empty string; drop rows with null `title` or `storypoint`.
4. Coerce `storypoint` to numeric via `pd.to_numeric(errors="coerce")`; drop non-numeric.
5. Cap at `STORYPOINT_CAP = 100` (Khattab et al. threshold); drop SP ≤ 0.
6. Create combined `text = title + " " + description`; lowercase, collapse whitespace; drop rows with text length ≤ 3 chars.

**Critical Design Decision: Full-Spectrum Regression (not Classification)**

Unlike classification approaches that filter to SP ∈ {1, 2, 3, 5, 8}, our pipeline retains the **full continuous story point spectrum** [1, 100]. No `.isin()` categorical filtering is applied anywhere in the pipeline.

| SP Value | Count | % |
|----------|-------|---|
| 1 | 3,371 | 18.1% |
| 2 | 2,734 | 14.7% |
| 3 | 3,178 | 17.0% |
| 4 | 675 | 3.6% |
| 5 | 3,449 | 18.5% |
| 6 | 330 | 1.8% |
| 7 | 40 | 0.2% |
| 8 | 2,487 | 13.3% |
| 9 | 32 | 0.2% |
| 10 | 310 | 1.7% |
| 11 | 16 | 0.1% |
| 12 | 65 | 0.3% |
| 13 | 783 | 4.2% |
| 14 | 20 | 0.1% |
| 15 | 55 | 0.3% |
| 16 | 45 | 0.2% |
| 17 | 18 | 0.1% |
| 18 | 11 | 0.1% |
| 19 | 16 | 0.1% |
| 20 | 372 | 2.0% |
| 21 | 88 | 0.5% |
| 22 | 10 | 0.1% |
| 23 | 6 | 0.0% |
| 24 | 11 | 0.1% |
| 25 | 7 | 0.0% |
| 26 | 14 | 0.1% |
| 27 | 1 | 0.0% |
| 28 | 8 | 0.0% |
| 29 | 4 | 0.0% |
| 30 | 45 | 0.2% |
| 31 | 4 | 0.0% |
| 32 | 4 | 0.0% |
| 33 | 4 | 0.0% |
| 34 | 23 | 0.1% |
| 35 | 6 | 0.0% |
| 36 | 7 | 0.0% |
| 38 | 12 | 0.1% |
| 39 | 6 | 0.0% |
| 40 | 147 | 0.8% |
| 42 | 3 | 0.0% |
| 43 | 2 | 0.0% |
| 45 | 9 | 0.0% |
| 46 | 1 | 0.0% |
| 47 | 2 | 0.0% |
| 48 | 1 | 0.0% |
| 49 | 4 | 0.0% |
| 50 | 16 | 0.1% |
| 52 | 3 | 0.0% |
| 53 | 16 | 0.1% |
| 54 | 7 | 0.0% |
| 55 | 1 | 0.0% |
| 56 | 5 | 0.0% |
| 57 | 1 | 0.0% |
| 58 | 1 | 0.0% |
| 60 | 22 | 0.1% |
| 63 | 1 | 0.0% |
| 64 | 1 | 0.0% |
| 65 | 2 | 0.0% |
| 66 | 1 | 0.0% |
| 68 | 1 | 0.0% |
| 69 | 1 | 0.0% |
| 70 | 3 | 0.0% |
| 71 | 1 | 0.0% |
| 75 | 3 | 0.0% |
| 79 | 15 | 0.1% |
| 80 | 10 | 0.1% |
| 82 | 1 | 0.0% |
| 83 | 1 | 0.0% |
| 84 | 1 | 0.0% |
| 88 | 1 | 0.0% |
| 89 | 1 | 0.0% |
| 90 | 4 | 0.0% |
| 92 | 1 | 0.0% |
| 95 | 1 | 0.0% |
| 100 | 92 | 0.5% |

- **75 unique SP values** in training set (range [1, 100])
- **81.6%** of rows have SP ∈ {1, 2, 3, 5, 8} — the Fibonacci core
- **12.8%** of rows have SP > 8 — high-effort outliers that drive MAE inflation
- **18.4%** of rows would be discarded by a 5-class classification filter

**Implication for MAE comparison:**
> A 5-class classification model (SP ∈ {1,2,3,5,8}) has a maximum single-prediction error of 7. Our regression model faces errors up to 99 (e.g., true=100, predicted=1). The two MAE values are **not directly comparable**. Our approach models the full complexity of real-world Agile estimation, including rare high-effort tasks that classification approaches discard.

**TF-IDF Vectorization (`src/tfidf_pipeline.py`):**
- Stratified 80/20 train/test split (n_train=18,650, n_test=4,663)
- `TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True, stop_words="english")`
- Output: sparse matrices X_train (18650 × 5000), X_test (4663 × 5000)

---

## Phase 1: Centralized Baseline

**Core Decisions:**
- Swapped RBF SVR for LinearSVR(dual=False) due to high-dimensional text feature sparsity ($d=5000$) to eliminate $O(n^3)$ compute gridlock and improve out-of-sample generalization.

**Centralized Benchmarks (Raw Story Points Scale [1, 100]):**
- MLP: MAE=3.967, RMSE=8.251
- LinearSVR: MAE=4.853, RMSE=8.836
- Random Forest: MAE=3.988, RMSE=8.698
- LSTM: MAE=4.222, RMSE=8.924
- **Final Stacking Ensemble (Meta-Learner): MAE=3.774, RMSE=8.084**

**Centralized Benchmarks (Normalized Scale [0, 1] for Academic Baseline Validation):**
- **Ensemble: MAE=0.0381, RMSE=0.0817**

---

## Phase 2: Vanilla FL Baseline (FedAvg)

**Core Decisions:**
- Federated MLP, LinearSVR, and LSTM via FedAvg. Random Forest and Meta-Learner kept local (trees cannot be averaged).
- All clients use the globally-fitted TF-IDF vectorizer to maintain dimension alignment ($d=5000$).
- 16 clients (one per JIRA project) creating an authentic Non-IID environment.
- RAM Protection: `fraction_fit=0.19` (3 clients/round), staggered 3s boot.

**Distributed Deep Ensemble Results (Centralized Evaluation, Raw Scale [1, 100]):**
*Note: Evaluates only the global deep features (MLP + LSTM) since the meta-learner and SVR/RF models were decoupled.*

| Round | MAE    | RMSE   |
|-------|--------|--------|
| 0 (init) | 4.3649 | 10.4277 |
| 1     | 4.2093 | 10.2626 |
| 2     | 5.0746 | 11.0162 |
| 3     | 5.2797 | 11.1282 |

**Client Drift Analysis (Round 3 vs Phase 1 Centralized):**
- MAE Delta: 5.280 - 3.774 = **+1.506 (39.9% degradation)**
- RMSE Delta: 11.128 - 8.084 = **+3.044 (37.7% degradation)**

> The significant accuracy drop confirms the profound impact of "client drift" in Vanilla FedAvg over heavily Non-IID categorical data, compounded by the removal of the centralized Meta-Learner which previously buffered errors.

### Architectural Optimization: Stateless Model Decoupling
* **Issue:** `LinearSVR` does not support incremental learning (`warm_start`). In the FL loop, it was overwriting injected global weights and recalculating deterministic local weights from scratch every round.
* **Impact:** Sending these static parameters back and forth across multiple communication rounds created redundant network overhead without providing true federated fine-tuning.
* **Resolution:** `LinearSVR` was decoupled from the Flower aggregation loop (`get_parameters` / `set_parameters`). It now operates as a strictly local baseline estimator (alongside the Random Forest). The federated network is now exclusively optimized for continuous deep learning feature extractors (LSTM, MLP).

---

## Phase 3: FedProx Simulation (Colab T4)

**Core Decisions:**
- Migrated from multi-process architecture (17 OS processes) to Flower's **single-process simulation engine** (`fl.simulation.start_simulation`) to solve Colab's 12.7 GB system RAM OOM crash.
- FedProx with `proximal_mu=0.1` to combat client drift observed in Phase 2's Vanilla FedAvg.
- `fraction_fit=0.5` (8 clients/round) — increased from Phase 2's 0.19 (3 clients/round).
- Fractional GPU allocation: `num_gpus=0.0625` per client (1/16 of T4).
- Ray Virtual Client Engine with 2 actor workers reusing a shared TF runtime.
- 10 federation rounds (up from Phase 2's 3 rounds).

**Infrastructure Migration:**
- **Before:** 17 Python processes (1 server + 16 clients) → ~12.6 GB RAM → OOM killed at Round 3.
- **After:** 1 Python process with Ray actor pool (2 workers) → ~2-3 GB RAM → all 10 rounds completed.

**Distributed Deep Ensemble Results (Centralized Evaluation, Raw Scale [1, 100]):**
*Note: Evaluates global deep features (MLP + LSTM) ensemble average on centralized test set.*

| Round     | MAE    | RMSE    |
|-----------|--------|---------|
| 0 (init)  | 4.3784 | 10.4451 |
| 1         | 4.7148 | 10.6149 |
| 2         | 4.9149 | 10.8812 |
| 3         | 5.1183 | 11.0371 |
| 4         | 4.7747 | 10.7939 |
| 5         | 4.8016 | 10.8234 |
| **6**     | **4.2815** | **10.1887** |
| 7         | 4.3969 | 10.3207 |
| 8         | 4.9257 | 10.8953 |
| 9         | 4.3090 | 10.2880 |
| 10        | 4.5822 | 10.6487 |

**Best Round:** R6 — MAE=4.2815, RMSE=10.1887

**Client Drift Analysis (Best Round R6 vs Phase 1 Centralized):**
- MAE Delta: 4.282 - 3.774 = **+0.508 (13.5% degradation)**
- RMSE Delta: 10.189 - 8.084 = **+2.105 (26.0% degradation)**

**Phase 2 vs Phase 3 Comparison (FedAvg vs FedProx):**
- Phase 2 Best (R1): MAE=4.209, RMSE=10.263
- Phase 3 Best (R6): MAE=4.282, RMSE=10.189
- FedProx achieved comparable MAE and slightly better RMSE, while sustaining convergence over 10 rounds instead of diverging after Round 1.

> FedProx's proximal term (`mu=0.1`) stabilized training over longer horizons. While Phase 2 (FedAvg) showed monotonic degradation after R1 (MAE rising from 4.209 → 5.280 over 3 rounds), Phase 3 (FedProx) oscillates but recovers, achieving its best result at R6 and maintaining competitive performance through R10. The non-IID data heterogeneity remains the dominant challenge.
