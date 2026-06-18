# Worktracker

This file serves as our empirical ledger for the final paper. It documents the metrics and architectural milestones across all phases of the Federated Agile Effort Estimation project.

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
