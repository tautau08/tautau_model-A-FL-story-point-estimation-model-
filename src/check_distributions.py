"""
check_distributions.py -- Stage B EDA artifact (see plan: i-need-a-proper-jaunty-moore)

Quantifies the per-project story-point distribution to diagnose which clients
are responsible for the Weighted-MAE inflation seen in Phase 4 evaluation
(models/phase4_personalized/phase4_evaluation_all.json).

For each of the 16 projects, computes: n, mean, std, skewness, kurtosis,
min/max, unique SP count, and % of rows with SP > 8 (the Fibonacci-core cutoff
used elsewhere in this project, e.g. worktracker.md).

Outputs:
  - reports/distribution_summary.csv   (one row per project, sortable)
  - reports/distribution_hist_<project>.png  (one histogram per project)
  - reports/distribution_overview.png  (small-multiples grid, all 16 projects)

Run: python src/check_distributions.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CLEANED_DATASET_PATH, PROJECTS

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIBONACCI_CUTOFF = 8  # SP > 8 counted as "high-effort outlier" per worktracker.md


def compute_project_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for project in PROJECTS:
        sp = df.loc[df["project_id"] == project, "storypoint"].dropna().to_numpy(dtype=float)
        if len(sp) == 0:
            continue
        rows.append({
            "project": project,
            "n": len(sp),
            "mean": np.mean(sp),
            "std": np.std(sp, ddof=1) if len(sp) > 1 else 0.0,
            "skew": stats.skew(sp),
            "kurtosis": stats.kurtosis(sp),
            "min": np.min(sp),
            "max": np.max(sp),
            "median": np.median(sp),
            "n_unique_sp": len(np.unique(sp)),
            "pct_sp_gt_8": 100.0 * np.mean(sp > FIBONACCI_CUTOFF),
        })
    out = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return out


def plot_overview(df: pd.DataFrame, sp_by_project: dict) -> None:
    n_projects = len(PROJECTS)
    ncols = 4
    nrows = int(np.ceil(n_projects / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6))
    axes = axes.flatten()

    for i, project in enumerate(PROJECTS):
        ax = axes[i]
        sp = sp_by_project.get(project)
        if sp is None or len(sp) == 0:
            ax.set_visible(False)
            continue
        ax.hist(sp, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.set_title(project, fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(n_projects, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Project Story Point Distributions (16 clients)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = REPORTS_DIR / "distribution_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_individual(project: str, sp: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(sp, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_title(f"{project} (n={len(sp)})")
    ax.set_xlabel("Story Point")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out_path = REPORTS_DIR / f"distribution_hist_{project}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(" Per-Project Story Point Distribution Check")
    print(" (diagnoses which clients drive Weighted-MAE inflation in Phase 4)")
    print("=" * 78)

    if not CLEANED_DATASET_PATH.exists():
        print(f"\n[!] {CLEANED_DATASET_PATH} not found. Run src/clean_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CLEANED_DATASET_PATH)
    summary = compute_project_stats(df)

    sp_by_project = {
        project: df.loc[df["project_id"] == project, "storypoint"].dropna().to_numpy(dtype=float)
        for project in PROJECTS
    }

    csv_path = REPORTS_DIR / "distribution_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n[+] Summary table saved to: {csv_path}\n")

    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 140):
        print(summary.to_string(index=False))

    print("\n[+] Generating histograms...")
    plot_overview(summary, sp_by_project)
    for project, sp in sp_by_project.items():
        if len(sp) > 0:
            plot_individual(project, sp)
    print(f"  Saved {len(sp_by_project)} individual histograms to: {REPORTS_DIR}")

    # Explicit outlier callout, mirroring the Phase 4 per-client MAE ranking
    print("\n[+] Outlier callout (matches Phase 4 per-client MAE ranking):")
    worst = summary.sort_values("pct_sp_gt_8", ascending=False).head(3)
    best = summary.sort_values("pct_sp_gt_8", ascending=True).head(3)
    print("\n  Highest %% SP>8 (near-continuous, hour-like scale -- hardest to estimate):")
    print(worst[["project", "n", "mean", "std", "n_unique_sp", "pct_sp_gt_8"]].to_string(index=False))
    print("\n  Lowest %% SP>8 (well-behaved, close to pure Fibonacci scale):")
    print(best[["project", "n", "mean", "std", "n_unique_sp", "pct_sp_gt_8"]].to_string(index=False))
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
