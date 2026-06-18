"""
partition_data.py -- Partition the cleaned dataset into client-specific splits for FL.

Reads `data/processed/cleaned_dataset.csv`.
Groups rows by `project_id`.
Creates `data/federated/client_{id}/train.csv` and `test.csv` using 80/20
stratified split (using storypoints).

Usage:
    python src/partition_data.py
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CLEANED_DATASET_PATH,
    FEDERATED_DATA_DIR,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.tfidf_pipeline import create_stratification_bins


def main():
    print("=" * 60)
    print(" Federated Agile Effort Estimation")
    print(" Phase 2 -- Data Partitioning (Non-IID by Project)")
    print("=" * 60)

    if not CLEANED_DATASET_PATH.exists():
        print(f"  [X] Missing: {CLEANED_DATASET_PATH}")
        sys.exit(1)

    print(f"\n  Loading cleaned dataset...")
    df = pd.read_csv(CLEANED_DATASET_PATH)

    projects = sorted(df["project_id"].unique())
    print(f"  Found {len(projects)} unique projects. Creating clients...")

    # Clear existing federated directory if it exists
    if FEDERATED_DATA_DIR.exists():
        import shutil
        shutil.rmtree(FEDERATED_DATA_DIR)
    
    FEDERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    client_mapping = []

    for i, project in enumerate(projects):
        client_id = str(i)
        client_dir = FEDERATED_DATA_DIR / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)

        df_proj = df[df["project_id"] == project].copy()
        
        # We need to stratify. If a project is very small, stratify might fail.
        # Fall back to unstratified if needed.
        try:
            strata = create_stratification_bins(df_proj["storypoint"])
            train_df, test_df = train_test_split(
                df_proj, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata
            )
        except ValueError:
            print(f"    [!] Project {project} too small/imbalanced for stratification. Falling back to random split.")
            train_df, test_df = train_test_split(
                df_proj, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

        train_path = client_dir / "train.csv"
        test_path = client_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"    Client {client_id:<2s} ({project:<20s}) -> Train: {len(train_df):>4d}, Test: {len(test_df):>4d}")
        client_mapping.append({"client_id": client_id, "project": project, "train_size": len(train_df), "test_size": len(test_df)})

    # Save mapping for reference
    pd.DataFrame(client_mapping).to_csv(FEDERATED_DATA_DIR / "client_mapping.csv", index=False)
    
    print("\n  [+] Partitioning complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
