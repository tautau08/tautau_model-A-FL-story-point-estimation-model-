"""
clean_data.py — Merge, clean, and normalize the raw Deep-SE CSVs.

Pipeline:
  1. Load all per-project CSVs from data/raw/
  2. Standardize columns → [issuekey, title, description, storypoint, project_id]
  3. Drop duplicates on issuekey
  4. Handle nulls (empty description OK, drop null title/storypoint)
  5. Cap storypoints at STORYPOINT_CAP
  6. Create combined 'text' column (title + description), normalize
  7. Save → data/processed/cleaned_dataset.csv

Usage:
    python src/clean_data.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from config import (
    RAW_DATA_DIR,
    CLEANED_DATASET_PATH,
    STORYPOINT_CAP,
    PROJECTS,
)


def load_and_tag_project(csv_path: Path, project_id: str) -> pd.DataFrame:
    """
    Load a single project CSV and normalize its column names.

    Different mirrors use slightly different column names and structures.
    This function handles the common variants.
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        print(f"  [!] Could not load {csv_path.name}: {e}")
        return pd.DataFrame()

    # Normalize column names: lowercase, strip whitespace
    df.columns = df.columns.str.strip().str.lower()

    # Handle common column name variants
    rename_map = {}
    for col in df.columns:
        if col in ("issue_key", "issue key", "key", "issuekey"):
            rename_map[col] = "issuekey"
        elif col in ("title", "summary", "issue_title"):
            rename_map[col] = "title"
        elif col in ("description", "desc", "issue_description"):
            rename_map[col] = "description"
        elif col in ("storypoint", "story_point", "story_points", "storypoints", "sp"):
            rename_map[col] = "storypoint"

    df = df.rename(columns=rename_map)

    # Add project identifier
    df["project_id"] = project_id

    # Keep only the columns we need (if they exist)
    expected = ["issuekey", "title", "description", "storypoint", "project_id"]
    available = [c for c in expected if c in df.columns]
    df = df[available]

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to the merged DataFrame."""

    print(f"\n  Raw merged shape: {df.shape}")

    # --- 1. Drop duplicates on issuekey ---
    before = len(df)
    df = df.drop_duplicates(subset="issuekey", keep="first")
    print(f"  Dropped {before - len(df)} duplicate issuekeys -> {len(df)} rows")

    # --- 2. Handle nulls ---
    # Description can be empty — fill NaN with ""
    df["description"] = df["description"].fillna("")

    # Title and storypoint must exist
    before = len(df)
    df = df.dropna(subset=["title", "storypoint"])
    print(f"  Dropped {before - len(df)} rows with null title/storypoint -> {len(df)} rows")

    # --- 3. Coerce storypoint to numeric, drop non-numeric ---
    df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["storypoint"])
    print(f"  Dropped {before - len(df)} rows with non-numeric storypoint -> {len(df)} rows")

    # --- 4. Cap storypoints ---
    before = len(df)
    df = df[df["storypoint"] <= STORYPOINT_CAP]
    print(f"  Dropped {before - len(df)} rows with SP > {STORYPOINT_CAP} -> {len(df)} rows")

    # Remove zero or negative SP (invalid)
    before = len(df)
    df = df[df["storypoint"] > 0]
    print(f"  Dropped {before - len(df)} rows with SP <= 0 -> {len(df)} rows")

    # --- 5. Create combined text column ---
    df["title"] = df["title"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["text"] = df["title"] + " " + df["description"]

    # --- 6. Normalize text ---
    df["text"] = (
        df["text"]
        .str.lower()
        .apply(lambda x: re.sub(r"\s+", " ", x))  # collapse whitespace
        .str.strip()
    )

    # Drop rows with effectively empty text
    before = len(df)
    df = df[df["text"].str.len() > 3]
    print(f"  Dropped {before - len(df)} rows with empty text -> {len(df)} rows")

    # --- 7. Reset index ---
    df = df.reset_index(drop=True)

    return df


def print_stats(df: pd.DataFrame) -> None:
    """Print summary statistics of the cleaned dataset."""
    print("\n" + "=" * 60)
    print(" Cleaned Dataset Summary")
    print("=" * 60)
    print(f"  Total rows:    {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Columns:       {list(df.columns)}")

    print(f"\n  Story Point Distribution:")
    print(f"    Mean:   {df['storypoint'].mean():.2f}")
    print(f"    Median: {df['storypoint'].median():.1f}")
    print(f"    Std:    {df['storypoint'].std():.2f}")
    print(f"    Min:    {df['storypoint'].min():.0f}")
    print(f"    Max:    {df['storypoint'].max():.0f}")

    print(f"\n  SP Value Counts (top 10):")
    for sp, count in df["storypoint"].value_counts().head(10).items():
        pct = count / len(df) * 100
        print(f"    SP={sp:<5.0f}  ->  {count:>5,} ({pct:5.1f}%)")

    print(f"\n  Per-Project Counts:")
    for proj, count in df["project_id"].value_counts().sort_index().items():
        print(f"    {proj:30s}  ->  {count:>5,} rows")

    print(f"\n  Text length (chars):")
    text_lens = df["text"].str.len()
    print(f"    Mean:   {text_lens.mean():.0f}")
    print(f"    Median: {text_lens.median():.0f}")
    print(f"    Min:    {text_lens.min()}")
    print(f"    Max:    {text_lens.max()}")
    print("=" * 60)


def main():
    print("=" * 60)
    print(" Federated Agile Effort Estimation")
    print(" Data Cleaning Pipeline")
    print("=" * 60)

    # Load all project CSVs
    all_frames = []
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"\n  [X] No CSV files found in {RAW_DATA_DIR}")
        print("    Run `python src/download_data.py` first.")
        sys.exit(1)

    print(f"\n  Found {len(csv_files)} CSV files in {RAW_DATA_DIR}")

    for csv_path in sorted(csv_files):
        project_id = csv_path.stem  # filename without extension
        df_proj = load_and_tag_project(csv_path, project_id)
        if not df_proj.empty:
            print(f"  [+] {project_id:30s} -> {len(df_proj):>5,} rows")
            all_frames.append(df_proj)
        else:
            print(f"  [X] {project_id:30s} -> empty / failed")

    if not all_frames:
        print("\n  [X] No data loaded. Aborting.")
        sys.exit(1)

    # Merge into one DataFrame
    df = pd.concat(all_frames, ignore_index=True)
    print(f"\n  Merged {len(all_frames)} projects -> {len(df):,} total rows")

    # Clean
    df = clean_dataset(df)

    # Save
    df.to_csv(CLEANED_DATASET_PATH, index=False)
    print(f"\n  [+] Saved cleaned dataset -> {CLEANED_DATASET_PATH}")

    # Stats
    print_stats(df)


if __name__ == "__main__":
    main()
