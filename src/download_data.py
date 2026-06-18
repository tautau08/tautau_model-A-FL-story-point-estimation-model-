"""
download_data.py — Fetch the 16-project Deep-SE benchmark dataset.

Downloads per-project CSV files from GitHub (with fallback URLs) into
data/raw/. Idempotent: skips files that already exist.

Usage:
    python src/download_data.py
"""

import sys
from pathlib import Path

# Allow running from project root: `python src/download_data.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from config import PROJECTS, DATASET_URL_TEMPLATES, RAW_DATA_DIR


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a single file. Returns True on success."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 100:
            dest.write_bytes(resp.content)
            return True
    except requests.RequestException:
        pass
    return False


def download_all_datasets() -> dict:
    """
    Download all 16 project CSVs, trying each URL template in order.

    Returns a summary dict: {project: "downloaded" | "skipped" | "FAILED"}.
    """
    summary = {}

    for project in PROJECTS:
        dest = RAW_DATA_DIR / f"{project}.csv"

        # Skip if already downloaded
        if dest.exists() and dest.stat().st_size > 100:
            summary[project] = "skipped (exists)"
            continue

        # Try each URL template
        downloaded = False
        for template in DATASET_URL_TEMPLATES:
            url = template.format(project=project)
            print(f"  Trying: {url} ... ", end="", flush=True)
            if download_file(url, dest):
                print("OK")
                summary[project] = "downloaded"
                downloaded = True
                break
            else:
                print("FAIL")

        if not downloaded:
            summary[project] = "FAILED"

    return summary


def print_summary(summary: dict) -> None:
    """Pretty-print the download summary."""
    downloaded = sum(1 for v in summary.values() if v == "downloaded")
    skipped    = sum(1 for v in summary.values() if "skipped" in v)
    failed     = sum(1 for v in summary.values() if v == "FAILED")

    print("\n" + "=" * 55)
    print(" Download Summary")
    print("=" * 55)
    for project, status in summary.items():
        icon = "+" if "downloaded" in status or "skipped" in status else "X"
        print(f"  {icon}  {project:30s} {status}")
    print("-" * 55)
    print(f"  Downloaded: {downloaded}  |  Skipped: {skipped}  |  Failed: {failed}")
    print("=" * 55)

    if failed > 0:
        print("\n[!] Some downloads failed. You can manually place the CSVs in:")
        print(f"   {RAW_DATA_DIR}")
        print("   Expected columns: issuekey, title, description, storypoint")
        print("   Source: https://github.com/morakotch/datasets/tree/master/storypoint")


if __name__ == "__main__":
    print("=" * 55)
    print(" Federated Agile Effort Estimation")
    print(" Dataset Downloader (Deep-SE Benchmark)")
    print("=" * 55)
    print(f"\nTarget directory: {RAW_DATA_DIR}")
    print(f"Projects to fetch: {len(PROJECTS)}\n")

    summary = download_all_datasets()
    print_summary(summary)
