#!/usr/bin/env python3
"""
run_colab.py
============
Google Colab execution launcher for Phase 3 – Federated Learning (FedProx).

Runs the single-process Flower simulation engine, which manages all 16
virtual clients within one Python process to stay within Colab's 12.7 GB
system RAM limit.

Previous architecture (17 processes):  ~12.6 GB RAM — OOM killed
Current architecture  (1 process):     ~2-3 GB RAM — safe

Usage (in a Colab cell):
    !python run_colab.py
"""

import subprocess
import sys
import time

PYTHON = sys.executable


def _log(msg: str) -> None:
    """Timestamped console log."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}]  {msg}", flush=True)


def main() -> None:
    print()
    print("=" * 60)
    print("  Phase 3 — Federated Learning Simulation (FedProx)")
    print("  Runtime: Google Colab  •  Mode: Single-process")
    print("=" * 60)
    print()

    _log("Launching Flower simulation (src/simulate_phase3.py) ...")
    print()

    result = subprocess.run([PYTHON, "src/simulate_phase3.py"])

    print()
    if result.returncode == 0:
        _log("Simulation completed successfully.")
    else:
        _log(f"Simulation failed (exit code {result.returncode}).")
        sys.exit(result.returncode)

    print()
    print("=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
