#!/usr/bin/env python3
"""
run_colab.py
============
Google Colab Pro execution launcher for Phase 3 – Federated Learning (FedProx).

Replaces the Windows-only run_phase3.ps1 script with a platform-agnostic Python
launcher that manages subprocesses and respects T4 GPU VRAM limits (16 GB).

Boot sequence (mirrors run_phase3.ps1):
    1. Partition data via src/partition_data.py
    2. Start FL server (src/server_phase3.py) as a background process
    3. Launch 16 FL clients (src/client.py --client_id <i>) with a 2-second
       stagger to prevent simultaneous Keras LSTM CUDA OOM on the T4
    4. Block until the server process exits (10 Flower rounds)
    5. Terminate all remaining subprocesses in a finally block

Usage (in a Colab cell):
    !python run_colab.py
"""

import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PYTHON = sys.executable          # Uses whichever Python is active in Colab
NUM_CLIENTS = 16
SERVER_BOOT_DELAY = 3            # seconds – let gRPC port open
CLIENT_STAGGER_DELAY = 2         # seconds – sequential VRAM allocation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Timestamped console log."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}]  {msg}", flush=True)


def _run_step(script: str, label: str) -> None:
    """Run a blocking subprocess; raise on failure."""
    _log(f"{label} ...")
    result = subprocess.run([PYTHON, script], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit code {result.returncode})")
    _log(f"{label} – done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 60)
    print("  Phase 3 — Federated Learning Orchestration (FedProx)")
    print("  Runtime: Google Colab Pro  •  GPU: T4 16 GB VRAM")
    print("=" * 60)
    print()

    # Step 1 – Partition data (blocking) --------------------------------
    _run_step("src/partition_data.py", "[Step 1] Partitioning data by project")

    server_proc = None
    client_procs: list[subprocess.Popen] = []

    try:
        # Step 2 – Start FL server (background) -------------------------
        _log("[Step 2] Starting FL server (port 8080) ...")
        server_proc = subprocess.Popen(
            [PYTHON, "src/server_phase3.py"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        _log(f"         Server PID: {server_proc.pid}")
        time.sleep(SERVER_BOOT_DELAY)

        # Step 3 – Launch 16 clients with VRAM-safe stagger -------------
        _log(f"[Step 3] Launching {NUM_CLIENTS} FL clients "
             f"({CLIENT_STAGGER_DELAY}s stagger) ...")
        for i in range(NUM_CLIENTS):
            proc = subprocess.Popen(
                [PYTHON, "src/client.py", "--client_id", str(i)],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            client_procs.append(proc)
            _log(f"         Client {i:>2d} launched  (PID {proc.pid})")
            if i < NUM_CLIENTS - 1:          # no sleep after the last client
                time.sleep(CLIENT_STAGGER_DELAY)

        # Step 4 – Wait for federation to finish ------------------------
        print()
        _log("[Step 4] All clients connected. Waiting for 10 federation "
             "rounds ...")
        print()
        server_proc.wait()
        _log("Federation finished!")

    finally:
        # Cleanup – terminate every subprocess to avoid Colab zombies ----
        print()
        _log("Cleaning up background processes ...")
        stopped = 0

        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            stopped += 1

        for proc in client_procs:
            if proc.poll() is None:
                proc.terminate()
                stopped += 1

        _log(f"Stopped {stopped} remaining process(es).")
        print()
        print("=" * 60)
        print("  Done.")
        print("=" * 60)


if __name__ == "__main__":
    main()
