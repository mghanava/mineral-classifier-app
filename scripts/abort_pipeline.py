"""Abort a running DVC pipeline while leaving the dashboard container up.

Scans /proc for pipeline processes (dvc repro, stage scripts, and the
dvc.yaml generator), sends SIGINT for a clean stop, and escalates to
SIGKILL after a short grace period if any process is still alive.
"""

import os
import signal
import time

SIGINT_GRACE_SECONDS = 3
STAGE_PATTERNS = ["dvc repro", "src/stages/", "setup_dvc.py"]


def find_pipeline_processes():
    """Return (pid, cmdline) pairs for running pipeline processes."""
    matches = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid() or int(pid) == 1:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode(errors="ignore")
        except OSError:
            continue
        if any(pattern in cmdline for pattern in STAGE_PATTERNS):
            matches.append((int(pid), cmdline.strip()))
    return matches


def signal_processes(sig, grace=0.0):
    """Send a signal to all pipeline processes, skipping already-dead ones."""
    for pid, cmdline in find_pipeline_processes():
        try:
            os.kill(pid, sig)
            print(f"Sent {signal.Signals(sig).name} to PID {pid}: {cmdline}")
        except (ProcessLookupError, PermissionError):
            pass
    if grace:
        time.sleep(grace)


def main():
    """Find running pipeline processes and stop them."""
    processes = find_pipeline_processes()
    if not processes:
        print("No running pipeline processes found.")
        return
    print("Stopping pipeline processes:")
    for pid, cmdline in processes:
        print(f"  PID {pid}: {cmdline}")
    signal_processes(signal.SIGINT, SIGINT_GRACE_SECONDS)
    signal_processes(signal.SIGKILL)


if __name__ == "__main__":
    main()
