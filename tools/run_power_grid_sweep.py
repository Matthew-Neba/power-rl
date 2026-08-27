#!/usr/bin/env python3
"""Robust sequential local sweep for power_grid.

The built-in sweep manager can lose its result queue when a worker exits. This
runner keeps one trial in the foreground, preserving every checkpoint and
benchmark result independently.
"""

import json
import math
import pathlib
import random
import re
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "sweep800_checkpoints"
RESULTS = ROOT / "sweep800_results.jsonl"
TRIALS = 800
STEPS = 20_000_000


def log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def trial_params(rng):
    return {
        "learning_rate": log_uniform(rng, 5e-4, 5e-3),
        "ent_coef": log_uniform(rng, 1e-5, 3e-3),
        "switch_penalty": log_uniform(rng, 3e-3, 3e-2),
        "gamma": rng.uniform(0.95, 0.999),
        "gae_lambda": rng.uniform(0.90, 0.99),
    }


def main():
    rng = random.Random(20260811)
    OUT.mkdir(exist_ok=True)
    completed = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                completed.add(int(json.loads(line)["trial"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
    with RESULTS.open("a", encoding="utf-8") as results:
        for trial in range(TRIALS):
            params = trial_params(rng)
            if trial in completed:
                continue
            trial_dir = OUT / f"trial_{trial:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            train_log = trial_dir / "train.log"
            command = [
                "puffer", "train", "power_grid",
                "--seed", str(10000 + trial),
                "--train.total-timesteps", str(STEPS),
                "--train.learning-rate", f"{params['learning_rate']:.12g}",
                "--train.ent-coef", f"{params['ent_coef']:.12g}",
                "--train.gamma", f"{params['gamma']:.12g}",
                "--train.gae-lambda", f"{params['gae_lambda']:.12g}",
                "--env.switch-penalty", f"{params['switch_penalty']:.12g}",
                "--checkpoint-dir", str(trial_dir),
            ]
            started = time.time()
            with train_log.open("w", encoding="utf-8") as log:
                train = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT)
            checkpoints = sorted(trial_dir.rglob("*.bin"),
                                 key=lambda path: path.stat().st_mtime)
            row = {"trial": trial, **params,
                   "train_exit": train.returncode,
                   "seconds": time.time() - started}
            if train.returncode == 0 and checkpoints:
                checkpoint = checkpoints[-1]
                bench = subprocess.run(
                    [sys.executable, "ocean/power_grid/benchmark.py",
                     "--checkpoint", str(checkpoint), "--episodes", "1024",
                     "--physics", "dc", "--random-event-probability", "0.25",
                     "--random-outages", "3", "--jobs", "4", "--ppo-only"],
                    cwd=ROOT, capture_output=True, text=True,
                )
                match = re.search(r"PPO agent\s+\d+\s+([0-9.]+)", bench.stdout)
                row["perf"] = float(match.group(1)) if match else None
                row["checkpoint"] = str(checkpoint)
                row["benchmark_exit"] = bench.returncode
            else:
                row["perf"] = None
            results.write(json.dumps(row) + "\n")
            results.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
