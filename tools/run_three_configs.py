#!/usr/bin/env python3
"""Train and benchmark the three supplied power-grid configurations."""
import json
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "three_config_runs"
CONFIGS = [
    {
        "name": "balanced",
        "hidden_size": 512, "num_layers": 1, "total_timesteps": 49_999_999,
        "horizon": 1024, "learning_rate": 0.0024955493,
        "ent_coef": 0.0491252503, "gamma": 0.9998925467,
        "gae_lambda": 0.9862453023, "replay_ratio": 1.8938161622,
        "clip_coef": 0.9167692184, "vf_clip_coef": 0.01,
        "vf_coef": 0.1, "max_grad_norm": 5.0, "beta1": 0.999,
        "beta2": 0.9900769490, "eps": 0.0000613270,
        "prio_alpha": 0.1935883133, "prio_beta0": 1.0,
        "failure_reward": -0.8632541950, "safe_step_reward": 0.0042193436,
        "switch_penalty": 0.0010496779, "congestion_cost_weight": 2.4561659097,
    },
    {
        "name": "second",
        "hidden_size": 512, "num_layers": 5, "total_timesteps": 54_246_748,
        "horizon": 512, "learning_rate": 0.0031997989,
        "ent_coef": 0.0554992588, "gamma": 0.9706140035,
        "gae_lambda": 0.9863009165, "replay_ratio": 2.6568749259,
        "clip_coef": 0.9663823993, "vf_clip_coef": 0.01,
        "vf_coef": 0.9985455035, "max_grad_norm": 5.0,
        "beta1": 0.9987949157, "beta2": 0.9995411215,
        "eps": 4.9112767e-11, "prio_alpha": 0.0440745105,
        "prio_beta0": 0.8994538930, "failure_reward": -0.9040679051,
        "safe_step_reward": 0.0061968221, "switch_penalty": 0.0028395687,
        "congestion_cost_weight": 1.8477378314,
    },
    {
        "name": "third",
        "hidden_size": 512, "num_layers": 3, "total_timesteps": 100_000_000,
        "horizon": 1024, "learning_rate": 0.0070666478,
        "ent_coef": 0.0270939793, "gamma": 0.9463051974,
        "gae_lambda": 0.9881208596, "replay_ratio": 1.9776114696,
        "clip_coef": 0.7116052813, "vf_clip_coef": 0.2502011190,
        "vf_coef": 0.6147175388, "max_grad_norm": 4.9779631263,
        "beta1": 0.999, "beta2": 0.9942211208, "eps": 0.0001,
        "prio_alpha": 0.2136785878, "prio_beta0": 0.3790478363,
        "failure_reward": -0.8311622752, "safe_step_reward": 0.0053052601,
        "switch_penalty": 0.0010772130, "congestion_cost_weight": 2.2354862219,
    },
]

TRAIN_KEYS = (
    "total_timesteps", "horizon", "learning_rate", "ent_coef", "gamma",
    "gae_lambda", "replay_ratio", "clip_coef", "vf_clip_coef", "vf_coef",
    "max_grad_norm", "beta1", "beta2", "eps", "prio_alpha", "prio_beta0",
)
ENV_KEYS = ("failure_reward", "safe_step_reward", "switch_penalty", "congestion_cost_weight")


def main():
    RUNS.mkdir(exist_ok=True)
    results = RUNS / "results.jsonl"
    with results.open("a", encoding="utf-8") as output:
        for index, config in enumerate(CONFIGS):
            directory = RUNS / config["name"]
            directory.mkdir(exist_ok=True)
            checkpoint_dir = directory / "checkpoint"
            command = [
                ".venv/bin/puffer", "train", "power_grid", "--seed", str(7000 + index),
                "--policy.hidden-size", str(config["hidden_size"]),
                "--policy.num-layers", str(config["num_layers"]),
            ]
            for key in TRAIN_KEYS:
                command += [f"--train.{key.replace('_', '-')}", str(config[key])]
            for key in ENV_KEYS:
                command += [f"--env.{key.replace('_', '-')}", str(config[key])]
            command += ["--checkpoint-dir", str(checkpoint_dir)]
            started = time.time()
            with (directory / "train.log").open("w", encoding="utf-8") as log:
                train = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            checkpoints = sorted(directory.rglob("*.bin"), key=lambda p: p.stat().st_mtime)
            row = {"name": config["name"], "train_exit": train.returncode,
                   "seconds": time.time() - started}
            if train.returncode == 0 and checkpoints:
                bench = subprocess.run([
                    ".venv/bin/python", "ocean/power_grid/benchmark.py",
                    "--checkpoint", str(checkpoints[-1]), "--episodes", "1024",
                    "--physics", "dc", "--random-event-probability", "0.25",
                    "--random-outages", "3", "--jobs", "4",
                ], cwd=ROOT, capture_output=True, text=True)
                match = re.search(r"PPO agent\s+\d+\s+([0-9.]+)\s+([0-9.]+)", bench.stdout)
                row["perf"] = float(match.group(1)) if match else None
                row["score"] = float(match.group(2)) if match else None
                row["checkpoint"] = str(checkpoints[-1])
                row["benchmark_exit"] = bench.returncode
                (directory / "benchmark.log").write_text(bench.stdout, encoding="utf-8")
            else:
                row["perf"] = None
                row["score"] = None
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
