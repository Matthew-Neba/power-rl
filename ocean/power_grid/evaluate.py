#!/usr/bin/env python3
"""Headless, apples-to-apples DC versus AC checkpoint replay."""

import argparse
import copy
import glob
import os
import sys

from pufferlib import pufferl
import pufferlib._C as backend


METRICS = (
    "perf",
    "score",
    "episode_return",
    "total_failure",
    "topology_failure",
    "solver_failure",
    "total_switches",
    "ac_voltage_violation_steps",
    "ac_generator_p_violation_steps",
    "ac_q_limit_events",
    "ac_mean_active_loss_mw",
    "ac_nonconvergence",
    "ac_thermal_trips",
    "ac_peak_thermal_stress",
    "maintenance_events",
    "n",
)


def resolve_checkpoint(path):
    if path != "latest":
        return path
    candidates = glob.glob("checkpoints/power_grid/**/*.bin", recursive=True)
    if not candidates:
        raise FileNotFoundError("No power_grid checkpoint exists under checkpoints/power_grid")
    return max(candidates, key=os.path.getctime)


def replay(base_args, checkpoint, ac_power_flow, rollouts, agents):
    args = copy.deepcopy(base_args)
    args["env"]["ac_power_flow"] = ac_power_flow
    args["env"]["evaluation_scenarios"] = True
    args["vec"]["total_agents"] = agents
    args["reset_state"] = False
    args["train"]["horizon"] = 48
    evaluator = backend.create_pufferl(args)
    try:
        backend.load_weights(evaluator, checkpoint)
        for _ in range(rollouts):
            backend.rollouts(evaluator)
        metrics = backend.eval_log(evaluator).get("env", {})
        if not metrics:
            raise RuntimeError("No completed episodes were logged; increase --rollouts")
        return {name: metrics[name] for name in METRICS}
    finally:
        backend.close(evaluator)


def main():
    parser = argparse.ArgumentParser(
        description="Replay one trained topology policy under both DC and AC physics"
    )
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--agents", type=int, default=1024)
    options = parser.parse_args()
    if backend.env_name != "power_grid":
        parser.error("PufferLib extension is not built for power_grid; run ./build.sh power_grid")
    if options.rollouts < 2:
        parser.error("--rollouts must be at least 2 so complete episodes are logged")
    if options.agents < 2 or options.agents % 2:
        parser.error("--agents must be a positive multiple of the default two buffers")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        base_args = pufferl.load_config("power_grid")
    finally:
        sys.argv = original_argv
    checkpoint = resolve_checkpoint(options.checkpoint)
    print(f"Checkpoint: {checkpoint}")
    results = {
        "DC": replay(base_args, checkpoint, False, options.rollouts, options.agents),
        "AC": replay(base_args, checkpoint, True, options.rollouts, options.agents),
    }
    print(f"{'Metric':34s} {'DC':>13s} {'AC':>13s}")
    print("-" * 62)
    for metric in METRICS:
        print(f"{metric:34s} {results['DC'][metric]:13.6f} {results['AC'][metric]:13.6f}")


if __name__ == "__main__":
    main()
