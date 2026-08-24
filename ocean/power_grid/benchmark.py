#!/usr/bin/env python3
"""Compare a trained Power Grid policy with deterministic C baselines."""

import argparse
import concurrent.futures
import csv
import io
import os
import pathlib
import subprocess
import sys
import tempfile

from pufferlib import pufferl
import pufferlib._C as backend

from evaluate import replay, resolve_checkpoint


ROOT = pathlib.Path(__file__).resolve().parents[2]


BASELINE_COUNT = 7
METRIC_COLUMNS = (
    "perf", "score", "total_failure", "event_failure", "total_switches",
    "random_events", "demand_fulfilled", "outage_completion",
    "all_outages_survived", "thermal_trips", "thermal_trip_episode",
    "peak_thermal_stress", "peak_line_loading", "overloaded_line_fraction",
)


def _episode_shards(episodes, shards):
    quotient, remainder = divmod(episodes, shards)
    offset = 0
    for shard in range(shards):
        count = quotient + (shard < remainder)
        yield offset, count
        offset += count


def run_baselines(
    episodes, ac_power_flow, random_event_probability, random_outages, jobs,
):
    source = ROOT / "ocean" / "power_grid" / "benchmark.c"
    with tempfile.TemporaryDirectory() as directory:
        executable = pathlib.Path(directory) / "power_grid_benchmark"
        subprocess.run(
            [
                "clang", "-std=c11", "-O3", "-march=native", "-Wall", "-Wextra",
                "-Werror", "-pedantic", f"-I{source.parent}", str(source), "-lm",
                "-o", str(executable),
            ],
            check=True,
            cwd=ROOT,
        )
        shards = min(jobs, episodes)
        tasks = [
            (policy, offset, count)
            for policy in range(BASELINE_COUNT)
            for offset, count in _episode_shards(episodes, shards)
        ]

        def run_task(task):
            policy, offset, count = task
            output = subprocess.run(
                [
                    str(executable), str(count), str(int(ac_power_flow)),
                    str(random_event_probability), str(random_outages),
                    str(policy), str(offset),
                ], check=True, cwd=ROOT, capture_output=True, text=True,
            ).stdout
            return next(csv.DictReader(io.StringIO(output)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            partial_rows = list(executor.map(run_task, tasks))

    rows = []
    for policy in range(BASELINE_COUNT):
        chunks = partial_rows[policy * shards:(policy + 1) * shards]
        row = {"policy": chunks[0]["policy"], "episodes": str(episodes)}
        for metric in METRIC_COLUMNS:
            row[metric] = sum(
                int(chunk["episodes"]) * float(chunk[metric]) for chunk in chunks
            ) / episodes
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare a trained policy with held-out seeded baselines"
    )
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--physics", choices=("dc", "ac"), default="dc")
    parser.add_argument("--random-event-probability", type=float, default=0.50)
    parser.add_argument("--random-outages", type=int, default=3)
    parser.add_argument(
        "--jobs", type=int, default=min(8, os.cpu_count() or 1),
        help="parallel deterministic baseline workers (default: up to 8)",
    )
    parser.add_argument(
        "--ppo-only", action="store_true",
        help="skip invariant C baselines while screening PPO checkpoints",
    )
    options = parser.parse_args()
    if options.episodes < 2 or options.episodes % 2:
        parser.error("--episodes must be a positive multiple of two")
    if not 0.0 <= options.random_event_probability <= 1.0:
        parser.error("--random-event-probability must be between zero and one")
    if not 1 <= options.random_outages <= 8:
        parser.error("--random-outages must be between one and eight")
    if options.jobs < 1:
        parser.error("--jobs must be positive")
    if backend.env_name != "power_grid":
        parser.error("run ./build.sh power_grid before benchmarking")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        args = pufferl.load_config("power_grid")
    finally:
        sys.argv = original_argv

    checkpoint = resolve_checkpoint(options.checkpoint)
    ac_power_flow = options.physics == "ac"
    learned = replay(
        args, checkpoint, ac_power_flow, 1, options.episodes, horizon=72,
        random_event_probability=options.random_event_probability,
        random_outage_count=options.random_outages,
    )
    rows = [{
        "policy": "PPO agent",
        "episodes": str(int(learned["n"])),
        "perf": learned["perf"],
        "score": learned["score"],
        "total_failure": learned["total_failure"],
        "event_failure": learned["event_failure"],
        "total_switches": learned["total_switches"],
        "random_events": learned["random_events"],
        "demand_fulfilled": learned["demand_fulfilled"],
        "outage_completion": learned["outage_completion"],
        "all_outages_survived": learned["all_outages_survived"],
        "thermal_trips": learned["thermal_trips"],
        "thermal_trip_episode": learned["thermal_trip_episode"],
        "peak_thermal_stress": learned["peak_thermal_stress"],
        "peak_line_loading": learned["peak_line_loading"],
        "overloaded_line_fraction": learned["overloaded_line_fraction"],
    }]
    if not options.ppo_only:
        rows.extend(run_baselines(
            options.episodes, ac_power_flow, options.random_event_probability,
            options.random_outages, options.jobs,
        ))

    print(f"Checkpoint: {checkpoint}")
    print(
        f"Held-out 2020 scenarios; {options.physics.upper()} physics; "
        f"random event probability {options.random_event_probability:.2f}; "
        f"{options.random_outages} outages per event episode; "
        f"environment seeds 0..{options.episodes - 1}"
    )
    print(
        f"{'Policy':18s} {'N':>6s} {'Perf':>8s} {'Score':>8s} "
        f"{'Fail':>8s} {'EvtFail':>8s} {'Switches':>10s} {'Events':>8s}"
    )
    print("-" * 84)
    for row in rows:
        print(
            f"{row['policy']:18s} {int(row['episodes']):6d} "
            f"{float(row['perf']):8.4f} {float(row['score']):8.4f} "
            f"{float(row['total_failure']):8.4f} "
            f"{float(row['event_failure']):8.4f} {float(row['total_switches']):10.3f} "
            f"{float(row['random_events']):8.4f}"
        )
    print()
    print(
        f"{'Policy':18s} {'Demand%':>9s} {'Outage%':>9s} {'Survive%':>9s} "
        f"{'TripEp%':>9s} {'Trips':>8s} {'PeakStress':>11s} {'PeakRho':>9s} "
        f"{'OLLine%':>9s}"
    )
    print("-" * 105)
    for row in rows:
        print(
            f"{row['policy']:18s} {100 * float(row['demand_fulfilled']):9.2f} "
            f"{100 * float(row['outage_completion']):9.2f} "
            f"{100 * float(row['all_outages_survived']):9.2f} "
            f"{100 * float(row['thermal_trip_episode']):9.2f} "
            f"{float(row['thermal_trips']):8.3f} "
            f"{float(row['peak_thermal_stress']):11.3f} "
            f"{float(row['peak_line_loading']):9.3f} "
            f"{100 * float(row['overloaded_line_fraction']):9.3f}"
        )


if __name__ == "__main__":
    main()
