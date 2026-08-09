#!/usr/bin/env python3
"""Generate the compile-time IEEE-118 network data used by power_grid.

The canonical MATPOWER case deliberately leaves branch ratings unlimited. This
builder assigns 225 MVA to 138/161 kV corridors and 550 MVA to 345 kV corridors
and transformers. These conservative synthetic voltage-class ratings are inspired
by MATPOWER's c118swf study case while retaining the canonical 186-branch topology.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "ocean/power_grid/power_grid_ieee118_data.h"
SOURCE_URL = "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case118.m"
SOURCE_SHA256 = "bc2e6f22b4b9e776572885ee4b50e4f4ab2ee0c5577e9126e86d906f14c4b5f7"


def matrix(source: str, name: str) -> list[list[float]]:
    match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", source, re.DOTALL)
    if match is None:
        raise ValueError(f"MATPOWER matrix {name!r} not found")
    rows = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("%", 1)[0].strip().removesuffix(";")
        if line:
            rows.append([float(value) for value in line.split()])
    return rows


def c_number(value: float) -> str:
    rendered = f"{value:.12g}"
    return rendered if "." in rendered or "e" in rendered.lower() else rendered + ".0"


def render_array(name: str, c_type: str, values, width: int = 6) -> list[str]:
    result = [f"static const {c_type} {name}[{len(values)}] = {{"]
    for start in range(0, len(values), width):
        result.append("    " + ", ".join(str(value) for value in values[start:start + width]) + ",")
    result.append("};")
    return result


def bridge_indexes(branches: list[list[float]], buses: int) -> set[int]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(buses)]
    for index, branch in enumerate(branches):
        start, end = int(branch[0]) - 1, int(branch[1]) - 1
        adjacency[start].append((end, index))
        adjacency[end].append((start, index))
    bridges = set()
    for removed, branch in enumerate(branches):
        start, target = int(branch[0]) - 1, int(branch[1]) - 1
        visited = {start}
        pending = [start]
        while pending:
            node = pending.pop()
            for neighbor, edge in adjacency[node]:
                if edge != removed and neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        if target not in visited:
            bridges.add(removed)
    return bridges


def build(source: bytes) -> str:
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"unexpected case118.m SHA-256 {digest}; expected {SOURCE_SHA256}")
    text = source.decode()
    buses = matrix(text, "bus")
    generators = matrix(text, "gen")
    branches = matrix(text, "branch")
    if (len(buses), len(generators), len(branches)) != (118, 54, 186):
        raise ValueError("unexpected IEEE-118 dimensions")

    slack_bus = next(int(bus[0]) for bus in buses if int(bus[1]) == 3)
    generators.sort(key=lambda generator: int(generator[0]) != slack_bus)
    loads = [bus for bus in buses if bus[2] != 0.0 or bus[3] != 0.0]
    bridges = bridge_indexes(branches, len(buses))

    branch_rows = []
    branch_r = []
    branch_b = []
    branch_names = []
    eligible = []
    for index, branch in enumerate(branches):
        start, end = int(branch[0]) - 1, int(branch[1]) - 1
        tap = branch[8] if branch[8] else 1.0
        high_voltage = max(buses[start][9], buses[end][9]) >= 300.0
        rating = 550.0 if high_voltage else 225.0
        branch_rows.append(
            "{" + ", ".join((str(start), str(end), c_number(branch[3]),
                              c_number(tap), c_number(rating))) + "}"
        )
        branch_r.append(c_number(branch[2]))
        branch_b.append(c_number(branch[4]))
        branch_names.append(f'"{start + 1}-{end + 1}"')
        eligible.append("0" if index in bridges else "1")

    generator_buses = [str(int(generator[0]) - 1) for generator in generators]
    generator_p = [c_number(generator[1]) for generator in generators]
    generator_q_max = [c_number(generator[3]) for generator in generators]
    generator_q_min = [c_number(generator[4]) for generator in generators]
    generator_v = [c_number(generator[5]) for generator in generators]
    generator_p_max = [c_number(generator[8]) for generator in generators]
    load_buses = [str(int(load[0]) - 1) for load in loads]
    load_p = [c_number(load[2]) for load in loads]
    load_q = [c_number(load[3]) for load in loads]
    bus_shunt_b = [c_number(bus[5]) for bus in buses]
    bus_base_kv = [c_number(bus[9]) for bus in buses]
    bus_v_reference = [c_number(bus[7]) for bus in buses]
    bus_angle_reference_deg = [c_number(bus[8]) for bus in buses]

    lines = [
        "/* Generated by build_ieee118_case.py; do not edit manually.",
        f" * MATPOWER case118.m SHA256: {digest}",
        " * Canonical topology with documented synthetic thermal ratings.",
        " */",
        "#ifndef POWER_GRID_IEEE118_DATA_H",
        "#define POWER_GRID_IEEE118_DATA_H",
        "",
        f"#define POWER_GRID_IEEE118_SLACK_BUS {slack_bus - 1}",
        f"#define POWER_GRID_IEEE118_BRIDGE_COUNT {len(bridges)}",
        "#define POWER_GRID_SOLAR_GENERATOR_BUS 79",
        "#define POWER_GRID_WIND_GENERATOR_BUS 88",
        "#define POWER_GRID_SOLAR_NAMEPLATE_MW 500.0",
        "#define POWER_GRID_WIND_NAMEPLATE_MW 700.0",
        "",
    ]
    lines += render_array("POWER_GRID_BRANCHES", "PowerGridBranch", branch_rows, 2)
    lines += [""] + render_array("POWER_GRID_BRANCH_NAMES", "char *const", branch_names, 8)
    lines += [""] + render_array("POWER_GRID_BRANCH_R", "double", branch_r)
    lines += [""] + render_array("POWER_GRID_BRANCH_B", "double", branch_b)
    lines += [""] + render_array("POWER_GRID_RANDOM_EVENT_ELIGIBLE", "unsigned char", eligible, 16)
    lines += [""] + render_array("POWER_GRID_GENERATOR_BUSES", "int", generator_buses, 12)
    lines += [""] + render_array("POWER_GRID_GENERATOR_P_NOMINAL", "double", generator_p)
    lines += [""] + render_array("POWER_GRID_GENERATOR_P_MAX", "double", generator_p_max)
    lines += [""] + render_array("POWER_GRID_GENERATOR_Q_MIN", "double", generator_q_min)
    lines += [""] + render_array("POWER_GRID_GENERATOR_Q_MAX", "double", generator_q_max)
    lines += [""] + render_array("POWER_GRID_GENERATOR_V_SETPOINT", "double", generator_v)
    lines += [""] + render_array("POWER_GRID_LOAD_BUSES", "int", load_buses, 12)
    lines += [""] + render_array("POWER_GRID_LOAD_P_NOMINAL", "double", load_p)
    lines += [""] + render_array("POWER_GRID_LOAD_Q_NOMINAL", "double", load_q)
    lines += [""] + render_array("POWER_GRID_BUS_SHUNT_B_MVAR", "double", bus_shunt_b, 12)
    lines += [""] + render_array("POWER_GRID_BUS_BASE_KV", "double", bus_base_kv, 12)
    lines += [""] + render_array("POWER_GRID_BUS_V_REFERENCE", "double", bus_v_reference, 12)
    lines += [""] + render_array("POWER_GRID_BUS_ANGLE_REFERENCE_DEG", "double",
                                  bus_angle_reference_deg, 12)
    lines += ["", "#endif", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.source:
        source = args.source.read_bytes()
    else:
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "power-rl/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            source = response.read()
    args.output.write_text(build(source))
    print(f"wrote IEEE-118 data to {args.output}")


if __name__ == "__main__":
    main()
