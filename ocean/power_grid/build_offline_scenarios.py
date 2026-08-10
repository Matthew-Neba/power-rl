#!/usr/bin/env python3
"""Build the compile-time power-grid scenario cache from public historical data.

This is deliberately an offline build tool. The generated C header is the only
artifact used by training; the environment never imports Python, reads a data
file, or makes a network request.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import io
import json
import math
import pathlib
import statistics
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "ocean/power_grid/power_grid_scenarios_data.h"
DEFAULT_CACHE = ROOT / "ocean/power_grid/.scenario_sources"
AESO_URL = (
    "https://www.aeso.ca/assets/Uploads/"
    "AILSMP-Wind-and-Solar-2016-to-2020.xlsx"
)
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FIELDS = (
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
)
PERIODS = 12
# These reference values define the network-independent cached renewable trace.
# At runtime the environment converts each series back to a capacity factor and
# applies the active network's solar and wind nameplates.
NOMINAL_TOTAL_LOAD_MW = 4242.0
FIXED_GENERATION_MW = 1500.0
SOLAR_NAMEPLATE_MW = 500.0
WIND_NAMEPLATE_MW = 700.0


def fetch_cached(url: str, path: pathlib.Path, refresh: bool = False) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "power-rl-scenario-builder/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def parse_aeso_xlsx(data: bytes) -> list[dict[str, object]]:
    """Read the single-sheet AESO workbook without a third-party XLSX package."""
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    result: list[dict[str, object]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as book:
        shared_root = ET.fromstring(book.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.iter(namespace + "t"))
                  for item in shared_root.findall(namespace + "si")]
        stream = book.open("xl/worksheets/sheet1.xml")
        for _, row in ET.iterparse(stream, events=("end",)):
            if row.tag != namespace + "row":
                continue
            values: dict[str, str] = {}
            for cell in row.findall(namespace + "c"):
                reference = cell.attrib.get("r", "")
                column = "".join(character for character in reference if character.isalpha())
                value_node = cell.find(namespace + "v")
                if value_node is None or value_node.text is None:
                    continue
                value = value_node.text
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
                values[column] = value
            row.clear()
            if not {"B", "C", "E", "F"}.issubset(values):
                continue
            try:
                timestamp = dt.datetime(1899, 12, 30) + dt.timedelta(days=float(values["B"]))
                load = float(values["C"])
                solar = float(values["E"])
                wind = float(values["F"])
            except ValueError:
                continue
            if all(math.isfinite(value) and value >= 0 for value in (load, solar, wind)):
                result.append({"time": timestamp, "load": load, "solar": solar, "wind": wind})
    return result


def parse_weather_json(data: bytes) -> list[dict[str, object]]:
    document = json.loads(data)
    hourly = document["hourly"]
    lengths = {len(hourly[field]) for field in ("time", *WEATHER_FIELDS)}
    if len(lengths) != 1:
        raise ValueError("weather arrays have inconsistent lengths")
    rows = []
    for index, timestamp in enumerate(hourly["time"]):
        values = [hourly[field][index] for field in WEATHER_FIELDS]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            continue
        rows.append({
            "time": dt.datetime.fromisoformat(timestamp),
            **{field: float(hourly[field][index]) for field in WEATHER_FIELDS},
        })
    return rows


def aggregate_two_hour_periods(rows: list[dict[str, object]], fields: tuple[str, ...]):
    buckets: dict[dt.date, list[dict[str, list[float]]]] = collections.defaultdict(
        lambda: [{field: [] for field in fields} for _ in range(PERIODS)]
    )
    for row in rows:
        timestamp = row["time"]
        assert isinstance(timestamp, dt.datetime)
        period = timestamp.hour // 2
        for field in fields:
            buckets[timestamp.date()][period][field].append(float(row[field]))

    complete = {}
    for date, periods in buckets.items():
        if any(not period[field] for period in periods for field in fields):
            continue
        complete[date] = {
            field: [statistics.fmean(period[field]) for period in periods]
            for field in fields
        }
    return complete


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_scenarios(aeso_rows, weather_rows, train_year: int, validation_year: int):
    aeso = aggregate_two_hour_periods(aeso_rows, ("load", "solar", "wind"))
    weather = aggregate_two_hour_periods(weather_rows, WEATHER_FIELDS)
    dates = sorted(date for date in aeso.keys() & weather.keys()
                   if date.year in (train_year, validation_year))
    if not dates:
        raise ValueError("AESO and weather sources have no complete overlapping days")

    train_loads = [value for date in dates if date.year == train_year
                   for value in aeso[date]["load"]]
    if not train_loads:
        raise ValueError(f"no complete training days for {train_year}")
    reference_load = statistics.median(train_loads)

    renewable_reference = {}
    for year in (train_year, validation_year):
        for field in ("solar", "wind"):
            values = [value for date in dates if date.year == year
                      for value in aeso[date][field]]
            renewable_reference[year, field] = max(percentile(values, 0.995), 1.0)

    scenarios = []
    for date in dates:
        load_scale = [min(1.35, max(0.65, value / reference_load))
                      for value in aeso[date]["load"]]
        solar_mw = [SOLAR_NAMEPLATE_MW * min(1.0, max(0.0, value /
                    renewable_reference[date.year, "solar"]))
                    for value in aeso[date]["solar"]]
        wind_mw = [WIND_NAMEPLATE_MW * min(1.0, max(0.0, value /
                   renewable_reference[date.year, "wind"]))
                   for value in aeso[date]["wind"]]

        # Preserve the observed renewable mix while keeping the reference
        # cache's residual generation non-negative.
        for period in range(PERIODS):
            maximum_renewable = max(
                0.0, NOMINAL_TOTAL_LOAD_MW * load_scale[period] - FIXED_GENERATION_MW - 5.0
            )
            renewable = solar_mw[period] + wind_mw[period]
            if renewable > maximum_renewable:
                factor = maximum_renewable / renewable
                solar_mw[period] *= factor
                wind_mw[period] *= factor

        scenarios.append({
            "date": int(date.strftime("%Y%m%d")),
            "load_scale": load_scale,
            "solar_mw": solar_mw,
            "wind_mw": wind_mw,
            "ambient_temperature_c": weather[date]["temperature_2m"],
            "wind_speed_mps": weather[date]["wind_speed_10m"],
            "solar_irradiance_wm2": weather[date]["shortwave_radiation"],
        })
    return scenarios


def render_header(scenarios, train_year: int, validation_year: int,
                  aeso_hash: str, weather_hash: str) -> str:
    train_count = sum(str(scenario["date"]).startswith(str(train_year)) for scenario in scenarios)
    validation_count = len(scenarios) - train_count
    if not train_count or not validation_count:
        raise ValueError("both training and validation scenario pools must be non-empty")
    lines = [
        "/* Generated by build_offline_scenarios.py; do not edit manually.",
        f" * AESO SHA256: {aeso_hash}",
        f" * Open-Meteo ERA5 SHA256: {weather_hash}",
        " */",
        "#ifndef POWER_GRID_SCENARIOS_DATA_H",
        "#define POWER_GRID_SCENARIOS_DATA_H",
        "",
        f"#define POWER_GRID_OFFLINE_TRAIN_YEAR {train_year}",
        f"#define POWER_GRID_OFFLINE_VALIDATION_YEAR {validation_year}",
        f"#define POWER_GRID_OFFLINE_TRAIN_COUNT {train_count}",
        f"#define POWER_GRID_OFFLINE_VALIDATION_OFFSET {train_count}",
        f"#define POWER_GRID_OFFLINE_VALIDATION_COUNT {validation_count}",
        f"#define POWER_GRID_OFFLINE_SCENARIO_COUNT {len(scenarios)}",
        "",
        "static const PowerGridOfflineScenario POWER_GRID_OFFLINE_SCENARIOS[",
        "    POWER_GRID_OFFLINE_SCENARIO_COUNT] = {",
    ]
    fields = (
        "load_scale", "solar_mw", "wind_mw", "ambient_temperature_c",
        "wind_speed_mps", "solar_irradiance_wm2",
    )
    for scenario in scenarios:
        lines.append(f"    {{.date_yyyymmdd = {scenario['date']}u,")
        for field in fields:
            rendered = []
            for value in scenario[field]:
                number = f"{value:.6g}"
                if "." not in number and "e" not in number:
                    number += ".0"
                rendered.append(number + "f")
            lines.append(f"     .{field} = {{{', '.join(rendered)}}},")
        lines.append("    },")
    lines.extend(("};", "", "#endif", ""))
    return "\n".join(lines)


def weather_request_url(start_year: int, end_year: int) -> str:
    query = urllib.parse.urlencode({
        "latitude": 53.5461,
        "longitude": -113.4938,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "hourly": ",".join(WEATHER_FIELDS),
        "timezone": "America/Edmonton",
        "wind_speed_unit": "ms",
        "models": "era5",
    })
    return f"{WEATHER_URL}?{query}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-year", type=int, default=2019)
    parser.add_argument("--validation-year", type=int, default=2020)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.train_year >= args.validation_year:
        parser.error("--train-year must precede --validation-year")

    aeso_data = fetch_cached(AESO_URL, args.cache_dir / "aeso_2016_2020.xlsx", args.refresh)
    weather_url = weather_request_url(args.train_year, args.validation_year)
    weather_data = fetch_cached(
        weather_url,
        args.cache_dir / f"era5_edmonton_{args.train_year}_{args.validation_year}.json",
        args.refresh,
    )
    scenarios = build_scenarios(
        parse_aeso_xlsx(aeso_data), parse_weather_json(weather_data),
        args.train_year, args.validation_year,
    )
    output = render_header(
        scenarios, args.train_year, args.validation_year,
        hashlib.sha256(aeso_data).hexdigest(), hashlib.sha256(weather_data).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    print(f"wrote {len(scenarios)} cached scenarios to {args.output}")


if __name__ == "__main__":
    main()
