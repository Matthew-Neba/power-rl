import datetime as dt
import importlib.util
import math
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "ocean" / "power_grid" / "build_offline_scenarios.py"
SPEC = importlib.util.spec_from_file_location("power_grid_scenario_builder", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def hourly_day(date, values):
    return [
        {"time": dt.datetime.combine(date, dt.time(hour)), **values(hour)}
        for hour in range(24)
    ]


def test_two_hour_aggregation_preserves_complete_days():
    date = dt.date(2019, 1, 1)
    rows = hourly_day(date, lambda hour: {"load": float(hour)})
    aggregated = BUILDER.aggregate_two_hour_periods(rows, ("load",))
    assert aggregated[date]["load"] == [hour + 0.5 for hour in range(0, 24, 2)]

    incomplete = rows[:10] + rows[12:]
    assert date not in BUILDER.aggregate_two_hour_periods(incomplete, ("load",))


def test_scenario_builder_keeps_years_separate_and_physical():
    aeso_rows = []
    weather_rows = []
    for year, load in ((2019, 9000.0), (2020, 9900.0)):
        date = dt.date(year, 1, 2)
        aeso_rows += hourly_day(date, lambda hour, load=load: {
            "load": load + 10 * hour,
            "solar": max(0.0, 100.0 - abs(hour - 12) * 20.0),
            "wind": 500.0 + hour,
        })
        weather_rows += hourly_day(date, lambda hour: {
            "temperature_2m": -10.0 + hour,
            "wind_speed_10m": 3.0,
            "shortwave_radiation": max(0.0, 500.0 - abs(hour - 12) * 100.0),
        })

    scenarios = BUILDER.build_scenarios(aeso_rows, weather_rows, 2019, 2020)
    assert [scenario["date"] for scenario in scenarios] == [20190102, 20200102]
    for scenario in scenarios:
        assert len(scenario["load_scale"]) == 12
        assert all(0.65 <= value <= 1.35 for value in scenario["load_scale"])
        assert all(0 <= value <= BUILDER.SOLAR_NAMEPLATE_MW
                   for value in scenario["solar_mw"])
        assert all(0 <= value <= BUILDER.WIND_NAMEPLATE_MW
                   for value in scenario["wind_mw"])
        for period in range(12):
            available = (BUILDER.NOMINAL_TOTAL_LOAD_MW * scenario["load_scale"][period]
                         - BUILDER.FIXED_GENERATION_MW - 5.0)
            assert scenario["solar_mw"][period] + scenario["wind_mw"][period] <= (
                available + 1e-9
            )

    header = BUILDER.render_header(scenarios, 2019, 2020, "a" * 64, "b" * 64)
    assert "POWER_GRID_OFFLINE_TRAIN_COUNT 1" in header
    assert "POWER_GRID_OFFLINE_VALIDATION_COUNT 1" in header
    assert "20190102u" in header and "20200102u" in header
    assert re.search(r"(?<![.0-9])[-+]?\d+f", header) is None


def test_checked_in_cache_has_provenance_and_expected_size():
    header = (ROOT / "ocean" / "power_grid" /
              "power_grid_scenarios_data.h").read_text()
    assert "AESO SHA256:" in header
    assert "Open-Meteo ERA5 SHA256:" in header
    assert "POWER_GRID_OFFLINE_TRAIN_COUNT 365" in header
    assert "POWER_GRID_OFFLINE_VALIDATION_COUNT 366" in header
    assert header.count(".date_yyyymmdd =") == 731
    assert not any(token in header for token in ("NaN", "nan", "inf", "Infinity"))


def test_weather_request_is_explicit_and_reproducible():
    url = BUILDER.weather_request_url(2019, 2020)
    assert "start_date=2019-01-01" in url
    assert "end_date=2020-12-31" in url
    assert "timezone=America%2FEdmonton" in url
    assert "models=era5" in url
    assert math.isfinite(BUILDER.NOMINAL_TOTAL_LOAD_MW)
