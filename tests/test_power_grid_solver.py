import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_power_grid_solver(tmp_path):
    executable = tmp_path / "test_power_grid_solver"
    subprocess.run(
        [
            "clang",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            f"-I{ROOT / 'ocean' / 'power_grid'}",
            str(ROOT / "tests" / "test_power_grid_solver.c"),
            str(ROOT / "ocean" / "power_grid" / "power_grid_solver.c"),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_power_grid_baselines(tmp_path):
    executable = tmp_path / "test_power_grid_baselines"
    subprocess.run(
        [
            "clang",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            f"-I{ROOT / 'ocean' / 'power_grid'}",
            str(ROOT / "tests" / "test_power_grid_baselines.c"),
            str(ROOT / "ocean" / "power_grid" / "power_grid_baselines.c"),
            str(ROOT / "ocean" / "power_grid" / "power_grid_solver.c"),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_power_grid_ac_solver(tmp_path):
    executable = tmp_path / "test_power_grid_ac"
    subprocess.run(
        [
            "clang",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            f"-I{ROOT / 'ocean' / 'power_grid'}",
            str(ROOT / "tests" / "test_power_grid_ac.c"),
            str(ROOT / "ocean" / "power_grid" / "power_grid_solver.c"),
            str(ROOT / "ocean" / "power_grid" / "power_grid_ac.c"),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_power_grid_environment(tmp_path):
    executable = tmp_path / "test_power_grid_env"
    subprocess.run(
        [
            "clang",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            f"-I{ROOT / 'ocean' / 'power_grid'}",
            str(ROOT / "tests" / "test_power_grid_env.c"),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(executable)], check=True, cwd=ROOT)
