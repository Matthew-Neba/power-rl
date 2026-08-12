import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
POWER_GRID = ROOT / "ocean" / "power_grid"


def compile_and_run(tmp_path, name, sources, *, optimized=False, dc_float=False):
    executable = tmp_path / name
    command = [
        "clang",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
    ]
    if optimized:
        command.append("-O2")
    if dc_float:
        command.append("-DPOWER_GRID_DC_FLOAT")
    command.extend([
        f"-I{POWER_GRID}",
        *(str(source) for source in sources),
        "-lm",
        "-o",
        str(executable),
    ])
    subprocess.run(command, check=True, cwd=ROOT)
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_power_grid_solver(tmp_path):
    compile_and_run(
        tmp_path,
        "test_power_grid_solver",
        [ROOT / "tests/test_power_grid_solver.c", POWER_GRID / "power_grid_solver.c"],
    )


def test_power_grid_float_solver(tmp_path):
    """Exercise the exact float-factorization path used by training."""
    compile_and_run(
        tmp_path,
        "test_power_grid_float_solver",
        [ROOT / "tests/test_power_grid_solver.c", POWER_GRID / "power_grid_solver.c"],
        optimized=True,
        dc_float=True,
    )


def test_power_grid_ac_solver(tmp_path):
    compile_and_run(
        tmp_path,
        "test_power_grid_ac",
        [
            ROOT / "tests/test_power_grid_ac.c",
            POWER_GRID / "power_grid_solver.c",
        ],
    )


def test_power_grid_environment(tmp_path):
    compile_and_run(
        tmp_path,
        "test_power_grid_env",
        [ROOT / "tests/test_power_grid_env.c"],
        optimized=True,
    )


def test_power_grid_float_environment(tmp_path):
    """Check environment caches and observations with training precision."""
    compile_and_run(
        tmp_path,
        "test_power_grid_float_env",
        [ROOT / "tests/test_power_grid_env.c"],
        optimized=True,
        dc_float=True,
    )


def test_power_grid_user_controller(tmp_path):
    compile_and_run(
        tmp_path,
        "test_power_grid_user",
        [ROOT / "tests/test_power_grid_user.c"],
        optimized=True,
    )


def test_power_grid_deployed_policy_architecture(tmp_path):
    compile_and_run(
        tmp_path,
        "test_power_grid_policy",
        [ROOT / "tests/test_power_grid_policy.c"],
        optimized=True,
    )
