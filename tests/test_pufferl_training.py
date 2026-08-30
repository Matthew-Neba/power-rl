import pytest
import torch

from pufferlib import pufferl


def test_sweep_cache_release_is_scoped_to_optimizer_device(monkeypatch):
    calls = []

    class DeviceContext:
        def __init__(self, device):
            calls.append(("device", torch.device(device)))

        def __enter__(self):
            calls.append(("enter",))

        def __exit__(self, *exc):
            calls.append(("exit",))

    monkeypatch.setattr(torch.cuda, "device", DeviceContext)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(("synchronize",)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty_cache",)))

    class Optimizer:
        device = torch.device("cuda:1")

    pufferl._release_sweep_cuda_cache(Optimizer())

    assert calls == [
        ("device", torch.device("cuda:1")),
        ("enter",),
        ("synchronize",),
        ("empty_cache",),
        ("exit",),
    ]


@pytest.mark.parametrize("optimizer", [object(), type("CPUOptimizer", (), {"device": "cpu"})()])
def test_sweep_cache_release_skips_non_cuda_optimizers(monkeypatch, optimizer):
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda: pytest.fail("CUDA synchronization should not run"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("CUDA cache release should not run"),
    )

    pufferl._release_sweep_cuda_cache(optimizer)


def test_train_loads_requested_initial_checkpoint(monkeypatch, tmp_path):
    class InitialWeightsLoaded(Exception):
        pass

    class Backend:
        @staticmethod
        def create_pufferl(args):
            return object()

        @staticmethod
        def load_weights(state, path):
            assert path == "initial.bin"
            raise InitialWeightsLoaded

    monkeypatch.setattr(pufferl, "_resolve_backend", lambda args: Backend)
    args = {
        "rank": 0,
        "wandb": False,
        "train": {"total_timesteps": 1},
        "sweep": {"metric": "perf"},
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "logs"),
        "env_name": "test",
        "load_model_path": "initial.bin",
    }

    with pytest.raises(InitialWeightsLoaded):
        pufferl._train("test", args)


def test_power_grid_preserves_memory_until_an_episode_terminates(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pytest"])
    args = pufferl.load_config("power_grid")
    assert args["reset_state"] is False
    assert args["train"]["horizon"] == args["env"]["max_episode_steps"] == 72
    assert args["train"]["minibatch_size"] % args["train"]["horizon"] == 0


@pytest.mark.parametrize("text, expected", [("True", True), ("False", False)])
def test_boolean_config_overrides_parse_their_value(monkeypatch, text, expected):
    monkeypatch.setattr(
        "sys.argv", ["pytest", "--env.end-episode-on-recovery", text]
    )
    args = pufferl.load_config("power_grid")
    assert args["env"]["end_episode_on_recovery"] is expected
