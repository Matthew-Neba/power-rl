from array import array
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "ocean/power_grid/expand_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("expand_checkpoint", MODULE_PATH)
expand_checkpoint = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(expand_checkpoint)


def read_floats(path):
    values = array("f")
    with path.open("rb") as checkpoint:
        values.fromfile(checkpoint, path.stat().st_size // values.itemsize)
    return values


def test_checkpoint_expansion_preserves_legacy_policy_layout(tmp_path):
    module = expand_checkpoint
    encoder_size = module.HIDDEN_SIZE * module.OLD_OBSERVATIONS
    decoder_size = (module.OLD_ACTIONS + 1) * module.HIDDEN_SIZE
    recurrent_size = (
        module.NUM_LAYERS * 3 * module.HIDDEN_SIZE * module.HIDDEN_SIZE
    )
    legacy = array("f", (float(index) for index in range(
        encoder_size + decoder_size + recurrent_size
    )))
    source = tmp_path / "legacy.bin"
    destination = tmp_path / "expanded.bin"
    with source.open("wb") as checkpoint:
        legacy.tofile(checkpoint)

    module.convert(source, destination)
    expanded = read_floats(destination)
    new_encoder_size = module.HIDDEN_SIZE * module.NEW_OBSERVATIONS
    new_decoder_size = (module.NEW_ACTIONS + 1) * module.HIDDEN_SIZE
    assert len(expanded) == new_encoder_size + new_decoder_size + recurrent_size

    for row in range(module.HIDDEN_SIZE):
        old_start = row * module.OLD_OBSERVATIONS
        new_start = row * module.NEW_OBSERVATIONS
        assert expanded[new_start : new_start + module.OLD_OBSERVATIONS] == (
            legacy[old_start : old_start + module.OLD_OBSERVATIONS]
        )
        assert expanded[
            new_start + module.OLD_OBSERVATIONS : new_start + module.NEW_OBSERVATIONS
        ] == array("f", [0.0] * (module.NEW_OBSERVATIONS - module.OLD_OBSERVATIONS))

    old_decoder = legacy[encoder_size : encoder_size + decoder_size]
    legacy_action_weights = module.OLD_ACTIONS * module.HIDDEN_SIZE
    assert expanded[
        new_encoder_size : new_encoder_size + legacy_action_weights
    ] == old_decoder[:legacy_action_weights]
    no_op = old_decoder[: module.HIDDEN_SIZE]
    for action in range(module.OLD_ACTIONS, module.NEW_ACTIONS):
        start = new_encoder_size + action * module.HIDDEN_SIZE
        assert expanded[start : start + module.HIDDEN_SIZE] == no_op
    value_start = new_encoder_size + module.NEW_ACTIONS * module.HIDDEN_SIZE
    assert expanded[value_start : value_start + module.HIDDEN_SIZE] == old_decoder[
        legacy_action_weights : legacy_action_weights + module.HIDDEN_SIZE
    ]
    assert expanded[new_encoder_size + new_decoder_size :] == legacy[
        encoder_size + decoder_size :
    ]


def test_checkpoint_expansion_rejects_wrong_architecture(tmp_path):
    source = tmp_path / "bad.bin"
    destination = tmp_path / "expanded.bin"
    with source.open("wb") as checkpoint:
        array("f", [1.0]).tofile(checkpoint)

    with pytest.raises(ValueError, match="expected .* floats, found 1"):
        expand_checkpoint.convert(source, destination)
