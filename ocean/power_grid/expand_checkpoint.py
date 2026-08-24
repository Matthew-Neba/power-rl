"""Expand the 221-observation/91-action checkpoint for emergency controls."""

import argparse
from array import array
from pathlib import Path


OLD_OBSERVATIONS = 221
NEW_OBSERVATIONS = 236
OLD_ACTIONS = 91
NEW_ACTIONS = 106
HIDDEN_SIZE = 256
NUM_LAYERS = 3


def convert(source: Path, destination: Path) -> None:
    weights = array("f")
    with source.open("rb") as checkpoint:
        weights.fromfile(checkpoint, source.stat().st_size // weights.itemsize)

    old_encoder_size = HIDDEN_SIZE * OLD_OBSERVATIONS
    old_decoder_size = (OLD_ACTIONS + 1) * HIDDEN_SIZE
    recurrent_size = NUM_LAYERS * 3 * HIDDEN_SIZE * HIDDEN_SIZE
    expected = old_encoder_size + old_decoder_size + recurrent_size
    if len(weights) != expected:
        raise ValueError(f"expected {expected} floats, found {len(weights)}")

    expanded = array("f")
    for row in range(HIDDEN_SIZE):
        start = row * OLD_OBSERVATIONS
        expanded.extend(weights[start : start + OLD_OBSERVATIONS])
        expanded.extend([0.0] * (NEW_OBSERVATIONS - OLD_OBSERVATIONS))

    decoder = weights[old_encoder_size : old_encoder_size + old_decoder_size]
    for action in range(OLD_ACTIONS):
        start = action * HIDDEN_SIZE
        expanded.extend(decoder[start : start + HIDDEN_SIZE])
    # Duplicating the old no-op row makes deterministic inference select the
    # earlier original action while still giving PPO finite exploration logits.
    no_op = decoder[:HIDDEN_SIZE]
    for _ in range(NEW_ACTIONS - OLD_ACTIONS):
        expanded.extend(no_op)
    value_start = OLD_ACTIONS * HIDDEN_SIZE
    expanded.extend(decoder[value_start : value_start + HIDDEN_SIZE])
    expanded.extend(weights[old_encoder_size + old_decoder_size :])

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as checkpoint:
        expanded.tofile(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.destination)
