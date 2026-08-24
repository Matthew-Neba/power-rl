"""Fit the deployed policy decoder to offline AC recovery actions."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


OBSERVATIONS = 236
ACTIONS = 106
HIDDEN = 256
LAYERS = 3
ENCODER_FLOATS = HIDDEN * OBSERVATIONS
DECODER_FLOATS = (ACTIONS + 1) * HIDDEN
RECURRENT_FLOATS = LAYERS * 3 * HIDDEN * HIDDEN
EXAMPLE_DTYPE = np.dtype([
    ("observations", "<f4", OBSERVATIONS),
    ("hidden", "<f4", HIDDEN),
    ("action", "<i4"),
    ("sequence_start", "u1"),
    ("train", "u1"),
    ("padding", "u1", 2),
])


def fit(source: Path, dataset_path: Path, destination: Path, epochs: int,
        no_op_weight_scale: float) -> None:
    weights = np.fromfile(source, dtype=np.float32)
    expected = ENCODER_FLOATS + DECODER_FLOATS + RECURRENT_FLOATS
    if weights.size != expected:
        raise ValueError(f"expected {expected} checkpoint floats, found {weights.size}")

    examples = np.fromfile(dataset_path, dtype=EXAMPLE_DTYPE)
    if examples.size == 0:
        raise ValueError("recovery dataset is empty")
    examples = examples[examples["train"] != 0]
    hidden = torch.from_numpy(examples["hidden"].copy()).cuda()
    actions = torch.from_numpy(examples["action"].astype(np.int64)).cuda()

    decoder_offset = ENCODER_FLOATS
    original = torch.from_numpy(
        weights[decoder_offset : decoder_offset + DECODER_FLOATS]
        .reshape(ACTIONS + 1, HIDDEN)
        .copy()
    ).cuda()
    decoder = torch.nn.Parameter(original[:ACTIONS].clone())
    optimizer = torch.optim.AdamW([decoder], lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cuda").manual_seed(4741)
    class_weights = torch.ones(ACTIONS, device="cuda")
    no_op_count = (actions == 0).sum().clamp_min(1)
    recovery_count = (actions != 0).sum().clamp_min(1)
    class_weights[0] = no_op_weight_scale * recovery_count / no_op_count

    split = max(1, int(0.9 * hidden.shape[0]))
    order = torch.randperm(hidden.shape[0], generator=generator, device="cuda")
    train_indices, valid_indices = order[:split], order[split:]
    batch_size = 1024
    for epoch in range(epochs):
        shuffled = train_indices[torch.randperm(
            train_indices.shape[0], generator=generator, device="cuda")]
        for start in range(0, shuffled.shape[0], batch_size):
            batch = shuffled[start : start + batch_size]
            logits = F.linear(hidden[batch], decoder)
            loss = F.cross_entropy(
                logits, actions[batch], weight=class_weights
            ) + 1e-5 * F.mse_loss(decoder, original[:ACTIONS])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch + 1 == epochs:
            with torch.no_grad():
                logits = F.linear(hidden[valid_indices], decoder)
                loss = F.cross_entropy(
                    logits, actions[valid_indices], weight=class_weights
                ).item()
                predicted = logits.argmax(1)
                accuracy = (predicted == actions[valid_indices]).float().mean().item()
                recovery = actions[valid_indices] != 0
                recovery_accuracy = (
                    (predicted[recovery] == actions[valid_indices][recovery])
                    .float().mean().item() if recovery.any() else 1.0
                )
            print(
                f"epoch {epoch + 1:3d}: validation loss {loss:.4f}, "
                f"accuracy {accuracy:.2%}, recovery {recovery_accuracy:.2%}"
            )

    updated = weights.copy()
    fitted = torch.cat((decoder.detach(), original[ACTIONS:]), dim=0)
    updated[decoder_offset : decoder_offset + DECODER_FLOATS] = (
        fitted.cpu().numpy().reshape(-1)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    updated.tofile(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no-op-weight-scale", type=float, default=1.0)
    args = parser.parse_args()
    fit(args.source, args.dataset, args.destination, args.epochs,
        args.no_op_weight_scale)
