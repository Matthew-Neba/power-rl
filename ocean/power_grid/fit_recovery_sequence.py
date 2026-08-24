"""Fine-tune the deployed MinGRU through complete offline recovery sequences."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fit_recovery_decoder import (
    ACTIONS,
    DECODER_FLOATS,
    ENCODER_FLOATS,
    EXAMPLE_DTYPE,
    HIDDEN,
    LAYERS,
    OBSERVATIONS,
    RECURRENT_FLOATS,
)


def fit(source: Path, dataset_path: Path, destination: Path, epochs: int,
        encoder_lr: float, decoder_lr: float, no_op_weight_scale: float) -> None:
    weights = np.fromfile(source, dtype=np.float32)
    expected = ENCODER_FLOATS + DECODER_FLOATS + RECURRENT_FLOATS
    if weights.size != expected:
        raise ValueError(f"expected {expected} checkpoint floats, found {weights.size}")
    examples = np.fromfile(dataset_path, dtype=EXAMPLE_DTYPE)
    starts = np.flatnonzero(examples["sequence_start"])
    if starts.size == 0 or starts[0] != 0:
        raise ValueError("recovery dataset has no complete sequences")
    ends = np.r_[starts[1:], len(examples)]
    sequences = list(zip(starts.tolist(), ends.tolist()))

    rng = np.random.default_rng(4741)
    order = rng.permutation(len(sequences))
    split = max(1, int(0.9 * len(order)))
    training = [sequences[index] for index in order[:split]]
    validation = [sequences[index] for index in order[split:]]

    device = "cuda"
    original_encoder = torch.from_numpy(
        weights[:ENCODER_FLOATS].reshape(HIDDEN, OBSERVATIONS).copy()
    ).to(device)
    decoder_offset = ENCODER_FLOATS
    original_decoder = torch.from_numpy(
        weights[decoder_offset : decoder_offset + DECODER_FLOATS]
        .reshape(ACTIONS + 1, HIDDEN).copy()
    ).to(device)
    recurrent = torch.from_numpy(
        weights[decoder_offset + DECODER_FLOATS :]
        .reshape(LAYERS, 3 * HIDDEN, HIDDEN).copy()
    ).to(device)
    encoder = torch.nn.Parameter(original_encoder.clone())
    decoder = torch.nn.Parameter(original_decoder[:ACTIONS].clone())
    optimizer = torch.optim.AdamW([
        {"params": [encoder], "lr": encoder_lr},
        {"params": [decoder], "lr": decoder_lr},
    ], weight_decay=1e-4)

    train_actions = examples["action"][examples["train"] != 0]
    no_op_count = max(1, int((train_actions == 0).sum()))
    recovery_count = max(1, int((train_actions != 0).sum()))
    class_weights = torch.ones(ACTIONS, device=device)
    class_weights[0] = no_op_weight_scale * recovery_count / no_op_count

    def batches(items, batch_size=128):
        ordered = sorted(items, key=lambda pair: pair[1] - pair[0])
        result = [
            ordered[start : start + batch_size]
            for start in range(0, len(ordered), batch_size)
        ]
        rng.shuffle(result)
        return result

    def run_batch(batch, update):
        batch_size = len(batch)
        steps = max(end - start for start, end in batch)
        observations = np.zeros((batch_size, steps, OBSERVATIONS), np.float32)
        actions = np.zeros((batch_size, steps), np.int64)
        acceptable = np.zeros((batch_size, steps, ACTIONS), np.bool_)
        mask = np.zeros((batch_size, steps), np.bool_)
        for row, (start, end) in enumerate(batch):
            length = end - start
            observations[row, :length] = examples["observations"][start:end]
            actions[row, :length] = examples["action"][start:end]
            acceptable[row, :length] = examples["acceptable"][start:end]
            mask[row, :length] = examples["train"][start:end] != 0
        observations = torch.from_numpy(observations).to(device)
        actions = torch.from_numpy(actions).to(device)
        acceptable = torch.from_numpy(acceptable).to(device)
        mask = torch.from_numpy(mask).to(device)
        state = torch.zeros(LAYERS, batch_size, HIDDEN, device=device)
        losses = []
        accepted = []
        targets = []
        for step in range(steps):
            value = F.linear(observations[:, step], encoder)
            next_state = []
            for layer in range(LAYERS):
                candidate, gate, highway = F.linear(
                    value, recurrent[layer]
                ).chunk(3, dim=1)
                candidate = torch.where(
                    candidate >= 0, candidate + 0.5, candidate.sigmoid()
                )
                output = torch.lerp(state[layer], candidate, gate.sigmoid())
                mix = highway.sigmoid()
                value = mix * output + (1.0 - mix) * value
                next_state.append(output)
            state = torch.stack(next_state)
            selected = mask[:, step]
            if selected.any():
                logits = F.linear(value[selected], decoder)
                valid = acceptable[selected, step]
                accepted_logits = logits.masked_fill(~valid, -torch.inf)
                losses.append(logits.logsumexp(1) - accepted_logits.logsumexp(1))
                target = actions[selected, step]
                targets.append(target)
                accepted.append(valid[
                    torch.arange(len(logits), device=device), logits.argmax(1)
                ])
        losses = torch.cat(losses)
        accepted = torch.cat(accepted)
        targets = torch.cat(targets)
        sample_weights = class_weights[targets]
        loss = (losses * sample_weights).sum() / sample_weights.sum()
        loss = loss + 1e-5 * F.mse_loss(decoder, original_decoder[:ACTIONS])
        loss = loss + 1e-4 * F.mse_loss(encoder, original_encoder)
        if update:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([encoder, decoder], 1.0)
            optimizer.step()
        return loss.detach().item(), accepted.float().mean().item()

    for epoch in range(epochs):
        train_stats = [run_batch(batch, True) for batch in batches(training)]
        with torch.no_grad():
            valid_stats = [run_batch(batch, False) for batch in batches(validation)]
        print(
            f"epoch {epoch + 1}: train {np.mean(train_stats, axis=0)}, "
            f"validation {np.mean(valid_stats, axis=0)}"
        )

    updated = weights.copy()
    updated[:ENCODER_FLOATS] = encoder.detach().cpu().numpy().reshape(-1)
    fitted = torch.cat((decoder.detach(), original_decoder[ACTIONS:]), dim=0)
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--encoder-lr", type=float, default=3e-6)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--no-op-weight-scale", type=float, default=1.0)
    args = parser.parse_args()
    fit(args.source, args.dataset, args.destination, args.epochs,
        args.encoder_lr, args.decoder_lr, args.no_op_weight_scale)
