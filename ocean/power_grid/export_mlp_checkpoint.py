#!/usr/bin/env python3
"""Export a one-hidden-layer PufferLib MLP checkpoint for the C QA harness."""

import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    keys = [
        "encoder.encoder.weight",
        "encoder.encoder.bias",
        "network.net.0.weight",
        "network.net.0.bias",
    ]
    if "network.net.2.weight" in state:
        keys += ["network.net.2.weight", "network.net.2.bias"]
    keys += [
        "decoder.decoder.weight",
        "decoder.decoder.bias",
    ]
    with open(args.output, "wb") as output:
        for key in keys:
            np.asarray(state[key], dtype=np.float32).tofile(output)


if __name__ == "__main__":
    main()
