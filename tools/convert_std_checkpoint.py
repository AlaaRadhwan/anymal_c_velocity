#!/usr/bin/env python3
"""Convert an RSL-RL 4.0.1 checkpoint from scalar std to log_std."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--minimum-std",
        type=float,
        default=0.05,
        help="Minimum allowed standard deviation.",
    )
    args = parser.parse_args()

    source = args.checkpoint.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")

    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
    )

    actor_state = checkpoint["actor_state_dict"]

    if "log_std" in actor_state:
        raise RuntimeError("Checkpoint already uses log_std.")

    if "std" not in actor_state:
        raise KeyError("Checkpoint does not contain actor parameter 'std'.")

    old_std = actor_state.pop("std")

    print("Checkpoint:", source)
    print("Stored std:", old_std.tolist())
    print("Minimum:", old_std.min().item())
    print("Maximum:", old_std.max().item())
    print("All finite:", torch.isfinite(old_std).all().item())

    if not torch.isfinite(old_std).all():
        raise RuntimeError(
            "The checkpoint contains NaN or infinite std values. "
            "Use an earlier checkpoint."
        )

    safe_std = old_std.clamp_min(args.minimum_std)
    actor_state["log_std"] = torch.log(safe_std)

    # Adam moments for std are not mathematically equivalent to
    # moments for log_std. Keep the optimizer configuration but
    # restart its accumulated moment estimates.
    optimizer_state = checkpoint.get("optimizer_state_dict")

    if optimizer_state is not None:
        optimizer_state["state"] = {}

    destination = source.with_name(
        f"{source.stem}_logstd.pt"
    )

    torch.save(checkpoint, destination)

    print("Converted std:", safe_std.tolist())
    print("Saved:", destination)


if __name__ == "__main__":
    main()

