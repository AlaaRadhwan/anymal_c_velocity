"""Play a Spot checkpoint from a selected training run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp

# Registers the project's custom MjLab tasks.
import anymal_c_velocity  # noqa: F401

from mjlab.scripts.play import PlayConfig, run_play


TASK_ID = "Mjlab-Velocity-Flat-Spot"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "rsl_rl"
    / "spot_velocity"
)

CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Play a Spot checkpoint. If no checkpoint is specified, "
            "the latest checkpoint in the selected run is used."
        )
    )

    parser.add_argument(
        "run_name",
        help="Run folder under logs/rsl_rl/spot_velocity.",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        help=(
            "Checkpoint number or filename. Examples: "
            "1500 or model_1500.pt. Default: latest checkpoint."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Playback device. Default: cuda:0 when available.",
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Optional number of playback environments.",
    )

    parser.add_argument(
        "--viewer",
        choices=("auto", "native", "viser"),
        default="auto",
        help="Viewer backend. Default: auto.",
    )

    parser.add_argument(
        "--no-terminations",
        action="store_true",
        help="Disable termination conditions during playback.",
    )

    return parser.parse_args()


def get_run_directory(run_name: str) -> Path:
    """Resolve the selected training-run directory."""

    run_directory = LOG_ROOT / run_name

    if not run_directory.is_dir():
        raise FileNotFoundError(
            f"Run folder does not exist: {run_directory}"
        )

    return run_directory.resolve()


def get_latest_checkpoint(run_directory: Path) -> Path:
    """Return the checkpoint with the highest model number."""

    checkpoints: list[tuple[int, Path]] = []

    for checkpoint_path in run_directory.glob("model_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(
            checkpoint_path.name
        )

        if match is None:
            continue

        model_number = int(match.group(1))
        checkpoints.append(
            (model_number, checkpoint_path)
        )

    if not checkpoints:
        raise FileNotFoundError(
            f"No model_*.pt checkpoints found in {run_directory}"
        )

    _, latest_checkpoint = max(
        checkpoints,
        key=lambda item: item[0],
    )

    return latest_checkpoint.resolve()


def get_checkpoint(
    run_directory: Path,
    checkpoint: str | None,
) -> Path:
    """Resolve the requested checkpoint or select the latest one."""

    if checkpoint is None:
        return get_latest_checkpoint(run_directory)

    if checkpoint.isdigit():
        checkpoint_name = f"model_{int(checkpoint)}.pt"
    else:
        checkpoint_name = checkpoint

    checkpoint_path = run_directory / checkpoint_name

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    return checkpoint_path.resolve()


def normalize_cuda_version(version: object) -> tuple[int, int]:
    """Convert Warp's CUDA driver version to (major, minor)."""

    if isinstance(version, tuple):
        return int(version[0]), int(version[1])

    if isinstance(version, list):
        return int(version[0]), int(version[1])

    if isinstance(version, int):
        return version // 1000, (version % 1000) // 10

    match = re.search(
        r"(\d+)\.(\d+)",
        str(version),
    )

    if match is None:
        raise RuntimeError(
            f"Unsupported CUDA driver version format: {version!r}"
        )

    return int(match.group(1)), int(match.group(2))


def apply_warp_compatibility() -> None:
    """Provide the legacy attribute expected by MjLab 1.1."""

    if hasattr(wp, "context"):
        return

    wp.init()

    driver_version = normalize_cuda_version(
        wp.get_cuda_driver_version()
    )

    wp.context = SimpleNamespace(
        runtime=SimpleNamespace(
            driver_version=driver_version,
        )
    )


def main() -> None:
    """Resolve the checkpoint and launch MjLab playback."""

    args = parse_args()

    run_directory = get_run_directory(
        args.run_name
    )

    checkpoint_path = get_checkpoint(
        run_directory=run_directory,
        checkpoint=args.checkpoint,
    )

    print(f"[INFO] Task:       {TASK_ID}")
    print(f"[INFO] Run:        {run_directory.name}")
    print(f"[INFO] Checkpoint: {checkpoint_path.name}")
    print(f"[INFO] Device:     {args.device}")

    apply_warp_compatibility()

    play_cfg = PlayConfig(
        checkpoint_file=str(checkpoint_path),
        num_envs=args.num_envs,
        device=args.device,
        viewer=args.viewer,
        no_terminations=args.no_terminations,
    )

    run_play(
        TASK_ID,
        play_cfg,
    )


if __name__ == "__main__":
    main()