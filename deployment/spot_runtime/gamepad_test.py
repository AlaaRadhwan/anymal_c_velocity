"""Display raw PS5-controller axes, buttons, and hats."""

from __future__ import annotations

import argparse
import time

from .gamepad import Gamepad, GamepadDisconnectedError


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Inspect the raw SDL mapping of a connected gamepad."
  )
  parser.add_argument(
    "--index",
    type=int,
    default=0,
    help="SDL device index to open. Default: 0.",
  )
  parser.add_argument(
    "--rate",
    type=float,
    default=60.0,
    help="Polling frequency in Hz. Default: 60.",
  )
  parser.add_argument(
    "--axis-threshold",
    type=float,
    default=0.08,
    help="Minimum axis change printed. Default: 0.08.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.rate <= 0.0:
    raise ValueError("--rate must be positive.")
  if not 0.0 < args.axis_threshold <= 2.0:
    raise ValueError("--axis-threshold must be in (0, 2].")

  with Gamepad(device_index=args.index) as gamepad:
    info = gamepad.info
    initial = gamepad.poll()

    print("Gamepad diagnostic started.")
    print(f"  name:        {info.name}")
    print(f"  GUID:        {info.guid}")
    print(f"  instance ID: {info.instance_id}")
    print(f"  axes:        {info.axis_count}")
    print(f"  buttons:     {info.button_count}")
    print(f"  hats:        {info.hat_count}")
    print(f"  power:       {info.power_level}")
    print()
    print("Initial axis values:")
    for index, value in enumerate(initial.axes):
      print(f"  axis {index:2d}: {value:+.3f}")

    print()
    print("Move or press one control at a time.")
    print("Record the indices for:")
    print("  left stick X/Y, right stick X, L1, Cross/X, and Circle.")
    print("Press Ctrl+C when finished.")
    print()

    reported_axes = list(initial.axes)
    previous_buttons = list(initial.buttons)
    previous_hats = list(initial.hats)
    period_s = 1.0 / args.rate

    try:
      while True:
        cycle_start = time.perf_counter()
        snapshot = gamepad.poll()

        for index, value in enumerate(snapshot.axes):
          if abs(value - reported_axes[index]) >= args.axis_threshold:
            print(f"axis {index:2d}: {value:+.3f}")
            reported_axes[index] = value

        for index, pressed in enumerate(snapshot.buttons):
          if pressed != previous_buttons[index]:
            state = "DOWN" if pressed else "UP"
            print(f"button {index:2d}: {state}")
            previous_buttons[index] = pressed

        for index, value in enumerate(snapshot.hats):
          if value != previous_hats[index]:
            print(f"hat {index:2d}: {value}")
            previous_hats[index] = value

        sleep_time = period_s - (time.perf_counter() - cycle_start)
        if sleep_time > 0.0:
          time.sleep(sleep_time)
    except KeyboardInterrupt:
      print("\nGamepad diagnostic stopped.")
    except GamepadDisconnectedError as error:
      raise SystemExit(str(error)) from error


if __name__ == "__main__":
  main()
