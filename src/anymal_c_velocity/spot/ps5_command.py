"""PS5 velocity-command source for manual Spot policy testing."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


@dataclass(kw_only=True)
class Ps5VelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for velocity commands from a PS5 controller."""

  controller_index: int = 0
  deadzone: float = 0.12

  # Default Linux/Pygame 2 DualSense axis mapping.
  left_x_axis: int = 0
  left_y_axis: int = 1
  right_x_axis: int = 3

  # Match the command distribution used during training.
  min_vx: float = 0.25
  max_vx: float = 1.50

  # During combined vx/vy commands, training used vy in [-0.4, 0.4].
  max_diagonal_vy: float = 0.40

  # Pure-lateral commands were trained in [0.25, 0.80].
  min_lateral_vy: float = 0.25
  max_lateral_vy: float = 0.80

  max_wz: float = 1.00

  def __post_init__(self) -> None:
    super().__post_init__()

    if not 0.0 <= self.deadzone < 1.0:
      raise ValueError("deadzone must be in the range [0, 1).")

    if self.min_vx <= 0.0 or self.max_vx < self.min_vx:
      raise ValueError("Invalid forward velocity limits.")

    if (
      self.min_lateral_vy <= 0.0
      or self.max_lateral_vy < self.min_lateral_vy
    ):
      raise ValueError("Invalid lateral velocity limits.")

  def build(self, env):
    return Ps5VelocityCommand(self, env)


class Ps5VelocityCommand(UniformVelocityCommand):
  """Continuously update velocity commands from PS5 joystick axes."""

  cfg: Ps5VelocityCommandCfg

  def __init__(
    self,
    cfg: Ps5VelocityCommandCfg,
    env,
  ) -> None:
    super().__init__(cfg, env)

    # Continue receiving joystick input while the MuJoCo window
    # has keyboard/mouse focus.
    os.environ.setdefault(
      "SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS",
      "1",
    )

    try:
      import pygame
    except ImportError as exc:
      raise RuntimeError(
        "Pygame is required for PS5 control. "
        "Run: uv add pygame==2.6.1"
      ) from exc

    self._pygame = pygame

    # A small hidden SDL window keeps Pygame's event queue active
    # without replacing the MuJoCo viewer.
    pygame.display.init()

    if pygame.display.get_surface() is None:
      pygame.display.set_mode(
        (1, 1),
        flags=pygame.HIDDEN,
      )

    pygame.joystick.init()

    controller_count = pygame.joystick.get_count()

    if controller_count == 0:
      raise RuntimeError(
        "No controller detected. Connect the PS5 controller "
        "before starting MjLab play."
      )

    if cfg.controller_index >= controller_count:
      raise RuntimeError(
        f"Controller index {cfg.controller_index} requested, "
        f"but only {controller_count} controller(s) were found."
      )

    self._joystick = pygame.joystick.Joystick(
      cfg.controller_index
    )
    self._joystick.init()

    highest_axis = max(
      cfg.left_x_axis,
      cfg.left_y_axis,
      cfg.right_x_axis,
    )

    if self._joystick.get_numaxes() <= highest_axis:
      raise RuntimeError(
        f"Controller exposes {self._joystick.get_numaxes()} axes, "
        f"but axis {highest_axis} was requested."
      )

    print()
    print("[PS5] Manual velocity control enabled")
    print(f"[PS5] Controller: {self._joystick.get_name()}")
    print("[PS5] Left stick vertical: forward/backward")
    print("[PS5] Left stick horizontal: lateral movement")
    print("[PS5] Right stick horizontal: yaw rate")
    print()

  def _resample_command(
    self,
    env_ids: torch.Tensor,
  ) -> None:
    """Disable random command sampling during manual control."""

    self.vel_command_b[env_ids] = 0.0
    self.is_heading_env[env_ids] = False
    self.is_standing_env[env_ids] = False

  def _apply_deadzone(
    self,
    value: float,
  ) -> float:
    """Remove stick drift and rescale the remaining range."""

    magnitude = abs(value)

    if magnitude <= self.cfg.deadzone:
      return 0.0

    scaled = (
      magnitude - self.cfg.deadzone
    ) / (
      1.0 - self.cfg.deadzone
    )

    return scaled if value > 0.0 else -scaled

  @staticmethod
  def _map_nonzero_speed(
    value: float,
    minimum: float,
    maximum: float,
  ) -> float:
    """Map nonzero stick input to a trained nonzero speed range."""

    if value == 0.0:
      return 0.0

    magnitude = minimum + (
      abs(value) * (maximum - minimum)
    )

    return magnitude if value > 0.0 else -magnitude

  def _zero_command(self) -> None:
    """Safely stop all robots."""

    self.vel_command_b.zero_()

  def _update_command(self) -> None:
    """Read controller axes and update all play environments."""

    try:
      # Pygame requires its event queue to be pumped regularly.
      self._pygame.event.pump()

      if not self._joystick.get_init():
        self._zero_command()
        return

      left_x = self._joystick.get_axis(
        self.cfg.left_x_axis
      )
      left_y = self._joystick.get_axis(
        self.cfg.left_y_axis
      )
      right_x = self._joystick.get_axis(
        self.cfg.right_x_axis
      )

    except self._pygame.error:
      # Controller was disconnected or became unavailable.
      self._zero_command()
      return

    # Pygame reports stick-up as negative.
    vx_input = self._apply_deadzone(-left_y)

    # Positive body-frame y is left, while axis 0 is positive right.
    vy_input = self._apply_deadzone(-left_x)

    # Positive yaw is counterclockwise/left.
    wz_input = self._apply_deadzone(-right_x)

    vx = self._map_nonzero_speed(
      vx_input,
      self.cfg.min_vx,
      self.cfg.max_vx,
    )

    if vx == 0.0:
      # Pure lateral: use the full trained lateral range.
      vy = self._map_nonzero_speed(
        vy_input,
        self.cfg.min_lateral_vy,
        self.cfg.max_lateral_vy,
      )
    else:
      # Combined vx/vy: stay inside the diagonal training range.
      vy = (
        vy_input
        * self.cfg.max_diagonal_vy
      )

    wz = (
      wz_input
      * self.cfg.max_wz
    )

    # Every simulated robot receives the same manual command.
    self.vel_command_b[:, 0] = vx
    self.vel_command_b[:, 1] = vy
    self.vel_command_b[:, 2] = wz