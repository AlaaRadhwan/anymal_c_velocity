"""SDL/Pygame gamepad access and Spot velocity-command mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

# These variables must be set before Pygame initializes SDL.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame

from .robot_types import VelocityCommand


class GamepadError(RuntimeError):
  """Base exception for gamepad initialization and communication errors."""


class GamepadDisconnectedError(GamepadError):
  """Raised when the active controller is removed."""


@dataclass(frozen=True)
class GamepadInfo:
  """Static information reported by SDL for one controller."""

  name: str
  guid: str
  instance_id: int
  axis_count: int
  button_count: int
  hat_count: int
  power_level: str


@dataclass(frozen=True)
class GamepadSnapshot:
  """One complete raw controller sample."""

  axes: tuple[float, ...]
  buttons: tuple[bool, ...]
  hats: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class GamepadMapping:
  """Raw SDL indices and signs used for Spot teleoperation."""

  left_x_axis: int = 0
  left_y_axis: int = 1
  right_x_axis: int = 3

  # PS5 DualSense mapping under Linux/SDL.
  enable_button: int = 4   # L1
  reset_button: int = 0    # Cross / X
  exit_button: int = 1     # Circle

  # SDL reports upward and leftward stick motion as negative.
  # These signs convert:
  # left stick up    -> +vx
  # left stick left  -> +vy
  # right stick left -> +wz
  vx_sign: float = -1.0
  vy_sign: float = -1.0
  wz_sign: float = -1.0


@dataclass(frozen=True)
class GamepadControl:
  """One semantic controller sample consumed by the teleoperation loop."""

  command: VelocityCommand
  enabled: bool
  reset_requested: bool
  exit_requested: bool


class Gamepad:
  """Open one Pygame joystick and provide validated polling."""

  def __init__(self, device_index: int = 0) -> None:
    if device_index < 0:
      raise ValueError("device_index must be non-negative.")

    self._joystick: pygame.joystick.JoystickType | None = None
    self._connected = False
    self._owns_display = False

    try:
      self._initialize_pygame()

      device_count = pygame.joystick.get_count()
      if device_count == 0:
        raise GamepadError(
          "No gamepad was detected. Connect the PS5 controller and rerun."
        )
      if device_index >= device_count:
        raise GamepadError(
          f"Requested gamepad index {device_index}, but SDL detected "
          f"{device_count} device(s)."
        )

      joystick = pygame.joystick.Joystick(device_index)
      joystick.init()
      self._joystick = joystick
      self._instance_id = int(joystick.get_instance_id())
      self._connected = True

      self.info = GamepadInfo(
        name=joystick.get_name(),
        guid=joystick.get_guid(),
        instance_id=self._instance_id,
        axis_count=joystick.get_numaxes(),
        button_count=joystick.get_numbuttons(),
        hat_count=joystick.get_numhats(),
        power_level=joystick.get_power_level(),
      )
    except Exception:
      self.close()
      raise

  @property
  def connected(self) -> bool:
    """Return whether the opened controller is still available."""

    return self._connected

  def poll(self) -> GamepadSnapshot:
    """Process SDL events and return the latest raw controller state."""

    joystick = self._require_joystick()

    for event in pygame.event.get():
      if (
        event.type == pygame.JOYDEVICEREMOVED
        and int(event.instance_id) == self._instance_id
      ):
        self._connected = False

    if not self._connected or not joystick.get_init():
      raise GamepadDisconnectedError(
        f"Gamepad {self.info.name!r} was disconnected."
      )

    axes = tuple(
      self._validate_axis(joystick.get_axis(index), index)
      for index in range(self.info.axis_count)
    )
    buttons = tuple(
      bool(joystick.get_button(index))
      for index in range(self.info.button_count)
    )
    hats = tuple(
      tuple(int(value) for value in joystick.get_hat(index))
      for index in range(self.info.hat_count)
    )

    return GamepadSnapshot(
      axes=axes,
      buttons=buttons,
      hats=hats,
    )

  def close(self) -> None:
    """Release the controller and the hidden SDL event window."""

    self._connected = False

    if self._joystick is not None:
      if self._joystick.get_init():
        self._joystick.quit()
      self._joystick = None

    if pygame.joystick.get_init():
      pygame.joystick.quit()

    if self._owns_display and pygame.display.get_init():
      pygame.display.quit()
      self._owns_display = False

  def __enter__(self) -> "Gamepad":
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()

  def _initialize_pygame(self) -> None:
    if not pygame.display.get_init():
      pygame.display.init()
      self._owns_display = True

    # SDL's event queue needs an initialized display. The window stays hidden.
    if pygame.display.get_surface() is None:
      try:
        pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
      except pygame.error as error:
        raise GamepadError(
          f"Could not initialize SDL's event system: {error}"
        ) from error

    if not pygame.joystick.get_init():
      pygame.joystick.init()

  def _require_joystick(self) -> pygame.joystick.JoystickType:
    if self._joystick is None:
      raise GamepadError("The gamepad is closed.")
    return self._joystick

  @staticmethod
  def _validate_axis(value: float, index: int) -> float:
    value = float(value)
    if not math.isfinite(value):
      raise GamepadError(f"Gamepad axis {index} returned a non-finite value.")
    return max(-1.0, min(1.0, value))


class SpotGamepad:
  """Convert raw gamepad samples into safe body-velocity commands."""

  def __init__(
    self,
    gamepad: Gamepad,
    *,
    mapping: GamepadMapping = GamepadMapping(),
    deadzone: float = 0.10,
    max_vx: float = 0.65,
    max_vy: float = 0.40,
    max_wz: float = 0.50,
  ) -> None:
    if not 0.0 <= deadzone < 1.0:
      raise ValueError("deadzone must be in [0, 1).")
    if max_vx <= 0.0 or max_vy <= 0.0 or max_wz <= 0.0:
      raise ValueError("Velocity limits must be positive.")

    self._gamepad = gamepad
    self._mapping = mapping
    self._deadzone = deadzone
    self._max_vx = float(max_vx)
    self._max_vy = float(max_vy)
    self._max_wz = float(max_wz)
    self._previous_reset = False
    self._previous_exit = False

    self._validate_mapping(gamepad.info)

  @property
  def mapping(self) -> GamepadMapping:
    return self._mapping

  def poll(self) -> GamepadControl:
    """Return one mapped sample with edge-triggered reset and exit buttons."""

    snapshot = self._gamepad.poll()
    mapping = self._mapping

    enabled = snapshot.buttons[mapping.enable_button]
    reset_pressed = snapshot.buttons[mapping.reset_button]
    exit_pressed = snapshot.buttons[mapping.exit_button]

    reset_requested = reset_pressed and not self._previous_reset
    exit_requested = exit_pressed and not self._previous_exit
    self._previous_reset = reset_pressed
    self._previous_exit = exit_pressed

    if enabled:
      vx_axis = self._axis(snapshot, mapping.left_y_axis)
      vy_axis = self._axis(snapshot, mapping.left_x_axis)
      wz_axis = self._axis(snapshot, mapping.right_x_axis)

      command = VelocityCommand(
        vx=mapping.vx_sign * vx_axis * self._max_vx,
        vy=mapping.vy_sign * vy_axis * self._max_vy,
        wz=mapping.wz_sign * wz_axis * self._max_wz,
      )
    else:
      command = VelocityCommand()

    return GamepadControl(
      command=command,
      enabled=enabled,
      reset_requested=reset_requested,
      exit_requested=exit_requested,
    )

  def _axis(self, snapshot: GamepadSnapshot, index: int) -> float:
    return self._apply_deadzone(snapshot.axes[index], self._deadzone)

  @staticmethod
  def _apply_deadzone(value: float, deadzone: float) -> float:
    magnitude = abs(value)
    if magnitude <= deadzone:
      return 0.0

    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return math.copysign(min(scaled, 1.0), value)

  def _validate_mapping(self, info: GamepadInfo) -> None:
    axis_indices = {
      "left_x_axis": self._mapping.left_x_axis,
      "left_y_axis": self._mapping.left_y_axis,
      "right_x_axis": self._mapping.right_x_axis,
    }
    for name, index in axis_indices.items():
      if not 0 <= index < info.axis_count:
        raise GamepadError(
          f"{name}={index} is invalid for a controller with "
          f"{info.axis_count} axes."
        )

    button_indices = {
      "enable_button": self._mapping.enable_button,
      "reset_button": self._mapping.reset_button,
      "exit_button": self._mapping.exit_button,
    }
    for name, index in button_indices.items():
      if not 0 <= index < info.button_count:
        raise GamepadError(
          f"{name}={index} is invalid for a controller with "
          f"{info.button_count} buttons."
        )
