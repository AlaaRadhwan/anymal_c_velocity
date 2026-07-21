"""Shared data structures for the standalone Spot runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


def _vector(
  value: NDArray[np.floating] | list[float] | tuple[float, ...],
  *,
  name: str,
  size: int | None = None,
) -> FloatArray:
  """Convert one runtime vector to a finite float32 array."""

  array = np.asarray(value, dtype=np.float32)
  if array.ndim != 1:
    raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")

  if size is not None and array.size != size:
    raise ValueError(
      f"{name} must contain {size} values; got {array.size}."
    )

  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} contains NaN or infinite values.")

  return array.copy()


@dataclass(frozen=True)
class VelocityCommand:
  """Body-frame velocity command consumed by the policy."""

  vx: float = 0.0
  vy: float = 0.0
  wz: float = 0.0

  def as_array(self) -> FloatArray:
    """Return ``[vx, vy, wz]`` as float32."""

    return _vector((self.vx, self.vy, self.wz), name="command", size=3)


@dataclass(frozen=True)
class RobotState:
  """Physical quantities required to construct one policy observation."""

  base_lin_vel: FloatArray
  base_ang_vel: FloatArray
  projected_gravity: FloatArray
  joint_pos: FloatArray
  joint_vel: FloatArray

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "base_lin_vel",
      _vector(self.base_lin_vel, name="base_lin_vel", size=3),
    )
    object.__setattr__(
      self,
      "base_ang_vel",
      _vector(self.base_ang_vel, name="base_ang_vel", size=3),
    )
    object.__setattr__(
      self,
      "projected_gravity",
      _vector(
        self.projected_gravity,
        name="projected_gravity",
        size=3,
      ),
    )
    object.__setattr__(
      self,
      "joint_pos",
      _vector(self.joint_pos, name="joint_pos"),
    )
    object.__setattr__(
      self,
      "joint_vel",
      _vector(self.joint_vel, name="joint_vel"),
    )

    if self.joint_pos.shape != self.joint_vel.shape:
      raise ValueError(
        "joint_pos and joint_vel must have the same shape."
      )


@dataclass(frozen=True)
class PolicyAction:
  """Normalized action produced directly by the neural network."""

  values: FloatArray

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "values",
      _vector(self.values, name="policy_action"),
    )


@dataclass(frozen=True)
class JointTargets:
  """Absolute joint-position targets in policy joint order."""

  positions: FloatArray

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "positions",
      _vector(self.positions, name="joint_targets"),
    )
