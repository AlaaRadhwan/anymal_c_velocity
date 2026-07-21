"""Common interface implemented by simulator and robot backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..robot_types import JointTargets, RobotState


class RobotBackend(ABC):
  """Minimal state-and-control interface required by the policy loop."""

  @property
  @abstractmethod
  def timestep_s(self) -> float:
    """Return the duration of one backend physics step."""

  @property
  @abstractmethod
  def time_s(self) -> float:
    """Return the backend simulation or robot time."""

  @abstractmethod
  def reset(self) -> None:
    """Reset the backend to its nominal initial state."""

  @abstractmethod
  def read_state(self) -> RobotState:
    """Read the physical quantities required by the policy observation."""

  @abstractmethod
  def write_joint_targets(self, targets: JointTargets) -> None:
    """Apply absolute joint-position targets in policy joint order."""

  @abstractmethod
  def step(self) -> None:
    """Advance the backend by one physics step."""

  def close(self) -> None:
    """Release backend resources when necessary."""
