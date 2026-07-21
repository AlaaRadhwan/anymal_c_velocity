"""TorchScript inference for the exported Spot actor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch

from .config import PolicyConfig
from .robot_types import PolicyAction


class TorchScriptPolicy:
  """Load and run the deterministic exported actor."""

  def __init__(
    self,
    policy_path: str | Path,
    config: PolicyConfig,
    device: str = "cpu",
  ) -> None:
    self._path = Path(policy_path).expanduser().resolve()
    if not self._path.is_file():
      raise FileNotFoundError(
        f"TorchScript policy not found: {self._path}"
      )

    self._config = config
    self._device = torch.device(device)
    self._model = torch.jit.load(
      str(self._path),
      map_location=self._device,
    )
    self._model.eval()

  @property
  def device(self) -> torch.device:
    """Return the device used for inference."""

    return self._device

  def act(
    self,
    observation: NDArray[np.floating] | list[float],
  ) -> PolicyAction:
    """Run one observation through the actor."""

    array = np.asarray(observation, dtype=np.float32)
    if array.ndim == 1:
      array = array[np.newaxis, :]

    expected_shape = (1, self._config.observation_size)
    if array.shape != expected_shape:
      raise ValueError(
        f"Policy observation must have shape {expected_shape}; "
        f"got {array.shape}."
      )

    if not np.all(np.isfinite(array)):
      raise ValueError("Policy observation contains NaN or infinite values.")

    observation_tensor = torch.from_numpy(array).to(self._device)

    with torch.inference_mode():
      output = self._model(observation_tensor)

    if not isinstance(output, torch.Tensor):
      raise TypeError(
        "The exported TorchScript policy did not return a tensor."
      )

    expected_output_shape = (1, self._config.action_size)
    if tuple(output.shape) != expected_output_shape:
      raise RuntimeError(
        f"Policy output must have shape {expected_output_shape}; "
        f"got {tuple(output.shape)}."
      )

    action = output.detach().to("cpu", dtype=torch.float32).numpy()[0]
    if not np.all(np.isfinite(action)):
      raise RuntimeError("Policy output contains NaN or infinite values.")

    return PolicyAction(action)
