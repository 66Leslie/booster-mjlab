"""Simple transition replay buffers for AMP state pairs."""

from __future__ import annotations

import torch


class AmpTransitionBuffer:
  """Circular replay buffer storing AMP state transitions."""

  def __init__(
    self,
    capacity: int,
    amp_obs_dim: int,
    device: str | torch.device,
  ) -> None:
    self.capacity = int(capacity)
    self.amp_obs_dim = int(amp_obs_dim)
    self.device = torch.device(device)
    self.state = torch.zeros(self.capacity, self.amp_obs_dim, device=self.device)
    self.next_state = torch.zeros_like(self.state)
    self._pos = 0
    self._size = 0

  @property
  def size(self) -> int:
    return self._size

  def insert(self, state: torch.Tensor, next_state: torch.Tensor) -> None:
    state = state.detach().to(self.device)
    next_state = next_state.detach().to(self.device)
    num = state.shape[0]
    if num >= self.capacity:
      self.state[:] = state[-self.capacity :]
      self.next_state[:] = next_state[-self.capacity :]
      self._pos = 0
      self._size = self.capacity
      return

    end = self._pos + num
    if end <= self.capacity:
      self.state[self._pos : end] = state
      self.next_state[self._pos : end] = next_state
    else:
      first = self.capacity - self._pos
      self.state[self._pos :] = state[:first]
      self.next_state[self._pos :] = next_state[:first]
      self.state[: end % self.capacity] = state[first:]
      self.next_state[: end % self.capacity] = next_state[first:]
    self._pos = end % self.capacity
    self._size = min(self.capacity, self._size + num)

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if self._size <= 0:
      raise RuntimeError("Cannot sample from an empty AMP replay buffer.")
    idx = torch.randint(self._size, (batch_size,), device=self.device)
    return self.state[idx], self.next_state[idx]
