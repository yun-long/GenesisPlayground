from collections.abc import Iterator
from typing import Final

import torch
from tensordict import TensorDict

from gs_agent.bases.buffer import BaseBuffer
from gs_agent.buffers.config.schema import BCBufferKey

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")


class BCBuffer(BaseBuffer):
    """
    A buffer for storing expert demonstrations for Behavior Cloning (BC) training.

    This buffer stores state-action pairs from expert demonstrations and provides
    functionality for sampling mini-batches for supervised learning.
    """

    def __init__(
        self,
        max_size: int,
        obs_size: int,
        action_size: int,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        """
        Initialize the BC buffer.

        Args:
            max_size: Maximum number of transitions to store
            obs_size: Dimension of the observation space
            action_size: Dimension of the action space
            device: Device to store the buffer on
        """
        super().__init__()
        self._max_size = max_size
        self._obs_size = obs_size
        self._action_size = action_size
        self._device = device

        # Track current size
        self._idx = 0
        self._size = 0

        # Initialize buffer
        self._buffer = self._init_buffers()

    def _init_buffers(self) -> TensorDict:
        """Initialize the buffer tensors."""
        buffer = TensorDict(
            {
                BCBufferKey.OBSERVATIONS: torch.zeros(
                    self._max_size, self._obs_size, device=self._device
                ),
                BCBufferKey.ACTIONS: torch.zeros(
                    self._max_size, self._action_size, device=self._device
                ),
            },
            batch_size=[self._max_size],
            device=self._device,
        )
        return buffer

    def reset(self) -> None:
        """Reset the buffer state."""
        self._idx = 0
        self._size = 0

    def append(self, transition: dict[BCBufferKey, torch.Tensor]) -> None:
        """
        Append a transition to the buffer.

        Args:
            transition: Dictionary containing:
                - 'obs': Current observation [obs_size]
                - 'act': Action taken [action_size]
        """
        idx = self._idx % self._max_size

        # Store the transition
        self._buffer[BCBufferKey.OBSERVATIONS][idx] = transition[BCBufferKey.OBSERVATIONS]
        self._buffer[BCBufferKey.ACTIONS][idx] = transition[BCBufferKey.ACTIONS]

        # increment index
        self._idx += 1
        self._size = min(self._size + 1, self._max_size)

    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return self._size >= self._max_size

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self._size

    def minibatch_gen(
        self, batch_size: int, num_epochs: int = 1, shuffle: bool = True
    ) -> Iterator[dict[BCBufferKey, torch.Tensor]]:
        """
        Generate mini-batches for training.

        Args:
            batch_size: Size of each mini-batch
            num_epochs: Number of epochs to iterate through the data
            shuffle: Whether to shuffle the data

        Yields:
            Dictionary containing mini-batch data
        """
        if self._size == 0:
            return

        # Create indices for all stored data
        indices = torch.arange(self._size, device=self._device)

        for _epoch in range(num_epochs):
            if shuffle:
                perm_indices = indices[torch.randperm(self._size, device=self._device)]
            else:
                perm_indices = indices

            # Split into mini-batches
            for start_idx in range(0, self._size, batch_size):
                end_idx = min(start_idx + batch_size, self._size)
                batch_indices = perm_indices[start_idx:end_idx]

                if batch_indices.numel() == 0:
                    continue

                yield {
                    BCBufferKey.OBSERVATIONS: self._buffer[BCBufferKey.OBSERVATIONS][batch_indices],
                    BCBufferKey.ACTIONS: self._buffer[BCBufferKey.ACTIONS][batch_indices],
                }

    def clear(self) -> None:
        """Clear all data from the buffer."""
        self.reset()
