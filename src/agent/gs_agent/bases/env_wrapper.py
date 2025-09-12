from abc import ABC, abstractmethod
from typing import Any, Final

import torch


class BaseEnvWrapper(ABC):
    """
    Base class for all environment wrappers.
    """

    def __init__(self, env: Any, device: torch.device) -> None:
        self.env: Final[Any] = env
        self.device: Final[torch.device] = device

    @abstractmethod
    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]: ...

    @abstractmethod
    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]: ...

    @abstractmethod
    def get_observations(self) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abstractmethod
    def actor_obs_dim(self) -> int: ...

    @property
    @abstractmethod
    def critic_obs_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_envs(self) -> int: ...
