from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Final

import torch

from gs_agent.bases.env_wrapper import BaseEnvWrapper
from gs_agent.bases.policy import Policy


class BaseAlgo(ABC):
    """
    Base class for all algorithms in the Genesis Playground system.

    This abstract base class defines the common interface that all algorithms
    must implement, ensuring consistency across different algorithm types.
    """

    def __init__(self, env: BaseEnvWrapper, cfg: Any, device: torch.device) -> None:
        """
        Initialize the base algorithm.

        Args:
            env: The environment to train the algorithm on.
            cfg: The configuration for the algorithm.
            device: The device to train the algorithm on.
        """
        self.env: Final[BaseEnvWrapper] = env
        self.cfg: Final[Any] = cfg
        self.device: Final[torch.device] = device

    @abstractmethod
    def train_one_iteration(self) -> dict[str, Any]:
        """
        Train the algorithm for a given iteration.

        Returns:
            A dictionary containing the training results.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the algorithm to a file.
        """
        ...

    @abstractmethod
    def load(self, path: Path, load_optimizer: bool = True) -> None:
        """
        Load the algorithm from a file.
        """
        ...

    @abstractmethod
    def train_mode(self) -> None:
        """
        Set the algorithm to train mode.
        """
        ...

    @abstractmethod
    def eval_mode(self) -> None:
        """
        Set the algorithm to eval mode.
        """
        ...

    @abstractmethod
    def get_inference_policy(self, device: torch.device | None = None) -> Policy:
        """
        Get the inference policy for evaluation.
        """
        ...
