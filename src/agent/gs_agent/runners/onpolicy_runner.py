import datetime
import time
from pathlib import Path
from typing import Any, Final

import torch

from gs_agent.bases.algo import BaseAlgo
from gs_agent.bases.policy import Policy
from gs_agent.bases.runner import BaseRunner
from gs_agent.configs.schema import RunnerArgs

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")


class OnPolicyRunner(BaseRunner):
    """Abstract base class for on-policy algorithm runners.

    This class provides a high-level interface for training on-policy algorithms
    with integrated logging, checkpointing, and evaluation capabilities.
    """

    def __init__(
        self,
        algorithm: BaseAlgo,
        runner_args: RunnerArgs,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        """Initialize the on-policy runner.

        Args:
            algorithm: The algorithm to train (e.g., PPO)
            runner_args: Configuration for the runner
            device: Device to run training on
        """
        super().__init__()
        self.algorithm: Final[BaseAlgo] = algorithm
        self.args: Final[RunnerArgs] = runner_args
        self.device: Final[torch.device] = device

        # Create save directory
        self.save_dir: Final[Path] = self.args.save_path / datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.args.save_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir: Final[Path] = self.args.save_path / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        print(f"OnPolicyRunner initialized with save path: {self.args.save_path}")

    def train(self, metric_logger: Any) -> dict[str, int | float | str]:
        """Train the algorithm for a specified number of episodes.

        Args:
            metric_logger: Metric logger to log metrics to

        Returns:
            Dictionary containing training results and final metrics
        """
        print(f"Starting training for {self.args.total_episodes} episodes")

        start_time = time.time()
        total_steps = 0
        total_episodes = 0
        reward_list = []

        for episode in range(self.args.total_episodes):
            # Training step
            train_one_episode_metrics = self.algorithm.train_one_episode()

            total_episodes += 1
            total_steps += train_one_episode_metrics["speed"]["rollout_step"]
            reward_list.append(train_one_episode_metrics["rollout"]["mean_reward"])

            # Logging
            if episode % self.args.log_interval == 0:
                self._log_metrics(metric_logger, train_one_episode_metrics, episode)

            # Regular checkpointing
            if episode % self.args.save_interval == 0:
                self._save_checkpoint(Path(f"checkpoint_{episode}.pt"))

        # Training summary
        training_time = time.time() - start_time

        return {
            "total_episodes": total_episodes,
            "total_steps": total_steps,
            "total_time": training_time,
            "final_reward": reward_list[-1],
        }

    def _log_metrics(
        self,
        metric_logger: Any,
        metrics: dict[str, Any],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to the provided logger.

        Args:
            metric_logger: Logger to log metrics to
            metrics: Dictionary of metrics to log
            step: Current training step
            prefix: Optional prefix for metric names
        """
        for key, value in metrics.items():
            metric_name = f"{prefix}{key}" if prefix else key
            if not isinstance(value, dict):
                continue
            for k, v in value.items():
                if isinstance(v, float | int):
                    metric_logger.record(f"{metric_name}/{k}", v)
                else:
                    print(f"Invalid metric type: {type(value)}")
        metric_logger.dump(step=step)

    def _save_checkpoint(self, filename: Path) -> None:
        """Save a checkpoint of the algorithm.

        Args:
            filename: Name of the checkpoint file
            episode: Current episode number
        """
        checkpoint_path = self.checkpoint_dir / filename
        self.algorithm.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load a checkpoint of the algorithm.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        self.algorithm.load(path)
        print(f"Checkpoint loaded: {path}")

    def get_inference_policy(self) -> Policy:
        """Get the trained policy for inference."""
        return self.algorithm.get_inference_policy()
