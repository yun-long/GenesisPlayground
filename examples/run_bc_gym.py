#!/usr/bin/env python3
"""Example: Train PPO on Pendulum-v1 environment using Genesis RL."""

from pathlib import Path

import fire
import gymnasium as gym
import torch
from gs_agent.algos.bc import BC
from gs_agent.algos.config.schema import BCArgs
from gs_agent.modules.config.registry import DEFAULT_MLP
from gs_agent.runners.config.registry import RUNNER_PENDULUM_BC_MLP
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.wrappers.gym_env_wrapper import GymEnvWrapper


def create_gym_env(env_name: str = "Pendulum-v1", render_mode: str | None = None) -> GymEnvWrapper:
    """Create gym environment wrapper."""
    # Properly detect available device (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    gym_env = gym.make(env_name, render_mode=render_mode)
    return GymEnvWrapper(gym_env, device=device)


def create_bc_runner_from_registry() -> OnPolicyRunner:
    """Create PPO runner using configuration from the registry."""
    # Environment setup
    wrapped_env = create_gym_env()

    BC_PENDULUM_MLP = BCArgs(
        policy_backbone=DEFAULT_MLP,
        teacher_backbone=DEFAULT_MLP,
        lr=3e-4,
        teacher_path=Path("./logs/ppo_gym_pendulum/20250907_171006/checkpoints/checkpoint_0200.pt"),
        num_epochs=1,
        batch_size=256,
        rollout_length=1000,
        max_buffer_size=1_000,
    )

    # Create PPO algorithm
    bc = BC(
        env=wrapped_env,
        cfg=BC_PENDULUM_MLP,
        device=wrapped_env.device,
    )

    # Create PPO runner
    runner = OnPolicyRunner(
        algorithm=bc,
        runner_args=RUNNER_PENDULUM_BC_MLP,
        device=wrapped_env.device,
    )
    return runner


def main(train: bool = True) -> None:
    """Main function demonstrating proper registry usage."""
    if train:
        # Get configuration and runner from registry
        runner = create_bc_runner_from_registry()
        # Set up logging with proper configuration
        logger = logger_configure(
            folder=str(runner.save_dir), format_strings=["stdout", "csv", "wandb"]
        )
        # Train using Runner
        train_summary_info = runner.train(metric_logger=logger)
        print("Training completed successfully!")
        print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
        print(f"Total episodes: {train_summary_info['total_episodes']}.")
        print(f"Total steps: {train_summary_info['total_steps']}.")
        print(f"Total reward: {train_summary_info['final_reward']:.2f}.")

    else:
        raise ValueError("Evaluation is not supported for BC")


if __name__ == "__main__":
    fire.Fire(main)
