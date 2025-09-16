#!/usr/bin/env python3
"""Example: Train PPO on Genesis Goal-reaching environment using Genesis RL."""

import fire
import torch
from gs_agent.algos.config.registry import PPO_WALKING_MLP
from gs_agent.algos.ppo import PPO
from gs_agent.runners.config.registry import RUNNER_WALKING_MLP
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.common.bases.base_env import BaseEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.locomotion.walking_env import WalkingEnv


def create_gs_env(
    env_name: str = "walk_default",
    show_viewer: bool = False,
    num_envs: int = 2048,
    device: str = "cuda",
) -> BaseEnv:
    """Create gym environment wrapper."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    return WalkingEnv(
        args=EnvArgsRegistry[env_name], num_envs=num_envs, show_viewer=show_viewer, device=device
    )


def create_ppo_runner_from_registry(env: BaseEnv) -> OnPolicyRunner:
    """Create PPO runner using configuration from the registry."""
    # Environment setup
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create PPO algorithm
    ppo = PPO(
        env=wrapped_env,
        cfg=PPO_WALKING_MLP,
        device=wrapped_env.device,
    )

    # Create PPO runner
    runner = OnPolicyRunner(
        algorithm=ppo,
        runner_args=RUNNER_WALKING_MLP,
        device=wrapped_env.device,
    )
    return runner


def main(num_envs: int = 2048, show_viewer: bool = False, device: str = "cuda") -> None:
    """Main function demonstrating proper registry usage."""
    # create environment
    env = create_gs_env(show_viewer=show_viewer, num_envs=num_envs, device=device)
    # Get configuration and runner from registry
    runner = create_ppo_runner_from_registry(env)
    # Set up logging with proper configuration
    logger = logger_configure(
        folder=str(runner.save_dir),
        format_strings=["stdout", "csv", "wandb"],
        mode="disabled",
        # mode="online" if not show_viewer else "disabled",
    )
    # Train using Runner
    train_summary_info = runner.train(metric_logger=logger)

    print("Training completed successfully!")
    print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
    print(f"Total episodes: {train_summary_info['total_episodes']}.")
    print(f"Total steps: {train_summary_info['total_steps']}.")
    print(f"Total reward: {train_summary_info['final_reward']:.2f}.")


if __name__ == "__main__":
    fire.Fire(main)
