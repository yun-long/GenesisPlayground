#!/usr/bin/env python3
"""Example: Train PPO on Genesis Goal-reaching environment using Genesis RL."""

import fire
import torch
from gs_agent.algos.config.registry import PPO_GOAL_REACHING_MLP
from gs_agent.algos.ppo import PPO
from gs_agent.runners.config.registry import RUNNER_GOAL_REACHING_MLP
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_latest_experiment
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.common.bases.base_env import BaseEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.manipulation.goal_reaching_env import GoalReachingEnv


def create_gs_env(
    env_name: str = "goal_reach_default", show_viewer: bool = False, num_envs: int = 2048
) -> BaseEnv:
    """Create gym environment wrapper."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    return GoalReachingEnv(
        args=EnvArgsRegistry[env_name], num_envs=num_envs, show_viewer=show_viewer, device=device
    )


def create_ppo_runner_from_registry(env: BaseEnv) -> OnPolicyRunner:
    """Create PPO runner using configuration from the registry."""
    # Environment setup
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create PPO algorithm
    ppo = PPO(
        env=wrapped_env,
        cfg=PPO_GOAL_REACHING_MLP,
        device=wrapped_env.device,
    )

    # Create PPO runner
    runner = OnPolicyRunner(
        algorithm=ppo,
        runner_args=RUNNER_GOAL_REACHING_MLP,
        device=wrapped_env.device,
    )
    return runner


def evaluate_policy(runner: OnPolicyRunner, device: torch.device = torch.device("cpu")) -> None:
    """Evaluate the policy."""
    log_dir = load_latest_experiment(exp_name="goal_reach", algo="gsppo")
    print(f"Loading policy from {log_dir}")



def main(num_envs: int = 2048, show_viewer: bool = False, train: bool = True) -> None:
    """Main function demonstrating proper registry usage."""
    # create environment
    env = create_gs_env(show_viewer=show_viewer, num_envs=num_envs)
    # Get configuration and runner from registry
    runner = create_ppo_runner_from_registry(env)
    if train:
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
        print(f"Evaluating policy on {env.device}")
        evaluate_policy(runner, device=env.device)


if __name__ == "__main__":
    fire.Fire(main)
