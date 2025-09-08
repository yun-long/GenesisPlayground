#!/usr/bin/env python3
"""Example: Train PPO on Pendulum-v1 environment using Genesis RL."""

import time
from pathlib import Path

import fire
import gymnasium as gym
import torch
from gs_agent.algos.config.registry import PPO_PENDULUM_MLP
from gs_agent.algos.ppo import PPO
from gs_agent.runners.config.registry import RUNNER_PENDULUM_PPO_MLP
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_latest_experiment, load_latest_model
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


def create_ppo_runner_from_registry() -> OnPolicyRunner:
    """Create PPO runner using configuration from the registry."""
    # Environment setup
    wrapped_env = create_gym_env()

    # Create PPO algorithm
    ppo = PPO(
        env=wrapped_env,
        cfg=PPO_PENDULUM_MLP,
        device=wrapped_env.device,
    )

    # Create PPO runner
    runner = OnPolicyRunner(
        algorithm=ppo,
        runner_args=RUNNER_PENDULUM_PPO_MLP,
        device=wrapped_env.device,
    )
    return runner


def evaluate_policy(checkpoint_path: Path, num_episodes: int = 10) -> None:
    """Evaluate a trained policy."""
    # Create environment with rendering
    wrapped_env = create_gym_env(render_mode="human")
    # Get config from checkpoint if available, otherwise use default
    ppo = PPO(env=wrapped_env, cfg=PPO_PENDULUM_MLP, device=wrapped_env.device)
    # Load checkpoint
    ppo.load(checkpoint_path)
    # Set to evaluation mode
    ppo.eval_mode()
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    inference_policy = ppo.get_inference_policy()
    for episode in range(num_episodes):
        obs, _info = wrapped_env.reset()
        episode_reward = 0.0
        episode_length = 0
        while True:
            with torch.no_grad():
                action, _log_prob = inference_policy(obs, deterministic=True)  # type: ignore
                obs, reward, done, _, _ = wrapped_env.step(action)  # type: ignore
            episode_reward += reward.item()
            episode_length += 1
            if done.item():
                break
            time.sleep(0.01)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: reward={episode_reward:.3f}, length={episode_length}")

    # Print results
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    std_reward = torch.std(torch.tensor(episode_rewards)).item()
    mean_length = sum(episode_lengths) / len(episode_lengths)

    print("Evaluation Results:")
    print(f"  Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"  Mean episode length: {mean_length:.1f}")
    print(f"  Min reward: {min(episode_rewards):.3f}")
    print(f"  Max reward: {max(episode_rewards):.3f}")


def main(train: bool = True) -> None:
    """Main function demonstrating proper registry usage."""
    if train:
        # Get configuration and runner from registry
        runner = create_ppo_runner_from_registry()
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
        log_dir = load_latest_experiment(exp_name="gym_pendulum", algo="ppo")
        print(f"Loading policy from {log_dir}")
        model_path = load_latest_model(Path(log_dir))
        evaluate_policy(model_path, num_episodes=10)


if __name__ == "__main__":
    fire.Fire(main)
