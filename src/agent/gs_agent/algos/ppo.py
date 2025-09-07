import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from gs_agent.algos.config.schema import PPOArgs
from gs_agent.bases.algo import BaseAlgo
from gs_agent.bases.env_wrapper import BaseEnvWrapper
from gs_agent.bases.policy import Policy
from gs_agent.buffers.config.schema import GAEBufferKey
from gs_agent.buffers.gae_buffer import GAEBuffer
from gs_agent.modules.critics import StateValueFunction
from gs_agent.modules.models import NetworkFactory
from gs_agent.modules.policies import GaussianPolicy

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")
"""Default device for the algorithm."""

_DEQUE_MAXLEN: Final[int] = 100
"""Max length of the deque for storing episode statistics."""


class PPO(BaseAlgo):
    """
    Proximal Policy Optimization (PPO) algorithm implementation.
    """

    def __init__(
        self, env: BaseEnvWrapper, cfg: PPOArgs, device: torch.device = _DEFAULT_DEVICE
    ) -> None:
        super().__init__(env, cfg, device)
        self._actor_obs_dim = self.env.actor_obs_dim
        self._critic_obs_dim = self.env.critic_obs_dim
        self._action_dim = self.env.action_dim

        self._num_envs = self.env.num_envs
        self._num_steps = cfg.rollout_length

        self.current_iter = 0
        self._rewbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._lenbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._curr_reward_sum = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.float
        )
        self._curr_ep_len = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float)
        #
        self._build_actor_critic()
        self._build_rollouts()

    def _build_actor_critic(self) -> None:
        policy_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.policy_backbone,
            input_dim=self._actor_obs_dim,
            output_dim=self._action_dim,
            device=self.device,
        )
        print(f"Policy backbone: {policy_backbone}")

        self._actor = GaussianPolicy(
            policy_backbone=policy_backbone,
            action_dim=self._action_dim,
        ).to(self.device)
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self.cfg.lr)

        critic_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.critic_backbone,
            input_dim=self._critic_obs_dim,
            output_dim=1,
            device=self.device,
        )
        print(f"Critic backbone: {critic_backbone}")
        self._critic = StateValueFunction(critic_backbone).to(self.device)
        value_lr = self.cfg.value_lr if self.cfg.value_lr is not None else self.cfg.lr
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=value_lr)

        self.actor_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        print("Device: ", self.device)

    def _build_rollouts(self) -> None:
        self._rollouts = GAEBuffer(
            num_envs=self._num_envs,
            max_steps=self._num_steps,
            actor_obs_size=self._actor_obs_dim,
            action_size=self._action_dim,
            device=self.device,
        )

    def _collect_rollouts(self, num_steps: int) -> dict[str, Any]:
        """Collect rollouts from the environment."""
        obs = self.env.get_observations()
        with torch.inference_mode():
            # collect rollouts and compute returns & advantages
            for _step in range(num_steps):
                action, log_prob = self._actor(obs)
                # Step environment
                next_obs, reward, terminated, truncated, _extra_infos = self.env.step(action)

                # all tensors are of shape: [num_envs, dim]
                transition = {
                    GAEBufferKey.ACTOR_OBS: obs,
                    GAEBufferKey.ACTIONS: action,
                    GAEBufferKey.REWARDS: reward,
                    GAEBufferKey.DONES: terminated,
                    GAEBufferKey.VALUES: self._critic(obs),
                    GAEBufferKey.ACTION_LOGPROBS: log_prob,
                }
                self._rollouts.append(transition)

                # Update episode tracking - handle reward and done sequences
                # Extract tensors from reward and done objects
                self._curr_reward_sum += reward.squeeze(-1)
                self._curr_ep_len += 1

                # Check for episode completions and reset tracking
                done_mask = terminated.unsqueeze(-1) | truncated.unsqueeze(-1)
                new_ids = (done_mask > 0).nonzero(as_tuple=False)
                if len(new_ids) > 0:
                    # Vectorized environment
                    self._rewbuffer.extend(
                        self._curr_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self._lenbuffer.extend(self._curr_ep_len[new_ids][:, 0].cpu().numpy().tolist())
                    # Reset tracking
                    self._curr_reward_sum[new_ids] = 0
                    self._curr_ep_len[new_ids] = 0

                obs = next_obs
        with torch.no_grad():
            last_value = self._critic(self.env.get_observations())
            self._rollouts.set_final_value(last_value)

        mean_reward = 0.0
        mean_ep_len = 0.0
        if len(self._rewbuffer) > 0:
            mean_reward = statistics.mean(self._rewbuffer)
            mean_ep_len = statistics.mean(self._lenbuffer)
        return {
            "mean_reward": mean_reward,
            "mean_ep_len": mean_ep_len,
        }

    def train_one_batch(self, mini_batch: dict[GAEBufferKey, torch.Tensor]) -> dict[str, Any]:
        """Train one batch of rollouts."""
        obs = mini_batch[GAEBufferKey.ACTOR_OBS]
        act = mini_batch[GAEBufferKey.ACTIONS]
        old_log_prob = mini_batch[GAEBufferKey.ACTION_LOGPROBS]
        advantage = mini_batch[GAEBufferKey.ADVANTAGES]
        returns = mini_batch[GAEBufferKey.RETURNS]

        #
        new_log_prob = self._actor.evaluate_log_prob(obs, act)
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = -advantage * ratio
        surr2 = -advantage * torch.clamp(
            ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
        )
        policy_loss = torch.max(surr1, surr2).mean()

        approx_kl = (new_log_prob - old_log_prob).mean()

        # Calculate value loss
        values = self._critic(obs)
        value_loss = (returns - values).pow(2).mean()

        # Calculate entropy loss
        entropy = self._actor.entropy_on(obs)
        entropy_loss = entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.cfg.value_loss_coef * value_loss
            - self.cfg.entropy_coef * entropy_loss
        )

        # Optimization step
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self._critic.parameters(), self.cfg.max_grad_norm)
        self._critic_optimizer.step()
        self._actor_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
        }

    def train_one_iteration(self) -> dict[str, Any]:
        """Train one iteration."""
        # collect rollouts
        t0 = time.time()
        rollout_infos = self._collect_rollouts(num_steps=self._num_steps)
        t1 = time.time()
        rollouts_time = t1 - t0
        fps = (self._num_steps * self._num_envs / rollouts_time) if rollouts_time > 0 else 0

        train_metrics_list: list[dict[str, Any]] = []
        for mini_batch in self._rollouts.minibatch_gen(
            num_mini_batches=self.cfg.num_mini_batches,
            num_epochs=self.cfg.num_epochs,
        ):
            metrics = self.train_one_batch(mini_batch)
            train_metrics_list.append(metrics)
        t2 = time.time()
        train_time = t2 - t1

        self._rollouts.reset()

        iteration_infos = {
            "rollout": {
                "mean_reward": rollout_infos["mean_reward"],
                "mean_length": rollout_infos["mean_ep_len"],
            },
            "train": {
                "policy_loss": statistics.mean(
                    [metrics["policy_loss"] for metrics in train_metrics_list]
                ),
                "value_loss": statistics.mean(
                    [metrics["value_loss"] for metrics in train_metrics_list]
                ),
                "entropy_loss": statistics.mean(
                    [metrics["entropy_loss"] for metrics in train_metrics_list]
                ),
                "approx_kl": statistics.mean(
                    [metrics["approx_kl"] for metrics in train_metrics_list]
                ),
            },
            "speed": {
                "rollout_time": rollouts_time,
                "rollout_fps": fps,
                "train_time": train_time,
                "rollout_step": self._num_steps * self._num_envs,
            },
        }
        return iteration_infos

    def save(self, path: Path) -> None:
        """Save the algorithm to a file."""
        saved_dict = {
            "model_state_dict": self._actor.state_dict(),
            "actor_optimizer_state_dict": self._actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self._critic_optimizer.state_dict(),
            "iter": self.current_iter,
        }
        torch.save(saved_dict, path)

    def load(self, path: Path, load_optimizer: bool = True) -> None:
        """Load the algorithm from a file."""
        checkpoint = torch.load(path)
        self._actor.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self._current_iter = checkpoint["iter"]
        if self.cfg.norm_obs:
            self.actor_obs_normalizer.load_state_dict(checkpoint["actor_obs_normalizer"])
            self.critic_obs_normalizer.load_state_dict(checkpoint["critic_obs_normalizer"])
        return checkpoint.get("infos", None)

    def train_mode(self) -> None:
        """Set the algorithm to train mode."""
        self._actor.train()
        self.actor_obs_normalizer.train()
        self.critic_obs_normalizer.train()

    def eval_mode(self) -> None:
        """Set the algorithm to eval mode."""
        self._actor.eval()
        self.actor_obs_normalizer.eval()
        self.critic_obs_normalizer.eval()

    def get_inference_policy(self, device: torch.device | None = None) -> Policy:
        """Get the inference policy for evaluation."""
        self.eval_mode()
        if device is not None:
            self._actor.to(device)
        policy = self._actor
        return policy
