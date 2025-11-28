from typing import Any, Final

import gymnasium as gym
import torch

from gs_agent.bases.env_wrapper import BaseEnvWrapper

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")


class GymEnvWrapper(BaseEnvWrapper):
    """Wrapper for a single Gymnasium env to typed (batch) step components (pseudo-batch)."""

    def __init__(self, env: gym.Env[Any, Any], device: torch.device = _DEFAULT_DEVICE) -> None:
        super().__init__(env, device)
        self._curr_obs: torch.Tensor = torch.tensor(
            self.env.reset()[0], device=self.device
        ).unsqueeze(0)

    # ---------------------------
    # BatchEnvWrapper API (batch)
    # ---------------------------
    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset()
        self._curr_obs = torch.tensor(obs, device=self.device).unsqueeze(0)
        return {"obs": self._curr_obs}, info

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        gym_actions = action.clone().cpu().numpy().squeeze(0)  # shape: [action_dim]

        obs, reward, terminated, truncated, info = self.env.step(gym_actions)

        done = terminated | truncated
        # If episode is done, reset the environment
        if done:
            obs, info = self.env.reset()

        self._curr_obs = torch.tensor(obs, device=self.device).unsqueeze(0)

        reward_batch = torch.as_tensor([[float(reward)]], device=self.device)
        terminated_batch = torch.as_tensor([[terminated]], device=self.device)
        truncated_batch = torch.as_tensor([[truncated]], device=self.device)
        return {"obs": self._curr_obs}, reward_batch, terminated_batch, truncated_batch, info

    def get_observations(self) -> dict[str, Any]:
        return {"obs": self._curr_obs}

    @property
    def action_dim(self) -> int:
        return self.env.action_space.shape[0]

    @property
    def actor_obs_dim(self) -> int:
        return self.env.observation_space.shape[0]

    @property
    def critic_obs_dim(self) -> int:
        return self.env.observation_space.shape[0]

    @property
    def num_envs(self) -> int:
        return 1

    def close(self) -> None:
        self.env.close()

    def render(self) -> None:
        self.env.render()
