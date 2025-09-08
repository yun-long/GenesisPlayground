from typing import Any, Final, TypeVar

import torch

from gs_agent.bases.env_wrapper import BaseEnvWrapper

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")

TGSEnv = TypeVar("TGSEnv")


class GenesisEnvWrapper(BaseEnvWrapper):
    def __init__(
        self,
        env: object,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(env, device)
        self.env.reset()
        self._curr_obs = self.env.get_observations()

    # ---------------------------
    # BatchEnvWrapper API (batch)
    # ---------------------------
    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        self.env.reset()
        self._curr_obs = self.env.get_observations()
        return self._curr_obs, self.env.get_extra_infos()

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # apply action
        self.env.apply_action(action)
        # get observations
        next_obs = self.env.get_observations()
        # get reward
        reward, reward_terms = self.env.get_reward()
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        # get terminated
        terminated = self.env.get_terminated()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)
        # get truncated
        truncated = self.env.get_truncated()
        if truncated.dim() == 1:
            truncated = truncated.unsqueeze(-1)
        # reset if terminated or truncated
        done_idx = terminated.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            self.env.reset_idx(done_idx)
        # get extra infos
        extra_infos = self.env.get_extra_infos()
        extra_infos["reward_terms"] = reward_terms
        return next_obs, reward, terminated, truncated, extra_infos

    def get_observations(self) -> torch.Tensor:
        return self.env.get_observations()

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def actor_obs_dim(self) -> int:
        return self.env.actor_obs_dim

    @property
    def critic_obs_dim(self) -> int:
        return self.env.critic_obs_dim

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def close(self) -> None:
        self.env.close()

    def render(self) -> None:
        self.env.render()
