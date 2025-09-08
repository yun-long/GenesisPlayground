import torch
from torch import nn
from torch.distributions import Normal

from gs_agent.bases.network_backbone import NetworkBackbone
from gs_agent.bases.policy import Policy


# === Gaussian Policy === #
class GaussianPolicy(Policy):
    def __init__(self, policy_backbone: NetworkBackbone, action_dim: int) -> None:
        super().__init__(policy_backbone, action_dim)
        self.mu = nn.Linear(self.backbone.output_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * 1.0)
        Normal.set_default_validate_args(False)

        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the policy.

        Args:
            obs: Typed observation batch.
            deterministic: Whether to use deterministic action.

        Returns:
            GaussianPolicyOutput: Policy output.
        """
        # Convert observation to tensor format
        dist = self._dist_from_obs(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        # Compute log probability with tanh transformation
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def _dist_from_obs(self, obs: torch.Tensor) -> Normal:
        feature = self.backbone(obs)
        action_mu = self.mu(feature)
        action_std = torch.exp(self.log_std)
        return Normal(action_mu, action_std.expand_as(action_mu))

    def evaluate_log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Evaluate the log probability of the action.

        Args:
            obs: Typed observation batch.
            act: Action tensor.

        Returns:
            Log probability of the action.
        """
        dist = self._dist_from_obs(obs)
        return dist.log_prob(act).sum(-1, keepdim=True)

    def entropy_on(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the entropy of the action distribution.

        Args:
            obs: Typed observation batch.

        Returns:
            Entropy of the action distribution.
        """
        dist = self._dist_from_obs(obs)
        return dist.entropy().sum(-1)

    def get_action_shape(self) -> tuple[int, ...]:
        return (self.action_dim,)


class DeterministicPolicy(Policy):
    def __init__(self, policy_backbone: NetworkBackbone, action_dim: int) -> None:
        super().__init__(policy_backbone, action_dim)
        self.mu = nn.Linear(self.backbone.output_dim, self.action_dim)
        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the policy."""
        action = self.mu(obs)
        return action


# === Categorical Policy === #
class CategoricalPolicy(Policy):
    # TODO: implement

    def __init__(self, policy_backbone: NetworkBackbone, action_dim: int) -> None:
        raise NotImplementedError("CategoricalPolicy is not implemented")
