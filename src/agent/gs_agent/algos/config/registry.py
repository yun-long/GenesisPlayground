from gs_agent.algos.config.schema import OptimizerType, PPOArgs
from gs_agent.modules.config.registry import DEFAULT_MLP

# default PPO config
PPO_DEFAULT = PPOArgs(
    policy_backbone=DEFAULT_MLP,
    critic_backbone=DEFAULT_MLP,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    max_grad_norm=1.0,
    target_kl=0.02,
    num_epochs=10,
    num_mini_batches=4,
    rollout_length=1000,
    optimizer_type=OptimizerType.ADAM,
    weight_decay=0.0,
)

# gym pendulum PPO config
PPO_PENDULUM_MLP = PPOArgs(
    # Network architecture
    policy_backbone=DEFAULT_MLP,
    critic_backbone=DEFAULT_MLP,
    lr=3e-4,
    value_lr=None,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    max_grad_norm=1.0,
    target_kl=0.02,
    num_epochs=10,
    num_mini_batches=4,
    rollout_length=1000,
    optimizer_type=OptimizerType.ADAM,
    weight_decay=0.0,
)


# goal reaching PPO config
PPO_GOAL_REACHING_MLP = PPOArgs(
    policy_backbone=DEFAULT_MLP,
    critic_backbone=DEFAULT_MLP,
    lr=3e-4,
    value_lr=None,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    max_grad_norm=1.0,
    target_kl=0.02,
    num_epochs=10,
    num_mini_batches=4,
    rollout_length=32,
    optimizer_type=OptimizerType.ADAM,
    weight_decay=0.0,
)
