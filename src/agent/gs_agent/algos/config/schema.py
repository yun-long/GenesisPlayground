from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat
from pathlib import Path
from gs_agent.modules.config.schema import MLPConfig, NetworkBackboneConfig


class OptimizerType(GenesisEnum):
    ADAM = "ADAM"
    ADAMW = "ADAMW"
    SGD = "SGD"


class AlgorithmType(GenesisEnum):
    PPO = "PPO"
    BC = "BC"


class PPOArgs(BaseModel):
    """Configuration for PPO algorithm."""

    model_config = genesis_pydantic_config(frozen=True)

    # Algorithm type
    algorithm_type: AlgorithmType = AlgorithmType.PPO

    # Network architecture
    policy_backbone: NetworkBackboneConfig = MLPConfig()
    critic_backbone: NetworkBackboneConfig = MLPConfig()

    # Learning rates
    lr: PositiveFloat = 3e-4
    """Policy learning rate"""

    # Value function learning rate
    value_lr: PositiveFloat | None = None
    """None means use the same learning rate as the policy"""

    # Discount and GAE
    gamma: PositiveFloat = Field(default=0.99, ge=0, le=1)
    gae_lambda: PositiveFloat = Field(default=0.95, ge=0, le=1)

    # PPO specific
    clip_ratio: PositiveFloat = 0.2
    value_loss_coef: PositiveFloat = 1.0
    entropy_coef: NonNegativeFloat = 0.0
    max_grad_norm: PositiveFloat = 1.0
    target_kl: PositiveFloat = 0.02

    # Training
    num_epochs: NonNegativeInt = 10
    num_mini_batches: NonNegativeInt = 4
    rollout_length: NonNegativeInt = 1000

    # Optimizer
    optimizer_type: OptimizerType = OptimizerType.ADAM
    weight_decay: NonNegativeFloat = 0.0


class BCArgs(BaseModel):
    """Configuration for BC algorithm."""

    model_config = genesis_pydantic_config(frozen=True)

    # Algorithm type
    algorithm_type: AlgorithmType = AlgorithmType.BC

    # Network architecture
    policy_backbone: NetworkBackboneConfig = MLPConfig()
    teacher_backbone: NetworkBackboneConfig = MLPConfig()

    # Learning rates
    lr: PositiveFloat = 3e-4
    """Policy learning rate"""

    # Teacher path
    teacher_path: Path 

    # Training
    num_epochs: NonNegativeInt = 10
    num_mini_batches: NonNegativeInt = 4
    rollout_length: NonNegativeInt = 1000
    max_buffer_size: NonNegativeInt = 1_000_000

    # Optimizer
    optimizer_type: OptimizerType = OptimizerType.ADAM
    weight_decay: NonNegativeFloat = 0.0


AlgorithmArgs = PPOArgs | BCArgs
