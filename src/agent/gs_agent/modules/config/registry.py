from gs_agent.modules.config.schema import ActivationType, MLPConfig

# ------------------------------------------------------------
# MLP Config
# ------------------------------------------------------------

DEFAULT_MLP = MLPConfig(
    hidden_dims=(256, 256, 128),
    activation=ActivationType.RELU,
)
