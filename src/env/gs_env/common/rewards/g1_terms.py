from .leggedrobot_terms import (
    ActionLimitPenalty,  # noqa
    ActionRatePenalty,  # noqa
    AngVelXYPenalty,  # noqa
    AngVelZReward,  # noqa
    BaseHeightPenalty,
    DofPosLimitPenalty,  # noqa
    LinVelXYReward,  # noqa
    LinVelZPenalty,  # noqa
    OrientationPenalty,  # noqa
    TorquePenalty,  # noqa  # noqa
)


### ---- Reward Terms ---- ###
class G1BaseHeightPenalty(BaseHeightPenalty):
    target_height = 0.75
