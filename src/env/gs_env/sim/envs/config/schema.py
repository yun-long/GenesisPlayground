import genesis as gs
from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel

from gs_env.sim.objects.config.schema import ObjectArgs
from gs_env.sim.robots.config.schema import (
    HumanoidRobotArgs,
    ManipulatorRobotArgs,
    QuadrupedRobotArgs,
)
from gs_env.sim.scenes.config.schema import SceneArgs
from gs_env.sim.sensors.config.schema import SensorArgs


class GenesisInitArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    seed: int
    precision: str
    logging_level: str
    backend: (
        gs.constants.backend | None
    )  # While we avoid using None, this is an exception where it finds a suitable backend automatically


class EnvArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)

    gs_init_args: GenesisInitArgs
    scene_args: SceneArgs
    robot_args: ManipulatorRobotArgs | QuadrupedRobotArgs | HumanoidRobotArgs
    objects_args: list[ObjectArgs]
    sensors_args: list[SensorArgs]
    reward_term: str = "reward"
    reward_args: dict[str, float]
    img_resolution: tuple[int, int]


class LeggedRobotEnvArgs(EnvArgs):
    action_latency: int = 1
    obs_history_len: int = 1
