import genesis as gs
from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel

from gs_env.sim.objects.config.schema import ObjectArgs
from gs_env.sim.robots.config.schema import ManipulatorRobotArgs
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
    robot_args: ManipulatorRobotArgs
    objects_args: list[ObjectArgs]
    sensors_args: list[SensorArgs]
    reward_args: dict[str, float]
    img_resolution: tuple[int, int]
