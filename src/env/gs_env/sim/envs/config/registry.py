from gs_env.sim.envs.config.schema import EnvArgs, GenesisInitArgs
from gs_env.sim.objects.config.registry import ObjectArgsRegistry
from gs_env.sim.robots.config.registry import RobotArgsRegistry
from gs_env.sim.scenes.config.registry import SceneArgsRegistry
from gs_env.sim.sensors.config.registry import SensorArgsRegistry

# ------------------------------------------------------------
# Genesis init
# ------------------------------------------------------------


GenesisInitArgsRegistry: dict[str, GenesisInitArgs] = {}


GenesisInitArgsRegistry["default"] = GenesisInitArgs(
    seed=0,
    precision="32",
    logging_level="info",
    backend=None,
)


# ------------------------------------------------------------
# Manipulation
# ------------------------------------------------------------


EnvArgsRegistry: dict[str, EnvArgs] = {}

EnvArgsRegistry["goal_reach_default"] = EnvArgs(
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_default"],
    objects_args=[ObjectArgsRegistry["box_default"]],
    sensors_args=[
        SensorArgsRegistry["oak_camera_default"],
        SensorArgsRegistry["ee_link_pos"],
        SensorArgsRegistry["ee_link_quat"],
        SensorArgsRegistry["joint_angles"],
        SensorArgsRegistry["gripper_width"],
    ],
    reward_args={
        "rew_actions": 0.0,
        "rew_keypoints": 1.0,
    },
    img_resolution=(480, 270),
)


EnvArgsRegistry["pick_cube_default"] = EnvArgs(
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_teleop"],
    objects_args=[ObjectArgsRegistry["box_default"]],
    sensors_args=[],
    reward_args={},
    img_resolution=(480, 270),
)
