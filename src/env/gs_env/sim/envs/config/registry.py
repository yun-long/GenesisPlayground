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


EnvArgsRegistry["put_bowl_inside_microwave_default"] = EnvArgs(
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_teleop"],
    objects_args=[],  # Objects are created directly in the environment
    sensors_args=[],
    reward_args={},
    img_resolution=(480, 270),
    env_config={
        "show_viewer": True,
        "show_FPS": False,
        "table_pos": (0.0, 0.0, 0.05),
        "table_size": (0.6, 0.6, 0.1),
        "bowl_scale": 1 / 5000,
        "microwave_scale": 0.3,
        "microwave_pos": (0.2, 0.2, 0.18),
        "microwave_euler": (0, 0, 30),
    },
)


EnvArgsRegistry["hang_lifebuoy_default"] = EnvArgs(
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_teleop"],
    objects_args=[],  # Objects are created directly in the environment
    sensors_args=[],
    reward_args={},
    img_resolution=(480, 270),
    env_config={
        "show_viewer": True,
        "show_FPS": False,
        "table_pos": (0.0, 0.0, 0.05),
        "table_size": (0.6, 0.6, 0.1),
        "lifebuoy_scale": 0.03,
        "hanger_scale": (10, 5, 10),
        "hanger_pos": (0.05, -0.2, 0.15),
        "hanger_euler": (90, 0, 90),
    },
)


EnvArgsRegistry["sweep_table_default"] = EnvArgs(
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_teleop"],
    objects_args=[],  # Objects are created directly in the environment
    sensors_args=[],
    reward_args={},
    img_resolution=(480, 270),
    env_config={
        "show_viewer": True,
        "show_FPS": False,
        "table_pos": (0.0, 0.0, 0.05),
        "table_size": (0.6, 0.6, 0.1),
        "broom_scale": (1 / 400, 1 / 800, 1 / 400),
        "broom_pos": (0.05, -0.2, 0.15),
        "broom_euler": (90, 0, 90),
        "trashbox_size": (0.03, 0.03, 0.03),
        "target_zone_pos": (0.1, 0.3, 0.045),
        "target_zone_size": (0.3, 0.3, 0.003),
    },
)


EnvArgsRegistry["pick_cube_default"] = EnvArgs(
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_teleop"],
    objects_args=[],  # Objects are created directly in the environment
    sensors_args=[],
    reward_args={},
    img_resolution=(480, 270),
    env_config={
        "show_viewer": True,
        "show_FPS": False,
        "cube_pos": (0.5, 0.0, 0.07),
        "cube_size": (0.04, 0.04, 0.04),
    },
)
