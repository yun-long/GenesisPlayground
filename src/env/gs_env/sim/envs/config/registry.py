from gs_env.sim.envs.config.schema import EnvArgs, GenesisInitArgs, LeggedRobotEnvArgs
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
    reward_term="reward",
    reward_args={
        "ActionL2Penalty": 0.0,
        "KeypointsAlign": 1.0,
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


EnvArgsRegistry["walk_default"] = LeggedRobotEnvArgs(
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_legged"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Velocity Tracking ###
        "LinVelXYReward": 1.0,
        "AngVelZReward": 1.0,
        "LinVelZPenalty": -0.2,
        "AngVelXYPenalty": -0.1,
        ### Pose Tracking ###
        "OrientationPenalty": -5.0,
        ### Regularization ###
        "TorquePenalty": -0.00001,
        "ActionRatePenalty": -0.01,
        "DofPosLimitPenalty": -10.0,
        "G1BaseHeightPenalty": -30.0,
        "ActionLimitPenalty": -0.1,
    },
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
)
