from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
import torch

from gs_env.common.bases.base_env import BaseEnv
from gs_env.common.rewards import ActionL2Penalty, KeypointsAlign
from gs_env.common.utils.math_utils import quat_mul
from gs_env.common.utils.misc_utils import get_space_dim
from gs_env.sim.envs.config.schema import EnvArgs
from gs_env.sim.robots.manipulators import FrankaRobot
from gs_env.sim.scenes import FlatScene

_DEFAULT_DEVICE = torch.device("cpu")


class GoalReachingEnv(BaseEnv):
    """
    Goal Reaching Environment for Manipulators.
    """

    def __init__(
        self,
        args: EnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(device=device)
        self._num_envs = num_envs
        self._show_viewer = show_viewer
        self._args = args

        if not gs._initialized:  # noqa: SLF001
            gs.init(performance_mode=True)

        # == setup the scene ==
        self._scene = FlatScene(
            num_envs=self._num_envs,
            args=args.scene_args,
            show_viewer=self._show_viewer,
            img_resolution=args.img_resolution,
        )

        # == setup the robot ==
        self._robot = FrankaRobot(
            num_envs=self._num_envs,
            scene=self._scene.scene,
            args=args.robot_args,
            device=self.device,
        )

        # == setup target entity
        self._target = self._scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), collision=False),
            material=gs.materials.Rigid(gravity_compensation=1),
        )
        # == build the scene ==
        self._scene.build()

        # == setup reward scalars and functions ==
        dt = self._scene.scene.dt
        self.rwd_action_l2 = ActionL2Penalty(scale=args.reward_args["rew_actions"] * dt)
        self.rwd_keypoints = KeypointsAlign(scale=args.reward_args["rew_keypoints"] * dt)

        # some auxiliary variables
        self._max_sim_time = 3.0  # seconds
        #
        self._init()
        self.reset()

    def _init(self) -> None:
        # specify the space attributes
        self._action_space = self._robot.action_space
        self._observation_space = gym.spaces.Dict(
            {
                "pose_vec": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "ee_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "ref_position": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),  # 3D position
                "ref_quat": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                ),  # 4D quaternion
            }
        )
        self._info_space = gym.spaces.Dict({})

        #
        self.goal_pose = torch.zeros(self.num_envs, 7, dtype=torch.float32, device=self._device)
        self.time_since_reset = torch.zeros(self.num_envs, device=self._device)
        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, device=self._device, unit_length=0.5
        )
        self.action_buf = torch.zeros((self.num_envs, self.action_dim), device=self._device)

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        if len(envs_idx) == 0:
            return
        self._robot.reset(envs_idx=envs_idx)
        # Generate random goal positions
        num_reset = len(envs_idx)
        # Random position within reachable workspace
        random_x = torch.rand(num_reset, device=self._device) * 0.3 + 0.15  # 0.15 to 0.45
        random_y = (torch.rand(num_reset, device=self._device) - 0.5) * 0.4  # -0.2 to 0.2
        random_z = torch.rand(num_reset, device=self._device) * 0.2 + 0.1  # 0.1 to 0.3

        self.goal_pose[envs_idx, 0] = random_x
        self.goal_pose[envs_idx, 1] = random_y
        self.goal_pose[envs_idx, 2] = random_z

        self.goal_pose[envs_idx] = torch.tensor(
            [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self._device
        )
        q_down = torch.tensor(
            [0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=self._device
        ).repeat(num_reset, 1)
        random_yaw = torch.rand(num_reset, device=self._device) * 2 * np.pi - np.pi  # -pi to pi
        random_yaw *= 0.25  # reduce the range to [-pi/4, pi/4]
        # random_yaw *= 0.0  # reduce the range to [-pi/4, pi/4]
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(num_reset, device=self._device),
                torch.zeros(num_reset, device=self._device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        self.goal_pose[envs_idx, 3:7] = quat_mul(q_yaw, q_down)

        #
        self.time_since_reset[envs_idx] = 0.0
        self._target.set_qpos(self.goal_pose[envs_idx], envs_idx=envs_idx)  # type: ignore

    def get_terminated(self) -> torch.Tensor:
        reset_buf = self.get_truncated()
        return reset_buf

    def get_truncated(self) -> torch.Tensor:
        time_out_buf = self.time_since_reset > self._max_sim_time
        return time_out_buf

    def get_observations(self) -> torch.Tensor:
        # Current end-effector pose
        ee_pos, ee_quat = self._robot.ee_pose[:, :3], self._robot.ee_pose[:, 3:7]
        #
        pos_diff = ee_pos - self.goal_pose[:, :3]
        obs_components = [
            pos_diff,  # 3D position difference
            ee_quat,  # current orientation (4D quaternion)
            self.goal_pose,  # goal pose (7D: pos + quat)
        ]
        return torch.cat(obs_components, dim=-1)

    def apply_action(self, action: torch.Tensor) -> None:
        action = self.rescale_action(action)
        self.action_buf[:] = action.clone().to(self._device)
        self.time_since_reset += self._scene.scene.dt
        self._robot.apply_action(action=action)
        self._scene.scene.step()

    def get_extra_infos(self) -> dict[str, Any]:
        return dict()

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        action_scale: torch.Tensor = torch.tensor(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32, device=self._device
        )
        return action * action_scale

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Squared-L2 action penalty
        reward_actions = self.rwd_action_l2(
            {
                "action": self.action_buf,  #
            }
        )
        # Key-point alignment
        reward_keypoints = self.rwd_keypoints(
            {
                "pose_a": self._robot.ee_pose,  # current pose
                "pose_b": self.goal_pose,  # goal pose
                "key_offsets": self.keypoints_offset,
            }
        )
        #
        reward_total = reward_actions + reward_keypoints

        reward_dict = {
            "reward_actions": reward_actions,
            "reward_keypoints": reward_keypoints,
            "reward_total": reward_total,
        }
        return reward_total, reward_dict

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    @staticmethod
    def get_keypoint_offsets(
        batch_size: int, device: torch.device, unit_length: float = 0.5
    ) -> torch.Tensor:
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at 0.
        """
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],  # origin
                    [-1.0, 0, 0],  # x-negative
                    [1.0, 0, 0],  # x-positive
                    [0, -1.0, 0],  # y-negative
                    [0, 1.0, 0],  # y-positive
                    [0, 0, -1.0],  # z-negative
                    [0, 0, 1.0],  # z-positive
                ],
                device=device,
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    @property
    def action_dim(self) -> int:
        act_dim = get_space_dim(self._action_space) - 1  # -1 for the gripper actio
        return act_dim

    @property
    def actor_obs_dim(self) -> int:
        return get_space_dim(self._observation_space)

    @property
    def critic_obs_dim(self) -> int:
        num_critic_obs = get_space_dim(self._observation_space)
        return num_critic_obs
