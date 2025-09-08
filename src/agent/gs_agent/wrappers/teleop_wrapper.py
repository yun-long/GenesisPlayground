import threading
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from gs_agent.bases.env_wrapper import BaseEnvWrapper

# Constants for trajectory management
TRAJECTORY_DIR = "trajectories"
TRAJECTORY_FILENAME_PREFIX = "so101_pick_place_"
TRAJECTORY_FILE_EXTENSION = ".pkl"

# Type alias for trajectory step data
TrajectoryStep = dict[str, Any]
_DEFAULT_DEVICE = torch.device("cpu")
_DEFAULT_MOVEMENT_SPEED = 0.01
_DEFAULT_ROTATION_SPEED = 0.05


class KeyboardCommand:
    """6-DOF end-effector command for robot control."""

    def __init__(
        self,
        position: NDArray[np.float64],  # [3] xyz position
        orientation: NDArray[np.float64],  # [4] wxyz quaternion
        gripper_close: bool = False,
        reset_scene: bool = False,
        quit_teleop: bool = False,
        # absolute_pose: bool = False,  # <-- NEW
        # NEW:
        # absolute_joints: bool = False,
        # joint_targets: NDArray[np.float64] | None = None,
    ) -> None:
        self.position: NDArray[np.float64] = position
        self.orientation: NDArray[np.float64] = orientation
        self.gripper_close: bool = gripper_close
        self.reset_scene: bool = reset_scene
        self.quit_teleop: bool = quit_teleop
        # self.absolute_pose: bool = absolute_pose
        # self.absolute_joints: bool = absolute_joints
        # self.joint_targets: NDArray[np.float64] | None = joint_targets


class KeyboardDevice:
    def __init__(self) -> None:
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def start(self) -> None:
        self.listener.start()

    def stop(self) -> None:
        self.listener.stop()
        self.listener.join()

    def on_press(self, key: Key | KeyCode | None) -> None:
        if key is not None:
            with self.lock:
                self.pressed_keys.add(key)

    def on_release(self, key: Key | KeyCode | None) -> None:
        if key is not None:
            with self.lock:
                self.pressed_keys.discard(key)

    def get_cmd(self) -> set[Key | KeyCode]:
        return self.pressed_keys


class KeyboardWrapper(BaseEnvWrapper):
    """Keyboard wrapper that follows the GenesisEnvWrapper pattern."""

    def __init__(
        self,
        env: Any,
        device: torch.device = _DEFAULT_DEVICE,
        movement_speed: float = _DEFAULT_MOVEMENT_SPEED,
        rotation_speed: float = _DEFAULT_ROTATION_SPEED,
        replay_steps_per_command: int = 10,
        viewer_init_delay: float = 2.0,
    ) -> None:
        super().__init__(env, device)

        # Movement parameters
        self.movement_speed = movement_speed * 2  # Doubled for faster movement
        self.rotation_speed = 0.05  # Match robot's direct_joint_change for consistent behavior

        # Replay parameters
        self.replay_steps_per_command = replay_steps_per_command
        self.viewer_init_delay = viewer_init_delay

        # Keyboard state
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = None
        self.running = False

        # Key press tracking for toggle actions
        self.last_recording_key_state = False

        # Current command state
        self.current_position: NDArray[np.float64] | None = None
        self.current_orientation: NDArray[np.float64] | None = None
        self.pending_reset: bool = False

        # Trajectory recording
        self.recording = False
        self.trajectory_data: list[TrajectoryStep] = []
        self.recording_start_time: float | None = None
        self.in_initial_state = True  # Track if we're in initial state after reset

        # input device
        self.clients = {}
        self.clients["keyboard"] = KeyboardDevice()
        self.clients["keyboard"].start()

        # Initialize current pose from environment if available
        try:
            self.target_position, self.target_orientation = self.env.get_ee_pose()
            print("self.target_position", self.target_position.shape)
            print("self.target_orientation", self.target_orientation.shape)
        except (AttributeError, RuntimeError):
            raise RuntimeError("Environment is not fully initialized yet") from None

    def start(self) -> None:
        """Start keyboard listener."""
        print("Starting teleop wrapper...")

        try:
            if self.listener is None:
                self.listener = keyboard.Listener(
                    on_press=self._on_press,
                    on_release=self._on_release,
                    suppress=False,  # Don't suppress system keys
                )
                self.listener.start()
                print("Keyboard listener started.")
        except Exception as e:
            print(f"Failed to start keyboard listener: {e}")
            print("This might be due to macOS accessibility permissions.")
            print(
                "Please grant accessibility permissions to your terminal/Python in System Preferences > Security & Privacy > Privacy > Accessibility"
            )
            return

        self.running = True
        print("Teleop wrapper started.")

        print("Teleop Controls:")
        print("↑ - Move Forward (North)")
        print("↓ - Move Backward (South)")
        print("← - Move Left (West)")
        print("→ - Move Right (East)")
        print("n - Move Up")
        print("m - Move Down")
        print("j - Rotate Counterclockwise")
        print("k - Rotate Clockwise")
        print("u - Reset Scene")
        print("space - Press to close gripper, release to open gripper")
        print("r - Start/Stop Recording Trajectory")
        print("esc - Quit")

    def stop(self) -> None:
        """Stop keyboard listener."""
        self.running = False
        if self.recording:
            self.stop_recording()
        if self.listener:
            self.listener.stop()

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment."""
        self.env.reset_idx(torch.IntTensor([0]))
        obs = self._convert_observation_to_dict() or {}
        return torch.tensor([]), obs

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment with teleop input."""
        # Process keyboard input and create command
        command = self._process_input()

        # Apply command to environment via apply_action
        if command:
            # handle reset and recording
            # If reset command was sent, mark for pose reinitialization in next step
            # if command.reset_scene:
            #     self.pending_reset = True
            #     # Stop recording when scene resets
            #     if self.recording:
            #         self.stop_recording()
            #     # Mark that we're now in initial state (can start recording)
            #     self.in_initial_state = True
            #     # NEW: prevent immediate follow-up movement from any stuck keys
            #     with self.lock:
            #         self.pressed_keys.clear()

            # Pass command directly to apply_action (like goal_reaching_env)
            self.env.apply_action(command)
        else:
            # No command - just step the environment
            self.env.apply_action(torch.tensor([]))

        # CHANGED: after a reset, sync cached pose from the actual env pose
        if self.pending_reset:
            self._sync_pose_from_env()
            self.pending_reset = False

        # Get observations
        obs = self._convert_observation_to_dict() or {}

        # Record trajectory data if recording
        if self.recording and command is not None:
            self._record_trajectory_step(command, obs)

        # Return teleop-specific format (rewards/termination not applicable)
        return (
            torch.tensor([]),  # next_obs
            torch.tensor([0.0]),  # reward
            torch.tensor([False]),  # terminated
            torch.tensor([False]),  # truncated
            obs,  # extra_infos
        )

    def get_observations(self) -> torch.Tensor:
        """Get current observations."""
        if hasattr(self, "env") and self.env is not None:
            obs = self.env.get_observations()
            if obs is None:
                return torch.tensor([])
        return torch.tensor([])

    def _convert_observation_to_dict(self) -> dict[str, Any] | None:
        """Convert tensor observation to dictionary format for teleop compatibility."""
        if self.env is None:
            return None

        # Get cube position
        cube_pos = np.array(self.env.entities["cube"].get_pos())
        cube_quat = np.array(self.env.entities["cube"].get_quat())

        # Create observation dictionary (for teleop compatibility)
        observation = {
            "ee_pose": self.env.entities["robot"].ee_pose,
            # "end_effector_pos": robot_obs["end_effector_pos"],
            # "end_effector_quat": robot_obs["end_effector_quat"],
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "rgb_images": {},  # No cameras in this simple setup
            "depth_images": {},  # No depth sensors in this simple setup
        }

        return observation

    # def _initialize_current_pose(self) -> None:
    #     """Initialize current pose from environment."""
    #     try:
    #         if self.env is not None:
    #             obs = self._convert_observation_to_dict()
    #             if obs is not None:
    #                 self.current_position = obs["end_effector_pos"].copy()
    #                 from scipy.spatial.transform import Rotation as R

    #                 quat = obs["end_effector_quat"]
    #                 rot = R.from_quat(quat)
    #                 self.current_orientation = rot.as_euler("xyz")
    #                 return  # success
    #     except Exception as e:
    #         print(f"Failed to initialize current pose: {e}")

    #     # Fallback only if we couldn't read from env
    #     self.current_position = np.array([0.0, 0.0, 0.3])
    #     self.current_orientation = np.array([0.0, 0.0, 0.0])

    # NEW: resync cached pose from the environment’s real EE pose
    def _sync_pose_from_env(self) -> None:
        """Reset teleop's cached pose to the environment's actual EE pose."""
        if self.env is None:
            return
        obs = self._convert_observation_to_dict()
        if obs is None:
            return
        from scipy.spatial.transform import Rotation as R

        self.current_position = obs["end_effector_pos"].copy()
        self.current_orientation = R.from_quat(obs["end_effector_quat"]).as_euler("xyz")

    def _process_input(self) -> KeyboardCommand | None:
        """Process keyboard input and return command."""
        with self.lock:
            pressed_keys = self.clients["keyboard"].pressed_keys.copy()
        # reset scene:
        reset_flag = False
        reset_flag |= keyboard.KeyCode.from_char("u") in pressed_keys

        # TODO: reset scene
        # if reset_flag:
        #     reset_scene()

        # stop teleoperation
        stop = keyboard.Key.esc in pressed_keys

        # get ee target pose
        is_close_gripper = False
        dpos = 0.002
        for key in pressed_keys:
            if key == keyboard.Key.up:
                self.target_position[0, 0] -= dpos
            elif key == keyboard.Key.down:
                self.target_position[0, 0] += dpos
            elif key == keyboard.Key.right:
                self.target_position[0, 1] += dpos
            elif key == keyboard.Key.left:
                self.target_position[0, 1] -= dpos
            elif key == keyboard.KeyCode.from_char("n"):
                self.target_position[0, 2] += dpos
            elif key == keyboard.KeyCode.from_char("m"):
                self.target_position[0, 2] -= dpos
            elif key == keyboard.KeyCode.from_char("j"):
                raise NotImplementedError("Rotation not implemented")
            elif key == keyboard.KeyCode.from_char("k"):
                raise NotImplementedError("Rotation not implemented")
            elif key == keyboard.Key.space:
                is_close_gripper = True

        command = KeyboardCommand(
            position=self.target_position,
            orientation=self.target_orientation,
            gripper_close=is_close_gripper,
            reset_scene=reset_flag,
            quit_teleop=stop,
        )
        return command

        # # control arm
        # target_quat = target_R.as_quat(scalar_first=True)
        # # target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        # q, err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
        # robot.control_dofs_position(q[:-2], motors_dof)
        # # control gripper
        # if is_close_gripper:
        #     robot.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        # else:
        #     robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

        # # Always process gripper and special commands, even if no movement keys are pressed
        # gripper_close = keyboard.Key.space in pressed_keys
        # reset_scene = keyboard.KeyCode.from_char("u") in pressed_keys
        # quit_teleop = keyboard.Key.esc in pressed_keys
        # keyboard.KeyCode.from_char("r") in pressed_keys

        # # Movement keys present?
        # movement_keys = {
        #     keyboard.Key.up,
        #     keyboard.Key.down,
        #     keyboard.Key.left,
        #     keyboard.Key.right,
        #     keyboard.KeyCode.from_char("n"),
        #     keyboard.KeyCode.from_char("m"),
        #     keyboard.KeyCode.from_char("j"),
        #     keyboard.KeyCode.from_char("k"),
        # }
        # has_movement = bool(pressed_keys & movement_keys)

        # if not pressed_keys:
        #     return None

        # # Initialize current pose if missing
        # # if self.current_position is None or self.current_orientation is None:
        # #     self._initialize_current_pose()
        # ee_pos, ee_quat = self.env.get_ee_pose()
        # print(ee_pos, ee_quat)

        # # Handle recording toggle (only on key press, not while held)
        # current_recording_key_state = keyboard.KeyCode.from_char("r") in pressed_keys
        # if current_recording_key_state and not self.last_recording_key_state:
        #     if self.recording:
        #         self.stop_recording()
        #     else:
        #         # Only allow starting recording if we're in initial state
        #         if self.in_initial_state:
        #             self.start_recording()
        #         else:
        #             print(
        #                 "⚠️  Can only start recording from initial state after reset. Press 'u' to reset first."
        #             )
        #             print(
        #                 "   💡 Recording must start immediately after scene reset to capture initial target and cube positions."
        #             )
        # self.last_recording_key_state = current_recording_key_state

        # # If still missing but special keys exist, send special-only command
        # if self.current_position is None or self.current_orientation is None:
        #     if gripper_close or reset_scene or quit_teleop:
        #         return KeyboardCommand(
        #             position=np.array([0.0, 0.0, 0.0]),
        #             orientation=np.array([0.0, 0.0, 0.0]),
        #             gripper_close=gripper_close,
        #             reset_scene=reset_scene,
        #             quit_teleop=quit_teleop,
        #         )
        #     return None

        # new_position = self.current_position.copy()
        # new_orientation = self.current_orientation.copy()

        # # Position controls
        # if keyboard.Key.up in pressed_keys:
        #     new_position[0] += self.movement_speed
        # if keyboard.Key.down in pressed_keys:
        #     new_position[0] -= self.movement_speed
        # if keyboard.Key.right in pressed_keys:
        #     new_position[1] += self.movement_speed
        # if keyboard.Key.left in pressed_keys:
        #     new_position[1] -= self.movement_speed
        # if keyboard.KeyCode.from_char("n") in pressed_keys:
        #     new_position[2] += self.movement_speed
        # if keyboard.KeyCode.from_char("m") in pressed_keys:
        #     new_position[2] -= self.movement_speed

        # # Orientation controls
        # if keyboard.KeyCode.from_char("j") in pressed_keys:
        #     new_orientation[2] += self.rotation_speed
        # if keyboard.KeyCode.from_char("k") in pressed_keys:
        #     new_orientation[2] -= self.rotation_speed

        # # If reset is pressed this tick, send only the reset flag (no motion)
        # if reset_scene:
        #     command = KeyboardCommand(
        #         position=np.array([0.0, 0.0, 0.0]),
        #         orientation=np.array([0.0, 0.0, 0.0]),
        #         gripper_close=gripper_close,
        #         reset_scene=reset_scene,
        #         quit_teleop=quit_teleop,
        #     )
        # else:
        #     command = KeyboardCommand(
        #         position=new_position,
        #         orientation=new_orientation,
        #         gripper_close=gripper_close,
        #         reset_scene=reset_scene,
        #         quit_teleop=quit_teleop,
        #     )

        # # Update cached pose only if there was movement (not on reset)
        # if has_movement and not reset_scene:
        #     self.current_position = new_position.copy()
        #     self.current_orientation = new_orientation.copy()
        #     # Mark that we're no longer in initial state once movement starts
        #     if self.in_initial_state:
        #         self.in_initial_state = False

        # return command

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key press events."""
        with self.lock:
            self.pressed_keys.add(key)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key release events."""
        with self.lock:
            self.pressed_keys.discard(key)

    # Required properties for BaseEnvWrapper
    @property
    def action_dim(self) -> int:
        return 0

    @property
    def actor_obs_dim(self) -> int:
        return 0

    @property
    def critic_obs_dim(self) -> int:
        return 0

    @property
    def num_envs(self) -> int:
        return 1

    def close(self) -> None:
        """Close the wrapper."""
        self.stop()

    def render(self) -> None:
        """Render the environment."""
        pass

    def stop_recording(self) -> None:
        """Stop trajectory recording."""
        self.recording = False
        # TODO: Implement trajectory saving logic
        pass

    def _record_trajectory_step(self, command: Any, obs: dict[str, Any]) -> None:
        """Record a step in the trajectory."""
        # TODO: Implement trajectory step recording logic
        pass
