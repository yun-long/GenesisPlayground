import os
import pickle
import threading
import time
from typing import Any

import numpy as np
import torch
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from gs_agent.bases.env_wrapper import BaseEnvWrapper

# Constants for trajectory management
TRAJECTORY_DIR = "trajectories"
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
        position: torch.Tensor,  # [3] xyz position
        orientation: torch.Tensor,  # [4] wxyz quaternion
        gripper_close: bool = False,
        # absolute_pose: bool = False,  # <-- NEW
        # NEW:
        # absolute_joints: bool = False,
        # joint_targets: NDArray[np.float64] | None = None,
    ) -> None:
        self.position: torch.Tensor = position
        self.orientation: torch.Tensor = orientation
        self.gripper_close: bool = gripper_close
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

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        with self.lock:
            self.pressed_keys.discard(key)

    def get_cmd(self) -> set[keyboard.Key]:
        return self.pressed_keys


class KeyboardWrapper(BaseEnvWrapper):
    """Keyboard wrapper that follows the GenesisEnvWrapper pattern."""

    def __init__(
        self,
        env: Any,
        device: torch.device = _DEFAULT_DEVICE,
        movement_speed: float = _DEFAULT_MOVEMENT_SPEED,
        rotation_speed: float = _DEFAULT_ROTATION_SPEED,
        replay_steps_per_command: int = 3,
        trajectory_filename_prefix: str = "franka_pick_place_",
    ) -> None:
        super().__init__(env, device)

        # Movement parameters
        self.movement_speed = movement_speed * 2  # Doubled for faster movement
        self.rotation_speed = 0.05  # Match robot's direct_joint_change for consistent behavior

        # Replay parameters
        self.replay_steps_per_command = replay_steps_per_command

        # Trajectory management
        self.trajectory_filename_prefix = trajectory_filename_prefix

        # Keyboard state
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = None
        self.running = False

        # Current command state
        self.last_command: KeyboardCommand | None = None

        # Trajectory recording
        self.recording = False
        self.record_requested = False
        self.reset_requested = False
        self.trajectory_data: list[TrajectoryStep] = []

        # input device
        self.clients = {}
        self.clients["keyboard"] = KeyboardDevice()
        self.clients["keyboard"].start()

        # Initialize current pose from environment if available
        # Note: This might fail if environment isn't fully initialized yet
        # The pose will be initialized later when needed

    def set_environment(self, env: Any) -> None:
        """Set the environment after creation."""
        # Store environment reference (can't reassign self.env due to Final)
        self._env = env

        self.target_position, self.target_orientation = self._env.get_ee_pose()
        print("🎮 Keyboard controls are now active!")

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
        print("j - Rotate around Z-axis (Left)")
        print("k - Rotate around Z-axis (Right)")
        print("u - Reset Scene")
        print("space - Press to close gripper, release to open gripper")
        print("r - Start/Stop Recording Trajectory")
        print("esc - Quit")

    def stop(self) -> None:
        """Stop keyboard listener."""
        self.running = False
        if self.recording:
            self._stop_recording()
        if self.listener:
            self.listener.stop()

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment."""
        self._env.reset_idx(torch.IntTensor([0]))
        obs = self._convert_observation_to_dict()
        return torch.tensor([]), obs

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment with teleop input."""
        # Process keyboard input and create command
        command = self._process_input()

        self.last_command = command
        self._env.apply_action(command)

        # Get observations
        obs = self._convert_observation_to_dict()

        # Record trajectory data if recording
        if self.recording:
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
        return self._env.get_observations()

    def _convert_observation_to_dict(self) -> dict[str, Any]:
        """Convert tensor observation to dictionary format for teleop compatibility."""

        # Get cube position
        cube_pos = np.array(self._env.entities["cube"].get_pos())
        cube_quat = np.array(self._env.entities["cube"].get_quat())

        # Create observation dictionary (for teleop compatibility)
        observation = {
            "ee_pose": self._env.entities["robot"].ee_pose,
            # "end_effector_pos": robot_obs["end_effector_pos"],
            # "end_effector_quat": robot_obs["end_effector_quat"],
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "rgb_images": {},  # No cameras in this simple setup
            "depth_images": {},  # No depth sensors in this simple setup
        }

        return observation

    def _process_input(self) -> KeyboardCommand:
        """Process keyboard input and return command."""

        with self.lock:
            pressed_keys = self.clients["keyboard"].pressed_keys.copy()
        # reset scene:

        # TODO: reset scene
        if self.reset_requested:
            # Reset the environment
            if hasattr(self, "_env") and hasattr(self._env, "reset_idx"):
                self.reset()
                if self.recording:
                    # create a new trajectory file
                    self.stop_recording()
                    self.start_recording()
                with self.lock:
                    self.pressed_keys.clear()
            self.reset_requested = False

        if self.record_requested:
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()
            self.record_requested = False

        # get ee target pose
        is_close_gripper = False
        dpos = 0.005
        drot = 0.01
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
                # 围绕Z轴逆时针旋转（向左）
                # 保持末端垂直向下，只改变围绕Z轴的旋转
                current_rotation = R.from_quat(self.target_orientation[0, :].cpu().numpy())
                # 获取当前Z轴旋转角度
                current_euler = current_rotation.as_euler("xyz", degrees=False)
                # 只修改Z轴旋转角度
                new_euler = [current_euler[0] + drot, current_euler[1], current_euler[2]]
                new_rotation = R.from_euler("xyz", new_euler)
                self.target_orientation[0, :] = torch.from_numpy(new_rotation.as_quat()).to(self.target_orientation.device)
            elif key == keyboard.KeyCode.from_char("k"):
                # 围绕x轴顺时针旋转（向右）
                # 保持末端垂直向下，只改变围绕x轴的旋转
                current_rotation = R.from_quat(self.target_orientation[0, :].cpu().numpy())
                # 获取当前x轴旋转角度
                current_euler = current_rotation.as_euler("xyz", degrees=False)
                # 只修改x轴旋转角度
                new_euler = [current_euler[0] - drot, current_euler[1], current_euler[2]]
                new_rotation = R.from_euler("xyz", new_euler)
                self.target_orientation[0, :] = torch.from_numpy(new_rotation.as_quat()).to(self.target_orientation.device)
            elif key == keyboard.Key.space:
                is_close_gripper = True

        command = KeyboardCommand(
            position=self.target_position,
            orientation=self.target_orientation,
            gripper_close=is_close_gripper,
        )
        return command

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key press events."""
        with self.lock:
            self.pressed_keys.add(key)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key release events."""
        with self.lock:
            self.pressed_keys.discard(key)
            if key == keyboard.KeyCode.from_char("r"):
                self.record_requested = True
            if key == keyboard.KeyCode.from_char("u"):
                self.reset_requested = True

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

    def _stop_recording(self) -> None:
        """Stop recording trajectory data."""
        if self.recording:
            self.recording = False
            print(f"Recording stopped. Captured {len(self.trajectory_data)} steps.")
            # Could save trajectory data here if needed
            self.trajectory_data.clear()

    def _record_trajectory_step(self, command: KeyboardCommand, obs: dict[str, Any]) -> None:
        """Record a step of trajectory data."""
        if not self.recording:
            return

        # Create trajectory step with simulation time
        step_data: TrajectoryStep = {
            "timestamp": self._env.scene.cur_t,
            "command": {
                "position": command.position.clone(),
                "orientation": command.orientation.clone(),
                "gripper_close": command.gripper_close,
            },
            "observation": obs.copy(),
        }

        self.trajectory_data.append(step_data)

    def close(self) -> None:
        """Close the wrapper."""
        self.stop()

    def render(self) -> None:
        """Render the environment."""
        pass

    # Trajectory Recording and Replay Methods

    def start_recording(self) -> None:
        """Start recording trajectory data."""
        if self.recording:
            print("⚠️  Already recording trajectory!")
            return

        self.recording = True
        self.trajectory_data = []
        print("🔴 Started recording trajectory...")
        print("   Press 'r' again to stop recording and save.")

    def stop_recording(self) -> None:
        """Stop recording and save trajectory data."""
        if not self.recording:
            print("⚠️  Not currently recording!")
            return

        self.recording = False

        print(f"🔴 Stopping recording... data_len={len(self.trajectory_data)}")

        if not self.trajectory_data:
            print("⚠️  No trajectory data recorded!")
            return

        # Save trajectory to file
        filename = self._save_trajectory()
        print("✅ Stopped recording trajectory!")
        print(f"   Steps recorded: {len(self.trajectory_data)}")
        print(f"   Saved to: {filename}")

        # Clear trajectory data
        self.trajectory_data = []

    def _save_trajectory(self) -> str:
        """Save trajectory data to file."""
        # Create trajectories directory if it doesn't exist
        os.makedirs(TRAJECTORY_DIR, exist_ok=True)

        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"{self.trajectory_filename_prefix}{timestamp}{TRAJECTORY_FILE_EXTENSION}"
        filepath = os.path.join(TRAJECTORY_DIR, filename)

        # Save trajectory data
        with open(filepath, "wb") as f:
            pickle.dump(self.trajectory_data, f)

        return filepath

    def _load_trajectory(self, filename: str | None = None) -> list[TrajectoryStep] | None:
        """Load a trajectory file. If no filename provided, loads the most recent one."""
        if not os.path.exists(TRAJECTORY_DIR):
            print("⚠️  No trajectories directory found!")
            return None

        # If no filename provided, find the most recent trajectory file
        if filename is None:
            # Find all trajectory files
            trajectory_files = [
                f
                for f in os.listdir(TRAJECTORY_DIR)
                if f.startswith(self.trajectory_filename_prefix)
                and f.endswith(TRAJECTORY_FILE_EXTENSION)
            ]

            if not trajectory_files:
                print("⚠️  No trajectory files found!")
                return None

            # Sort by modification time and get the latest
            trajectory_files.sort(
                key=lambda x: os.path.getmtime(os.path.join(TRAJECTORY_DIR, x)), reverse=True
            )
            filename = trajectory_files[0]

        filepath = os.path.join(TRAJECTORY_DIR, filename)

        if not os.path.exists(filepath):
            print(f"⚠️  Trajectory file not found: {filename}")
            return None

        # Load trajectory data
        try:
            with open(filepath, "rb") as f:
                trajectory_data = pickle.load(f)
            print(f"📁 Loaded trajectory from: {filename}")
            print(f"   Steps: {len(trajectory_data)}")
            return trajectory_data
        except Exception as e:
            print(f"❌ Failed to load trajectory: {e}")
            return None

    def replay_trajectory(self, filename: str | None = None) -> None:
        """Replay a specific trajectory file or the most recent one if no filename provided."""
        print("🎬 Starting trajectory replay...")

        # Set running flag to allow replay
        self.running = True

        try:
            # Load trajectory data
            trajectory_data = self._load_trajectory(filename)

            if trajectory_data is None:
                return

            if not trajectory_data:
                print("⚠️  Empty trajectory data!")
                return

            print(f"🎯 Replaying {len(trajectory_data)} steps...")

            self._env.reset_idx(torch.IntTensor([0]))
            print("🔄 Environment reset to initial state")

            # Replay each step
            for i, step_data in enumerate(trajectory_data):
                if not self.running:
                    print("⏹️  Replay stopped by user")
                    break

                # Extract command from step data
                cmd_data = step_data["command"]
                command = KeyboardCommand(
                    position=cmd_data["position"],
                    orientation=cmd_data["orientation"],
                    gripper_close=cmd_data["gripper_close"],
                )

                # Apply command to environment
                self._env.apply_action(command)

                if i % 50 == 0:  # Progress update every 50 steps
                    print(f"   Step {i}/{len(trajectory_data)}")

            print("✅ Trajectory replay completed!")

        finally:
            # Always reset running flag
            self.running = False

    def replay_latest_trajectory(self) -> None:
        """Replay the most recent trajectory."""
        self.replay_trajectory()  # Call the new function without filename
