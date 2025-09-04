import threading
import time
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pynput import keyboard

from gs_agent.bases.env_wrapper import BaseEnvWrapper


class TeleopCommand:
    """6-DOF end-effector command for robot control."""

    def __init__(
        self,
        position: NDArray[np.float64],  # [3] xyz position
        orientation: NDArray[np.float64],  # [3] roll, pitch, yaw in radians
        gripper_close: bool = False,
        reset_scene: bool = False,
        quit_teleop: bool = False,
    ) -> None:
        self.position: NDArray[np.float64] = position
        self.orientation: NDArray[np.float64] = orientation
        self.gripper_close: bool = gripper_close
        self.reset_scene: bool = reset_scene
        self.quit_teleop: bool = quit_teleop


class TeleopWrapper(BaseEnvWrapper):
    """Teleop wrapper that follows the GenesisEnvWrapper pattern."""
    def __init__(
        self,
        env: Any | None = None,
        device: torch.device = torch.device("cpu"),
        movement_speed: float = 0.01,
        rotation_speed: float = 0.05,
    ) -> None:
        super().__init__(env, device)

        # Movement parameters
        self.movement_speed = movement_speed * 2  # Doubled for faster movement
        self.rotation_speed = rotation_speed * 2  # Doubled for faster movement

        # Keyboard state
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = None
        self.running = False

        # Current command state
        self.current_position: NDArray[np.float64] | None = None
        self.current_orientation: NDArray[np.float64] | None = None
        self.last_command: TeleopCommand | None = None
        self.pending_reset: bool = False

        # Initialize current pose from environment if available
        if self.env is not None:
            self._initialize_current_pose()

    def set_environment(self, env: Any) -> None:
        """Set the environment after creation."""
        self._teleop_env = env
        self._teleop_env.initialize()
        self._initialize_current_pose()

    def start(self) -> None:
        """Start keyboard listener."""
        print("Starting teleop wrapper...")

        try:
            if self.listener is None:
                self.listener = keyboard.Listener(
                    on_press=self._on_press,
                    on_release=self._on_release,
                    suppress=False  # Don't suppress system keys
                )
                self.listener.start()
                print("Keyboard listener started.")
        except Exception as e:
            print(f"Failed to start keyboard listener: {e}")
            print("This might be due to macOS accessibility permissions.")
            print("Please grant accessibility permissions to your terminal/Python in System Preferences > Security & Privacy > Privacy > Accessibility")
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
        print("esc - Quit")

    def stop(self) -> None:
        """Stop keyboard listener."""
        self.running = False
        if self.listener:
            self.listener.stop()

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment."""
        if hasattr(self, '_teleop_env') and self._teleop_env is not None:
            self._teleop_env.reset_idx(None)
            obs = self._teleop_env.get_observation()
            if obs is None:
                obs = {}
            return torch.tensor([]), obs
        return torch.tensor([]), {}

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment with teleop input."""
        # Process keyboard input and create command
        command = self._process_input()

        # Apply command to environment
        if command and hasattr(self, '_teleop_env') and self._teleop_env is not None:
            self._teleop_env.apply_command(command)
            self.last_command = command

            # If reset command was sent, mark for pose reinitialization in next step
            if command.reset_scene:
                self.pending_reset = True
                # NEW: prevent immediate follow-up movement from any stuck keys
                with self.lock:
                    self.pressed_keys.clear()

        # Step the environment
        if hasattr(self, '_teleop_env') and self._teleop_env is not None:
            self._teleop_env.step()

        # CHANGED: after a reset, sync cached pose from the actual env pose
        if self.pending_reset:
            self._sync_pose_from_env()
            self.pending_reset = False

        # Get observations
        if hasattr(self, '_teleop_env') and self._teleop_env is not None:
            obs = self._teleop_env.get_observation()
            if obs is None:
                obs = {}
        else:
            obs = {}

        # Return standard step format (empty tensors for compatibility)
        return (
            torch.tensor([]),          # next_obs
            torch.tensor([0.0]),       # reward
            torch.tensor([False]),     # terminated
            torch.tensor([False]),     # truncated
            obs                        # extra_infos
        )

    def get_observations(self) -> torch.Tensor:
        """Get current observations."""
        if hasattr(self, '_teleop_env') and self._teleop_env is not None:
            obs = self._teleop_env.get_observation()
            if obs is None:
                return torch.tensor([])
        return torch.tensor([])

    def _initialize_current_pose(self) -> None:
        """Initialize current pose from environment."""
        try:
            env = getattr(self, '_teleop_env', None) or self.env
            if env is not None:
                obs = env.get_observation()
                if obs is not None:
                    self.current_position = obs['end_effector_pos'].copy()
                    from scipy.spatial.transform import Rotation as R
                    quat = obs['end_effector_quat']
                    rot = R.from_quat(quat)
                    self.current_orientation = rot.as_euler('xyz')
        except Exception as e:
            print(f"Failed to initialize current pose: {e}")
            self.current_position = np.array([0.0, 0.0, 0.3])
            self.current_orientation = np.array([0.0, 0.0, 0.0])

    # NEW: resync cached pose from the environment’s real EE pose
    def _sync_pose_from_env(self) -> None:
        """Reset teleop's cached pose to the environment's actual EE pose."""
        try:
            env = getattr(self, '_teleop_env', None) or self.env
            if env is None:
                return
            obs = env.get_observation()
            if obs is None:
                return
            from scipy.spatial.transform import Rotation as R
            self.current_position = obs['end_effector_pos'].copy()
            self.current_orientation = R.from_quat(obs['end_effector_quat']).as_euler('xyz')
        except Exception as e:
            print(f"Failed to sync teleop pose: {e}")

    def _process_input(self) -> TeleopCommand | None:
        """Process keyboard input and return command."""
        with self.lock:
            pressed_keys = self.pressed_keys.copy()

        # Always process gripper and special commands, even if no movement keys are pressed
        gripper_close = keyboard.Key.space in pressed_keys
        reset_scene = keyboard.KeyCode.from_char('u') in pressed_keys
        quit_teleop = keyboard.Key.esc in pressed_keys

        # Movement keys present?
        movement_keys = {
            keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right,
            keyboard.KeyCode.from_char('n'), keyboard.KeyCode.from_char('m'),
            keyboard.KeyCode.from_char('j'), keyboard.KeyCode.from_char('k')
        }
        has_movement = bool(pressed_keys & movement_keys)

        if not pressed_keys:
            return None

        # Initialize current pose if missing
        if self.current_position is None or self.current_orientation is None:
            self._initialize_current_pose()

        # If still missing but special keys exist, send special-only command
        if self.current_position is None or self.current_orientation is None:
            if gripper_close or reset_scene or quit_teleop:
                return TeleopCommand(
                    position=np.array([0.0, 0.0, 0.0]),
                    orientation=np.array([0.0, 0.0, 0.0]),
                    gripper_close=gripper_close,
                    reset_scene=reset_scene,
                    quit_teleop=quit_teleop
                )
            return None

        new_position = self.current_position.copy()
        new_orientation = self.current_orientation.copy()

        # Position controls
        if keyboard.Key.up in pressed_keys:
            new_position[0] += self.movement_speed
        if keyboard.Key.down in pressed_keys:
            new_position[0] -= self.movement_speed
        if keyboard.Key.right in pressed_keys:
            new_position[1] += self.movement_speed
        if keyboard.Key.left in pressed_keys:
            new_position[1] -= self.movement_speed
        if keyboard.KeyCode.from_char('n') in pressed_keys:
            new_position[2] += self.movement_speed
        if keyboard.KeyCode.from_char('m') in pressed_keys:
            new_position[2] -= self.movement_speed

        # Orientation controls
        if keyboard.KeyCode.from_char('j') in pressed_keys:
            new_orientation[2] += self.rotation_speed
        if keyboard.KeyCode.from_char('k') in pressed_keys:
            new_orientation[2] -= self.rotation_speed

        # If reset is pressed this tick, send only the reset flag (no motion)
        if reset_scene:
            command = TeleopCommand(
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([0.0, 0.0, 0.0]),
                gripper_close=gripper_close,
                reset_scene=reset_scene,
                quit_teleop=quit_teleop
            )
        else:
            command = TeleopCommand(
                position=new_position,
                orientation=new_orientation,
                gripper_close=gripper_close,
                reset_scene=reset_scene,
                quit_teleop=quit_teleop
            )

        # Update cached pose only if there was movement (not on reset)
        if has_movement and not reset_scene:
            self.current_position = new_position.copy()
            self.current_orientation = new_orientation.copy()

        return command

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