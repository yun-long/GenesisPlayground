import json
import os
import threading
import time
from datetime import datetime
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

        # Trajectory recording
        self.recording = False
        self.trajectory_data = []
        self.recording_start_time = None

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
        print("r - Start/Stop Recording Trajectory")
        print("p - Replay Latest Trajectory")
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
                # Stop recording when scene resets
                if self.recording:
                    self.stop_recording()
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

        # Record trajectory data if recording
        if self.recording and command is not None:
            self._record_trajectory_step(command, obs)

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
        toggle_recording = keyboard.KeyCode.from_char('r') in pressed_keys
        replay_trajectory = keyboard.KeyCode.from_char('p') in pressed_keys

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

        # Handle recording toggle
        if toggle_recording:
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()

        # Handle replay
        if replay_trajectory:
            self.replay_latest_trajectory()

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

    def start_recording(self) -> None:
        """Start recording trajectory data."""
        if not self.recording:
            self.recording = True
            self.trajectory_data = []
            self.recording_start_time = time.time()
            print("🎬 Started recording trajectory...")

    def stop_recording(self) -> None:
        """Stop recording and save trajectory to disk."""
        if self.recording:
            self.recording = False
            if self.trajectory_data:
                self._save_trajectory()
            else:
                print("No trajectory data to save.")
            self.trajectory_data = []
            self.recording_start_time = None

    def _record_trajectory_step(self, command: TeleopCommand, obs: dict[str, Any]) -> None:
        """Record a single step of trajectory data."""
        if not self.recording:
            return

        timestamp = time.time() - (self.recording_start_time or 0)
        
        # Extract relevant data
        step_data = {
            "timestamp": timestamp,
            "command": {
                "position": command.position.tolist(),
                "orientation": command.orientation.tolist(),
                "gripper_close": command.gripper_close,
                "reset_scene": command.reset_scene,
                "quit_teleop": command.quit_teleop
            },
            "observation": {}
        }

        # Add observation data if available
        if obs:
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    step_data["observation"][key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    step_data["observation"][key] = value.detach().cpu().numpy().tolist()
                else:
                    step_data["observation"][key] = value

        self.trajectory_data.append(step_data)

    def _save_trajectory(self) -> None:
        """Save trajectory data to disk."""
        if not self.trajectory_data:
            return

        # Create trajectories directory if it doesn't exist
        os.makedirs("trajectories", exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectories/so101_pick_place_{timestamp}.json"

        # Prepare trajectory metadata
        trajectory_info = {
            "metadata": {
                "robot": "SO101",
                "task": "pick_and_place",
                "recording_date": datetime.now().isoformat(),
                "duration_seconds": self.trajectory_data[-1]["timestamp"] if self.trajectory_data else 0,
                "num_steps": len(self.trajectory_data),
                "movement_speed": self.movement_speed,
                "rotation_speed": self.rotation_speed
            },
            "trajectory": self.trajectory_data
        }

        try:
            with open(filename, 'w') as f:
                json.dump(trajectory_info, f, indent=2)
            print(f"💾 Trajectory saved to: {filename}")
            print(f"   - Duration: {trajectory_info['metadata']['duration_seconds']:.2f}s")
            print(f"   - Steps: {trajectory_info['metadata']['num_steps']}")
        except Exception as e:
            print(f"❌ Failed to save trajectory: {e}")

    def replay_latest_trajectory(self) -> None:
        """Replay the most recent trajectory."""
        trajectories = self.list_trajectories()
        if not trajectories:
            print("❌ No trajectory files found. Record a trajectory first with 'r'.")
            return
        
        latest_trajectory = trajectories[0]
        print(f"🎬 Replaying latest trajectory: {os.path.basename(latest_trajectory)}")
        self.replay(latest_trajectory)

    def replay(self, traj_file_path: str) -> None:
        """Replay a trajectory from a saved file."""
        if not os.path.exists(traj_file_path):
            print(f"❌ Trajectory file not found: {traj_file_path}")
            return

        if not hasattr(self, '_teleop_env') or self._teleop_env is None:
            print("❌ No environment set. Call set_environment() first.")
            return

        try:
            # Load trajectory
            with open(traj_file_path, 'r') as f:
                trajectory_info = json.load(f)
            
            trajectory = trajectory_info['trajectory']
            print(f"🎬 Replaying trajectory: {traj_file_path}")
            print(f"   - Steps: {len(trajectory)}")
            print(f"   - Duration: {trajectory_info['metadata']['duration_seconds']:.2f}s")
            print(f"   - Task: {trajectory_info['metadata']['task']}")
            
            # Reset environment to initial state
            print("🔄 Resetting environment...")
            self._teleop_env.reset_idx(None)
            
            # Wait for viewer to be ready
            print("⏳ Waiting for viewer to be ready...")
            print("👀 Look for the Genesis viewer window - it should show the robot and cube!")
            time.sleep(2.0)  # 2 second pause for viewer initialization
            
            # Replay each step
            for i, step_data in enumerate(trajectory):
                # Create command from step data
                command = TeleopCommand(
                    position=np.array(step_data['command']['position']),
                    orientation=np.array(step_data['command']['orientation']),
                    gripper_close=step_data['command']['gripper_close'],
                    reset_scene=step_data['command']['reset_scene'],
                    quit_teleop=step_data['command']['quit_teleop']
                )
                
                # Apply command to environment
                self._teleop_env.apply_command(command)
                
                # Step the environment multiple times to keep viewer active
                for _ in range(10):  # Step 10 times per command for smooth visualization
                    self._teleop_env.step()
                    time.sleep(0.02)  # 20ms between simulation steps
                
                # Print progress every 10% of trajectory
                if (i + 1) % max(1, len(trajectory) // 10) == 0:
                    progress = (i + 1) / len(trajectory) * 100
                    print(f"   Progress: {progress:.0f}% ({i + 1}/{len(trajectory)} steps)")
            
            print("✅ Trajectory replay completed!")
            
        except Exception as e:
            print(f"❌ Failed to replay trajectory: {e}")

    def list_trajectories(self) -> list[str]:
        """List all available trajectory files."""
        if not os.path.exists("trajectories"):
            return []
        
        trajectory_files = []
        for filename in os.listdir("trajectories"):
            if filename.startswith("so101_pick_place_") and filename.endswith(".json"):
                filepath = os.path.join("trajectories", filename)
                trajectory_files.append(filepath)
        
        # Sort by modification time (newest first)
        trajectory_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return trajectory_files