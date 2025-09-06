import pickle
import os
import threading
import time
from datetime import datetime
from typing import Any, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from pynput import keyboard

from gs_agent.bases.env_wrapper import BaseEnvWrapper
from gs_env.common.bases.base_env import BaseEnv


# Constants for trajectory management
TRAJECTORY_DIR = "trajectories"
TRAJECTORY_FILENAME_PREFIX = "so101_pick_place_"
TRAJECTORY_FILE_EXTENSION = ".pkl"

# Type alias for trajectory step data
TrajectoryStep = dict[str, Any]


class TeleopCommand:
    """6-DOF end-effector command for robot control."""

    def __init__(
        self,
        position: NDArray[np.float64],  # [3] xyz position
        orientation: NDArray[np.float64],  # [3] roll, pitch, yaw in radians
        gripper_close: bool = False,
        reset_scene: bool = False,
        quit_teleop: bool = False,
        absolute_pose: bool = False,   # <-- NEW
        # NEW:
        absolute_joints: bool = False,
        joint_targets: NDArray[np.float64] | None = None,
    ) -> None:
        self.position: NDArray[np.float64] = position
        self.orientation: NDArray[np.float64] = orientation
        self.gripper_close: bool = gripper_close
        self.reset_scene: bool = reset_scene
        self.quit_teleop: bool = quit_teleop
        self.absolute_pose: bool = absolute_pose
        self.absolute_joints: bool = absolute_joints
        self.joint_targets: NDArray[np.float64] | None = joint_targets


class TeleopWrapper(BaseEnvWrapper):
    """Teleop wrapper that follows the GenesisEnvWrapper pattern."""
    def __init__(
        self,
        env: Any,
        device: torch.device = torch.device("cpu"),
        movement_speed: float = 0.01,
        rotation_speed: float = 0.05,
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
        self.last_command: TeleopCommand | None = None
        self.pending_reset: bool = False

        # Trajectory recording
        self.recording = False
        self.trajectory_data: list[TrajectoryStep] = []
        self.recording_start_time: float | None = None
        self.in_initial_state = True  # Track if we're in initial state after reset

        # Initialize current pose from environment if available
        # Note: This might fail if environment isn't fully initialized yet
        # The pose will be initialized later when needed

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
        obs = self.env.get_observation() or {}
        return torch.tensor([]), obs

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment with teleop input."""
        # Process keyboard input and create command
        command = self._process_input()

        # Apply command to environment
        if command:
            self.env.apply_command(command)
            self.last_command = command

            # If reset command was sent, mark for pose reinitialization in next step
            if command.reset_scene:
                self.pending_reset = True
                # Stop recording when scene resets
                if self.recording:
                    self.stop_recording()
                # Mark that we're now in initial state (can start recording)
                self.in_initial_state = True
                # NEW: prevent immediate follow-up movement from any stuck keys
                with self.lock:
                    self.pressed_keys.clear()

        # Scene stepping is now handled by the environment's apply_command method

        # CHANGED: after a reset, sync cached pose from the actual env pose
        if self.pending_reset:
            self._sync_pose_from_env()
            self.pending_reset = False

        # Get observations
        obs = self.env.get_observation() or {}

        # Record trajectory data if recording
        if self.recording and command is not None:
            self._record_trajectory_step(command, obs)

        # Return teleop-specific format (rewards/termination not applicable)
        return (
            torch.tensor([]),          # next_obs (not used in teleop)
            torch.tensor([]),          # reward (not applicable for teleop)
            torch.tensor([]),          # terminated (not applicable for teleop)
            torch.tensor([]),          # truncated (not applicable for teleop)
            obs                        # observations
        )

    def get_observations(self) -> torch.Tensor:
        """Get current observations."""
        if hasattr(self, 'self.env') and self.env is not None:
            obs = self.env.get_observation()
            if obs is None:
                return torch.tensor([])
        return torch.tensor([])

    def _initialize_current_pose(self) -> None:
        """Initialize current pose from environment."""
        try:
            if self.env is not None:
                obs = self.env.get_observation()
                if obs is not None:
                    self.current_position = obs['end_effector_pos'].copy()
                    from scipy.spatial.transform import Rotation as R
                    quat = obs['end_effector_quat']
                    rot = R.from_quat(quat)
                    self.current_orientation = rot.as_euler('xyz')
                    return  # success
        except Exception as e:
            print(f"Failed to initialize current pose: {e}")

        # Fallback only if we couldn't read from env
        self.current_position = np.array([0.0, 0.0, 0.3])
        self.current_orientation = np.array([0.0, 0.0, 0.0])

    # NEW: resync cached pose from the environment’s real EE pose
    def _sync_pose_from_env(self) -> None:
        """Reset teleop's cached pose to the environment's actual EE pose."""
        if self.env is None:
            return
        obs = self.env.get_observation()
        if obs is None:
            return
        from scipy.spatial.transform import Rotation as R
        self.current_position = obs['end_effector_pos'].copy()
        self.current_orientation = R.from_quat(obs['end_effector_quat']).as_euler('xyz')

    def _process_input(self) -> TeleopCommand | None:
        """Process keyboard input and return command."""
        with self.lock:
            pressed_keys = self.pressed_keys.copy()

        # Always process gripper and special commands, even if no movement keys are pressed
        gripper_close = keyboard.Key.space in pressed_keys
        reset_scene = keyboard.KeyCode.from_char('u') in pressed_keys
        quit_teleop = keyboard.Key.esc in pressed_keys
        toggle_recording = keyboard.KeyCode.from_char('r') in pressed_keys

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

        # Handle recording toggle (only on key press, not while held)
        current_recording_key_state = keyboard.KeyCode.from_char('r') in pressed_keys
        if current_recording_key_state and not self.last_recording_key_state:
            if self.recording:
                self.stop_recording()
            else:
                # Only allow starting recording if we're in initial state
                if self.in_initial_state:
                    self.start_recording()
                else:
                    print("⚠️  Can only start recording from initial state after reset. Press 'u' to reset first.")
                    print("   💡 Recording must start immediately after scene reset to capture initial target and cube positions.")
        self.last_recording_key_state = current_recording_key_state


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
            # Mark that we're no longer in initial state once movement starts
            if self.in_initial_state:
                self.in_initial_state = False

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
            
            # Record initial state (target and cube positions)
            self._record_initial_state()
            
            print("🎬 Started recording trajectory...")

    def _record_initial_state(self) -> None:
        """Record the initial state of the environment (target and cube positions)."""
        if not hasattr(self, 'self.env') or self.env is None:
            return
        
        initial_state = {
            "timestamp": 0.0,
            "command": {
                "position": [0.0, 0.0, 0.0],  # Dummy command for initial state
                "orientation": [0.0, 0.0, 0.0],
                "gripper_close": False,
                "reset_scene": False,
                "quit_teleop": False
            },
            "target": {
                "position": [0.0, 0.0, 0.0],  # Will be filled below
                "orientation": [0.0, 0.0, 0.0]
            },
            "observation": {},
            "is_initial_state": True  # Mark this as initial state
        }
        
        # Add target location (debug sphere position)
        if hasattr(self.env, 'target_location'):
            initial_state["target_location"] = self.env.target_location.tolist()
            initial_state["target"]["position"] = self.env.target_location.tolist()
        
        # Add cube position and orientation
        if hasattr(self.env, 'entities') and 'cube' in self.env.entities:
            cube_entity = self.env.entities['cube']
            initial_state["cube_state"] = {
                "position": np.array(cube_entity.get_pos()).tolist(),
                "orientation": np.array(cube_entity.get_quat()).tolist()
            }
        
        # Add robot initial state
        if hasattr(self.env, 'entities') and 'robot' in self.env.entities:
            robot_obs = self.env.entities['robot'].get_observation()
            if robot_obs:
                for key, value in robot_obs.items():
                    if isinstance(value, np.ndarray):
                        initial_state["observation"][key] = value.tolist()
                    elif isinstance(value, torch.Tensor):
                        initial_state["observation"][key] = value.detach().cpu().numpy().tolist()
                    else:
                        initial_state["observation"][key] = value
        
        self.trajectory_data.append(initial_state)

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
            "target": {
                "position": command.position.tolist(),  # Target position (same as command for teleop)
                "orientation": command.orientation.tolist()  # Target orientation (same as command for teleop)
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
        
        # Add target location (debug sphere position) if available
        if hasattr(self, 'self.env') and self.env is not None:
            if hasattr(self.env, 'target_location'):
                step_data["target_location"] = self.env.target_location.tolist()
            
            # Add cube position and orientation to trajectory data
            if hasattr(self.env, 'entities') and 'cube' in self.env.entities:
                cube_entity = self.env.entities['cube']
                step_data["cube_state"] = {
                    "position": np.array(cube_entity.get_pos()).tolist(),
                    "orientation": np.array(cube_entity.get_quat()).tolist()
                }

        # Add robot joint positions (authoritative)
        if hasattr(self, 'self.env') and self.env is not None:
            robot = getattr(self.env, 'entities', {}).get('robot')
            if robot is not None and hasattr(robot, 'entity'):
                try:
                    q = robot.entity.get_qpos()
                    if isinstance(q, torch.Tensor):
                        q = q.detach().cpu().numpy()
                    step_data["robot_joints"] = np.asarray(q).tolist()
                except Exception as e:
                    print(f"Warn: failed to read robot joints for recording: {e}")

        self.trajectory_data.append(step_data)

    def _save_trajectory(self) -> None:
        """Save trajectory data to disk."""
        if not self.trajectory_data:
            return

        # Create trajectories directory if it doesn't exist
        os.makedirs(TRAJECTORY_DIR, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{TRAJECTORY_DIR}/{TRAJECTORY_FILENAME_PREFIX}{timestamp}{TRAJECTORY_FILE_EXTENSION}"

        # Prepare trajectory metadata
        trajectory_info = {
            "metadata": {
                "robot": "SO101",
                "task": "pick_and_place",
                "recording_date": datetime.now().isoformat(),
                "duration_seconds": self.trajectory_data[-1]["timestamp"] if self.trajectory_data else 0,
                "num_steps": len(self.trajectory_data),
                "movement_speed": self.movement_speed,
                "rotation_speed": self.rotation_speed,
                "includes_target_data": True,
                "data_format": {
                    "command": "Robot control commands (position, orientation, gripper, etc.)",
                    "target": "Target positions and orientations for visualization",
                    "target_location": "Debug sphere position (where cube should be placed for success)",
                    "observation": "Robot state observations (joint positions, end-effector pose, etc.)"
                }
            },
            "trajectory": self.trajectory_data
        }

        with open(filename, 'wb') as f:
            pickle.dump(trajectory_info, f)
        print(f"💾 Trajectory saved to: {filename}")
        print(f"   - Duration: {trajectory_info['metadata']['duration_seconds']:.2f}s")
        print(f"   - Steps: {trajectory_info['metadata']['num_steps']}")

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

        if not hasattr(self, 'self.env') or self.env is None:
            print("❌ No environment set. Call set_environment() first.")
            return

        # Load trajectory
        with open(traj_file_path, 'rb') as f:
            trajectory_info = pickle.load(f)
        
        trajectory = trajectory_info['trajectory']
        print(f"🎬 Replaying trajectory: {traj_file_path}")
        print(f"   - Steps: {len(trajectory)}")
        print(f"   - Duration: {trajectory_info['metadata']['duration_seconds']:.2f}s")
        print(f"   - Task: {trajectory_info['metadata']['task']}")
        
        # Reset environment to initial state
        print("🔄 Resetting environment...")
        self.env.reset_idx(torch.IntTensor([0]))
        
        # Clear any existing debug objects to prevent duplicates
        if hasattr(self.env, 'scene'):
            self.env.scene.clear_debug_objects()
        
        # Wait for viewer to be ready
        print("⏳ Waiting for viewer to be ready...")
        time.sleep(self.viewer_init_delay)  # Configurable pause for viewer initialization
        
        # Restore initial cube and target positions from recorded trajectory
        initial_state = None
        for step_data in trajectory:
            if step_data.get('is_initial_state', False):
                initial_state = step_data
                break
        
        if initial_state:
            print("📍 Restoring initial cube and target positions...")
            # Restore target location
            if 'target_location' in initial_state and hasattr(self.env, 'target_location'):
                # Clear any existing debug objects first
                if hasattr(self.env, 'scene'):
                    self.env.scene.clear_debug_objects()
                self.env.target_location = np.array(initial_state['target_location'])
                self.env._draw_target_visualization(self.env.target_location)
            
            # Restore cube position and orientation
            if 'cube_state' in initial_state and hasattr(self.env, 'entities') and 'cube' in self.env.entities:
                cube_entity = self.env.entities['cube']
                cube_state = initial_state['cube_state']
                cube_entity.set_pos(cube_state['position'])
                cube_entity.set_quat(cube_state['orientation'])
                print(f"   - Cube restored to position: {cube_state['position']}")
                print(f"   - Target restored to position: {initial_state['target_location']}")
            
            # Reset robot's internal target pose tracking to match recorded initial state
            if hasattr(self.env, 'entities') and 'robot' in self.env.entities:
                robot = self.env.entities['robot']
                if hasattr(robot, 'target_position') and hasattr(robot, 'target_orientation'):
                    # Use recorded robot initial state if available
                    if 'observation' in initial_state and 'end_effector_pos' in initial_state['observation']:
                        # Set robot target to recorded initial pose
                        robot.target_position = np.array(initial_state['observation']['end_effector_pos'])
                        if 'end_effector_quat' in initial_state['observation']:
                            from scipy.spatial.transform import Rotation as R
                            quat = initial_state['observation']['end_effector_quat']
                            rot = R.from_quat(quat)
                            robot.target_orientation = rot.as_euler('xyz')
                        else:
                            # Fallback to current pose if no recorded orientation
                            from scipy.spatial.transform import Rotation as R
                            pos, quat = robot.get_ee_pose()
                            if pos is not None:
                                rot = R.from_quat(quat)
                                robot.target_orientation = rot.as_euler('xyz')
                        
                        # Set previous targets to match current targets (no delta movement initially)
                        robot.previous_target_position = robot.target_position.copy()
                        robot.previous_target_orientation = robot.target_orientation.copy()
                        
                        # Also restore joint positions if available in recorded data
                        if 'joint_positions' in initial_state['observation']:
                            joint_pos = np.array(initial_state['observation']['joint_positions'])
                            if hasattr(robot, 'entity') and hasattr(robot, 'motors_dof'):
                                robot.entity.set_qpos(joint_pos[:-1], robot.motors_dof)
                                print(f"   - Robot joints restored to recorded initial positions")
                        
                        print(f"   - Robot target pose reset to recorded initial pose: {robot.target_position}")
                    else:
                        # Fallback: get current robot pose and set it as the baseline
                        pos, quat = robot.get_ee_pose()
                        if pos is not None:
                            robot.target_position = pos.copy()
                            from scipy.spatial.transform import Rotation as R
                            rot = R.from_quat(quat)
                            robot.target_orientation = rot.as_euler('xyz')
                            robot.previous_target_position = robot.target_position.copy()
                            robot.previous_target_orientation = robot.target_orientation.copy()
                            print(f"   - Robot target pose reset to current pose (fallback)")
        else:
            print("⚠️  No initial state found in trajectory")
        
        # Replay each step with proper timing
        last_timestamp = 0.0
        
        for i, step_data in enumerate(trajectory):
            # Skip initial state step (timestamp 0.0) as it's just for reference
            if step_data.get('is_initial_state', False):
                print("📍 Initial state recorded - target and cube positions captured")
                continue
            
            # Calculate time delay based on actual trajectory timing
            current_timestamp = step_data['timestamp']
            if i > 0:
                time_delay = current_timestamp - last_timestamp
                # Cap the delay to reasonable bounds (10ms to 200ms)
                time_delay = max(0.01, min(0.2, time_delay))
                time.sleep(time_delay)
            
            # Use the actual recorded positions from the trajectory
            # The trajectory contains the correct end-effector positions that the robot should move to
            current_position = np.array(step_data['command']['position'])
            current_orientation = np.array(step_data['command']['orientation'])
            
            
            
            
            # Create command using the recorded positions directly
            joint_targets = None
            if "robot_joints" in step_data:
                joint_targets = np.array(step_data["robot_joints"], dtype=float)

            command = TeleopCommand(
                position=current_position,
                orientation=current_orientation,
                gripper_close=step_data['command']['gripper_close'],
                reset_scene=step_data['command']['reset_scene'],
                quit_teleop=step_data['command']['quit_teleop'],
                absolute_joints=joint_targets is not None,   # NEW
                joint_targets=joint_targets,                  # NEW
            )
            
            
            # Apply command to environment
            self.env.apply_command(command)
            
            # Restore cube position if available in trajectory data
            if 'cube_state' in step_data and hasattr(self.env, 'entities') and 'cube' in self.env.entities:
                cube_entity = self.env.entities['cube']
                cube_state = step_data['cube_state']
                cube_entity.set_pos(cube_state['position'])
                cube_entity.set_quat(cube_state['orientation'])
            
            # Step the environment multiple times to ensure the robot reaches the target
            for _ in range(5):  # Increased for better movement completion
                self.env.step()
            
            # Update last timestamp
            last_timestamp = current_timestamp
            
            # Print progress every 10% of trajectory
            if (i + 1) % max(1, len(trajectory) // 10) == 0:
                progress = (i + 1) / len(trajectory) * 100
                print(f"   Progress: {progress:.0f}% ({i + 1}/{len(trajectory)} steps)")
        
        print("✅ Trajectory replay completed!")

    def list_trajectories(self) -> list[str]:
        """List all available trajectory files."""
        if not os.path.exists(TRAJECTORY_DIR):
            return []
        
        trajectory_files = []
        for filename in os.listdir(TRAJECTORY_DIR):
            if filename.startswith(TRAJECTORY_FILENAME_PREFIX) and filename.endswith(TRAJECTORY_FILE_EXTENSION):
                filepath = os.path.join(TRAJECTORY_DIR, filename)
                trajectory_files.append(filepath)
        
        # Sort by modification time (newest first)
        trajectory_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return trajectory_files
