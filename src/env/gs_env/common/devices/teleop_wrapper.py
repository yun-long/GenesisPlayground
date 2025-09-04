import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pynput import keyboard

from gs_env.common.bases.device import Device


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


class TeleopObservation:
    """Observation data from task environment."""

    def __init__(
        self,
        joint_positions: NDArray[np.float64],  # [n_joints] current joint angles
        end_effector_pos: NDArray[np.float64],  # [3] current end-effector position
        end_effector_quat: NDArray[np.float64],  # [4] current end-effector quaternion
        rgb_images: dict[str, NDArray[np.float64]] | None = None,  # camera images
        depth_images: dict[str, NDArray[np.float64]] | None = None,  # depth images
    ) -> None:
        self.joint_positions: NDArray[np.float64] = joint_positions
        self.end_effector_pos: NDArray[np.float64] = end_effector_pos
        self.end_effector_quat: NDArray[np.float64] = end_effector_quat
        self.rgb_images = rgb_images or {}
        self.depth_images = depth_images or {}


class TeleopWrapper(Device[TeleopCommand]):
    """Robot-agnostic teleop wrapper that sends 6-DOF commands to task environments."""

    def __init__(self, environment: Any, movement_speed: float = 0.01, rotation_speed: float = 0.05) -> None:
        super().__init__()

        # Environment that this teleop wrapper controls
        self.environment = environment

        # Movement parameters
        self.movement_speed = movement_speed * 2  # Doubled for faster movement
        self.rotation_speed = rotation_speed * 2  # Doubled for faster movement

        # Keyboard state
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = None
        self.running = False

        # Current command state - will be initialized by the robot
        self.current_position: NDArray[np.float64] | None = None  # Will be set by robot's initial pose
        self.current_orientation: NDArray[np.float64] | None = None  # Will be set by robot's initial pose

    def start(self) -> None:
        """Start keyboard listener (like original KeyboardDevice)."""
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

    def run(self) -> None:
        """Run the main control loop in a separate thread."""
        import threading
        self.control_thread = threading.Thread(target=self._run_control_loop, daemon=True)
        self.control_thread.start()
        
        # Wait for the control thread to finish
        self.control_thread.join()

    def stop(self) -> None:
        """Stop keyboard listener."""
        self.running = False

        if self.listener:
            self.listener.stop()
            self.listener.join()
            self.listener = None

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key press events."""
        with self.lock:
            self.pressed_keys.add(key)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key release events."""
        with self.lock:
            self.pressed_keys.discard(key)

    def process_input(self) -> bool:
        """Process keyboard input and send commands. Returns True if should continue, False if should quit."""
        # Get current key state
        with self.lock:
            keys = self.pressed_keys.copy()

        # Only process if there are actually keys pressed
        if not keys:
            return True

        # Process movement keys
        position_delta: NDArray[np.float64] = np.zeros(3)
        orientation_delta: NDArray[np.float64] = np.zeros(3)
        gripper_close = False
        reset_scene = False
        quit_teleop = False

        for key in keys:
            if key == keyboard.Key.up:
                position_delta[0] += self.movement_speed
            elif key == keyboard.Key.down:
                position_delta[0] -= self.movement_speed
            elif key == keyboard.Key.right:
                position_delta[1] += self.movement_speed
            elif key == keyboard.Key.left:
                position_delta[1] -= self.movement_speed
            elif hasattr(key, 'char') and key.char == 'n':
                position_delta[2] += self.movement_speed
            elif hasattr(key, 'char') and key.char == 'm':
                position_delta[2] -= self.movement_speed
            elif hasattr(key, 'char') and key.char == 'j':
                orientation_delta[2] += self.rotation_speed
            elif hasattr(key, 'char') and key.char == 'k':
                orientation_delta[2] -= self.rotation_speed
            elif key == keyboard.Key.space:
                gripper_close = True
            elif hasattr(key, 'char') and key.char == 'u':
                reset_scene = True
            elif key == keyboard.Key.esc:
                quit_teleop = True

        # Only send command if there's actual movement or action
        if (np.any(position_delta != 0) or np.any(orientation_delta != 0) or
            gripper_close or reset_scene or quit_teleop):

            # Initialize position if not set
            if self.current_position is None:
                self.current_position = np.array([0.0, 0.0, 0.3])  # Default position
            if self.current_orientation is None:
                self.current_orientation = np.array([0.0, 0.0, 0.0])  # Default orientation

            # Update current position and orientation
            self.current_position = self.current_position + position_delta
            self.current_orientation = self.current_orientation + orientation_delta

            # Create command
            command = TeleopCommand(
                position=self.current_position.copy(),
                orientation=self.current_orientation.copy(),
                gripper_close=gripper_close,
                reset_scene=reset_scene,
                quit_teleop=quit_teleop
            )

            # Send command to task environment if callback is set
            if self.command_callback:
                try:
                    self.command_callback(command)
                except Exception as e:
                    print(f"Error sending command: {e}")

            # Handle special commands
            if reset_scene:
                print("Resetting scene...")
            elif quit_teleop:
                print("Quitting teleop...")
                return False

        return True

    def get_state(self) -> TeleopCommand:
        """Get current teleop state (for compatibility with Device base class)."""
        if self.current_position is not None and self.current_orientation is not None:
            return TeleopCommand(
                position=self.current_position.copy(),
                orientation=self.current_orientation.copy(),
                gripper_close=False,
                reset_scene=False,
                quit_teleop=False
            )
        else:
            return TeleopCommand(
                position=np.zeros(3),
                orientation=np.zeros(3),
                gripper_close=False,
                reset_scene=False,
                quit_teleop=False
            )

    def send_cmd(self, cmd: TeleopCommand) -> None:
        """Send command to device (for compatibility with Device base class)."""
        # This method is not used in teleop mode
        pass

    def set_command_callback(self, callback: Any) -> None:
        """Set callback function for sending commands to task environment."""
        self.command_callback = callback

    def set_observation_callback(self, callback: Any) -> None:
        """Set callback function for receiving observations from task environment."""
        self.observation_callback = callback

    def update_current_pose(self, position: NDArray[np.float64], orientation: NDArray[np.float64]) -> None:
        """Update current position and orientation from task environment."""
        self.current_position = position.copy()
        self.current_orientation = orientation.copy()

    def get_current_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get current position and orientation."""
        if self.current_position is not None and self.current_orientation is not None:
            return self.current_position.copy(), self.current_orientation.copy()
        else:
            return np.zeros(3), np.zeros(3)

    def _initialize_current_pose(self) -> None:
        """Initialize current pose from environment."""
        try:
            # Get initial pose from environment
            obs = self.environment.get_observation()
            if obs is not None:
                self.current_position = obs['end_effector_pos'].copy()
                # Convert quaternion to euler angles
                from scipy.spatial.transform import Rotation as R
                quat = obs['end_effector_quat']
                rot = R.from_quat(quat)
                self.current_orientation = rot.as_euler('xyz')
                print(f"Initialized teleop pose: pos={self.current_position}, orient={self.current_orientation}")
        except Exception as e:
            print(f"Failed to initialize current pose: {e}")
            # Use default values
            self.current_position = np.array([0.0, 0.0, 0.3])
            self.current_orientation = np.array([0.0, 0.0, 0.0])

    def _run_control_loop(self) -> None:
        """Run the main control loop."""
        print("SO101 Cube Environment ready for teleop!")
        print("Use keyboard controls to move the robot.")
        print("Press ESC to quit.")

        try:
            # Main simulation loop
            step_count = 0
            while self.running:
                # Process keyboard input
                self._process_input()

                # Step the environment
                self.environment.step()
                step_count += 1

                # Print status every 1000 steps
                if step_count % 1000 == 0:
                    print(f"Running... Step {step_count}")

                # Check for quit command
                if (hasattr(self, 'last_command') and
                    self.last_command and
                    hasattr(self.last_command, 'quit_teleop') and
                    self.last_command.quit_teleop):
                    print("Quit command received, exiting...")
                    break

                # Safety check - exit after 1 hour of running
                if step_count > 180000:  # 1 hour at 50Hz
                    print("Maximum runtime reached, exiting...")
                    break

                # Small delay to control simulation frequency
                time.sleep(0.02)  # 50 Hz

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"Error in control loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            self.stop()
            print("Teleop session ended.")

    def _process_input(self) -> None:
        """Process keyboard input and send commands to environment."""
        with self.lock:
            if not self.pressed_keys:
                return  # No keys pressed, no command to send

        # Initialize current pose if not set
        if self.current_position is None or self.current_orientation is None:
            self._initialize_current_pose()

        # Calculate new position and orientation based on pressed keys
        new_position = self.current_position.copy()
        new_orientation = self.current_orientation.copy()

        with self.lock:
            # Position controls
            if keyboard.Key.up in self.pressed_keys:
                new_position[0] += self.movement_speed  # Move forward (X+)
            if keyboard.Key.down in self.pressed_keys:
                new_position[0] -= self.movement_speed  # Move backward (X-)
            if keyboard.Key.right in self.pressed_keys:
                new_position[1] += self.movement_speed  # Move right (Y+)
            if keyboard.Key.left in self.pressed_keys:
                new_position[1] -= self.movement_speed  # Move left (Y-)
            if keyboard.KeyCode.from_char('n') in self.pressed_keys:
                new_position[2] += self.movement_speed  # Move up (Z+)
            if keyboard.KeyCode.from_char('m') in self.pressed_keys:
                new_position[2] -= self.movement_speed  # Move down (Z-)

            # Orientation controls
            if keyboard.KeyCode.from_char('j') in self.pressed_keys:
                new_orientation[2] += self.rotation_speed  # Rotate counterclockwise
            if keyboard.KeyCode.from_char('k') in self.pressed_keys:
                new_orientation[2] -= self.rotation_speed  # Rotate clockwise

            # Gripper control
            gripper_close = keyboard.KeyCode.from_char('g') in self.pressed_keys

            # Special commands
            reset_scene = keyboard.KeyCode.from_char('r') in self.pressed_keys
            quit_teleop = keyboard.Key.esc in self.pressed_keys

        # Create command
        command = TeleopCommand(
            position=new_position,
            orientation=new_orientation,
            gripper_close=gripper_close,
            reset_scene=reset_scene,
            quit_teleop=quit_teleop
        )

        # Send command to environment
        self.environment.apply_command(command)

        # Update current pose
        self.current_position = new_position.copy()
        self.current_orientation = new_orientation.copy()
