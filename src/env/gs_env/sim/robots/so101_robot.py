import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from typing import Any, Optional, Tuple
from numpy.typing import NDArray

import genesis as gs
from gs_env.common.bases.base_robot import BaseGymRobot


class SO101Robot(BaseGymRobot):
    """SO101 robot implementation with 6-DOF end-effector control."""
    
    def __init__(self, scene: gs.Scene):
        super().__init__()
        
        # Load SO101 robot model
        self.entity: Any = scene.add_entity(
            material=gs.materials.Rigid(gravity_compensation=1),
            morph=gs.morphs.MJCF(
                file="genesis/assets/xml/so101_robot/so101_robot.xml",
                euler=(0, 0, 0),
                convexify=True,
                decompose_robot_error_threshold=0,
            ),
        )

        # SO101 has 6 DOFs: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        n_dofs = self.entity.n_dofs
        self.motors_dof = np.arange(n_dofs - 1)  # All joints except gripper
        self.gripper_dof = np.array([n_dofs - 1])  # Gripper joint
        self.ee_link = self.entity.get_link("gripper")

        # Set initial pose to prevent sideways claw
        try:
            # Compensate for natural tilt with wrist joint adjustment
            adjusted_q = np.array([0.0, 0.0, 0.0, 0.0, -1.5708, 0.0])
            self.entity.set_qpos(adjusted_q)
        except Exception as e:
            print(f"Failed to set initial pose: {e}")

        # Store current target pose for smooth movement
        self.target_position = np.array([0.0, 0.0, 0.3])
        self.target_orientation = np.array([0.0, 0.0, 0.0])
        
        # Store previous target pose for delta calculation
        self.previous_target_position = self.target_position.copy()
        self.previous_target_orientation = self.target_orientation.copy()
    
    def initialize(self) -> None:
        """Initialize the robot after scene is built."""
        # Get current end-effector pose as initial target
        try:
            pos, quat = self.get_ee_pose()
            if pos is not None:
                self.target_position = pos.copy()
                # Convert quaternion to euler angles
                rot = R.from_quat(quat)
                self.target_orientation = rot.as_euler('xyz')
                
                # Initialize previous target positions
                self.previous_target_position = self.target_position.copy()
                self.previous_target_orientation = self.target_orientation.copy()
                
                print(f"Initialized robot target position: {self.target_position}")
                print(f"Initialized robot target orientation: {self.target_orientation}")
        except Exception as e:
            print(f"Failed to get initial pose: {e}")
    
    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        """Reset the robot."""
        # Reset to initial pose
        initial_q: NDArray[np.float64] = np.array([0.0, -0.3, 0.5, 0.0, 0.0, 0.0])
        self.reset_to_pose(initial_q)

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply action to robot (for compatibility with BaseGymRobot interface)."""
        # This method is not used in our teleop setup
        pass
    
    def apply_teleop_command(self, command) -> None:
        """Apply teleop command using hybrid IK + direct joint control (like original script)."""
        # Update target pose
        self.target_position = command.position.copy()
        self.target_orientation = command.orientation.copy()
        
        # Get current joint positions
        current_q = self.entity.get_qpos()
        
        # Use direct joint control for smooth, predictable movement (like original script)
        direct_joint_change = 0.05  # Increased for faster movement
        
        # Calculate position deltas from previous target
        position_delta = self.target_position - self.previous_target_position
        orientation_delta = self.target_orientation - self.previous_target_orientation
        
        # Apply direct joint control based on movement direction
        # This creates smooth, responsive movement in all 6 directions
        if position_delta[0] > 0:  # Move forward (X+)
            current_q[2] -= direct_joint_change  # elbow_flex - extend arm forward
        elif position_delta[0] < 0:  # Move backward (X-)
            current_q[2] += direct_joint_change  # elbow_flex - retract arm backward
            
        if position_delta[1] > 0:  # Move right (Y+)
            current_q[0] -= direct_joint_change  # shoulder_pan - rotate right
        elif position_delta[1] < 0:  # Move left (Y-)
            current_q[0] += direct_joint_change  # shoulder_pan - rotate left
            
        if position_delta[2] > 0:  # Move up (Z+)
            current_q[1] -= direct_joint_change  # shoulder_lift - lift arm up
        elif position_delta[2] < 0:  # Move down (Z-)
            current_q[1] += direct_joint_change  # shoulder_lift - lower arm down
            
        if orientation_delta[2] > 0:  # Rotate counterclockwise
            current_q[4] -= direct_joint_change  # wrist_roll - rotate gripper counter-clockwise
        elif orientation_delta[2] < 0:  # Rotate clockwise
            current_q[4] += direct_joint_change  # wrist_roll - rotate gripper clockwise
        
        # Apply direct joint control for smooth movement
        try:
            self.entity.control_dofs_position(current_q[:-1], self.motors_dof)
        except Exception as e:
            print(f"Direct joint control failed: {e}")
        
        # Update the target visualization to follow the robot's actual end-effector position
        # This ensures the axis and robot move together (like original script)
        actual_ee_pos = None
        actual_ee_quat = None
        try:
            actual_ee_pos = np.array(self.ee_link.get_pos())
            actual_ee_quat = np.array(self.ee_link.get_quat())
            # Update target entity if it exists (for visualization)
            # Note: target_entity is managed by the environment, not the robot
            pass
        except Exception as e:
            print(f"Failed to update target visualization: {e}")
        
        # Optional: Use IK to verify the target is reachable (but don't apply it)
        # This helps debug IK issues without affecting movement (like original script)
        if actual_ee_pos is not None and actual_ee_quat is not None:
            try:
                q, err = self.entity.inverse_kinematics(
                    link=self.ee_link, 
                    pos=actual_ee_pos,  # Use actual position instead of target
                    quat=actual_ee_quat,  # Use actual orientation instead of target
                    return_error=True
                )
                
                # Handle tensor error - take the maximum error value if it's a tensor
                if hasattr(err, 'shape') and len(err.shape) > 0:
                    max_err = float(err.max())
                else:
                    max_err = float(err)
                
                # Only print IK info occasionally to avoid spam
                if max_err > 0.1:
                    print(f"IK target error: {max_err:.4f}")
                    
            except Exception as e:
                # IK failure is not critical since we're using direct joint control
                pass
        
        # Update previous target for next iteration
        self.previous_target_position = self.target_position.copy()
        self.previous_target_orientation = self.target_orientation.copy()
        
        # Control gripper
        if command.gripper_close:
            self.entity.control_dofs_force(np.array([-1.0]), self.gripper_dof)
        else:
            self.entity.control_dofs_force(np.array([1.0]), self.gripper_dof)
    
    def update_teleop_pose(self, teleop_wrapper) -> None:
        """Update teleop wrapper with current robot pose."""
        if teleop_wrapper and self.target_position is not None:
            teleop_wrapper.current_position = self.target_position.copy()
            teleop_wrapper.current_orientation = self.target_orientation.copy()
    
    def reset_to_pose(self, joint_angles: NDArray[np.float64]) -> None:
        """Reset robot to specified joint configuration."""
        self.entity.set_qpos(joint_angles[:-1], self.motors_dof)
        
        # Update target pose to match new configuration
        try:
            pos, quat = self.get_ee_pose()
            if pos is not None:
                self.target_position = pos.copy()
                rot = R.from_quat(quat)
                self.target_orientation = rot.as_euler('xyz')
        except Exception as e:
            print(f"Failed to update target pose: {e}")
    
    def get_ee_pose(self) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """Get current end-effector pose."""
        try:
            pos = np.array(self.ee_link.get_pos())
            quat = np.array(self.ee_link.get_quat())
            return pos, quat
        except Exception as e:
            print(f"Failed to get EE pose: {e}")
            return None, None
    
    def get_joint_positions(self) -> NDArray[np.float64]:
        """Get current joint positions."""
        try:
            return self.entity.get_qpos().copy()
        except Exception as e:
            print(f"Failed to get joint positions: {e}")
            return np.zeros(6)
    
    def get_observation(self):
        """Get robot observation for teleop feedback."""
        joint_pos = self.get_joint_positions()
        ee_pos, ee_quat = self.get_ee_pose()
        
        if ee_pos is None or ee_quat is None:
            return None
        
        return {
            'joint_positions': joint_pos,
            'end_effector_pos': ee_pos,
            'end_effector_quat': ee_quat,
            'target_position': self.target_position.copy(),
            'target_orientation': self.target_orientation.copy()
        }
    
    def is_moving(self) -> bool:
        """Check if robot is currently moving."""
        try:
            current_q = self.entity.get_qpos()
            target_q = self.entity.inverse_kinematics(
                link=self.ee_link,
                pos=self.target_position,
                quat=R.from_euler('xyz', self.target_orientation).as_quat()
            )
            
            # Check if current joints are close to target
            joint_error = np.linalg.norm(current_q[:-1] - target_q[:-1])
            return bool(joint_error > 0.01)  # Threshold for "moving"
            
        except Exception:
            return False
