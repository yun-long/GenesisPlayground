from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import torch


class BaseEnv(ABC):
    """Core simulator/task with a minimal, tensor-only API."""

    def __init__(self, device: torch.device, seed: int | None = None) -> None:
        self.device: Final[torch.device] = device
        self._rng = torch.Generator(device=self.device)  # seeding for any stochastic ops
        if seed is not None:
            self._rng.manual_seed(int(seed))
        self._episode_steps: int = 0
        self._episode_length_limit: int | None = None

    def reset(self) -> None:
        envs_idx = torch.IntTensor(range(self.num_envs))
        self.reset_idx(envs_idx=envs_idx)

    @abstractmethod
    def reset_idx(self, envs_idx: torch.IntTensor) -> None: ...

    @abstractmethod
    def apply_action(self, action: torch.Tensor) -> None: ...

    @abstractmethod
    def get_observations(self) -> dict[str, Any]: ...

    @abstractmethod
    def get_extra_infos(self) -> dict[str, Any]: ...

    @abstractmethod
    def get_terminated(self) -> torch.Tensor: ...

    @abstractmethod
    def get_truncated(self) -> torch.Tensor: ...

    @abstractmethod
    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    @property
    @abstractmethod
    def num_envs(self) -> int: ...

    def observation_spec(self) -> Mapping[str, Any]:
        return {}

    def _setup_camera(
        self,
        pos: tuple[float, float, float] = (1.5, 0.0, 0.7),
        lookat: tuple[float, float, float] = (0.2, 0.0, 0.1),
        fov: int = 50,
        resolution: tuple[int, int] = (640, 480),
    ) -> None:
        """Setup camera for image capture. Override in subclasses that support cameras."""
        # Default implementation does nothing - environments without cameras can ignore this
        # This is intentionally empty to allow environments without cameras to skip camera setup
        return

    def _process_rgb_tensor(
        self, rgb: torch.Tensor | np.ndarray[Any, Any] | None, normalize: bool = True
    ) -> torch.Tensor | None:
        """Process raw RGB tensor from camera render. Common processing logic."""
        if rgb is None:
            return None

        try:
            # Convert numpy array to torch tensor if needed
            if not isinstance(rgb, torch.Tensor):
                # Make a copy to avoid negative stride issues
                rgb = torch.from_numpy(rgb.copy())

            # Convert to proper format: (H, W, C) or (B, H, W, C) -> (B, C, H, W)
            if rgb.dim() == 3:  # Single image (H, W, C)
                rgb = rgb.permute(2, 0, 1).unsqueeze(0)[:, :3]  # -> (1, 3, H, W)
            else:  # Batch (B, H, W, C)
                rgb = rgb.permute(0, 3, 1, 2)[:, :3]  # -> (B, 3, H, W)

            # Normalize if requested
            if normalize:
                rgb = torch.clamp(rgb, min=0.0, max=255.0) / 255.0

            return rgb
        except Exception as e:
            print(f"Warning: Could not process RGB tensor: {e}")
            return None

    def get_rgb_image(self, normalize: bool = True) -> torch.Tensor | None:
        """Capture RGB image from camera (if available). Override in subclasses."""
        return None

    def save_camera_image(self, filename: str = "camera_capture.png") -> None:
        """Save the current camera image to a file for inspection."""
        try:
            import matplotlib.pyplot as plt

            # Get the raw image (not normalized)
            rgb_image = self.get_rgb_image(normalize=False)

            if rgb_image is not None:
                # Convert from (B, C, H, W) to (H, W, C) for display
                if rgb_image.dim() == 4:  # Batch
                    image = rgb_image[0].permute(1, 2, 0).cpu().numpy()
                else:  # Single image
                    image = rgb_image.permute(1, 2, 0).cpu().numpy()

                # Ensure values are in 0-255 range
                image = np.clip(image, 0, 255).astype(np.uint8)

                # Save the image
                plt.imsave(filename, image)
                print(f"Camera image saved to: {filename}")
            else:
                print("No camera image available to save")
        except Exception as e:
            print(f"Error saving camera image: {e}")

    def _add_rgb_to_observations(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Add RGB images to observations dictionary. Call this from get_observations()."""
        # Capture RGB image
        rgb_images = {}
        rgb_image = self.get_rgb_image()
        if rgb_image is not None:
            rgb_images["camera"] = rgb_image

        # Add RGB images to observations
        observations["rgb_images"] = rgb_images
        observations["depth_images"] = {}  # Placeholder for future depth camera support

        return observations

    def action_spec(self) -> Mapping[str, Any]:
        return {}

    def episode_steps(self) -> int:
        return self._episode_steps

    def set_time_limit(self, max_steps: int | None) -> None:
        self._episode_length_limit = max_steps
