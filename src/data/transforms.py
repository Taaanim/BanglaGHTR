"""
Computer vision transforms and aspect-ratio preserving resizing for HTR.
Includes training-time augmentations for robustness.
"""

import numpy as np
import torch
import random
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple
from scipy.ndimage import map_coordinates, gaussian_filter


class AspectRatioPadResize:
    """
    Resizes text line/word images by fixing the target height and scaling the width
    proportionally to preserve character aspect ratio. Pads remainder with white pixels (255).
    """
    def __init__(self, target_height: int = 64, max_width: int = 1024, pad_value: int = 255):
        self.target_height = target_height
        self.max_width = max_width
        self.pad_value = pad_value

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, int]:
        """
        Args:
            img: PIL Image in grayscale ('L') mode.
        Returns:
            tensor: [1, target_height, max_width] normalized to [0, 1].
            valid_width: The unpadded width before white padding.
        """
        w, h = img.size
        if h == 0:
            h = 1
        scale = self.target_height / float(h)
        new_w = max(1, min(int(w * scale), self.max_width))

        resized = img.resize((new_w, self.target_height), Image.Resampling.BILINEAR)

        # Create padded canvas
        canvas = Image.new("L", (self.max_width, self.target_height), self.pad_value)
        canvas.paste(resized, (0, 0))

        # Convert to float tensor [1, H, W] in [0, 1]
        arr = np.array(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        return tensor, new_w


class CharacterTransform:
    """Transform for 28x28 isolated character images."""
    def __init__(self, image_size: Tuple[int, int] = (28, 28), invert: bool = True):
        self.image_size = image_size
        self.invert = invert

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.Resampling.BILINEAR)

        arr = np.array(img, dtype=np.uint8)
        if self.invert:
            # Converts black background (0) with white strokes (255)
            # to natural white paper (255) with dark strokes (0)
            arr = 255 - arr

        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor


class HTRAugmentation:
    """
    Training-time data augmentation for handwritten text recognition.
    Applies randomized visual distortions to improve model robustness.

    Each augmentation is applied independently with a configurable probability.
    Designed to simulate real-world handwriting variations:
    - Geometric: slight rotation, shear, perspective changes
    - Photometric: contrast, brightness, noise
    - Morphological: elastic distortion, erosion/dilation
    """
    def __init__(
        self,
        p: float = None,
        p_affine: float = 0.3,
        p_elastic: float = 0.2,
        p_noise: float = 0.3,
        p_contrast: float = 0.3,
        p_blur: float = 0.15,
        p_erosion: float = 0.1,
        rotation_range: float = 3.0,
        elastic_alpha: float = 8.0,
        elastic_sigma: float = 3.0,
        noise_std: float = 0.03
    ):
        if p is not None:
            p_affine = p_elastic = p_noise = p_contrast = p_blur = p_erosion = p
        self.p_affine = p_affine
        self.p_elastic = p_elastic
        self.p_noise = p_noise
        self.p_contrast = p_contrast
        self.p_blur = p_blur
        self.p_erosion = p_erosion
        self.rotation_range = rotation_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.noise_std = noise_std

    def _random_affine(self, img: Image.Image) -> Image.Image:
        """Applies small random rotation and scale."""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        # Slight scale perturbation (0.95 to 1.05)
        scale = random.uniform(0.95, 1.05)
        w, h = img.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((max(1, new_w), max(1, new_h)), Image.Resampling.BILINEAR)
        img = img.rotate(angle, fillcolor=255, resample=Image.Resampling.BILINEAR)
        return img

    def _elastic_distortion(self, arr: np.ndarray) -> np.ndarray:
        """Applies elastic distortion to simulate pen pressure variations."""
        h, w = arr.shape
        dx = gaussian_filter(np.random.randn(h, w), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter(np.random.randn(h, w), self.elastic_sigma) * self.elastic_alpha

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = (
            np.clip(y + dy, 0, h - 1).astype(np.float64),
            np.clip(x + dx, 0, w - 1).astype(np.float64)
        )

        return map_coordinates(arr.astype(np.float64), indices, order=1, mode='constant', cval=255).astype(np.uint8)

    def _add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Adds Gaussian noise to simulate scanner noise."""
        noise = torch.randn_like(tensor) * self.noise_std
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def _adjust_contrast(self, img: Image.Image) -> Image.Image:
        """Randomly adjusts contrast."""
        factor = random.uniform(0.7, 1.4)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def _apply_blur(self, img: Image.Image) -> Image.Image:
        """Applies mild Gaussian blur."""
        radius = random.choice([0.5, 0.8, 1.0])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def _morphological_erosion(self, arr: np.ndarray) -> np.ndarray:
        """Simulates pen thickness variation via erosion/dilation."""
        # Simple 3x3 min filter (erosion) or max filter (dilation)
        from scipy.ndimage import minimum_filter, maximum_filter
        if random.random() < 0.5:
            return minimum_filter(arr, size=2)  # Thin strokes
        else:
            return maximum_filter(arr, size=2)  # Thick strokes

    def augment_pil(self, img: Image.Image) -> Image.Image:
        """Applies augmentations to a PIL Image (before resize/pad)."""
        if random.random() < self.p_affine:
            img = self._random_affine(img)

        if random.random() < self.p_contrast:
            img = self._adjust_contrast(img)

        if random.random() < self.p_blur:
            img = self._apply_blur(img)

        if random.random() < self.p_elastic:
            arr = np.array(img)
            arr = self._elastic_distortion(arr)
            img = Image.fromarray(arr, mode='L')

        if random.random() < self.p_erosion:
            arr = np.array(img)
            arr = self._morphological_erosion(arr)
            img = Image.fromarray(arr, mode='L')

        return img

    def augment_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies tensor-level augmentations (after normalization)."""
        if random.random() < self.p_noise:
            tensor = self._add_gaussian_noise(tensor)
        return tensor

    def __call__(self, x):
        if isinstance(x, Image.Image):
            return self.augment_pil(x)
        elif isinstance(x, torch.Tensor):
            return self.augment_tensor(x)
        return x


class AugmentedAspectRatioPadResize:
    """
    AspectRatioPadResize with optional training augmentations.
    Wraps the base transform and applies HTRAugmentation before resizing.
    """
    def __init__(
        self,
        target_height: int = 64,
        max_width: int = 1024,
        pad_value: int = 255,
        augment: bool = True,
        is_training: bool = None
    ):
        if is_training is not None:
            augment = is_training
        self.base_transform = AspectRatioPadResize(target_height, max_width, pad_value)
        self.augmentor = HTRAugmentation() if augment else None

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, int]:
        # Apply PIL-level augmentations before resize
        if self.augmentor is not None:
            img = self.augmentor.augment_pil(img)

        tensor, valid_width = self.base_transform(img)

        # Apply tensor-level augmentations after normalization
        if self.augmentor is not None:
            tensor = self.augmentor.augment_tensor(tensor)

        return tensor, valid_width
