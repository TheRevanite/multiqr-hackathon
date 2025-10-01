"""
Advanced data augmentation pipeline for MED1C QR code detection.
Includes geometric and photometric augmentations specifically designed for robustness.
"""

import torch
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image, ImageFilter
from typing import Tuple, Dict

class QRDetectionAugmentation:
    """
    Advanced augmentation pipeline specifically designed for QR code detection.
    Handles both image transformations and bounding box coordinate updates.
    """
    
    def __init__(self, 
                 image_size: int = 640,
                 geometric_prob: float = 0.5,
                 photometric_prob: float = 0.7,
                 training: bool = True):
        self.image_size = image_size
        self.geometric_prob = geometric_prob
        self.photometric_prob = photometric_prob
        self.training = training
        
        self.rotation_degrees = (-15, 15)  # Conservative for QR codes
        self.scale_range = (0.8, 1.2)
        self.shear_range = (-10, 10)
        self.translate_range = (0.1, 0.1)
        
        self.brightness_factor = (0.7, 1.3)
        self.contrast_factor = (0.8, 1.2)
        self.saturation_factor = (0.8, 1.2)
        self.hue_factor = (-0.1, 0.1)
        self.gamma_range = (0.8, 1.2)
    
    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Apply augmentations to image and update target accordingly."""
        if not self.training:
            image = F.resize(image, (self.image_size, self.image_size))
            return image, target
        
        image, target = self._apply_geometric_augmentations(image, target)
        image = self._apply_photometric_augmentations(image)
        
        image = F.resize(image, (self.image_size, self.image_size))
        
        return image, target
    
    def _apply_geometric_augmentations(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Apply geometric transformations and update bounding boxes."""
        if random.random() > self.geometric_prob:
            return image, target
        
        original_size = image.size
        boxes = target["boxes"].clone()
        
        if random.random() < 0.5:
            image = F.hflip(image)
            boxes[:, [0, 2]] = original_size[0] - boxes[:, [2, 0]]
        
        if random.random() < 0.3:
            angle = random.uniform(*self.rotation_degrees)
            image = F.rotate(image, angle, expand=True, fill=255)
            boxes = self._rotate_boxes(boxes, angle, original_size)
            original_size = image.size
        
        if random.random() < 0.4:
            degrees = random.uniform(-5, 5)
            translate = [random.uniform(-0.05, 0.05) * original_size[0], 
                        random.uniform(-0.05, 0.05) * original_size[1]]
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-5, 5)
            
            image = F.affine(image, degrees, translate, scale, shear, fill=255)
            boxes = self._affine_boxes(boxes, degrees, translate, scale, shear, original_size)
        
        target["boxes"] = boxes
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        return image, target
    
    def _apply_photometric_augmentations(self, image: Image.Image) -> Image.Image:
        """Apply photometric augmentations to simulate real-world conditions."""
        if random.random() > self.photometric_prob:
            return image
        
        if random.random() < 0.6:
            brightness = random.uniform(*self.brightness_factor)
            contrast = random.uniform(*self.contrast_factor)
            saturation = random.uniform(*self.saturation_factor)
            hue = random.uniform(*self.hue_factor)
            
            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)
            image = F.adjust_saturation(image, saturation)
            image = F.adjust_hue(image, hue)
        
        if random.random() < 0.2:
            blur_radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        if random.random() < 0.3:
            image = self._add_gaussian_noise(image)
        
        if random.random() < 0.4:
            gamma = random.uniform(*self.gamma_range)
            image = self._adjust_gamma(image, gamma)
        
        if random.random() < 0.3:
            image = self._simulate_lighting_conditions(image)
        
        return image
    
    def _rotate_boxes(self, boxes: torch.Tensor, angle: float, image_size: Tuple[int, int]) -> torch.Tensor:
        """Rotate bounding boxes with the image."""
        cx, cy = image_size[0] / 2, image_size[1] / 2
        angle_rad = np.radians(-angle)
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        rotated_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            
            corners = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ])
            
            rotated_corners = []
            for x, y in corners:
                x_new = cx + (x - cx) * cos_a - (y - cy) * sin_a
                y_new = cy + (x - cx) * sin_a + (y - cy) * cos_a
                rotated_corners.append([x_new, y_new])
            
            rotated_corners = np.array(rotated_corners)
            
            x_min, y_min = rotated_corners.min(axis=0)
            x_max, y_max = rotated_corners.max(axis=0)
            
            rotated_boxes.append([x_min, y_min, x_max, y_max])
        
        return torch.tensor(rotated_boxes, dtype=boxes.dtype)
    
    def _affine_boxes(self, boxes: torch.Tensor, degrees: float, translate: list, 
                     scale: float, shear: float, image_size: Tuple[int, int]) -> torch.Tensor:
        """Apply affine transformation to bounding boxes."""
        w, h = image_size
        boxes = boxes * scale
        
        boxes[:, [0, 2]] += translate[0]
        boxes[:, [1, 3]] += translate[1]
        
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, h)
        
        return boxes
    
    def _add_gaussian_noise(self, image: Image.Image, noise_factor: float = 0.1) -> Image.Image:
        """Add Gaussian noise to simulate sensor noise."""
        img_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, noise_factor * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def _adjust_gamma(self, image: Image.Image, gamma: float) -> Image.Image:
        """Apply gamma correction."""
        img_array = np.array(image, dtype=np.float32) / 255.0
        corrected = np.power(img_array, gamma)
        corrected = (corrected * 255).astype(np.uint8)
        return Image.fromarray(corrected)
    
    def _simulate_lighting_conditions(self, image: Image.Image) -> Image.Image:
        """Simulate different lighting conditions."""
        img_array = np.array(image, dtype=np.float32)
        h, w = img_array.shape[:2]
        
        x_gradient = np.linspace(0, 1, w)
        y_gradient = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x_gradient, y_gradient)
        
        lighting_factor = 0.8 + 0.4 * (0.5 * xx + 0.3 * yy + 0.2 * np.random.random())
        lighting_factor = np.expand_dims(lighting_factor, axis=2)
        
        modified_img = img_array * lighting_factor
        modified_img = np.clip(modified_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(modified_img)

class MixUp:
    """MixUp augmentation for detection tasks."""
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.3):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch_images: list, batch_targets: list):
        """Apply MixUp to a batch of images and targets."""
        if random.random() > self.prob or len(batch_images) < 2:
            return batch_images, batch_targets
        
        mixed_images = []
        mixed_targets = []
        
        for i in range(len(batch_images)):
            if random.random() < 0.5:
                mix_idx = random.randint(0, len(batch_images) - 1)
                if mix_idx == i:
                    mix_idx = (i + 1) % len(batch_images)
                
                lam = np.random.beta(self.alpha, self.alpha)
                
                mixed_img = lam * batch_images[i] + (1 - lam) * batch_images[mix_idx]
                
                mixed_target = {}
                target1, target2 = batch_targets[i], batch_targets[mix_idx]
                
                mixed_target["boxes"] = torch.cat([target1["boxes"], target2["boxes"]], dim=0)
                mixed_target["labels"] = torch.cat([target1["labels"], target2["labels"]], dim=0)
                mixed_target["image_id"] = target1["image_id"]
                mixed_target["area"] = torch.cat([target1["area"], target2["area"]], dim=0)
                mixed_target["iscrowd"] = torch.cat([target1["iscrowd"], target2["iscrowd"]], dim=0)
                
                mixed_images.append(mixed_img)
                mixed_targets.append(mixed_target)
            else:
                mixed_images.append(batch_images[i])
                mixed_targets.append(batch_targets[i])
        
        return mixed_images, mixed_targets

def get_training_transforms(image_size: int = 640) -> QRDetectionAugmentation:
    """Get training augmentation pipeline."""
    return QRDetectionAugmentation(
        image_size=image_size,
        geometric_prob=0.6,
        photometric_prob=0.8,
        training=True
    )

def get_validation_transforms(image_size: int = 640) -> QRDetectionAugmentation:
    """Get validation augmentation pipeline (minimal augmentation)."""
    return QRDetectionAugmentation(
        image_size=image_size,
        geometric_prob=0.0,
        photometric_prob=0.0,
        training=False
    )

def get_test_time_augmentation_transforms(image_size: int = 640) -> list:
    """Get Test Time Augmentation (TTA) transforms for inference."""
    return [
        QRDetectionAugmentation(image_size, 0.0, 0.0, False),
        lambda img, target: (F.hflip(F.resize(img, (image_size, image_size))), target),
        lambda img, target: (F.rotate(F.resize(img, (image_size, image_size)), 5), target),
        lambda img, target: (F.rotate(F.resize(img, (image_size, image_size)), -5), target),
        lambda img, target: (F.adjust_brightness(F.resize(img, (image_size, image_size)), 1.1), target),
        lambda img, target: (F.adjust_brightness(F.resize(img, (image_size, image_size)), 0.9), target),
    ]