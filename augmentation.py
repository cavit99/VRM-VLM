import cv2
import numpy as np
import glob
import os
import random
from typing import Tuple

# Define a minimum plate width in pixels (adjust as needed)
MIN_PLATE_WIDTH = 300

class ImageAugmenter:
    def __init__(self, canvas_size: int = 448, min_plate_width: int = 300):
        self.canvas_size = canvas_size
        self.MIN_PLATE_WIDTH = min_plate_width

    class PhotometricAugmenter:
        @staticmethod
        def adjust_brightness_contrast(img: np.ndarray) -> np.ndarray:
            """Random brightness and contrast adjustment."""
            alpha = random.uniform(0.5, 1.5)  # Contrast multiplier
            # Convert percentage (-50% to +50%) to pixel value adjustment
            brightness_percent = random.uniform(-0.5, 0.5)  # -50% to +50%
            beta = int(brightness_percent * 255)  # Convert to pixel value adjustment
            return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        @staticmethod
        def apply_blur(img: np.ndarray) -> np.ndarray:
            """Randomly apply either a Gaussian or a motion blur."""
            if random.random() < 0.3:  # Reduced probability for Gaussian blur (was 0.5)
                # Gaussian blur
                ksize = random.choice([3, 5, 7])
                return cv2.GaussianBlur(img, (ksize, ksize), 0)
            
            # Simulated motion blur with larger kernel size and stronger effect
            size = random.choice([13, 15, 17])  # Increased kernel sizes (was [9, 11, 13])
            kernel = np.zeros((size, size))
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])
            
            if direction == 'horizontal':
                kernel[size // 2, :] = np.ones(size)
            elif direction == 'vertical':
                kernel[:, size // 2] = np.ones(size)
            else:
                np.fill_diagonal(kernel, 1)
            
            kernel = kernel / size  
            return cv2.filter2D(img, -1, kernel)

        @staticmethod
        def add_gaussian_noise(img: np.ndarray) -> np.ndarray:
            """Add random Gaussian noise."""
            mean = 0
            std = random.uniform(10, 30)
            noise = np.random.normal(mean, std, img.shape).astype(np.int16)
            return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        @staticmethod
        def add_salt_and_pepper(img: np.ndarray) -> np.ndarray:
            """Add salt and pepper noise."""
            prob = random.uniform(0.01, 0.05)  # Random probability between 1-5%
            noisy = np.copy(img)
            # Salt
            salt = np.random.random(img.shape) < prob/2
            noisy[salt] = 255
            # Pepper
            pepper = np.random.random(img.shape) < prob/2
            noisy[pepper] = 0
            return noisy

        @staticmethod
        def jpeg_compression(img: np.ndarray) -> np.ndarray:
            """Apply simulated JPEG compression artifacts."""
            quality = random.randint(5, 20)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            return cv2.imdecode(encimg, 1) if result else img

    class GeometricAugmenter:
        def __init__(self, canvas_size: int, min_plate_width: int):
            self.canvas_size = canvas_size
            self.MIN_PLATE_WIDTH = min_plate_width

        def apply_pure_photometric(self, plate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Apply minimal geometric transformation."""
            height_original, width_original = plate.shape[:2]
            
            new_width = random.uniform(
                max(self.canvas_size * 0.5, self.MIN_PLATE_WIDTH), 
                self.canvas_size
            )
            scale_factor = new_width / width_original
            new_height = int(height_original * scale_factor)
            
            plate_resized = cv2.resize(plate, (int(new_width), new_height))
            mask = np.full((plate_resized.shape[0], plate_resized.shape[1]), 255, dtype=np.uint8)
            
            return plate_resized, mask

        def apply_geometric(self, plate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Apply mild geometric transformation with perspective."""
            height_original, width_original = plate.shape[:2]
            
            target_width = random.uniform(max(150, self.MIN_PLATE_WIDTH), self.canvas_size)
            target_height = target_width * (height_original / width_original)
            
            # Setup source and destination points
            src_points = np.array([
                [0, 0],
                [width_original, 0],
                [0, height_original],
                [width_original, height_original]
            ], dtype=np.float32)

            # Calculate destination points with random placement
            w, h = target_width, target_height
            cx = random.uniform(w/2, self.canvas_size - w/2)
            cy = random.uniform(h/2, self.canvas_size - h/2)
            
            corners = np.array([
                [cx - w/2, cy - h/2],
                [cx + w/2, cy - h/2],
                [cx - w/2, cy + h/2],
                [cx + w/2, cy + h/2]
            ], dtype=np.float32)

            # Apply rotation
            theta = random.uniform(-15, 15)
            rad = np.deg2rad(theta)
            rot_matrix = np.array([
                [np.cos(rad), -np.sin(rad)],
                [np.sin(rad),  np.cos(rad)]
            ])
            corners_rot = np.dot(corners - [cx, cy], rot_matrix) + [cx, cy]

            # Apply perspective perturbation
            perturbation = 0.05 * w
            perturbations = np.random.uniform(-perturbation, perturbation, (4, 2))
            dst_points = np.clip(corners_rot + perturbations, 0, self.canvas_size)

            # Compute and apply homography
            H, _ = cv2.findHomography(src_points, dst_points)
            mask = np.full((height_original, width_original), 255, dtype=np.uint8)
            
            warped_plate = cv2.warpPerspective(
                plate, H, (self.canvas_size, self.canvas_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            warped_mask = cv2.warpPerspective(
                mask, H, (self.canvas_size, self.canvas_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            warped_mask = (warped_mask > 128).astype(np.uint8) * 255
            return warped_plate, warped_mask

    def augment_plate_image(self, image_path: str, output_path: str) -> None:
        """Augment a license plate image with geometric and photometric transforms."""
        plate = cv2.imread(image_path)
        if plate is None:
            print(f"Error reading {image_path}")
            return

        # Initialize augmenters
        geometric = self.GeometricAugmenter(self.canvas_size, self.MIN_PLATE_WIDTH)
        photometric = self.PhotometricAugmenter()

        # Decide transformation type and apply geometric transform
        pure_photometric = random.random() < 0.6
        background_color = np.random.randint(0, 256, 3).tolist()
        background = np.full((self.canvas_size, self.canvas_size, 3), background_color, dtype=np.uint8)
        
        if pure_photometric:
            plate_transformed, mask = geometric.apply_pure_photometric(plate)
            max_x = self.canvas_size - plate_transformed.shape[1]
            max_y = self.canvas_size - plate_transformed.shape[0]
            tx, ty = int(random.uniform(0, max_x)), int(random.uniform(0, max_y))
            
            composite = background.copy()
            # Convert mask to float32 and normalize to 0-1 range
            mask_float = mask.astype(np.float32) / 255.0
            mask_3ch = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2BGR)
            
            roi = composite[ty:ty+plate_transformed.shape[0], tx:tx+plate_transformed.shape[1]]
            # Proper alpha blending
            roi[:] = (plate_transformed * mask_3ch + background[ty:ty+plate_transformed.shape[0], 
                     tx:tx+plate_transformed.shape[1]] * (1 - mask_3ch))
        else:
            warped_plate, warped_mask = geometric.apply_geometric(plate)
            composite = background.copy()
            # Convert mask to float32 and normalize to 0-1 range
            mask_float = warped_mask.astype(np.float32) / 255.0
            mask_3ch = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2BGR)
            # Proper alpha blending
            composite = warped_plate * mask_3ch + background * (1 - mask_3ch)

        # Apply photometric augmentations
        augmented = composite
        augmented = photometric.adjust_brightness_contrast(augmented)
        if random.random() < 0.55:
            augmented = photometric.apply_blur(augmented)
        if random.random() < 0.5:
            augmented = photometric.add_gaussian_noise(augmented)
        if random.random() < 0.33: 
            augmented = photometric.add_salt_and_pepper(augmented)
        if random.random() < 0.5:
            augmented = photometric.jpeg_compression(augmented)

        # Save the augmented image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, augmented)
        print(f"Saved augmented image to {output_path}")

def main():
    input_folder = "generated"
    output_folder = "augmented"
    num_augmentations = 1
    
    image_files = glob.glob(os.path.join(input_folder, "*.jpg"))
    if not image_files:
        print("No images found in the 'generated' folder.")
        return

    augmenter = ImageAugmenter()
    for image_path in image_files:
        for i in range(num_augmentations):
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            new_path = os.path.join(output_folder, f"{name}_aug{i}{ext}")
            augmenter.augment_plate_image(image_path, new_path)

if __name__ == "__main__":
    main()
    print("Augmentation complete!")