import os
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAMImageProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="preprocessing/sam_vit_h_4b8939.pth", 
                 center_weight=1.0, area_weight=1.0, enhance_green=True):
        """
        Initialize the Segment Anything Model predictor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.center_weight = center_weight
        self.area_weight = area_weight
        self.enhance_green = enhance_green
        logger.info(f"Using device: {self.device}")
        logger.info(f"Center weight: {center_weight}, Area weight: {area_weight}, Enhance green: {enhance_green}")
        
        try:
            # Load SAM model
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            # Also initialize automatic mask generator for better object detection
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            logger.info("SAM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {str(e)}")
            raise
    
    def process_image(self, image_path, output_path):
        """
        Process a single image: segment subject, create transparent background, and save
        """
        try:
            # Verify it's actually an image file
            if not self._is_valid_image(image_path):
                logger.warning(f"Skipping non-image file: {image_path}")
                return False
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return False
            
            logger.info(f"Processing image: {os.path.basename(image_path)}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Enhance green contrast for pomelo images
            if self.enhance_green:
                image_rgb = self._enhance_green_contrast(image_rgb)
            
            # Method 1: Try automatic mask generation first (better for object detection)
            masks = self._generate_masks_automatically(image_rgb)
            
            # Method 2: If automatic fails, try the original point-based approach
            if masks is None or len(masks) == 0:
                logger.info("Automatic mask generation failed, trying point-based approach")
                masks = self._generate_masks_with_points(image_rgb)
            
            if masks is None or len(masks) == 0:
                logger.warning(f"No valid masks found for {image_path}")
                return False
            
            # Choose the best mask using the scoring system
            best_mask = self._select_best_mask_with_scoring(masks, image_rgb.shape)
            
            if best_mask is None:
                logger.warning(f"No suitable mask found for {image_path}")
                return False
            
            # Create transparent image
            rgba_image = self._create_transparent_image(image_rgb, best_mask)
            
            # Trim transparent areas
            trimmed_image = self._trim_transparent(rgba_image)
            
            # Verify the output is valid (not just a tiny speck)
            if self._is_valid_output(trimmed_image):
                trimmed_image.save(output_path, 'PNG')
                logger.info(f"Successfully processed and saved: {output_path}")
                return True
            else:
                logger.warning(f"Output image too small or invalid for {image_path}")
                return False
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return False
    
    def _enhance_green_contrast(self, image_rgb):
        """
        Enhance contrast specifically for green pomelos on green foliage
        """
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Enhance the a channel (green-red axis) to separate pomelos from foliage
        # Pomelos tend to be more yellow-green, foliage more pure green
        a_enhanced = a.astype(np.float32)
        a_enhanced = np.clip(a_enhanced * 1.2, 0, 255).astype(np.uint8)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b])
        enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb
    
    def _is_valid_image(self, file_path):
        """Check if file is a valid image"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        return Path(file_path).suffix.lower() in valid_extensions
    
    def _generate_masks_automatically(self, image_rgb):
        """Generate masks using automatic mask generation"""
        try:
            masks = self.mask_generator.generate(image_rgb)
            return masks
        except Exception as e:
            logger.warning(f"Automatic mask generation failed: {str(e)}")
            return None
    
    def _generate_masks_with_points(self, image_rgb):
        """Generate masks using point-based prediction"""
        try:
            self.predictor.set_image(image_rgb)
            
            # Try multiple approaches for point selection
            height, width = image_rgb.shape[:2]
            
            # Approach 1: Center point
            center_point = np.array([[width // 2, height // 2]])
            masks, scores, _ = self.predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),
                multimask_output=True,
            )
            
            # Approach 2: Multiple points if center approach fails
            if np.max(scores) < 0.5:
                points = np.array([
                    [width // 2, height // 2],          # center
                    [width // 4, height // 2],          # left center
                    [3 * width // 4, height // 2],      # right center
                    [width // 2, height // 4],          # top center
                    [width // 2, 3 * height // 4],      # bottom center
                ])
                masks, scores, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=np.array([1, 1, 1, 1, 1]),
                    multimask_output=True,
                )
            
            # Convert to same format as automatic masks
            formatted_masks = []
            for i, mask in enumerate(masks):
                formatted_masks.append({
                    'segmentation': mask,
                    'area': np.sum(mask),
                    'predicted_iou': scores[i],
                    'stability_score': scores[i]
                })
            
            return formatted_masks
            
        except Exception as e:
            logger.warning(f"Point-based mask generation failed: {str(e)}")
            return None
    
    def _select_best_mask_with_scoring(self, masks, image_shape):
        """Select the best mask using the scoring system based on center position and area"""
        if not masks:
            return None
        
        height, width = image_shape[:2]
        total_area = height * width
        scored_masks = []
        
        for mask_data in masks:
            if 'segmentation' in mask_data:
                mask = mask_data['segmentation']
                area = mask_data.get('area', np.sum(mask))
                
                # Get bounding box coordinates
                coords = np.column_stack(np.where(mask))
                if len(coords) == 0:
                    continue
                
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Calculate center scores
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                x_center_score = 1 - abs(center_x - width/2) / (width/2)
                y_center_score = 1 - abs(center_y - height/2) / (height/2)
                
                # Calculate area score
                area_score = 1 - 2 * abs(area - total_area/3) / total_area  # Target area is 1/3 of image
                
                # Combine scores with weights
                total_score = (self.center_weight * (x_center_score + y_center_score) + 
                              self.area_weight * area_score)
                
                # Add confidence score if available
                confidence = mask_data.get('predicted_iou', 0.5) * mask_data.get('stability_score', 0.5)
                total_score *= confidence
                
                scored_masks.append((mask, total_score, area, (x_min, y_min, x_max, y_max)))
        
        if not scored_masks:
            return None
        
        # Sort by total score (highest first)
        scored_masks.sort(key=lambda x: x[1], reverse=True)
        
        # Log top 3 masks for debugging
        logger.info(f"Top mask scores: {[f'{score:.3f}' for _, score, _, _ in scored_masks[:3]]}")
        
        return scored_masks[0][0]  # Return the mask with highest score
    
    def _create_transparent_image(self, image_rgb, mask):
        """
        Create an image with transparent background
        """
        # Convert to RGBA
        rgba = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = image_rgb
        
        # Set alpha channel based on mask
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        
        return Image.fromarray(rgba)
    
    def _trim_transparent(self, image):
        """
        Trim transparent areas from the image
        """
        image_np = np.array(image)
        alpha = image_np[:, :, 3]
        
        # Find bounding box of non-transparent pixels
        coords = np.column_stack(np.where(alpha > 10))  # Small threshold for near-transparent pixels
        if len(coords) == 0:
            return image
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add small padding
        padding = 5
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(image_np.shape[0] - 1, y_max + padding)
        x_max = min(image_np.shape[1] - 1, x_max + padding)
        
        # Crop the image
        cropped = image_np[y_min:y_max+1, x_min:x_max+1]
        return Image.fromarray(cropped)
    
    def _is_valid_output(self, image):
        """Check if the output image is valid (not just a tiny speck)"""
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            alpha = image_np[:, :, 3]
            non_transparent_pixels = np.sum(alpha > 10)  # Count non-transparent pixels
            return non_transparent_pixels > 1000  # At least 1000 pixels should be non-transparent
        return False

def process_images(input_folder, output_folder, max_images=None, center_weight=1.0, area_weight=1.0, enhance_green=True):
    """
    Process all images in the input folder and save results to output folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Initialize SAM processor with hyperparameters
        processor = SAMImageProcessor(
            center_weight=center_weight,
            area_weight=area_weight,
            enhance_green=enhance_green
        )
    except Exception as e:
        logger.error(f"Failed to initialize SAM processor: {str(e)}")
        return
    
    # Get all image files
    image_files = []
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if processor._is_valid_image(file_path):
            image_files.append(file_path)
    
    if not image_files:
        logger.warning("No valid images found in the input folder.")
        return
    
    # Sort files for consistent processing order
    image_files.sort()
    
    # Limit number of images if specified
    if max_images is not None:
        image_files = image_files[:max_images]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        output_filename = f"segmented_{Path(image_path).stem}.png"
        output_path = os.path.join(output_folder, output_filename)
        
        if processor.process_image(image_path, output_path):
            successful += 1
    
    logger.info(f"Processing complete! {successful}/{len(image_files)} images processed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Segment images using SAM and remove background')
    parser.add_argument('--input', '-i', required=True, help='Path to input folder with images')
    parser.add_argument('--output', '-o', required=True, help='Path to output folder for processed images')
    parser.add_argument('--max', '-m', type=int, default=None, help='Maximum number of images to process (optional)')
    parser.add_argument('--model', default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    parser.add_argument('--center-weight', type=float, default=1.0, help='Weight for center position scoring')
    parser.add_argument('--area-weight', type=float, default=1.0, help='Weight for area scoring')
    parser.add_argument('--no-enhance-green', action='store_false', dest='enhance_green', 
                       help='Disable green contrast enhancement')
    
    args = parser.parse_args()
    
    # Process images with hyperparameters
    process_images(
        args.input, 
        args.output, 
        args.max, 
        args.center_weight, 
        args.area_weight, 
        args.enhance_green
    )

if __name__ == "__main__":
    # For testing without command line arguments
    INPUT_FOLDER = "test_images"
    OUTPUT_FOLDER = "test_output"
    MAX_IMAGES = None  # Start with a small number for testing
    CENTER_WEIGHT = 1.0  # Hyperparameter for center scoring
    AREA_WEIGHT = 1.0    # Hyperparameter for area scoring
    ENHANCE_GREEN = False # Enable green contrast enhancement
    
    # Uncomment for testing with hardcoded paths
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, MAX_IMAGES, CENTER_WEIGHT, AREA_WEIGHT, ENHANCE_GREEN)
    
    # Or use command line arguments
    # main()