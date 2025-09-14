import os
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import SamPredictor, sam_model_registry
import argparse
from pathlib import Path

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
        print(f"Using device: {self.device}")
        print(f"Center weight: {center_weight}, Area weight: {area_weight}, Enhance green: {enhance_green}")
        
        # Load SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
    
    def process_image(self, image_path, output_path):
        """
        Process a single image: segment subject, create transparent background, and save
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return False
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Enhance green contrast for pomelo images
            if self.enhance_green:
                image_rgb = self._enhance_green_contrast(image_rgb)
            
            # Set image in predictor
            self.predictor.set_image(image_rgb)
            
            # Get the mask for the entire image (subject segmentation)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                multimask_output=True,
            )
            
            # Use scoring system to choose the best mask instead of just highest score
            best_mask = self._select_best_mask_with_scoring(masks, scores, image_rgb.shape)
            
            if best_mask is None:
                print(f"No suitable mask found for {image_path}")
                return False
            
            # Find the largest connected component (biggest individual)
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, 8, cv2.CV_32S)
            
            if num_labels > 1:  # If there are multiple components
                # Find the largest component (skip background at index 0)
                largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                best_mask = (labels == largest_component_idx)
            
            # Create transparent image
            rgba_image = self._create_transparent_image(image_rgb, best_mask)
            
            # Trim transparent areas
            trimmed_image = self._trim_transparent(rgba_image)
            
            # Save the result
            trimmed_image.save(output_path, 'PNG')
            print(f"Processed and saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
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
    
    def _select_best_mask_with_scoring(self, masks, scores, image_shape):
        """Select the best mask using the scoring system based on center position and area"""
        if len(masks) == 0:
            return None
        
        height, width = image_shape[:2]
        total_area = height * width
        scored_masks = []
        
        for i, mask in enumerate(masks):
            area = np.sum(mask)
            
            # Get bounding box coordinates
            coords = np.column_stack(np.where(mask))
            if len(coords) == 0:
                continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Calculate center scores using your specified equations
            x_center_score = 1 - abs((x_min + x_max) - width) / width
            y_center_score = 1 - abs((y_min + y_max) - height) / height
            
            # Calculate area score
            area_score = 1 - 2 * abs(area - total_area/2) / total_area
            
            # Combine scores with weights
            total_score = (self.center_weight * (x_center_score + y_center_score) + 
                          self.area_weight * area_score)
            
            # Incorporate the original SAM confidence score
            confidence = scores[i]
            total_score *= confidence
            
            scored_masks.append((mask, total_score, area, (x_min, y_min, x_max, y_max)))
        
        if not scored_masks:
            return None
        
        # Sort by total score (highest first)
        scored_masks.sort(key=lambda x: x[1], reverse=True)
        
        # Log top mask score for debugging
        print(f"Best mask score: {scored_masks[0][1]:.3f}")
        
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
        coords = np.column_stack(np.where(alpha > 0))
        if len(coords) == 0:
            return image
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Crop the image
        cropped = image_np[y_min:y_max+1, x_min:x_max+1]
        return Image.fromarray(cropped)

def process_images(input_folder, output_folder, max_images=None, center_weight=1.0, area_weight=1.0, enhance_green=True):
    """
    Process all images in the input folder and save results to output folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize SAM processor with hyperparameters
    processor = SAMImageProcessor(
        center_weight=center_weight,
        area_weight=area_weight,
        enhance_green=enhance_green
    )
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file in os.listdir(input_folder):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(os.path.join(input_folder, file))
    
    if not image_files:
        print("No images found in the input folder.")
        return
    
    # Limit number of images if specified
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    successful = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        output_filename = f"segmented_{Path(image_path).stem}.png"
        output_path = os.path.join(output_folder, output_filename)
        
        if processor.process_image(image_path, output_path):
            successful += 1
    
    print(f"Processing complete! {successful}/{len(image_files)} images processed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Segment images using SAM and remove background')
    parser.add_argument('--input', '-i', required=True, help='Path to input folder with images')
    parser.add_argument('--output', '-o', required=True, help='Path to output folder for processed images')
    parser.add_argument('--max', '-m', type=int, default=None, help='Maximum number of images to process (optional)')
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
    # For testing without command line arguments, you can set these variables:
    INPUT_FOLDER = "test_images"  # Change this to your input folder path
    OUTPUT_FOLDER = "test_output"  # Change this to your output folder path
    MAX_IMAGES = None  # Set to None to process all images, or a number to limit
    CENTER_WEIGHT = 0.5  # Hyperparameter for center scoring
    AREA_WEIGHT = 2.0    # Hyperparameter for area scoring
    ENHANCE_GREEN = False # Enable green contrast enhancement
    
    # Uncomment the next line to use the hardcoded paths for testing
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, MAX_IMAGES, CENTER_WEIGHT, AREA_WEIGHT, ENHANCE_GREEN)
    
    # Or use command line arguments (comment out the line above)
    # main()