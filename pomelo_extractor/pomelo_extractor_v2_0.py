import os
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import SamPredictor, sam_model_registry
import argparse
from pathlib import Path

class SAMImageProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="cropper/sam_vit_h_4b8939.pth"):
        """
        Initialize the Segment Anything Model predictor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
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
            
            # Set image in predictor
            self.predictor.set_image(image_rgb)
            
            # Get the mask for the entire image (subject segmentation)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                multimask_output=True,
            )
            
            # Choose the mask with the highest score
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            
            # Find the largest connected component (biggest individual)
            mask_uint8 = (mask * 255).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, 8, cv2.CV_32S)
            
            if num_labels > 1:  # If there are multiple components
                # Find the largest component (skip background at index 0)
                largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask = (labels == largest_component_idx)
            
            # Create transparent image
            rgba_image = self._create_transparent_image(image_rgb, mask)
            
            # Trim transparent areas
            trimmed_image = self._trim_transparent(rgba_image)
            
            # Save the result
            trimmed_image.save(output_path, 'PNG')
            print(f"Processed and saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False
    
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

def process_images(input_folder, output_folder, max_images=None):
    """
    Process all images in the input folder and save results to output folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize SAM processor
    processor = SAMImageProcessor()
    
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
    
    args = parser.parse_args()
    
    # Process images
    process_images(args.input, args.output, args.max)

if __name__ == "__main__":
    # For testing without command line arguments, you can set these variables:
    INPUT_FOLDER = "test_images"  # Change this to your input folder path
    OUTPUT_FOLDER = "test_output"  # Change this to your output folder path
    MAX_IMAGES = None  # Set to None to process all images, or a number to limit
    
    # Uncomment the next line to use the hardcoded paths for testing
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, MAX_IMAGES)
    
    # Or use command line arguments (comment out the line above)
    # main()