import os
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import SamPredictor, sam_model_registry
import argparse
from pathlib import Path
import time

class SAMImageProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="preprocessing/sam_vit_h_4b8939.pth"):
        """
        Initialize the Segment Anything Model predictor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
    
    def process_image(self, image_path, output_path, monitoring_path=None, mask_override_dict=None):
        """
        Process a single image: segment subject, create transparent background, and save
        """
        start_time = time.time()
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return False, 0, None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set image in predictor
            self.predictor.set_image(image_rgb)
            
            # Get image dimensions for center point
            height, width = image_rgb.shape[:2]
            
            # Use center point prompt to guide SAM to the main subject
            center_point = np.array([[width // 2, height // 2]])
            point_labels = np.array([1])  # 1 = foreground point
            
            # Get the mask with center point guidance
            masks, scores, _ = self.predictor.predict(
                point_coords=center_point,
                point_labels=point_labels,
                multimask_output=True,
            )
            
            # Create monitoring visualization if monitoring path is provided
            if monitoring_path:
                self._create_monitoring_visualization(image_rgb, masks, scores, monitoring_path, center_point)
            
            # Check for mask override (convert 1-based to 0-based)
            image_name = Path(image_path).stem
            selected_mask_index = None
            
            if mask_override_dict and image_name in mask_override_dict:
                override_index = mask_override_dict[image_name] - 1  # Convert 1-based to 0-based
                if 0 <= override_index < len(masks):
                    print(f"Using override mask index {override_index + 1} for {image_name}")
                    best_mask = masks[override_index]
                    selected_mask_index = override_index + 1  # Store as 1-based for logging
                else:
                    print(f"Invalid override index {override_index + 1} for {image_name}, using circularity selection")
                    best_mask, selected_mask_index = self._select_most_circular_mask(masks, scores, image_rgb.shape)
            else:
                # Use circularity-based selection
                best_mask, selected_mask_index = self._select_most_circular_mask(masks, scores, image_rgb.shape)
            
            if best_mask is None:
                print(f"No suitable mask found for {image_path}")
                return False, 0, None
            
            # Apply morphological reconstruction to fix fragmented segments (e.g., anthracnose)
            reconstructed_mask = self._reconstruct_complete_fruit(best_mask, image_rgb.shape)
            
            # Find the largest connected component (biggest individual)
            mask_uint8 = (reconstructed_mask * 255).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, 8, cv2.CV_32S)
            
            if num_labels > 1:  # If there are multiple components
                # Find the largest component (skip background at index 0)
                largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                reconstructed_mask = (labels == largest_component_idx)
            
            # Create transparent image
            rgba_image = self._create_transparent_image(image_rgb, reconstructed_mask)
            
            # Trim transparent areas
            trimmed_image = self._trim_transparent(rgba_image)
            
            # Save the result
            trimmed_image.save(output_path, 'PNG')
            
            processing_time = time.time() - start_time
            print(f"Processed and saved: {output_path} (Mask {selected_mask_index}, Time: {processing_time:.2f}s)")
            
            return True, processing_time, selected_mask_index
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False, 0, None
    
    def _reconstruct_complete_fruit(self, mask, image_shape):
        """
        Reconstruct the complete fruit mask from potentially fragmented segments (e.g., anthracnose)
        """
        # Convert mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 1. Morphological closing to connect nearby segments
        kernel = np.ones((15, 15), np.uint8)  # Adjust kernel size based on image resolution
        closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # 2. Find the largest contour
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 3. Create a convex hull to get the full fruit shape
        hull = cv2.convexHull(largest_contour)
        
        # 4. Create a new mask from the convex hull
        complete_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(complete_mask, [hull], 255)
        
        # 5. Optional: Apply mild dilation to ensure complete coverage
        kernel_small = np.ones((5, 5), np.uint8)
        complete_mask = cv2.dilate(complete_mask, kernel_small, iterations=1)
        
        return complete_mask.astype(bool)
    
    def _create_monitoring_visualization(self, image_rgb, masks, scores, monitoring_path, center_point):
        """
        Create a visualization of all segments found in the image for monitoring
        """
        # Create a grid of visualizations
        num_masks = len(masks)
        if num_masks == 0:
            return
        
        # Calculate grid dimensions
        cols = min(4, num_masks)  # Maximum 4 columns
        rows = (num_masks + cols - 1) // cols
        
        # Create a large canvas for the grid
        grid_height = rows * image_rgb.shape[0]
        grid_width = cols * image_rgb.shape[1]
        grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Create individual visualizations for each mask
        for i, (mask, score) in enumerate(zip(masks, scores)):
            row = i // cols
            col = i % cols
            
            # Create visualization for this mask
            mask_visualization = self._visualize_mask(image_rgb, mask, score, i, center_point)
            
            # Place in grid
            y_start = row * image_rgb.shape[0]
            y_end = y_start + image_rgb.shape[0]
            x_start = col * image_rgb.shape[1]
            x_end = x_start + image_rgb.shape[1]
            
            grid_canvas[y_start:y_end, x_start:x_end] = mask_visualization
        
        # Save the monitoring visualization
        monitoring_image = Image.fromarray(grid_canvas)
        monitoring_image.save(monitoring_path)
        print(f"Monitoring visualization saved: {monitoring_path}")
    
    def _visualize_mask(self, image_rgb, mask, score, mask_index, center_point):
        """
        Create a visualization of a single mask with bounding box and score
        """
        # Create a copy of the image
        visualization = image_rgb.copy()
        
        # Apply mask as semi-transparent overlay
        mask_rgb = np.zeros_like(visualization)
        mask_rgb[mask] = [255, 0, 0]  # Red color for mask
        
        # Blend mask with original image
        alpha = 0.3
        visualization = cv2.addWeighted(visualization, 1.0, mask_rgb, alpha, 0)
        
        # Draw center point (prompt point)
        center_x, center_y = center_point[0]
        cv2.circle(visualization, (center_x, center_y), 10, (0, 255, 255), -1)  # Yellow center point
        cv2.circle(visualization, (center_x, center_y), 12, (0, 0, 0), 2)  # Black outline
        
        # Draw bounding box
        coords = np.column_stack(np.where(mask))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Draw rectangle
            cv2.rectangle(visualization, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw center point of mask
            mask_center_x = (x_min + x_max) // 2
            mask_center_y = (y_min + y_max) // 2
            cv2.circle(visualization, (mask_center_x, mask_center_y), 5, (255, 0, 255), -1)  # Purple mask center
        
        # Add score text
        score_text = f"Score: {score:.3f}"
        cv2.putText(visualization, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add mask number (1-based for user readability)
        mask_text = f"Mask {mask_index + 1}"
        cv2.putText(visualization, mask_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add circularity score
        circularity = self._calculate_circularity(mask)
        circularity_text = f"Circ: {circularity:.3f}"
        cv2.putText(visualization, circularity_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return visualization
    
    def _calculate_circularity(self, mask):
        """
        Calculate circularity of a mask (1 = perfect circle, 0 = not circular)
        """
        # Convert mask to uint8 for contour detection
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity formula: 4 * π * area / perimeter²
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        return circularity
    
    def _select_most_circular_mask(self, masks, scores, image_shape):
        """
        Select the mask that is most circular/elliptical
        """
        if len(masks) == 0:
            return None, None
        
        circularity_scores = []
        
        for i, mask in enumerate(masks):
            # Calculate circularity
            circularity = self._calculate_circularity(mask)
            
            # Combine with SAM confidence score
            combined_score = circularity * scores[i]
            
            circularity_scores.append((mask, combined_score, circularity, scores[i], i))
        
        # Sort by circularity score (highest first)
        circularity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Log the best circularity score for debugging
        best_circularity = circularity_scores[0][2]
        best_confidence = circularity_scores[0][3]
        best_index = circularity_scores[0][4]
        print(f"Best circularity: {best_circularity:.3f}, Confidence: {best_confidence:.3f}")
        
        return circularity_scores[0][0], best_index + 1  # Return 1-based index
    
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

def process_images(input_folder, output_folder, max_images=None, mask_override_dict=None):
    """
    Process all images in the input folder and save results to output folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create monitoring folder inside output folder
    monitoring_folder = os.path.join(output_folder, "monitoring")
    os.makedirs(monitoring_folder, exist_ok=True)
    
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
    total_processing_time = 0
    processing_times = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        output_filename = f"{Path(image_path).stem}.png"
        output_path = os.path.join(output_folder, output_filename)
        
        # Create monitoring path
        monitoring_filename = f"monitoring_{Path(image_path).stem}.jpg"
        monitoring_path = os.path.join(monitoring_folder, monitoring_filename)
        
        success, processing_time, mask_index = processor.process_image(
            image_path, output_path, monitoring_path, mask_override_dict
        )
        
        if success:
            successful += 1
            total_processing_time += processing_time
            processing_times.append(processing_time)
    
    # Calculate and log average processing time
    if processing_times:
        avg_time = total_processing_time / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        print(f"\nProcessing complete! {successful}/{len(image_files)} images processed successfully.")
        print(f"Average processing time: {avg_time:.2f}s per image")
        print(f"Minimum time: {min_time:.2f}s, Maximum time: {max_time:.2f}s")
    else:
        print(f"\nProcessing complete! {successful}/{len(image_files)} images processed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Segment images using SAM and remove background')
    parser.add_argument('--input', '-i', required=True, help='Path to input folder with images')
    parser.add_argument('--output', '-o', required=True, help='Path to output folder for processed images')
    parser.add_argument('--max', '-m', type=int, default=None, help='Maximum number of images to process (optional)')
    
    args = parser.parse_args()
    
    # Mask override dictionary - use 1-based indices for user input
    mask_override_dict = {
        # Example: "image_name": mask_index (1-based)
        # "IMG_1234": 3,  # This will use the 3rd mask (index 2 internally)
        # "DSC_5678": 2,  # This will use the 2nd mask (index 1 internally)
    }
    
    # Process images
    process_images(
        args.input, 
        args.output, 
        args.max, 
        mask_override_dict
    )

if __name__ == "__main__":
    # For testing without command line arguments, you can set these variables:
    INPUT_FOLDER = "test_images"  # Change this to your input folder path
    OUTPUT_FOLDER = "test_output"  # Change this to your output folder path
    MAX_IMAGES = None  # Set to None to process all images, or a number to limit
    
    # Mask override dictionary - use 1-based indices for user input
    MASK_OVERRIDE_DICT = {
        # Example: "image_name": mask_index (1-based)
        # "IMG_1234": 3,  # This will use the 3rd mask (index 2 internally)
        # "DSC_5678": 2,  # This will use the 2nd mask (index 1 internally)
        "IMG20250703084935": 2,
        "IMG20250703085534": 3
    }
    
    # Uncomment the next line to use the hardcoded paths for testing
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, MAX_IMAGES, MASK_OVERRIDE_DICT)
    
    # Or use command line arguments (comment out the line above)
    # main()