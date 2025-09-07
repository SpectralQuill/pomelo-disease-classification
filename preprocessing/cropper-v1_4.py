import os
import cv2
import torch
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

class CitrusDetector:
    def __init__(self, input_folder, output_folder, log_file_path, process_limit=None, monitor_images=None):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.log_file_path = Path(log_file_path)
        self.process_limit = process_limit
        self.monitor_images = set(monitor_images) if monitor_images else set()
        
        # Load YOLOv5 model with adjusted parameters (Adjustment #7)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = 0.02  # Confidence threshold
        self.model.iou = 0.99
        self.model.agnostic = False
        self.model.multi_label = False
        self.model.max_det = 1000
        self.min_pomelo_area = 5000
        self.max_aspect_ratio = 1.4
        
        # Citrus-related classes and additional specified classes
        self.citrus_classes = {
            'orange', 'apple', 'lemon', 'lime', 'fruit', 'banana', 
            'cake', 'vase', 'suitcase', 'broccoli', 'teddy bear', 'tie'
        }
        
        self.all_detected_classes = set()
        self.non_citrus_classes = set()
        self.detection_stats = Counter()  # Track filtering statistics
    
    def setup_folders(self):
        """Create necessary folders if they don't exist"""
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create monitoring folder if monitoring is enabled
        if self.monitor_images:
            self.monitor_folder = self.output_folder / "monitoring"
            self.monitor_folder.mkdir(parents=True, exist_ok=True)
    
    def get_image_files(self):
        """Get all image files from input folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file_path in self.input_folder.iterdir():
            if file_path.suffix.lower() in image_extensions and file_path.is_file():
                image_files.append(file_path)
        
        # Apply process limit if specified
        if self.process_limit:
            image_files = image_files[:self.process_limit]
        
        return image_files
    
    def enhance_contrast(self, image):
        """
        Enhance image contrast using CLAHE (Adjustment #3)
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"Contrast enhancement failed: {e}")
            return image
    
    def is_valid_aspect_ratio(self, detection):
        """
        Check if detection has valid aspect ratio for citrus
        Returns True if aspect ratio is below the maximum threshold
        """
        width = detection['xmax'] - detection['xmin']
        height = detection['ymax'] - detection['ymin']
        
        # Avoid division by zero
        if height == 0:
            return False
        
        aspect_ratio = width / height if width >= height else height / width
        return aspect_ratio < self.max_aspect_ratio
    
    def draw_bounding_boxes(self, image, detections, citrus_detections, valid_detections):
        """
        Draw bounding boxes on image for monitoring purposes
        """
        img_with_boxes = image.copy()
        
        # Define colors
        citrus_color = (0, 255, 0)  # Green for valid citrus detections
        other_color = (255, 0, 0)   # Blue for other detections
        rejected_color = (0, 0, 255) # Red for rejected citrus detections
        
        # Draw all detections
        for _, detection in detections.iterrows():
            xmin = int(detection['xmin'])
            ymin = int(detection['ymin'])
            xmax = int(detection['xmax'])
            ymax = int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Determine color based on detection type
            if class_name.lower() in self.citrus_classes:
                # Check if this is a valid citrus detection
                is_valid = any((valid_detection['xmin'] == xmin and 
                               valid_detection['ymin'] == ymin and 
                               valid_detection['xmax'] == xmax and 
                               valid_detection['ymax'] == ymax) 
                              for valid_detection in valid_detections)
                color = citrus_color if is_valid else rejected_color
            else:
                color = other_color
            
            # Draw bounding box
            cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img_with_boxes, label, (xmin, ymin - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_with_boxes
    
    def save_monitoring_image(self, image_path, image_with_boxes):
        """Save monitoring image with bounding boxes"""
        output_filename = f"monitor_{image_path.stem}{image_path.suffix}"
        output_path = self.monitor_folder / output_filename
        cv2.imwrite(str(output_path), image_with_boxes)
        return output_path
    
    def process_image(self, image_path):
        """Process a single image and return detection results with counts"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None, [], {}, None
        
        # Enhance contrast for better detection (Adjustment #3)
        enhanced_img = self.enhance_contrast(img)
        
        # Run YOLO detection
        results = self.model(enhanced_img)
        detections = results.pandas().xyxy[0]
        
        # Count occurrences of each class in this image
        class_counts = Counter()
        for _, detection in detections.iterrows():
            class_name = detection['name']
            class_counts[class_name] += 1
        
        # Filter for citrus-related classes
        citrus_detections = detections[detections['name'].str.lower().isin(self.citrus_classes)]
        
        # Log all detected classes
        detected_classes = set(detections['name'].unique())
        self.all_detected_classes.update(detected_classes)
        self.non_citrus_classes.update(detected_classes - self.citrus_classes)
        
        # Apply size and aspect ratio filtering
        valid_detections = []
        if not citrus_detections.empty:
            citrus_detections = citrus_detections.copy()
            citrus_detections['area'] = (citrus_detections['xmax'] - citrus_detections['xmin']) * \
                                      (citrus_detections['ymax'] - citrus_detections['ymin'])
            
            # Apply both size and aspect ratio filtering
            for _, detection in citrus_detections.iterrows():
                area = detection['area']
                valid_size = area >= self.min_pomelo_area
                valid_aspect = self.is_valid_aspect_ratio(detection)
                
                if valid_size and valid_aspect:
                    valid_detections.append(detection)
                else:
                    # Track why detection was rejected
                    if not valid_size:
                        self.detection_stats['rejected_small'] += 1
                    if not valid_aspect:
                        self.detection_stats['rejected_aspect_ratio'] += 1
        
        # Convert back to DataFrame for consistency
        if valid_detections:
            citrus_detections = pd.DataFrame(valid_detections)
        else:
            citrus_detections = pd.DataFrame(columns=detections.columns)
        
        # Create monitoring image if this image is in the monitoring list
        monitoring_image = None
        if self.monitor_images and image_path.name in self.monitor_images:
            monitoring_image = self.draw_bounding_boxes(img, detections, citrus_detections, valid_detections)
        
        # If no valid citrus detected, return None with class counts
        if citrus_detections.empty:
            return None, list(detected_classes), dict(class_counts), monitoring_image
        
        # Find the largest bounding box (by area)
        largest_detection = citrus_detections.loc[citrus_detections['area'].idxmax()]
        
        # Convert coordinates to integers
        xmin = int(largest_detection['xmin'])
        ymin = int(largest_detection['ymin'])
        xmax = int(largest_detection['xmax'])
        ymax = int(largest_detection['ymax'])
        
        # Crop the image
        cropped_img = img[ymin:ymax, xmin:xmax]
        
        return cropped_img, list(detected_classes), dict(class_counts), monitoring_image
    
    def save_cropped_image(self, original_path, cropped_img):
        """Save cropped image with appropriate naming"""
        output_filename = f"{original_path.stem}{original_path.suffix}"
        output_path = self.output_folder / output_filename
        cv2.imwrite(str(output_path), cropped_img)
        return output_path
    
    def write_log_entry(self, file_handle, image_name, detected_classes, class_counts, success):
        """Write log entry for a single image with class counts"""
        status = "SUCCESS" if success else "NO_CITRUS_DETECTED"
        
        # Format class information with counts
        class_info = []
        for cls in detected_classes:
            count = class_counts.get(cls, 0)
            class_info.append(f"{cls} ({count})")
        
        classes_str = ", ".join(class_info) if class_info else "None"
        file_handle.write(f"{image_name}: {status} - Detected classes: [{classes_str}]\n")
    
    def process_all_images(self):
        """Process all images in the input folder"""
        self.setup_folders()
        image_files = self.get_image_files()
        
        print(f"Found {len(image_files)} images to process")
        print(f"Using confidence threshold: {self.model.conf}")
        print(f"Using IOU threshold: {self.model.iou}")
        print(f"Minimum detection area: {self.min_pomelo_area}px")
        print(f"Maximum aspect ratio: {self.max_aspect_ratio}")
        if self.monitor_images:
            print(f"Monitoring {len(self.monitor_images)} images for manual checking")
        
        # Open log file
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("CITRUS DETECTION PROCESSING LOG\n")
            log_file.write("=" * 50 + "\n\n")
            log_file.write(f"Configuration: conf={self.model.conf}, iou={self.model.iou}, ")
            log_file.write(f"min_area={self.min_pomelo_area}px, max_aspect_ratio={self.max_aspect_ratio}\n\n")
            
            processed_count = 0
            successful_detections = 0
            
            for image_path in image_files:
                print(f"Processing: {image_path.name}")
                
                # Process image (now returns class_counts as third element and monitoring_image as fourth)
                cropped_img, detected_classes, class_counts, monitoring_image = self.process_image(image_path)
                
                # Save monitoring image if available
                if monitoring_image is not None:
                    self.save_monitoring_image(image_path, monitoring_image)
                    print(f"  📊 Monitoring image created")
                
                # Log results with class counts
                success = cropped_img is not None
                self.write_log_entry(log_file, image_path.name, detected_classes, class_counts, success)
                
                if success:
                    # Save cropped image
                    self.save_cropped_image(image_path, cropped_img)
                    successful_detections += 1
                    print(f"  ✓ Citrus detected and cropped")
                else:
                    print(f"  ✗ No citrus detected")
                
                processed_count += 1
            
            # Write summary with detailed filtering statistics
            log_file.write(f"\n{'=' * 50}\n")
            log_file.write("PROCESSING SUMMARY\n")
            log_file.write(f"{'=' * 50}\n")
            log_file.write(f"Total images processed: {processed_count}\n")
            log_file.write(f"Successful citrus detections: {successful_detections}\n")
            log_file.write(f"Failed detections: {processed_count - successful_detections}\n")
            log_file.write(f"Small detections skipped: {self.detection_stats['rejected_small']}\n")
            log_file.write(f"Invalid aspect ratio skipped: {self.detection_stats['rejected_aspect_ratio']}\n")
            
            log_file.write(f"\nUnique non-citrus classes detected:\n")
            if self.non_citrus_classes:
                for cls in sorted(self.non_citrus_classes):
                    log_file.write(f"  - {cls}\n")
            else:
                log_file.write("  None\n")
        
        print(f"\nProcessing complete!")
        print(f"Processed {processed_count} images, found citrus in {successful_detections}")
        print(f"Skipped {self.detection_stats['rejected_small']} small detections")
        print(f"Skipped {self.detection_stats['rejected_aspect_ratio']} invalid aspect ratio detections")
        if self.monitor_images:
            print(f"Monitoring images saved to: {self.monitor_folder}")
        print(f"Log file saved to: {self.log_file_path}")
        print(f"Cropped images saved to: {self.output_folder}")

def main():
    # Configuration - modify these paths as needed
    INPUT_FOLDER = "test_images"  # Change this to your input folder
    OUTPUT_FOLDER = "test_output"  # Change this to your output folder
    LOG_FILE_PATH = "test_output/citrus_detection_log.txt"  # Change this to your log file path
    PROCESS_LIMIT = None  # Set to None to process all images, or a number to limit
    
    # Optional: List of image names to create monitoring images for
    MONITOR_IMAGES = [
        "IMG20250703093342.jpg",
        "IMG20250703093342_2.jpg"
    ]
    
    # Initialize and run detector
    detector = CitrusDetector(
        INPUT_FOLDER, 
        OUTPUT_FOLDER, 
        LOG_FILE_PATH, 
        PROCESS_LIMIT,
        monitor_images=MONITOR_IMAGES  # Pass monitoring images list
    )
    detector.process_all_images()

if __name__ == "__main__":
    main()