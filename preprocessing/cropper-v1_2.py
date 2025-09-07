import os
import cv2
import torch
import shutil
import numpy as np
from pathlib import Path
from collections import Counter

class CitrusDetector:
    def __init__(self, input_folder, output_folder, log_file_path, process_limit=None):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.log_file_path = Path(log_file_path)
        self.process_limit = process_limit
        
        # Load YOLOv5 model with adjusted parameters (Adjustment #7)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = 0.01  # Confidence threshold
        self.model.iou = 0.7    # IOU threshold (Adjustment #2)
        self.model.agnostic = False
        self.model.multi_label = False
        self.model.max_det = 1000
        
        # Minimum area for valid detection (Adjustment #5)
        self.min_pomelo_area = 5000
        
        # Citrus-related classes and additional specified classes
        self.citrus_classes = {
            'orange', 'apple', 'lemon', 'lime', 'fruit', 'banana', 
            'cake', 'vase', 'suitcase', 'broccoli'
        }
        
        self.all_detected_classes = set()
        self.non_citrus_classes = set()
    
    def setup_folders(self):
        """Create necessary folders if they don't exist"""
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    def process_image(self, image_path):
        """Process a single image and return detection results with counts"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None, [], {}
        
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
        
        # Apply size filtering (Adjustment #5)
        if not citrus_detections.empty:
            citrus_detections = citrus_detections.copy()
            citrus_detections['area'] = (citrus_detections['xmax'] - citrus_detections['xmin']) * \
                                      (citrus_detections['ymax'] - citrus_detections['ymin'])
            # Filter out small detections
            citrus_detections = citrus_detections[citrus_detections['area'] >= self.min_pomelo_area]
        
        # If no citrus detected, return None with class counts
        if citrus_detections.empty:
            return None, list(detected_classes), dict(class_counts)
        
        # Find the largest bounding box (by area)
        largest_detection = citrus_detections.loc[citrus_detections['area'].idxmax()]
        
        # Convert coordinates to integers
        xmin = int(largest_detection['xmin'])
        ymin = int(largest_detection['ymin'])
        xmax = int(largest_detection['xmax'])
        ymax = int(largest_detection['ymax'])
        
        # Crop the image
        cropped_img = img[ymin:ymax, xmin:xmax]
        
        return cropped_img, list(detected_classes), dict(class_counts)
    
    def save_cropped_image(self, original_path, cropped_img):
        """Save cropped image with appropriate naming"""
        output_filename = f"cropped_{original_path.stem}{original_path.suffix}"
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
        
        # Open log file
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("CITRUS DETECTION PROCESSING LOG\n")
            log_file.write("=" * 50 + "\n\n")
            log_file.write(f"Configuration: conf={self.model.conf}, iou={self.model.iou}, min_area={self.min_pomelo_area}px\n\n")
            
            processed_count = 0
            successful_detections = 0
            skipped_small_detections = 0
            
            for image_path in image_files:
                print(f"Processing: {image_path.name}")
                
                # Process image (now returns class_counts as third element)
                cropped_img, detected_classes, class_counts = self.process_image(image_path)
                
                # Check if detection was skipped due to size
                citrus_detected = any(cls in self.citrus_classes for cls in detected_classes)
                citrus_but_too_small = citrus_detected and cropped_img is None
                if citrus_but_too_small:
                    skipped_small_detections += 1
                    print(f"  ⚠ Citrus detected but too small - skipped")
                
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
            
            # Write summary and unique non-citrus classes
            log_file.write(f"\n{'=' * 50}\n")
            log_file.write("PROCESSING SUMMARY\n")
            log_file.write(f"{'=' * 50}\n")
            log_file.write(f"Total images processed: {processed_count}\n")
            log_file.write(f"Successful citrus detections: {successful_detections}\n")
            log_file.write(f"Failed detections: {processed_count - successful_detections}\n")
            log_file.write(f"Small detections skipped: {skipped_small_detections}\n")
            
            log_file.write(f"\nUnique non-citrus classes detected:\n")
            if self.non_citrus_classes:
                for cls in sorted(self.non_citrus_classes):
                    log_file.write(f"  - {cls}\n")
            else:
                log_file.write("  None\n")
        
        print(f"\nProcessing complete!")
        print(f"Processed {processed_count} images, found citrus in {successful_detections}")
        print(f"Skipped {skipped_small_detections} small detections")
        print(f"Log file saved to: {self.log_file_path}")
        print(f"Cropped images saved to: {self.output_folder}")

def main():
    # Configuration - modify these paths as needed
    INPUT_FOLDER = "test_images"  # Change this to your input folder
    OUTPUT_FOLDER = "test_output"  # Change this to your output folder
    LOG_FILE_PATH = "test_output/citrus_detection_log.txt"  # Change this to your log file path
    PROCESS_LIMIT = None  # Set to None to process all images, or a number to limit
    
    # Initialize and run detector
    detector = CitrusDetector(INPUT_FOLDER, OUTPUT_FOLDER, LOG_FILE_PATH, PROCESS_LIMIT)
    detector.process_all_images()

if __name__ == "__main__":
    main()