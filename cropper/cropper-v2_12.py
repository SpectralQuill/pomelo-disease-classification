import os
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import SamPredictor, sam_model_registry
import argparse
from pathlib import Path
import time
import csv
import signal
import sys

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\nGraceful shutdown requested. Finishing current task and stopping...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

class SAMImageProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="cropper/sam_vit_h_4b8939.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
    
    def process_image(self, image_path, output_path, monitoring_path=None,
                      mask_override_dict=None, point_override_dict=None):
        start_time = time.time()
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return False, 0, None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)
            
            height, width = image_rgb.shape[:2]
            image_name = Path(image_path).stem

            # Use point override if provided
            if point_override_dict and image_name in point_override_dict:
                override_x, override_y = point_override_dict[image_name]
                if override_x is None:
                    override_x = width // 2
                if override_y is None:
                    override_y = height // 2
                center_point = np.array([[override_x, override_y]])
                print(f"Using override point {override_x}, {override_y} for {image_name}")
            else:
                center_point = np.array([[width // 2, height // 2]])
            point_labels = np.array([1])

            masks, scores, _ = self.predictor.predict(
                point_coords=center_point,
                point_labels=point_labels,
                multimask_output=True,
            )

            # Clean masks
            filtered_masks, filtered_scores = [], []
            for mask, score in zip(masks, scores):
                cleaned_mask = self._remove_small_segments(mask)
                filtered_masks.append(cleaned_mask)
                filtered_scores.append(score)
            masks, scores = np.array(filtered_masks), np.array(filtered_scores)

            # Select mask (override if available)
            selected_mask_index = None
            if mask_override_dict and image_name in mask_override_dict:
                override_index = mask_override_dict[image_name] - 1
                if 0 <= override_index < len(masks):
                    best_mask = masks[override_index]
                    selected_mask_index = override_index + 1
                    print(f"Using override mask index {selected_mask_index} for {image_name}")
                else:
                    best_mask, selected_mask_index = self._select_best_mask(masks, scores, image_rgb.shape)
            else:
                best_mask, selected_mask_index = self._select_best_mask(masks, scores, image_rgb.shape)

            if best_mask is None:
                print(f"No suitable mask found for {image_path}")
                return False, 0, None

            if monitoring_path:
                self._create_monitoring_visualization(image_rgb, masks, scores,
                                                      monitoring_path, center_point, selected_mask_index)

            rgba_image = self._create_transparent_image(image_rgb, best_mask)
            trimmed_image = self._trim_transparent(rgba_image)
            trimmed_image.save(output_path, 'PNG')

            processing_time = time.time() - start_time
            print(f"Processed and saved: {output_path} (Mask {selected_mask_index}, Time: {processing_time:.2f}s)")
            return True, processing_time, selected_mask_index

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False, 0, None

    def _remove_small_segments(self, mask, min_area_ratio=0.05):
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        total_area = np.sum(mask)
        new_mask = np.zeros_like(mask, dtype=bool)
        for contour in contours:
            temp_mask = np.zeros_like(mask_uint8)
            cv2.drawContours(temp_mask, [contour], -1, 255, -1)
            segment_area = np.sum(temp_mask > 0)
            if segment_area / total_area > min_area_ratio:
                new_mask |= (temp_mask > 0)
        return new_mask

    def _create_monitoring_visualization(self, image_rgb, masks, scores, monitoring_path, center_point, chosen_index=None):
        num_masks = len(masks)
        if num_masks == 0:
            return
        cols = min(4, num_masks)
        rows = (num_masks + cols - 1) // cols
        grid_height = rows * image_rgb.shape[0]
        grid_width = cols * image_rgb.shape[1]
        grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        for i, (mask, score) in enumerate(zip(masks, scores)):
            row, col = i // cols, i % cols
            chosen = (i + 1 == chosen_index)
            mask_visualization = self._visualize_mask(image_rgb, mask, score, i, center_point, chosen)
            y_start, y_end = row * image_rgb.shape[0], (row + 1) * image_rgb.shape[0]
            x_start, x_end = col * image_rgb.shape[1], (col + 1) * image_rgb.shape[1]
            grid_canvas[y_start:y_end, x_start:x_end] = mask_visualization
        Image.fromarray(grid_canvas).save(monitoring_path)
        print(f"Monitoring visualization saved: {monitoring_path}")

    def _visualize_mask(self, image_rgb, mask, score, mask_index, center_point, chosen=False):
        visualization = image_rgb.copy()
        mask_rgb = np.zeros_like(visualization)
        mask_rgb[mask] = [255, 0, 0]
        visualization = cv2.addWeighted(visualization, 1.0, mask_rgb, 0.3, 0)
        center_x, center_y = center_point[0]
        cv2.circle(visualization, (center_x, center_y), 10, (0, 255, 255), -1)
        cv2.circle(visualization, (center_x, center_y), 12, (0, 0, 0), 2)

        coords = np.column_stack(np.where(mask))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            cv2.rectangle(visualization, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mask_center_x = (x_min + x_max) // 2
            mask_center_y = (y_min + y_max) // 2
            cv2.circle(visualization, (mask_center_x, mask_center_y), 5, (255, 0, 255), -1)

        combined_score, circularity, area_ratio, position_score = self._calculate_combined_score(mask, score, image_rgb.shape)
        y_offset, line_height = 30, 25
        text_color = [255, 0, 0]
        mask_text = f"Mask {mask_index + 1}{' (Selected)' if chosen else ''}"
        cv2.putText(visualization, mask_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        y_offset += line_height
        total_score_text = f"Total Score: {combined_score * 100:.1f}%"
        cv2.putText(visualization, total_score_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        y_offset += line_height
        circularity_text = f"Circularity: {circularity * 100:.1f}%"
        cv2.putText(visualization, circularity_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        y_offset += line_height
        sam_score_text = f"SAM Confidence: {score * 100:.1f}%"
        cv2.putText(visualization, sam_score_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        y_offset += line_height
        area_text = f"Image Area: {area_ratio * 100:.1f}%"
        cv2.putText(visualization, area_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        y_offset += line_height
        position_text = f"Position Score: {position_score * 100:.1f}%"
        cv2.putText(visualization, position_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        return visualization

    def _select_best_mask(self, masks, scores, image_shape):
        if len(masks) == 0:
            return None, None
        mask_scores = []
        for i, mask in enumerate(masks):
            combined_score, _, _, _ = self._calculate_combined_score(mask, scores[i], image_shape)
            mask_scores.append((mask, combined_score, i))
        mask_scores.sort(key=lambda x: x[1], reverse=True)
        best_mask, _, best_index = mask_scores[0]
        return best_mask, best_index + 1

    def _calculate_combined_score(self, mask, sam_score, image_shape):
        circularity = self._calculate_circularity(mask)
        mask_area = np.sum(mask)
        image_area = image_shape[0] * image_shape[1]
        area_ratio = mask_area / image_area
        combined_score = 0.4 * circularity + 0.3 * sam_score + 0.2 * min(1.0, area_ratio * 3)
        return combined_score, circularity, area_ratio, 0

    def _calculate_circularity(self, mask):
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest_contour = max(contours, key=cv2.contourArea)
        area, perimeter = cv2.contourArea(largest_contour), cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return 0.0
        return (4 * np.pi * area) / (perimeter * perimeter)

    def _create_transparent_image(self, image_rgb, mask):
        rgba = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = image_rgb
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        return Image.fromarray(rgba)

    def _trim_transparent(self, image):
        image_np = np.array(image)
        alpha = image_np[:, :, 3]
        coords = np.column_stack(np.where(alpha > 0))
        if len(coords) == 0:
            return image
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = image_np[y_min:y_max+1, x_min:x_max+1]
        return Image.fromarray(cropped)

def read_csv_status_and_overrides(csv_path):
    status_dict = {}
    mask_override_dict = {}
    point_override_dict = {}
    statuses_to_skip = {"Processed", "Partial", "Unusable"}
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 6:
                    image_name = row[0]
                    is_skipped = (row[2].upper() in statuses_to_skip)
                    status_dict[image_name] = is_skipped

                    # Mask override (col 4)
                    if row[3].strip():
                        try:
                            mask_override_dict[image_name] = int(row[3])
                        except ValueError:
                            pass

                    # Point overrides (cols 5 and 6)
                    override_x = int(row[4]) if row[4].strip() else None
                    override_y = int(row[5]) if row[5].strip() else None
                    if override_x is not None or override_y is not None:
                        point_override_dict[image_name] = (override_x, override_y)

        print(f"Read status and overrides for {len(status_dict)} images from CSV")
    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}. All images will be processed.")
    return status_dict, mask_override_dict, point_override_dict

def process_images(input_folder, output_folder, max_images=None, csv_path=None):
    global shutdown_requested
    os.makedirs(output_folder, exist_ok=True)
    monitoring_folder = os.path.join(output_folder, "monitoring")
    os.makedirs(monitoring_folder, exist_ok=True)

    csv_status, mask_override_dict, point_override_dict = read_csv_status_and_overrides(csv_path) if csv_path else ({}, {}, {})
    processor = SAMImageProcessor()

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    all_files = [f for f in os.listdir(input_folder) if Path(f).suffix.lower() in image_extensions]

    # Filter out processed images BEFORE limiting
    unprocessed_files = [os.path.join(input_folder, f) for f in all_files if not (csv_status.get(Path(f).stem, False))]

    if max_images is not None:
        unprocessed_files = unprocessed_files[:max_images]
    if not unprocessed_files:
        print("No unprocessed images to process.")
        return

    print(f"Found {len(unprocessed_files)} images to process...")
    successful, total_processing_time, processing_times = 0, 0, []
    for i, image_path in enumerate(unprocessed_files, 1):
        if shutdown_requested:
            print("Shutdown signal detected. Stopping further processing.")
            break
        image_name = Path(image_path).stem
        print(f"Processing image {i}/{len(unprocessed_files)}: {os.path.basename(image_path)}")
        output_path = os.path.join(output_folder, f"{image_name}.png")
        monitoring_path = os.path.join(monitoring_folder, f"monitoring_{image_name}.jpg")
        success, processing_time, _ = processor.process_image(
            image_path, output_path, monitoring_path,
            mask_override_dict, point_override_dict
        )
        if success:
            successful += 1
            total_processing_time += processing_time
            processing_times.append(processing_time)
    if processing_times:
        avg_time = total_processing_time / len(processing_times)
        print(f"\nProcessing complete! {successful}/{len(unprocessed_files)} images processed successfully.")
        print(f"Average processing time: {avg_time:.2f}s")
    else:
        print(f"\nProcessing complete! No images processed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Segment images using SAM and remove background')
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--max', '-m', type=int, default=None)
    parser.add_argument('--csv', default=None)
    args = parser.parse_args()

    process_images(args.input, args.output, args.max, args.csv)

if __name__ == "__main__":
    INPUT_FOLDER = "test_images"
    OUTPUT_FOLDER = "test_output"
    MAX_IMAGES = 3
    CSV_PATH = "tracker/tracker.csv"
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, MAX_IMAGES, CSV_PATH)
