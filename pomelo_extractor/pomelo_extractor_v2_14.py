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

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\nGraceful shutdown requested. Finishing current task and stopping...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

class PomeloExtractor:
    CSV_COLUMNS = {
        'NAME': 0,
        'CLASS': 1,
        'STATUS': 2,
        'MASK_INDEX': 3,
        'X_POINT_OVERRIDE': 4,
        'Y_POINT_OVERRIDE': 5,
        'DEBUG': 6
    }
    
    def __init__(self, model_type="vit_h", checkpoint_path="pomelo_extractor/sam_vit_h_4b8939.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        ideal_area_ratio = 1 / 3
        self.ideal_area_ratio = ideal_area_ratio
        self.area_score_coefficient_1 = ((1 - 2 * ideal_area_ratio) / (2 * ideal_area_ratio * (1 - ideal_area_ratio)))
        self.area_score_coefficient_2 = (1 / (2 * ideal_area_ratio * (1 - ideal_area_ratio)))
    
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
            mask_overridden = False
            if mask_override_dict and image_name in mask_override_dict:
                override_index = mask_override_dict[image_name] - 1
                if 0 <= override_index < len(masks):
                    best_mask = masks[override_index]
                    selected_mask_index = override_index + 1
                    mask_overridden = True
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
                                                      monitoring_path, center_point, selected_mask_index, mask_overridden)

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

    def _create_monitoring_visualization(self, image_rgb, masks, scores, monitoring_path,
                                         center_point, chosen_index=None, mask_overriden=False):
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
            mask_visualization = self._visualize_mask(image_rgb, mask, score, i, center_point,
                                                      chosen, mask_overriden)
            y_start, y_end = row * image_rgb.shape[0], (row + 1) * image_rgb.shape[0]
            x_start, x_end = col * image_rgb.shape[1], (col + 1) * image_rgb.shape[1]
            grid_canvas[y_start:y_end, x_start:x_end] = mask_visualization
        Image.fromarray(grid_canvas).save(monitoring_path)
        print(f"Monitoring visualization saved: {monitoring_path}")

    def _visualize_mask(self, image_rgb, mask, score, mask_index, center_point, chosen=False,
                        mask_overridden=False):
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

        combined_score, circularity, area_score, position_score = self._calculate_combined_score(mask, score, image_rgb.shape)
        y_offset, line_height = 30, 25
        text_color = [255, 0, 0]
        mask_text = f"Mask {mask_index + 1}{'' if not chosen else ' (Overridden)' if mask_overridden else ' (Selected)'}"
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
        area_text = f"Area Score: {area_score * 100:.1f}%"
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
        area_score = self._calculate_area_score(mask, image_shape)
        position_score = self._calculate_position_score(mask, image_shape)
        combined_score = (0.4 * circularity + 0.3 * sam_score + 0.2 * area_score
                          + 0.1 * position_score)
        return combined_score, circularity, area_score, position_score

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

    def _calculate_area_score(self, mask, image_shape):
        mask_area = np.sum(mask)
        image_area = image_shape[0] * image_shape[1]
        area_ratio = mask_area / image_area
        area_score = (
            1
            + self.area_score_coefficient_1 * (area_ratio - self.ideal_area_ratio)
            - self.area_score_coefficient_2 * abs(area_ratio - self.ideal_area_ratio)
        )
        return area_score

    def _calculate_position_score(self, mask, image_shape):
        coords = np.column_stack(np.where(mask))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            mask_center_x = (x_min + x_max) // 2
            mask_center_y = (y_min + y_max) // 2
            image_center = (image_shape[1] // 2, image_shape[0] // 2)
            center_distance = np.sqrt((mask_center_x - image_center[0])**2 + 
                                    (mask_center_y - image_center[1])**2)
            max_distance = np.sqrt(image_center[0]**2 + image_center[1]**2)
            position_score = 1 - (center_distance / max_distance)
        else:
            position_score = 0
        return position_score

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
    debug_dict = {}
    
    statuses_to_skip = set()
    with open("configs\\image_statuses.csv", 'r', newline='') as config_file:
        reader = csv.reader(config_file)
        headers = next(reader)
        included_col_index = None
        for i, header in enumerate(headers):
            if header.strip().lower() == "included in extraction":
                included_col_index = i
                break
        if included_col_index is not None:
            for row in reader:
                if len(row) > 0:
                    status_name = row[0].strip()
                    if len(row) > included_col_index and row[included_col_index].upper() == "TRUE":
                        continue
                    else:
                        statuses_to_skip.add(status_name)
    
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if len(row) > PomeloExtractor.CSV_COLUMNS['NAME']:
                image_name = row[PomeloExtractor.CSV_COLUMNS['NAME']]
                
                # Check if we should skip this image (unless it's a debug image)
                is_skipped = (len(row) > PomeloExtractor.CSV_COLUMNS['STATUS'] and 
                                row[PomeloExtractor.CSV_COLUMNS['STATUS']] in statuses_to_skip)
                
                # Check if this is a debug image
                is_debug = (len(row) > PomeloExtractor.CSV_COLUMNS['DEBUG'] and 
                            row[PomeloExtractor.CSV_COLUMNS['DEBUG']].upper() == "TRUE")
                
                # Debug images override skip status
                status_dict[image_name] = is_skipped and not is_debug
                debug_dict[image_name] = is_debug
                
                # Read mask override if available
                if (len(row) > PomeloExtractor.CSV_COLUMNS['MASK_INDEX'] and 
                    row[PomeloExtractor.CSV_COLUMNS['MASK_INDEX']].strip()):
                    try:
                        mask_override_dict[image_name] = int(row[PomeloExtractor.CSV_COLUMNS['MASK_INDEX']])
                    except ValueError:
                        pass
                
                # Read point overrides if available
                override_x = None
                override_y = None
                if len(row) > PomeloExtractor.CSV_COLUMNS['X_POINT_OVERRIDE']:
                    try:
                        override_x = int(row[PomeloExtractor.CSV_COLUMNS['X_POINT_OVERRIDE']]) if row[PomeloExtractor.CSV_COLUMNS['X_POINT_OVERRIDE']].strip() else None
                    except ValueError:
                        pass
                
                if len(row) > PomeloExtractor.CSV_COLUMNS['Y_POINT_OVERRIDE']:
                    try:
                        override_y = int(row[PomeloExtractor.CSV_COLUMNS['Y_POINT_OVERRIDE']]) if row[PomeloExtractor.CSV_COLUMNS['Y_POINT_OVERRIDE']].strip() else None
                    except ValueError:
                        pass
                
                if override_x is not None or override_y is not None:
                    point_override_dict[image_name] = (override_x, override_y)
    
    print(f"Read status and overrides for {len(status_dict)} images from CSV")
    return status_dict, mask_override_dict, point_override_dict, debug_dict

def run_pomelo_extractor(input_folder, output_folder, max_images=None, csv_path=None,
                         ignore_subfolders=None):
    global shutdown_requested
    os.makedirs(output_folder, exist_ok=True)
    
    if ignore_subfolders is None:
        ignore_subfolders = set()
    
    csv_status, mask_override_dict, point_override_dict,debug_dict = read_csv_status_and_overrides(csv_path) if csv_path else ({}, {}, {}, {})
    processor = PomeloExtractor()

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    all_files = []
    for root, dirs, files in os.walk(input_folder):
        # Skip ignored subfolders
        rel_path = Path(root).relative_to(input_folder)
        if any(part in ignore_subfolders for part in rel_path.parts):
            continue
            
        for f in files:
            if Path(f).suffix.lower() in image_extensions and len(rel_path.parts) <= 1:
                all_files.append(os.path.relpath(os.path.join(root, f), input_folder))

    all_files = sorted(all_files, key=lambda x: Path(x).stem.lower())

    # Separate debug images from regular images
    debug_files = []
    regular_files = []
    
    for f in all_files:
        image_name = Path(f).stem
        if debug_dict.get(image_name, False):
            debug_files.append(os.path.join(input_folder, f))
        elif not csv_status.get(image_name, False):
            regular_files.append(os.path.join(input_folder, f))

    # Process regular images first
    if max_images is not None:
        regular_files = regular_files[:max_images]
    
    # Create monitoring folder only if we have regular images to process
    monitoring_folder = None
    if regular_files:
        monitoring_folder = os.path.join(output_folder, "monitoring")
        os.makedirs(monitoring_folder, exist_ok=True)
    
    # Create debugging folder only if we have debug images
    debugging_folder = None
    if debug_files:
        debugging_folder = os.path.join(output_folder, "debugging")
        os.makedirs(debugging_folder, exist_ok=True)
        debugging_monitoring_folder = os.path.join(debugging_folder, "monitoring")
        os.makedirs(debugging_monitoring_folder, exist_ok=True)

    files_to_process = regular_files + debug_files
    if not files_to_process:
        print("No images to process.")
        return

    print(f"Found {len(files_to_process)} images to process ({len(regular_files)} regular, {len(debug_files)} debug)...")
    successful, total_processing_time, processing_times = 0, 0, []
    
    for i, image_path in enumerate(files_to_process, 1):
        if shutdown_requested:
            print("Shutdown signal detected. Stopping further processing.")
            break
            
        image_name = Path(image_path).stem
        is_debug = debug_dict.get(image_name, False)
        print(f"Processing image {i}/{len(files_to_process)}: {os.path.basename(image_path)} {'(DEBUG)' if is_debug else ''}")
        
        # Determine output paths based on whether it's a debug image
        if is_debug:
            output_path = os.path.join(debugging_folder, f"{image_name}.png")
            monitoring_path = os.path.join(debugging_monitoring_folder, f"monitoring_{image_name}.jpg")
        else:
            output_path = os.path.join(output_folder, f"{image_name}.png")
            monitoring_path = os.path.join(monitoring_folder, f"monitoring_{image_name}.jpg") if monitoring_folder else None

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
        print(f"\nProcessing complete! {successful}/{len(files_to_process)} images processed successfully.")
        print(f"Average processing time: {avg_time:.2f}s")
    else:
        print(f"\nProcessing complete! No images processed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Segment pomelo images using SAM and remove background')
    parser.add_argument('--input', '-i', default=r"images\raw", required=True)
    parser.add_argument('--output', '-o', default=r"images\extracted", required=True)
    parser.add_argument('--max', '-m', type=int, default=None)
    parser.add_argument('--csv', default=r"tracker\tracker.csv")
    parser.add_argument('--ignore', nargs='+', default=[], help='Subfolder names to ignore')
    args = parser.parse_args()
    run_pomelo_extractor(args.input, args.output, args.max, args.csv, set(args.ignore))

if __name__ == "__main__":
    main()
