import os
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import shutil

load_dotenv()

def rgba_to_bgr_avg_background(image_bgra, alpha_threshold=128):
    """Convert BGRA to BGR, replacing transparent pixels with avg BGR of non-transparent regions."""
    bgr = image_bgra[..., :3].astype(np.float32)
    alpha = image_bgra[..., 3].astype(np.float32) / 255.0
    mask = alpha >= (alpha_threshold / 255.0)

    if np.any(mask):
        mean_color = bgr[mask].mean(axis=0)
    else:
        mean_color = np.array([127, 127, 127], dtype=np.float32)

    background = np.full_like(bgr, mean_color, dtype=np.float32)
    out = np.where(mask[..., None], bgr, background)
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_pomelo_image(image_bgr):
    # Convert to LAB for luminance and chroma control
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Step 1: Luminance equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Step 2: Decrease overall green dominance (negative 'a' shift → greener tones)
    a_mean, b_mean = np.mean(a), np.mean(b)
    a = np.clip((a - a_mean) * 1.20 + a_mean, 0, 255).astype(np.uint8)
    b = np.clip((b - b_mean) * 1.15 + b_mean, 0, 255).astype(np.uint8)
    l = l.astype(np.uint8)

    a = cv2.resize(a, (l.shape[1], l.shape[0]))
    b = cv2.resize(b, (l.shape[1], l.shape[0]))

    lab = cv2.merge((l, a, b))
    enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Step 3: Targeted color rebalancing (HSV domain)
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Boost brown, tan, yellow ranges (H 1–50°)
    warm_mask = (h >= 1) & (h <= 50)
    s[warm_mask] *= 1.2
    v[warm_mask] *= 1.1

    # Define a broader green mask to cover light and dark greens (approx 32° to 160°)
    green_mask = (h >= 32) & (h <= 160)
    s[green_mask] = np.clip(s[green_mask] * 1.05, 0, 255)

    # Enhance black/white/tan visibility by range normalization
    global_mask = (v > 30) & (v < 220) & ((h < 32) | (h > 160))
    v[global_mask] = np.power(v[global_mask] / 255.0, 0.8) * 255.0

    hsv = cv2.merge((h, np.clip(s, 0, 255), np.clip(v, 0, 255)))
    enhanced_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Step 4: Tone balancing and mild sharpening
    enhanced_bgr = np.clip(np.power(enhanced_bgr.astype(np.float32) / 255.0, 1.05) * 255, 0, 255).astype(np.uint8)
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=1.0)
    enhanced_bgr = cv2.addWeighted(enhanced_bgr, 1.4, gaussian, -0.40, 0)

    gamma = 1.1
    enhanced_bgr = np.clip((enhanced_bgr / 255.0) ** (1 / gamma) * 255, 0, 255).astype(np.uint8)

    # Step 5: Normalize output for model use
    enhanced_bgr = cv2.normalize(enhanced_bgr, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced_bgr


def process_image_file(input_path, output_path):
    """Process and save a single image as RGB JPEG."""
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[WARN] Skipping unreadable file: {input_path}")
        return False

    # --- Handle RGBA → BGR conversion ---
    if image.shape[-1] == 4:
        image = rgba_to_bgr_avg_background(image)
    elif image.shape[-1] == 3:
        image = image.astype(np.uint8)
    else:
        print(f"[WARN] Unsupported image shape {image.shape} for {input_path}")
        return False

    # --- Apply enhancement ---
    enhanced_bgr = enhance_pomelo_image(image)

    # --- Convert to RGB for model / Pillow ---
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    # --- Ensure output folder exists ---
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Save as true RGB JPEG using Pillow ---
    Image.fromarray(enhanced_rgb).save(output_path, quality=95, subsampling=0)
    return True


def build_image_dictionary(input_dir, output_dir):
    """Build dictionary of image names with their input and output paths."""
    image_dict = {}
    input_duplicates = []
    output_duplicates = []
    
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    
    # Scan input directory
    print(f"[INFO] Scanning input directory: {input_dir}")
    for path in input_dir.rglob("*"):
        if path.suffix.lower() in image_extensions and path.is_file():
            image_name = path.stem  # filename without extension
            
            if image_name in image_dict:
                # Already found this image name, mark as duplicate
                input_duplicates.append(path)
            else:
                # First time seeing this image name
                image_dict[image_name] = [path, None]
    
    # Scan output directory
    print(f"[INFO] Scanning output directory: {output_dir}")
    for path in output_dir.rglob("*"):
        if path.suffix.lower() in image_extensions and path.is_file():
            image_name = path.stem  # filename without extension
            
            if image_name in image_dict:
                if image_dict[image_name][1] is None:
                    # First output path for this image name
                    image_dict[image_name][1] = path
                else:
                    # Already have an output path, mark as duplicate
                    output_duplicates.append(path)
            else:
                # Image name not in input, create entry with None input
                image_dict[image_name] = [None, path]
    
    print(f"[INFO] Found {len(image_dict)} unique image names")
    print(f"[INFO] Found {len(input_duplicates)} input duplicates")
    print(f"[INFO] Found {len(output_duplicates)} output duplicates")
    
    return image_dict, input_duplicates, output_duplicates


def delete_duplicates(duplicate_list):
    """Delete duplicate files from the provided list."""
    if not duplicate_list:
        print("[INFO] No duplicates to delete")
        return
    
    print(f"[INFO] Deleting {len(duplicate_list)} duplicate files...")
    for duplicate_path in tqdm(duplicate_list, desc="Deleting duplicates"):
        try:
            duplicate_path.unlink()
            print(f"[INFO] Deleted duplicate: {duplicate_path}")
        except Exception as e:
            print(f"[ERROR] Failed to delete {duplicate_path}: {e}")


def ensure_same_subfolder_structure(input_path, output_path, input_base, output_base):
    """Ensure output file has same subfolder structure as input."""
    if input_path is None:
        return output_path
    
    rel_path = input_path.relative_to(input_base)
    expected_output_path = output_base / rel_path
    
    if output_path != expected_output_path:
        # Move file to correct location
        expected_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if output_path.exists():
                shutil.move(str(output_path), str(expected_output_path))
                print(f"[INFO] Moved {output_path} to {expected_output_path}")
            return expected_output_path
        except Exception as e:
            print(f"[ERROR] Failed to move {output_path} to {expected_output_path}: {e}")
    
    return output_path


def process_smart_dataset(input_dir, output_dir):
    """Smart processing with duplicate management and file organization."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Step 1: Build image dictionary and identify duplicates
    image_dict, input_duplicates, output_duplicates = build_image_dictionary(input_dir, output_dir)
    
    # Step 2: Delete all redundant duplicates
    all_duplicates = input_duplicates + output_duplicates
    delete_duplicates(all_duplicates)
    
    # Step 3: Process images based on their state
    processed_count = 0
    skipped_count = 0
    deleted_count = 0
    error_count = 0
    
    print("[INFO] Processing images based on their state...")
    for image_name, (input_path, output_path) in tqdm(image_dict.items(), desc="Processing images"):
        # Ensure output path has correct subfolder structure if it exists
        if output_path is not None:
            output_path = ensure_same_subfolder_structure(input_path, output_path, input_dir, output_dir)
            image_dict[image_name] = [input_path, output_path]  # Update with potentially moved path
        
        # Case 3c: Output exists but input doesn't - delete output
        if input_path is None and output_path is not None:
            try:
                output_path.unlink()
                deleted_count += 1
                print(f"[INFO] Deleted orphaned output: {output_path}")
            except Exception as e:
                print(f"[ERROR] Failed to delete {output_path}: {e}")
                error_count += 1
        
        # Case 3a: Both input and output exist - ensure correct structure
        elif input_path is not None and output_path is not None:
            # Already handled by ensure_same_subfolder_structure above
            skipped_count += 1
        
        # Case 3b: Input exists but output doesn't - process image
        elif input_path is not None and output_path is None:
            # Determine output path based on input structure
            rel_path = input_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.jpg')
            
            success = process_image_file(input_path, output_path)
            if success:
                processed_count += 1
                # Update the dictionary with the new output path
                image_dict[image_name][1] = output_path
            else:
                error_count += 1
    
    print(f"[DONE] Summary:")
    print(f"  - Processed: {processed_count} new images")
    print(f"  - Skipped: {skipped_count} existing images")
    print(f"  - Deleted: {deleted_count} orphaned outputs")
    print(f"  - Errors: {error_count} files")
    print(f"  - Total unique images: {len(image_dict)}")


if __name__ == "__main__":
    input_folder = os.getenv("POMELO_IMAGE_ENHANCER_INPUT_FOLDER")
    output_folder = os.getenv("POMELO_IMAGE_ENHANCER_OUTPUT_FOLDER")

    if not input_folder or not output_folder:
        print("Error: Environment variables POMELO_IMAGE_ENHANCER_INPUT_FOLDER and POMELO_IMAGE_ENHANCER_OUTPUT_FOLDER must be set.")
        exit(1)

    process_smart_dataset(input_folder, output_folder)
