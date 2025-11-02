import os
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

load_dotenv()

# ============================================================
# Pomelo Image Enhancer v1.4
# Natural color preservation + RGB-safe output
# ============================================================

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
    """
    Balanced pomelo enhancer for disease detection:
    - Boosts micro-texture contrast (tiny white/black spots)
    - Preserves natural tone and color
    - Avoids blown-out highlights
    """

    # --- Step 1: LAB luminance contrast ---
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for even local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Mild chroma stretch to enhance subtle color tone differences
    a_mean, b_mean = np.mean(a), np.mean(b)
    a = np.clip((a - a_mean) * 1.15 + a_mean, 0, 255).astype(np.uint8)
    b = np.clip((b - b_mean) * 1.10 + b_mean, 0, 255).astype(np.uint8)

    lab = cv2.merge((l, a, b))
    enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- Step 2: Micro-texture high-pass enhancement ---
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    detail = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
    # Mix back as a luminance detail layer (gentle 0.4–0.5)
    enhanced_bgr = cv2.addWeighted(enhanced_bgr, 1.0,
                                   cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR),
                                   0.45, 0)

    # --- Step 3: Hue-targeted adjustment (maintain color realism) ---
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Slightly boost rust and yellow tones
    rust_mask = ((h < 20) | (h > 160))
    s[rust_mask] *= 1.15
    v[rust_mask] *= 1.08

    yellow_mask = (h >= 20) & (h <= 50)
    s[yellow_mask] *= 1.10
    v[yellow_mask] *= 1.05

    hsv = cv2.merge((h, np.clip(s, 0, 255), np.clip(v, 0, 255)))
    enhanced_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # --- Step 4: Tone compression to avoid overexposure ---
    enhanced_bgr = np.clip(enhanced_bgr.astype(np.float32) / 255.0, 0, 1.0)
    # Compress bright regions slightly while preserving midtones
    enhanced_bgr = np.power(enhanced_bgr, 0.95)
    enhanced_bgr = np.clip(enhanced_bgr * 255, 0, 255).astype(np.uint8)

    # --- Step 5: Mild unsharp mask for crisp edges ---
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=0.7)
    enhanced_bgr = cv2.addWeighted(enhanced_bgr, 1.25, gaussian, -0.25, 0)

    # --- Step 6: Normalize final output for model consistency ---
    enhanced_bgr = cv2.normalize(enhanced_bgr, None, 0, 255, cv2.NORM_MINMAX)

    return enhanced_bgr

def process_image_file(input_path, output_path):
    """Process and save a single image as RGB JPEG."""
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[WARN] Skipping unreadable file: {input_path}")
        return

    # --- Handle RGBA → BGR conversion ---
    if image.shape[-1] == 4:
        image = rgba_to_bgr_avg_background(image)
    elif image.shape[-1] == 3:
        image = image.astype(np.uint8)
    else:
        print(f"[WARN] Unsupported image shape {image.shape} for {input_path}")
        return

    # --- Apply enhancement ---
    enhanced_bgr = enhance_pomelo_image(image)

    # --- Convert to RGB for model / Pillow ---
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    # --- Ensure output folder exists ---
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Save as true RGB JPEG using Pillow ---
    Image.fromarray(enhanced_rgb).save(output_path, quality=95, subsampling=0)


def process_dataset(input_dir, output_dir, limit_per_folder=None):
    """Recursively process all images from input_dir → output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    print(f"[INFO] Scanning for images under {input_dir}...")

    folder_to_images = {}
    for path in input_dir.rglob("*"):
        if path.suffix.lower() in image_extensions and path.is_file():
            folder_to_images.setdefault(path.parent, []).append(path)

    total_folders = len(folder_to_images)
    print(f"[INFO] Found {total_folders} folders with image files.")

    total_images = 0
    for folder_idx, (folder, image_paths) in enumerate(folder_to_images.items(), start=1):
        image_paths.sort()
        if limit_per_folder is not None:
            image_paths = image_paths[:limit_per_folder]

        total_images += len(image_paths)
        print(f"[{folder_idx}/{total_folders}] Processing {len(image_paths)} images in: {folder}")

        for input_path in tqdm(image_paths, desc=f"Folder {folder.name}", leave=False):
            rel_path = input_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            process_image_file(input_path, output_path)

    print(f"[DONE] Processed {total_images} images across {total_folders} folders.")


if __name__ == "__main__":
    input_folder = os.getenv("POMELO_IMAGE_ENHANCER_INPUT_FOLDER")
    output_folder = os.getenv("POMELO_IMAGE_ENHANCER_OUTPUT_FOLDER")

    if not input_folder or not output_folder:
        print("Error: Environment variables POMELO_IMAGE_ENHANCER_INPUT_FOLDER and POMELO_IMAGE_ENHANCER_OUTPUT_FOLDER must be set.")
        exit(1)

    limit_per_folder = None
    process_dataset(input_folder, output_folder, limit_per_folder)
