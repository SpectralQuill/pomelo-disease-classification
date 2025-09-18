import os
import numpy as np
from PIL import Image
from scipy import ndimage

class ImageTrimmer:
    def __init__(self, input_folder: str, output_folder: str, alpha_threshold: int = 10, min_area_ratio: float = 0.05):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.alpha_threshold = alpha_threshold  # ignore faint halos
        self.min_area_ratio = min_area_ratio    # minimum cluster area relative to total image area
        os.makedirs(self.output_folder, exist_ok=True)  # create if missing

    def trim_transparent_edges(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        image_np = np.array(image)
        alpha = image_np[:, :, 3]

        # Step 1: threshold alpha
        mask = alpha > self.alpha_threshold

        # Step 2: connected component labeling
        labeled_array, num_features = ndimage.label(mask)

        if num_features == 0:
            return image  # fully transparent

        # Step 3: compute cluster sizes
        cluster_sizes = np.bincount(labeled_array.ravel())

        # Step 4: determine minimum cluster size
        total_area = mask.shape[0] * mask.shape[1]
        min_area = int(total_area * self.min_area_ratio)

        # Step 5: keep only large-enough clusters
        valid_labels = np.where(cluster_sizes >= min_area)[0]
        valid_labels = valid_labels[valid_labels != 0]  # exclude background

        if valid_labels.size == 0:
            return image  # nothing significant remains

        filtered_mask = np.isin(labeled_array, valid_labels)

        # Step 6: bounding box from filtered mask
        coords = np.column_stack(np.where(filtered_mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = image_np[y_min:y_max+1, x_min:x_max+1]

        return Image.fromarray(cropped)

    def process_folder(self):
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith((".png", ".webp")):
                input_path = os.path.join(self.input_folder, filename)
                output_path = os.path.join(self.output_folder, filename)
                try:
                    with Image.open(input_path) as img:
                        trimmed_img = self.trim_transparent_edges(img)
                        trimmed_img.save(output_path)
                        print(f"Trimmed: {filename} → saved to {self.output_folder}")
                except Exception as e:
                    print(f"Skipping {filename}: {e}")


def main():
    input_folder = r"images\to_trim"
    output_folder = r"images\trimmed"

    trimmer = ImageTrimmer(input_folder, output_folder, alpha_threshold=10, min_area_ratio=0.05)
    trimmer.process_folder()


if __name__ == "__main__":
    main()
