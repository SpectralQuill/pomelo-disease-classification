import os
import numpy as np
from PIL import Image

class TransparentImageTrimmer:
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)  # create if missing

    def trim_transparent_edges(self, image: Image.Image) -> Image.Image:
        """Trim extra transparent pixels around the edges of a PIL Image using NumPy."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        arr = np.array(image)
        alpha_channel = arr[:, :, 3]

        non_empty_rows = np.where(alpha_channel.max(axis=1) > 0)[0]
        non_empty_cols = np.where(alpha_channel.max(axis=0) > 0)[0]

        if non_empty_rows.size and non_empty_cols.size:
            row_min, row_max = non_empty_rows[0], non_empty_rows[-1]
            col_min, col_max = non_empty_cols[0], non_empty_cols[-1]
            trimmed_arr = arr[row_min:row_max + 1, col_min:col_max + 1, :]
            return Image.fromarray(trimmed_arr, "RGBA")
        else:
            # Fully transparent → return original
            return image

    def process_folder(self):
        """Process all images in the input folder and save trimmed versions to output folder."""
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
    # Put your paths here
    input_folder = r"C:\path\to\your\images"
    output_folder = r"C:\path\to\your\trimmed"

    trimmer = TransparentImageTrimmer(input_folder, output_folder)
    trimmer.process_folder()


if __name__ == "__main__":
    main()
