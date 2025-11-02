import os
from PIL import Image

class AlphaBackgroundRGBReplacer:
    """
    Replaces RGB values of pixels whose alpha < threshold with a specified background RGB color,
    preserving the alpha channel and folder structure in the output directory.
    """

    SUPPORTED_FORMATS = (".png", ".webp", ".tiff")

    def __init__(self, input_dir: str, output_dir: str, background_rgb: tuple[int, int, int], alpha_threshold: int):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.background_rgb = background_rgb
        self.alpha_threshold = alpha_threshold

    def process(self):
        """Walk through all subfolders and process each image."""
        for root, _, files in os.walk(self.input_dir):
            for filename in files:
                if filename.lower().endswith(self.SUPPORTED_FORMATS):
                    input_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(root, self.input_dir)
                    output_path = os.path.join(self.output_dir, rel_path, filename)

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    self._process_image(input_path, output_path)

    def _process_image(self, input_path: str, output_path: str):
        """Modify RGB values but preserve alpha channel."""
        try:
            img = Image.open(input_path).convert("RGBA")
            pixels = img.load()

            width, height = img.size
            bg_r, bg_g, bg_b = self.background_rgb

            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixels[x, y]
                    if a < self.alpha_threshold:
                        pixels[x, y] = (bg_r, bg_g, bg_b, a)

            img.save(output_path, "PNG")
            print(f"Processed: {input_path} → {output_path}")

        except Exception as e:
            print(f"⚠️ Error processing {input_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Replace RGB values of low-alpha pixels with a background color (preserve alpha)."
    )
    parser.add_argument("input_dir", help="Input folder containing images.")
    parser.add_argument("output_dir", help="Output folder where processed images will be saved.")
    parser.add_argument("--background", nargs=3, type=int, metavar=("R", "G", "B"),
                        required=True, help="Background RGB values (e.g., 255 255 255).")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Alpha threshold (0–255). Pixels below this will have RGB replaced (default: 10).")

    args = parser.parse_args()

    replacer = AlphaBackgroundRGBReplacer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        background_rgb=tuple(args.background),
        alpha_threshold=args.threshold,
    )
    replacer.process()
