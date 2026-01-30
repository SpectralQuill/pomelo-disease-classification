import os
import sys
from PIL import Image
import colorsys
import argparse

class HSVAlphaBackgroundRGBReplacer:
    """
    Replaces RGB values of pixels that match optional HSV + Alpha range filters,
    preserving the alpha channel and folder structure in the output directory.
    """

    SUPPORTED_FORMATS = (".png", ".jpg", ".webp", ".tiff")

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        background_rgb: tuple[int, int, int],
        hue_range: tuple[float, float] | None = None,
        sat_range: tuple[float, float] | None = None,
        val_range: tuple[float, float] | None = None,
        alpha_range: tuple[int, int] | None = None,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.background_rgb = background_rgb
        self.hue_range = hue_range
        self.sat_range = sat_range
        self.val_range = val_range
        self.alpha_range = alpha_range

        # Ensure at least one filter exists
        if not any([hue_range, sat_range, val_range, alpha_range]):
            raise ValueError("At least one filter range must be provided (Hue, Saturation, Value, or Alpha).")

    def process(self):
        """Walk through all subfolders and process each supported image."""
        for root, _, files in os.walk(self.input_dir):
            for filename in files:
                if filename.lower().endswith(self.SUPPORTED_FORMATS):
                    input_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(root, self.input_dir)
                    output_path = os.path.join(self.output_dir, rel_path, filename)

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    self._process_image(input_path, output_path)

    def _process_image(self, input_path: str, output_path: str):
        """Apply filters and replace matching pixel RGB while keeping original alpha."""
        try:
            img = Image.open(input_path).convert("RGBA")
            pixels = img.load()
            width, height = img.size
            bg_r, bg_g, bg_b, bg_a = self.background_rgb

            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixels[x, y]

                    # Normalize RGB → HSV
                    rn, gn, bn = r/255, g/255, b/255
                    h, s, v = colorsys.rgb_to_hsv(rn, gn, bn)
                    h_deg = h * 360  # Convert hue to 0-360

                    # Apply optional filters
                    if self.hue_range:
                        h_min, h_max = self.hue_range
                        if h_max < h_min:
                            h_min1, h_max1 = 0, h_max
                            h_min2, h_max2 = h_min, 360
                            if not (h_min1 <= h_deg <= h_max1 or h_min2 <= h_deg <= h_max2):
                                continue
                        elif not (h_min <= h_deg <= h_max):
                                continue

                    if self.sat_range:
                        s_min, s_max = self.sat_range
                        if not (s_min <= s <= s_max):
                            continue

                    if self.val_range:
                        v_min, v_max = self.val_range
                        if not (v_min <= v <= v_max):
                            continue

                    if self.alpha_range:
                        a_min, a_max = self.alpha_range
                        if not (a_min <= a <= a_max):
                            continue

                    # If pixel passed all enabled filters → replace RGB, keep A
                    pixels[x, y] = (bg_r, bg_g, bg_b, bg_a)

            img.save(output_path, "PNG")
            print(f"Processed: {input_path} → {output_path}")

        except Exception as e:
            print(f"⚠️ Error processing {input_path}: {e}")

def parse_optional_range(values: list[float | int]) -> tuple[float, float] | tuple[int, int] | None:
    """
    Converts input list into a (min, max) range.
    - If list has 2 values → use directly
    - If list has 1 value → (0, value)
    - If list is empty/None → return None (disabled filter)
    """
    if not values:
        return None
    if len(values) == 1:
        return (0, values[0])
    if len(values) >= 2:
        return (values[0], values[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace RGB values of pixels matching optional HSV and/or Alpha ranges, preserving alpha and folder structure."
    )

    parser.add_argument("input_dir", help="Input folder containing images.")
    parser.add_argument("output_dir", help="Output folder where processed images will be saved.")
    parser.add_argument("--background", nargs=4, type=int, metavar=("R", "G", "B", "A"),
                        required=True, help="Background RGBA values (e.g., 255 255 255 255).")

    # Optional filters
    parser.add_argument("--hue", nargs="*", type=float, metavar=("H_MIN", "H_MAX"),
                        help="Hue range in degrees (0–360). One value = max, min defaults to 0.")
    parser.add_argument("--sat", nargs="*", type=float, metavar=("S_MIN", "S_MAX"),
                        help="Saturation range (0.0–1.0). One value = max, min defaults to 0.")
    parser.add_argument("--val", nargs="*", type=float, metavar=("V_MIN", "V_MAX"),
                        help="Value/Brightness range (0.0–1.0). One value = max, min defaults to 0.")
    parser.add_argument("--alpha", nargs="*", type=int, metavar=("A_MIN", "A_MAX"),
                        help="Alpha range (0–255). One value = max, min defaults to 0.")

    args = parser.parse_args()

    # Parse ranges properly
    hue_range = parse_optional_range(args.hue)
    sat_range = parse_optional_range(args.sat)
    val_range = parse_optional_range(args.val)
    alpha_range = parse_optional_range(args.alpha)

    try:
        replacer = HSVAlphaBackgroundRGBReplacer(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            background_rgb=tuple(args.background),
            hue_range=hue_range,
            sat_range=sat_range,
            val_range=val_range,
            alpha_range=alpha_range,
        )
        replacer.process()

    except ValueError as ve:
        print(f"❌ {ve}")
        sys.exit(1)
