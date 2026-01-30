import os
import sys

class SampleSelector:
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}

    def __init__(self, folder: str, step: int):
        self.folder = folder
        self.step = step

    def get_images(self):
        return [
            os.path.join(self.folder, f)
            for f in os.listdir(self.folder)
            if os.path.isfile(os.path.join(self.folder, f)) and os.path.splitext(f)[1].lower() in self.IMAGE_EXTS
        ]

    def log_every_nth(self):
        for i, img_path in enumerate(self.get_images(), start=1):
            if i % self.step == 1:
                print(img_path)


if __name__ == "__main__":
    folder = sys.argv[1]
    n = int(sys.argv[2])

    SampleSelector(folder, n).log_every_nth()
