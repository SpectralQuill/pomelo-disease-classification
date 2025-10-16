#!/usr/bin/env python3
import os
import csv
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv


class PomeloPreclassifier:
    def __init__(self, verbose=False, dry_run=False):
        load_dotenv()

        self.verbose = verbose
        self.dry_run = dry_run

        self.input_folder = Path(
            os.getenv("PRECLASSIFIER_INPUT_FOLDER", r"input")
        ).resolve()
        self.output_folder = Path(
            os.getenv("PRECLASSIFIER_OUTPUT_FOLDER", r"output")
        ).resolve()
        self.config_file = Path(r"configs\pomelo_classes.csv").resolve()

        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_folder}")
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        self.classes = self._load_config()

    def _load_config(self):
        mapping = {}
        with open(self.config_file, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            headers = {h.lower(): h for h in reader.fieldnames}
            for row in reader:
                name = row.get(headers.get("name"), "").strip()
                folder = row.get(headers.get("folder name"), "").strip().lower()
                abbr = row.get(headers.get("abbreviation"), "").strip()
                if name:
                    mapping[name.lower()] = {
                        "name": name,
                        "folder": folder,
                        "abbrev": abbr,
                    }
        if self.verbose:
            print(f"✅ Loaded {len(mapping)} classes from {self.config_file}")
            for k, v in self.classes.items():
                print(f"  {v['name']} -> {v['abbrev']} ({v['folder']})")
        return mapping

    def classify_image(self, image_path):
        image_path = Path(image_path).resolve()
        image_name = image_path.stem

        # Run backend prediction via npm
        try:
            cmd = f'npm run backend:predict "{image_path}"'
            if self.verbose:
                print(f"🔍 Running: {cmd}")

            result = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True
            )

            # Expect backend to output valid JSON at end of stdout
            output = result.stdout.strip().splitlines()
            json_str = output[-1] if output else "{}"
            response = json.loads(json_str)

        except subprocess.CalledProcessError as e:
            print(f"❌ Error predicting {image_path.name}: {e}")
            return None
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse backend response for {image_path.name}")
            return None

        predictions = response.get("all_predictions", {})
        if not predictions:
            print(f"⚠️ No predictions found for {image_path.name}")
            return None

        # Sort by confidence descending
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        # Prepare abbreviations & percentages
        log_parts = []
        for cls, conf in sorted_preds:
            abbrev = self.classes.get(cls.lower(), {}).get("abbrev", cls[:2].upper())
            log_parts.append(f"{abbrev}: {conf*100:.3f}%")

        log_line = f"{image_name} - {', '.join(log_parts)}"
        print(log_line)

        # Return top class key
        return sorted_preds[0][0]

    def copy_image(self, image_path, assigned_class):
        image_path = Path(image_path).resolve()
        class_key = assigned_class.strip().lower()
        class_info = self.classes.get(class_key)

        if class_info:
            subfolder_name = class_info["folder"].lower()
        else:
            subfolder_name = class_key

        dest_dir = self.output_folder / subfolder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / image_path.name
        if self.dry_run:
            print(f"🧾 [Dry Run] Would copy {image_path} -> {dest_path}")
            return

        shutil.copy2(image_path, dest_path)
        if self.verbose:
            print(f"📁 Copied to {dest_path}")

    def process_images(self):
        print(f"📂 Scanning input folder: {self.input_folder}")
        for root, dirs, files in os.walk(self.input_folder):
            # Only go 1 subfolder deep
            rel_depth = len(Path(root).relative_to(self.input_folder).parts)
            if rel_depth > 1:
                continue

            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = Path(root) / file
                    try:
                        assigned_class = self.classify_image(image_path)
                        if assigned_class:
                            self.copy_image(image_path, assigned_class)
                    except Exception as e:
                        print(f"❌ Error processing {file}: {e}")
        print("✅ Preclassification complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Pomelo Preclassifier — preclassify unlabeled pomelos using backend model"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable detailed logs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate classification and copying without making changes",
    )

    args = parser.parse_args()
    preclassifier = PomeloPreclassifier(verbose=args.verbose, dry_run=args.dry_run)
    preclassifier.process_images()

if __name__ == "__main__":
    main()
