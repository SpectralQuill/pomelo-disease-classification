#!/usr/bin/env python3
import os
import sys
import csv
import argparse
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class ClassCsvGenerator:
    CONFIG_FILE = r"configs\pomelo_classes.csv"
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

    def __init__(self, input_folder: str, input_csv: str):
        self.input_folder = input_folder
        self.input_csv = input_csv
        self.config_rows: List[Dict[str, str]] = []
        self.csv_fieldnames: List[str] = []
        self.csv_rows: List[Dict[str, str]] = []

    def load_config(self):
        config_path = self.CONFIG_FILE
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found at '{config_path}' (hard-coded).")

        with open(config_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            if "Name" not in headers or "Folder Name" not in headers:
                raise ValueError(
                    f"Config file '{config_path}' must contain headers 'Name' and 'Folder Name'. Headers found: {headers}"
                )
            self.config_rows = [row for row in reader]

    def load_input_csv(self):
        csv_path = self.input_csv
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Input CSV file not found at '{csv_path}'.")

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "Name" not in fieldnames or "Class" not in fieldnames:
                raise ValueError(
                    f"Input CSV '{csv_path}' must contain headers 'Name' and 'Class'. Headers found: {fieldnames}"
                )
            self.csv_fieldnames = fieldnames[:]
            self.csv_rows = [row for row in reader]

    def check_csv_writable(self):
        csv_path = self.input_csv

        if not os.access(csv_path, os.W_OK):
            try:
                with open(csv_path, "r+", newline="", encoding="utf-8"):
                    pass
            except Exception as exc:
                raise PermissionError(f"No write permission for CSV '{csv_path}': {exc}")

    def _find_matching_subfolder(self, folder_name: str) -> str:
        base = self.input_folder
        try:
            entries = os.listdir(base)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input folder '{base}' does not exist.")

        target_lower = folder_name.lower()
        for entry in entries:
            entry_path = os.path.join(base, entry)
            if os.path.isdir(entry_path) and entry.lower() == target_lower:
                return os.path.abspath(entry_path)
        return ""

    def _list_images_in_folder(self, folder_path: str) -> List[str]:
        result: List[str] = []
        try:
            entries = os.listdir(folder_path)
        except Exception:
            return result

        for entry in entries:
            full = os.path.join(folder_path, entry)
            if os.path.isfile(full):
                _, ext = os.path.splitext(entry)
                if ext.lower() in self.IMAGE_EXTENSIONS:
                    result.append(entry)
        return result

    def _name_without_extension(self, filename: str) -> str:
        base, _ = os.path.splitext(filename)
        return base.lower()

    def process_folders_and_update(self):
        name_to_row_indices: Dict[str, List[int]] = {}

        for idx, row in enumerate(self.csv_rows):
            csv_name_raw = row.get("Name", "")
            csv_name_key = self._name_without_extension(csv_name_raw)
            if csv_name_key not in name_to_row_indices:
                name_to_row_indices[csv_name_key] = []
            name_to_row_indices[csv_name_key].append(idx)

        for cfg in self.config_rows:
            pomelo_class_name = cfg.get("Name", "").strip()
            folder_name = cfg.get("Folder Name", "").strip()
            if not folder_name:
                continue

            matched_folder = self._find_matching_subfolder(folder_name)
            if not matched_folder:
                print(f"Warning: folder '{folder_name}' from config not found under input folder '{self.input_folder}'. Skipping.")
                continue

            images = self._list_images_in_folder(matched_folder)
            if not images:
                continue

            for image_filename in images:
                image_key = self._name_without_extension(image_filename)
                if image_key in name_to_row_indices:
                    for row_idx in name_to_row_indices[image_key]:
                        self.csv_rows[row_idx]["Class"] = pomelo_class_name

    def save_csv(self):
        csv_path = self.input_csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
            for row in self.csv_rows:
                safe_row = {k: row.get(k, "") for k in self.csv_fieldnames}
                writer.writerow(safe_row)

    def run(self):
        self.load_config()

        self.load_input_csv()

        self.check_csv_writable()

        self.process_folders_and_update()

        self.save_csv()


def run_class_csv_generator(input_folder: str, input_csv: str):
    folder_to_use = input_folder
    csv_to_use = input_csv

    if (not folder_to_use) and ("CLASS_CSV_GENERATOR_INPUT_FOLDER" in os.environ):
        folder_to_use = os.environ["CLASS_CSV_GENERATOR_INPUT_FOLDER"]
    if (not csv_to_use) and ("CLASS_CSV_GENERATOR_INPUT_CSV" in os.environ):
        csv_to_use = os.environ["CLASS_CSV_GENERATOR_INPUT_CSV"]

    if not folder_to_use or not csv_to_use:
        raise ValueError("Both input_folder and input_csv must be provided, either as arguments or via environment variables.")

    generator = ClassCsvGenerator(folder_to_use, csv_to_use)
    generator.run()


def main(input_folder: str, input_csv: str):
    folder_to_use = input_folder
    csv_to_use = input_csv

    if (not folder_to_use) and ("CLASS_CSV_GENERATOR_INPUT_FOLDER" in os.environ):
        folder_to_use = os.environ["CLASS_CSV_GENERATOR_INPUT_FOLDER"]
    if (not csv_to_use) and ("CLASS_CSV_GENERATOR_INPUT_CSV" in os.environ):
        csv_to_use = os.environ["CLASS_CSV_GENERATOR_INPUT_CSV"]

    if not folder_to_use or not csv_to_use:
        raise SystemExit("Error: input folder and input CSV must be provided either as command-line arguments or environment variables.")

    try:
        run_class_csv_generator(folder_to_use, csv_to_use)
        print("CSV successfully updated.")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate/Update Class column in CSV based on folder structure and config.")
    parser.add_argument("--input-folder", dest="input_folder", help="Path to pomelo images input folder", default=None)
    parser.add_argument("--input-csv", dest="input_csv", help="Path to input CSV file to edit", default=None)
    args = parser.parse_args()

    try:
        main(args.input_folder, args.input_csv)
    except Exception as exc:
        sys.exit(1)
