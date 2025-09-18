import os
import csv
import shutil

class PomeloStatusUpdater:
    def __init__(self, csv_path, class_folders):
        self.csv_path = csv_path
        self.class_folders = class_folders
        self.class_counters = {cls: 0 for cls in class_folders.keys()}
        self.not_found_images = []
        self.rows = []
        self.header = []
        self.csv_lookup = {}

    def make_backup(self):
        base, ext = os.path.splitext(self.csv_path)
        counter = 0
        new_path = f"{base}_backup ({counter}){ext}"
        while os.path.exists(new_path):
            counter += 1
            new_path = f"{base}_backup ({counter}){ext}"
        shutil.copy2(self.csv_path, new_path)
        print(f"Backup created at: {new_path}")
        return new_path

    def load_csv(self):
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.header = reader[0]
            self.rows = reader[1:]  # skip header

        # Make dict for fast lookup (image_name -> row index)
        self.csv_lookup = {row[0]: i for i, row in enumerate(self.rows)}

        # Ensure header has at least 3 columns
        while len(self.header) < 3:
            self.header.append(f"col{len(self.header)+1}")

    def process_class_folders(self):
        for class_name, folder in self.class_folders.items():
            if not os.path.isdir(folder):
                print(f"Warning: folder '{folder}' not found. Skipping...")
                continue

            for file in os.listdir(folder):
                if not os.path.isfile(os.path.join(folder, file)):
                    continue

                image_name, _ = os.path.splitext(file)

                if image_name in self.csv_lookup:
                    row_index = self.csv_lookup[image_name]
                    row = self.rows[row_index]

                    # Ensure row has at least 3 columns
                    while len(row) < 3:
                        row.append("")

                    # Write class name to column 3
                    row[2] = class_name

                    # make coyunter not update when alrewDY IN

                    # Update counter
                    self.class_counters[class_name] += 1
                else:
                    self.not_found_images.append(image_name)

    def save_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(self.rows)

    def print_results(self):
        print("\n=== Rewrite Summary ===")
        for cls, count in self.class_counters.items():
            print(f"{cls}: {count} successful rewrites")

        if self.not_found_images:
            print("\n=== Images not found in CSV ===")
            for img in self.not_found_images:
                print(img)
        else:
            print("\nNo missing images.")

    def run(self):
        self.make_backup()
        self.load_csv()
        self.process_class_folders()
        self.save_csv()
        self.print_results()


def main():
    # ------------------------
    # Arbitrary variables
    # ------------------------
    csv_path = r"your_file.csv"  # <-- Change this
    class_folders = {
        "Processed": r"images\processed",
        "Partial": r"images\partial",
        "Incorrect": r"images\incorrect",
        "Unusable": r"images\unusable",
        # "Unprocessed": r""
    }

    updater = PomeloStatusUpdater(csv_path, class_folders)
    updater.run()


if __name__ == "__main__":
    main()
