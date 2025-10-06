import argparse
import os
from dotenv import load_dotenv
from image_status_updater.image_status_updater_v2_2 import main as update_image_statuses
from image_trimmer.image_trimmer_v1_0 import main as trim_images
from pomelo_dataset_organizer.pomelo_dataset_organizer_v2_3 import run_pomelo_dataset_organizer
from pomelo_extractor.pomelo_extractor_v2_15 import run_pomelo_extractor

load_dotenv()

def load_pomelo_dataset_organizer():
    google_drive_folder = os.environ['DATASET_GOOGLE_DRIVE_ID']
    local_images_folder = r"images\processed"
    labeling_csv = r"tracker\tracker.csv"

    run_pomelo_dataset_organizer(google_drive_folder, local_images_folder, labeling_csv)

def load_pomelo_extractor():
    input_folder = r"images\raw"
    output_folder = r"images\extracted"
    max_images = 60
    csv_path = r"tracker\tracker.csv"
    ignore_subfolders = []
    run_pomelo_extractor(input_folder, output_folder, max_images, csv_path,
                         ignore_subfolders)

def load_image_class_updater():
    print("The script is a Jupyter notebook file and cannot be run in this console. Please run the latest version in image_status_updater folder.")

def load_image_status_updater():
    update_image_statuses()
    print('Done')

def load_trimmer():
    trim_images()
    print('Done')

choices = {
    "Load pomelo extractor": load_pomelo_extractor,
    "Load image statuses updater": load_image_status_updater,
    "Load trimmer": load_trimmer,
    "Load image class updater": load_image_class_updater,
    "Load pomelo dataset organizer": load_pomelo_dataset_organizer
}

def main():
    parser = argparse.ArgumentParser(
        description="Enter number of script to run or leave blank to choose from menu"
    )
    parser.add_argument(
        "--choice", type=int, default=None, help="Number of script to run"
    )
    args = parser.parse_args()
    choice_keys = list(choices.keys())
    choice_range = range(len(choice_keys))
    print()
    for index, choice in enumerate(choices):
        print(f"({index+1}) {choice}")
    print()
    if args.choice is None:
        choice_index = int(input("Enter number: ")) - 1
    else:
        print(f"(Default) Running choice #{args.choice}")
        choice_index = args.choice - 1
    print()
    if choice_index not in choice_range:
        print("\033[31mInvalid choice\033[0m")
    else:
        choices[choice_keys[choice_index]]()

if __name__ == "__main__":
    main()
