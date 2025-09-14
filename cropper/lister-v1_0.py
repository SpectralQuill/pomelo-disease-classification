import os
import csv
from pathlib import Path
from typing import List, Set

def get_image_names(dataset_folder: str) -> List[str]:
    """
    Get all image names from the dataset folder and one level of subfolders,
    excluding the 'Processed' subfolder. Returns only filenames without paths or extensions.
    
    Args:
        dataset_folder: Path to the main dataset folder
        
    Returns:
        List of image names (without extensions) alphabetized
    """
    dataset_path = Path(dataset_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_names = []
    
    # Get images from the main folder
    for item in dataset_path.iterdir():
        if item.is_file() and item.suffix.lower() in image_extensions:
            image_names.append(item.stem)  # Get filename without extension
    
    # Get images from one level of subfolders (excluding 'Processed')
    for subfolder in dataset_path.iterdir():
        if (subfolder.is_dir() and 
            subfolder.name != "Processed" and 
            not subfolder.name.startswith('.')):  # Skip hidden folders
            
            for item in subfolder.iterdir():
                if item.is_file() and item.suffix.lower() in image_extensions:
                    image_names.append(item.stem)  # Get filename without extension
    
    return sorted(image_names)  # Alphabetize

def get_existing_images(csv_file: str) -> Set[str]:
    """
    Get set of image names already in the CSV file.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Set of image names from the first column (excluding header)
    """
    existing_images = set()
    
    if not os.path.exists(csv_file):
        return existing_images
    
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header row
            for row in reader:
                if row:  # Check if row is not empty
                    existing_images.add(row[0])  # First column contains image names
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    
    return existing_images

def update_csv_with_new_images(csv_file: str, image_names: List[str], existing_images: Set[str]):
    """
    Update CSV file with new images, maintaining alphabetical order.
    
    Args:
        csv_file: Path to the CSV file
        image_names: List of all image names (alphabetized, without extensions)
        existing_images: Set of images already in CSV
    """
    # Read existing data (excluding header)
    existing_rows = []
    header = ["Image Name"]  # Default header
    
    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                existing_header = next(reader, None)  # Save header
                if existing_header:
                    header = existing_header
                existing_rows = list(reader)
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            return
    
    # Create a dictionary of existing rows for easy lookup (using first column as key)
    existing_data = {}
    for row in existing_rows:
        if row:  # Skip empty rows
            image_name = row[0]
            existing_data[image_name] = row
    
    # Add new images to the dictionary
    for image_name in image_names:
        if image_name not in existing_data:
            # Create new row with image name in first column
            existing_data[image_name] = [image_name]
    
    # Convert to sorted list by image name (first column)
    all_rows = sorted(existing_data.values(), key=lambda x: x[0])
    
    # Write back to CSV
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write header
            writer.writerows(all_rows)  # Write all rows in alphabetical order
        
        print(f"CSV updated successfully. Total rows: {len(all_rows)}")
        print(f"New images added: {len(image_names) - len(existing_data) + len(existing_rows)}")
        
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def main():
    """
    Main function to update the CSV with new images from the dataset folder.
    """
    # Configuration - UPDATE THESE PATHS
    DATASET_FOLDER = "dataset"  # Your main dataset folder
    CSV_FILE = "tracker/tracker.csv"  # Your CSV file path
    
    # Get all image names from dataset (without extensions or paths)
    print("Scanning for images...")
    image_names = get_image_names(DATASET_FOLDER)
    print(f"Found {len(image_names)} images in dataset")
    
    # Get images already in CSV
    existing_images = get_existing_images(CSV_FILE)
    print(f"Found {len(existing_images)} images already in CSV")
    
    # Find new images to add
    new_images = [img for img in image_names if img not in existing_images]
    print(f"Found {len(new_images)} new images to add")
    
    if new_images:
        print("New images to be added:", new_images)
        # Update CSV with new images
        update_csv_with_new_images(CSV_FILE, image_names, existing_images)
    else:
        print("No new images to add. CSV is up to date.")

if __name__ == "__main__":
    main()