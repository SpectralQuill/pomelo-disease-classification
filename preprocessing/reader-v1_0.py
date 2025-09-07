import os
import csv
from pathlib import Path
from typing import Dict, List

def get_image_class_mappings(folder_path: str) -> Dict[str, str]:
    """
    Scan subfolders and map image names to their class names.
    
    Args:
        folder_path: Path to the main folder with class subfolders
        
    Returns:
        Dictionary mapping image names (without extension) to class names
    """
    main_path = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_class_map = {}
    
    # Loop through all subfolders (one level down)
    for subfolder in main_path.iterdir():
        if subfolder.is_dir() and not subfolder.name.startswith('.'):  # Skip hidden folders
            class_name = subfolder.name
            
            # Process all images in this subfolder
            for image_file in subfolder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    image_name = image_file.stem  # Get filename without extension
                    image_class_map[image_name] = class_name
    
    return image_class_map

def update_csv_with_classes(csv_file: str, image_class_map: Dict[str, str]):
    """
    Update CSV file with class names in the second column.
    
    Args:
        csv_file: Path to the CSV file
        image_class_map: Dictionary mapping image names to class names
    """
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} does not exist!")
        return
    
    updated_count = 0
    new_rows = []
    
    try:
        # Read existing CSV
        with open(csv_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read header row
            
            # Ensure we have at least 2 columns
            if len(header) < 2:
                header = ["Image Name", "Class"]  # Update header if needed
            else:
                header[1] = "Class"  # Ensure second column is named "Class"
            
            new_rows.append(header)
            
            # Process each row
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                
                image_name = row[0]
                
                # Create a new row with proper number of columns
                new_row = [image_name]  # First column: image name
                
                # Add class name if available
                if image_name in image_class_map:
                    new_row.append(image_class_map[image_name])
                    updated_count += 1
                elif len(row) > 1:
                    # Keep existing class if present
                    new_row.append(row[1])
                else:
                    # Add empty class if none exists
                    new_row.append("")
                
                # Copy any additional columns from original row
                if len(row) > 1:
                    new_row.extend(row[2:])
                
                new_rows.append(new_row)
        
        # Write updated CSV back
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(new_rows)
        
        print(f"CSV updated successfully! Updated {updated_count} rows with class information.")
        print(f"Total images processed: {len(image_class_map)}")
        
    except Exception as e:
        print(f"Error updating CSV: {e}")

def main():
    """
    Main function to update CSV with class information from subfolders.
    """
    # Configuration - UPDATE THESE PATHS
    FOLDER_PATH = "dataset/Processed"  # Folder with class subfolders
    CSV_FILE = "preprocessing/tracker.csv"  # Your CSV file path
    
    print("Scanning subfolders for class information...")
    
    # Get image to class mappings
    image_class_map = get_image_class_mappings(FOLDER_PATH)
    
    if not image_class_map:
        print("No images found in subfolders. Please check your folder path.")
        return
    
    print(f"Found {len(image_class_map)} images in {len(set(image_class_map.values()))} classes")
    print("Classes found:", sorted(set(image_class_map.values())))
    
    # Update CSV with class information
    update_csv_with_classes(CSV_FILE, image_class_map)

if __name__ == "__main__":
    main()