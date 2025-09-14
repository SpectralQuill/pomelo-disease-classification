import os
import csv

# Configuration - set these as needed
CSV_FILE_PATH = "tracker/tracker.csv"      # Path to your CSV file
IMAGE_FOLDER_PATH = "test_output"  # Path to your image folder
FILL_CROPPED = True                      # Set to True to fill the third column
FILL_TRANSPARENT = True                  # Set to True to fill the fourth column

def process_csv_with_images():
    """
    Process the CSV file and update boolean columns based on image presence
    """
    # Read all image names from the folder and one subfolder level
    image_names = set()
    
    # Get images from main folder
    for file in os.listdir(IMAGE_FOLDER_PATH):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # Remove extension and add to set
            name_without_ext = os.path.splitext(file)[0]
            image_names.add(name_without_ext)
    
    # Get images from subfolders (one level down)
    for item in os.listdir(IMAGE_FOLDER_PATH):
        item_path = os.path.join(IMAGE_FOLDER_PATH, item)
        if os.path.isdir(item_path):
            for file in os.listdir(item_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    name_without_ext = os.path.splitext(file)[0]
                    image_names.add(name_without_ext)
    
    print(f"Found {len(image_names)} unique image names in the folder structure")
    
    # Read and process the CSV file
    rows = []
    with open(CSV_FILE_PATH, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        # Read header row
        header = next(reader)
        rows.append(header)
        
        # Process each data row
        for row in reader:
            if not row:  # Skip empty rows
                continue
                
            image_name = row[0].strip() if len(row) > 0 else ""
            
            # Ensure row has enough columns
            while len(row) < 4:
                row.append("")
            
            # Update boolean columns if image exists
            if image_name in image_names:
                if FILL_CROPPED and len(row) > 2:
                    row[2] = "TRUE"
                if FILL_TRANSPARENT and len(row) > 3:
                    row[3] = "TRUE"
            
            rows.append(row)
    
    # Write the updated CSV back to file
    with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    
    print(f"CSV file '{CSV_FILE_PATH}' has been updated successfully!")
    print(f"FILL_CROPPED: {FILL_CROPPED}, FILL_TRANSPARENT: {FILL_TRANSPARENT}")

if __name__ == "__main__":
    # Check if files exist
    if not os.path.isfile(CSV_FILE_PATH):
        print(f"Error: CSV file '{CSV_FILE_PATH}' not found!")
    elif not os.path.isdir(IMAGE_FOLDER_PATH):
        print(f"Error: Image folder '{IMAGE_FOLDER_PATH}' not found!")
    else:
        process_csv_with_images()