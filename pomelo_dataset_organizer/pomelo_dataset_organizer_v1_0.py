import argparse
import os
import shutil
import pandas as pd

class PomeloDatasetOrganizer:
    excel_extensions = ['.xlsx', '.xls']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

    def __init__(self, images_folder=r"images\raw", labeling_file=r"tracker\tracker.csv", 
                 sheet_name="Classes", name_column="Name", class_column=None):
        """
        Initialize the PomeloDatasetOrganizer with given hyperparameters.
        """
        self.images_folder = images_folder
        self.labeling_file = labeling_file
        self.sheet_name = sheet_name
        self.name_column = name_column
        self.class_column = class_column or self._get_default_class_column()
        
        # Initialize other attributes
        self.pomelo_classes = {}
        self.labeling_data = None
        
        # Validate inputs
        self._validate_parameters()
        
    def _get_default_class_column(self):
        """Set default class column based on file type."""
        # Use string methods for file extension detection
        file_lower = self.labeling_file.lower()
        if file_lower.endswith('.csv'):
            return "Class"
        elif any(file_lower.endswith(ext) for ext in self.excel_extensions):
            return "Final"
        return None
    
    def _validate_parameters(self):
        """Validate the input parameters."""
        if not os.path.exists(self.labeling_file):
            raise FileNotFoundError(f"Labeling file not found: {self.labeling_file}")
        
        # Use string methods for file extension detection
        file_lower = self.labeling_file.lower()
        if any(file_lower.endswith(ext) for ext in self.excel_extensions):
            if self.sheet_name is None:
                raise ValueError("sheet_name must be provided for Excel files")
            
            # Check if sheet exists
            try:
                excel_file = pd.ExcelFile(self.labeling_file)
                if self.sheet_name not in excel_file.sheet_names:
                    raise ValueError(f"Sheet '{self.sheet_name}' not found in Excel file")
            except Exception as e:
                raise ValueError(f"Error reading Excel file: {e}")
    
    def load_pomelo_classes(self):
        """
        Load pomelo classes from configs/pomelo_classes.csv
        """
        config_file = r"configs\pomelo_classes.csv"
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Pomelo classes config file not found: {config_file}")
        
        try:
            config_df = pd.read_csv(config_file)
            self.pomelo_classes = dict(zip(
                config_df['Name'].str.strip().str.lower(), 
                config_df['Folder Name'].str.strip()
            ))
            print(f"Loaded {len(self.pomelo_classes)} pomelo classes from config")
        except Exception as e:
            raise ValueError(f"Error loading pomelo classes config: {e}")
    
    def load_labeling_data(self):
        """
        Load labeling data from CSV or Excel file.
        """
        try:
            # Use string methods for file extension detection
            file_lower = self.labeling_file.lower()
            if file_lower.endswith('.csv'):
                self.labeling_data = pd.read_csv(self.labeling_file)
            else:
                self.labeling_data = pd.read_excel(
                    self.labeling_file, 
                    sheet_name=self.sheet_name,
                    engine='openpyxl'
                )
            
            # Convert column names to lowercase for case-insensitive comparison
            self.labeling_data.columns = self.labeling_data.columns.str.strip().str.lower()
            
            # Check if required columns exist
            name_col_lower = self.name_column.strip().lower()
            class_col_lower = self.class_column.strip().lower()
            
            if name_col_lower not in self.labeling_data.columns:
                raise ValueError(f"Name column '{self.name_column}' not found in labeling file")
            
            if class_col_lower not in self.labeling_data.columns:
                raise ValueError(f"Class column '{self.class_column}' not found in labeling file")
                
            print(f"Loaded labeling data with {len(self.labeling_data)} entries")
            
        except Exception as e:
            raise ValueError(f"Error loading labeling data: {e}")
    
    def create_class_folders(self):
        """
        Create directories for each class folder if they don't exist.
        """
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder, exist_ok=True)
            print(f"Created images folder: {self.images_folder}")
        
        for folder_name in self.pomelo_classes.values():
            class_folder = os.path.join(self.images_folder, folder_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder, exist_ok=True)
                print(f"Created class folder: {class_folder}")
    
    def find_image_file(self, image_name):
        """
        Search for image file in images_folder (directly and one subfolder level down).
        Returns the file path if found, None otherwise.
        """
        # Remove file extension from search using string methods
        image_name = os.path.splitext(image_name)[0] if '.' in image_name else image_name
        
        # Search directly in images_folder
        for ext in self.image_extensions:
            image_path = os.path.join(self.images_folder, f"{image_name}{ext}")
            if os.path.exists(image_path):
                return image_path
            
            # Search with different case variations
            image_path_lower = os.path.join(self.images_folder, f"{image_name.lower()}{ext}")
            if os.path.exists(image_path_lower):
                return image_path_lower
            image_path_upper = os.path.join(self.images_folder, f"{image_name.upper()}{ext}")
            if os.path.exists(image_path_upper):
                return image_path_upper
        
        # Search one subfolder level down
        if os.path.exists(self.images_folder):
            for item in os.listdir(self.images_folder):
                subfolder_path = os.path.join(self.images_folder, item)
                if os.path.isdir(subfolder_path):
                    for ext in self.image_extensions:
                        image_path = os.path.join(subfolder_path, f"{image_name}{ext}")
                        if os.path.exists(image_path):
                            return image_path
                        
                        # Search with different case variations
                        image_path_lower = os.path.join(subfolder_path, f"{image_name.lower()}{ext}")
                        if os.path.exists(image_path_lower):
                            return image_path_lower
                        image_path_upper = os.path.join(subfolder_path, f"{image_name.upper()}{ext}")
                        if os.path.exists(image_path_upper):
                            return image_path_upper
        
        return None
    
    def organize_images(self):
        """
        Organize images into their appropriate class folders based on labeling data.
        """
        name_col_lower = self.name_column.strip().lower()
        class_col_lower = self.class_column.strip().lower()
        
        moved_count = 0
        not_found_count = 0
        no_class_count = 0
        
        for _, row in self.labeling_data.iterrows():
            image_name = str(row[name_col_lower]).strip()
            class_name = str(row[class_col_lower]).strip().lower() if pd.notna(row[class_col_lower]) else None
            
            if not class_name or class_name not in self.pomelo_classes:
                no_class_count += 1
                continue
            
            # Find the image file
            image_path = self.find_image_file(image_name)
            if not image_path:
                not_found_count += 1
                continue
            
            # Get target folder and destination path
            target_folder = self.pomelo_classes[class_name]
            destination_folder = os.path.join(self.images_folder, target_folder)
            destination_path = os.path.join(destination_folder, os.path.basename(image_path))
            if image_path.lower() == destination_path.lower():
                continue
            
            # Move the file
            try:
                shutil.move(image_path, destination_path)
                moved_count += 1
                print(f"Moved: {os.path.basename(image_path)} -> {target_folder}/")
            except Exception as e:
                print(f"Error moving {image_path}: {e}")
        
        print(f"\nOrganization completed:")
        print(f"  - Moved: {moved_count} images")
        print(f"  - Not found: {not_found_count} images")
        print(f"  - No valid class: {no_class_count} images")
    
    def run_organization(self):
        """
        Main method to run the entire organization process.
        """
        print("Starting Pomelo Dataset Organization...")
        
        # Load configuration and data
        self.load_pomelo_classes()
        self.load_labeling_data()
        
        # Create folders
        self.create_class_folders()
        
        # Organize images
        self.organize_images()
        
        print("Pomelo Dataset Organization completed!")


def run_pomelo_dataset_organizer(images_folder=r"images\raw", labeling_file=r"tracker\tracker.csv", 
                               sheet_name="Classes", name_column="Name", class_column=None):
    """
    Function for importing this script and running the organizer.
    
    Parameters:
    - images_folder: Directory for pomelo images. Default r"images\raw"
    - labeling_file: CSV or Excel file path. Default r"tracker\tracker.csv"
    - sheet_name: Excel sheet name. Default "Classes"
    - name_column: Header for pomelo image names. Default "Name"
    - class_column: Header for classes. Default based on file type
    """
    organizer = PomeloDatasetOrganizer(
        images_folder=images_folder,
        labeling_file=labeling_file,
        sheet_name=sheet_name,
        name_column=name_column,
        class_column=class_column
    )
    organizer.run_organization()


def main():
    """
    Main function for when running the script as standalone.
    """
    
    parser = argparse.ArgumentParser(description='Organize pomelo images into class folders')
    parser.add_argument('--images_folder', default=r'images\raw', 
                       help='Directory for pomelo images (default: images\\raw)')
    parser.add_argument('--labeling_file', default=r'tracker\tracker.csv',
                       help='CSV or Excel file path (default: tracker\\tracker.csv)')
    parser.add_argument('--sheet_name', default='Classes',
                       help='Excel sheet name (default: Classes)')
    parser.add_argument('--name_column', default='Name',
                       help='Header for pomelo image names (default: Name)')
    parser.add_argument('--class_column', 
                       help='Header for classes (default: Class for CSV, Final for Excel)')
    
    args = parser.parse_args()
    
    try:
        run_pomelo_dataset_organizer(
            images_folder=args.images_folder,
            labeling_file=args.labeling_file,
            sheet_name=args.sheet_name,
            name_column=args.name_column,
            class_column=args.class_column
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
