import argparse
import os
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

load_dotenv()

class PomeloDatasetOrganizer:
    """
    Organizes pomelo images across Google Drive and local folders based on class assignments.
    Handles class conflicts and moves images to appropriate directories.
    """
    
    def __init__(
        self,
        google_drive_images_folder: str = os.environ['DATASET_GOOGLE_DRIVE_ID'],
        local_images_folder: str = r"images\processed",
        labeling_csv: str = r"tracker\tracker.csv"
    ):
        """
        Initialize the PomeloDatasetOrganizer with folder paths and configuration.
        
        Args:
            google_drive_images_folder: URL to Google Drive folder containing pomelo class subfolders
            local_images_folder: Local folder path containing pomelo class subfolders  
            labeling_csv: Path to CSV file tracking image classifications
        """
        self.google_drive_images_folder = google_drive_images_folder
        self.local_images_folder = Path(local_images_folder)
        self.labeling_csv = Path(labeling_csv)
        
        # Check labeling CSV permissions before proceeding
        self._check_csv_permissions()
        
        # Extract Google Drive folder ID from URL
        self.drive_folder_id = self._extract_drive_folder_id(google_drive_images_folder)
        
        # Load pomelo classes configuration
        self.pomelo_classes = self._load_pomelo_classes()
        
        # Initialize Google Drive service
        self.drive_service = self._initialize_google_drive()
        
        # Setup logging
        self._setup_logging()
        
        # Track statistics
        self.changed_classes_count = 0
        self.conflicted_images_count = 0
    
    def _check_csv_permissions(self):
        """Check if labeling CSV has write permissions."""
        if not self.labeling_csv.exists():
            raise FileNotFoundError(f"Labeling CSV not found: {self.labeling_csv}")
        
        try:
            # Test write permissions by attempting to open in append mode
            with open(self.labeling_csv, 'a', newline='', encoding='utf-8') as f:
                pass
        except PermissionError:
            raise PermissionError(
                f"No write permission for labeling CSV: {self.labeling_csv}. "
                "Please ensure the file is not open in another program and you have write access."
            )
        except Exception as e:
            raise Exception(f"Unable to access labeling CSV: {str(e)}")
    
    def _extract_drive_folder_id(self, drive_url: str) -> str:
        """Extract folder ID from Google Drive URL."""
        match = re.search(r'/folders/([a-zA-Z0-9_-]+)', drive_url)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract folder ID from URL: {drive_url}")
    
    def _load_pomelo_classes(self) -> Dict[str, int]:
        """Load pomelo classes and their priority weights from configuration file."""
        classes_config_path = Path(r"configs\pomelo_classes.csv")
        
        if not classes_config_path.exists():
            raise FileNotFoundError(f"Pomelo classes config not found: {classes_config_path}")
        
        df = pd.read_csv(classes_config_path)
        classes_dict = {}
        
        for _, row in df.iterrows():
            class_name = str(row['Name']).strip().lower()
            priority_weight = int(row['Priority Weight'])
            classes_dict[class_name] = priority_weight
        
        # Return classes sorted alphabetically
        return dict(sorted(classes_dict.items()))
    
    def _initialize_google_drive(self):
        """Initialize and authenticate Google Drive service."""
        try:
            gauth = GoogleAuth()
            
            # Try to load existing credentials
            credentials_file = Path('credentials.json')
            if credentials_file.exists():
                gauth.LoadCredentialsFile(str(credentials_file))
            
            # Authenticate if needed
            if gauth.credentials is None or gauth.access_token_expired:
                gauth.LocalWebserverAuth()
                gauth.SaveCredentialsFile(str(credentials_file))
            
            return GoogleDrive(gauth)
        except Exception as e:
            logging.error(f"Failed to initialize Google Drive: {str(e)}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _case_insensitive_equals(self, str1: str, str2: str) -> bool:
        """Check if two strings are equal case-insensitively."""
        return str1.lower() == str2.lower()
    
    def _get_image_files_local(self, folder_path: Path) -> Set[str]:
        """Get set of image filenames (without extensions) from local folder (direct and one subfolder level)."""
        image_files = set()
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Check direct files
        if folder_path.exists():
            for file_path in folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    # Get filename without extension
                    filename_without_ext = file_path.stem
                    image_files.add(filename_without_ext.lower())
        
        # Check one subfolder level down
        if folder_path.exists():
            for subfolder in folder_path.iterdir():
                if subfolder.is_dir():
                    for file_path in subfolder.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                            # Get filename without extension
                            filename_without_ext = file_path.stem
                            image_files.add(filename_without_ext.lower())
        
        return image_files

    def _get_image_files_drive(self, folder_id: str) -> Set[str]:
        """Get set of image filenames (without extensions) from Google Drive folder (direct and one subfolder level)."""
        image_files = set()
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        try:
            # Get ALL files in the main folder (including subfolders)
            query = f"'{folder_id}' in parents and trashed=false"
            file_list = self.drive_service.ListFile({'q': query}).GetList()
            
            for file_item in file_list:
                # Check if it's a file (not a folder)
                if file_item['mimeType'] != 'application/vnd.google-apps.folder':
                    file_title = file_item['title']
                    file_extension = Path(file_title).suffix.lower()
                    if file_extension in image_extensions:
                        # Get filename without extension
                        filename_without_ext = Path(file_title).stem
                        image_files.add(filename_without_ext.lower())
                else:
                    # It's a folder - search inside it
                    subfolder_files = self.drive_service.ListFile({
                        'q': f"'{file_item['id']}' in parents and trashed=false"
                    }).GetList()
                    
                    for sub_file in subfolder_files:
                        if sub_file['mimeType'] != 'application/vnd.google-apps.folder':
                            file_title = sub_file['title']
                            file_extension = Path(file_title).suffix.lower()
                            if file_extension in image_extensions:
                                filename_without_ext = Path(file_title).stem
                                image_files.add(filename_without_ext.lower())
                            
        except Exception as e:
            logging.error(f"Error accessing Google Drive folder {folder_id}: {str(e)}")
        
        return image_files

    def _get_class_subfolders_local(self) -> Dict[str, Path]:
        """Get local subfolders that match pomelo classes."""
        class_folders = {}
        
        if not self.local_images_folder.exists():
            return class_folders
        
        for item in self.local_images_folder.iterdir():
            if item.is_dir():
                folder_name = item.name.lower()
                if folder_name in self.pomelo_classes:
                    class_folders[folder_name] = item
        return class_folders
    
    def _get_class_subfolders_drive(self) -> Dict[str, str]:
        """Get Google Drive subfolders that match pomelo classes."""
        class_folders = {}
        
        try:
            folder_list = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            for folder in folder_list:
                folder_name = folder['title'].lower()
                if folder_name in self.pomelo_classes:
                    class_folders[folder_name] = folder['id']
                    
        except Exception as e:
            logging.error(f"Error getting Google Drive subfolders: {str(e)}")
        
        return class_folders
    
    def _create_multi_hot_encoding(self, image_name: str) -> Tuple[bool]:
        """
        Create multi-hot encoding for an image across all pomelo classes.
        
        Returns:
            Tuple of booleans indicating presence in each class folder
        """
        encoding = [False] * len(self.pomelo_classes)
        image_name_lower = image_name.lower()
        
        # Check local folders
        local_class_folders = self._get_class_subfolders_local()
        for class_name, folder_path in local_class_folders.items():
            class_images = self._get_image_files_local(folder_path)
            if image_name_lower in class_images:
                class_index = list(self.pomelo_classes.keys()).index(class_name)
                encoding[class_index] = True
        
        # Check Google Drive folders
        drive_class_folders = self._get_class_subfolders_drive()
        for class_name, folder_id in drive_class_folders.items():
            class_images = self._get_image_files_drive(folder_id)
            if image_name_lower in class_images:
                class_index = list(self.pomelo_classes.keys()).index(class_name)
                encoding[class_index] = True
        
        return tuple(encoding)
    
    def _resolve_class_conflicts(self, encoding: Tuple[bool]) -> Tuple[bool]:
        """
        Resolve class conflicts by removing less dominant classes.
        
        Args:
            encoding: Original multi-hot encoding
            
        Returns:
            Resolved multi-hot encoding with conflicts removed
        """
        if sum(encoding) <= 1:
            return encoding
        
        # Get indices of positive classes
        positive_indices = [i for i, present in enumerate(encoding) if present]
        class_names = list(self.pomelo_classes.keys())
        
        # Find class with highest priority weight
        max_priority = -1
        best_class_index = -1
        
        for idx in positive_indices:
            class_name = class_names[idx]
            priority = self.pomelo_classes[class_name]
            if priority > max_priority:
                max_priority = priority
                best_class_index = idx
        
        # If tie, keep all positive classes (handled later)
        tied_classes = [idx for idx in positive_indices 
                       if self.pomelo_classes[class_names[idx]] == max_priority]
        
        if len(tied_classes) > 1:
            # Tie - return original encoding with all tied classes
            new_encoding = [False] * len(encoding)
            for idx in tied_classes:
                new_encoding[idx] = True
            return tuple(new_encoding)
        else:
            # Clear winner - keep only the dominant class
            new_encoding = [False] * len(encoding)
            new_encoding[best_class_index] = True
            return tuple(new_encoding)
    
    def _move_image_local(self, image_name: str, target_folder: Path):
        """Move image to target local folder by searching entire local structure."""
        # Find the image anywhere in the local folder structure
        source_path = None
        
        # Search in main folder and all subfolders
        for root, dirs, files in os.walk(self.local_images_folder):
            for file in files:
                file_path = Path(root) / file
                if (file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'} and
                    file_path.stem.lower() == image_name.lower()):
                    source_path = file_path
                    break
            if source_path:
                break
        
        if not source_path:
            logging.warning(f"Image {image_name} not found in local folder structure")
            return False
        
        # Ensure target folder exists
        target_folder.mkdir(parents=True, exist_ok=True)
        target_path = target_folder / (image_name + source_path.suffix)
        
        # Move file if not already in target location
        if source_path != target_path:
            try:
                source_path.rename(target_path)
                logging.info(f"Moved local image: {image_name} -> {target_folder.name}")
                return True
            except Exception as e:
                logging.error(f"Failed to move local image {image_name}: {str(e)}")
                return False
        
        return True

    def _move_image_drive(self, image_name: str, target_folder_id: str):
        """Move image to target Google Drive folder by searching entire drive structure."""
        try:
            # Search for the file across the entire drive folder
            file_item = None
            current_parent_id = None
            
            # Search in the main drive folder and all subfolders
            search_query = f"'{self.drive_folder_id}' in parents and trashed=false"
            all_items = self.drive_service.ListFile({'q': search_query}).GetList()
            
            for item in all_items:
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    # Search in subfolder
                    sub_query = f"'{item['id']}' in parents and trashed=false"
                    sub_items = self.drive_service.ListFile({'q': sub_query}).GetList()
                    for sub_item in sub_items:
                        if (sub_item['mimeType'] != 'application/vnd.google-apps.folder' and 
                            Path(sub_item['title']).stem.lower() == image_name.lower()):
                            file_item = sub_item
                            current_parent_id = item['id']  # Store the subfolder as current parent
                            break
                else:
                    # Check file in main folder
                    if (Path(item['title']).stem.lower() == image_name.lower()):
                        file_item = item
                        current_parent_id = self.drive_folder_id  # Store main folder as current parent
                        break
                
                if file_item:
                    break
            
            if not file_item:
                logging.warning(f"Image {image_name} not found in Google Drive folder structure")
                return False
            
            # Check if file is already in the target folder
            if current_parent_id == target_folder_id:
                return True
            
            # Get current parents to remove them
            current_parents = file_item.get('parents', [])
            if not current_parents:
                logging.warning(f"Image {image_name} has no parent folders")
                return False
            
            # Remove from current parent and add to target folder
            file_item['parents'] = [{'id': target_folder_id}]
            file_item.Upload()
            
            logging.info(f"Moved Drive image: {image_name} from {current_parent_id} -> {target_folder_id}")
            return True
                
        except Exception as e:
            logging.error(f"Failed to move Drive image {image_name}: {str(e)}")
            return False

    def _create_conflicted_folder_name(self, positive_classes: List[str]) -> str:
        """Create folder name for conflicted images."""
        sorted_classes = sorted(positive_classes)
        class_names = "_".join(sorted_classes)
        return f"conflicted_{class_names}"
    
    def _get_or_create_conflicted_folder_local(self, folder_name: str) -> Path:
        """Get existing conflicted folder or create if doesn't exist."""
        conflicted_folder = self.local_images_folder / folder_name
        conflicted_folder.mkdir(parents=True, exist_ok=True)
        return conflicted_folder
    
    def _get_or_create_conflicted_folder_drive(self, folder_name: str) -> Optional[str]:
        """Get existing conflicted folder ID or create if doesn't exist."""
        try:
            # Check if folder already exists
            existing_folders = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and title='{folder_name}' and trashed=false"
            }).GetList()
            
            if existing_folders:
                return existing_folders[0]['id']
            
            # Create new folder if it doesn't exist
            folder_metadata = {
                'title': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [{'id': self.drive_folder_id}]
            }
            
            folder = self.drive_service.CreateFile(folder_metadata)
            folder.Upload()
            logging.info(f"Created Drive conflicted folder: {folder_name}")
            return folder['id']
            
        except Exception as e:
            logging.error(f"Failed to create Drive conflicted folder {folder_name}: {str(e)}")
            return None
    
    def _update_labeling_csv(self, image_name: str, new_class: str):
        """Update the class assignment in the labeling CSV file."""
        try:
            df = pd.read_csv(self.labeling_csv)
            
            # Find row with matching image name (case-insensitive)
            mask = df['Name'].str.lower() == image_name.lower()
            if not mask.any():
                logging.warning(f"Image {image_name} not found in labeling CSV")
                return
            
            # Capitalize first letter if not "Conflicted"
            if new_class.lower() != "conflicted":
                new_class = new_class.capitalize()
            
            # Update class
            old_class = df.loc[mask, 'Class'].iloc[0]
            df.loc[mask, 'Class'] = new_class
            
            # Save updated CSV
            df.to_csv(self.labeling_csv, index=False)
            
            if old_class != new_class:
                logging.info(f"Updated {image_name}: {old_class} -> {new_class}")
                if new_class.lower() != "conflicted":
                    self.changed_classes_count += 1
            
        except Exception as e:
            logging.error(f"Failed to update labeling CSV for {image_name}: {str(e)}")
    
    def process_image(self, image_name: str):
        """Process a single image and organize it across folders."""
        logging.info(f"Processing image: {image_name}")
        
        # Get multi-hot encoding
        encoding = self._create_multi_hot_encoding(image_name)
        resolved_encoding = self._resolve_class_conflicts(encoding)
        
        class_names = list(self.pomelo_classes.keys())
        positive_classes = [class_names[i] for i, present in enumerate(resolved_encoding) if present]
        
        if len(positive_classes) == 1:
            # Single class assignment
            assigned_class = positive_classes[0]
            self._handle_single_class(image_name, assigned_class)
            
        elif len(positive_classes) > 1:
            # Multiple classes - conflict
            self._handle_multiple_classes(image_name, positive_classes)
            self.conflicted_images_count += 1
    
    def _handle_single_class(self, image_name: str, assigned_class: str):
        """Handle image with single class assignment."""
        # Get target folders
        local_target = self.local_images_folder / assigned_class.lower()
        drive_target_id = self._get_class_subfolders_drive().get(assigned_class.lower())
        
        # Move images - search entire structure, not just class folders
        if self.local_images_folder.exists():
            self._move_image_local(image_name, local_target)
        
        if drive_target_id:
            self._move_image_drive(image_name, drive_target_id)
        
        # Update labeling CSV
        self._update_labeling_csv(image_name, assigned_class)

    def _handle_multiple_classes(self, image_name: str, positive_classes: List[str]):
        """Handle image with multiple class assignments (conflict)."""
        conflicted_folder_name = self._create_conflicted_folder_name(positive_classes)
        
        # Create and move to conflicted folders
        if self.local_images_folder.exists():
            local_conflicted = self._get_or_create_conflicted_folder_local(conflicted_folder_name)
            self._move_image_local(image_name, local_conflicted)
        
        drive_conflicted_id = self._get_or_create_conflicted_folder_drive(conflicted_folder_name)
        if drive_conflicted_id:
            self._move_image_drive(image_name, drive_conflicted_id)
        
        # Update labeling CSV
        self._update_labeling_csv(image_name, "Conflicted")

    def delete_empty_subfolders(self):
        """
        Delete empty subfolders in both local and Google Drive folders after organization.
        Preserves main class folders and conflicted folders that might be needed.
        """
        logging.info("Cleaning up empty subfolders...")
        
        local_empty_count = self._delete_empty_local_subfolders()
        drive_empty_count = self._delete_empty_drive_subfolders()
        
        logging.info(f"Cleanup completed: {local_empty_count} local empty folders deleted, "
                    f"{drive_empty_count} Drive empty folders deleted")
    
    def _delete_empty_local_subfolders(self) -> int:
        """Delete empty subfolders in local images directory."""
        empty_count = 0

        if not self.local_images_folder.exists():
            print("1 - Local images folder does not exist.")
            return empty_count
        
        try:
            # Use os.walk with topdown=False to process deepest folders first
            for root, dirs, files in os.walk(self.local_images_folder, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    
                    # Skip if it's the main images folder itself
                    if dir_path == self.local_images_folder:
                        print("2 - Skipping main images folder.")
                        continue

                    if self._is_folder_empty(dir_path) and not self._is_protected_folder(dir_path):
                        try:
                            dir_path.rmdir()
                            logging.info(f"Deleted empty local folder: {dir_path.relative_to(self.local_images_folder)}")
                            empty_count += 1
                        except OSError as e:
                            # Folder might not be empty due to recent file operations
                            logging.debug(f"Could not delete {dir_path}: {e}")
            
            return empty_count
            
        except Exception as e:
            logging.error(f"Error deleting empty local folders: {e}")
            return empty_count

    def _delete_empty_drive_subfolders(self) -> int:
        """Delete empty subfolders in Google Drive directory (recursively)."""
        empty_count = 0
        
        try:
            # Recursively get all subfolders and process them
            folders_to_process = self._get_all_drive_subfolders_recursive(self.drive_folder_id)
            
            # Sort by path depth (deepest first) to handle nested empty folders
            folders_to_process.sort(key=lambda x: len(x['path']), reverse=True)
            
            for folder_info in folders_to_process:
                folder = folder_info['folder']
                if self._is_drive_folder_empty(folder['id']) and not self._is_protected_drive_folder(folder):
                    try:
                        folder.Delete()
                        logging.info(f"Deleted empty Drive folder: {folder_info['path']}")
                        empty_count += 1
                    except Exception as e:
                        logging.debug(f"Could not delete Drive folder {folder_info['path']}: {e}")
            
            return empty_count
            
        except Exception as e:
            logging.error(f"Error deleting empty Drive folders: {e}")
            return empty_count

    def _get_all_drive_subfolders_recursive(self, parent_folder_id: str, current_path: str = "") -> List[Dict]:
        """
        Recursively get all subfolders from a parent folder.
        
        Returns:
            List of dicts with 'folder' object and 'path' string
        """
        all_folders = []
        
        try:
            # Get direct subfolders
            folder_list = self.drive_service.ListFile({
                'q': f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            for folder in folder_list:
                folder_path = f"{current_path}/{folder['title']}" if current_path else folder['title']
                all_folders.append({
                    'folder': folder,
                    'path': folder_path
                })
                
                # Recursively get subfolders
                subfolders = self._get_all_drive_subfolders_recursive(folder['id'], folder_path)
                all_folders.extend(subfolders)
                
        except Exception as e:
            logging.error(f"Error getting Drive subfolders recursively: {e}")
        
        return all_folders

    def _is_folder_empty(self, folder_path: Path) -> bool:
        """Check if a local folder is completely empty (no files, no subfolders)."""
        if not folder_path.exists() or not folder_path.is_dir():
            return False
        
        try:
            # Check if folder has any items at all
            with os.scandir(folder_path) as it:
                return not any(True for _ in it)
        except Exception as e:
            logging.debug(f"Error checking if folder is empty {folder_path}: {e}")
            return False

    def _is_drive_folder_empty(self, folder_id: str) -> bool:
        """Check if a Google Drive folder is completely empty."""
        try:
            # Check for any items (files or folders)
            item_list = self.drive_service.ListFile({
                'q': f"'{folder_id}' in parents and trashed=false"
            }).GetList()
            
            return len(item_list) == 0
            
        except Exception as e:
            logging.debug(f"Error checking if Drive folder is empty {folder_id}: {e}")
            return False

    def _is_protected_folder(self, folder_path: Path) -> bool:
        """
        Check if a folder should be protected from deletion.
        Protects main class folders and conflicted folders.
        """
        if not folder_path.is_relative_to(self.local_images_folder):
            return True
        
        folder_name = folder_path.name.lower()
        
        # Protect direct subfolders of main images folder (class folders and conflicted folders)
        if folder_path.parent == self.local_images_folder and folder_name in self.pomelo_classes:
            return True  # Main class folder
        
        # Protect the main images folder itself
        if folder_path == self.local_images_folder:
            return True
        
        return False

    def _is_protected_drive_folder(self, folder) -> bool:
        """
        Check if a Google Drive folder should be protected from deletion.
        """
        folder_name = folder['title'].lower()
        parent_id = folder['parents'][0]['id'] if folder.get('parents') else None
        
        # Protect direct subfolders of main drive folder
        if parent_id == self.drive_folder_id and folder_name in self.pomelo_classes:
            return True  # Main class folder
        
        return False

    def organize_dataset(self):
        """Main method to organize the entire dataset."""
        logging.info("Starting pomelo dataset organization...")
        
        try:
            # Read labeling CSV
            df = pd.read_csv(self.labeling_csv)
            image_names = df['Name'].tolist()
            
            # Process each image
            for image_name in image_names:
                self.process_image(str(image_name))
            
            # Clean up empty folders after organization
            self.delete_empty_subfolders()
            
            # Log statistics
            logging.info(f"Organization completed!")
            logging.info(f"Images with changed classes: {self.changed_classes_count}")
            logging.info(f"Images with conflicting classes: {self.conflicted_images_count}")
            
        except Exception as e:
            logging.error(f"Error during dataset organization: {str(e)}")
            raise

def run_pomelo_dataset_organizer(
    google_drive_folder: str = None,
    local_images_folder: str = None,
    labeling_csv: str = None
):
    """
    Run the PomeloDatasetOrganizer with optional custom parameters.
    
    Args:
        google_drive_folder: Optional custom Google Drive folder URL
        local_images_folder: Optional custom local images folder path
        labeling_csv: Optional custom labeling CSV file path
    """
    kwargs = {}
    if google_drive_folder:
        kwargs['google_drive_images_folder'] = google_drive_folder
    if local_images_folder:
        kwargs['local_images_folder'] = local_images_folder
    if labeling_csv:
        kwargs['labeling_csv'] = labeling_csv
    
    organizer = PomeloDatasetOrganizer(**kwargs)
    organizer.organize_dataset()

def main():
    parser = argparse.ArgumentParser(
        description="Organize pomelo images across Google Drive and local folders based on class assignments."
    )
    
    parser.add_argument(
        '--google-drive-images-folder',
        type=str,
        default=os.environ['DATASET_GOOGLE_DRIVE_ID'],
        help='Google Drive folder URL containing pomelo class subfolders'
    )
    
    parser.add_argument(
        '--local-images-folder', 
        type=str,
        default=r"images\processed",
        help='Local folder path containing pomelo class subfolders'
    )
    
    parser.add_argument(
        '--labeling-csv',
        type=str, 
        default=r"tracker\tracker.csv",
        help='Path to CSV file tracking image classifications'
    )
    
    args = parser.parse_args()
    
    run_pomelo_dataset_organizer(
        google_drive_folder=args.google_drive_images_folder,
        local_images_folder=args.local_images_folder,
        labeling_csv=args.labeling_csv
    )

if __name__ == "__main__":
    main()
