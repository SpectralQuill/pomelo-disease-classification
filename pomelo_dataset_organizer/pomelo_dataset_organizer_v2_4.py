import argparse
import logging
import os
import pandas as pd
import re
import shutil
from dotenv import load_dotenv
from pathlib import Path
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from typing import Dict, List, Set, Optional, Any

load_dotenv()

class PomeloDatasetOrganizer:
    """
    Organizes pomelo images across Google Drive and local folders based on class assignments.
    Handles class conflicts and moves images to appropriate directories.
    """
    
    class PomeloClass:
        """Represents a pomelo class with its properties and encoding information."""
        
        def __init__(self, name: str, priority_weight: int):
            self.name = name.lower()
            self.priority_weight = priority_weight
        
        def __str__(self):
            return f"PomeloClass(name='{self.name}', weight={self.priority_weight})"
    
    class PomeloImage:
        """Represents a pomelo image with its file information and manipulation methods."""
        
        def __init__(self, name: str, organizer: 'PomeloDatasetOrganizer'):
            self.name = name
            self.organizer = organizer
            self.local_files = []
            self.drive_files = []
            self.is_moved_local = False
            self.is_moved_drive = False
        
        def add_local_classes(self, classes_set: Set[str], file_path: Path):
            """Add a local classes set to this image."""
            self.local_files.append({
                'path': file_path, 
                'classes_set': classes_set, 
                'is_main': False
            })
        
        def add_drive_classes(self, classes_set: Set[str], file_item: Any, parent_id: str):
            """Add a drive classes set to this image."""
            self.drive_files.append({
                'file_item': file_item, 
                'classes_set': classes_set,
                'parent_id': parent_id,
                'is_main': False
            })
        
        def get_resolved_classes(self) -> Set[str]:
            """Get resolved classes set by merging drive and local classes and resolving conflicts."""
            # Start with empty set
            merged_classes = set()
            
            # Merge all classes from drive and local
            for file_info in self.local_files + self.drive_files:
                classes_set = file_info['classes_set']
                merged_classes.update(classes_set)
            
            # Handle root folder case ("__root__" in classes)
            if "__root__" in merged_classes:
                merged_classes.discard("__root__")
                # If no other classes are present, assign to first pomelo class
                if not any(cls in self.organizer.pomelo_classes for cls in merged_classes):
                    if self.organizer.pomelo_classes:
                        first_class = list(self.organizer.pomelo_classes.keys())[0]
                        merged_classes.add(first_class)
            
            # Apply overriding CSV if available
            if self.organizer.overriding_csv_data is not None:
                if self.name in self.organizer.overriding_csv_data:
                    override_classes = self.organizer.overriding_csv_data[self.name]
                    # Only keep classes that are registered in pomelo classes
                    valid_override_classes = {cls for cls in override_classes if cls in self.organizer.pomelo_classes}
                    if valid_override_classes:
                        merged_classes = valid_override_classes
            
            # Filter to keep only highest priority classes
            if len(merged_classes) > 1:
                max_priority = -1
                best_classes = set()
                
                for class_name in merged_classes:
                    if class_name in self.organizer.pomelo_classes:
                        pomelo_class = self.organizer.pomelo_classes[class_name]
                        if pomelo_class.priority_weight > max_priority:
                            max_priority = pomelo_class.priority_weight
                            best_classes = {class_name}
                        elif pomelo_class.priority_weight == max_priority:
                            best_classes.add(class_name)
                
                merged_classes = best_classes
            
            return merged_classes
        
        def mark_main_files(self):
            """Mark main files based on resolved classes set match."""
            resolved_classes = self.get_resolved_classes()
            
            # Mark main files for local files
            main_local_found = False
            for file_info in self.local_files:
                if file_info['classes_set'] == resolved_classes:
                    file_info['is_main'] = True
                    main_local_found = True
                    break
            
            # If no exact match found, mark first local file as main
            if not main_local_found and self.local_files:
                self.local_files[0]['is_main'] = True
            
            # Mark main files for drive files
            main_drive_found = False
            for file_info in self.drive_files:
                if file_info['classes_set'] == resolved_classes:
                    file_info['is_main'] = True
                    main_drive_found = True
                    break
            
            # If no exact match found, mark first drive file as main
            if not main_drive_found and self.drive_files:
                self.drive_files[0]['is_main'] = True
        
        def get_positive_classes(self) -> List[str]:
            """Get the list of positive classes from resolved classes set."""
            resolved_classes = self.get_resolved_classes()
            return [cls for cls in resolved_classes if cls != "__root__"]
        
        def get_original_drive_subfolder(self):
            """Get original drive subfolder using first element of drive_files."""
            if not self.drive_files:
                return None
            
            # Use the classes set from the first drive file
            first_classes_set = self.drive_files[0]['classes_set']
            classes_key = self.organizer._classes_set_to_key(first_classes_set)
            return self.organizer.pomelo_class_subfolders.get(classes_key)
        
        def get_original_local_subfolder(self):
            """Get original local subfolder using first element of local_files."""
            if not self.local_files:
                return None
            
            # Use the classes set from the first local file
            first_classes_set = self.local_files[0]['classes_set']
            classes_key = self.organizer._classes_set_to_key(first_classes_set)
            return self.organizer.pomelo_class_subfolders.get(classes_key)
        
        def get_new_subfolder(self):
            """Get new subfolder using resolved classes set."""
            resolved_classes = self.get_resolved_classes()
            classes_key = self.organizer._classes_set_to_key(resolved_classes)
            
            if classes_key not in self.organizer.pomelo_class_subfolders:
                # Create new subfolder
                self.organizer.pomelo_class_subfolders[classes_key] = (
                    self.organizer.PomeloClassSubfolder(resolved_classes, self.organizer, is_new=True)
                )
            
            return self.organizer.pomelo_class_subfolders[classes_key]
        
        def move_to_new_subfolder(self) -> bool:
            """Move image to new subfolder based on resolved classes set."""
            
            new_subfolder = self.get_new_subfolder()
            original_drive_subfolder = self.get_original_drive_subfolder()
            original_local_subfolder = self.get_original_local_subfolder()
            
            success_local = True
            success_drive = True
            
            # Move main local file
            main_local_files = [f for f in self.local_files if f['is_main']]
            if main_local_files and (not original_local_subfolder or original_local_subfolder != new_subfolder):
                success_local = self._move_local_to_subfolder(new_subfolder, main_local_files[0])
                if success_local and original_local_subfolder:
                    original_local_subfolder.local_dir_count -= len(main_local_files)
                    self.is_moved_local = True
                new_subfolder.local_dir_count += len(main_local_files) if success_local else 0
            
            # Move main drive file
            main_drive_files = [f for f in self.drive_files if f['is_main']]
            if main_drive_files and (not original_drive_subfolder or original_drive_subfolder != new_subfolder):
                success_drive = self._move_drive_to_subfolder(new_subfolder, main_drive_files[0])
                if success_drive and original_drive_subfolder:
                    original_drive_subfolder.drive_dir_count -= len(main_drive_files)
                    self.is_moved_drive = True
                new_subfolder.drive_dir_count += len(main_drive_files) if success_drive else 0
            
            return success_local and success_drive
        
        def _move_local_to_subfolder(self, target_subfolder, file_info: dict) -> bool:
            """Move local file to target subfolder."""
            target_folder_path = self.organizer._get_or_create_local_folder(target_subfolder.name)
            
            source_path = file_info['path']
            target_path = target_folder_path / source_path.name
            
            try:
                if source_path != target_path:
                    shutil.move(str(source_path), str(target_path))
                    file_info['path'] = target_path
                    logging.info(f"Moved local image {self.name} to {target_subfolder.name}")
                    return True
            except Exception as e:
                logging.error(f"Failed to move local image {self.name}: {str(e)}")
                return False
            
            return True
        
        def _move_drive_to_subfolder(self, target_subfolder, file_info: dict) -> bool:
            """Move drive file to target subfolder."""
            target_folder_id = self.organizer._get_or_create_drive_folder(target_subfolder.name)
            if not target_folder_id:
                return False
            
            file_item = file_info['file_item']
            current_parent_id = file_info['parent_id']
            
            if current_parent_id != target_folder_id:
                try:
                    file_item['parents'] = [{'id': target_folder_id}]
                    file_item.Upload()
                    file_info['parent_id'] = target_folder_id
                    logging.info(f"Moved drive image {self.name} to {target_subfolder.name}")
                    return True
                except Exception as e:
                    logging.error(f"Failed to move drive image {self.name}: {str(e)}")
                    return False
            
            return True

        def _get_unique_duplicate_name(self, file_info: dict, base_index: int, duplicates_folder: Path, is_local: bool) -> str:
            """Generate unique duplicate image name with conflict resolution."""
            # Extract name and extension based on file type
            if 'path' in file_info:  # Local file
                source_path = file_info['path']
                base_name = source_path.stem
                extension = source_path.suffix
            else:  # Drive file
                file_item = file_info['file_item']
                file_title = file_item['title']
                base_name = Path(file_title).stem
                extension = Path(file_title).suffix
            
            # Remove existing duplicate pattern if present
            base_name = re.sub(r'\s*\(\d+\)$', '', base_name)
            
            # Try different indices until we find a unique name
            index = base_index
            while True:
                if index == 1:
                    candidate_name = f"{base_name}{extension}"
                else:
                    candidate_name = f"{base_name} ({index}){extension}"
                
                if is_local:
                    # Check if file exists in local duplicates folder
                    if not (duplicates_folder / candidate_name).exists():
                        return candidate_name
                else:
                    # For drive, we'll check during upload by trying to create the file
                    return candidate_name
                
                index += 1

        def move_duplicates(self):
            """Move duplicate images to duplicates folder."""
            # Move local duplicates
            duplicate_local_files = [f for f in self.local_files if not f['is_main']]
            if duplicate_local_files:
                self._move_local_duplicates(duplicate_local_files)
            
            # Move drive duplicates
            duplicate_drive_files = [f for f in self.drive_files if not f['is_main']]
            if duplicate_drive_files:
                self._move_drive_duplicates(duplicate_drive_files)

        def _move_local_duplicates(self, duplicate_files: List[dict]):
            """Move duplicate local images to duplicates folder."""
            duplicates_folder = self.organizer.local_images_folder / "duplicates"
            duplicates_folder.mkdir(parents=True, exist_ok=True)
            
            # Move duplicates to duplicates folder
            for i, file_info in enumerate(duplicate_files, start=1):
                source_path = file_info['path']
                new_filename = self._get_unique_duplicate_name(file_info, i, duplicates_folder, is_local=True)
                target_path = duplicates_folder / new_filename
                
                try:
                    shutil.move(str(source_path), str(target_path))
                    file_info['path'] = target_path
                    logging.info(f"Moved duplicate local image: {self.name} -> duplicates/{new_filename}")
                except Exception as e:
                    logging.error(f"Failed to move duplicate local image {self.name}: {str(e)}")

        def _move_drive_duplicates(self, duplicate_files: List[dict]):
            """Move duplicate drive images to duplicates folder."""
            duplicates_folder_id = self.organizer._get_or_create_drive_folder("duplicates")
            if not duplicates_folder_id:
                logging.error("Failed to create duplicates folder in Google Drive")
                return
            
            # Move duplicates to duplicates folder
            for i, file_info in enumerate(duplicate_files, start=1):
                file_item = file_info['file_item']
                new_filename = self._get_unique_duplicate_name(file_info, i, None, is_local=False)
                
                try:
                    file_item['title'] = new_filename
                    file_item['parents'] = [{'id': duplicates_folder_id}]
                    file_item.Upload()
                    logging.info(f"Moved duplicate drive image: {self.name} -> duplicates/{new_filename}")
                except Exception as e:
                    logging.error(f"Failed to move duplicate drive image {self.name}: {str(e)}")

        def __str__(self):
            return f"PomeloImage(name='{self.name}', local_files={len(self.local_files)}, drive_files={len(self.drive_files)})"
        
    class PomeloClassSubfolder:
        """Represents a pomelo class subfolder with its classes set and file counts."""
        
        def __init__(self, classes_set: Set[str], organizer: 'PomeloDatasetOrganizer', drive_id: Optional[str] = None, is_new: bool = False):
            self.classes_set = classes_set
            self.drive_id = drive_id
            self.local_dir_count = 0
            self.drive_dir_count = 0
            self.is_new = is_new
            self._name = None
            self.organizer = organizer
        
        @property
        def name(self):
            """Get the subfolder name based on classes set."""
            if self._name is None:
                self._name = self.load_classes_subfolder_name()
            return self._name
        
        def load_classes_subfolder_name(self) -> str:
            """Generate subfolder name based on classes set."""
            # Handle root folder
            if "__root__" in self.classes_set:
                return "__root__"
            
            # Handle duplicates folder
            if "duplicates" in self.classes_set:
                return "duplicates"
            
            # If only one class, return the class name
            if len(self.classes_set) == 1:
                return next(iter(self.classes_set))
            
            # If multiple classes, create conflicted folder name
            sorted_classes = sorted(self.classes_set)
            return f"conflicted_{'_'.join(sorted_classes)}"
        
        def delete(self, location: str) -> bool:
            """Delete subfolder if empty in specified location."""
            if location == "local" and self.local_dir_count == 0:
                logging.info(f"Deleting empty local subfolder: {self.name}")
                return True
            elif location == "drive" and self.drive_dir_count == 0 and self.drive_id:
                logging.info(f"Deleting empty drive subfolder: {self.name}")
                return True
            return False
        
        def __str__(self):
            return f"PomeloClassSubfolder(name='{self.name}', classes_set={self.classes_set}, local_count={self.local_dir_count}, drive_count={self.drive_dir_count})"
    
    def __init__(
        self,
        google_drive_id: str = os.environ['DATASET_GOOGLE_DRIVE_ID'],
        local_images_folder: str = r"images\processed",
        labeling_csv: str = r"tracker\tracker.csv",
        overriding_csv: str = r"tracker\override.csv",
        ignore_subfolders: Set[str] = None
    ):
        """
        Initialize the PomeloDatasetOrganizer with folder paths and configuration.
        """
        if ignore_subfolders is None:
            ignore_subfolders = {"protected"}
        
        self._setup_logging()
        logging.info("Initializing PomeloDatasetOrganizer...")
        
        self.google_drive_images_folder = google_drive_id
        self.local_images_folder = Path(local_images_folder)
        self.labeling_csv = Path(labeling_csv)
        self.overriding_csv = Path(overriding_csv) if overriding_csv else None
        self.ignore_subfolders = {folder.lower() for folder in ignore_subfolders}
        
        self._check_csv_permissions()
        self.drive_folder_id = google_drive_id
        self.drive_service = self._initialize_google_drive()
        
        logging.info(f"Using Google Drive folder: {google_drive_id}")
        logging.info(f"Using local images folder: {local_images_folder}")
        logging.info(f"Using labeling CSV: {labeling_csv}")
        if self.overriding_csv:
            logging.info(f"Using overriding CSV: {self.overriding_csv}")
        
        self.pomelo_classes = self._load_pomelo_classes()
        class_names = list(self.pomelo_classes.keys())
        logging.info(f"Pomelo classes found: {', '.join(class_names)}")
        
        self.pomelo_class_subfolders = {}
        self.pomelo_images = {}
        self.overriding_csv_data = self._load_overriding_csv() if self.overriding_csv else None
        
        self._load_pomelo_class_subfolders()
        self._load_images_from_subfolders()
    
    def _load_pomelo_classes(self) -> Dict[str, 'PomeloDatasetOrganizer.PomeloClass']:
        """Load pomelo classes as PomeloClass objects."""
        classes_config_path = Path(r"configs\pomelo_classes.csv")
        
        if not classes_config_path.exists():
            raise FileNotFoundError(f"Pomelo classes config not found: {classes_config_path}")
        
        df = pd.read_csv(classes_config_path)
        classes_dict = {}
        
        for _, row in df.iterrows():
            class_name = str(row['Name']).strip().lower()
            priority_weight = int(row['Priority Weight'])
            
            classes_dict[class_name] = self.PomeloClass(
                name=class_name,
                priority_weight=priority_weight
            )
        
        return dict(sorted(classes_dict.items()))
    
    def _load_overriding_csv(self) -> Optional[Dict[str, Set[str]]]:
        """Load overriding CSV data."""
        if not self.overriding_csv or not self.overriding_csv.exists():
            return None
        
        try:
            df = pd.read_csv(self.overriding_csv)
            overriding_data = {}
            
            for _, row in df.iterrows():
                image_name = str(row['Name'])
                classes_str = str(row['Class']) if 'Class' in df.columns else ""
                
                # Parse classes (assuming comma-separated)
                classes_set = {cls.strip().lower() for cls in classes_str.split(',') if cls.strip()}
                if classes_set:
                    overriding_data[image_name] = classes_set
            
            logging.info(f"Loaded overriding data for {len(overriding_data)} images")
            return overriding_data
            
        except Exception as e:
            logging.error(f"Error loading overriding CSV: {str(e)}")
            return None
    
    def _load_pomelo_class_subfolders(self):
        """Load pomelo class subfolders from both drive and local."""
        logging.info("Loading pomelo class subfolders...")
        
        # Add root folder
        root_classes = {"__root__"}
        root_key = self._classes_set_to_key(root_classes)
        self.pomelo_class_subfolders[root_key] = self.PomeloClassSubfolder(
            root_classes, self, self.drive_folder_id
        )
        
        # Add duplicates folder
        duplicates_classes = {"duplicates"}
        duplicates_key = self._classes_set_to_key(duplicates_classes)
        self.pomelo_class_subfolders[duplicates_key] = self.PomeloClassSubfolder(
            duplicates_classes, self, is_new=False
        )
        
        # Load drive subfolders
        drive_subfolders = self._get_subfolders_drive()
        
        # Load local subfolders
        local_subfolders = self._get_subfolders_local()
        
        # Create subfolder instances for all found subfolders
        all_subfolders = set(drive_subfolders.keys()).union(set(local_subfolders.keys()))
        
        for folder_name in all_subfolders:
            if folder_name.lower() in self.ignore_subfolders:
                continue
                
            # Determine classes set for this subfolder
            classes_set = self._subfolder_name_to_classes_set(folder_name)
            if not classes_set:  # Skip if empty
                continue
                
            classes_key = self._classes_set_to_key(classes_set)
            
            if classes_key not in self.pomelo_class_subfolders:
                drive_id = drive_subfolders.get(folder_name)
                self.pomelo_class_subfolders[classes_key] = self.PomeloClassSubfolder(
                    classes_set, self, drive_id
                )
    
    def _load_images_from_subfolders(self):
        """Load images from all pomelo class folders."""
        logging.info("Loading images from pomelo class folders...")
        
        total_subfolders = len(self.pomelo_class_subfolders)
        logging.info(f"Reading {total_subfolders} folders...")
        
        for i, (classes_key, subfolder) in enumerate(self.pomelo_class_subfolders.items(), 1):
            logging.info(f"Reading folder {i}/{total_subfolders}: {subfolder.name}")
            
            local_folder_path = self.local_images_folder / subfolder.name if subfolder.name != "__root__" else self.local_images_folder
            if local_folder_path.exists() and local_folder_path.is_dir():
                self._load_images_from_local_subfolder(subfolder, local_folder_path)
            if subfolder.drive_id:
                self._load_images_from_drive_subfolder(subfolder, subfolder.drive_id)
        
        logging.info(f"Found {len(self.pomelo_images)} unique images")
    
    def _load_images_from_local_subfolder(self, subfolder: PomeloClassSubfolder, folder_path: Path):
        """Load images from a local subfolder and add to pomelo_images."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Use regex to match image name with duplicates pattern
                image_name = self._extract_image_name_from_filename(file_path.name)
                
                if image_name not in self.pomelo_images:
                    self.pomelo_images[image_name] = self.PomeloImage(image_name, self)
                
                self.pomelo_images[image_name].add_local_classes(subfolder.classes_set, file_path)
                subfolder.local_dir_count += 1
    
    def _load_images_from_drive_subfolder(self, subfolder: PomeloClassSubfolder, folder_id: str):
        """Load images from a drive subfolder and add to pomelo_images."""
        try:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
            
            file_list = self.drive_service.ListFile({
                'q': f"'{folder_id}' in parents and trashed=false"
            }).GetList()
            
            for file_item in file_list:
                if file_item['mimeType'] != 'application/vnd.google-apps.folder':
                    file_extension = Path(file_item['title']).suffix.lower()
                    if file_extension in image_extensions:
                        # Use regex to match image name with duplicates pattern
                        image_name = self._extract_image_name_from_filename(file_item['title'])
                        
                        if image_name not in self.pomelo_images:
                            self.pomelo_images[image_name] = self.PomeloImage(image_name, self)
                        
                        self.pomelo_images[image_name].add_drive_classes(subfolder.classes_set, file_item, folder_id)
                        subfolder.drive_dir_count += 1
                        
        except Exception as e:
            logging.error(f"Error loading images from drive subfolder {subfolder.name}: {str(e)}")
    
    def _extract_image_name_from_filename(self, filename: str) -> str:
        """Extract base image name from filename using regex pattern."""
        # Pattern to match base name before any duplicate indicators
        pattern = r'^(.+?)(?:\s*\(\d+\))?\.[^.]+$'
        match = re.match(pattern, filename)
        if match:
            return match.group(1)
        return Path(filename).stem  # Fallback to stem if pattern doesn't match
    
    def _subfolder_name_to_classes_set(self, folder_name: str) -> Set[str]:
        """Convert subfolder name to classes set."""
        # Handle root folder
        if folder_name == "__root__":
            return {"__root__"}
        
        # Handle duplicates folder
        if folder_name == "duplicates":
            return {"duplicates"}
        
        # Handle conflicted folders
        if folder_name.startswith("conflicted_"):
            classes_str = folder_name[11:]  # Remove "conflicted_" prefix
            class_names = classes_str.split("_")
            return {name for name in class_names if name in self.pomelo_classes}
        
        # Handle regular class folders
        if folder_name in self.pomelo_classes:
            return {folder_name}
        
        return set()
    
    def _classes_set_to_key(self, classes_set: Set[str]) -> str:
        """Convert classes set to string key."""
        return "_".join(sorted(classes_set))
    
    def _get_subfolders_drive(self) -> Dict[str, str]:
        """Get direct subfolder names and IDs from Google Drive folder."""
        folder_dict = {}
        try:
            folder_list = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            for folder in folder_list:
                folder_name = folder['title'].lower()
                if folder_name not in self.ignore_subfolders:
                    folder_dict[folder_name] = folder['id']
        except Exception as e:
            logging.error(f"Error getting Drive subfolders: {str(e)}")
        return folder_dict
    
    def _get_subfolders_local(self) -> Dict[str, Path]:
        """Get direct subfolder names and paths from local folder."""
        folder_dict = {}
        if self.local_images_folder.exists():
            for item in self.local_images_folder.iterdir():
                if item.is_dir() and item.name.lower() not in self.ignore_subfolders:
                    folder_dict[item.name.lower()] = item
        return folder_dict
    
    def _download_drive_file_to_local(self, file_item: Any, target_path: Path) -> bool:
        """Download a drive file to local path."""
        try:
            # Ensure parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file
            file_item.GetContentFile(str(target_path))
            logging.info(f"Downloaded drive file to local: {target_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to download drive file {file_item['title']} to {target_path}: {str(e)}")
            return False
    
    def _get_or_create_drive_folder(self, folder_name: str) -> Optional[str]:
        """Get existing folder ID or create if it doesn't exist."""
        try:
            existing_folders = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and title='{folder_name}' and trashed=false"
            }).GetList()
            
            if existing_folders:
                return existing_folders[0]['id']
            
            # Create new folder
            folder_metadata = {
                'title': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [{'id': self.drive_folder_id}]
            }
            
            folder = self.drive_service.CreateFile(folder_metadata)
            folder.Upload()
            logging.info(f"Created Drive folder: {folder_name}")
            return folder['id']
            
        except Exception as e:
            logging.error(f"Failed to create Drive folder {folder_name}: {str(e)}")
            return None
    
    def _get_or_create_local_folder(self, folder_name: str) -> Path:
        """Get existing local folder path or create if it doesn't exist."""
        folder_path = self.local_images_folder / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def _check_csv_permissions(self):
        """Check if labeling CSV has write permissions."""
        if not self.labeling_csv.exists():
            raise FileNotFoundError(f"Labeling CSV not found: {self.labeling_csv}")
        
        try:
            with open(self.labeling_csv, 'a', newline='', encoding='utf-8') as f:
                pass
        except PermissionError:
            raise PermissionError(
                f"No write permission for labeling CSV: {self.labeling_csv}. "
                "Please ensure the file is not open in another program and you have write access."
            )
        except Exception as e:
            raise Exception(f"Unable to access labeling CSV: {str(e)}")
    
    def _initialize_google_drive(self):
        """Initialize and authenticate Google Drive service."""
        try:
            gauth = GoogleAuth()
            
            credentials_file = Path('credentials.json')
            if credentials_file.exists():
                gauth.LoadCredentialsFile(str(credentials_file))
            
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
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    def _get_drive_subfolder_name(self, folder_id: str) -> str:
        """Get drive subfolder name by ID."""
        try:
            folder = self.drive_service.CreateFile({'id': folder_id})
            folder.FetchMetadata()
            return folder['title']
        except Exception:
            return "__root__"

    def _delete_empty_subfolders(self):
        """Delete empty subfolders in both local and Google Drive."""
        logging.info("Deleting empty subfolders...")
        
        local_deleted = 0
        drive_deleted = 0
        
        for classes_key, subfolder in list(self.pomelo_class_subfolders.items()):
            # Skip root folder, duplicates folder, and newly created folders
            if "_root_" in subfolder.classes_set or "duplicates" in subfolder.classes_set or subfolder.is_new:
                continue
                
            # Delete local subfolder if empty (this should work for all accounts)
            if subfolder.local_dir_count == 0:
                local_folder_path = self.local_images_folder / subfolder.name
                if local_folder_path.exists() and local_folder_path.is_dir():
                    try:
                        # Check if folder is actually empty before deleting
                        if not any(local_folder_path.iterdir()):
                            local_folder_path.rmdir()
                            local_deleted += 1
                            logging.info(f"Deleted empty local subfolder: {subfolder.name}")
                        else:
                            logging.warning(f"Local folder {subfolder.name} is not empty, skipping deletion")
                    except Exception as e:
                        logging.warning(f"Could not delete local folder {subfolder.name}: {e}")
            
            # Only attempt Drive deletion if we're confident we have owner permissions
            if subfolder.drive_id and subfolder.drive_dir_count == 0:
                try:
                    # Check if we have delete permissions by testing metadata access first
                    drive_folder = self.drive_service.CreateFile({'id': subfolder.drive_id})
                    drive_folder.FetchMetadata()
                    
                    # Only attempt deletion if we're the owner or have explicit delete rights
                    if ('owners' in drive_folder and 
                        any(owner.get('permissionId') == 'me' for owner in drive_folder['owners'])):
                        drive_folder.Delete()
                        drive_deleted += 1
                        logging.info(f"Deleted empty drive subfolder: {subfolder.name}")
                    else:
                        logging.info(f"Skipping drive folder deletion (no owner permissions): {subfolder.name}")
                        
                except Exception as e:
                    logging.warning(f"Could not delete drive folder {subfolder.name}: {e}")
        
        logging.info(f"Deleted {local_deleted} empty local subfolders and {drive_deleted} empty drive subfolders")

    def organize_dataset(self):
        """Main method to organize the entire dataset."""
        logging.info("Starting pomelo dataset organization...")
        
        try:
            moved_local_count = 0
            moved_drive_count = 0
            conflicted_count = 0
            duplicates_local_count = 0
            duplicates_drive_count = 0
            downloaded_count = 0
            
            # Load image names from labeling CSV in the specified order
            df = pd.read_csv(self.labeling_csv)
            image_names = df['Name'].tolist()
            total_images = len(image_names)
            
            logging.info(f"Processing {total_images} images from labeling CSV...")
            
            for index, image_name in enumerate(image_names, 1):
                image_name_str = str(image_name)
                logging.info(f"Processing image {index}/{total_images}: {image_name_str}")
                
                # Check if image exists in our loaded images
                if image_name_str not in self.pomelo_images:
                    logging.warning(f"Image {image_name_str} not found in loaded images, skipping...")
                    continue
                    
                image = self.pomelo_images[image_name_str]

                # Download missing local files from Drive
                if not image.local_files and image.drive_files:
                    logging.info(f"Downloading missing local file for {image_name_str} from Drive...")
                    # Use the main drive file if available, otherwise first one
                    main_drive_files = [f for f in image.drive_files if f['is_main']]
                    drive_file_info = main_drive_files[0] if main_drive_files else image.drive_files[0]
                    
                    # Determine target local path based on drive folder structure
                    drive_subfolder = self._get_drive_subfolder_name(drive_file_info['parent_id'])
                    local_target_folder = self.local_images_folder / drive_subfolder
                    local_target_path = local_target_folder / drive_file_info['file_item']['title']
                    
                    if self._download_drive_file_to_local(drive_file_info['file_item'], local_target_path):
                        # Add the downloaded file to local files
                        image.add_local_classes(drive_file_info['classes_set'], local_target_path)
                        downloaded_count += 1

                # Mark main files
                image.mark_main_files()
                
                # Move to new subfolder
                if image.move_to_new_subfolder():
                    if image.is_moved_local:
                        moved_local_count += 1
                    if image.is_moved_drive:
                        moved_drive_count += 1
                
                # Move duplicates
                image.move_duplicates()
                duplicate_local_files = [f for f in image.local_files if not f['is_main']]
                duplicate_drive_files = [f for f in image.drive_files if not f['is_main']]
                duplicates_local_count += len(duplicate_local_files)
                duplicates_drive_count += len(duplicate_drive_files)
                
                # Check for conflicts
                resolved_classes = image.get_resolved_classes()
                if len(resolved_classes) > 1:
                    conflicted_count += 1
            
            # Delete empty subfolders
            self._delete_empty_subfolders()
            
            logging.info(f"Organization completed!")
            logging.info(f"Images downloaded from Drive: {downloaded_count}")
            logging.info(f"Images moved locally: {moved_local_count}")
            logging.info(f"Images moved on Drive: {moved_drive_count}")
            logging.info(f"Images with conflicting classes: {conflicted_count}")
            logging.info(f"Duplicates in local folder: {duplicates_local_count}")
            logging.info(f"Duplicates in Drive folder: {duplicates_drive_count}")
            
        except Exception as e:
            logging.error(f"Error during dataset organization: {str(e)}")
            raise

def run_pomelo_dataset_organizer(
    google_drive_id: str = None,
    local_images_folder: str = None,
    labeling_csv: str = None,
    overriding_csv: str = None,
    ignore_subfolders: Set[str] = None
):
    """
    Run the PomeloDatasetOrganizer with optional custom parameters.
    
    Args:
        google_drive_folder: Optional custom Google Drive folder URL
        local_images_folder: Optional custom local images folder path
        labeling_csv: Optional custom labeling CSV file path
        overriding_csv: Optional custom overriding CSV file path
        ignore_subfolders: Optional set of subfolder names to ignore
    """
    kwargs = {}
    if google_drive_id:
        kwargs['google_drive_id'] = google_drive_id
    if local_images_folder:
        kwargs['local_images_folder'] = local_images_folder
    if labeling_csv:
        kwargs['labeling_csv'] = labeling_csv
    if overriding_csv:
        kwargs['overriding_csv'] = overriding_csv
    if ignore_subfolders:
        kwargs['ignore_subfolders'] = ignore_subfolders
    
    organizer = PomeloDatasetOrganizer(**kwargs)
    organizer.organize_dataset()

def main():
    parser = argparse.ArgumentParser(
        description="Organize pomelo images across Google Drive and local folders based on class assignments."
    )
    
    parser.add_argument(
        '--google-drive-id',
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
    
    parser.add_argument(
        '--overriding-csv',
        type=str,
        default=r"tracker\override.csv",
        help='Path to CSV file with overriding class assignments'
    )
    
    parser.add_argument(
        '--ignore-subfolders',
        type=str,
        nargs='*',
        default=["protected"],
        help='Subfolder names to ignore across all operations'
    )
    
    args = parser.parse_args()
    
    run_pomelo_dataset_organizer(
        google_drive_id=args.google_drive_id,
        local_images_folder=args.local_images_folder,
        labeling_csv=args.labeling_csv,
        overriding_csv=args.overriding_csv,
        ignore_subfolders=set(args.ignore_subfolders)
    )

if __name__ == "__main__":
    main()
