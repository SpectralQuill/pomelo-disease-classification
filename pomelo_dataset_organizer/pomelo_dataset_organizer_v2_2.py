import argparse
import re
import logging
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any
from pathlib import Path
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import shutil

class PomeloDatasetOrganizer:
    """
    Organizes pomelo images across Google Drive and local folders based on class assignments.
    Handles class conflicts and moves images to appropriate directories.
    """
    
    class PomeloClass:
        """Represents a pomelo class with its properties and encoding information."""
        
        def __init__(self, name: str, priority_weight: int, encoding: Tuple[bool]):
            self.name = name.lower()
            self.priority_weight = priority_weight
            self.encoding = encoding
        
        def __str__(self):
            return f"PomeloClass(name='{self.name}', weight={self.priority_weight}, encoding={self.encoding})"
    
    class PomeloImage:
        """Represents a pomelo image with its file information and manipulation methods."""
        
        def __init__(self, name: str, organizer: 'PomeloDatasetOrganizer'):
            self.name = name
            self.organizer = organizer
            self.drive_encodings = set()
            self.local_encodings = set()
            self.local_files = []
            self.drive_files = []
            self.is_moved_local = False
            self.is_moved_drive = False
        
        def add_local_encoding(self, encoding: Tuple[bool], file_path: Path):
            """Add a local encoding to this image."""
            self.local_encodings.add(encoding)
            self.local_files.append({'path': file_path, 'encoding': encoding})
        
        def add_drive_encoding(self, encoding: Tuple[bool], file_item: Any, parent_id: str):
            """Add a drive encoding to this image."""
            self.drive_encodings.add(encoding)
            self.drive_files.append({'file_item': file_item, 'encoding': encoding, 'parent_id': parent_id})
        
        def get_resolved_encoding(self) -> Tuple[bool]:
            """Get resolved encoding by merging drive and local encodings and resolving conflicts."""
            # Start with all False encoding
            encoding_length = len(self.organizer.pomelo_classes) + 1
            merged_encoding = [False] * encoding_length
            
            # Merge all encodings from drive and local
            for encoding in self.drive_encodings.union(self.local_encodings):
                for i, value in enumerate(encoding):
                    if value and i < encoding_length:
                        merged_encoding[i] = True
            
            # Handle root folder case (index 0 is True)
            if merged_encoding[0]:
                merged_encoding[0] = False
                # If no other classes are True, assign to Unlabeled (index 1)
                if not any(merged_encoding[1:]):
                    if encoding_length > 1:
                        merged_encoding[1] = True
            
            # Filter to keep only highest priority classes
            positive_indices = [i for i, present in enumerate(merged_encoding) if present and i > 0]
            if len(positive_indices) > 1:
                # Find highest priority
                max_priority = -1
                best_indices = []
                
                for idx in positive_indices:
                    class_name = self.organizer._get_class_name_by_index(idx)
                    if class_name in self.organizer.pomelo_classes:
                        pomelo_class = self.organizer.pomelo_classes[class_name]
                        if pomelo_class.priority_weight > max_priority:
                            max_priority = pomelo_class.priority_weight
                            best_indices = [idx]
                        elif pomelo_class.priority_weight == max_priority:
                            best_indices.append(idx)
                
                # Reset encoding and set only highest priority classes
                merged_encoding = [False] * encoding_length
                for idx in best_indices:
                    merged_encoding[idx] = True
            
            return tuple(merged_encoding)
        
        def get_positive_classes(self) -> List[str]:
            """Get the list of positive classes from resolved encoding."""
            resolved_encoding = self.get_resolved_encoding()
            positive_classes = []
            
            for i, present in enumerate(resolved_encoding):
                if present:
                    class_name = self.organizer._get_class_name_by_index(i)
                    if class_name and class_name != "_root_":
                        positive_classes.append(class_name)
            
            return positive_classes
        
        def get_original_drive_subfolder(self):
            """Get original drive subfolder using first element of drive_encodings."""
            if not self.drive_encodings:
                return None
            
            first_encoding = next(iter(self.drive_encodings))
            encoding_key = self.organizer._encoding_to_key(first_encoding)
            return self.organizer.pomelo_class_subfolders.get(encoding_key)
        
        def get_original_local_subfolder(self):
            """Get original local subfolder using first element of local_encodings."""
            if not self.local_encodings:
                return None
            
            first_encoding = next(iter(self.local_encodings))
            encoding_key = self.organizer._encoding_to_key(first_encoding)
            return self.organizer.pomelo_class_subfolders.get(encoding_key)
        
        def get_new_subfolder(self):
            """Get new subfolder using resolved encoding."""
            resolved_encoding = self.get_resolved_encoding()
            encoding_key = self.organizer._encoding_to_key(resolved_encoding)
            
            if encoding_key not in self.organizer.pomelo_class_subfolders:
                # Create new subfolder
                self.organizer.pomelo_class_subfolders[encoding_key] = (
                    self.organizer.PomeloClassSubfolder(resolved_encoding, self.organizer, is_new=True)
                )
            
            return self.organizer.pomelo_class_subfolders[encoding_key]
        
        def move_to_new_subfolder(self) -> bool:
            """Move image to new subfolder based on resolved encoding."""
            new_subfolder = self.get_new_subfolder()
            original_drive_subfolder = self.get_original_drive_subfolder()
            original_local_subfolder = self.get_original_local_subfolder()
            
            success_local = True
            success_drive = True
            
            # Move local files
            if self.local_files and (not original_local_subfolder or original_local_subfolder != new_subfolder):
                success_local = self._move_local_to_subfolder(new_subfolder)
                if success_local and original_local_subfolder:
                    original_local_subfolder.local_dir_count -= len(self.local_files)
                    self.is_moved_local = True
                new_subfolder.local_dir_count += len(self.local_files) if success_local else 0
            
            # Move drive files
            if self.drive_files and (not original_drive_subfolder or original_drive_subfolder != new_subfolder):
                success_drive = self._move_drive_to_subfolder(new_subfolder)
                if success_drive and original_drive_subfolder:
                    original_drive_subfolder.drive_dir_count -= len(self.drive_files)
                    self.is_moved_drive = True
                new_subfolder.drive_dir_count += len(self.drive_files) if success_drive else 0
            
            return success_local and success_drive
        
        def _move_local_to_subfolder(self, target_subfolder) -> bool:
            """Move local files to target subfolder."""
            target_folder_path = self.organizer._get_or_create_local_folder(target_subfolder.name)
            
            success = True
            for file_info in self.local_files:
                source_path = file_info['path']
                target_path = target_folder_path / source_path.name
                
                try:
                    if source_path != target_path:
                        shutil.move(str(source_path), str(target_path))
                        file_info['path'] = target_path
                        logging.info(f"Moved local image {self.name} to {target_subfolder.name}")
                except Exception as e:
                    logging.error(f"Failed to move local image {self.name}: {str(e)}")
                    success = False
            
            return success
        
        def _move_drive_to_subfolder(self, target_subfolder) -> bool:
            """Move drive files to target subfolder."""
            target_folder_id = self.organizer._get_or_create_drive_folder(target_subfolder.name)
            if not target_folder_id:
                return False
            
            success = True
            for file_info in self.drive_files:
                file_item = file_info['file_item']
                current_parent_id = file_info['parent_id']
                
                if current_parent_id != target_folder_id:
                    try:
                        file_item['parents'] = [{'id': target_folder_id}]
                        file_item.Upload()
                        file_info['parent_id'] = target_folder_id
                        logging.info(f"Moved drive image {self.name} to {target_subfolder.name}")
                    except Exception as e:
                        logging.error(f"Failed to move drive image {self.name}: {str(e)}")
                        success = False
            
            return success
        
        def move_duplicates(self):
            """Move duplicate images to duplicates folder."""
            # Move local duplicates
            if len(self.local_files) > 1:
                self._move_local_duplicates()
            
            # Move drive duplicates
            if len(self.drive_files) > 1:
                self._move_drive_duplicates()
        
        def _move_local_duplicates(self):
            """Move duplicate local images to duplicates folder."""
            duplicates_folder = self.organizer._get_or_create_local_folder("duplicates")
            
            # Keep the first file, move the rest to duplicates
            for i, file_info in enumerate(self.local_files[1:], 1):
                source_path = file_info['path']
                new_filename = f"{self.name}_dup{i:02d}{source_path.suffix}"
                target_path = duplicates_folder / new_filename
                
                try:
                    shutil.move(str(source_path), str(target_path))
                    file_info['path'] = target_path
                    logging.info(f"Moved duplicate local image: {self.name} -> duplicates/{new_filename}")
                except Exception as e:
                    logging.error(f"Failed to move duplicate local image {self.name}: {str(e)}")
        
        def _move_drive_duplicates(self):
            """Move duplicate drive images to duplicates folder."""
            duplicates_folder_id = self.organizer._get_or_create_drive_folder("duplicates")
            if not duplicates_folder_id:
                logging.error("Failed to create duplicates folder in Google Drive")
                return
            
            # Keep the first file, move the rest to duplicates
            for i, file_info in enumerate(self.drive_files[1:], 1):
                file_item = file_info['file_item']
                new_filename = f"{self.name}_dup{i:02d}{Path(file_item['title']).suffix}"
                
                try:
                    file_item['title'] = new_filename
                    file_item['parents'] = [{'id': duplicates_folder_id}]
                    file_item.Upload()
                    logging.info(f"Moved duplicate drive image: {self.name} -> duplicates/{new_filename}")
                except Exception as e:
                    logging.error(f"Failed to move duplicate drive image {self.name}: {str(e)}")
        
        def __str__(self):
            return f"PomeloImage(name='{self.name}', drive_encodings={len(self.drive_encodings)}, local_encodings={len(self.local_encodings)})"
    
    class PomeloClassSubfolder:
        """Represents a pomelo class subfolder with its encoding and file counts."""
        
        def __init__(self, encoding: Tuple[bool], organizer: 'PomeloDatasetOrganizer', drive_id: Optional[str] = None, is_new: bool = False):
            self.encoding = encoding
            self.drive_id = drive_id
            self.local_dir_count = 0
            self.drive_dir_count = 0
            self.is_new = is_new
            self._name = None
            self.organizer = organizer
        
        @property
        def name(self):
            """Get the subfolder name based on encoding."""
            if self._name is None:
                self._name = self.load_encoding_subfolder_name()
            return self._name
        
        def load_encoding_subfolder_name(self) -> str:
            """Generate subfolder name based on encoding."""
            # Handle root folder (index 0 is True)
            if self.encoding[0]:
                return "_root_"
            
            # Count True values
            true_indices = [i for i, present in enumerate(self.encoding) if present]
            
            # If only one True value, return corresponding class name
            if len(true_indices) == 1:
                class_name = self.organizer._get_class_name_by_index(true_indices[0])
                return class_name if class_name else "unknown"
            
            # If multiple True values, create conflicted folder name
            class_names = []
            for idx in true_indices:
                class_name = self.organizer._get_class_name_by_index(idx)
                if class_name:
                    class_names.append(class_name)
            
            sorted_classes = sorted(class_names)
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
            return f"PomeloClassSubfolder(name='{self.name}', encoding={self.encoding}, local_count={self.local_dir_count}, drive_count={self.drive_dir_count})"
    
    def __init__(
        self,
        google_drive_images_folder: str = os.environ['DATASET_GOOGLE_DRIVE_ID'],
        local_images_folder: str = r"images\processed",
        labeling_csv: str = r"tracker\tracker.csv",
        ignore_subfolders: Set[str] = None
    ):
        """
        Initialize the PomeloDatasetOrganizer with folder paths and configuration.
        """
        if ignore_subfolders is None:
            ignore_subfolders = {"protected"}
        
        self._setup_logging()
        logging.info("Initializing PomeloDatasetOrganizer...")
        
        self.google_drive_images_folder = google_drive_images_folder
        self.local_images_folder = Path(local_images_folder)
        self.labeling_csv = Path(labeling_csv)
        self.ignore_subfolders = {folder.lower() for folder in ignore_subfolders}
        
        self._check_csv_permissions()
        self.drive_folder_id = self._extract_drive_folder_id(google_drive_images_folder)
        self.drive_service = self._initialize_google_drive()
        
        logging.info(f"Using Google Drive folder: {google_drive_images_folder}")
        logging.info(f"Using local images folder: {local_images_folder}")
        logging.info(f"Using labeling CSV: {labeling_csv}")
        
        self.pomelo_classes = self._load_pomelo_classes()
        class_names = list(self.pomelo_classes.keys())
        logging.info(f"Pomelo classes found: {', '.join(class_names)}")
        
        self.pomelo_class_subfolders = {}
        self.pomelo_images = {}
        
        self._load_pomelo_class_subfolders()
        self._load_images_from_subfolders()
    
    def _load_pomelo_classes(self) -> Dict[str, 'PomeloDatasetOrganizer.PomeloClass']:
        """Load pomelo classes as PomeloClass objects."""
        classes_config_path = Path(r"configs\pomelo_classes.csv")
        
        if not classes_config_path.exists():
            raise FileNotFoundError(f"Pomelo classes config not found: {classes_config_path}")
        
        df = pd.read_csv(classes_config_path)
        classes_dict = {}
        
        # Create encodings for each class
        class_names = [str(row['Name']).strip().lower() for _, row in df.iterrows()]
        
        for i, (_, row) in enumerate(df.iterrows()):
            class_name = str(row['Name']).strip().lower()
            priority_weight = int(row['Priority Weight'])
            
            # Create encoding: index 0 is root, index i+1 is this class
            encoding = [False] * (len(class_names) + 1)
            encoding[i + 1] = True  # +1 because index 0 is root
            
            classes_dict[class_name] = self.PomeloClass(
                name=class_name,
                priority_weight=priority_weight,
                encoding=tuple(encoding)
            )
        
        return dict(sorted(classes_dict.items()))
    
    def _load_pomelo_class_subfolders(self):
        """Load pomelo class subfolders from both drive and local."""
        logging.info("Loading pomelo class subfolders...")
        
        # Add root folder
        root_encoding = [True] + [False] * len(self.pomelo_classes)
        root_key = self._encoding_to_key(tuple(root_encoding))
        self.pomelo_class_subfolders[root_key] = self.PomeloClassSubfolder(
            tuple(root_encoding), self, self.drive_folder_id
        )
        
        # Load drive subfolders
        drive_subfolders = self._get_direct_subfolders_drive()
        
        # Load local subfolders
        local_subfolders = self._get_direct_subfolders_local()
        
        # Create subfolder instances for all found subfolders
        all_subfolders = set(drive_subfolders.keys()).union(set(local_subfolders.keys()))
        
        for folder_name in all_subfolders:
            if folder_name.lower() in self.ignore_subfolders or folder_name == "duplicates":
                continue
                
            # Determine encoding for this subfolder
            encoding = self._subfolder_name_to_encoding(folder_name)
            if not any(encoding):  # Skip if all False
                continue
                
            encoding_key = self._encoding_to_key(encoding)
            
            if encoding_key not in self.pomelo_class_subfolders:
                drive_id = drive_subfolders.get(folder_name)
                self.pomelo_class_subfolders[encoding_key] = self.PomeloClassSubfolder(
                    encoding, self, drive_id
                )
    
    def _load_images_from_subfolders(self):
        """Load images from all pomelo class folders."""
        logging.info("Loading images from pomelo class folders...")
        
        total_subfolders = len(self.pomelo_class_subfolders)
        logging.info(f"Reading {total_subfolders} folders...")
        
        for i, (encoding_key, subfolder) in enumerate(self.pomelo_class_subfolders.items(), 1):
            logging.info(f"Reading folder {i}/{total_subfolders}: {subfolder.name}")
            
            local_folder_path = self.local_images_folder / subfolder.name if subfolder.name != "_root_" else self.local_images_folder
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
                image_name = file_path.stem
                
                if image_name not in self.pomelo_images:
                    self.pomelo_images[image_name] = self.PomeloImage(image_name, self)
                
                self.pomelo_images[image_name].add_local_encoding(subfolder.encoding, file_path)
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
                        image_name = Path(file_item['title']).stem
                        
                        if image_name not in self.pomelo_images:
                            self.pomelo_images[image_name] = self.PomeloImage(image_name, self)
                        
                        self.pomelo_images[image_name].add_drive_encoding(subfolder.encoding, file_item, folder_id)
                        subfolder.drive_dir_count += 1
                        
        except Exception as e:
            logging.error(f"Error loading images from drive subfolder {subfolder.name}: {str(e)}")
    
    def _subfolder_name_to_encoding(self, folder_name: str) -> Tuple[bool]:
        """Convert subfolder name to encoding tuple."""
        encoding_length = len(self.pomelo_classes) + 1
        encoding = [False] * encoding_length
        
        # Handle root folder
        if folder_name == "_root_":
            encoding[0] = True
            return tuple(encoding)
        
        # Handle conflicted folders
        if folder_name.startswith("conflicted_"):
            classes_str = folder_name[11:]  # Remove "conflicted_" prefix
            class_names = classes_str.split("_")
            
            for class_name in class_names:
                if class_name in self.pomelo_classes:
                    class_idx = self._get_class_index_by_name(class_name)
                    if class_idx is not None:
                        encoding[class_idx] = True
            
            return tuple(encoding)
        
        # Handle regular class folders
        if folder_name in self.pomelo_classes:
            class_idx = self._get_class_index_by_name(folder_name)
            if class_idx is not None:
                encoding[class_idx] = True
        
        return tuple(encoding)
    
    def _get_class_index_by_name(self, class_name: str) -> Optional[int]:
        """Get the index of a class in the encoding."""
        class_names = list(self.pomelo_classes.keys())
        try:
            return class_names.index(class_name.lower()) + 1  # +1 because index 0 is root
        except ValueError:
            return None
    
    def _get_class_name_by_index(self, index: int) -> Optional[str]:
        """Get class name by index in encoding."""
        if index == 0:
            return "_root_"
        
        class_idx = index - 1  # -1 because index 0 is root
        class_names = list(self.pomelo_classes.keys())
        
        if 0 <= class_idx < len(class_names):
            return class_names[class_idx]
        
        return None
    
    def _encoding_to_key(self, encoding: Tuple[bool]) -> str:
        """Convert encoding tuple to string key."""
        return "".join("1" if val else "0" for val in encoding)
    
    def _get_direct_subfolders_drive(self) -> Dict[str, str]:
        """Get direct subfolder names and IDs from Google Drive folder."""
        folder_dict = {}
        try:
            folder_list = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            for folder in folder_list:
                folder_name = folder['title'].lower()
                if folder_name not in self.ignore_subfolders and folder_name != "duplicates":
                    folder_dict[folder_name] = folder['id']
        except Exception as e:
            logging.error(f"Error getting Drive subfolders: {str(e)}")
        return folder_dict
    
    def _get_direct_subfolders_local(self) -> Dict[str, Path]:
        """Get direct subfolder names and paths from local folder."""
        folder_dict = {}
        if self.local_images_folder.exists():
            for item in self.local_images_folder.iterdir():
                if item.is_dir() and item.name.lower() not in self.ignore_subfolders and item.name.lower() != "duplicates":
                    folder_dict[item.name.lower()] = item
        return folder_dict
    
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
    
    def _extract_drive_folder_id(self, drive_url: str) -> str:
        """Extract folder ID from Google Drive URL."""
        match = re.search(r'/folders/([a-zA-Z0-9_-]+)', drive_url)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract folder ID from URL: {drive_url}")
    
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
    
    def organize_dataset(self):
        """Main method to organize the entire dataset."""
        logging.info("Starting pomelo dataset organization...")
        
        try:
            moved_local_count = 0
            moved_drive_count = 0
            conflicted_count = 0
            duplicates_local_count = 0
            duplicates_drive_count = 0
            
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
                
                # Move to new subfolder
                if image.move_to_new_subfolder():
                    if image.is_moved_local:
                        moved_local_count += 1
                    if image.is_moved_drive:
                        moved_drive_count += 1
                
                # Check for conflicts
                positive_classes = image.get_positive_classes()
                if len(positive_classes) > 1:
                    conflicted_count += 1
                
                # Move duplicates
                image.move_duplicates()
                if len(image.local_files) > 1:
                    duplicates_local_count += len(image.local_files) - 1
                if len(image.drive_files) > 1:
                    duplicates_drive_count += len(image.drive_files) - 1
            
            # Delete empty subfolders
            self._delete_empty_subfolders()
            
            logging.info(f"Organization completed!")
            logging.info(f"Images moved locally: {moved_local_count}")
            logging.info(f"Images moved on Drive: {moved_drive_count}")
            logging.info(f"Images with conflicting classes: {conflicted_count}")
            logging.info(f"Duplicates in local folder: {duplicates_local_count}")
            logging.info(f"Duplicates in Drive folder: {duplicates_drive_count}")
            
        except Exception as e:
            logging.error(f"Error during dataset organization: {str(e)}")
            raise

    def _delete_empty_subfolders(self):
        """Delete empty subfolders in both local and Google Drive."""
        logging.info("Deleting empty subfolders...")
        
        local_deleted = 0
        drive_deleted = 0
        
        for encoding_key, subfolder in list(self.pomelo_class_subfolders.items()):
            # Skip root folder and newly created folders
            if subfolder.encoding[0] or subfolder.is_new:
                continue
                
            if subfolder.delete("local"):
                # Actually delete the local folder
                local_folder_path = self.local_images_folder / subfolder.name
                if local_folder_path.exists() and local_folder_path.is_dir():
                    try:
                        local_folder_path.rmdir()
                        local_deleted += 1
                    except OSError as e:
                        logging.warning(f"Could not delete local folder {subfolder.name}: {e}")
            
            if subfolder.drive_id and subfolder.delete("drive"):
                # Actually delete the drive folder
                try:
                    drive_folder = self.drive_service.CreateFile({'id': subfolder.drive_id})
                    drive_folder.Delete()
                    drive_deleted += 1
                except Exception as e:
                    logging.warning(f"Could not delete drive folder {subfolder.name}: {e}")
        
        logging.info(f"Deleted {local_deleted} empty local subfolders and {drive_deleted} empty drive subfolders")

def run_pomelo_dataset_organizer(
    google_drive_folder: str = None,
    local_images_folder: str = None,
    labeling_csv: str = None,
    ignore_subfolders: Set[str] = None
):
    """
    Run the PomeloDatasetOrganizer with optional custom parameters.
    
    Args:
        google_drive_folder: Optional custom Google Drive folder URL
        local_images_folder: Optional custom local images folder path
        labeling_csv: Optional custom labeling CSV file path
        ignore_subfolders: Optional set of subfolder names to ignore
    """
    kwargs = {}
    if google_drive_folder:
        kwargs['google_drive_images_folder'] = google_drive_folder
    if local_images_folder:
        kwargs['local_images_folder'] = local_images_folder
    if labeling_csv:
        kwargs['labeling_csv'] = labeling_csv
    if ignore_subfolders:
        kwargs['ignore_subfolders'] = ignore_subfolders
    
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
    
    parser.add_argument(
        '--ignore-subfolders',
        type=str,
        nargs='*',
        default=["protected"],
        help='Subfolder names to ignore across all operations'
    )
    
    args = parser.parse_args()
    
    run_pomelo_dataset_organizer(
        google_drive_folder=args.google_drive_images_folder,
        local_images_folder=args.local_images_folder,
        labeling_csv=args.labeling_csv,
        ignore_subfolders=set(args.ignore_subfolders)
    )

if __name__ == "__main__":
    main()
