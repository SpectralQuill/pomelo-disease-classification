import argparse
import re
import logging
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

class PomeloDatasetOrganizer:
    """
    Organizes pomelo images across Google Drive and local folders based on class assignments.
    Handles class conflicts and moves images to appropriate directories.
    """
    
    class PomeloClass:
        """Represents a pomelo class with its properties and folder existence information."""
        
        def __init__(self, name: str, priority_weight: int, exists_in_drive: bool, exists_local: bool):
            self.name = name.lower()
            self.priority_weight = priority_weight
            self.exists_in_drive = exists_in_drive
            self.exists_local = exists_local
        
        def should_create_folder_drive(self) -> bool:
            """Determine if this class folder should be created in Google Drive."""
            return not self.exists_in_drive
        
        def should_create_folder_local(self) -> bool:
            """Determine if this class folder should be created locally."""
            return not self.exists_local
        
        def __str__(self):
            return f"PomeloClass(name='{self.name}', weight={self.priority_weight}, drive={self.exists_in_drive}, local={self.exists_local})"
    
    class PomeloImage:
        """Represents a pomelo image with its file information and manipulation methods."""
        
        def __init__(self, name: str, organizer: 'PomeloDatasetOrganizer'):
            self.name = name
            self.name_lower = name.lower()
            self.organizer = organizer
            self.local_files = []
            self.drive_files = []
            self.encoding = None
            self.resolved_encoding = None
            self.assigned_class = None
            self.is_conflicted = None
        
        def add_local_file(self, file_path: Path, parent_name: str):
            """Add a local file path to this image."""
            
            self.local_files.append({
                'path': file_path,
                'parent_name': parent_name.lower()
            })
        
        def add_drive_file(self, file_item, parent_id: str, parent_name: str):
            """Add a Google Drive file to this image."""
            self.drive_files.append({
                'file_item': file_item,
                'parent_id': parent_id,
                'parent_name': parent_name.lower()
            })

        def load_encodings(self) -> Tuple[Tuple[bool], Tuple[bool]]:
            """Load both multi-hot and resolved multi-hot encodings."""
            encoding = self._load_multi_hot_encoding()
            resolved_encoding = self._load_resolved_multi_hot_encoding()
            return encoding, resolved_encoding

        def _load_multi_hot_encoding(self) -> Tuple[bool]:
            """Create multi-hot encoding for this image across all pomelo classes."""
            encoding = [False] * len(self.organizer.pomelo_classes)
            class_names = list(self.organizer.pomelo_classes.keys())

            for file_info in self.local_files:
                parent_name = file_info['parent_name']
                for i, class_name in enumerate(class_names):
                    if parent_name == class_name:
                        encoding[i] = True
            
            for file_info in self.drive_files:
                parent_name = file_info['parent_name']
                for i, class_name in enumerate(class_names):
                    if parent_name == class_name:
                        encoding[i] = True
            
            self.encoding = encoding
            return encoding
        
        def _load_resolved_multi_hot_encoding(self) -> Tuple[bool]:
            """Resolve class conflicts by removing less dominant classes."""
            if self.encoding is None:
                self._load_multi_hot_encoding()
            
            if sum(self.encoding) <= 1:
                self.resolved_encoding = self.encoding
                return self.encoding
            
            positive_indices = [i for i, present in enumerate(self.encoding) if present]
            class_names = list(self.organizer.pomelo_classes.keys())
            
            max_priority = -1
            best_class_index = -1
            
            for idx in positive_indices:
                class_name = class_names[idx]
                pomelo_class = self.organizer.pomelo_classes[class_name]
                if pomelo_class.priority_weight > max_priority:
                    max_priority = pomelo_class.priority_weight
                    best_class_index = idx
            
            tied_classes = [idx for idx in positive_indices 
                           if self.organizer.pomelo_classes[class_names[idx]].priority_weight == max_priority]
            
            if len(tied_classes) > 1:
                new_encoding = [False] * len(self.encoding)
                for idx in tied_classes:
                    new_encoding[idx] = True
            else:
                new_encoding = [False] * len(self.encoding)
                new_encoding[best_class_index] = True
            
            self.resolved_encoding = new_encoding
            return new_encoding
        
        def get_positive_classes(self) -> List[str]:
            """Get the list of positive classes from resolved encoding."""
            if self.resolved_encoding is None:
                self._load_resolved_multi_hot_encoding()
            
            class_names = list(self.organizer.pomelo_classes.keys())
            return [class_names[i] for i, present in enumerate(self.resolved_encoding) if present]
        
        def move_to_class(self, target_class_name: str) -> bool:
            """Move this image to the specified class folder."""
            success_local = True
            success_drive = True
            
            if self.organizer.local_images_folder.exists() and self.local_files:
                success_local = self._move_local_to_class(target_class_name)
            
            if self.drive_files:
                success_drive = self._move_drive_to_class(target_class_name)
            
            self.assigned_class = target_class_name
            return success_local and success_drive
        
        def move_to_conflicted(self, positive_classes: List[str]) -> bool:
            """Move this image to conflicted folder."""
            conflicted_folder_name = self.organizer._create_conflicted_folder_name(positive_classes)
            success_local = True
            success_drive = True
            
            if self.organizer.local_images_folder.exists() and self.local_files:
                success_local = self._move_local_to_conflicted(conflicted_folder_name)
            
            if self.drive_files:
                success_drive = self._move_drive_to_conflicted(conflicted_folder_name)
            self.is_conflicted = True
            return success_local and success_drive
        
        def _move_local_to_class(self, target_class_name: str) -> bool:
            """Move local files to target class folder."""
            if not self.local_files:
                return True
            
            first_file = self.local_files[0]
            source_path = first_file['path']
            target_folder = self.organizer.local_images_folder / target_class_name
            
            if target_class_name in self.organizer.pomelo_classes:
                target_folder.mkdir(parents=True, exist_ok=True)
            
            target_path = target_folder / (self.name + source_path.suffix)
            
            moved_successfully = False
            if source_path != target_path:
                try:
                    source_path.rename(target_path)
                    first_file['path'] = target_path
                    first_file['parent_name'] = target_class_name
                    logging.info(f"Moved local image: {self.name} -> {target_class_name}")
                    moved_successfully = True
                except Exception as e:
                    logging.error(f"Failed to move local image {self.name}: {str(e)}")
                    return False
            
            if len(self.local_files) > 1:
                self._handle_local_duplicates()
            
            return moved_successfully
        
        def _move_drive_to_class(self, target_class_name: str) -> bool:
            """Move drive files to target class folder."""
            if not self.drive_files:
                return True
            
            first_file = self.drive_files[0]
            file_item = first_file['file_item']
            current_parent_id = first_file['parent_id']
            
            target_folder_id = self.organizer._get_or_create_class_folder_drive(target_class_name)
            if not target_folder_id:
                return False
            
            moved_successfully = True
            if current_parent_id != target_folder_id:
                try:
                    file_item['parents'] = [{'id': target_folder_id}]
                    file_item.Upload()
                    
                    first_file['parent_id'] = target_folder_id
                    first_file['parent_name'] = target_class_name
                    
                    logging.info(f"Moved Drive image: {self.name} -> {target_class_name}")
                except Exception as e:
                    logging.error(f"Failed to move Drive image {self.name}: {str(e)}")
                    moved_successfully = False
            
            if len(self.drive_files) > 1:
                self._handle_drive_duplicates()
            
            return moved_successfully
        
        def _move_local_to_conflicted(self, conflicted_folder_name: str) -> bool:
            """Move local files to conflicted folder."""
            if not self.local_files:
                return True
            
            local_conflicted = self.organizer._get_or_create_conflicted_folder_local(conflicted_folder_name)
            first_file = self.local_files[0]
            source_path = first_file['path']
            target_path = local_conflicted / source_path.name
            
            if source_path != target_path:
                try:
                    source_path.rename(target_path)
                    first_file['path'] = target_path
                    first_file['parent_name'] = conflicted_folder_name
                    logging.info(f"Moved local image to conflicted: {self.name} -> {conflicted_folder_name}")
                    return True
                except Exception as e:
                    logging.error(f"Failed to move local image {self.name} to conflicted: {str(e)}")
                    return False
            
            return True
        
        def _move_drive_to_conflicted(self, conflicted_folder_name: str) -> bool:
            """Move drive files to conflicted folder."""
            if not self.drive_files:
                return True
            
            drive_conflicted_id = self.organizer._get_or_create_conflicted_folder_drive(conflicted_folder_name)
            if not drive_conflicted_id:
                return False
            
            first_file = self.drive_files[0]
            file_item = first_file['file_item']
            
            try:
                file_item['parents'] = [{'id': drive_conflicted_id}]
                file_item.Upload()
                
                first_file['parent_id'] = drive_conflicted_id
                first_file['parent_name'] = conflicted_folder_name
                
                logging.info(f"Moved Drive image to conflicted: {self.name} -> {conflicted_folder_name}")
                return True
            except Exception as e:
                logging.error(f"Failed to move Drive image {self.name} to conflicted: {str(e)}")
                return False
        
        def _handle_local_duplicates(self):
            """Move duplicate local images to duplicates folder."""
            duplicates_folder = self.organizer.local_images_folder / "duplicates"
            duplicates_folder.mkdir(parents=True, exist_ok=True)
            
            max_count = len(self.local_files)
            padding_length = len(str(max_count))
            
            for index, file_info in enumerate(self.local_files[1:], start=2):
                source_path = file_info['path']
                count_str = str(index).zfill(padding_length)
                new_filename = f"{self.name}_{count_str}{source_path.suffix}"
                target_path = duplicates_folder / new_filename
                
                try:
                    source_path.rename(target_path)
                    logging.info(f"Moved duplicate local image: {self.name} -> duplicates/{new_filename}")
                    file_info['path'] = target_path
                    file_info['parent_name'] = 'duplicates'
                except Exception as e:
                    logging.error(f"Failed to move duplicate local image {self.name}: {str(e)}")
        
        def _handle_drive_duplicates(self):
            """Move duplicate drive images to duplicates folder."""
            duplicates_folder_id = self.organizer._get_or_create_duplicates_folder_drive()
            if not duplicates_folder_id:
                logging.error("Failed to create duplicates folder in Google Drive")
                return
            
            max_count = len(self.drive_files)
            padding_length = len(str(max_count))
            
            for index, file_info in enumerate(self.drive_files[1:], start=2):
                file_item = file_info['file_item']
                count_str = str(index).zfill(padding_length)
                new_filename = f"{self.name}_{count_str}{Path(file_item['title']).suffix}"
                
                try:
                    file_item['title'] = new_filename
                    file_item['parents'] = [{'id': duplicates_folder_id}]
                    file_item.Upload()
                    
                    logging.info(f"Moved duplicate Drive image: {self.name} -> duplicates/{new_filename}")
                    file_info['parent_id'] = duplicates_folder_id
                    file_info['parent_name'] = 'duplicates'
                except Exception as e:
                    logging.error(f"Failed to move duplicate Drive image {self.name}: {str(e)}")
        
        def __str__(self):
            return f"PomeloImage(name='{self.name}', local_files={len(self.local_files)}, drive_files={len(self.drive_files)})"
    
    def __init__(
        self,
        google_drive_images_folder = os.environ['DATASET_GOOGLE_DRIVE_ID'],
        local_images_folder = r"images\processed",
        labeling_csv = r"tracker\tracker.csv"
    ):
        """
        Initialize the PomeloDatasetOrganizer with folder paths and configuration.
        """
        self.google_drive_images_folder = google_drive_images_folder
        self.local_images_folder = Path(local_images_folder)
        self.labeling_csv = Path(labeling_csv)
        
        self._check_csv_permissions()
        self.drive_folder_id = self._extract_drive_folder_id(google_drive_images_folder)
        self.drive_service = self._initialize_google_drive()
        self._setup_logging()
        
        self.pomelo_classes = self._load_pomelo_classes()
        self.pomelo_images = self._load_pomelo_images()
        
        self.changed_classes_count = 0
        self.conflicted_images_count = 0
    
    def _load_pomelo_classes(self) -> Dict[str, 'PomeloDatasetOrganizer.PomeloClass']:
        """Load pomelo classes as PomeloClass objects."""
        classes_config_path = Path(r"configs\pomelo_classes.csv")
        
        if not classes_config_path.exists():
            raise FileNotFoundError(f"Pomelo classes config not found: {classes_config_path}")
        
        df = pd.read_csv(classes_config_path)
        classes_dict = {}
        
        drive_class_folders = self._get_direct_subfolders_drive()
        local_class_folders = self._get_direct_subfolders_local()
        
        for _, row in df.iterrows():
            class_name = str(row['Name']).strip().lower()
            priority_weight = int(row['Priority Weight'])
            exists_in_drive = class_name in drive_class_folders
            exists_local = class_name in local_class_folders
            
            classes_dict[class_name] = self.PomeloClass(
                name=class_name,
                priority_weight=priority_weight,
                exists_in_drive=exists_in_drive,
                exists_local=exists_local
            )
        
        return dict(sorted(classes_dict.items()))
    
    def _load_pomelo_images(self) -> Dict[str, 'PomeloDatasetOrganizer.PomeloImage']:
        """Load pomelo images as PomeloImage objects."""
        images = {}
        
        if self.local_images_folder.exists():
            locations_to_search = [self.local_images_folder]
            locations_to_search.extend([item for item in self.local_images_folder.iterdir() if item.is_dir()])
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
            
            for location in locations_to_search:
                for file_path in location.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                        filename_without_ext = file_path.stem
                        parent_name = location.name.lower() if location != self.local_images_folder else 'root'
                        
                        if filename_without_ext not in images:
                            images[filename_without_ext] = self.PomeloImage(filename_without_ext, self)
                        
                        images[filename_without_ext].add_local_file(file_path, parent_name)
        
        try:
            locations_to_search = [{'id': self.drive_folder_id, 'name': 'root'}]
            folder_list = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            for folder in folder_list:
                locations_to_search.append({'id': folder['id'], 'name': folder['title']})
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
            
            for location in locations_to_search:
                file_list = self.drive_service.ListFile({
                    'q': f"'{location['id']}' in parents and trashed=false"
                }).GetList()
                
                for file_item in file_list:
                    if file_item['mimeType'] != 'application/vnd.google-apps.folder':
                        file_title = file_item['title']
                        file_extension = Path(file_title).suffix.lower()
                        if file_extension in image_extensions:
                            filename_without_ext = Path(file_title).stem
                            
                            if filename_without_ext not in images:
                                images[filename_without_ext] = self.PomeloImage(filename_without_ext, self)
                            
                            images[filename_without_ext].add_drive_file(
                                file_item, location['id'], location['name']
                            )
                            
        except Exception as e:
            logging.error(f"Error loading Drive images: {str(e)}")
        
        return images

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
    
    def _extract_drive_folder_id(self, drive_url) -> str:
        """Extract folder ID from Google Drive URL."""
        match = re.search(r'/folders/([a-zA-Z0-9_-]+)', drive_url)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract folder ID from URL: {drive_url}")
    
    def _get_direct_subfolders_drive(self) -> Set[str]:
        """Get direct subfolder names from Google Drive folder."""
        folder_names = set()
        try:
            folder_list = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            for folder in folder_list:
                folder_names.add(folder['title'].lower())
        except Exception as e:
            logging.error(f"Error getting Drive subfolders: {str(e)}")
        return folder_names
    
    def _get_direct_subfolders_local(self) -> Set[str]:
        """Get direct subfolder names from local folder."""
        folder_names = set()
        if self.local_images_folder.exists():
            for item in self.local_images_folder.iterdir():
                if item.is_dir():
                    folder_names.add(item.name.lower())
        return folder_names
    
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
    
    def _get_or_create_class_folder_drive(self, folder_name) -> Optional[str]:
        """Get existing class folder ID or create if it exists in config and doesn't exist."""
        existing_folders = self.drive_service.ListFile({
            'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and title='{folder_name}' and trashed=false"
        }).GetList()
        
        if existing_folders:
            return existing_folders[0]['id']
        
        if folder_name in self.pomelo_classes:
            pomelo_class = self.pomelo_classes[folder_name]
            if pomelo_class.should_create_folder_drive():
                try:
                    folder_metadata = {
                        'title': folder_name,
                        'mimeType': 'application/vnd.google-apps.folder',
                        'parents': [{'id': self.drive_folder_id}]
                    }
                    
                    folder = self.drive_service.CreateFile(folder_metadata)
                    folder.Upload()
                    logging.info(f"Created Drive class folder: {folder_name}")
                    
                    pomelo_class.exists_in_drive = True
                    return folder['id']
                except Exception as e:
                    logging.error(f"Failed to create Drive folder {folder_name}: {str(e)}")
        
        return None

    def _get_or_create_duplicates_folder_drive(self) -> Optional[str]:
        """Get existing duplicates folder ID or create if doesn't exist."""
        try:
            existing_folders = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and title='duplicates' and trashed=false"
            }).GetList()
            
            if existing_folders:
                return existing_folders[0]['id']
            
            folder_metadata = {
                'title': 'duplicates',
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [{'id': self.drive_folder_id}]
            }
            
            folder = self.drive_service.CreateFile(folder_metadata)
            folder.Upload()
            logging.info("Created Drive duplicates folder")
            return folder['id']
            
        except Exception as e:
            logging.error(f"Failed to create Drive duplicates folder: {str(e)}")
            return None

    def _create_conflicted_folder_name(self, positive_classes: List[str]) -> str:
        """Create folder name for conflicted images."""
        sorted_classes = sorted(positive_classes)
        class_names = "_".join(sorted_classes)
        return f"conflicted_{class_names}"
    
    def _get_or_create_conflicted_folder_local(self, folder_name) -> Path:
        """Get existing conflicted folder or create if doesn't exist."""
        conflicted_folder = self.local_images_folder / folder_name
        conflicted_folder.mkdir(parents=True, exist_ok=True)
        return conflicted_folder
    
    def _get_or_create_conflicted_folder_drive(self, folder_name) -> Optional[str]:
        """Get existing conflicted folder ID or create if doesn't exist."""
        try:
            existing_folders = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and title='{folder_name}' and trashed=false"
            }).GetList()
            
            if existing_folders:
                return existing_folders[0]['id']
            
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
    
    def _update_labeling_csv(self, image_name, new_class):
        """Update the class assignment in the labeling CSV file."""
        try:
            df = pd.read_csv(self.labeling_csv)
            
            mask = df['Name'].str.lower() == image_name.lower()
            if not mask.any():
                logging.warning(f"Image {image_name} not found in labeling CSV")
                return
            
            if new_class.lower() != "conflicted":
                new_class = new_class.capitalize()
            
            old_class = df.loc[mask, 'Class'].iloc[0]
            df.loc[mask, 'Class'] = new_class
            
            df.to_csv(self.labeling_csv, index=False)
            
            if old_class != new_class:
                logging.info(f"Updated {image_name}: {old_class} -> {new_class}")
                if new_class.lower() != "conflicted":
                    self.changed_classes_count += 1
            
        except Exception as e:
            logging.error(f"Failed to update labeling CSV for {image_name}: {str(e)}")
    
    def process_image(self, image_name):
        """Process a single image using PomeloImage object."""
        if image_name not in self.pomelo_images:
            logging.warning(f"Image {image_name} not found in preloaded images")
            return
        
        image = self.pomelo_images[image_name]
        logging.info(f"Processing image: {image_name}")
        
        image.load_encodings()
        positive_classes = image.get_positive_classes()
        
        if len(positive_classes) == 1:
            assigned_class = positive_classes[0]
            if image.move_to_class(assigned_class):
                self._update_labeling_csv(image_name, assigned_class)
            
        elif len(positive_classes) > 1:
            if image.move_to_conflicted(positive_classes):
                self._update_labeling_csv(image_name, "Conflicted")
                self.conflicted_images_count += 1
    
    def delete_empty_subfolders(self):
        """
        Delete empty subfolders in both local and Google Drive folders after organization.
        Only checks direct subfolders, not deep search.
        """
        logging.info("Cleaning up empty subfolders...")
        
        local_empty_count = self._delete_empty_local_subfolders()
        drive_empty_count = self._delete_empty_drive_subfolders()
        
        logging.info(f"Cleanup completed: {local_empty_count} local empty folders deleted, "
                    f"{drive_empty_count} Drive empty folders deleted")
    
    def _delete_empty_local_subfolders(self) -> int:
        """Delete empty direct subfolders in local images directory."""
        empty_count = 0

        if not self.local_images_folder.exists():
            return empty_count
        
        try:
            for item in self.local_images_folder.iterdir():
                if item.is_dir() and self._is_local_folder_empty(item):
                    try:
                        item.rmdir()
                        logging.info(f"Deleted empty local folder: {item.name}")
                        empty_count += 1
                    except OSError as e:
                        logging.debug(f"Could not delete {item}: {e}")
            
            return empty_count
            
        except Exception as e:
            logging.error(f"Error deleting empty local folders: {e}")
            return empty_count

    def _delete_empty_drive_subfolders(self) -> int:
        """Delete empty direct subfolders in Google Drive directory."""
        empty_count = 0
        
        try:
            folder_list = self.drive_service.ListFile({
                'q': f"'{self.drive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            for folder in folder_list:
                if self._is_drive_folder_empty(folder['id']):
                    try:
                        folder.Delete()
                        logging.info(f"Deleted empty Drive folder: {folder['title']}")
                        empty_count += 1
                    except Exception as e:
                        logging.debug(f"Could not delete Drive folder {folder['title']}: {e}")
            
            return empty_count
            
        except Exception as e:
            logging.error(f"Error deleting empty Drive folders: {e}")
            return empty_count

    def _is_local_folder_empty(self, folder_path: Path) -> bool:
        """Check if a local folder is completely empty."""
        if not folder_path.exists() or not folder_path.is_dir():
            return False
        
        try:
            return not any(folder_path.iterdir())
        except Exception as e:
            logging.debug(f"Error checking if folder is empty {folder_path}: {e}")
            return False

    def _is_drive_folder_empty(self, folder_id) -> bool:
        """Check if a Google Drive folder is completely empty."""
        try:
            item_list = self.drive_service.ListFile({
                'q': f"'{folder_id}' in parents and trashed=false"
            }).GetList()
            
            return len(item_list) == 0
            
        except Exception as e:
            logging.debug(f"Error checking if Drive folder is empty {folder_id}: {e}")
            return False

    def organize_dataset(self):
        """Main method to organize the entire dataset."""
        logging.info("Starting pomelo dataset organization...")
        
        try:
            df = pd.read_csv(self.labeling_csv)
            image_names = df['Name'].tolist()
            
            for image_name in image_names:
                self.process_image(str(image_name))
            
            self.delete_empty_subfolders()
            
            logging.info(f"Organization completed!")
            logging.info(f"Images with changed classes: {self.changed_classes_count}")
            logging.info(f"Images with conflicting classes: {self.conflicted_images_count}")
            
        except Exception as e:
            logging.error(f"Error during dataset organization: {str(e)}")
            raise

def run_pomelo_dataset_organizer(
    google_drive_folder = None,
    local_images_folder = None,
    labeling_csv = None
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
