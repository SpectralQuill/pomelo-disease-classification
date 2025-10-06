#!/usr/bin/env python3
"""
Citrus Maxima (Pomelo) Disease Classifier
EfficientNet-B0 + SVM Hybrid Approach
Author: AI Assistant
Date: 2024
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import joblib
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from albumentations.core.composition import OneOf
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

class CitrusDiseaseClassifier:
    def __init__(self, config):
        self.config = config
        self.img_size = (config['IMG_SIZE'], config['IMG_SIZE'])
        self.classes = config['CLASSES']
        self.num_classes = len(self.classes)
        self.le = LabelEncoder()
        self.efficientnet_model = None
        self.svm_model = None
        self.feature_extractor = None
        self.is_trained = False
        
        # Create directories
        os.makedirs(config['SAVE_DIR'], exist_ok=True)
        os.makedirs(os.path.join(config['SAVE_DIR'], 'plots'), exist_ok=True)
        os.makedirs(os.path.join(config['SAVE_DIR'], 'reports'), exist_ok=True)
        os.makedirs(os.path.join(config['SAVE_DIR'], 'augmentation_samples'), exist_ok=True)
        os.makedirs(os.path.join(config['SAVE_DIR'], 'inference_results'), exist_ok=True)
        
        set_seeds(config['RANDOM_SEED'])
        
    def load_and_preprocess_data(self):
        """Load images from dataset directory and preprocess"""
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        class_counts = {cls: 0 for cls in self.classes}
        
        for class_name in self.classes:
            class_dir = os.path.join(self.config['DATASET_DIR'], class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found. Skipping...")
                continue
                
            print(f"Loading images from {class_dir}")
            for img_file in tqdm(os.listdir(class_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Debug: Print image shape for first few images
                            if len(images) < 3:  # Print for first 3 images only
                                print(f"Image shape: {img.shape}, dtype: {img.dtype}")
                            
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            
                            # Ensure image has 3 channels (convert grayscale to RGB)
                            if len(img.shape) == 2:  # Grayscale image
                                print(f"Converting grayscale image to RGB: {img_path}")
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            elif img.shape[2] == 4:  # RGBA image
                                print(f"Converting RGBA image to RGB: {img_path}")
                                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                            elif img.shape[2] == 1:  # Single channel
                                print(f"Converting single channel image to RGB: {img_path}")
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            
                            # Apply contrast enhancement (CLAHE)
                            if self.config.get('APPLY_CLAHE', True):
                                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                                lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
                                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                            
                            images.append(img)
                            labels.append(class_name)
                            class_counts[class_name] += 1
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        if not images:
            raise ValueError("No images found in the dataset directory!")
        
        print(f"Loaded {len(images)} images")
        print("Class distribution:", class_counts)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Debug: Check final image shapes and channels
        print(f"Final images array shape: {images.shape}")
        print(f"Image data type: {images.dtype}")
        if len(images) > 0:
            print(f"Sample image shape: {images[0].shape}")
            print(f"Sample image range: [{images[0].min()}, {images[0].max()}]")

        # Encode labels
        label_encoded = self.le.fit_transform(labels)
        
        return images, label_encoded, class_counts
    
    def create_augmentation_pipeline(self):
        """Create data augmentation pipeline using albumentations"""
        augmentations = []
        
        if self.config.get('AUGMENTATION_OPTIONS', {}).get('random_rotation', True):
            augmentations.append(A.Rotate(limit=30, p=0.5))
        
        # Skip RandomResizedCrop entirely and use safer alternatives
        if self.config.get('AUGMENTATION_OPTIONS', {}).get('random_crop', True):
            # Use ShiftScaleRotate as a safer alternative that includes scaling
            augmentations.append(A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=0,  # We already have rotation above
                p=0.5
            ))
        
        if self.config.get('AUGMENTATION_OPTIONS', {}).get('gaussian_blur', True):
            augmentations.append(A.GaussianBlur(blur_limit=3, p=0.3))
        
        if self.config.get('AUGMENTATION_OPTIONS', {}).get('contrast_jitter', True):
            augmentations.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        
        if self.config.get('AUGMENTATION_OPTIONS', {}).get('scaling', True):
            augmentations.append(A.Affine(scale=(0.9, 1.1), p=0.5))
        
        if self.config.get('AUGMENTATION_OPTIONS', {}).get('cutout', True):
            augmentations.append(A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3))
        
        if self.config.get('AUGMENTATION_OPTIONS', {}).get('salt_pepper_noise', True):
            augmentations.append(A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3))
        
        # Always include horizontal flip as it's generally safe
        augmentations.append(A.HorizontalFlip(p=0.5))
    
        return A.Compose(augmentations)
    
    def augment_images(self, images, labels, target_samples_per_class=1000):
        """Augment images to balance classes"""
        print("Augmenting images...")
        
        augmentation_pipeline = self.create_augmentation_pipeline()
        augmented_images = []
        augmented_labels = []
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        for label in unique_labels:
            class_images = images[labels == label]
            current_count = len(class_images)
            
            if current_count < target_samples_per_class:
                num_to_generate = target_samples_per_class - current_count
                
                for _ in tqdm(range(num_to_generate), desc=f"Augmenting class {self.le.inverse_transform([label])[0]}"):
                    # Randomly select an image from the class
                    img = class_images[np.random.randint(0, current_count)]
                    
                    # Apply augmentation
                    augmented = augmentation_pipeline(image=img)
                    augmented_img = augmented['image']
                    
                    augmented_images.append(augmented_img)
                    augmented_labels.append(label)
        
        if augmented_images:
            all_images = np.concatenate([images, np.array(augmented_images)], axis=0)
            all_labels = np.concatenate([labels, np.array(augmented_labels)], axis=0)
        else:
            all_images, all_labels = images, labels
        
        print(f"After augmentation: {len(all_images)} total images")
        return all_images, all_labels
    
    def visualize_augmentation(self, images, labels, num_samples=5):
        """Visualize original and augmented images"""
        print("Generating augmentation samples...")
        
        augmentation_pipeline = self.create_augmentation_pipeline()
        
        for class_idx in range(self.num_classes):
            class_name = self.le.inverse_transform([class_idx])[0]
            class_images = images[labels == class_idx]
            
            if len(class_images) == 0:
                continue
                
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
            fig.suptitle(f'Augmentation Samples - {class_name}', fontsize=16)
            
            for i in range(num_samples):
                if i < len(class_images):
                    # Original image
                    axes[0, i].imshow(class_images[i])
                    axes[0, i].set_title('Original')
                    axes[0, i].axis('off')
                    
                    # Augmented image
                    augmented = augmentation_pipeline(image=class_images[i])
                    axes[1, i].imshow(augmented['image'])
                    axes[1, i].set_title('Augmented')
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['SAVE_DIR'], 'augmentation_samples', f'augmentation_{class_name}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def create_feature_extractor(self, fine_tune=False):
        """Create EfficientNet-B0 feature extractor using Keras Applications"""
        print("Creating EfficientNet-B0 feature extractor...")
        
        try:
            # First try with pre-trained weights
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
            print("✓ Loaded EfficientNet-B0 with pre-trained ImageNet weights")
        except ValueError as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
            print("Creating EfficientNet-B0 with random initialization...")
            # Fallback to random weights if shape mismatch occurs
            base_model = EfficientNetB0(
                weights=None,  # No pre-trained weights
                include_top=False,
                pooling='avg',
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
        
        base_model.trainable = fine_tune
        
        # Create a simple model that outputs features
        inputs = tf.keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = preprocess_input(inputs)
        features = base_model(x, training=False)
        
        self.feature_extractor = tf.keras.Model(inputs, features)
        print("✓ Successfully created EfficientNet-B0 feature extractor")
        
        return self.feature_extractor
     
            
    def extract_features(self, images, batch_size=32):
        """Extract features using EfficientNet-B0"""
        print("Extracting features...")
        
        if self.feature_extractor is None:
            self.create_feature_extractor(fine_tune=False)
        
        # Extract features in batches
        features = []
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i+batch_size]
            
            # Preprocess the batch for EfficientNet
            batch = preprocess_input(batch)
            
            # Extract features using predict
            batch_features = self.feature_extractor.predict(batch, verbose=0)
            features.append(batch_features)
        
        features = np.concatenate(features, axis=0)
        print(f"Extracted features shape: {features.shape}")
        
        return features
    
    def fine_tune_efficientnet(self, train_images, train_labels, val_images, val_labels):
        """Fine-tune EfficientNet-B0 using Keras model"""
        print("Fine-tuning EfficientNet-B0...")
        
        try:
            # First try with pre-trained weights
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
            print("✓ Loaded EfficientNet-B0 with pre-trained ImageNet weights")
        except ValueError as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
            print("Creating EfficientNet-B0 with random initialization for fine-tuning...")
            # Fallback to random weights if shape mismatch occurs
            base_model = EfficientNetB0(
                weights=None,  # No pre-trained weights
                include_top=False,
                pooling='avg',
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
        
        # Make the base model trainable for fine-tuning
        base_model.trainable = True
        
        # Create the classification model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.img_size[0], self.img_size[1], 3)),
            base_model,
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['LEARNING_RATE']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        print("Fine-tuning model summary:")
        model.summary()
        
        # Prepare data - ensure images are properly preprocessed
        def preprocess_for_training(images):
            """Preprocess images for EfficientNet training"""
            # Convert to float32 and preprocess for EfficientNet
            images = images.astype('float32')
            return preprocess_input(images)
        
        # Preprocess the images
        train_images_processed = preprocess_for_training(train_images)
        val_images_processed = preprocess_for_training(val_images)
        
        # Debug: Check image shapes and ranges
        print(f"Train images shape: {train_images_processed.shape}, dtype: {train_images_processed.dtype}")
        print(f"Train images range: [{train_images_processed.min():.3f}, {train_images_processed.max():.3f}]")
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images_processed, train_labels)
        ).batch(self.config['BATCH_SIZE']).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_images_processed, val_labels)
        ).batch(self.config['BATCH_SIZE']).prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Train model
        print("Starting fine-tuning training...")
        history = model.fit(
            train_dataset,
            epochs=self.config['EPOCHS'],
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Update feature extractor to use the fine-tuned model
        # Extract the feature extraction part from the trained model
        self.feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[0].output  # Output of EfficientNet base
        )
        
        print("Fine-tuning completed!")
        return history
        
    def plot_training_history(self, history, use_fine_tuning=False):
        """Plot training history for accuracy and loss"""
        if history is None:
            print("No training history available to plot.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_name = 'fine_tuning_history.png' if use_fine_tuning else 'training_history.png'
        plt.savefig(os.path.join(self.config['SAVE_DIR'], 'plots', plot_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved as {plot_name}")
    
    def train_svm(self, features, labels, use_grid_search=False):
        """Train SVM classifier on extracted features"""
        print("Training SVM classifier...")
        
        if use_grid_search:
            print("Performing grid search for SVM hyperparameters...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.0001],
                'kernel': ['rbf']
            }
            
            svm = SVC(probability=True, random_state=self.config['RANDOM_SEED'])
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(features, labels)
            
            self.svm_model = grid_search.best_estimator_
            print(f"Best SVM parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Use configured parameters
            svm_config = self.config.get('SVM', {})
            self.svm_model = SVC(
                kernel=svm_config.get('kernel', 'rbf'),
                C=svm_config.get('C', 1.0),
                gamma=svm_config.get('gamma', 'scale'),
                probability=svm_config.get('probability', True),
                random_state=self.config['RANDOM_SEED']
            )
            self.svm_model.fit(features, labels)
        
        # Save SVM model
        joblib.dump(self.svm_model, self.config['SVM_MODEL_PATH'])
        print(f"Saved SVM model to {self.config['SVM_MODEL_PATH']}")
        
        return self.svm_model
    
    def evaluate_model(self, test_features, test_labels, test_images=None):
        """Comprehensive model evaluation"""
        print("Evaluating model...")
        
        # Predictions
        y_pred = self.svm_model.predict(test_features)
        y_pred_proba = self.svm_model.predict_proba(test_features)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, y_pred)
        report = classification_report(test_labels, y_pred, target_names=self.classes, output_dict=True)
        cm = confusion_matrix(test_labels, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(test_labels, y_pred, target_names=self.classes))
        
        # Save results
        self.save_evaluation_results(report, cm, accuracy, y_pred_proba, test_labels)
        self.plot_evaluation_metrics(report, cm, y_pred_proba, test_labels)
        
        return report, cm, accuracy
    
    def save_evaluation_results(self, report, cm, accuracy, y_pred_proba, test_labels):
        """Save evaluation results to files"""
        
        # Save classification report
        with open(os.path.join(self.config['SAVE_DIR'], 'reports', 'classification_report.txt'), 'w') as f:
            f.write(f"Citrus Disease Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            for class_name in self.classes:
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
                f.write(f"  Recall: {report[class_name]['recall']:.4f}\n")
                f.write(f"  F1-Score: {report[class_name]['f1-score']:.4f}\n")
                f.write(f"  Support: {report[class_name]['support']}\n\n")
        
        # Save metrics summary CSV
        metrics_data = []
        for class_name in self.classes:
            metrics_data.append({
                'class': class_name,
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1_score': report[class_name]['f1-score'],
                'support': report[class_name]['support']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(self.config['SAVE_DIR'], 'reports', 'metrics_summary.csv'), index=False)
        
        # Save confusion matrix
        np.savetxt(os.path.join(self.config['SAVE_DIR'], 'reports', 'confusion_matrix.csv'), cm, delimiter=',', fmt='%d')
    
    def plot_evaluation_metrics(self, report, cm, y_pred_proba, test_labels):
        """Create and save evaluation plots"""
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['SAVE_DIR'], 'plots', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Normalized Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['SAVE_DIR'], 'plots', 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Metrics Bar Chart
        metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(self.classes))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in self.classes]
            plt.bar(x + i * width, values, width, label=metric.capitalize())
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Classification Metrics by Class')
        plt.xticks(x + width, self.classes, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['SAVE_DIR'], 'plots', 'metrics_by_class.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Class Distribution (if we have the original data stats)
        if hasattr(self, 'original_class_counts'):
            plt.figure(figsize=(10, 6))
            plt.bar(self.original_class_counts.keys(), self.original_class_counts.values())
            plt.title('Class Distribution in Dataset')
            plt.xlabel('Classes')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['SAVE_DIR'], 'plots', 'class_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_training_pipeline(self, use_fine_tuning=False, use_svm_grid_search=False):
        """Run complete training pipeline"""
        print("Starting Citrus Disease Classification Pipeline...")
        
        # 1. Load and preprocess data
        images, labels, class_counts = self.load_and_preprocess_data()
        self.original_class_counts = class_counts
        
        # 2. Visualize augmentation
        self.visualize_augmentation(images, labels)
        
        # 3. Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, 
            test_size=self.config['TEST_SPLIT'],
            stratify=labels,
            random_state=self.config['RANDOM_SEED']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['VALIDATION_SPLIT']/(1-self.config['TEST_SPLIT']),
            stratify=y_temp,
            random_state=self.config['RANDOM_SEED']
        )
        
        print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 4. Augment training data
        X_train_aug, y_train_aug = self.augment_images(X_train, y_train)
        
        # 5. Fine-tune or use pre-trained EfficientNet
        history = None
        if use_fine_tuning:
            print("Using fine-tuning approach...")
            history = self.fine_tune_efficientnet(X_train_aug, y_train_aug, X_val, y_val)
            
            # Extract features with fine-tuned model
            train_features = self.extract_features(X_train_aug)
            val_features = self.extract_features(X_val)
            test_features = self.extract_features(X_test)
            
            # Plot training history
            self.plot_training_history(history, use_fine_tuning=True)
        else:
            print("Using pre-trained EfficientNet for feature extraction...")
            # Create feature extractor
            self.create_feature_extractor(fine_tune=False)
            
            # Extract features
            train_features = self.extract_features(X_train_aug)
            val_features = self.extract_features(X_val)
            test_features = self.extract_features(X_test)
        
        # Save features
        np.save(self.config['FEATURE_CACHE_PATH'] + '_train.npy', train_features)
        np.save(self.config['FEATURE_CACHE_PATH'] + '_val.npy', val_features)
        np.save(self.config['FEATURE_CACHE_PATH'] + '_test.npy', test_features)
        np.save(self.config['FEATURE_CACHE_PATH'] + '_train_labels.npy', y_train_aug)
        np.save(self.config['FEATURE_CACHE_PATH'] + '_test_labels.npy', y_test)
        
        # 6. Train SVM on combined train + val features
        combined_features = np.concatenate([train_features, val_features], axis=0)
        combined_labels = np.concatenate([y_train_aug, y_val], axis=0)
        
        self.train_svm(combined_features, combined_labels, use_grid_search=use_svm_grid_search)
        
        # 7. Evaluate on test set
        report, cm, accuracy = self.evaluate_model(test_features, y_test)
        
        # 8. Save summary
        self.save_training_summary(accuracy, report, use_fine_tuning)
        
        self.is_trained = True
        print("Training pipeline completed successfully!")
        return accuracy
    
    def save_training_summary(self, accuracy, report, use_fine_tuning):
        """Save training summary JSON"""
        summary = {
            'model_architecture': 'EfficientNet-B0 + SVM',
            'test_accuracy': float(accuracy),
            'fine_tuning_used': use_fine_tuning,
            'classes': self.classes,
            'image_size': self.img_size,
            'saved_models': {
                'svm_model': self.config['SVM_MODEL_PATH'],
                'features': self.config['FEATURE_CACHE_PATH'] + '_*.npy'
            },
            'evaluation_metrics': {
                'macro_avg': {
                    'precision': float(report['macro avg']['precision']),
                    'recall': float(report['macro avg']['recall']),
                    'f1_score': float(report['macro avg']['f1-score'])
                },
                'weighted_avg': {
                    'precision': float(report['weighted avg']['precision']),
                    'recall': float(report['weighted avg']['recall']),
                    'f1_score': float(report['weighted avg']['f1-score'])
                }
            }
        }
        
        # Add per-class metrics
        for class_name in self.classes:
            summary[f'class_{class_name}'] = {
                'precision': float(report[class_name]['precision']),
                'recall': float(report[class_name]['recall']),
                'f1_score': float(report[class_name]['f1-score'])
            }
        
        with open(os.path.join(self.config['SAVE_DIR'], 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to {os.path.join(self.config['SAVE_DIR'], 'summary.json')}")

    # =========================================================================
    # INFERENCE METHODS
    # =========================================================================
    
    def load_trained_models(self):
        """Load pre-trained models for inference"""
        print("Loading trained models for inference...")
        
        # Load SVM model
        if os.path.exists(self.config['SVM_MODEL_PATH']):
            self.svm_model = joblib.load(self.config['SVM_MODEL_PATH'])
            print(f"Loaded SVM model from {self.config['SVM_MODEL_PATH']}")
        else:
            raise FileNotFoundError(f"SVM model not found at {self.config['SVM_MODEL_PATH']}")
        
        # Create feature extractor
        self.create_feature_extractor(fine_tune=False)
        
        # Load label encoder classes
        if os.path.exists(os.path.join(self.config['SAVE_DIR'], 'label_encoder_classes.npy')):
            self.le.classes_ = np.load(os.path.join(self.config['SAVE_DIR'], 'label_encoder_classes.npy'))
        else:
            # If no saved classes, use the default classes from config
            self.le.fit(self.classes)
            np.save(os.path.join(self.config['SAVE_DIR'], 'label_encoder_classes.npy'), self.le.classes_)
        
        self.is_trained = True
        print("All models loaded successfully!")
    
    def preprocess_single_image(self, image_path):
        """Preprocess a single image for inference"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, self.img_size)
        
        # Apply CLAHE if configured
        if self.config.get('APPLY_CLAHE', True):
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return img
    
    def predict_single_image(self, image_path, return_probabilities=True, visualize=True):
        """Predict disease class for a single image"""
        if not self.is_trained:
            self.load_trained_models()
        
        # Preprocess image
        img = self.preprocess_single_image(image_path)
        
        # Extract features
        features = self.extract_features(np.array([img]))
        
        # Make prediction
        if return_probabilities:
            probabilities = self.svm_model.predict_proba(features)[0]
            prediction = self.svm_model.predict(features)[0]
        else:
            prediction = self.svm_model.predict(features)[0]
            probabilities = None
        
        # Get class name
        class_name = self.le.inverse_transform([prediction])[0]
        
        # Visualize result if requested
        if visualize:
            self.visualize_prediction(img, class_name, probabilities, image_path)
        
        result = {
            'class_name': class_name,
            'class_id': int(prediction),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'confidence': float(probabilities[prediction]) if probabilities is not None else None
        }
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=True):
        """Predict disease classes for multiple images"""
        if not self.is_trained:
            self.load_trained_models()
        
        results = []
        processed_images = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Preprocess image
                img = self.preprocess_single_image(image_path)
                processed_images.append(img)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        if not processed_images:
            return results
        
        # Extract features for all images
        features = self.extract_features(np.array(processed_images))
        
        # Make predictions
        if return_probabilities:
            all_probabilities = self.svm_model.predict_proba(features)
            all_predictions = self.svm_model.predict(features)
        else:
            all_predictions = self.svm_model.predict(features)
            all_probabilities = [None] * len(all_predictions)
        
        # Compile results
        for i, (image_path, prediction, probabilities) in enumerate(zip(image_paths, all_predictions, all_probabilities)):
            if 'error' not in results[i] if i < len(results) else True:
                class_name = self.le.inverse_transform([prediction])[0]
                
                result = {
                    'image_path': image_path,
                    'class_name': class_name,
                    'class_id': int(prediction),
                    'probabilities': probabilities.tolist() if probabilities is not None else None,
                    'confidence': float(probabilities[prediction]) if probabilities is not None else None
                }
                results.append(result)
        
        return results
    
    def visualize_prediction(self, image, class_name, probabilities, image_path=None):
        """Visualize prediction result"""
        plt.figure(figsize=(12, 5))
        
        # Show original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Input Image\nPredicted: {class_name}')
        plt.axis('off')
        
        # Show probability distribution
        if probabilities is not None:
            plt.subplot(1, 2, 2)
            y_pos = np.arange(len(self.classes))
            plt.barh(y_pos, probabilities, align='center', alpha=0.7)
            plt.yticks(y_pos, self.classes)
            plt.xlabel('Probability')
            plt.title('Class Probabilities')
            plt.xlim(0, 1)
            
            # Highlight the predicted class
            predicted_idx = self.le.transform([class_name])[0]
            plt.barh(predicted_idx, probabilities[predicted_idx], color='red', alpha=0.8)
        
        plt.tight_layout()
        
        # Save or show
        if image_path:
            save_path = os.path.join(
                self.config['SAVE_DIR'], 
                'inference_results', 
                f"prediction_{os.path.basename(image_path)}"
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Prediction visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_inference_report(self, results, output_path=None):
        """Generate a comprehensive inference report"""
        if output_path is None:
            output_path = os.path.join(self.config['SAVE_DIR'], 'inference_results', 'inference_report.csv')
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save CSV report
        df.to_csv(output_path, index=False)
        
        # Generate summary statistics
        if len(results) > 0 and 'class_name' in results[0]:
            class_counts = df['class_name'].value_counts()
            
            print("\n" + "="*50)
            print("INFERENCE SUMMARY")
            print("="*50)
            print(f"Total images processed: {len(results)}")
            print("\nClass distribution:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} images")
            
            if 'confidence' in df.columns:
                avg_confidence = df['confidence'].mean()
                print(f"\nAverage confidence: {avg_confidence:.4f}")
        
        print(f"\nDetailed report saved to: {output_path}")
        return df

def load_config(config_path):
    """Load configuration from JSON file with better error handling"""
    # If config_path is relative, make it relative to the script directory
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            f"Please make sure 'config.json' is in the same directory as the script.\n"
            f"Script location: {os.path.dirname(os.path.abspath(__file__))}"
        )
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Successfully loaded config from: {config_path}")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading config file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Citrus Disease Classifier Training & Inference')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='Path to configuration file')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Use fine-tuning instead of feature extraction')
    parser.add_argument('--svm_grid_search', action='store_true',
                       help='Use grid search for SVM hyperparameters')
    parser.add_argument('--dataset_dir', type=str,
                       help='Override dataset directory from config')
    
    # Inference arguments
    parser.add_argument('--predict', type=str,
                       help='Path to single image for prediction')
    parser.add_argument('--predict_batch', type=str, nargs='+',
                       help='Paths to multiple images for batch prediction')
    parser.add_argument('--predict_dir', type=str,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable prediction visualization')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override dataset directory if provided
    if args.dataset_dir:
        config['DATASET_DIR'] = args.dataset_dir
    
    # Create classifier
    classifier = CitrusDiseaseClassifier(config)
    
    # Run inference if requested
    if args.predict or args.predict_batch or args.predict_dir:
        print("Running inference...")
        
        # Collect image paths
        image_paths = []
        
        if args.predict:
            image_paths.append(args.predict)
        
        if args.predict_batch:
            image_paths.extend(args.predict_batch)
        
        if args.predict_dir:
            if os.path.exists(args.predict_dir):
                for img_file in os.listdir(args.predict_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(args.predict_dir, img_file))
            else:
                print(f"Warning: Directory {args.predict_dir} not found.")
        
        if not image_paths:
            print("No valid images found for inference.")
            return
        
        # Run prediction
        if len(image_paths) == 1:
            # Single image prediction
            result = classifier.predict_single_image(
                image_paths[0], 
                visualize=not args.no_visualize
            )
            print(f"\nPrediction Result:")
            print(f"  Image: {os.path.basename(image_paths[0])}")
            print(f"  Predicted Class: {result['class_name']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            
            if result['probabilities']:
                print(f"  All probabilities:")
                for class_name, prob in zip(classifier.classes, result['probabilities']):
                    print(f"    {class_name}: {prob:.4f}")
        
        else:
            # Batch prediction
            results = classifier.predict_batch(image_paths, return_probabilities=True)
            classifier.generate_inference_report(results)
    
    else:
        # Run training
        accuracy = classifier.run_training_pipeline(
            use_fine_tuning=args.fine_tune,
            use_svm_grid_search=args.svm_grid_search
        )
        
        print(f"\n{'='*50}")
        print(f"Training completed! Final Test Accuracy: {accuracy:.4f}")
        print(f"Models and results saved to: {config['SAVE_DIR']}")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()