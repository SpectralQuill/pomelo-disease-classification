# -*- coding: utf-8 -*-
"""Citrus Maxima (Pomelo) Fruit Disease Classification using EfficientNet-B0 + SVM"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PomeloDiseaseClassifier:
    def __init__(self, config_path="Models/EfficientNetB0+SVM/config.yaml"):
        """Initialize the classifier with configuration"""
        self.load_config(config_path)
        self.create_directories()
        self.setup_gpu()
        self.class_names = self.config['classes']
        self.num_classes = len(self.class_names)
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        print("✅ Configuration loaded successfully")
        
    def create_directories(self):
        """Create output directories"""
        base_path = self.config['data']['output_path']
        self.dirs = {
            'plots': os.path.join(base_path, 'plots'),
            'metrics': os.path.join(base_path, 'metrics'),
            'weights': os.path.join(base_path, 'weights'),
            'logs': os.path.join(base_path, 'logs'),
            'samples': os.path.join(base_path, 'samples'),
            'features': os.path.join(base_path, 'features')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        print("✅ Output directories created")
        
    def setup_gpu(self):
        """Setup GPU configuration"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU setup complete: {len(gpus)} GPU(s) available")
            except RuntimeError as e:
                print(f"❌ GPU setup error: {e}")
        else:
            print("ℹ️ No GPU available, using CPU")
    
    def load_dataset(self):
        """Load and organize the pomelo disease dataset"""
        dataset_path = self.config['data']['dataset_path']
        image_paths = []
        labels = []
        
        print("📁 Loading dataset...")
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Warning: Directory {class_path} not found")
                continue
                
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith('.png'):
                    img_path = os.path.join(class_path, img_file)
                    image_paths.append(img_path)
                    labels.append(class_idx)
        
        if not image_paths:
            raise ValueError("❌ No images found in dataset directory")
            
        print(f"✅ Loaded {len(image_paths)} images across {self.num_classes} classes")
        return image_paths, labels
    
    def plot_data_distribution(self, train_labels, val_labels, test_labels):
        """Plot data distribution pie chart"""
        sizes = [len(train_labels), len(val_labels), len(test_labels)]
        labels = ['Training (80%)', 'Validation (10%)', 'Testing (10%)']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Data Partition Distribution')
        
        # Save the plot
        plt.savefig(os.path.join(self.dirs['plots'], 'data_distribution.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("✅ Data distribution plot saved")
    
    def custom_augmentation(self, image):
        """Apply custom data augmentation techniques using pure TensorFlow operations"""
        # Random rotation
        def apply_rotation(img):
            angle = tf.random.uniform([], -30, 30) * np.pi / 180  # Convert to radians
            return tf.keras.preprocessing.image.apply_affine_transform(
                img, theta=angle, channel_axis=2, fill_mode='reflect'
            )
        
        image = tf.cond(
            tf.random.uniform([]) > 0.5,
            lambda: tf.py_function(apply_rotation, [image], tf.float32),
            lambda: image
        )
        
        # Gaussian blur using TensorFlow operations
        def apply_gaussian_blur(img):
            # Use separable convolution for Gaussian blur approximation
            kernel_size = 5
            sigma = 1.0
            
            # Create 1D Gaussian kernel
            x = tf.range(kernel_size, dtype=tf.float32) - (kernel_size - 1) / 2.0
            kernel = tf.exp(-0.5 * tf.square(x / sigma))
            kernel = kernel / tf.reduce_sum(kernel)
            kernel = tf.reshape(kernel, [kernel_size, 1, 1, 1])
            
            # Apply separable convolution
            blurred = tf.nn.separable_conv2d(
                tf.expand_dims(img, 0),  # Add batch dimension
                tf.tile(kernel, [1, 1, 3, 1]),  # Depthwise kernel
                tf.eye(3, batch_shape=[1, 1]),  # Pointwise kernel
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            return tf.squeeze(blurred, 0)  # Remove batch dimension
        
        image = tf.cond(
            tf.random.uniform([]) > 0.7,
            lambda: apply_gaussian_blur(image),
            lambda: image
        )
        
        # Salt and pepper noise using TensorFlow
        def apply_salt_pepper_noise(img):
            salt_vs_pepper = 0.5
            amount = 0.05
            
            # Generate random mask
            random_mask = tf.random.uniform(tf.shape(img)[:2])
            
            # Salt pixels
            salt_mask = random_mask < (amount * salt_vs_pepper)
            salt_mask = tf.expand_dims(salt_mask, -1)
            salt_mask = tf.tile(salt_mask, [1, 1, 3])
            
            # Pepper pixels  
            pepper_mask = (random_mask > (1 - amount * (1 - salt_vs_pepper))) & (random_mask <= 1.0)
            pepper_mask = tf.expand_dims(pepper_mask, -1)
            pepper_mask = tf.tile(pepper_mask, [1, 1, 3])
            
            # Apply noise
            img = tf.where(salt_mask, 1.0, img)  # Set to white
            img = tf.where(pepper_mask, 0.0, img)  # Set to black
            
            return img
        
        image = tf.cond(
            tf.random.uniform([]) > 0.8,
            lambda: apply_salt_pepper_noise(image),
            lambda: image
        )
        
        # Random adjustment of contrast
        image = tf.cond(
            tf.random.uniform([]) > 0.5,
            lambda: tf.image.random_contrast(image, 0.8, 1.2),
            lambda: image
        )
        
        # Random cropping
        def apply_random_crop(img):
            crop_size = self.config['model']['image_size'] - 20
            cropped = tf.image.random_crop(img, [crop_size, crop_size, 3])
            return tf.image.resize(cropped, [self.config['model']['image_size'], 
                                           self.config['model']['image_size']])
        
        image = tf.cond(
            tf.random.uniform([]) > 0.5,
            lambda: apply_random_crop(image),
            lambda: image
        )
        
        # Cutout using TensorFlow
        def apply_cutout(img):
            mask_size = 20
            h, w = tf.shape(img)[0], tf.shape(img)[1]
            
            # Random position
            x = tf.random.uniform([], 0, w - mask_size, dtype=tf.int32)
            y = tf.random.uniform([], 0, h - mask_size, dtype=tf.int32)
            
            # Create mask
            mask = tf.ones([mask_size, mask_size, 3])
            paddings = [[y, h - (y + mask_size)], [x, w - (x + mask_size)], [0, 0]]
            mask = tf.pad(mask, paddings, mode='CONSTANT', constant_values=1)
            mask = 1 - mask  # Invert mask
            
            return img * mask
        
        image = tf.cond(
            tf.random.uniform([]) > 0.7,
            lambda: apply_cutout(image),
            lambda: image
        )
        
        # Ensure image is properly shaped
        image.set_shape([self.config['model']['image_size'], 
                        self.config['model']['image_size'], 3])
        
        return image
    
    def preprocess_image(self, image_path, label, augment=False):
        """Preprocess single image"""
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [self.config['model']['image_size'], 
                                      self.config['model']['image_size']])
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        
        # Apply augmentation if training
        if augment:
            image = self.custom_augmentation(image)
        
        # EfficientNet preprocessing (scale to [-1, 1])
        image = image * 2.0 - 1.0
        
        return image, label
    
    def create_data_pipeline(self, image_paths, labels, augment=False):
        """Create TensorFlow data pipeline"""
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        if augment:
            dataset = dataset.map(
                lambda x, y: self.preprocess_image(x, y, augment=True),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.shuffle(len(image_paths))
        else:
            dataset = dataset.map(
                lambda x, y: self.preprocess_image(x, y, augment=False),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        dataset = dataset.batch(self.config['training']['batch_size'])
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def plot_augmentation_samples(self, dataset, num_samples=5):
        """Plot augmented image samples"""
        plt.figure(figsize=(15, 8))
        for batch_idx, (images, labels) in enumerate(dataset.take(1)):
            for i in range(min(num_samples, len(images))):
                plt.subplot(2, 5, i + 1)
                
                # Reverse preprocessing for visualization
                img = images[i].numpy()
                img = (img + 1.0) / 2.0  # Scale back to [0, 1]
                img = np.clip(img, 0, 1)
                
                plt.imshow(img)
                plt.title(f'Class: {self.class_names[labels[i]]}')
                plt.axis('off')
            
            plt.suptitle('Data Augmentation Samples')
            plt.savefig(os.path.join(self.dirs['samples'], 'augmentation_samples.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            break
        print("✅ Augmentation samples plot saved")
    
    def build_feature_extractor(self):
        """Build EfficientNet-B0 feature extractor"""
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.config['model']['image_size'], 
                        self.config['model']['image_size'], 3),
            pooling='avg'
        )
        
        # Fine-tune the last few layers
        base_model.trainable = False
        for layer in base_model.layers[-20:]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
        
        print("✅ EfficientNet-B0 feature extractor built")
        print(f"📊 Trainable layers: {sum([layer.trainable for layer in base_model.layers])}")
        print(f"📊 Total layers: {len(base_model.layers)}")
        
        return base_model
    
    def plot_model_architecture(self, model):
        """Plot and save model architecture"""
        try:
            keras.utils.plot_model(
                model,
                to_file=os.path.join(self.dirs['plots'], 'model_architecture.png'),
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB'
            )
            print("✅ Model architecture plot saved")
        except Exception as e:
            print(f"⚠️ Could not plot model architecture: {e}")
    
    def extract_features(self, model, dataset):
        """Extract features using EfficientNet-B0"""
        features = []
        labels_list = []
        
        print("🔍 Extracting features...")
        for images, labels in tqdm(dataset, desc="Feature Extraction"):
            batch_features = model(images, training=False)
            features.extend(batch_features.numpy())
            labels_list.extend(labels.numpy())
        
        return np.array(features), np.array(labels_list)
    
    def train_svm(self, features, labels):
        """Train SVM classifier"""
        print("🤖 Training SVM classifier...")
        
        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Train SVM
        self.svm = SVC(
            C=self.config['svm']['C'],
            kernel=self.config['svm']['kernel'],
            gamma=self.config['svm']['gamma'],
            probability=True,
            random_state=42
        )
        
        self.svm.fit(features_scaled, labels)
        print("✅ SVM training completed")
        
        return self.svm
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(15, 6))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], 'o-', label='Train Accuracy', color='tab:blue')
        plt.plot(history.history['val_accuracy'], 's-', label='Validation Accuracy', color='tab:orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Model Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], 'o-', label='Train Loss', color='tab:red')
        plt.plot(history.history['val_loss'], 's-', label='Validation Loss', color='tab:green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['plots'], 'training_history.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("✅ Training history plot saved")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        plt.savefig(os.path.join(self.dirs['plots'], 'confusion_matrix.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("✅ Confusion matrix plot saved")
        
        return cm
    
    def plot_feature_distribution(self, features, labels):
        """Plot feature distribution using PCA"""
        from sklearn.decomposition import PCA
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Feature Distribution (PCA)')
        
        # Create legend
        for i, class_name in enumerate(self.class_names):
            plt.scatter([], [], c=[plt.cm.viridis(i / len(self.class_names))], 
                       label=class_name, alpha=0.7)
        plt.legend()
        
        plt.savefig(os.path.join(self.dirs['plots'], 'feature_distribution.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("✅ Feature distribution plot saved")
    
    def evaluate_model(self, y_true, y_pred, y_proba=None):
        """Evaluate model performance and save metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save metrics to JSON
        with open(os.path.join(self.dirs['metrics'], 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save metrics summary to text file
        with open(os.path.join(self.dirs['metrics'], 'metrics_summary.txt'), 'w') as f:
            f.write("Pomelo Fruit Disease Classification - Evaluation Metrics\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, target_names=self.class_names))
        
        print("✅ Evaluation metrics saved")
        return metrics

    def train_pipeline(self):
        """Main training pipeline"""
        print("🚀 Starting Pomelo Fruit Disease Classification Pipeline...")
        
        # Load dataset
        image_paths, labels = self.load_dataset()
        
        # Split data (80% train, 10% validation, 10% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, 
            test_size=self.config['data']['test_ratio'],
            random_state=42,
            stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['data']['val_ratio'] / (1 - self.config['data']['test_ratio']),
            random_state=42,
            stratify=y_temp
        )
        
        print(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Plot data distribution
        self.plot_data_distribution(y_train, y_val, y_test)
        
        # Create data pipelines
        train_dataset = self.create_data_pipeline(X_train, y_train, augment=True)
        val_dataset = self.create_data_pipeline(X_val, y_val, augment=False)
        test_dataset = self.create_data_pipeline(X_test, y_test, augment=False)
        
        # Plot augmentation samples
        self.plot_augmentation_samples(train_dataset)
        
        # Build feature extractor
        feature_extractor = self.build_feature_extractor()
        
        # Plot model architecture
        self.plot_model_architecture(feature_extractor)
        
        # Build fine-tuning model
        inputs = keras.Input(shape=(self.config['model']['image_size'], 
                                  self.config['model']['image_size'], 3))
        x = feature_extractor(inputs)
        x = keras.layers.Dropout(self.config['model']['dropout_rate'])(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(self.config['training']['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=self.config['training']['reduce_lr_patience'],
                factor=0.5,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.dirs['weights'], 'best_model.keras'),
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                os.path.join(self.dirs['logs'], 'training_log.csv')
            )
        ]
        
        # Train the model
        print("🎯 Starting model training...")
        history = model.fit(
            train_dataset,
            epochs=self.config['training']['epochs'],
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        # Extract features for SVM training
        print("🔍 Extracting features for SVM...")
        train_features, train_labels_feat = self.extract_features(feature_extractor, train_dataset)
        val_features, val_labels_feat = self.extract_features(feature_extractor, val_dataset)
        test_features, test_labels_feat = self.extract_features(feature_extractor, test_dataset)
        
        # Combine train and val for SVM training
        svm_features = np.vstack([train_features, val_features])
        svm_labels = np.hstack([train_labels_feat, val_labels_feat])
        
        # Train SVM
        self.train_svm(svm_features, svm_labels)
        
        # Evaluate on test set
        test_features_scaled = self.scaler.transform(test_features)
        y_pred_svm = self.svm.predict(test_features_scaled)
        y_proba_svm = self.svm.predict_proba(test_features_scaled)
        
        # Generate plots and metrics
        self.plot_confusion_matrix(test_labels_feat, y_pred_svm)
        self.plot_feature_distribution(test_features, test_labels_feat)
        metrics = self.evaluate_model(test_labels_feat, y_pred_svm, y_proba_svm)
        
        # Print results
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED - FINAL RESULTS")
        print("="*60)
        print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
        print(f"🎯 Precision: {metrics['precision']:.4f}")
        print(f"🔍 Recall: {metrics['recall']:.4f}")
        print(f"⭐ F1-Score: {metrics['f1_score']:.4f}")
        print("="*60)
        
        # Save final model
        model.save(os.path.join(self.dirs['weights'], 'final_model.keras'))
        
        # Save feature extractor and SVM
        feature_extractor.save(os.path.join(self.dirs['weights'], 'feature_extractor.keras'))
        
        import joblib
        joblib.dump(self.svm, os.path.join(self.dirs['weights'], 'svm_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.dirs['weights'], 'feature_scaler.pkl'))
        
        print("💾 All models and weights saved successfully!")
        
        return model, self.svm, metrics
    
    def predict_image(self, image_path):
        """Predict the class of a single pomelo image"""
        import joblib
        
        if not hasattr(self, 'svm') or not hasattr(self, 'scaler'):
            # Load models if not already loaded
            feature_extractor = keras.models.load_model(
                os.path.join(self.dirs['weights'], 'feature_extractor.keras')
            )
            self.svm = joblib.load(os.path.join(self.dirs['weights'], 'svm_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.dirs['weights'], 'feature_scaler.pkl'))
        else:
            feature_extractor = self.build_feature_extractor()
        
        # Preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.config['model']['image_size'], 
                            self.config['model']['image_size']))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = image_array * 2.0 - 1.0  # EfficientNet preprocessing
        
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Extract features
        features = feature_extractor.predict(image_batch, verbose=0)
        
        # Scale features and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.svm.predict(features_scaled)[0]
        probability = self.svm.predict_proba(features_scaled)[0]
        
        # Get confidence score
        confidence = probability[prediction]
        
        # Display result
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(image)
        plt.title(f'Prediction: {self.class_names[prediction]} (Confidence: {confidence:.2f})')
        plt.axis('off')
        
        plt.subplot(2, 1, 2)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        bars = plt.bar(self.class_names, probability, color=colors)
        plt.xticks(rotation=45)
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        
        # Add probability values on bars
        for bar, prob in zip(bars, probability):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return self.class_names[prediction], confidence

def main():
    """Main execution function"""
    try:
        # Initialize classifier
        classifier = PomeloDiseaseClassifier()
        
        # Run training pipeline
        model, svm, metrics = classifier.train_pipeline()
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved in: {classifier.config['data']['output_path']}")
        
        # Example prediction (uncomment to test)
        # if os.path.exists('test_image.png'):
        #     prediction, confidence = classifier.predict_image('test_image.png')
        #     print(f"🧪 Test Prediction: {prediction} (Confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()