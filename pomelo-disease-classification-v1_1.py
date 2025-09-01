import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import cv2
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Class names
class_names = ['Anthracnose', 'Borer', 'Canker', 'Healthy', 'Mites']

class PomeloDiseaseClassifier:
    def __init__(self, img_size=224, batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.base_model = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.svm_classifier = None
        
    def load_and_preprocess_data(self, data_dir):
        """
        Load images from directory structure: data_dir/class_name/*.jpg
        """
        print("Loading and preprocessing images...")
        
        # Get class names from subdirectories
        
        images = []
        labels = []
        
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_file in tqdm(os.listdir(class_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    
                    # Read and preprocess image
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = preprocess_input(img)  # EfficientNet specific preprocessing
                        
                        images.append(img)
                        labels.append(class_name)
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images with {len(class_names)} classes")
        print(f"Classes: {class_names}")
        
        return images, labels
    
    def create_feature_extractor(self):
        """Create EfficientNetB0 base model for feature extraction"""
        print("Creating EfficientNetB0 feature extractor...")
        
        # Load pre-trained EfficientNetB0 without top layers
        self.base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3),
            pooling=None
        )
        
        # Freeze base model layers
        self.base_model.trainable = False
        
        # Create feature extraction model
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
        x = self.base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        
        self.model = Model(inputs, x)
        
        return self.model
    
    def extract_features(self, images):
        """Extract features using EfficientNetB0"""
        print("Extracting features...")
        features = self.model.predict(images, verbose=1)
        return features
    
    def train_svm(self, features, labels):
        """Train SVM classifier on extracted features"""
        print("Training SVM classifier...")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # Train SVM with hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.0001],
            'kernel': ['rbf', 'linear']
        }
        
        self.svm_classifier = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        self.svm_classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Best SVM parameters: {self.svm_classifier.best_params_}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return X_test, y_test, y_pred
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def predict_new_image(self, image_path):
        """Predict class for a new image"""
        if self.model is None or self.svm_classifier is None:
            raise ValueError("Model not trained yet. Please train first.")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        features = self.model.predict(img, verbose=0)
        scaled_features = self.scaler.transform(features)
        
        # Predict
        prediction = self.svm_classifier.predict(scaled_features)
        probability = self.svm_classifier.predict_proba(scaled_features)
        
        predicted_class = self.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probability)
        
        return predicted_class, confidence, probability[0]
    
    def save_model(self, model_path):
        """Save the entire pipeline"""
        import joblib
        model_data = {
            'svm_classifier': self.svm_classifier,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'class_names': class_names,
            'img_size': self.img_size
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a saved model"""
        import joblib
        model_data = joblib.load(model_path)
        self.svm_classifier = model_data['svm_classifier']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.img_size = model_data['img_size']
        
        # Recreate feature extractor
        self.create_feature_extractor()
        print(f"Model loaded from {model_path}")

# Example usage and training pipeline
def main():
    # Initialize classifier
    classifier = PomeloDiseaseClassifier(img_size=224, batch_size=32)
    
    # Load your dataset - replace with your dataset path
    data_dir = "./dataset"  # You'll provide this
    
    try:
        # Load and preprocess data
        images, labels = classifier.load_and_preprocess_data(data_dir)
        
        # Create feature extractor
        classifier.create_feature_extractor()
        
        # Extract features
        features = classifier.extract_features(images)
        
        # Train SVM
        X_test, y_test, y_pred = classifier.train_svm(features, labels)
        
        # Plot confusion matrix
        classifier.plot_confusion_matrix(y_test, y_pred)
        
        # Save model
        classifier.save_model("pomelo_disease_classifier.pkl")
        
        # Example prediction on new image
        # test_image_path = "path/to/test/image.jpg"
        # predicted_class, confidence, probabilities = classifier.predict_new_image(test_image_path)
        # print(f"Predicted: {predicted_class}, Confidence: {confidence:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure:")
        print("1. The dataset path is correct")
        print("2. Dataset structure is: data_dir/class_name/*.jpg")
        print("3. You have sufficient memory for processing")

if __name__ == "__main__":
    main()