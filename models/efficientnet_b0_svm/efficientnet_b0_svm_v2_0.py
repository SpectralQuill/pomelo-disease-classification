#!/usr/bin/env python3
"""
EffB0SVMTrainer - Monolithic object-oriented pipeline:
- Loads config.yaml
- Creates output/analysis/weights/false_predictions structure
- Stratified split (70/20/10)
- Augmentation (albumentations + optional Keras augmentation)
- Fine-tune EfficientNet-B0 (temporary classifier head), stop on plateau
- Remove head, use feature extractor (GlobalAveragePooling2D -> 1280-dim)
- Cache features
- Scale features and train / tune SVM (GridSearchCV)
- Evaluate: confusion matrix, classification report, training history plot
- Save false_predictions.json in analysis folder
"""

import argparse
import os
import sys
import shutil
import random
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import urllib.request

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Albumentations for augmentations
import albumentations as A

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# For reproducibility
import numpy.random as npr

plt.switch_backend("agg")  # headless-friendly

# ---------------------------
# Helper functions
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    npr.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def timestamp_now(fmt="%Y%m%d%H%M%S"):
    return datetime.now().strftime(fmt)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"Downloading weights from {url} -> {dest} ...")
    urllib.request.urlretrieve(url, str(dest))
    return dest

def load_image_paths_from_folder(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    return sorted(paths)

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def plot_and_save_pie(class_counts: Dict[str,int], outpath: Path, title="Partition distribution"):
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_and_save_bar(class_counts_before: Dict[str,int], class_counts_after: Dict[str,int], outpath: Path):
    labels = list(class_counts_before.keys())
    before = [class_counts_before[k] for k in labels]
    after = [class_counts_after.get(k,0) for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, before, width, label="Before")
    plt.bar(x + width/2, after, width, label="After")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_and_save_history(history, outpath: Path):
    plt.figure(figsize=(8,4))
    if "accuracy" in history.history:
        plt.plot(history.history.get("accuracy", []), label="train_acc")
        plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    if "loss" in history.history:
        plt.plot(history.history.get("loss", []), label="train_loss")
        plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_and_save_confusion_matrix(y_true, y_pred, labels, outpath: Path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label', title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close()

# ---------------------------
# Main Trainer Class
# ---------------------------

class EffB0SVMTrainer:
    def __init__(self, config_path: str):
        # Load YAML
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Set seed
        self.seed = int(self.cfg.get("general", {}).get("seed", 42))
        set_seed(self.seed)

        # Paths
        self.dataset_dir = Path(self.cfg["general"]["dataset_dir"])
        self.augments_dir = Path(self.cfg["general"]["augments_dir"])
        self.outputs_root = Path(self.cfg["general"]["outputs_dir"])
        self.timestamp = timestamp_now(self.cfg["general"].get("timestamp_format", "%Y%m%d%H%M%S"))
        self.run_name = f"effb0svm_{self.timestamp}"
        self.run_dir = self.outputs_root / self.run_name

        # Subfolders
        self.analysis_dir = self.run_dir / "analysis"
        self.false_preds_dir = self.run_dir / "false_predictions"
        self.weights_dir = self.run_dir / "weights"
        self.create_output_dirs()

        # Config short-hands
        self.image_size = int(self.cfg["general"].get("image_size", 224))
        self.batch_size = int(self.cfg["general"].get("batch_size", 32))
        self.num_workers = int(self.cfg["general"].get("num_workers", 8))

        # Splits
        self.train_frac = float(self.cfg["splits"].get("train_frac", 0.7))
        self.val_frac = float(self.cfg["splits"].get("val_frac", 0.2))
        self.test_frac = float(self.cfg["splits"].get("test_frac", 0.1))
        assert abs(self.train_frac + self.val_frac + self.test_frac - 1.0) < 1e-6, "Splits must sum to 1.0"

        # Augmentation config
        self.aug_cfg = self.cfg.get("augmentation", {})
        self.per_class_target = self.aug_cfg.get("per_class_target", None)
        self.max_aug_per_source = int(self.aug_cfg.get("max_aug_per_source", 5))

        # Model config
        self.model_cfg = self.cfg.get("model", {})
        self.weights_url = self.model_cfg.get("efficientnet", {}).get("weights_url", None)
        self.unfreeze_top_n = int(self.model_cfg.get("efficientnet", {}).get("unfreeze_top_n_layers", 0))
        self.fine_tune_cfg = self.model_cfg.get("efficientnet", {}).get("fine_tune", {})

        # Feature extraction config
        self.feat_cfg = self.cfg.get("feature_extraction", {})
        self.cache_features = bool(self.feat_cfg.get("cache_features", True))
        self.features_filename = self.feat_cfg.get("features_filename", "features_{split}.npz")
        self.pooling = self.feat_cfg.get("pooling", "avg")
        self.flatten_map = bool(self.feat_cfg.get("flatten_map", False))

        # SVM config
        self.svm_cfg = self.cfg.get("svm", {})
        self.scale_features = bool(self.svm_cfg.get("scale_features", True))
        self.scaler_name = self.svm_cfg.get("scaler", "StandardScaler")
        self.grid_cfg = self.svm_cfg.get("grid_search", {})
        self.final_svm_cfg = self.svm_cfg.get("final", {})

        # Analysis options
        self.analysis_cfg = self.cfg.get("analysis", {})
        self.save_history_plot = bool(self.analysis_cfg.get("save_history_plot", True))

        # Internal placeholders
        self.class_names = []
        self.label_encoder = LabelEncoder()
        self.train_files = []
        self.val_files = []
        self.test_files = []
        self.train_labels = []
        self.val_labels = []
        self.test_labels = []

        # Models / objects
        self.base_model = None
        self.feature_extractor = None
        self.history = None
        self.scaler = None
        self.svm = None

        # Save config copy
        shutil.copy(config_path, self.run_dir / "config.yaml")

    def create_output_dirs(self):
        ensure_dir(self.run_dir)
        ensure_dir(self.analysis_dir)
        ensure_dir(self.false_preds_dir)
        ensure_dir(self.weights_dir)

    # ---------------------------
    # Data utilities
    # ---------------------------

    def collect_dataset(self):
        # Expect dataset_dir to have class subfolders
        classes = [p for p in self.dataset_dir.iterdir() if p.is_dir()]
        classes = sorted(classes)
        if not classes:
            raise FileNotFoundError(f"No class subfolders found in {self.dataset_dir}")
        self.class_names = [p.name for p in classes]
        all_files, all_labels = [], []
        for cname in self.class_names:
            folder = self.dataset_dir / cname
            files = load_image_paths_from_folder(folder)
            all_files.extend(files)
            all_labels.extend([cname] * len(files))
        # Convert to arrays
        self.df = pd.DataFrame({"filepath": [str(p) for p in all_files], "class": all_labels})
        print(f"Collected {len(self.df)} images across {len(self.class_names)} classes.")
        # Save class mapping
        save_json(self.weights_dir / "class_mapping.json", {i: n for i, n in enumerate(self.class_names)})

    def stratified_split(self):
        # First split train and temp (val+test)
        X = self.df["filepath"].values
        y = self.df["class"].values
        stratify = y
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=self.train_frac, stratify=stratify, random_state=self.seed
        )
        # Compute relative val fraction within temp
        val_rel = self.val_frac / (self.val_frac + self.test_frac)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_rel, stratify=y_temp, random_state=self.seed
        )
        # Save lists
        self.train_files = list(X_train)
        self.val_files = list(X_val)
        self.test_files = list(X_test)
        self.train_labels = list(y_train)
        self.val_labels = list(y_val)
        self.test_labels = list(y_test)

        # Save file lists
        pd.DataFrame({"filepath": self.train_files, "class": self.train_labels}).to_csv(self.run_dir / "train_files.csv", index=False)
        pd.DataFrame({"filepath": self.val_files, "class": self.val_labels}).to_csv(self.run_dir / "val_files.csv", index=False)
        pd.DataFrame({"filepath": self.test_files, "class": self.test_labels}).to_csv(self.run_dir / "test_files.csv", index=False)

        # Pie
        counts = dict(pd.Series(self.train_labels).value_counts())
        plot_and_save_pie(counts, self.analysis_dir / "partition_pie.png", title="Training partition distribution (train set)")

        print(f"Split into train={len(self.train_files)}, val={len(self.val_files)}, test={len(self.test_files)}")

    # ---------------------------
    # Augmentation
    # ---------------------------

    def build_alb_transforms(self):
        a_cfg = self.aug_cfg.get("albumentations", {})
        transforms = []
        # Random brightness/contrast
        if a_cfg.get("brightness_contrast", {}).get("enabled", False):
            params = a_cfg.get("brightness_contrast", {})
            transforms.append(A.RandomBrightnessContrast(brightness_limit=params.get("brightness_limit", 0.2),
                                                         contrast_limit=params.get("contrast_limit", 0.2),
                                                         p=params.get("p", 0.5)))
        # Gaussian blur
        if a_cfg.get("gaussian_blur", {}).get("enabled", False):
            params = a_cfg.get("gaussian_blur", {})
            transforms.append(A.GaussianBlur(blur_limit=params.get("blur_limit", 3), p=params.get("p", 0.3)))
        # Salt and pepper
        if a_cfg.get("salt_and_pepper", {}).get("enabled", False):
            params = a_cfg.get("salt_and_pepper", {})
            transforms.append(A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.5), p=params.get("p",0.2)))
        # Noise (gaussian)
        if a_cfg.get("noise_scaling", {}).get("enabled", False):
            params = a_cfg.get("noise_scaling", {})
            transforms.append(A.GaussNoise(var_limit=params.get("var_limit", 10.0), p=params.get("p", 0.3)))
        # Rotation
        if a_cfg.get("rotation", {}).get("enabled", False):
            params = a_cfg.get("rotation", {})
            transforms.append(A.Rotate(limit=params.get("limit", 25), p=params.get("p", 0.5)))
        # Add always resize to desired image_size
        transforms.append(A.Resize(self.image_size, self.image_size))
        # Compose
        return A.Compose(transforms)

    def generate_augmented_set(self):
        """
        Generates an augmented and balanced training set per class.
        - Augmentations are applied on the original full-resolution images.
        - Saved images (originals + augmented) are resized to the configured input size to save disk space.
        """
        ensure_dir(self.augments_dir)
        for cls in self.class_names:
            ensure_dir(self.augments_dir / cls)

        # Count occurrences per class in train set
        train_counts = pd.Series(self.train_labels).value_counts().to_dict()

        # Determine target per-class amount
        if self.per_class_target is None:
            target = max(train_counts.values())
        else:
            target = int(self.per_class_target)

        per_class_after = {}
        alb_transform = self.build_alb_transforms()
        keras_aug_cfg = self.aug_cfg.get("keras_augment", {})
        target_size = (
            (self.image_size, self.image_size)
            if isinstance(self.image_size, int)
            else tuple(self.image_size)
        )

        datagen = ImageDataGenerator(
            horizontal_flip=keras_aug_cfg.get("horizontal_flip", False),
            vertical_flip=keras_aug_cfg.get("vertical_flip", False),
            rotation_range=keras_aug_cfg.get("rotation_range", 0),
            width_shift_range=keras_aug_cfg.get("width_shift_range", 0.0),
            height_shift_range=keras_aug_cfg.get("height_shift_range", 0.0),
            shear_range=keras_aug_cfg.get("shear_range", 0.0),
            zoom_range=keras_aug_cfg.get("zoom_range", 0.0),
            fill_mode='nearest'
        )

        for cls in self.class_names:
            cls_train_files = [f for f, l in zip(self.train_files, self.train_labels) if l == cls]
            n_existing = len(cls_train_files)
            n_target = target
            n_needed = max(0, n_target - n_existing)
            saved = 0
            out_cls_dir = self.augments_dir / cls
            print(f"[Augment] Class '{cls}': existing={n_existing}, target={n_target}, needed={n_needed}")

            # Step 1️⃣ — Generate augmentations first (using originals)
            if n_needed > 0:
                per_source = min(self.max_aug_per_source, math.ceil(n_needed / max(1, n_existing)))
                aug_per_image = {src: 0 for src in cls_train_files}

                while saved < n_needed:
                    # Choose next source that still has augment quota left
                    available = [src for src, count in aug_per_image.items() if count < per_source]
                    if not available:
                        break  # all reached max
                    src = random.choice(available)

                    img = cv2.imread(src)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Albumentations
                    augmented = alb_transform(image=img_rgb)
                    aug_img = augmented["image"]

                    # Keras datagen
                    x = np.expand_dims(aug_img.astype("float32"), 0)
                    it = datagen.flow(x, batch_size=1, shuffle=False)
                    aug_img = next(it)[0].astype("uint8")

                    # Resize to target size
                    aug_img_resized = cv2.resize(aug_img, target_size, interpolation=cv2.INTER_AREA)

                    # Save
                    fname = f"aug_{saved}_{Path(src).stem}.jpg"
                    out_path = out_cls_dir / fname
                    Image.fromarray(aug_img_resized).save(out_path)
                    saved += 1
                    aug_per_image[src] += 1

            # Step 2️⃣ — Copy & resize original images AFTER augmentation
            for src in cls_train_files:
                dst = out_cls_dir / Path(src).name
                if not dst.exists():
                    img = cv2.imread(src)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
                    Image.fromarray(img_resized).save(dst)

            per_class_after[cls] = len(list(out_cls_dir.glob("*")))

        # Step 3️⃣ — Save class distribution summary
        class_before = dict(pd.Series(self.train_labels).value_counts().to_dict())
        save_json(self.analysis_dir / "class_counts_before.json", class_before)
        save_json(self.analysis_dir / "class_counts_after.json", per_class_after)
        plot_and_save_bar(
            class_before, per_class_after,
            self.analysis_dir / "class_distribution_before_after.png"
        )
        print("Augmentation finished. Augmented images saved to:", str(self.augments_dir))

        # Step 4️⃣ — Update train lists to point to augments dir (balanced)
        new_train_files, new_train_labels = [], []
        for cls in self.class_names:
            files = sorted([
                str(p) for p in (self.augments_dir / cls).glob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])
            new_train_files.extend(files)
            new_train_labels.extend([cls] * len(files))
        self.train_files = new_train_files
        self.train_labels = new_train_labels

        # Step 5️⃣ — Save new train list CSV
        pd.DataFrame({
            "filepath": self.train_files,
            "class": self.train_labels
        }).to_csv(self.run_dir / "train_files_after_augment.csv", index=False)

    # ---------------------------
    # Preprocessing & Model (Fine-tune)
    # ---------------------------

    def build_efficientnet_base(self):
        try:
            # Prefer loading notop weights if provided
            if self.weights_url:
                weights_path = self.weights_dir / Path(self.weights_url).name
                if not weights_path.exists():
                    print("Downloading EfficientNet weights...")
                    download_file(self.weights_url, weights_path)
                # instantiate model with no weights first and then load
                base = EfficientNetB0(include_top=False, weights=None, input_shape=(self.image_size, self.image_size, 3))
                base.load_weights(str(weights_path), by_name=True)
                print("Loaded provided EfficientNet-B0 notop weights.")
            else:
                base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(self.image_size, self.image_size, 3))
                print("Loaded EfficientNet-B0 imagenet weights.")
        except Exception as e:
            print("Warning: failed to load provided weights, falling back to imagenet:", e)
            base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(self.image_size, self.image_size, 3))
        self.base_model = base

    def fine_tune_effb0(self):
        # Build base if not present
        if self.base_model is None:
            self.build_efficientnet_base()

        # Build temp classifier model for fine-tuning
        x = layers.GlobalAveragePooling2D()(self.base_model.output)
        num_classes = len(self.class_names)
        out = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs=self.base_model.input, outputs=out)

        # Freeze everything except top N layers
        total_layers = len(self.base_model.layers)
        n_unfreeze = max(0, min(self.unfreeze_top_n, total_layers))
        for i, layer in enumerate(self.base_model.layers):
            layer.trainable = False if i < (total_layers - n_unfreeze) else True

        # Compile
        lr = float(self.fine_tune_cfg.get("learning_rate", 1e-5))
        opt_name = self.fine_tune_cfg.get("optimizer", "adam")
        optimizer = optimizers.Adam(learning_rate=lr) if opt_name.lower() == "adam" else optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # Prepare tf.data generators for training/validation
        def flow_from_list(filepaths, labels, batch_size, shuffle=True, augment=False):
            # Create simple generator yields (x,y) batches preprocessed
            lb = LabelEncoder()
            lb.fit(self.class_names)
            y_encoded = lb.transform(labels)
            n = len(filepaths)
            idxs = np.arange(n)
            while True:
                if shuffle:
                    np.random.shuffle(idxs)
                for i in range(0, n, batch_size):
                    batch_idx = idxs[i:i+batch_size]
                    batch_x = []
                    batch_y = []
                    for j in batch_idx:
                        p = filepaths[j]
                        img = Image.open(p).convert("RGB").resize((self.image_size, self.image_size))
                        arr = np.array(img).astype("float32") / 255.0
                        batch_x.append(arr)
                        batch_y.append(y_encoded[j])
                    batch_x = np.stack(batch_x, axis=0)
                    batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=num_classes)
                    yield batch_x, batch_y

        train_gen = flow_from_list(self.train_files, self.train_labels, int(self.fine_tune_cfg.get("batch_size", self.batch_size)), shuffle=True)
        val_gen = flow_from_list(self.val_files, self.val_labels, int(self.fine_tune_cfg.get("batch_size", self.batch_size)), shuffle=False)

        # Steps
        steps_per_epoch = max(1, len(self.train_files) // int(self.fine_tune_cfg.get("batch_size", self.batch_size)))
        validation_steps = max(1, len(self.val_files) // int(self.fine_tune_cfg.get("batch_size", self.batch_size)))

        # Callbacks
        es_cfg = self.fine_tune_cfg.get("early_stopping", {})
        es = callbacks.EarlyStopping(monitor=es_cfg.get("monitor", "val_accuracy"), patience=int(es_cfg.get("patience", 3)), restore_best_weights=True)
        ckpt_path = self.weights_dir / "effb0_finetuned_best.keras"
        mc = callbacks.ModelCheckpoint(str(ckpt_path), monitor=es_cfg.get("monitor", "val_accuracy"), save_best_only=True, save_weights_only=False)
        rlr_cfg = self.fine_tune_cfg.get("reduce_lr", {})
        reduce_lr = callbacks.ReduceLROnPlateau(monitor=rlr_cfg.get("monitor", "val_loss"), factor=rlr_cfg.get("factor", 0.5), patience=int(rlr_cfg.get("patience", 2)))

        epochs = int(self.fine_tune_cfg.get("epochs", 10))
        print(f"Starting fine-tuning for up to {epochs} epochs. Steps per epoch: {steps_per_epoch}, val_steps: {validation_steps}")
        # Fit
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=[es, mc, reduce_lr],
            verbose=1
        )
        self.history = history
        # Save weights / model
        model.save(self.weights_dir / "effb0_finetuned_full_model.keras")
        print("Saved fine-tuned model to weights folder.")

        # After fine-tuning, remove classifier head and create feature extractor
        # Reinstantiate base model to ensure we reference the same convolutional backbone with trained weights
        # We will load the saved model and use its conv layers up to the last conv block
        try:
            saved = tf.keras.models.load_model(self.weights_dir / "effb0_finetuned_full_model.keras")
            # find layer in saved model that corresponds to base_model.layers[-1]
            # Use saved.layers up to the penultimate (GlobalAveragePooling2D is before Dense)
            # We assume the saved model structure: base_model -> GAP -> Dense
            # So we can locate the layer before GAP in saved model and create new model whose output is that layer's output processed with GAP.
            # Simpler: create new model using saved.input and take the layer named like 'top_activation' or take the convolutional base by name.
            # We'll create a feature extractor by taking the saved model and slicing before the Dense output.
            # Find index of GlobalAveragePooling2D layer
            gap_index = None
            for i, layer in enumerate(saved.layers):
                if isinstance(layer, layers.GlobalAveragePooling2D):
                    gap_index = i
                    break
            if gap_index is None:
                # fallback: use original base model with loaded weights
                print("Warning: GAP layer not found in saved model. Falling back to base model + GAP.")
                base = EfficientNetB0(include_top=False, weights=None, input_shape=(self.image_size, self.image_size,3))
                base.load_weights(str(self.weights_dir / "effb0_finetuned_best.keras"), by_name=True)
                x = layers.GlobalAveragePooling2D()(base.output)
                self.feature_extractor = models.Model(inputs=base.input, outputs=x)
            else:
                # build a new model from saved that outputs the GAP layer output (so excludes Dense)
                gap_layer = saved.layers[gap_index]
                feat_model = models.Model(inputs=saved.input, outputs=gap_layer.output)
                self.feature_extractor = feat_model
            print("Feature extractor ready.")
        except Exception as e:
            print("Failed to load saved fine-tuned model to create feature extractor:", e)
            # fallback to base_model + GAP
            if self.base_model is None:
                self.build_efficientnet_base()
            x = layers.GlobalAveragePooling2D()(self.base_model.output)
            self.feature_extractor = models.Model(inputs=self.base_model.input, outputs=x)

        # Save simple feature-extractor model
        try:
            self.feature_extractor.save(self.weights_dir / "effb0_feature_extractor.keras")
        except Exception:
            pass

        # Plot history
        if self.save_history_plot and self.history is not None:
            plot_and_save_history(self.history, self.analysis_dir / "training_history.png")

    # ---------------------------
    # Feature extraction (cache)
    # ---------------------------

    def extract_features_for_list(self, filepaths: List[str], split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        cache_path = self.weights_dir / self.features_filename.format(split=split_name)
        if self.cache_features and cache_path.exists():
            print(f"Loading cached features from {cache_path}")
            d = np.load(cache_path, allow_pickle=True)
            return d["features"], d["labels"]
        # else extract
        n = len(filepaths)
        batch = self.batch_size
        feat_list = []
        y_list = []
        for i in range(0, n, batch):
            chunk = filepaths[i:i+batch]
            imgs = []
            for p in chunk:
                img = Image.open(p).convert("RGB").resize((self.image_size, self.image_size))
                arr = np.array(img).astype("float32") / 255.0
                imgs.append(arr)
            X = np.stack(imgs, axis=0)
            feats = self.feature_extractor.predict(X, verbose=0)
            # feats shape: (b, 1280) if GAP
            if self.flatten_map:
                feats = feats.reshape((feats.shape[0], -1))
            feat_list.append(feats)
            # labels?
            # label extraction: get label from filepath mapping
            y_chunk = []
            for p in chunk:
                # filename in train/val/test lists has class in separate arrays; we'll create mapping
                # Find in which list this filepath exists and get its label
                pstr = str(p)
                # default mapping using df
                row = self.df[self.df["filepath"] == pstr]
                if not row.empty:
                    y_chunk.append(row["class"].values[0])
                else:
                    # fallback: extract parent folder name
                    y_chunk.append(Path(p).parent.name)
            y_list.extend(y_chunk)
        feats_all = np.concatenate(feat_list, axis=0)
        labels = np.array(y_list)
        if self.cache_features:
            np.savez_compressed(cache_path, features=feats_all, labels=labels)
        return feats_all, labels

    # ---------------------------
    # SVM training / tuning
    # ---------------------------

    def scale_features_if_needed(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray = None):
        if not self.scale_features:
            return X_train, X_val, X_test
        scaler = StandardScaler() if self.scaler_name == "StandardScaler" else StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test) if X_test is not None else None
        self.scaler = scaler
        joblib.dump(scaler, self.weights_dir / "scaler.joblib")
        return X_train_s, X_val_s, X_test_s

    def train_and_tune_svm(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        # Encode labels
        self.label_encoder.fit(self.class_names)
        y_train_enc = self.label_encoder.transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)

        # Grid search if requested
        if self.grid_cfg.get("enabled", True):
            param_grid = self.grid_cfg.get("param_grid", {"C":[1,10], "kernel":["rbf"], "gamma":["scale"]})
            cv = int(self.grid_cfg.get("cv", 5))
            n_jobs = int(self.grid_cfg.get("n_jobs", -1))
            svc = SVC(probability=self.final_svm_cfg.get("probability", True), class_weight=self.final_svm_cfg.get("class_weight", None))
            gs = GridSearchCV(svc, param_grid, cv=cv, verbose=2, n_jobs=n_jobs, scoring='accuracy')
            print("Starting GridSearchCV for SVM...")
            gs.fit(X_train, y_train_enc)
            best = gs.best_estimator_
            # Evaluate on val
            y_val_pred = best.predict(X_val)
            val_acc = accuracy_score(y_val_enc, y_val_pred)
            print("Grid search best params:", gs.best_params_, "val_acc:", val_acc)
            self.svm = best
            # Save CV results
            try:
                results = { "best_params": gs.best_params_, "best_score": float(gs.best_score_) }
                save_json(self.analysis_dir / "svm_grid_results.json", results)
            except Exception:
                pass
        else:
            # Train final SVM with provided final params
            params = dict(self.final_svm_cfg)
            kernel = params.pop("kernel", "rbf")
            C = params.pop("C", 1.0)
            gamma = params.pop("gamma", "scale")
            svc = SVC(kernel=kernel, C=C, gamma=gamma, probability=params.get("probability", True), class_weight=params.get("class_weight", None))
            svc.fit(X_train, y_train_enc)
            self.svm = svc

        # Save final SVM
        joblib.dump(self.svm, self.weights_dir / "svm_model.joblib")
        joblib.dump(self.label_encoder, self.weights_dir / "label_encoder.joblib")

        return self.svm

    # ---------------------------
    # Evaluation & False Predictions
    # ---------------------------

    def evaluate_and_save(self, X_test: np.ndarray, y_test: np.ndarray, filepaths_test: List[str]):
        y_test_enc = self.label_encoder.transform(y_test)
        y_pred_enc = self.svm.predict(X_test)
        y_proba = None
        try:
            y_proba = self.svm.predict_proba(X_test)
        except Exception:
            y_proba = None
        report = classification_report(y_test_enc, y_pred_enc, output_dict=True, target_names=self.class_names)
        save_json(self.analysis_dir / "results.json", report)
        plot_and_save_confusion_matrix(y_test_enc, y_pred_enc, labels=self.class_names, outpath=self.analysis_dir / "confusion_matrix.png")
        false_preds = []
        for i, (true_label, pred_label) in enumerate(zip(y_test, self.label_encoder.inverse_transform(y_pred_enc))):
            if true_label != pred_label:
                record = {
                    "filename": str(filepaths_test[i]),
                    "true_label": str(true_label),
                    "predicted_label": str(pred_label),
                }
                if y_proba is not None:
                    conf = float(np.max(y_proba[i]))
                    record["predicted_confidence"] = conf
                false_preds.append(record)
        save_json(self.analysis_dir / "false_predictions.json", false_preds)
        for rec in false_preds:
            src = Path(rec["filename"])
            true_label = rec["true_label"]
            predicted_label = rec["predicted_label"]
            if src.exists():
                dst = self.false_preds_dir / src.name
                try:
                    img = cv2.imread(str(src))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_rgb)
                        width, height = img_pil.size
                        new_height = height + 80
                        new_img = Image.new('RGB', (width, new_height), color='white')
                        new_img.paste(img_pil, (0, 0))
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(new_img)
                        
                        try:
                            font = ImageFont.truetype("arial.ttf", 20)
                        except:
                            try:
                                font = ImageFont.truetype("DejaVuSans.ttf", 20)
                            except:
                                font = ImageFont.load_default()
                        
                        filename_text = f"File: {src.name}"
                        true_text = f"True: {true_label}"
                        pred_text = f"Pred: {predicted_label}"
                        
                        y_offset = height + 5
                        draw.text((10, y_offset), filename_text, fill='black', font=font)
                        draw.text((10, y_offset + 25), true_text, fill='red', font=font)  # True in red
                        draw.text((10, y_offset + 50), pred_text, fill='blue', font=font)  # Predicted in blue
                        
                        new_img.save(dst)
                        
                except Exception as e:
                    print(f"Error processing {src}: {e}")
                    try:
                        shutil.copy(src, dst)
                    except Exception:
                        pass

        print(f"Evaluation done. Accuracy: {accuracy_score(y_test_enc, y_pred_enc):.4f}. False predictions saved: {len(false_preds)}")

    # ---------------------------
    # Orchestration
    # ---------------------------

    def run(self):
        # 1. Collect dataset and split
        self.collect_dataset()
        self.stratified_split()

        # 2. Augmentation (applies only on training)
        if self.aug_cfg.get("enable_augmentation", True):
            self.generate_augmented_set()

        # 3. Build & fine-tune EfficientNet (if enabled)
        if self.fine_tune_cfg.get("enabled", True):
            self.fine_tune_effb0()
        else:
            # If not fine-tuning, just create feature extractor from imagenet weights
            self.build_efficientnet_base()
            x = layers.GlobalAveragePooling2D()(self.base_model.output)
            self.feature_extractor = models.Model(inputs=self.base_model.input, outputs=x)
            try:
                self.feature_extractor.save(self.weights_dir / "effb0_feature_extractor.keras")
            except Exception:
                pass

        # 4. Feature extraction for train/val/test
        X_train_feat, y_train_f = self.extract_features_for_list(self.train_files, "train")
        X_val_feat, y_val_f = self.extract_features_for_list(self.val_files, "val")
        X_test_feat, y_test_f = self.extract_features_for_list(self.test_files, "test")

        # Ensure label arrays align with label encoder later
        # 5. Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features_if_needed(X_train_feat, X_val_feat, X_test_feat)

        # 6. Train & tune SVM
        self.train_and_tune_svm(X_train_scaled, y_train_f, X_val_scaled, y_val_f)

        # 7. Evaluate on test set and save false_predictions.json
        self.evaluate_and_save(X_test_scaled, y_test_f, self.test_files)

        print("Run complete. Artifacts saved to:", str(self.run_dir))


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="EfficientNet-B0 + SVM training pipeline")
    parser.add_argument("--config", "-c", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    trainer = EffB0SVMTrainer(args.config)
    trainer.run()

if __name__ == "__main__":
    main()
