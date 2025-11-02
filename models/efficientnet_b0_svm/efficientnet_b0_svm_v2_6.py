#!/usr/bin/env python3
"""
EffB0SVMTrainer - Hybrid pipeline combining v1.0's performance with v2.x's features:
- Uses v1.0's proper EfficientNet preprocessing and training strategy
- Keeps v2.x's balanced augmentation and SVM pipeline
- Maintains structured output and analysis
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
from PIL import Image, ImageDraw, ImageFont
import cv2

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

# Albumentations for augmentations
import albumentations as A

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import joblib

# For reproducibility
import numpy.random as npr

from tqdm import tqdm

plt.switch_backend("agg")

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
        self.logs_dir = self.run_dir / "logs"
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
        ensure_dir(self.logs_dir)

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

        # --- Pie chart showing train/val/test split proportions ---
        split_counts = {
            "Training Set": len(self.train_files),
            "Validation Set": len(self.val_files),
            "Test Set": len(self.test_files),
        }
        plot_and_save_pie(
            split_counts,
            self.analysis_dir / "partition_pie.png",
            title="Dataset Split Distribution (Train/Val/Test)"
        )

        print(f"Split into train={len(self.train_files)}, val={len(self.val_files)}, test={len(self.test_files)})")

    # ---------------------------
    # Augmentation (v2.x approach)
    # ---------------------------

    def build_alb_transforms(self):
        """Build Albumentations pipeline tuned for pomelo disease images (small features)."""
        a_cfg = self.aug_cfg.get("albumentations", {})
        transforms = []

        if a_cfg.get("brightness_contrast", {}).get("enabled", False):
            p = a_cfg["brightness_contrast"]
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=p.get("brightness_limit", 0.15),
                contrast_limit=p.get("contrast_limit", 0.15),
                p=p.get("p", 0.5)
            ))

        if a_cfg.get("hue_saturation", {}).get("enabled", False):
            p = a_cfg["hue_saturation"]
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=p.get("hue_shift_limit", 10),
                sat_shift_limit=p.get("sat_shift_limit", 15),
                val_shift_limit=p.get("val_shift_limit", 10),
                p=p.get("p", 0.4)
            ))

        if a_cfg.get("clahe", {}).get("enabled", False):
            p = a_cfg["clahe"]
            transforms.append(A.CLAHE(
                clip_limit=p.get("clip_limit", 2.0),
                tile_grid_size=tuple(p.get("tile_grid_size", (8, 8))),
                p=p.get("p", 0.3)
            ))

        if a_cfg.get("gaussian_blur", {}).get("enabled", False):
            p = a_cfg["gaussian_blur"]
            transforms.append(A.GaussianBlur(
                blur_limit=p.get("blur_limit", (1, 3)),
                p=p.get("p", 0.2)
            ))

        if a_cfg.get("gauss_noise", {}).get("enabled", False):
            p = a_cfg["gauss_noise"]
            transforms.append(A.GaussNoise(
                var_limit=p.get("var_limit", (5.0, 25.0)),
                p=p.get("p", 0.2)
            ))

        if a_cfg.get("sharpen", {}).get("enabled", False):
            p = a_cfg["sharpen"]
            transforms.append(A.Sharpen(
                alpha=p.get("alpha", (0.1, 0.3)),
                lightness=p.get("lightness", (0.9, 1.1)),
                p=p.get("p", 0.3)
            ))

        if a_cfg.get("rotation", {}).get("enabled", False):
            p = a_cfg["rotation"]
            transforms.append(A.Rotate(
                limit=p.get("limit", 10),  # small rotation to preserve details
                border_mode=cv2.BORDER_REFLECT_101,
                p=p.get("p", 0.5)
            ))

        if a_cfg.get("flip", {}).get("enabled", True):
            p = a_cfg["flip"]
            transforms.append(A.HorizontalFlip(p=p.get("p", 0.5)))
            transforms.append(A.VerticalFlip(p=p.get("p", 0.3)))

        if a_cfg.get("shift_scale", {}).get("enabled", False):
            p = a_cfg["shift_scale"]
            transforms.append(A.ShiftScaleRotate(
                shift_limit=p.get("shift_limit", 0.02),
                scale_limit=p.get("scale_limit", 0.05),
                rotate_limit=0,  # disable rotation to avoid redundancy
                border_mode=cv2.BORDER_REFLECT_101,
                p=p.get("p", 0.3)
            ))

        if a_cfg.get("rgb_shift", {}).get("enabled", False):
            p = a_cfg["rgb_shift"]
            transforms.append(A.RGBShift(
                r_shift_limit=p.get("r_shift_limit", 10),
                g_shift_limit=p.get("g_shift_limit", 10),
                b_shift_limit=p.get("b_shift_limit", 10),
                p=p.get("p", 0.2)
            ))

        if a_cfg.get("multiplicative_noise", {}).get("enabled", False):
            p = a_cfg["multiplicative_noise"]
            transforms.append(A.MultiplicativeNoise(
                multiplier=p.get("shift_limit", (0.9, 1.1)),
                p=p.get("p", 0.2)
            ))
        
        if a_cfg.get("random_gamma", {}).get("enabled", False):
            p = a_cfg["random_gamma"]
            transforms.append(A.RandomGamma(
                gamma_limit=p.get("gamma_limit", (90, 110)),
                p=p.get("p", 0.3)
            ))

        transforms.append(A.Resize(self.image_size, self.image_size))

        return A.Compose(transforms)

    def generate_augmented_set(self):
        """
        Generates an augmented and balanced training set per class using Albumentations only.
        """
        ensure_dir(self.augments_dir)
        for cls in self.class_names:
            ensure_dir(self.augments_dir / cls)

        # Count class occurrences in training set
        train_counts = pd.Series(self.train_labels).value_counts().to_dict()
        target = int(self.per_class_target or max(train_counts.values()))
        per_class_after = {}

        alb_transform = self.build_alb_transforms()
        target_size = (self.image_size, self.image_size)

        for cls in self.class_names:
            cls_train_files = [f for f, l in zip(self.train_files, self.train_labels) if l == cls]
            n_existing = len(cls_train_files)
            n_target = target
            n_needed = max(0, n_target - n_existing)
            saved = 0
            out_cls_dir = self.augments_dir / cls

            # Clear existing augmented images for this class
            for existing_file in out_cls_dir.glob("*"):
                if existing_file.is_file():
                    existing_file.unlink()

            print(f"[Augment] Class '{cls}': existing={n_existing}, target={n_target}, needed={n_needed}")

            # Step 1 — Copy & resize original images first
            for src in cls_train_files:
                dst = out_cls_dir / Path(src).name
                if not dst.exists():
                    img = cv2.imread(str(src))
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
                    Image.fromarray(img_resized).save(dst)

            # Step 2 — Generate augmentations if needed
            if n_needed > 0:
                per_source = min(self.max_aug_per_source, math.ceil(n_needed / max(1, n_existing)))
                aug_per_image = {src: 0 for src in cls_train_files}

                while saved < n_needed:
                    available = [src for src, count in aug_per_image.items() if count < per_source]
                    if not available:
                        break  # all reached max per source
                    src = random.choice(available)

                    img = cv2.imread(str(src))
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Apply Albumentations transformation
                    try:
                        augmented = alb_transform(image=img_rgb)
                        aug_img = augmented["image"]
                    except Exception as e:
                        print(f"  Albumentations failed for {src}: {e}")
                        continue

                    aug_img_resized = cv2.resize(aug_img, target_size, interpolation=cv2.INTER_AREA)

                    # Save augmented image
                    fname = f"aug_{saved:04d}_{Path(src).stem}.jpg"
                    out_path = out_cls_dir / fname
                    try:
                        Image.fromarray(aug_img_resized).save(out_path)
                        saved += 1
                        aug_per_image[src] += 1
                    except Exception as e:
                        print(f"  Failed to save {out_path}: {e}")

            # Count final images for this class
            final_files = list(out_cls_dir.glob("*"))
            per_class_after[cls] = len(final_files)
            print(f"  Final count for '{cls}': {per_class_after[cls]}")

        # Step 3 — Save class distribution summary
        class_before = dict(pd.Series(self.train_labels).value_counts().to_dict())
        save_json(self.analysis_dir / "class_counts_before.json", class_before)
        save_json(self.analysis_dir / "class_counts_after.json", per_class_after)
        plot_and_save_bar(
            class_before, per_class_after,
            self.analysis_dir / "class_distribution_before_after.png"
        )
        print("Augmentation finished. Augmented images saved to:", str(self.augments_dir))

        # Step 4 — Update train lists to point to augments dir (balanced)
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

        # Step 5 — Save new train list CSV
        pd.DataFrame({
            "filepath": self.train_files,
            "class": self.train_labels
        }).to_csv(self.run_dir / "train_files_after_augment.csv", index=False)

    # ---------------------------
    # Model Building (v1.0 approach)
    # ---------------------------

    def build_efficientnet_base(self):
        """Build EfficientNet base with proper weights loading (v1.0 approach)"""
        try:
            if self.weights_url:
                weights_path = self.weights_dir / Path(self.weights_url).name
                if not weights_path.exists():
                    print("Downloading EfficientNet weights...")
                    download_file(self.weights_url, weights_path)
                # Build architecture (no weights yet)
                base = EfficientNetB0(
                    include_top=False,
                    weights=None,
                    input_shape=(self.image_size, self.image_size, 3),
                    pooling="avg"
                )
                base.load_weights(str(weights_path), by_name=True)
                print("Loaded provided EfficientNet-B0 notop weights.")
            else:
                base = EfficientNetB0(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(self.image_size, self.image_size, 3),
                    pooling="avg"
                )
                print("Loaded EfficientNet-B0 imagenet weights.")
        except Exception as e:
            print("Warning: failed to load provided weights, falling back to imagenet:", e)
            base = EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=(self.image_size, self.image_size, 3),
                pooling="avg"
            )
        self.base_model = base
        return base

    def build_finetune_model(self, base_model, num_classes):
        """Enhanced fine-tuning model for EfficientNetB0 hybrid pipeline."""
        x = base_model.output
        if len(x.shape) == 4:
            x = layers.GlobalAveragePooling2D()(x)

        # Stabilize input from base model
        x = layers.BatchNormalization()(x)

        # Bottleneck Dense Block 1
        x = layers.Dense(
            512,
            activation="swish",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        # Dense Block 2 - smaller with stronger regularization
        x = layers.Dense(
            256,
            activation="selu",
            kernel_regularizer=regularizers.l2(5e-5)
        )(x)
        x = layers.AlphaDropout(0.35)(x)  # works well with SELU

        # Final compression before classifier
        x = layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)

        preds = layers.Dense(num_classes, activation="softmax")(x)

        model = models.Model(inputs=base_model.input, outputs=preds)
        return model

    def fine_tune_effb0(self):
        """Flexible multi-phase fine-tuning of EfficientNet-B0 Hybrid."""

        print("Starting fine-tuning (EfficientNet-B0 Hybrid)...")

        base = self.base_model or self.build_efficientnet_base()
        num_classes = len(self.class_names)
        model = self.build_finetune_model(base, num_classes)

        self.label_encoder.fit(self.class_names)
        y_train_enc = self.label_encoder.transform(self.train_labels)
        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train_enc),
            y=y_train_enc
        )
        class_weights = {i: w for i, w in enumerate(class_weights_array)}
        print("Computed class weights:", class_weights)

        # Choose dataset loading method based on config
        load_mode = self.fine_tune_cfg.get("data_loading_mode", "stream").lower()
        if load_mode == "memory":
            print("🔹 Loading datasets into memory (faster, higher RAM use)")
            X_train, y_train = self.load_images_to_memory(self.train_files, self.train_labels)
            X_val, y_val = self.load_images_to_memory(self.val_files, self.val_labels)
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.batch_size)
        else:
            print("🔹 Streaming datasets from disk (lower RAM usage)")
            train_ds, val_ds = self.build_datasets_from_files()

        es_cfg = self.fine_tune_cfg.get("early_stopping", {})
        es = callbacks.EarlyStopping(
            monitor=es_cfg.get("monitor", "val_accuracy"),
            patience=int(es_cfg.get("patience", 5)),
            restore_best_weights=True
        )

        rlr_cfg = self.fine_tune_cfg.get("reduce_lr", {})
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=rlr_cfg.get("monitor", "val_loss"),
            factor=rlr_cfg.get("factor", 0.5),
            patience=int(rlr_cfg.get("patience", 3)),
            min_lr=float(rlr_cfg.get("min_lr", 1e-7))
        )

        print("\nPHASE 1: Training classifier head (base frozen)")
        base.trainable = False

        lr_phase1 = float(self.fine_tune_cfg.get("learning_rate", 1e-4))
        optimizer = optimizers.Adam(learning_rate=lr_phase1)
        model.compile(optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

        ckpt_phase1 = self.weights_dir / "effb0_phase1_best.keras"
        mc1 = callbacks.ModelCheckpoint(
            str(ckpt_phase1),
            monitor=es_cfg.get("monitor", "val_accuracy"),
            save_best_only=True,
            verbose=1
        )

        history_all = []
        hist1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(self.fine_tune_cfg.get("epochs", 20)),
            callbacks=[es, reduce_lr, mc1],
            class_weight=class_weights,
            verbose=1
        )
        history_all.append(hist1.history)

        # === Gradual unfreezing ===
        unfreeze_stages = self.fine_tune_cfg.get("unfreeze_stages", [])
        print(f"\nFound {len(unfreeze_stages)} configurable unfreeze stages.")
        total_layers = len(base.layers)

        for idx, stage in enumerate(unfreeze_stages, start=2):
            layers_to_unfreeze = int(stage.get("layers_to_unfreeze", 0))
            epochs_stage = int(stage.get("epochs", 5))
            lr_stage = float(stage.get("learning_rate", lr_phase1 / 10))

            # Limit unfreezing if exceeds total layer count
            layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
            print(f"\nPHASE {idx}: Unfreezing top {layers_to_unfreeze} layers | "
                f"LR={lr_stage:.1e} | Epochs={epochs_stage}")

            base.trainable = True
            for layer in base.layers[:-layers_to_unfreeze]:
                layer.trainable = False

            optimizer_stage = optimizers.Adam(learning_rate=lr_stage)
            model.compile(
                optimizer=optimizer_stage,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            ckpt_stage = self.weights_dir / f"effb0_phase{idx}_best.keras"
            mc_stage = callbacks.ModelCheckpoint(
                str(ckpt_stage),
                monitor=es_cfg.get("monitor", "val_accuracy"),
                save_best_only=True,
                verbose=1
            )

            hist_stage = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs_stage,
                callbacks=[es, reduce_lr, mc_stage],
                class_weight=class_weights,
                verbose=1
            )
            history_all.append(hist_stage.history)
        
        # === Save model & feature extractor ===
        final_model_path = self.weights_dir / "effb0_finetuned_full_model.keras"
        model.save(final_model_path)
        print(f"✅ Saved fine-tuned model: {final_model_path}")

        self.feature_extractor = models.Model(inputs=base.input, outputs=base.output)
        self.feature_extractor.save(self.weights_dir / "effb0_feature_extractor.keras")

        # === Embedding-Level Balancing ===
        print("\n🔸 Generating balanced embeddings for SVM training...")
        embeddings, y_enc = self.extract_embeddings(self.train_files)
        X_bal, y_bal = self.balance_embeddings_within_classes(embeddings, y_enc)

        emb_path = self.weights_dir / "balanced_embeddings.joblib"
        joblib.dump({"features": X_bal, "labels": y_bal}, emb_path)
        print(f"💾 Saved balanced embeddings: {emb_path}")

        # -------------------------------
        # Save training history plot
        # -------------------------------

        if self.save_history_plot and self.history is not None:
            class MockHistory:
                def __init__(self, history_dict):
                    self.history = history_dict
            plot_and_save_history(MockHistory(self.history), self.analysis_dir / "training_history.png")

        print("\nFine-tuning complete (multi-phase + embedding balancing).")

    def load_images_to_memory(self, filepaths: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Load images into memory (v1.0 approach for better performance)"""
        X = []
        y = []
        
        # Fit label encoder if not already fitted
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(self.class_names)
        
        y_encoded = self.label_encoder.transform(labels)
        
        for i, filepath in enumerate(filepaths):
            try:
                # Load and resize image
                img = Image.open(filepath).convert("RGB").resize((self.image_size, self.image_size))
                arr = np.array(img, dtype=np.float32)
                X.append(arr)
                y.append(y_encoded[i])
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
                # Add zero array as placeholder
                X.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.float32))
                y.append(y_encoded[i])
        
        return np.array(X), np.array(y)

    def build_datasets_from_files(self):
        """
        Builds TensorFlow datasets (train/val) directly from image file paths.
        Avoids loading all images into RAM.
        """

        def decode_img(img_path, label):
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [self.image_size, self.image_size])
            img = img / 255.0
            return img, label

        # Encode class labels as integers
        self.label_encoder.fit(self.class_names)
        y_train_enc = self.label_encoder.transform(self.train_labels)
        y_val_enc = self.label_encoder.transform(self.val_labels)

        # Convert to TensorFlow tensors
        train_paths = tf.convert_to_tensor(self.train_files)
        val_paths = tf.convert_to_tensor(self.val_files)
        train_labels = tf.convert_to_tensor(y_train_enc, dtype=tf.int32)
        val_labels = tf.convert_to_tensor(y_val_enc, dtype=tf.int32)

        # Build raw datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

        # Apply decoding and optional augmentations
        train_ds = train_ds.shuffle(buffer_size=len(self.train_files))
        train_ds = train_ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        print(f"✅ Built datasets: {len(self.train_files)} train, {len(self.val_files)} val")

        return train_ds, val_ds

    # ======================================================================
    # EMBEDDING-LEVEL CLUSTERING & BALANCING
    # ======================================================================

    def extract_embeddings(self, filepaths):
        """Extract embeddings using the fine-tuned feature extractor."""
        print(f"Extracting embeddings for {len(filepaths)} samples...")
        embeddings = []
        y_encoded = []

        label_to_index = {cls: i for i, cls in enumerate(self.class_names)}

        for i in tqdm(range(0, len(filepaths), self.batch_size)):
            batch_paths = filepaths[i:i + self.batch_size]
            batch_imgs = []
            for p in batch_paths:
                img = image.load_img(p, target_size=(self.image_size, self.image_size))
                arr = image.img_to_array(img)
                arr = preprocess_input(arr)
                batch_imgs.append(arr)

            batch_imgs = np.array(batch_imgs)
            feats = self.feature_extractor.predict(batch_imgs, verbose=0)
            embeddings.append(feats)
            y_encoded.extend([label_to_index[self.train_labels[j]]
                            for j in range(i, i + len(batch_paths))])

        embeddings = np.vstack(embeddings)
        y_encoded = np.array(y_encoded)
        print(f"✅ Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
        return embeddings, y_encoded

    def balance_embeddings_within_classes(
        self,
        embeddings,
        labels,
        n_clusters_per_class=3,
        strategy="hybrid",
        min_ratio=0.7,
        max_ratio=1.3
    ):
        """
        Cluster embeddings within each class and balance clusters via hybrid up/down sampling.
        """
        balanced_X, balanced_y = [], []
        print("\n🔍 Performing embedding-level clustering & balancing...")

        for cls_idx, cls_name in enumerate(self.class_names):
            cls_emb = embeddings[labels == cls_idx]
            if len(cls_emb) < n_clusters_per_class:
                balanced_X.append(cls_emb)
                balanced_y.extend([cls_idx] * len(cls_emb))
                continue

            print(f"\n📦 Class '{cls_name}' — {len(cls_emb)} embeddings before balancing")
            kmeans = KMeans(n_clusters=n_clusters_per_class, random_state=42)
            cluster_ids = kmeans.fit_predict(cls_emb)
            cluster_counts = np.bincount(cluster_ids)
            target_size = int(np.mean(cluster_counts))
            print(f"Target cluster size: {target_size}")

            for c in range(n_clusters_per_class):
                cluster_emb = cls_emb[cluster_ids == c]
                size = len(cluster_emb)

                if strategy == "hybrid":
                    if size < target_size * min_ratio:
                        n_samples = int(target_size * min_ratio)
                        up = resample(cluster_emb, replace=True, n_samples=n_samples, random_state=42)
                        noise = np.random.normal(0, 0.01, up.shape)
                        up = up + noise
                        balanced_X.append(up)
                        balanced_y.extend([cls_idx] * len(up))
                    elif size > target_size * max_ratio:
                        n_samples = int(target_size * max_ratio)
                        down = resample(cluster_emb, replace=False, n_samples=n_samples, random_state=42)
                        balanced_X.append(down)
                        balanced_y.extend([cls_idx] * len(down))
                    else:
                        balanced_X.append(cluster_emb)
                        balanced_y.extend([cls_idx] * len(cluster_emb))
                else:
                    resampled = resample(cluster_emb, replace=True, n_samples=target_size, random_state=42)
                    balanced_X.append(resampled)
                    balanced_y.extend([cls_idx] * len(resampled))

        balanced_X = np.vstack(balanced_X)
        balanced_y = np.array(balanced_y)
        print(f"\n✅ Balanced embedding set: {balanced_X.shape[0]} samples total")
        return balanced_X, balanced_y

    # ---------------------------
    # Feature extraction (cache)
    # ---------------------------

    def extract_features_for_list(self, filepaths: List[str], split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        cache_path = self.weights_dir / self.features_filename.format(split=split_name)
        if self.cache_features and cache_path.exists():
            print(f"Loading cached features from {cache_path}")
            d = np.load(cache_path, allow_pickle=True)
            return d["features"], d["labels"]
        
        # Extract features
        X, y_encoded = self.load_images_to_memory(filepaths, [self.get_label_from_filepath(p) for p in filepaths])
        
        # Apply EfficientNet preprocessing
        X_pp = preprocess_input(X.copy())
        
        # Extract features
        feats = self.feature_extractor.predict(X_pp, verbose=1)
        
        # Get labels
        labels = []
        for p in filepaths:
            pstr = str(p)
            row = self.df[self.df["filepath"] == pstr]
            if not row.empty:
                labels.append(row["class"].values[0])
            else:
                labels.append(Path(p).parent.name)
        
        if self.cache_features:
            np.savez_compressed(cache_path, features=feats, labels=labels)
            
        return feats, np.array(labels)

    def get_label_from_filepath(self, filepath: str) -> str:
        """Extract label from filepath"""
        pstr = str(filepath)
        row = self.df[self.df["filepath"] == pstr]
        if not row.empty:
            return row["class"].values[0]
        else:
            return Path(filepath).parent.name

    # ---------------------------
    # SVM training / tuning (v2.x approach)
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
            svc = SVC(probability=self.final_svm_cfg.get("probability", True), class_weight=self.final_svm_cfg.get("class_weight", "balanced"))
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
            svc = SVC(kernel=kernel, C=C, gamma=gamma, probability=params.get("probability", True), class_weight=params.get("class_weight", "balanced"))
            svc.fit(X_train, y_train_enc)
            self.svm = svc

        # Save final SVM
        joblib.dump(self.svm, self.weights_dir / "svm_model.joblib")
        joblib.dump(self.label_encoder, self.weights_dir / "label_encoder.joblib")

        return self.svm

    # ---------------------------
    # Evaluation & False Predictions (v2.x approach)
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
                        draw.text((10, y_offset + 25), true_text, fill='red', font=font)
                        draw.text((10, y_offset + 50), pred_text, fill='blue', font=font)
                        
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
            self.feature_extractor = self.base_model  # Base already has pooling='avg'

        # 4. Feature extraction / use pre-balanced embeddings
        emb_path = self.weights_dir / "balanced_embeddings.joblib"

        if emb_path.exists():
            print(f"\n🔸 Found balanced embeddings at {emb_path}, loading instead of raw extraction...")
            emb_data = joblib.load(emb_path)
            X_train_feat = emb_data["features"]
            y_train_f = emb_data["labels"]
        else:
            print("\n⚠️ No pre-balanced embeddings found. Extracting raw training features instead...")
            X_train_feat, y_train_f = self.extract_features_for_list(self.train_files, "train")

        # Validation and test embeddings are always freshly extracted
        X_val_feat, y_val_f = self.extract_features_for_list(self.val_files, "val")
        X_test_feat, y_test_f = self.extract_features_for_list(self.test_files, "test")

        # 5. Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features_if_needed(
            X_train_feat, X_val_feat, X_test_feat
        )

        # 6. Train & tune SVM
        self.train_and_tune_svm(X_train_scaled, y_train_f, X_val_scaled, y_val_f)

        # 7. Evaluate on test set and save false_predictions.json
        self.evaluate_and_save(X_test_scaled, y_test_f, self.test_files)

        print("✅ Run complete. Artifacts saved to:", str(self.run_dir))

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
