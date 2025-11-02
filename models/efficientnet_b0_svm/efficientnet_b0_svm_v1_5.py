#!/usr/bin/env python3
"""
efficientnet_b0_svm_v1_5.py

EfficientNetB0 + SVM pipeline (v1.5)
- Fresh CLAHE per run (old clahe_preprocessed deleted)
- Augmentation for balanced dataset
- EfficientNetB0 as feature extractor ONLY (no classification training)
- SVM as final classifier
- manual_rgb pretrained weights
- SMOTE optional
- Metrics JSON saved
- RESIZING ONLY DURING MODEL FEEDING (not in CLAHE)
"""

import argparse
import datetime
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Enabled GPU memory growth")
    except:
        pass

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DISABLE_PREFETCH_AUTOTUNER'] = '1'
from tensorflow import keras
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Optional: imblearn.SMOTE
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

# seaborn for plotting confusion matrix
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

# -------------------------
# Helpers and IO
# -------------------------

def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_output_dirs(base_out):
    ts = datetime.datetime.now().strftime("effb0svm_%Y%m%d_%H%M%S")
    base = Path(base_out) / ts
    subs = {
        "root": base,
        "clahe_preprocessed": base / "clahe_preprocessed",
        "augment_samples": base / "augment_samples",
        "prediction_samples": base / "prediction_samples",
        "plots": base / "plots",
        "metrics": base / "metrics",
        "weights": base / "weights",
        "log": base / "log",
        "features": base / "features"
    }
    for d in subs.values():
        d.mkdir(parents=True, exist_ok=True)
    return base, subs

def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "training.log"
    logger = logging.getLogger("effb0svm")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(str(logfile))
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    logger.info("Logging to %s", logfile)
    return logger

# -------------------------
# Find previous runs
# -------------------------

def find_latest_clahe_folder(base_out):
    base = Path(base_out)
    if not base.exists():
        return None
    outs = sorted(list(base.glob("effb0svm_*")), key=lambda p: p.stat().st_mtime, reverse=True)
    for o in outs:
        c = o / "clahe_preprocessed"
        if c.exists() and any(c.iterdir()):
            return o
    return None

# -------------------------
# CLAHE preprocessing (fresh each run)
# -------------------------

def ensure_rgb(pil: Image.Image) -> Image.Image:
    """Convert RGBA to RGB with smart background handling for pomelo"""
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        
        if pil.mode == "RGBA":
            # Remove completely transparent pixels by making them white
            r, g, b, a = pil.split()
            mask = a.point(lambda i: 0 if i < 30 else 255)
            bg.paste(pil, mask=mask)
        return bg
    return pil.convert("RGB")

def apply_clahe_pil(pil_img, clipLimit=2.0, tileGridSize=(16,16)):
    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    
    # Apply color normalization first
    arr = apply_color_normalization(arr)
    
    # Convert to LAB color space for better CLAHE
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply only to lightness channel
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out

def delete_prev_clahe(prev_run: Path, logger):
    """Delete prev_run/clahe_preprocessed to free space (if exists)."""
    if prev_run is None:
        return
    prev_clahe = prev_run / "clahe_preprocessed"
    if prev_clahe.exists():
        try:
            logger.info("Deleting previous CLAHE folder: %s", prev_clahe)
            shutil.rmtree(prev_clahe)
            logger.info("Deleted previous CLAHE folder.")
        except Exception as e:
            logger.warning("Could not delete previous CLAHE folder: %s", e)

def preprocess_clahe(src_dataset_dir: str, dst_dir: Path, logger, clipLimit=3.0, tileGridSize=(8,8)):
    """CLAHE preprocessing WITHOUT resizing - keep original image sizes"""
    src = Path(src_dataset_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    logger.info("Starting fresh CLAHE preprocessing: %s -> %s", src, dst)
    classes = sorted([p for p in src.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError("No class subfolders found in dataset_dir: " + str(src))
    for cls in classes:
        out_cls = dst / cls.name
        out_cls.mkdir(parents=True, exist_ok=True)
        files = sorted([f for ext in ("*.jpg","*.jpeg","*.png","*.bmp") for f in cls.glob(ext)])
        for f in tqdm(files, desc=f"CLAHE {cls.name}"):
            try:
                p = ensure_rgb(Image.open(f))
                arr = apply_clahe_pil(p, clipLimit=clipLimit, tileGridSize=tileGridSize)
                outp = out_cls / f.name
                cv2.imwrite(str(outp), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            except Exception as e:
                logger.warning("Failed CLAHE on %s: %s", f, e)
    logger.info("CLAHE preprocessing finished (original sizes preserved).")

# -------------------------
# Augmentation helpers - WORK WITH ORIGINAL SIZES
# -------------------------

def random_contrast(img: Image.Image, factor_min=0.8, factor_max=1.2):
    factor = random.uniform(factor_min, factor_max)
    return ImageEnhance_Contrast(img).enhance(factor)

def ImageEnhance_Contrast(img):
    # small helper to avoid import clash with ImageEnhance repeated usage
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(img)

def random_rotation(img: Image.Image, max_deg=25):
    ang = random.uniform(-max_deg, max_deg)
    return img.rotate(ang, resample=Image.BICUBIC, expand=False)

def gaussian_blur(img: Image.Image, max_radius=1.5):
    r = random.uniform(0.0, max_radius)
    if r <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=r))

def salt_pepper(img: np.ndarray, amount=0.004, s_vs_p=0.5):
    out = img.copy()
    h, w, c = out.shape
    num_salt = np.ceil(amount * h * w * s_vs_p)
    num_pepper = np.ceil(amount * h * w * (1.0 - s_vs_p))
    # Salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in (h, w)]
    out[coords[0], coords[1], :] = 255
    # Pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in (h, w)]
    out[coords[0], coords[1], :] = 0
    return out

def random_crop(img: Image.Image, crop_scale_min=0.8):
    w, h = img.size
    scale = random.uniform(crop_scale_min, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BICUBIC)

def mosaic_combine(image_paths, target_size):
    """
    Simple mosaic: pick up to 4 images from the same class, resize to half, tile in 2x2.
    image_paths: list of paths (>=1)
    target_size: final output size for the model
    """
    imgs = []
    for i in range(4):
        p = image_paths[i % len(image_paths)]
        img = Image.open(p).convert("RGB").resize((target_size//2, target_size//2), Image.BICUBIC)
        imgs.append(img)
    new = Image.new("RGB", (target_size, target_size))
    new.paste(imgs[0], (0,0))
    new.paste(imgs[1], (target_size//2,0))
    new.paste(imgs[2], (0,target_size//2))
    new.paste(imgs[3], (target_size//2,target_size//2))
    return new

def augment_image_and_save(img_path: Path, save_dir: Path, target_size: int, n_aug=1, logger=None):
    """Enhanced augmentation - images will be resized to target_size only when saved for model input"""
    augmented_paths = []
    
    try:
        orig = Image.open(img_path).convert("RGB")
    except Exception as e:
        if logger: 
            logger.warning("Failed to open %s for augmentation: %s", img_path, e)
        return augmented_paths
    
    for i in range(n_aug):
        img = orig.copy()
        
        if random.random() < 0.7:
            img = smart_zoom(img, orig.size, zoom_range=(0.7, 1.0))
        if random.random() < 0.7:
            img = random_rotation(img, max_deg=25)
        if random.random() < 0.6:
            img = enhance_contrast_simple(img)
        if random.random() < 0.7:
            img = random_crop(img, crop_scale_min=0.85)
        if random.random() < 0.6:
            img = Image.fromarray(salt_pepper(np.array(img), amount=0.003))
        if random.random() < 0.6:
            img = gaussian_blur(img, max_radius=1.2)
        if random.random() < 0.7:
            img = random_contrast(img, 0.85, 1.25)
        
        img = img.resize((target_size, target_size), Image.LANCZOS)
        
        timestamp = int(time.time() * 1000) % 1000000
        outp = save_dir / f"{img_path.stem}_aug_{timestamp}_{i}.jpg"
        
        counter = 0
        while outp.exists():
            outp = save_dir / f"{img_path.stem}_aug_{timestamp}_{i}_{counter}.jpg"
            counter += 1
            
        img.save(outp, quality=90)
        augmented_paths.append(outp)
    
    return augmented_paths

def enhance_contrast_simple(img: Image.Image):
    """Simple contrast enhancement without heavy processing"""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(random.uniform(1.1, 1.4))

def smart_zoom(img: Image.Image, original_size: tuple, zoom_range=(0.7, 1.0)):
    """Smart zoom that preserves important regions - works with original image size"""
    w, h = original_size
    scale = random.uniform(zoom_range[0], zoom_range[1])
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Try to avoid always centering - random offsets
    max_offset_x = max(0, w - new_w)
    max_offset_y = max(0, h - new_h)
    
    left = random.randint(0, max_offset_x)
    top = random.randint(0, max_offset_y)
    
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)  # Resize back to original size

def enhance_disease_features(img: Image.Image):
    """Enhance visibility of pale spots and small features"""
    from PIL import ImageEnhance, ImageFilter
    
    # Convert to numpy for processing
    arr = np.array(img)
    
    # Enhance contrast specifically in problematic ranges
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to lightness channel for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back
    lab = cv2.merge([l, a, b])
    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    img = Image.fromarray(arr)
    
    # Additional contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(1.1, 1.3))
    
    # Sharpening to enhance small features
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    return img

# -------------------------
# Build dataset & visualization
# -------------------------

def gather_image_paths_and_labels(dataset_dir):
    p = Path(dataset_dir)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    paths, labels = [], []
    for c in classes:
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            for f in (p / c).glob(ext):
                paths.append(str(f)); labels.append(c)
    df = pd.DataFrame({"filepath": paths, "label": labels})
    return df, classes

def plot_and_save_confusion(cm, classes, out_path):
    if plt is None or sns is None:
        return
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_metrics_json(metrics: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

# -------------------------
# tf.data pipeline helpers - RESIZING HAPPENS HERE
# -------------------------

def get_keras_augmentation_layer(cfg):
    L = keras.layers
    return keras.Sequential([
        L.RandomFlip("horizontal_and_vertical"),
        L.RandomRotation(cfg.get("keras_random_rotation", 0.12)),
        L.RandomZoom(cfg.get("keras_random_zoom", 0.12)),
        L.RandomTranslation(cfg.get("keras_random_translate", 0.06), cfg.get("keras_random_translate", 0.06)),
        L.RandomContrast(cfg.get("keras_random_contrast", 0.12)),
    ], name="keras_aug")

def make_tf_dataset_from_paths(df: pd.DataFrame, cfg: dict, batch_size: int, training: bool):
    img_size = int(cfg["image_size"])
    keras_aug = get_keras_augmentation_layer(cfg)
    filepaths = df["filepath"].astype(str).tolist()
    labels = df["label"].tolist()
    class_names = sorted(list(set(labels)))
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    y = [class_to_idx[l] for l in labels]

    ds = tf.data.Dataset.from_tensor_slices((filepaths, y))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        # RESIZE TO MODEL INPUT SIZE HERE
        img = tf.image.resize(img, [img_size, img_size])
        
        if training:
            img = tf.expand_dims(img, 0)
            img = keras_aug(img, training=True)
            img = tf.squeeze(img, 0)
        return img, label

    # MEMORY OPTIMIZED pipeline
    ds = ds.map(_load, num_parallel_calls=2)  # Reduced parallel calls
    ds = ds.apply(tf.data.experimental.ignore_errors())  # Skip corrupted images
    
    if training:
        ds = ds.shuffle(buffer_size=512)  # Reduced buffer size
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)  # Minimal prefetch
    return ds, class_to_idx, class_names

# -------------------------
# Model building (Feature extractor ONLY)
# -------------------------

def build_feature_extractor(img_size, pretrained="manual_rgb", logger=None):
    """Build EfficientNetB0 that processes multiple scales"""
    from tensorflow.keras import layers, Model
    
    base_model = keras.applications.EfficientNetB0(
        include_top=False, 
        weights="imagenet" if pretrained == "imagenet" else None,
        input_shape=(img_size, img_size, 3), 
        pooling=None  # No global pooling yet
    )
    
    if pretrained != "imagenet":
        weights_path = keras.utils.get_file(
            "efficientnetb0_notop_rgb.h5",
            "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5",
        )
        base_model.load_weights(weights_path)
    
    base_model.trainable = False
    
    # Multi-scale feature fusion
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Original scale features
    features_orig = base_model(inputs)
    
    # 2x zoom features (center crop)
    zoom_factor = 0.6
    new_size = int(img_size * zoom_factor)
    start = (img_size - new_size) // 2
    cropped = layers.Cropping2D(
        cropping=((start, start), (start, start))
    )(inputs)
    zoomed = layers.Resizing(img_size, img_size)(cropped)
    features_zoom = base_model(zoomed)
    
    # Concatenate multi-scale features
    combined = layers.Concatenate()([
        layers.GlobalAveragePooling2D()(features_orig),
        layers.GlobalAveragePooling2D()(features_zoom)
    ])
    
    model = Model(inputs=inputs, outputs=combined)
    return model

def extract_features_from_model(feature_model, dataset, logger, desc="extract"):
    feats, labs = [], []
    logger.info("Extracting features (%s)…", desc)
    for imgs, labels in tqdm(dataset, desc=f"[Features {desc}]"):
        f = feature_model.predict(imgs, verbose=0)
        feats.append(f)
        labs.append(labels.numpy())
    if len(feats) == 0:
        return np.zeros((0, feature_model.output_shape[-1])), np.array([], dtype=int)
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labs, axis=0)
    return X, y

# -------------------------
# SVM train, prune, eval
# -------------------------

def normalize_features(X_train, X_val, X_test, subs, logger):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, subs["weights"] / "feature_scaler.joblib")
    logger.info("Applied feature standardization")
    return X_train_scaled, X_val_scaled, X_test_scaled

def apply_color_normalization(img_array):
    """Normalize color distribution across dataset"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    # Normalize lightness channel
    lab[:,:,0] = cv2.normalize(lab[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def apply_smote_if_available(X, y, cfg, logger):
    if not cfg.get("enable_smote", True):
        logger.info("SMOTE disabled in config.")
        return X, y
    if SMOTE is None:
        logger.warning("imblearn.SMOTE not available; skipping SMOTE.")
        return X, y
    logger.info("Applying SMOTE to balance classes.")
    sm = SMOTE(random_state=cfg.get("seed", 42))
    Xb, yb = sm.fit_resample(X, y)
    logger.info("SMOTE done: %s -> %s", X.shape, Xb.shape)
    return Xb, yb

def train_svm_and_prune(X_train, y_train, X_val, y_val, cfg, subs, logger):
    """Legacy function - now uses the new tracking system"""
    svm, score = track_svm_training(X_train, y_train, X_val, y_val, cfg, subs, logger)
    return svm, X_val

# -------------------------
# Evaluate & save metrics
# -------------------------

def track_svm_training(X_train, y_train, X_val, y_val, cfg, subs, logger):
    """Track SVM training progress with validation metrics using progressive training"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weight = np.array([class_weights[y] for y in y_train])
    
    logger.info("Starting SVM training with progress tracking...")
    
    # Use progressive subset training as "epochs"
    return track_svm_with_subsets(X_train, y_train, X_val, y_val, sample_weight, cfg, subs, logger)

def track_svm_with_subsets(X_train, y_train, X_val, y_val, sample_weight, cfg, subs, logger):
    """Train SVM on progressively larger subsets to track learning progress"""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.svm import SVC
    
    # Create training subsets (10%, 20%, ..., 100%)
    subset_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_losses = []
    val_losses = []
    epochs = []
    
    best_score = 0
    best_svm = None
    
    # Final SVM with best parameters from config
    final_svm = SVC(
        kernel=cfg.get("svm_kernel", "rbf"),
        C=float(cfg.get("svm_C", 1.0)),
        gamma=cfg.get("svm_gamma", "scale"),
        probability=True,
        random_state=cfg.get("seed", 42)
    )
    
    logger.info("Training SVM on progressive data subsets...")
    
    for epoch, subset_size in enumerate(subset_sizes):
        # Select subset of training data
        subset_size_actual = int(len(X_train) * subset_size)
        indices = np.random.choice(len(X_train), subset_size_actual, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        sample_weight_subset = sample_weight[indices]
        
        # Train on subset
        epoch_svm = SVC(
            kernel=cfg.get("svm_kernel", "rbf"),
            C=float(cfg.get("svm_C", 1.0)),
            gamma=cfg.get("svm_gamma", "scale"),
            probability=True,
            random_state=cfg.get("seed", 42)
        )
        epoch_svm.fit(X_subset, y_subset, sample_weight=sample_weight_subset)
        
        # Also train final SVM on this subset (cumulative)
        if epoch == 0:
            final_svm.fit(X_subset, y_subset, sample_weight=sample_weight_subset)
        else:
            # For linear kernel, we can use partial_fit, for RBF we retrain on full data up to this point
            final_svm.fit(X_train[:subset_size_actual], y_train[:subset_size_actual], 
                         sample_weight=sample_weight[:subset_size_actual])
        
        # Predictions with epoch SVM
        y_train_pred = epoch_svm.predict(X_train)
        y_val_pred = epoch_svm.predict(X_val)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        # Approximate "loss" as 1 - accuracy
        train_loss = 1 - train_acc
        val_loss = 1 - val_acc
        
        # Store metrics
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs.append(epoch + 1)
        
        # Check if this is the best model
        if val_acc > best_score:
            best_score = val_acc
            best_svm = epoch_svm
        
        # Log progress
        logger.info("Epoch %2d/10 (%3d%%) - Train Acc: %.4f, Val Acc: %.4f, Val F1: %.4f, Train Loss: %.4f, Val Loss: %.4f",
                   epoch + 1, int(subset_size * 100), train_acc, val_acc, val_f1, train_loss, val_loss)
    
    # Train final SVM on full dataset
    final_svm.fit(X_train, y_train, sample_weight=sample_weight)
    y_val_pred_final = final_svm.predict(X_val)
    final_val_acc = accuracy_score(y_val, y_val_pred_final)
    
    logger.info("Final SVM validation accuracy: %.4f", final_val_acc)
    
    # Plot training history
    plot_svm_training_history(epochs, train_accuracies, val_accuracies, train_losses, 
                             val_losses, train_f1_scores, val_f1_scores, subs, logger)
    
    # Use the final model trained on full data
    return final_svm, final_val_acc

def plot_svm_training_history(epochs, train_acc, val_acc, train_loss, val_loss, 
                             train_f1, val_f1, subs, logger):
    """Create comprehensive training history plots for SVM"""
    if plt is None:
        logger.warning("Matplotlib not available, skipping training plots")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Accuracy
    ax1.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('SVM Training Progress - Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs (Data Subset %)')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax2.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('SVM Training Progress - Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs (Data Subset %)')
    ax2.set_ylabel('Loss (1 - Accuracy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: F1 Score
    ax3.plot(epochs, train_f1, 'b-', label='Training F1 Score', linewidth=2)
    ax3.plot(epochs, val_f1, 'r-', label='Validation F1 Score', linewidth=2)
    ax3.set_title('SVM Training Progress - F1 Score', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epochs (Data Subset %)')
    ax3.set_ylabel('F1 Score (Weighted)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined metrics
    ax4.plot(epochs, train_acc, 'b-', label='Train Accuracy', alpha=0.7)
    ax4.plot(epochs, val_acc, 'r-', label='Val Accuracy', alpha=0.7)
    ax4.plot(epochs, train_f1, 'g-', label='Train F1', alpha=0.7)
    ax4.plot(epochs, val_f1, 'orange', label='Val F1', alpha=0.7)
    ax4.set_title('SVM Training - All Metrics', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epochs (Data Subset %)')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = subs["plots"] / "svm_training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved SVM training history plot to %s", plot_path)

def evaluate_and_save(y_true, y_pred, classes, subs, logger):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    # save confusion matrix plot if seaborn available
    if plt is not None and sns is not None:
        plot_path = subs["plots"] / "confusion_matrix.png"
        plot_and_save_confusion(cm, classes, plot_path)
        logger.info("Saved confusion matrix to %s", plot_path)
    # summary metrics
    summ = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    }
    metrics_path = subs["metrics"] / "metrics_summary.json"
    save_metrics_json({"summary": summ, "classification_report": rep}, metrics_path)
    logger.info("Saved metrics JSON to %s", metrics_path)
    return summ, rep, cm

# -------------------------
# Orchestration: main flow
# -------------------------

# -------------------------
# Orchestration: main flow
# -------------------------

def main(args):
    cfg = load_config(args.config)
    # required keys check
    if "dataset_dir" not in cfg or "output_dir" not in cfg:
        raise ValueError("config must contain dataset_dir and output_dir")

    base_out = cfg["output_dir"]
    out_base, subs = make_output_dirs(base_out)
    logger = setup_logger(subs["log"])
    logger.info("Output base: %s", out_base)

    # Create class distribution plot for original dataset (before any processing)
    try:
        df_original, classes_original = gather_image_paths_and_labels(cfg["dataset_dir"])
        # Check if matplotlib is available before plotting
        if plt is not None:
            plt.figure(figsize=(10, 6))
            df_original["label"].value_counts().sort_index().plot(kind="bar", title="Class Distribution (Original Dataset)")
            plt.xlabel("Class"); plt.ylabel("Count"); plt.xticks(rotation=45)
            plt.tight_layout(); plt.savefig(subs["plots"] / "class_dist_before.png"); plt.close()
            logger.info("Saved original class distribution plot.")
        else:
            logger.info("Matplotlib not available, skipping class_dist_before.png")
    except Exception as e:
        logger.warning("Failed to create original class distribution plot: %s", e)

    # Delete previous run's CLAHE folder to save space (if present)
    prev = find_latest_clahe_folder(base_out)
    if prev:
        delete_prev_clahe(prev, logger)

    # Make fresh CLAHE in this run's output - WITHOUT RESIZING
    preprocess_clahe(cfg["dataset_dir"], subs["clahe_preprocessed"], logger,
                    clipLimit=cfg.get("clahe_clipLimit", 3.0),
                    tileGridSize=tuple(cfg.get("clahe_tileGridSize", (8,8))))
                    # Removed target_size parameter

    # Augmentation: balance dataset to reach total * augment_factor with equal class distribution
    augment_factor = cfg.get("augment_factor", 1)
    sample_augments_count = int(cfg.get("sample_augments_count", 10))
    target_size = int(cfg["image_size"])
    logger.info("augment_factor=%d, sample_augments_count=%d", augment_factor, sample_augments_count)

    # Collect all augmented samples for sampling later
    all_augmented_samples = []

    logger.info("Starting balanced augmentation")
    
    # Get initial dataset statistics
    df_before, classes = gather_image_paths_and_labels(subs["clahe_preprocessed"])
    
    # Calculate targets
    initial_total = len(df_before)
    target_total = math.ceil(initial_total * augment_factor)
    num_classes = len(classes)
    target_per_class = math.ceil(target_total / num_classes)
    
    logger.info("Initial total: %d, Target total: %d, Classes: %d, Target per class: %d", 
                initial_total, target_total, num_classes, target_per_class)
    
    # Process each class to reach target_per_class
    for cls in classes:
        cls_dir = subs["clahe_preprocessed"] / cls
        original_imgs = sorted([p for p in cls_dir.glob("*.jpg")] + [p for p in cls_dir.glob("*.png")])
        n_orig = len(original_imgs)
        
        logger.info("Class %s: original=%d target=%d", cls, n_orig, target_per_class)
        
        if n_orig == target_per_class:
            logger.info("  Class %s already at target, no changes needed", cls)
            continue
            
        elif n_orig > target_per_class:
            # Undersampling: randomly select target_per_class images and delete the rest
            logger.info("  Undersampling: removing %d images", n_orig - target_per_class)
            imgs_to_keep = random.sample(original_imgs, target_per_class)
            
            # Delete images not in the keep list
            for img_path in original_imgs:
                if img_path not in imgs_to_keep:
                    try:
                        img_path.unlink()
                    except Exception as e:
                        logger.warning("Failed to delete %s: %s", img_path, e)
            
        else:
            # Augmentation needed
            to_create = target_per_class - n_orig
            logger.info("  Augmentation: creating %d new images", to_create)
            
            # Create a working copy of original images for random selection
            available_imgs = original_imgs.copy()
            created_count = 0
            
            while created_count < to_create:
                if not available_imgs:
                    # Reset the available images if we've used them all
                    available_imgs = original_imgs.copy()
                
                # Pick a random image to augment
                src_img = random.choice(available_imgs)
                available_imgs.remove(src_img)  # Remove to avoid immediate reuse
                
                # Occasionally create mosaic (5% chance) if we have enough images
                if (created_count < to_create and 
                    random.random() < 0.05 and 
                    len(original_imgs) >= 4):
                    try:
                        mosaic_imgs = random.sample(original_imgs, 4)
                        mosaic_img = mosaic_combine(mosaic_imgs, target_size)
                        outp = cls_dir / f"mosaic_{created_count}_{int(time.time()*1000)}.jpg"
                        mosaic_img.save(outp, quality=90)
                        all_augmented_samples.append((outp, cls))
                        created_count += 1
                        continue
                    except Exception as e:
                        logger.debug("Mosaic failed: %s", e)
                
                # Regular single image augmentation - RESIZE TO TARGET SIZE ONLY WHEN SAVING
                augmented_paths = augment_image_and_save(src_img, cls_dir, target_size, n_aug=1, logger=logger)
                for aug_path in augmented_paths:
                    all_augmented_samples.append((aug_path, cls))
                    created_count += 1
                    if created_count >= to_create:
                        break
        
        logger.info("Class %s balancing complete", cls)
    
    logger.info("Balanced augmentation finished. Total augmented samples created: %d", len(all_augmented_samples))
    
    # Save random samples to augment_samples folder
    if all_augmented_samples and sample_augments_count > 0:
        # Shuffle and select samples, ensuring we don't exceed available samples
        random.shuffle(all_augmented_samples)
        samples_to_save = all_augmented_samples[:min(sample_augments_count, len(all_augmented_samples))]
        
        for i, (sample_path, cls) in enumerate(samples_to_save):
            try:
                dst_path = subs["augment_samples"] / f"sample_{i:03d}_{cls}_{sample_path.name}"
                shutil.copy2(sample_path, dst_path)
            except Exception as e:
                logger.warning("Failed to copy augmentation sample %s: %s", sample_path, e)
        
        logger.info("Saved %d augmentation samples to %s", len(samples_to_save), subs["augment_samples"])

    # gather dataset after augmentation
    df_all, classes = gather_image_paths_and_labels(subs["clahe_preprocessed"])
    if df_all.empty:
        raise RuntimeError("No images found after CLAHE+augmentation in " + str(subs["clahe_preprocessed"]))
    logger.info("Total images after augment: %d", len(df_all))
    logger.info("Per-class counts:\n%s", df_all["label"].value_counts().to_dict())

    # Split dataset: stratified train/val/test
    val_split = float(cfg.get("validation_split", 0.2))
    test_split = float(cfg.get("test_split", 0.1))
    seed = int(cfg.get("seed", 42))
    from sklearn.model_selection import train_test_split
    test_size = val_split + test_split
    train_df, temp_df = train_test_split(df_all, test_size=test_size, stratify=df_all["label"], random_state=seed)
    val_rel = val_split / test_size if test_size > 0 else 0.5
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_rel, stratify=temp_df["label"], random_state=seed)
    logger.info("Split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # Save partition pie and class distributions (simple safe plots)
    try:
        # Check if matplotlib is available
        if plt is not None:
            plt.figure(figsize=(4,4))
            counts = [len(train_df), len(val_df), len(test_df)]
            labels = [f"train ({counts[0]})", f"val ({counts[1]})", f"test ({counts[2]})"]
            if sum(counts) > 0:
                plt.pie(counts, labels=labels, autopct="%1.1f%%"); plt.title("Dataset partition")
                plt.savefig(subs["plots"] / "partition_pie.png"); plt.close()
            
            # class dist after augmentation (entire dataset used for training/val/test)
            plt.figure(figsize=(10, 6))
            df_all["label"].value_counts().sort_index().plot(kind="bar", title="Class Distribution (After Augmentation - Entire Dataset)")
            plt.xlabel("Class"); plt.ylabel("Count"); plt.xticks(rotation=45)
            plt.tight_layout(); plt.savefig(subs["plots"] / "class_dist_after.png"); plt.close()
        else:
            logger.info("Matplotlib not available, skipping plots")
    except Exception as e:
        logger.warning("Plotting failed: %s", e)

    # Build tf.data datasets - RESIZING HAPPENS HERE
    batch_size = int(cfg.get("batch_size", 16))
    train_ds, class_to_idx, class_names = make_tf_dataset_from_paths(train_df, cfg, batch_size, training=True)
    val_ds, _, _ = make_tf_dataset_from_paths(val_df, cfg, batch_size, training=False)
    test_ds, _, _ = make_tf_dataset_from_paths(test_df, cfg, batch_size, training=False)

    # Build feature extractor (EfficientNetB0 as fixed feature extractor)
    img_size = int(cfg["image_size"])
    feature_extractor = build_feature_extractor(img_size, pretrained=cfg.get("pretrained_weights","manual_rgb"), logger=logger)
    
    # Save feature extractor model
    feature_extractor.save(subs["weights"] / "efficientnetb0_feature_extractor.keras")
    logger.info("Saved EfficientNetB0 feature extractor.")

    # Save model info JSON
    model_info = {
        "model_name": "EfficientNetB0 + SVM Hybrid",
        "input_size": img_size,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "pretrained_weights": cfg.get("pretrained_weights", "manual_rgb"),
        "feature_extractor": "EfficientNetB0",
        "classifier": "SVM",
        "svm_config": {
            "kernel": cfg.get("svm_kernel", "rbf"),
            "C": cfg.get("svm_C", 1.0),
            "gamma": cfg.get("svm_gamma", "scale")
        }
    }
    with open(subs["weights"] / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info("Saved model info to weights folder.")

    # Extract features (train/val/test)
    X_train, y_train = extract_features_from_model(feature_extractor, train_ds, logger, desc="train")
    X_val, y_val = extract_features_from_model(feature_extractor, val_ds, logger, desc="val")
    X_test, y_test = extract_features_from_model(feature_extractor, test_ds, logger, desc="test")
    logger.info("Feature shapes: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)

    # NORMALIZE FEATURES
    X_train, X_val, X_test = normalize_features(X_train, X_val, X_test, subs, logger)

    # SMOTE balancing (on training features)
    Xb, yb = apply_smote_if_available(X_train, y_train, cfg, logger)

    # Train SVM with progress tracking
    logger.info("Starting SVM training with live progress tracking...")
    svm_model, best_val_score = track_svm_training(Xb, yb, X_val, y_val, cfg, subs, logger)

    # Save the trained SVM
    joblib.dump(svm_model, subs["weights"] / "svm_classifier.joblib")
    logger.info("Saved SVM classifier to weights folder. Best validation score: %.4f", best_val_score)

    # Evaluate SVM on test features
    y_pred = svm_model.predict(X_test)
    summ, rep, cm = evaluate_and_save(y_test, y_pred, class_names, subs, logger)

    logger.info("Final summary: %s", summ)
    logger.info("Classification report saved in metrics JSON.")

    # Save prediction samples with true and predicted labels
    preds_idx = list(map(int, y_pred))
    true_idx = list(map(int, y_test))
    out_pred_dir = subs["prediction_samples"]
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    
    # build mapping from filepath to label for test_df
    test_paths = test_df["filepath"].tolist()
    
    for i, p in enumerate(test_paths[: min(len(test_paths), int(cfg.get("sample_predictions_count", 20)))]):
        pred_label = class_names[preds_idx[i]]
        true_label = class_names[true_idx[i]]
        
        try:
            # Open image and add text labels - RESIZE TO TARGET SIZE FOR DISPLAY
            img = Image.open(p).convert("RGB")
            img = img.resize((img_size, img_size))
            
            # Create a copy with labels
            img_with_text = img.copy()
            draw = ImageDraw.Draw(img_with_text)
            
            # Try to use a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Add text background for better readability
            text_bg_height = 60
            draw.rectangle([0, 0, img_size, text_bg_height], fill="black")
            
            # Add true and predicted labels
            draw.text((10, 10), f"True: {true_label}", fill="white", font=font)
            draw.text((10, 35), f"Pred: {pred_label}", fill="white", font=font)
            
            # Color code based on correctness
            if true_label == pred_label:
                draw.rectangle([img_size-30, 10, img_size-10, 30], fill="green")
            else:
                draw.rectangle([img_size-30, 10, img_size-10, 30], fill="red")
            
            # Save the image
            out_path = out_pred_dir / f"pred_{i:03d}_{true_label}_vs_{pred_label}.jpg"
            img_with_text.save(out_path, quality=90)
            
        except Exception as e:
            logger.warning("Failed to create prediction sample %s: %s", p, e)

    logger.info("Saved prediction samples to %s", out_pred_dir)

    # Save final config snapshot and metrics
    with open(subs["metrics"] / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)

    logger.info("Run complete. Outputs in %s", out_base)
    print("Complete. Outputs in:", out_base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args)
