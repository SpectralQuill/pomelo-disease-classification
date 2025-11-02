#!/usr/bin/env python3
"""
Updated EfficientNet-B0 + SVM pipeline
Implements:
 - YAML-config driven pipeline (required --config)
 - RGBA -> RGB conversion
 - Augmentation suite (Keras + custom): contrast, clipping, rotate, blur,
   salt-and-pepper, random crop, mosaic, random cropping/resizing
 - Augment_factor to produce dataset_size * augment_factor total samples
 - Augmentation fusion (compose multiple aug transforms)
 - Image enhancement: histogram equalization + CLAHE
 - Min-max normalization (0..1)
 - SMOTE-based balancing in feature space (configurable)
 - Fine-tune EfficientNet-B0 as feature extractor
 - SVM training and RFE-based post-pruning (feature selection), then final SVM
 - Evaluation + saved artifacts
Notes:
 - CLI requires --config (no defaults)
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import random
import json
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image

# scikit-learn / imblearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE
import joblib

# imblearn (SMOTE)
try:
    from imblearn.over_sampling import SMOTE
except Exception as e:
    SMOTE = None

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.efficientnet import preprocess_input

# plotting (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Helpers
# -----------------------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def make_output_dirs(base_out):
    ts = datetime.now().strftime("effb0svm_%Y%m%d_%H%M%S")
    out = Path(base_out) / ts
    subs = {
        "plots": out / "plots",
        "metrics": out / "metrics",
        "weights": out / "weights",
        "logs": out / "logs",
        "features": out / "features",
        "samples": out / "samples",
        "model_vis": out / "model_vis",
    }
    for v in subs.values():
        v.mkdir(parents=True, exist_ok=True)
    return out, subs

# -----------------------------
# Image utilities (RGBA->RGB, resize, enhancements)
# -----------------------------
def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert PIL Image to RGB. Preserve alpha channel separately if present."""
    if img.mode == "RGBA":
        # Composite on white background (keeps alpha effect); convert to RGB
        background = Image.new("RGBA", img.size, (255,255,255,255))
        background.paste(img, mask=img.split()[3])  # 3 is alpha
        return background.convert("RGB")
    elif img.mode == "LA":  # grayscale with alpha
        return img.convert("RGB")
    else:
        return img.convert("RGB")

def pil_to_np(img: Image.Image, image_size):
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2] > 3:
        arr = arr[..., :3]
    return arr

def load_image_to_array(path, image_size):
    try:
        img = Image.open(path)
        img = ensure_rgb(img)
        arr = pil_to_np(img, image_size)
        return arr
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return np.zeros((image_size, image_size, 3), dtype=np.float32)

def save_img(arr, path):
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR))

# -----------------------------
# Enhancement: histogram eq / CLAHE
# -----------------------------
def apply_hist_eq_rgb(img_arr):
    """Apply histogram equalization per channel (simple). img_arr uint8 or float."""
    img = img_arr.copy().astype(np.uint8)
    out = np.zeros_like(img)
    for ch in range(3):
        out[..., ch] = cv2.equalizeHist(img[..., ch])
    return out.astype(np.float32)

def apply_clahe_rgb(img_arr, clipLimit=2.0, tileGridSize=(8,8)):
    img = img_arr.copy().astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    out = np.zeros_like(img)
    for ch in range(3):
        out[..., ch] = clahe.apply(img[..., ch])
    return out.astype(np.float32)

# -----------------------------
# Augmentations (Keras + custom)
# -----------------------------
def get_keras_augmentation(image_size, cfg):
    """Return a keras Sequential of augmentation layers; more can be added from cfg."""
    layers_list = []
    # use flips / rotations / zooms etc (keras preprocessing)
    layers_list.append(layers.RandomFlip("horizontal_and_vertical"))
    layers_list.append(layers.RandomRotation(cfg.get("keras_random_rotation", 0.12)))
    layers_list.append(layers.RandomZoom(cfg.get("keras_random_zoom", 0.12)))
    layers_list.append(layers.RandomTranslation(cfg.get("keras_random_translate", 0.06), cfg.get("keras_random_translate", 0.06)))
    layers_list.append(layers.RandomContrast(cfg.get("keras_random_contrast", 0.12)))
    return keras.Sequential(layers_list, name="keras_augmentation")

def salt_and_pepper(img, amount=0.005):
    img = img.copy()
    h, w, c = img.shape
    num = int(amount * h * w)
    # salt
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    img[ys, xs] = 255
    # pepper
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    img[ys, xs] = 0
    return img

def random_gaussian_blur(img, max_ksize=5, p=0.35):
    if random.random() > p:
        return img
    k = random.choice([1,3,5])
    if k == 1:
        return img
    return cv2.GaussianBlur(img.astype(np.uint8), (k,k), 0).astype(np.float32)

def random_contrast_clip(img, contrast_range=(0.7, 1.3), clip_percent=0.01, p=0.5):
    if random.random() > p:
        return img
    factor = random.uniform(*contrast_range)
    img2 = img * factor
    lo = np.percentile(img2, clip_percent*100)
    hi = np.percentile(img2, 100 - clip_percent*100)
    img2 = np.clip(img2, lo, hi)
    # rescale back
    img2 = (img2 - img2.min()) / (max(1e-8, img2.max()-img2.min())) * 255.0
    return img2

def random_crop_and_resize(img, min_scale=0.7, p=0.4, target_size=None):
    if random.random() > p:
        return img
    h, w = img.shape[:2]
    scale = random.uniform(min_scale, 1.0)
    ch = int(h * scale)
    cw = int(w * scale)
    if ch < 2 or cw < 2:
        return img
    y0 = random.randint(0, h-ch)
    x0 = random.randint(0, w-cw)
    crop = img[y0:y0+ch, x0:x0+cw]
    if target_size is None:
        return crop
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA).astype(np.float32)

def mosaic_augment(paths_for_mosaic, image_size):
    """Given 4 image paths (or fewer), return a 2x2 mosaic image sized image_size x image_size."""
    imgs = []
    for p in paths_for_mosaic:
        try:
            img = Image.open(p)
            img = ensure_rgb(img)
            imgs.append(pil_to_np(img, image_size//2))
        except:
            imgs.append(np.zeros((image_size//2, image_size//2, 3), dtype=np.float32))
    # If fewer than 4, pad with zeros
    while len(imgs) < 4:
        imgs.append(np.zeros((image_size//2, image_size//2, 3), dtype=np.float32))
    top = np.hstack([imgs[0], imgs[1]])
    bottom = np.hstack([imgs[2], imgs[3]])
    mosaic = np.vstack([top, bottom])
    mosaic = cv2.resize(mosaic.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return mosaic

def augmentation_fusion(img_arr, keras_aug=None, cfg=None, all_paths_for_mosaic=None):
    """Apply a chain of augmentations. Accepts numpy RGB float arrays 0..255."""
    out = img_arr.copy()
    # Keras augmentation (operates on batch)
    if keras_aug is not None and random.random() < cfg.get("p_keras_aug", 0.8):
        try:
            out = keras_aug(np.expand_dims(out, 0), training=True).numpy()[0]
        except Exception:
            pass
    # Random contrast and clipping
    out = random_contrast_clip(out, contrast_range=tuple(cfg.get("contrast_range", (0.7,1.3))), clip_percent=cfg.get("clip_percent", 0.01), p=cfg.get("p_contrast_clip", 0.5))
    # Random gaussian blur
    out = random_gaussian_blur(out, p=cfg.get("p_gaussian_blur", 0.35))
    # Salt and pepper
    if random.random() < cfg.get("p_salt_pepper", 0.15):
        out = salt_and_pepper(out, amount=cfg.get("salt_pepper_amount", 0.004))
    # Random crop/resize
    out = random_crop_and_resize(out, min_scale=cfg.get("random_crop_min_scale", 0.7), p=cfg.get("p_random_crop", 0.4), target_size=cfg.get("image_size"))
    # Optionally mosaic (with low prob) - requires other image paths
    if cfg.get("enable_mosaic", True) and all_paths_for_mosaic is not None and random.random() < cfg.get("p_mosaic", 0.08):
        # choose 3 other random images
        others = random.sample(all_paths_for_mosaic, min(3, len(all_paths_for_mosaic)))
        # include current image path if provided in list
        mosaic_paths = others
        out = mosaic_augment(mosaic_paths, cfg.get("image_size"))
    return out

# -----------------------------
# Dataset gathering + augmentation expansion
# -----------------------------
def gather_image_paths_and_labels(dataset_dir):
    dataset_dir = Path(dataset_dir)
    class_names = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    filepaths = []
    labels = []
    for cls in class_names:
        p = dataset_dir / cls
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
            for f in p.glob(ext):
                filepaths.append(str(f))
                labels.append(cls)
    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    return df, class_names

def expand_dataset_with_augmentation(df, cfg, subs):
    """
    Expand dataset so final count ≈ original_count * augment_factor.
    This function builds a new DataFrame where each row points to either the original image
    or an "on-the-fly" augmentation reference (we'll generate arrays later).
    For simplicity we will produce augmented images in-memory and optionally save samples.
    """
    image_size = cfg["image_size"]
    augment_factor = cfg.get("augment_factor", 1)
    if augment_factor <= 1:
        return df.copy()  # no expansion

    print(f"Expanding dataset by augment_factor={augment_factor} ...")
    all_paths = df['filepath'].tolist()
    target_total = int(len(df) * augment_factor)
    expanded_rows = []
    # Keep original images
    for _, r in df.iterrows():
        expanded_rows.append({"filepath": r['filepath'], "label": r['label'], "aug": False})

    # We'll generate augmented samples by class to aim for balanced augmentation
    idx = 0
    attempt_cap = target_total * 3
    while len(expanded_rows) < target_total and attempt_cap > 0:
        attempt_cap -= 1
        # choose a random base image
        row = df.sample(1).iloc[0]
        base_path = row['filepath']
        label = row['label']
        # we'll mark this entry as an augmentation to create later
        expanded_rows.append({"filepath": base_path, "label": label, "aug": True})
        idx += 1

    expanded_df = pd.DataFrame(expanded_rows)
    print(f"Expanded dataset: {len(expanded_df)} samples (target {target_total})")
    return expanded_df

def build_numpy_dataset_from_expanded(df_expanded, cfg, subs):
    """
    Memory-efficient dataset builder.
    Returns a tf.data.Dataset pipeline instead of a giant NumPy array.
    """

    image_size = cfg["image_size"]
    batch_size = cfg.get("batch_size", 32)
    use_tf_data = cfg.get("use_tf_data", True)

    if not use_tf_data:
        # Fallback: old in-memory behavior (float16 for safety)
        X, y = [], []
        keras_aug = get_keras_augmentation(image_size, cfg)
        all_paths = df_expanded["filepath"].unique().tolist()
        max_samples_to_save = int(cfg.get("sample_images_to_save", 0))
        step = max(1, len(df_expanded) // max_samples_to_save) if max_samples_to_save > 0 else None

        for idx, row in tqdm(df_expanded.iterrows(), total=len(df_expanded), desc="Building images (RAM)"):
            arr = load_image_to_array(row["filepath"], image_size)
            if cfg.get("enhance_hist_eq", True):
                arr = apply_clahe_rgb(arr) if cfg.get("use_clahe", True) else apply_hist_eq_rgb(arr)
            if bool(row.get("aug", False)):
                arr = augmentation_fusion(arr, keras_aug=keras_aug, cfg=cfg, all_paths_for_mosaic=all_paths)
            arr = cv2.resize(arr.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32)
            if step and idx % step == 0 and idx // step < max_samples_to_save:
                save_img(arr, subs["samples"] / f"sample_{idx}_{row['label']}.png")
            X.append(arr)
            y.append(row["label"])
        X = np.array(X, dtype=np.float16)
        y = np.array(y)
        return X, y

    # ------------------------------------------------------------------
    # TF.DATA STREAMING PIPELINE (Recommended)
    # ------------------------------------------------------------------
    filepaths = df_expanded["filepath"].astype(str).tolist()
    labels = df_expanded["label"].tolist()
    class_to_idx = {c: i for i, c in enumerate(sorted(set(labels)))}
    labels = [class_to_idx[l] for l in labels]
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    keras_aug = get_keras_augmentation(image_size, cfg)
    all_paths_tf = tf.constant(filepaths)

    def _load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.image.convert_image_dtype(img, tf.float32)  # 0–1 float32

        # Apply CLAHE/hist-eq via OpenCV on CPU (optional)
        if cfg.get("enhance_hist_eq", True):
            img_uint8 = tf.image.convert_image_dtype(img, tf.uint8)
            img_np = tf.numpy_function(
                lambda im: apply_clahe_rgb(im.numpy()) if cfg.get("use_clahe", True)
                else apply_hist_eq_rgb(im.numpy()), [img_uint8], tf.float32)
            img_np = tf.clip_by_value(img_np / 255.0, 0.0, 1.0)
            img = img_np

        # Random augmentations (fusion)
        if tf.random.uniform([]) < 0.8:
            img = keras_aug(img, training=True)

        # Normalize to [0,1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=min(1000, len(filepaths)), seed=cfg.get("seed", 42))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Optionally save deterministic sample subset
    max_samples_to_save = int(cfg.get("sample_images_to_save", 0))
    if max_samples_to_save > 0:
        save_dir = subs["samples"]
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, (imgs, lbls) in enumerate(ds.take(max_samples_to_save)):
            for j in range(imgs.shape[0]):
                fname = f"sample_{i*batch_size+j}_{lbls[j].numpy()}.png"
                arr = (imgs[j].numpy() * 255).astype(np.uint8)
                save_img(arr, save_dir / fname)

    return ds, class_to_idx

# -----------------------------
# Splitting and label encoding
# -----------------------------
def create_splits(df, seed, val_frac=0.1, test_frac=0.1):
    # use stratified splitting based on label counts
    train_df, temp_df = train_test_split(df, test_size=(val_frac+test_frac), stratify=df['label'], random_state=seed)
    rel = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(temp_df, test_size=(1-rel), stratify=temp_df['label'], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

# -----------------------------
# Model building (EfficientNetB0 finetune)
# -----------------------------
def build_finetune_model(img_size, num_classes, dropout_rate=0.4):
    """
    Builds EfficientNetB0 fine-tuning model with manually loaded RGB notop weights.
    This avoids stem_conv shape mismatch errors in some TensorFlow versions.
    """
    from tensorflow.keras import backend as K
    from tensorflow.keras import layers, regularizers, models
    from tensorflow.keras.applications import EfficientNetB0

    K.clear_session()

    # ⚙️ Build base model without top, no weights yet
    base = EfficientNetB0(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        pooling="avg",
        weights=None  # load manually below
    )

    # ✅ Load the official Keras EfficientNetB0 RGB weights explicitly
    weights_path = tf.keras.utils.get_file(
        "efficientnetb0_notop_rgb.h5",
        "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5",
    )
    base.load_weights(weights_path)

    # 🔧 Add custom classification head
    x = base.output
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    preds = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=preds)

    return model, base

# -----------------------------
# Training / evaluation helpers
# -----------------------------
def plot_history_and_save(history, out_plots, prefix="history"):
    h = history.history
    epochs = range(1, len(h['loss']) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, h.get('accuracy', []), label='train_acc')
    plt.plot(epochs, h.get('val_accuracy', []), label='val_acc')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(epochs, h.get('loss', []), label='train_loss')
    plt.plot(epochs, h.get('val_loss', []), label='val_loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.tight_layout()
    p = out_plots / f"{prefix}_acc_loss.png"
    plt.savefig(p); plt.close()
    return p

def save_classification_metrics(y_true, y_pred, class_names, out_metrics):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.title("Confusion Matrix")
    cm_path = out_metrics / "confusion_matrix.png"
    plt.savefig(cm_path); plt.close()
    rpt_json = out_metrics / "classification_report.json"
    with open(rpt_json, "w") as f:
        json.dump(report, f, indent=2)
    rpt_csv = out_metrics / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(rpt_csv)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    summary = {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1}
    with open(out_metrics / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return cm_path, rpt_json, out_metrics / "summary.json"

# -----------------------------
# Main pipeline
# -----------------------------
def main(args):
    cfg = load_config(args.config)
    # Basic seeds
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # outputs
    master_out, subs = make_output_dirs(cfg["output_dir"])
    print("Outputs will be saved to:", master_out)

    # gather dataset
    print("------------------------------------------------------------")
    print(f"📂 Dataset directory: {cfg['dataset_dir']}")
    print("------------------------------------------------------------")

    df, class_names = gather_image_paths_and_labels(cfg["dataset_dir"])
    if df.empty:
        raise SystemExit("No images found in dataset directory. Exiting.")
    print("Found classes:", class_names)
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    inv_label_map = {v:k for k,v in class_to_idx.items()}

    # Expand dataset per augment_factor
    df_expanded = expand_dataset_with_augmentation(df, cfg, subs)

    # Splits
    train_df0, val_df0, test_df0 = create_splits(df, seed, val_frac=cfg["validation_split"], test_frac=cfg["validation_split"])
    print(f"Base splits => train: {len(train_df0)}, val: {len(val_df0)}, test: {len(test_df0)}")

    # We will expand per split separately to avoid leakage (best practice).
    train_exp = df_expanded[df_expanded['label'].isin(train_df0['label']) & df_expanded['filepath'].isin(train_df0['filepath']) == False]
    # Simpler approach: filter expanded rows whose base path is in train_df0 etc.
    train_exp = df_expanded[df_expanded['filepath'].isin(train_df0['filepath'])]
    val_exp = df_expanded[df_expanded['filepath'].isin(val_df0['filepath'])]
    test_exp = df_expanded[df_expanded['filepath'].isin(test_df0['filepath'])]

    # Build numpy datasets (with augmentations applied for rows marked aug=True)
    print("Building training numpy arrays (augmentation applied where requested)...")
    X_train, y_train_labels = build_numpy_dataset_from_expanded(train_exp, cfg, subs)
    X_val, y_val_labels = build_numpy_dataset_from_expanded(val_exp, cfg, subs)
    X_test, y_test_labels = build_numpy_dataset_from_expanded(test_exp, cfg, subs)

    # Label encode
    y_train = np.array([class_to_idx[l] for l in y_train_labels])
    y_val = np.array([class_to_idx[l] for l in y_val_labels])
    y_test = np.array([class_to_idx[l] for l in y_test_labels])

    # Save dataset stats
    stats = {
        "total_images_raw": len(df),
        "total_images_expanded": len(df_expanded),
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test),
        "classes": class_names
    }
    with open(subs["logs"] / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Normalize images min-max to [0,1] (after any enhancement)
    def normalize_minmax(arr):
        arr = arr.astype(np.float32)
        lo = arr.min(axis=(1,2,3), keepdims=True)
        hi = arr.max(axis=(1,2,3), keepdims=True)
        denom = np.maximum(hi - lo, 1e-6)
        return (arr - lo) / denom

    X_train = normalize_minmax(X_train) if cfg.get("normalize_minmax", True) else X_train / 255.0
    X_val = normalize_minmax(X_val) if cfg.get("normalize_minmax", True) else X_val / 255.0
    X_test = normalize_minmax(X_test) if cfg.get("normalize_minmax", True) else X_test / 255.0

    # Preprocess for EfficientNet
    X_train_pp = preprocess_input((X_train * 255.0).astype(np.float32))
    X_val_pp = preprocess_input((X_val * 255.0).astype(np.float32))
    X_test_pp = preprocess_input((X_test * 255.0).astype(np.float32))

    # Build and train EfficientNet finetune model
    num_classes = len(class_names)
    model, base = build_finetune_model(cfg["image_size"], num_classes)
    try:
        keras.utils.plot_model(model, to_file=str(subs["model_vis"] / "model_architecture.png"), show_shapes=True, show_layer_names=True)
    except Exception:
        pass

    # Freeze base initially
    base.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=str(subs["weights"] / cfg.get("checkpoint_name", "effnet_ckpt.h5")),
                                        monitor='val_accuracy', save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.get("early_stop_patience", 6), restore_best_weights=True, verbose=1)
    ]

    print("Starting initial training (top head)...")
    history = model.fit(X_train_pp, y_train,
                        validation_data=(X_val_pp, y_val),
                        epochs=cfg["epochs"],
                        batch_size=cfg["batch_size"],
                        callbacks=callbacks,
                        verbose=1)

    plot_history_and_save(history, subs["plots"], prefix="initial")

    # Fine-tune
    print("Fine-tuning base model...")
    base.trainable = True
    fine_tune_at = int(len(base.layers) * cfg.get("fine_tune_unfreeze_frac", 0.25))
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(cfg["fine_tune_learning_rate"])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history_ft = model.fit(X_train_pp, y_train,
                           validation_data=(X_val_pp, y_val),
                           epochs=cfg["fine_tune_epochs"],
                           batch_size=cfg["batch_size"],
                           callbacks=callbacks,
                           verbose=1)
    plot_history_and_save(history_ft, subs["plots"], prefix="finetune")
    model.save(str(subs["weights"] / cfg.get("checkpoint_name", "effnet_finetuned.h5")))

    # Feature extraction using base (pooling='avg')
    feature_model = keras.models.Model(inputs=base.input, outputs=base.output)
    print("Extracting features...")
    features_train = feature_model.predict(X_train_pp, verbose=1)
    features_val = feature_model.predict(X_val_pp, verbose=1)
    features_test = feature_model.predict(X_test_pp, verbose=1)

    np.save(subs["features"] / "features_train.npy", features_train)
    np.save(subs["features"] / "features_val.npy", features_val)
    np.save(subs["features"] / "features_test.npy", features_test)
    np.save(subs["features"] / "y_train.npy", y_train)
    np.save(subs["features"] / "y_val.npy", y_val)
    np.save(subs["features"] / "y_test.npy", y_test)

    # SMOTE balancing in feature space (train only)
    if cfg.get("enable_smote", True):
        if SMOTE is None:
            print("[WARN] imblearn not available. Skipping SMOTE balancing.")
            features_train_bal, y_train_bal = features_train, y_train
        else:
            print("Applying SMOTE in feature space to balance classes...")
            # Determine target: either specified or max class count
            unique, counts = np.unique(y_train, return_counts=True)
            class_counts = dict(zip(unique.tolist(), counts.tolist()))
            target_per_class = cfg.get("smote_target_per_class", max(counts))
            # Build y labels for SMOTE (1D)
            sm = SMOTE(random_state=seed, sampling_strategy={int(k): int(target_per_class) for k in class_counts})
            try:
                features_train_bal, y_train_bal = sm.fit_resample(features_train, y_train)
                print("After SMOTE, training set size:", features_train_bal.shape[0])
            except Exception as e:
                print("[WARN] SMOTE failed:", e, " -> falling back to original train set")
                features_train_bal, y_train_bal = features_train, y_train
    else:
        features_train_bal, y_train_bal = features_train, y_train

    # SVM training (initial)
    print("Training initial SVM on features...")
    svm_clf = SVC(kernel=cfg.get("svm_kernel", "rbf"),
                  C=cfg.get("svm_C", 1.0),
                  gamma=cfg.get("svm_gamma", "scale"),
                  probability=True,
                  random_state=seed)
    svm_clf.fit(features_train_bal, y_train_bal)
    joblib.dump(svm_clf, subs["weights"] / "svm_classifier_initial.joblib")

    # Post-pruning of SVM via feature selection (RFE)
    if cfg.get("enable_svm_prune", True):
        print("Performing RFE-based feature selection (post-pruning)...")
        prune_percent = float(cfg.get("svm_prune_percent", 0.5))
        n_features = features_train_bal.shape[1]
        n_select = max(1, int(n_features * (1.0 - prune_percent)))
        # Use a fast linear estimator for RFE
        try:
            linear_est = LinearSVC(max_iter=5000, random_state=seed)
            selector = RFE(linear_est, n_features_to_select=n_select, step=0.1)
            selector = selector.fit(features_train_bal, y_train_bal)
            features_train_sel = selector.transform(features_train_bal)
            features_val_sel = selector.transform(features_val)
            features_test_sel = selector.transform(features_test)
            print(f"Selected {features_train_sel.shape[1]} features ({100.0*features_train_sel.shape[1]/n_features:.2f}% kept)")
            # Retrain SVM on selected features
            svm_pruned = SVC(kernel=cfg.get("svm_kernel", "rbf"),
                             C=cfg.get("svm_C", 1.0),
                             gamma=cfg.get("svm_gamma", "scale"),
                             probability=True,
                             random_state=seed)
            svm_pruned.fit(features_train_sel, y_train_bal)
            joblib.dump(svm_pruned, subs["weights"] / "svm_classifier_pruned.joblib")
            svm_clf = svm_pruned  # use pruned version for evaluation
            # store selector for later transform
            joblib.dump(selector, subs["weights"] / "feature_selector_rfe.joblib")
        except Exception as e:
            print("[WARN] RFE failed or LinearSVC not viable:", e)
            # fall back to original svm_clf
    else:
        print("Skipping SVM pruning.")

    # Evaluate
    # transform test features if selector exists
    selector_path = subs["weights"] / "feature_selector_rfe.joblib"
    if selector_path.exists():
        selector = joblib.load(selector_path)
        features_test_final = selector.transform(features_test)
    else:
        features_test_final = features_test

    y_pred_test = svm_clf.predict(features_test_final)
    cm_path, rpt_json, summary_path = save_classification_metrics(y_test, y_pred_test, class_names, subs["metrics"])
    print("Saved classification metrics & confusion matrix at", subs["metrics"])

    # Save model info
    with open(subs["weights"] / "model_info.json", "w") as f:
        info = {
            "class_names": class_names,
            "config": cfg,
            "feature_shape": features_train.shape[1]
        }
        json.dump(info, f, indent=2)

    eval_summary = {
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
    }
    pd.DataFrame([eval_summary]).to_csv(subs["metrics"] / "evaluation_summary.csv", index=False)

    # Save some sample predictions
    sample_dir = subs["plots"] / "sample_predictions"
    sample_dir.mkdir(exist_ok=True)
    for i in range(min(12, len(X_test))):
        img = (X_test[i] * 255.0).astype(np.uint8)
        feat = features_test[i:i+1]
        if selector_path.exists():
            feat = selector.transform(feat)
        pred = svm_clf.predict(feat)[0]
        pred_label = inv_label_map[pred]
        true_label = inv_label_map[int(y_test[i])]
        fig_path = sample_dir / f"sample_{i}_{pred_label}_true_{true_label}.png"
        plt.figure(figsize=(3,3)); plt.imshow(img.astype(np.uint8)); plt.axis('off'); plt.title(f"P:{pred_label} / T:{true_label}")
        plt.savefig(fig_path); plt.close()

    # Combined report
    with open(subs["logs"] / "full_report.txt", "w") as f:
        f.write("Dataset stats:\n")
        json.dump(stats, f, indent=2)
        f.write("\nEvaluation summary:\n")
        json.dump(eval_summary, f, indent=2)
    print("Full pipeline complete. Outputs in:", master_out)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientNetB0 + SVM pipeline (updated). Requires --config")
    parser.add_argument("--config", type=str, required=True, help="path to config yaml (required)")
    args = parser.parse_args()
    main(args)
