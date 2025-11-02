#!/usr/bin/env python3
"""
efficientnet_b0_svm_v1_3.py

EfficientNetB0 + SVM pipeline (v1.3)
- Fresh CLAHE per run (old clahe_preprocessed deleted)
- Augmentation to reach dataset * augment_factor
- manual_rgb pretrained weights
- SMOTE optional
- Metrics JSON saved
"""

import argparse
import datetime
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps, ImageFilter
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.svm import SVC
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
        "features": base / "features",
        "svm": base / "svm"
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
    """
    Return the most recent effb0svm_* Path that contains clahe_preprocessed and is not the new folder.
    We keep function as-is; caller will delete previous clahe folder content.
    """
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
    if pil.mode == "RGBA":
        bg = Image.new("RGBA", pil.size, (255,255,255,255))
        bg.paste(pil, mask=pil.split()[3])
        return bg.convert("RGB")
    return pil.convert("RGB")

def apply_clahe_pil(pil_img, clipLimit=3.0, tileGridSize=(8,8)):
    """Apply CLAHE to a PIL image (RGB) and return uint8 RGB ndarray."""
    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    out = np.stack([clahe.apply(arr[..., c]) for c in range(3)], axis=-1)
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
                # cv2.imwrite expects BGR
                cv2.imwrite(str(outp), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            except Exception as e:
                logger.warning("Failed CLAHE on %s: %s", f, e)
    logger.info("CLAHE preprocessing finished.")

# -------------------------
# Augmentation helpers
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

def mosaic_combine(image_paths, out_size):
    """
    Simple mosaic: pick up to 4 images from the same class, resize to half, tile in 2x2.
    image_paths: list of paths (>=1)
    """
    imgs = []
    for i in range(4):
        p = image_paths[i % len(image_paths)]
        img = Image.open(p).convert("RGB").resize((out_size//2, out_size//2), Image.BICUBIC)
        imgs.append(img)
    new = Image.new("RGB", (out_size, out_size))
    new.paste(imgs[0], (0,0))
    new.paste(imgs[1], (out_size//2,0))
    new.paste(imgs[2], (0,out_size//2))
    new.paste(imgs[3], (out_size//2,out_size//2))
    return new

def augment_image_and_save(img_path: Path, save_dir: Path, image_size: int, n_aug=1, logger=None):
    """
    Given a source image path, generate n_aug variants and save them in save_dir.
    The source is not deleted; we only save augmented copies.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        orig = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    except Exception as e:
        if logger: logger.warning("Failed to open %s for augmentation: %s", img_path, e)
        return
    for i in range(n_aug):
        img = orig.copy()
        # Apply random pipeline
        if random.random() < 0.7:
            img = random_rotation(img, max_deg=25)
        if random.random() < 0.7:
            img = random_crop(img, crop_scale_min=0.85)
        if random.random() < 0.6:
            img = Image.fromarray(salt_pepper(np.array(img), amount=0.003))
        if random.random() < 0.6:
            img = gaussian_blur(img, max_radius=1.2)
        if random.random() < 0.7:
            img = random_contrast(img, 0.85, 1.25)
        # Save
        outp = save_dir / f"{img_path.stem}_aug_{i}.jpg"
        img.save(outp, quality=90)

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
# tf.data pipeline helpers
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
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.image.convert_image_dtype(img, tf.float32)  # 0-1
        # normalize min-max: already 0-1, keep
        img.set_shape([img_size, img_size, 3])
        if training:
            img = tf.expand_dims(img, 0)
            img = keras_aug(img, training=True)
            img = tf.squeeze(img, 0)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=min(2048, max(1000, len(filepaths))))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, class_to_idx, class_names

# -------------------------
# Model, training, features
# -------------------------

def build_finetune_model(img_size, num_classes, pretrained="manual_rgb", logger=None):
    from tensorflow.keras import layers, models, regularizers, backend as K
    K.clear_session()
    if pretrained == "imagenet":
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                    input_shape=(img_size,img_size,3), pooling="avg")
        if logger: logger.info("Loaded EfficientNetB0 weights='imagenet'")
    else:
        base = keras.applications.EfficientNetB0(include_top=False, weights=None,
                                                    input_shape=(img_size,img_size,3), pooling="avg")
        # manual rgb weights (notop)
        weights_path = keras.utils.get_file(
            "efficientnetb0_notop_rgb.h5",
            "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5",
        )
        base.load_weights(weights_path)
        if logger: logger.info("Loaded EfficientNetB0 manual_rgb weights from %s", weights_path)

    x = base.output
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation="swish", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    preds = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=preds)
    return model, base

def train_finetune(model, train_ds, val_ds, cfg, logger):
    # initial head-only training
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-20:]:
        layer.trainable = True

    lr = float(cfg.get("learning_rate", 1e-3))
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    opt = keras.optimizers.Adam(lr)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.get("early_stop_patience", 8), restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]
    logger.info("Starting head training: epochs=%s lr=%s", cfg.get("epochs", 10), lr)
    history_head = model.fit(train_ds, validation_data=val_ds, epochs=int(cfg.get("epochs", 10)),
                             callbacks=callbacks, verbose=1)

    # fine-tune: unfreeze top fraction
    total_layers = len(model.layers)
    frac = float(cfg.get("fine_tune_unfreeze_frac", 0.25))
    unfreeze_from = max(0, int(total_layers * (1.0 - frac)))
    for i, layer in enumerate(model.layers):
        layer.trainable = True if i >= unfreeze_from else False
    lr_ft = float(cfg.get("fine_tune_learning_rate", 1e-4))
    logger.info("Starting fine-tune: unfreeze top %.2f%% layers (from %d), lr=%s", frac, unfreeze_from, lr_ft)
    opt_ft = keras.optimizers.Adam(lr_ft)
    model.compile(optimizer=opt_ft, loss=loss_fn, metrics=["accuracy"])
    history_ft = model.fit(train_ds, validation_data=val_ds, epochs=int(cfg.get("fine_tune_epochs", 5)),
                           callbacks=callbacks, verbose=1)
    # combine histories if needed (we'll just return both)
    return history_head, history_ft

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

def train_svm_and_prune(X_train, y_train, X_val, X_test, cfg, subs, logger):
    svm = SVC(kernel=cfg.get("svm_kernel","rbf"), C=float(cfg.get("svm_C",1.0)),
              gamma=cfg.get("svm_gamma","scale"), probability=True, random_state=cfg.get("seed",42))
    logger.info("Training SVM on features: %s", X_train.shape)
    svm.fit(X_train, y_train)
    joblib.dump(svm, subs["svm"] / "svm_initial.joblib")
    logger.info("Saved initial SVM.")
    return svm, X_test

# -------------------------
# Evaluate & save metrics
# -------------------------

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

def main(args):
    cfg = load_config(args.config)
    # required keys check
    if "dataset_dir" not in cfg or "output_dir" not in cfg:
        raise ValueError("config must contain dataset_dir and output_dir")

    base_out = cfg["output_dir"]
    out_base, subs = make_output_dirs(base_out)
    logger = setup_logger(subs["log"])
    logger.info("Output base: %s", out_base)

    # Delete previous run's CLAHE folder to save space (if present)
    prev = find_latest_clahe_folder(base_out)
    if prev:
        delete_prev_clahe(prev, logger)

    # Make fresh CLAHE in this run's output
    preprocess_clahe(cfg["dataset_dir"], subs["clahe_preprocessed"], logger,
                     clipLimit=cfg.get("clahe_clipLimit", 3.0),
                     tileGridSize=tuple(cfg.get("clahe_tileGridSize", (8,8))))

    # Augmentation: for each image produce augment_factor-1 images saved in-place (clahe_preprocessed class folders)
    augment_factor = int(cfg.get("augment_factor", 1))
    logger.info("augment_factor=%d", augment_factor)
    if augment_factor > 1:
        logger.info("Starting augmentation to reach dataset * %d", augment_factor)
        # For each class, compute target count = original_count * augment_factor
        df_before, classes = gather_image_paths_and_labels(subs["clahe_preprocessed"])
        by_class = df_before["label"].value_counts().to_dict()
        for cls in classes:
            cls_dir = subs["clahe_preprocessed"] / cls
            imgs = sorted([p for p in cls_dir.glob("*.jpg")] + [p for p in cls_dir.glob("*.png")])
            n_orig = len(imgs)
            if n_orig == 0:
                logger.warning("No images for class %s; skipping augmentation", cls)
                continue
            target = int(n_orig * augment_factor)
            to_create = target - n_orig
            logger.info("Class %s: original=%d target=%d to_create=%d", cls, n_orig, target, to_create)
            # generate augmentations in loop; reuse random images and mosaic sometimes
            idx = 0
            while to_create > 0:
                src = imgs[idx % n_orig]
                # occasionally create mosaic from other images in same class
                if to_create >= 1 and random.random() < 0.12 and n_orig >= 4:
                    try:
                        mosaic_img = mosaic_combine(imgs[idx:idx+4] if idx+4 <= n_orig else imgs[:4], int(cfg["image_size"]))
                        outp = cls_dir / f"mosaic_{idx}_{to_create}.jpg"
                        mosaic_img.save(outp, quality=90)
                        to_create -= 1
                        idx += 1
                        continue
                    except Exception as e:
                        logger.debug("Mosaic failed: %s", e)
                # else augment single
                augment_image_and_save(src, cls_dir, int(cfg["image_size"]), n_aug=1, logger=logger)
                to_create -= 1
                idx += 1
        logger.info("Augmentation finished.")

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
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4,4))
        counts = [len(train_df), len(val_df), len(test_df)]
        labels = [f"train ({counts[0]})", f"val ({counts[1]})", f"test ({counts[2]})"]
        if sum(counts) > 0:
            plt.pie(counts, labels=labels, autopct="%1.1f%%"); plt.title("Dataset partition")
            plt.savefig(subs["plots"] / "partition_pie.png"); plt.close()
        # class dist before
        df_all["label"].value_counts().sort_index().plot(kind="bar", title="Class distribution (after augment)")
        plt.tight_layout(); plt.savefig(subs["plots"] / "class_dist_after.png"); plt.close()
    except Exception as e:
        logger.warning("Plotting failed: %s", e)

    # Build tf.data datasets
    batch_size = int(cfg.get("batch_size", 16))
    train_ds, class_to_idx, class_names = make_tf_dataset_from_paths(train_df, cfg, batch_size, training=True)
    val_ds, _, _ = make_tf_dataset_from_paths(val_df, cfg, batch_size, training=False)
    test_ds, _, _ = make_tf_dataset_from_paths(test_df, cfg, batch_size, training=False)

    # Build/compile model
    img_size = int(cfg["image_size"])
    model, base = build_finetune_model(img_size, len(class_names), pretrained=cfg.get("pretrained_weights","manual_rgb"), logger=logger)
    # compile head before training (train function compiles later for fine-tune)
    model.compile(optimizer=keras.optimizers.Adam(float(cfg.get("learning_rate",1e-3))),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    # Train & fine-tune
    history_head, history_ft = train_finetune(model, train_ds, val_ds, cfg, logger)
    # save model
    model.save(subs["weights"] / "effnet_finetuned.keras")
    logger.info("Saved finetuned model.")

    # Build feature extractor (use base pooling output)
    # Since base was created with pooling='avg' in build_finetune_model, it might not be accessible directly; we use model.layers to find a layer with shape (None, channels) and treat it as feature output
    # Simpler: create a model that outputs the layer before final Dense: model.layers[-3] used earlier
    feat_layer = model.layers[-3].output if len(model.layers) >= 3 else model.layers[-2].output
    feat_model = keras.Model(inputs=model.input, outputs=feat_layer)

    # Extract features (train/val/test)
    X_train, y_train = extract_features_from_model(feat_model, train_ds, logger, desc="train")
    X_val, y_val = extract_features_from_model(feat_model, val_ds, logger, desc="val")
    X_test, y_test = extract_features_from_model(feat_model, test_ds, logger, desc="test")
    logger.info("Feature shapes: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)

    # SMOTE balancing (on training features)
    Xb, yb = apply_smote_if_available(X_train, y_train, cfg, logger)

    # Train SVM
    svm_model, X_test_for_eval = train_svm_and_prune(Xb, yb, X_val, X_test, cfg, subs, logger)

    # Evaluate SVM on test features
    y_pred = svm_model.predict(X_test_for_eval)
    summ, rep, cm = evaluate_and_save(y_test, y_pred, class_names, subs, logger)

    logger.info("Final summary: %s", summ)
    logger.info("Classification report saved in metrics JSON.")

    # Save some prediction samples (copy test images with predicted labels to prediction_samples)
    preds_idx = list(map(int, y_pred))
    out_pred_dir = subs["prediction_samples"]
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    # build mapping from filepath to label for test_df
    test_paths = test_df["filepath"].tolist()
    for i, p in enumerate(test_paths[: min(len(test_paths), int(cfg.get("sample_predictions_count", 20)))]):
        pred_label = class_names[preds_idx[i]]
        dst = out_pred_dir / f"{pred_label}_{Path(p).name}"
        try:
            shutil.copy(p, dst)
        except Exception as e:
            logger.warning("Could not copy prediction sample %s: %s", p, e)

    # Save final config snapshot and metrics
    with open(subs["metrics"] / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)

    logger.info("Run complete. Outputs in %s", out_base)
    print("Complete. Outputs in:", out_base)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to YAML config")
    args = p.parse_args()
    main(args)
