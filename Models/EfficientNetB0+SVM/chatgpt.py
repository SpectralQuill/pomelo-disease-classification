#!/usr/bin/env python3
"""
Citrus maxima (Pomelo) Disease Classification
EfficientNet-B0 (feature extractor) + SVM (classifier) hybrid pipeline.
Saves plots, metrics, weights, features, augmentation samples, etc.
"""

import os
import sys
import argparse
import shutil
import time
import math
from datetime import datetime
from pathlib import Path
import yaml
import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from rich.progress import track
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

import cv2

# -----------------------------
# Helpers: config + paths
# -----------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_output_dirs(base_out):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
# Image utilities
# -----------------------------
def load_image_to_array(path, image_size):
    """Load image as RGB even if it's grayscale."""
    try:
        # Force RGB mode, regardless of source format
        img = kimage.load_img(path, target_size=(image_size, image_size), color_mode="rgb")
        arr = kimage.img_to_array(img)

        # If any image somehow loads with 1 channel, expand to 3
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return arr
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        # Return a blank RGB image placeholder to keep dataset consistent
        return np.zeros((image_size, image_size, 3), dtype=np.float32)

def save_img(arr, path):
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR))

def display_and_save_sample(img_arr, label, save_path):
    plt.figure(figsize=(3,3))
    plt.imshow(np.clip(img_arr.astype("uint8"), 0, 255).astype(np.uint8))
    plt.axis('off')
    plt.title(label)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# -----------------------------
# Augmentations (Keras + custom)
# -----------------------------
def get_keras_augmentation(image_size):
    # Use Keras preprocessing layers for safe, deterministic augmentation components
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.06, 0.06),
        layers.RandomContrast(0.12),
    ], name="augmentation_pipeline")

def salt_and_pepper(img, amount=0.004):
    """Apply salt-and-pepper noise; img: float array 0-255"""
    img = img.copy()
    h, w, c = img.shape
    num = int(amount * h * w)
    # salt
    coords = [np.random.randint(0, i - 1, num) for i in (h, w)]
    img[coords[0], coords[1]] = 255
    # pepper
    coords = [np.random.randint(0, i - 1, num) for i in (h, w)]
    img[coords[0], coords[1]] = 0
    return img

def random_gaussian_blur(img, max_ksize=5, p=0.3):
    if random.random() > p:
        return img
    k = random.choice([1,3,5])  # kernel sizes
    if k == 1:
        return img
    return cv2.GaussianBlur(img.astype(np.uint8), (k,k), 0)

# -----------------------------
# Dataset loading + splitting
# -----------------------------
def gather_image_paths_and_labels(dataset_dir):
    dataset_dir = Path(dataset_dir)
    class_names = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    filepaths = []
    labels = []
    for i, cls in enumerate(class_names):
        p = dataset_dir / cls
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for f in p.glob(ext):
                filepaths.append(str(f))
                labels.append(cls)
    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    return df, class_names

def create_splits(df, seed, val_frac=0.1, test_frac=0.1):
    # First take 80% train, 20% temp
    train_df, temp_df = train_test_split(df, test_size=(val_frac+test_frac), stratify=df['label'], random_state=seed)
    # split temp into val and test equally (val_frac and test_frac are fractions of the total)
    rel = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(temp_df, test_size=(1-rel), stratify=temp_df['label'], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

# -----------------------------
# Feature extraction using EfficientNetB0
# -----------------------------
def build_finetune_model(img_size, num_classes, dropout_rate=0.4):
    from tensorflow.keras import backend as K
    K.clear_session()  # 🔥 Clears any previous model graph that used 1-channel input

    base = EfficientNetB0(include_top=False, input_shape=(img_size, img_size, 3),
                          pooling='avg', weights='imagenet')
    x = base.output
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    preds = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=base.input, outputs=preds)
    return model, base

# -----------------------------
# Training utilities
# -----------------------------
def plot_history_and_save(history, out_plots, prefix="history"):
    # history is returned by Keras model.fit
    h = history.history
    epochs = range(1, len(h['loss']) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, h['accuracy'], label='train_acc')
    plt.plot(epochs, h.get('val_accuracy', []), label='val_acc')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(epochs, h['loss'], label='train_loss')
    plt.plot(epochs, h.get('val_loss', []), label='val_loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.tight_layout()
    p = out_plots / f"{prefix}_acc_loss.png"
    plt.savefig(p); plt.close()
    return p

def save_classification_metrics(y_true, y_pred, class_names, out_metrics):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    # Save confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.title("Confusion Matrix")
    cm_path = out_metrics / "confusion_matrix.png"
    plt.savefig(cm_path); plt.close()
    # Save classification report to json and csv
    rpt_json = out_metrics / "classification_report.json"
    with open(rpt_json, "w") as f:
        json.dump(report, f, indent=2)
    rpt_csv = out_metrics / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(rpt_csv)
    # Basic summary
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    summary = {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1}
    with open(out_metrics / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return cm_path, rpt_json, out_metrics / "summary.json"

# -----------------------------
# Pipeline: build datasets arrays (in-memory)
# -----------------------------
def build_numpy_dataset(df, image_size, augment=False, aug_pipeline=None, sample_count=None, save_samples_dir=None):
    X = []
    y = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading images", unit="img"):
        try:
            arr = load_image_to_array(row['filepath'], image_size)
            if augment and aug_pipeline is not None:
                # apply Keras augmentations (batch dim required)
                arr_aug = aug_pipeline(np.expand_dims(arr, 0), training=True).numpy()[0]
                # apply a random salt & pepper and gaussian sometimes
                if random.random() < 0.2:
                    arr_aug = salt_and_pepper(arr_aug, amount=0.004)
                arr_aug = random_gaussian_blur(arr_aug, p=0.35)
                arr = arr_aug
                # optionally save sample
                if save_samples_dir is not None and random.random() < 0.04:
                    fp = save_samples_dir / f"aug_{Path(row['filepath']).stem}.png"
                    save_img(arr, fp)
            X.append(arr)
            y.append(row['label'])
        except Exception as e:
            print("Failed to load", row['filepath'], ":", e)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y

# -----------------------------
# Prediction utility
# -----------------------------
def predict_image(path, img_size, base_model, svm_clf, class_names, preprocess=True):
    arr = load_image_to_array(path, img_size)
    display = arr.copy()
    x = np.expand_dims(arr, 0)
    if preprocess:
        x = preprocess_input(x)
    # get features from base_model (pooling='avg' layer outputs shape [1, features])
    features = base_model.predict(x, verbose=0)
    if hasattr(svm_clf, "predict_proba"):
        probs = svm_clf.predict_proba(features)[0]
        pred_idx = np.argmax(probs)
        conf = float(probs[pred_idx])
    else:
        pred_idx = int(svm_clf.predict(features)[0])
        conf = None
    pred_label = class_names[pred_idx]
    return pred_label, conf, display

# -----------------------------
# Main
# -----------------------------
def main(args):
    cfg = load_config(args.config)
    np.random.seed(cfg.get("seed", 42))
    random.seed(cfg.get("seed", 42))
    tf.random.set_seed(cfg.get("seed", 42))

    # outputs
    master_out, subs = make_output_dirs(cfg["output_dir"])
    print("Outputs will be saved to:", master_out)

    # gather dataset
    df, class_names = gather_image_paths_and_labels(cfg["dataset_dir"])
    if df.empty:
        print("No images found in dataset directory. Exiting.")
        return
    print("Found classes:", class_names)
    class_to_idx = {c:i for i,c in enumerate(class_names)}

    # Save a small manifest
    df.sample(min(5, len(df))).to_csv(subs["logs"] / "sample_manifest.csv", index=False)

    # Make splits 80/10/10
    train_df, val_df, test_df = create_splits(df, cfg["random_state"], val_frac=cfg["validation_split"], test_frac=cfg["validation_split"])
    # Save partition plot
    part_counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    plt.figure(figsize=(4,4))
    plt.pie(list(part_counts.values()), labels=list(part_counts.keys()), autopct="%1.1f%%", colors=None, startangle=90)
    plt.title("Data Partition (Train/Val/Test)")
    plt.savefig(subs["plots"] / "data_partition_pie.png"); plt.close()

    # augmentation pipeline
    aug = get_keras_augmentation(cfg["image_size"])

    # Build in-memory numpy arrays (Note: you can adapt to tf.data generator if dataset is huge)
    print("Building training numpy arrays (with augmentation samples saved)...")
    X_train, y_train_labels = build_numpy_dataset(train_df, cfg["image_size"], augment=True, aug_pipeline=aug, save_samples_dir=subs["samples"])
    X_val, y_val_labels = build_numpy_dataset(val_df, cfg["image_size"], augment=False, aug_pipeline=aug, save_samples_dir=subs["samples"])
    X_test, y_test_labels = build_numpy_dataset(test_df, cfg["image_size"], augment=False, aug_pipeline=aug, save_samples_dir=subs["samples"])

    # Encode labels to ints
    label_map = class_to_idx
    inv_label_map = {v:k for k,v in label_map.items()}
    y_train = np.array([label_map[l] for l in y_train_labels])
    y_val = np.array([label_map[l] for l in y_val_labels])
    y_test = np.array([label_map[l] for l in y_test_labels])

    # Save dataset statistics
    stats = {
        "total_images": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "classes": class_names
    }
    with open(subs["logs"] / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Preprocess images for EfficientNet
    X_train_pp = preprocess_input(X_train.copy())
    X_val_pp = preprocess_input(X_val.copy())
    X_test_pp = preprocess_input(X_test.copy())

    # Build model
    num_classes = len(class_names)
    print("Building EfficientNetB0 finetune model...")
    model, base = build_finetune_model(cfg["image_size"], num_classes)
    # visualize model
    try:
        keras.utils.plot_model(model, to_file=str(subs["model_vis"] / "model_architecture.png"), show_shapes=True, show_layer_names=True)
        print("Saved model visualization.")
    except Exception as e:
        print("Could not create model visualiser image (pydot/graphviz missing?):", e)

    # Freeze base for initial training
    base.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=str(subs["weights"] / cfg.get("checkpoint_name", "effnet_ckpt.h5")),
                                        monitor='val_accuracy', save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    ]

    # Train top head (feature extractor frozen)
    print("Starting initial training (top classifier head)...")
    history = model.fit(X_train_pp, y_train,
                        validation_data=(X_val_pp, y_val),
                        epochs=cfg["epochs"],
                        batch_size=cfg["batch_size"],
                        callbacks=callbacks,
                        verbose=1)

    # plot and save history
    hist_path = plot_history_and_save(history, subs["plots"], prefix="initial")
    print("Saved training history plot to", hist_path)

    # Fine-tune: unfreeze some base layers and recompile
    print("Fine-tuning base model...")
    base.trainable = True
    # Optionally, freeze first N layers
    fine_tune_at = int(len(base.layers) * 0.75)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["fine_tune_learning_rate"]),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history_ft = model.fit(X_train_pp, y_train,
                           validation_data=(X_val_pp, y_val),
                           epochs=cfg["fine_tune_epochs"],
                           batch_size=cfg["batch_size"],
                           callbacks=callbacks,
                           verbose=1)

    plot_history_and_save(history_ft, subs["plots"], prefix="finetune")

    # Save final Keras weights
    model.save(str(subs["weights"] / cfg.get("checkpoint_name", "effnet_finetuned.h5")))
    print("Saved finetuned Keras model weights.")

    # Extract features using the *base* (pooling='avg') or using the trained model in a truncated form:
    # Use a model that outputs the pooling features (we created base with pooling='avg')
    feature_model = keras.models.Model(inputs=base.input, outputs=base.output)

    # Get features for train/val/test
    print("Extracting features with EfficientNetB0 pooling output...")
    features_train = feature_model.predict(X_train_pp, verbose=1)
    features_val = feature_model.predict(X_val_pp, verbose=1)
    features_test = feature_model.predict(X_test_pp, verbose=1)
    # Save features
    np.save(subs["features"] / "features_train.npy", features_train)
    np.save(subs["features"] / "features_val.npy", features_val)
    np.save(subs["features"] / "features_test.npy", features_test)
    np.save(subs["features"] / "y_train.npy", y_train)
    np.save(subs["features"] / "y_val.npy", y_val)
    np.save(subs["features"] / "y_test.npy", y_test)

    # Train SVM
    print("Training SVM on extracted features... This may take time for large datasets.")
    svm_clf = SVC(kernel=cfg.get("svm_kernel", "rbf"),
                  C=cfg.get("svm_C", 1.0),
                  gamma=cfg.get("svm_gamma", "scale"),
                  probability=True,
                  random_state=cfg.get("random_state", 42))

    # Fit with progress bar wrapper
    svm_clf.fit(features_train, y_train)
    joblib.dump(svm_clf, subs["weights"] / "svm_classifier.joblib")
    print("Saved SVM classifier.")

    # Evaluate with SVM
    y_pred_test = svm_clf.predict(features_test)
    # Save metrics and confusion matrix
    cm_path, rpt_json, summary_path = save_classification_metrics(y_test, y_pred_test, class_names, subs["metrics"])
    print("Saved classification metrics & confusion matrix at", subs["metrics"])

    # Save weights / model info
    with open(subs["weights"] / "model_info.json", "w") as f:
        info = {
            "class_names": class_names,
            "config": cfg,
            "feature_shape": features_train.shape[1]
        }
        json.dump(info, f, indent=2)

    # Save training and evaluation summary CSV
    eval_summary = {
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
    }
    pd.DataFrame([eval_summary]).to_csv(subs["metrics"] / "evaluation_summary.csv", index=False)

    # Plot sample predictions
    sample_dir = subs["plots"] / "sample_predictions"
    sample_dir.mkdir(exist_ok=True)
    for i in range(min(12, len(X_test))):
        img = X_test[i]
        feat = features_test[i:i+1]
        pred = svm_clf.predict(feat)[0]
        pred_label = inv_label_map[pred]
        true_label = inv_label_map[int(y_test[i])]
        fig_path = sample_dir / f"sample_{i}_{pred_label}_true_{true_label}.png"
        display_and_save_sample(img, f"P:{pred_label} / T:{true_label}", fig_path)

    print("Saved sample prediction plots.")

    # Save a combined report
    with open(subs["logs"] / "full_report.txt", "w") as f:
        f.write("Dataset stats:\n")
        json.dump(stats, f, indent=2)
        f.write("\nEvaluation summary:\n")
        json.dump(eval_summary, f, indent=2)
    print("Full pipeline complete. Outputs in:", master_out)

    # Example usage of predict_image - saves one sample
    test_path = test_df.iloc[0]['filepath']
    pred_label, conf, display_img = predict_image(test_path, cfg["image_size"], feature_model, svm_clf, class_names)
    print("Prediction example:", test_path, "->", pred_label, "confidence:", conf)
    # save overlay
    plt.figure(figsize=(4,4)); plt.imshow(display_img.astype(int)); plt.axis('off'); plt.title(f"{pred_label} ({conf:.3f})" if conf else pred_label)
    plt.savefig(subs["plots"] / "prediction_example.png"); plt.close()

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientNetB0 + SVM pipeline for Citrus maxima disease classification")
    parser.add_argument("--config", type=str, default="Models/EfficientNetB0+SVM/chatgpt-config.yaml", help="path to config yaml")
    args = parser.parse_args()
    main(args)
