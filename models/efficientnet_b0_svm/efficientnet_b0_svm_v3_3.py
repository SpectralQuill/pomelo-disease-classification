#!/usr/bin/env python3
"""
pomelo_effb0_svm_trainer.py
Object-oriented trainer: PomeloDiseaseTrainer

Usage:
    python pomelo_effb0_svm_trainer.py --config config.yaml
"""

import sys
import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path
import random
import json

from PIL import Image, ImageDraw, ImageFont

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# image ops
import cv2

# albumentations
import albumentations as A

# sklearn
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, efficientnet, MobileNetV2, mobilenet_v2, ResNet50, resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

# ---------------------------
# Utility helpers
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_ts():
    return datetime.now().strftime("%Y%m%d%H%M%S")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def image_save_with_text(dst_path: Path, pil_img: Image.Image, true_label: str, pred_label: str):
    # append labels as caption under image
    font_size = max(12, pil_img.width // 25)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    # create new canvas
    caption = f"true: {true_label}    pred: {pred_label}"
    w, h = pil_img.size
    bbox = font.getbbox(caption)
    caption_h = bbox[3] - bbox[1] + 8
    new_img = Image.new("RGB", (w, h + caption_h), (255, 255, 255))
    new_img.paste(pil_img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    draw.text((4, h + 2), caption, fill=(0, 0, 0), font=font)
    new_img.save(dst_path)

# ---------------------------
# Utility classes
# ---------------------------
class LoggingCallback(callbacks.Callback):
    def __init__(self, target_model, logger, phase_name):
        super().__init__()
        self.target_model = target_model
        self.logger = logger
        self.phase_name = phase_name
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Universal learning rate extraction
        optimizer = self.target_model.optimizer
        if hasattr(optimizer, 'learning_rate'):
            lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
        elif hasattr(optimizer, 'lr'):
            lr = float(tf.keras.backend.get_value(optimizer.lr))
        else:
            lr = 0.0
            
        self.logger.info(
            f"{self.phase_name} - Epoch {epoch+1}: "
            f"loss={logs.get('loss', 0):.4f}, "
            f"accuracy={logs.get('accuracy', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"val_accuracy={logs.get('val_accuracy', 0):.4f}, "
            f"lr={lr:.2e}"
        )

# ---------------------------
# Trainer class
# ---------------------------
class PomeloDiseaseTrainer:
    def __init__(self, config_path: str, base: str="effb0svm"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.seed = int(self.cfg.get("seed", 42))
        set_seed(self.seed)
        self.load_dirs(base)
        self.setup_logging()

        # other runtime attributes
        self.df_splits = None  # DataFrame holding filepaths + labels + set
        self.augments_df = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.model = None
        self.feature_extractor = None

        self.logger.info(f"Output dir: {self.output_dir}")
    
    def load_dirs(self, base: str="effb0svm"):
        # Setting output dir and base model
        self.base = base
        output_dir = ""
        self.base_model_folder = None
        if base == "effb0svm" or base == "mobv2" or base == "res50":
            output_dir = base
        elif base.startswith("effb0svm"):
            output_dir = "effb0soft"
            self.base_model_folder = base
        elif base.startswith("effb0soft"):
            output_dir = "effb0svm"
            self.base_model_folder = base
        output_dir += "_" + now_ts()

        # Directories
        self.dataset_dir = Path(self.cfg["dataset_dir"]).resolve()
        self.outputs_root = Path(self.cfg["outputs_dir"]).resolve()
        ensure_dir(self.outputs_root)

        # create run-specific output dir
        self.output_dir = self.outputs_root / output_dir
        ensure_dir(self.output_dir)
        # create required subfolders
        self.analysis_dir = self.output_dir / "analysis"
        self.augments_dir = self.output_dir / "augments"
        self.test_false_pred_dir = self.output_dir / "test_false_predictions"
        self.validation_false_pred_dir = self.output_dir / "validation_false_predictions"
        self.weights_dir = self.output_dir / "weights"
        for d in [self.analysis_dir, self.augments_dir, self.test_false_pred_dir, 
                  self.validation_false_pred_dir, self.weights_dir]:
            ensure_dir(d)
        
        if base.startswith("effb0svm") or base.startswith("effb0soft"):
            with open(self.analysis_dir / "base.json", "w") as f:
                json.dump({"base": base}, f, indent=2)

    # ---------------------------
    # Logging / utils
    # ---------------------------
    def setup_logging(self):
        ensure_dir(self.analysis_dir)
        log_file = self.analysis_dir / "train.log"  # Now inside analysis folder
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("PomeloDiseaseTrainer")
        # dump config for record
        cfg_dump = self.output_dir / "config_used.json"
        with open(cfg_dump, "w") as f:
            json.dump(self.cfg, f, indent=2)

    # ---------------------------
    # Dataset discovery
    # ---------------------------
    def discover_dataset(self):
        """
        Walks dataset_dir expecting structure:
            dataset_dir/
              class1/
                infraclass1/
                  img1.jpg
                infraclass2/
              class2/
                ...
        Produces a dataframe with columns: filepath, class, infraclass, filename
        """
        rows = []
        for class_dir in sorted(self.dataset_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for infraclass_dir in sorted(class_dir.iterdir()):
                if not infraclass_dir.is_dir():
                    continue
                infraclass_name = infraclass_dir.name
                for img in sorted(infraclass_dir.iterdir()):
                    if not img.is_file():
                        continue
                    # optionally allow non-image skipping
                    rows.append({
                        "filepath": str(img.resolve()),
                        "class": class_name,
                        "infraclass": infraclass_name,
                        "filename": img.name
                    })
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"No image files found in dataset dir {self.dataset_dir}")
        df = df.sort_values(by=["filename"]).reset_index(drop=True)
        self.df_all = df
        self.logger.info(f"Discovered {len(df)} images across {df['class'].nunique()} classes.")
        return df

    # ---------------------------
    # Splitting (stratified across infraclass)
    # ---------------------------
    def create_splits(self):
        """
        Creates training/validation/test splits using stratified sampling across infraclass label.
        Save CSV with columns: filepath, class, infraclass, set
        """
        df = self.df_all.copy()
        # create a single stratify key e.g. "class_infraclass" so that each infraclass is represented
        df["stratify_key"] = df["class"] + "__" + df["infraclass"]
        # config percentages
        val_pct = float(self.cfg.get("val_pct", 0.15))
        test_pct = float(self.cfg.get("test_pct", 0.15))
        train_pct = 1.0 - val_pct - test_pct
        if train_pct <= 0:
            raise ValueError("val_pct + test_pct must be < 1.0")

        # first split train+temp / test
        strat = df["stratify_key"].values
        train_temp_idx, test_idx = next(StratifiedShuffleSplit(
            n_splits=1, test_size=test_pct, random_state=self.seed
        ).split(df.index, strat))
        df_train_temp = df.loc[train_temp_idx].copy()
        df_test = df.loc[test_idx].copy()

        # then split train / val from train_temp
        strat2 = df_train_temp["stratify_key"].values
        val_relative = val_pct / (train_pct + val_pct)  # relative to train_temp size
        train_idx2, val_idx2 = next(StratifiedShuffleSplit(
            n_splits=1, test_size=val_relative, random_state=self.seed
        ).split(df_train_temp.index, strat2))
        df_train = df_train_temp.iloc[train_idx2].copy()
        df_val = df_train_temp.iloc[val_idx2].copy()

        # assign set labels
        df_train["set"] = "train"
        df_val["set"] = "val"
        df_test["set"] = "test"
        df_all_splits = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)
        # remove temp column
        df_all_splits.drop(columns=["stratify_key"], inplace=True)
        # sort by filename as requested
        df_all_splits = df_all_splits.sort_values(by=["filename"]).reset_index(drop=True)
        # save CSV
        split_csv = self.analysis_dir / "data_splits.csv"
        df_all_splits.to_csv(split_csv, index=False)
        self.df_splits = df_all_splits
        self.logger.info(f"Data splits saved to {split_csv}")
        
        # Save partition pie chart
        self._create_partition_plots(df_all_splits)
        
        return df_all_splits

    def _create_partition_plots(self, df_all_splits):
        """Create partition pie chart and infraclass distribution plots"""
        # Save partition pie chart
        counts = df_all_splits["set"].value_counts()
        plt.figure(figsize=(4,4))
        counts.plot.pie(autopct="%1.1f%%")
        plt.title("Dataset partition")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "partition_pie.png")
        plt.close()

        # Create infraclass distribution for training set before augmentation
        train_df = df_all_splits[df_all_splits["set"] == "train"]
        infraclass_counts = train_df.groupby(["class", "infraclass"]).size().reset_index(name="count")
        
        # Calculate percentages for each infraclass within each class
        class_totals = train_df.groupby("class").size()
        infraclass_counts["percentage"] = infraclass_counts.apply(
            lambda row: (row["count"] / class_totals[row["class"]] * 100), axis=1
        )
        
        # Plot 1: Infraclass distribution with counts and percentages
        plt.figure(figsize=(14, 8))
        infraclass_pivot = infraclass_counts.pivot(index="class", columns="infraclass", values="count").fillna(0)
        percentage_pivot = infraclass_counts.pivot(index="class", columns="infraclass", values="percentage").fillna(0)
        
        ax = infraclass_pivot.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Infraclass Distribution in Training Set (Before Augmentation)")
        plt.xlabel("Class")
        plt.ylabel("Count")
        
        # Add percentage labels on each bar segment
        for i, (class_name, class_row) in enumerate(infraclass_pivot.iterrows()):
            cumulative_height = 0
            for infraclass in infraclass_pivot.columns:
                count = class_row[infraclass]
                percentage = percentage_pivot.loc[class_name, infraclass]
                if count > 0:  # Only label if there are samples
                    # Position the text in the middle of each bar segment
                    ax.text(i, cumulative_height + count/2, 
                        f'{percentage:.1f}%', 
                        ha='center', va='center', 
                        fontsize=8, fontweight='bold',
                        color='white' if count > max(class_row)/2 else 'black')
                cumulative_height += count
        
        plt.legend(title="Infraclass", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "infraclass_distribution_before_aug.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Percentage-only visualization with labels
        plt.figure(figsize=(14, 8))
        percentage_pivot = infraclass_counts.pivot(index="class", columns="infraclass", values="percentage").fillna(0)
        ax2 = percentage_pivot.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Infraclass Distribution Percentages in Training Set (Before Augmentation)")
        plt.xlabel("Class")
        plt.ylabel("Percentage (%)")
        
        # Add percentage labels on each bar segment for the percentage plot
        for i, (class_name, class_row) in enumerate(percentage_pivot.iterrows()):
            cumulative_height = 0
            for infraclass in percentage_pivot.columns:
                percentage = class_row[infraclass]
                if percentage > 0:  # Only label if percentage > 0
                    # Position the text in the middle of each bar segment
                    ax2.text(i, cumulative_height + percentage/2, 
                        f'{percentage:.1f}%', 
                        ha='center', va='center', 
                        fontsize=8, fontweight='bold',
                        color='white' if percentage > 25 else 'black')  # Adjust color threshold for percentages
                    cumulative_height += percentage
        
        plt.legend(title="Infraclass", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "infraclass_percentages_before_aug.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ---------------------------
    # Augmentation pipeline
    # ---------------------------
    def build_augmenter(self):
        aug_cfg = self.cfg.get("augmentations", {})
        # create albumentations transforms from config
        transforms = []
        if aug_cfg.get("random_rotate", True):
            transforms.append(A.Rotate(limit=aug_cfg.get("rotate_limit", 20), p=aug_cfg.get("rotate_p", 0.5), border_mode=cv2.BORDER_REFLECT_101))
        if aug_cfg.get("flip", True):
            transforms.append(A.HorizontalFlip(p=aug_cfg.get("hflip_p", 0.5)))
            transforms.append(A.VerticalFlip(p=aug_cfg.get("vflip_p", 0.1)))
        if aug_cfg.get("brightness_contrast", True):
            transforms.append(A.RandomBrightnessContrast(p=aug_cfg.get("brightness_p", 0.5)))
        if aug_cfg.get("hue_sat", True):
            transforms.append(A.HueSaturationValue(p=aug_cfg.get("hsv_p", 0.3)))
        if aug_cfg.get("rgb_shift", True):
            transforms.append(A.RGBShift(p=aug_cfg.get("rgb_p", 0.2)))
        if aug_cfg.get("gamma", True):
            transforms.append(A.RandomGamma(p=aug_cfg.get("gamma_p", 0.2)))
        if aug_cfg.get("blur", True):
            transforms.append(A.GaussianBlur(p=aug_cfg.get("blur_p", 0.1)))
        aug = A.Compose(transforms)
        return aug

    def do_augmentations(self):
        """
        Enhanced augmentation with balanced strategy for handling infraclass imbalance
        """
        df_train = self.df_splits[self.df_splits["set"] == "train"].copy()
        target_per_class_cfg = int(self.cfg.get("target_images_per_class", 1000))
        image_size = tuple(self.cfg.get("image_size", [224, 224]))
        augmenter = self.build_augmenter()
        seed = self.seed

        # prepare output CSV rows
        aug_rows = []

        classes = sorted(df_train["class"].unique())
        # compute current max-class count (largest class) so that target can't be smaller than that
        counts_by_class = df_train.groupby("class").size().to_dict()
        max_class_count = max(counts_by_class.values())
        target_per_class = max(target_per_class_cfg, max_class_count)

        self.logger.info(f"Augmentation target per class set to {target_per_class}")

        for cls in classes:
            cls_df = df_train[df_train["class"] == cls]
            infraclasses = sorted(cls_df["infraclass"].unique())
            # counts per infraclass
            infracounts = {inf: cls_df[cls_df["infraclass"] == inf].shape[0] for inf in infraclasses}
            original_count = len(cls_df)
            target_augments_total = max(0, target_per_class - original_count)
            
            self.logger.info(f"Class {cls} infraclass distribution: {infracounts}")

            # Handle case where no augmentation needed
            if target_augments_total == 0:
                self._copy_originals_to_augments(cls_df, image_size, aug_rows)
                continue

            # Calculate augmentation targets using balanced strategy
            infratargets = self._calculate_balanced_targets(
                infracounts, target_augments_total
            )

            # Perform augmentation for each infraclass
            self._perform_infraclass_augmentation(
                cls, cls_df, infratargets, image_size, augmenter, seed, aug_rows
            )

        # Save results and create visualizations
        return self._finalize_augmentation_results(aug_rows, df_train)

    def _copy_originals_to_augments(self, cls_df, image_size, aug_rows):
        """Copy original images to augments folder"""
        for _, r in cls_df.iterrows():
            src = Path(r["filepath"])
            img = cv2.imread(str(src))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
            src = Path(src).resolve()
            rel = src.relative_to(self.dataset_dir)
            dst = self.augments_dir / rel.parent / f"{src.stem}.jpg"
            ensure_dir(dst.parent)
            pil = Image.fromarray(img_resized)
            pil.save(dst, format="JPEG", quality=95)
            aug_rows.append({
                "augpath": str(dst),
                "source": r["filename"],
                "augment_index": 0,
                "class": r["class"],
                "infraclass": r["infraclass"]
            })

    def _calculate_balanced_targets(self, infracounts, target_augments_total):
        """
        Compute balanced target counts using a refined version of 'Class-Balanced Loss' (Cui et al., 2019),
        with adaptive beta and mixing ratio depending on imbalance severity.
        """

        infraclasses = list(infracounts.keys())
        current_counts = np.array(list(infracounts.values()), dtype=float)

        total = np.sum(current_counts)
        mean_count = np.mean(current_counts)
        min_count, max_count = np.min(current_counts), np.max(current_counts)
    
        # Adaptive beta based on dataset size (Cui et al. 2019 recommendation)
        beta = 1 - 1 / (mean_count + 1e-8)
        effective_numbers = (1 - np.power(beta, current_counts)) / (1 - beta)

        # Normalize effective proportions
        effective_props = effective_numbers / np.sum(effective_numbers)

        # Compute imbalance severity (0 = balanced, 1 = very imbalanced)
        imbalance_ratio = max_count / (min_count + 1e-8)
        imbalance_severity = np.clip((imbalance_ratio - 1) / (10 - 1), 0, 1)  # cap at ratio=10

        # Blend weights dynamically
        w_effective = 0.6 + 0.4 * imbalance_severity  # 0.6→1.0 depending on imbalance
        w_original = 1 - w_effective

        original_props = current_counts / total
        final_props = w_effective * effective_props + w_original * original_props

        # Optional: minimum smoothing to ensure all classes get at least baseline share
        min_share = 0.02  # 2% minimum of total per class
        final_props = np.maximum(final_props, min_share)
        final_props = final_props / np.sum(final_props)

        # Convert proportions to integer targets
        infratargets = {
            inf: int(round(final_props[i] * target_augments_total))
            for i, inf in enumerate(infraclasses)
        }

        # Logging
        self.logger.info(f"[Balanced Targets] β={beta:.4f}, imbalance_severity={imbalance_severity:.2f}")
        self.logger.info(f"Original counts: {dict(zip(infraclasses, current_counts.astype(int)))}")
        self.logger.info(f"Effective props: {dict(zip(infraclasses, effective_props.round(3)))}")
        self.logger.info(f"Final props (blended): {dict(zip(infraclasses, final_props.round(3)))}")

        return self._adjust_rounding(infratargets, target_augments_total)

    def _adjust_rounding(self, infratargets, target_total):
        """Adjust for rounding differences"""
        s = sum(infratargets.values())
        diff = target_total - s
        inf_sorted = sorted(infratargets.items(), key=lambda x: -x[1])
        idx = 0
        while diff != 0:
            inf_name, _ = inf_sorted[idx % len(inf_sorted)]
            infratargets[inf_name] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            idx += 1
        return infratargets

    def _perform_infraclass_augmentation(self, cls, cls_df, infratargets, image_size, augmenter, seed, aug_rows):
        """Perform augmentation for each infraclass"""
        for inf in cls_df["infraclass"].unique():
            inf_df = cls_df[cls_df["infraclass"] == inf]
            files = list(inf_df["filepath"].values)
            if not files:
                continue
                
            original_count = len(files)
            # Copy original images first
            for f in files:
                src = Path(f)
                img = cv2.imread(str(src))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
                src = Path(src).resolve()
                rel = src.relative_to(self.dataset_dir)
                dst = self.augments_dir / rel.parent / f"{src.stem}.jpg"
                ensure_dir(dst.parent)
                pil = Image.fromarray(img_resized)
                pil.save(dst, format="JPEG", quality=95)

                aug_rows.append({
                    "augpath": str(dst),
                    "source": src.name,
                    "augment_index": 0,
                    "class": cls,
                    "infraclass": inf
                })
                
            target_count = infratargets.get(inf, 0)
            self.logger.info(
                f"Class {cls} - Infraclass '{inf}': original {original_count}, "
                f"target total {original_count + target_count}, "
                f"augments to create {target_count}"
            )
            
            if target_count <= 0:
                continue
                
            # Iterative augmentation
            cur_count = 0
            idxs = list(range(len(files)))
            random.Random(seed).shuffle(idxs)
            pointer = 0
            augment_index = 1
            
            while cur_count < target_count:
                src_path = Path(files[idxs[pointer]])
                img = cv2.imread(str(src_path))
                if img is None:
                    pointer = (pointer + 1) % len(idxs)
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                aug_img = augmenter(image=img)["image"]
                aug_img = cv2.resize(aug_img, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
                # Use the same relative path structure
                rel = src_path.relative_to(self.dataset_dir)
                dst = self.augments_dir / rel.parent / f"{src_path.stem}_aug{augment_index}.jpg"
                ensure_dir(dst.parent)
                pil = Image.fromarray(aug_img)
                pil.save(dst, format="JPEG", quality=95)
                aug_rows.append({
                    "augpath": str(dst),
                    "source": src_path.name,
                    "augment_index": augment_index,
                    "class": cls,
                    "infraclass": inf
                })
                cur_count += 1
                augment_index += 1
                pointer += 1
                if pointer >= len(idxs):
                    random.Random(seed + augment_index).shuffle(idxs)
                    pointer = 0

    def _finalize_augmentation_results(self, aug_rows, df_train):
        """Save augmentation results and create visualizations"""
        # save augments CSV sorted by source name then augment index
        df_aug = pd.DataFrame(aug_rows)
        df_aug = df_aug.sort_values(by=["source", "augment_index"]).reset_index(drop=True)
        aug_csv = self.analysis_dir / "augments.csv"
        df_aug.to_csv(aug_csv, index=False)
        self.augments_df = df_aug
        self.logger.info(f"Augments saved. CSV: {aug_csv}")

        # Create enhanced distribution plots
        self._create_distribution_plots(df_train, df_aug)

        return df_aug

    def _create_distribution_plots(self, df_train, df_aug):
        """Create enhanced distribution plots for balanced strategy"""
        
        # Class distribution before/after augmentation
        before_counts = df_train["class"].value_counts().sort_index()
        after_counts = df_aug["class"].value_counts().sort_index()
        
        # Calculate percentages
        before_pct = (before_counts / before_counts.sum() * 100).round(1)
        after_pct = (after_counts / after_counts.sum() * 100).round(1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Class distribution plot
        x = np.arange(len(before_counts))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, before_counts.values, width, label="Original", alpha=0.7, color='skyblue')
        bars2 = ax1.bar(x + width/2, after_counts.reindex(index=before_counts.index).fillna(0).values, width, label="After Balanced Aug", alpha=0.7, color='lightcoral')
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars1, before_pct)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(before_counts.values),
                    f'{pct}%', ha='center', va='bottom', fontsize=8)
        
        for i, (bar, pct) in enumerate(zip(bars2, after_pct)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(after_counts.values),
                    f'{pct}%', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(before_counts.index, rotation=45, ha="right")
        ax1.set_ylabel("Count")
        ax1.set_title("Class Distribution: Before vs After Balanced Augmentation")
        ax1.legend()
        
        # Infraclass distribution after augmentation with percentages
        infra_after_counts = df_aug.groupby(["class", "infraclass"]).size().reset_index(name="count")
        
        # Calculate percentages for each infraclass within each class
        class_totals_after = df_aug.groupby("class").size()
        infra_after_counts["percentage"] = infra_after_counts.apply(
            lambda row: (row["count"] / class_totals_after[row["class"]] * 100), axis=1
        )
        
        # Create pivot tables for counts and percentages
        infra_after_pivot = infra_after_counts.pivot(index="class", columns="infraclass", values="count").fillna(0)
        percentage_after_pivot = infra_after_counts.pivot(index="class", columns="infraclass", values="percentage").fillna(0)
        
        # Plot infraclass distribution after augmentation
        ax2 = infra_after_pivot.plot(kind="bar", stacked=True, ax=ax2, alpha=0.7)
        ax2.set_title("Infraclass Distribution After Balanced Augmentation")
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Count")
        
        # Add percentage labels on each bar segment
        for i, (class_name, class_row) in enumerate(infra_after_pivot.iterrows()):
            cumulative_height = 0
            for infraclass in infra_after_pivot.columns:
                count = class_row[infraclass]
                percentage = percentage_after_pivot.loc[class_name, infraclass]
                if count > 0:  # Only label if there are samples
                    # Position the text in the middle of each bar segment
                    ax2.text(i, cumulative_height + count/2, 
                        f'{percentage:.1f}%', 
                        ha='center', va='center', 
                        fontsize=8, fontweight='bold',
                        color='white' if count > max(class_row)/3 else 'black')
                cumulative_height += count
        
        ax2.legend(title="Infraclass", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "balanced_augmentation_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Log balancing statistics
        self._log_balancing_stats(df_train, df_aug)

    def _log_balancing_stats(self, df_train, df_aug):
        """Log detailed statistics about the balancing effect"""
        self.logger.info("=== BALANCING STATISTICS ===")
        
        for cls in df_train["class"].unique():
            cls_train = df_train[df_train["class"] == cls]
            cls_aug = df_aug[df_aug["class"] == cls]
            
            train_dist = cls_train["infraclass"].value_counts().to_dict()
            aug_dist = cls_aug["infraclass"].value_counts().to_dict()
            
            self.logger.info(f"Class {cls}:")
            self.logger.info(f"  Before: {train_dist}")
            self.logger.info(f"  After:  {aug_dist}")
            
            # Calculate imbalance reduction
            if len(train_dist) > 1:
                train_imbalance = max(train_dist.values()) / min(train_dist.values())
                aug_imbalance = max(aug_dist.values()) / min(aug_dist.values())
                self.logger.info(f"  Imbalance ratio: {train_imbalance:.1f} → {aug_imbalance:.1f}")

    # ---------------------------
    # Build model, fine-tune phases
    # ---------------------------
    def build_model_head(self, base_model, num_classes):
        """
        Attach classification head to base_model (which is not include_top).
        """
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.BatchNormalization(name="bn_gap")(x)
        # dense with regularization
        head_units = int(self.cfg.get("head_units", 512))
        dropout = float(self.cfg.get("dropout", 0.4))
        x = layers.Dense(head_units, activation="relu",
                         kernel_regularizer=regularizers.l2(float(self.cfg.get("l2", 1e-4))),
                         name="head_dense")(x)
        x = layers.BatchNormalization(name="bn_head")(x)
        x = layers.Dropout(dropout, name="head_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
        model = models.Model(inputs=base_model.input, outputs=outputs)
        return model

    def prepare_feature_extractor(self):
        """
        Build base CNN (EfficientNetB0, MobileNetV2, or ResNet50 depending on self.base)
        and attach classification head. Saves a feature extractor after fine-tuning.
        """

        # Determine architecture type
        base_key = self.base.lower()
        input_shape = tuple(self.cfg.get("image_size", [224, 224])) + (3,)
        num_classes = len(sorted(self.df_splits["class"].unique()))
        pretrained_dir = Path("models/base")

        # Select backbone
        if base_key.startswith("effb0"):
            pretrained = pretrained_dir / "efficientnetb0_notop.h5"
            base_model = EfficientNetB0(include_top=False, weights=pretrained, input_shape=input_shape)
            preprocess_func = efficientnet.preprocess_input
            base_name = "EfficientNetB0"

        elif base_key.startswith("mobv2"):
            pretrained = pretrained_dir / "mobilenetv2_notop.h5"
            base_model = MobileNetV2(include_top=False, weights=pretrained, input_shape=input_shape)
            preprocess_func = mobilenet_v2.preprocess_input
            base_name = "MobileNetV2"

        elif base_key.startswith("res50"):
            pretrained = pretrained_dir / "resnet50_notop.h5"
            base_model = ResNet50(include_top=False, weights=pretrained, input_shape=input_shape)
            preprocess_func = resnet.preprocess_input
            base_name = "ResNet50"

        else:
            raise ValueError(f"❌ Unknown base architecture: {self.base}")

        # Save preprocess function for later use
        self.preprocess_func = preprocess_func

        # Build classifier head
        model = self.build_model_head(base_model, num_classes)
        self.logger.info(f"✅ Built model with {base_name} base and head units {self.cfg.get('head_units', 512)}")

        # Create feature extractor (output before classification head)
        if "gap" in [l.name for l in model.layers]:
            gap_layer = "gap"
        else:
            # fallback search for global average pooling
            gap_candidates = [l.name for l in model.layers if "global_average" in l.name or "avg_pool" in l.name]
            gap_layer = gap_candidates[-1] if gap_candidates else model.layers[-2].name

        self.model = model
        self.feature_extractor = models.Model(inputs=model.input, outputs=model.get_layer(gap_layer).output)

        return model

    def train_finetune_phases(self):
        """
        Conduct configurable fine-tuning phases. Each phase config should be:
        - unfreeze_last_n_layers : int (0 means none)
        - epochs, lr, batch_size
        """
        phases = self.cfg.get("finetune_phases", [
            {"name":"phase0", "unfreeze_last_n":0, "epochs":30, "lr":1e-4, "batch_size":32},
        ])
        image_size = tuple(self.cfg.get("image_size", [224, 224]))

        # Determine train image paths: prefer augments folder if exists and non-empty else use original train
        train_source_dir = self.augments_dir if any(self.augments_dir.iterdir()) else None
        # build mapping for train/val/test
        if train_source_dir:
            # load augmented CSV and map augpaths that are in train set
            aug_csv_path = self.analysis_dir / "augments.csv"
            if aug_csv_path.exists():
                df_aug = pd.read_csv(aug_csv_path)
                # path strings are absolute in our code path creation
                # use only augmented images that correspond to classes in train split (train split already used to create aug)
                train_paths = df_aug["augpath"].values.tolist()
                train_labels = df_aug["class"].values.tolist()
            else:
                raise FileNotFoundError("Augments CSV not found but augments folder exists.")
        else:
            train_df = self.df_splits[self.df_splits["set"] == "train"]
            train_paths = train_df["filepath"].values.tolist()
            train_labels = train_df["class"].values.tolist()

        val_df = self.df_splits[self.df_splits["set"] == "val"]
        val_paths = val_df["filepath"].values.tolist()
        val_labels = val_df["class"].values.tolist()

        # encode labels using LabelEncoder
        self.label_encoder.fit(sorted(self.df_splits["class"].unique()))
        train_y = self.label_encoder.transform(train_labels)
        val_y = self.label_encoder.transform(val_labels)

        # convert paths to strings of bytes for tf dataset
        train_paths_tf = np.array(train_paths, dtype=str)
        val_paths_tf = np.array(val_paths, dtype=str)

        # phase names duplication tracking
        phase_names = ["softmax"]

        history_all = {}
        for phase in phases:
            name = phase.get("name", f"phase_{phase}")
            base_name = name
            name_index = 0
            while name in phase_names:
                name_index += 1
                name = base_name + str(name_index)
            phase_names.append(name)

            unfreeze_last_n = int(phase.get("unfreeze_last_n", 0))
            epochs = int(phase.get("epochs", 15))
            lr = float(phase.get("lr", 1e-4))
            batch_size = int(phase.get("batch_size", 32))

            self.logger.info(f"Starting fine-tune phase \"{name}\": unfreeze_last_n={unfreeze_last_n}, epochs={epochs}, lr={lr}, batch_size={batch_size}")

            if unfreeze_last_n == 0:
                self._set_model_trainability(0)
            elif unfreeze_last_n > 0:
                self._set_model_trainability(unfreeze_last_n)

            # compile
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                               loss="sparse_categorical_crossentropy",
                               metrics=["accuracy"])
            # create datasets
            ds_train = self.paths_to_dataset(train_paths_tf, train_y, batch_size, image_size, shuffle=True)
            ds_val = self.paths_to_dataset(val_paths_tf, val_y, batch_size, image_size, shuffle=False)

            # callbacks
            cb = []
            monitor = phase.get("monitor", "val_loss")
            es = callbacks.EarlyStopping(monitor=monitor, patience=phase.get("earlystop_patience", 4), restore_best_weights=True)
            rlr = callbacks.ReduceLROnPlateau(monitor=monitor, factor=phase.get("reduce_lr_factor", 0.5), patience=phase.get("reduce_lr_patience", 3))
            ckpt_path = self.weights_dir / f"{name}_best.keras"
            ckpt = callbacks.ModelCheckpoint(str(ckpt_path), monitor=monitor, save_best_only=True)
            cb.extend([es, rlr, ckpt])

            logging_cb = LoggingCallback(self.model, self.logger, name)
            cb.append(logging_cb)

            hist = self.model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=cb, verbose=1)
            history_all[name] = hist.history

            # save history plots
            plt.figure()
            plt.plot(hist.history.get("loss", []), label="train_loss")
            plt.plot(hist.history.get("val_loss", []), label="val_loss")
            plt.title(f"{name} Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(self.analysis_dir / f"{name}_losses.png")
            plt.close()

            plt.figure()
            plt.plot(hist.history.get("accuracy", []), label="train_accuracy")
            plt.plot(hist.history.get("val_accuracy", []), label="val_accuracy")
            plt.title(f"{name} Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(self.analysis_dir / f"{name}_accuracies.png")
            plt.close()

            # after phase, save intermediate feature extractor
            # feature extractor is model up to GAP
            self.feature_extractor = models.Model(inputs=self.model.input, outputs=self.model.get_layer("gap").output)
            feat_path = self.weights_dir / f"feature_extractor_{name}.keras"
            self.feature_extractor.save(str(feat_path))
            self.logger.info(f"Saved feature extractor for phase {name} to {feat_path}")

        # save final model weights
        final_weights = self.weights_dir / "final_model.keras"
        self.model.save(final_weights)
        self.logger.info(f"Saved final full model to {final_weights}")

        # also save label encoder
        joblib.dump(self.label_encoder, self.weights_dir / "label_encoder.joblib")
        self.logger.info("Label encoder saved.")

        # store histories
        hist_json = self.analysis_dir / "training_history.json"
        with open(hist_json, "w") as f:
            json.dump(history_all, f, indent=2)
        self.logger.info(f"Training histories saved to {hist_json}")

        return history_all

    def paths_to_dataset(self, paths, labels, batch_size, image_size, shuffle=False):
        def load_and_preprocess(p, lab):
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, image_size)
            img = eff_preprocess(img * 255.0)
            return img, lab

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=self.seed)
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _set_model_trainability(self, unfreeze_last_n):
        """Set model trainability with precise control"""
        
        if unfreeze_last_n == 0:
            # For warmup: freeze everything except the classification head
            # The head is typically the last few dense layers
            head_start = max(0, len(self.model.layers) - 4)  # Last 4 layers are usually head
            
            for i, layer in enumerate(self.model.layers):
                if i < head_start:
                    layer.trainable = False
                else:
                    layer.trainable = True
                    
            self.logger.info(f"Frozen base model (layers 0-{head_start-1}), kept head trainable (layers {head_start}-end)")
                    
        elif unfreeze_last_n > 0:
            # For fine-tuning: unfreeze last N layers
            self.model.trainable = True
            total_layers = len(self.model.layers)
            layers_to_unfreeze = min(unfreeze_last_n, total_layers)
            
            for i, layer in enumerate(self.model.layers):
                layer.trainable = (i >= total_layers - layers_to_unfreeze)
            
            self.logger.info(f"Unfroze last {layers_to_unfreeze} layers (layers {total_layers - layers_to_unfreeze}-end)")
        
        # Log final trainable status
        trainable_layers = [layer.name for layer in self.model.layers if layer.trainable]
        self.logger.info(f"Trainable layers: {len(trainable_layers)} - {trainable_layers}")

    # ---------------------------
    # Feature extraction + SVM training
    # ---------------------------
    def extract_features_for_split(self, df_split):
        """
        Given a dataframe of {filepath, class, ...}, returns (X_features, y_labels, filepaths)
        using current self.feature_extractor.
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not prepared.")
        image_size = tuple(self.cfg.get("image_size", [224, 224]))
        paths = df_split["filepath"].values.tolist()
        labels = self.label_encoder.transform(df_split["class"].values.tolist())
        batch = 64
        feats = []
        for i in range(0, len(paths), batch):
            batch_paths = paths[i:i+batch]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB").resize(image_size)
                arr = np.array(img)
                arr = eff_preprocess(arr.astype(np.float32))
                imgs.append(arr)
            imgs = np.stack(imgs, axis=0)
            f = self.feature_extractor.predict(imgs, verbose=0)
            feats.append(f)
        X = np.vstack(feats)
        y = np.array(labels)
        return X, y, paths

    def train_svm(self):
        """
        Extract features for train/val/test and train SVM with GridSearchCV.
        Save models and results.
        """
        # use original splits (not augments) for SVM training
        df_train = self.df_splits[self.df_splits["set"] == "train"]
        df_val = self.df_splits[self.df_splits["set"] == "val"]
        df_test = self.df_splits[self.df_splits["set"] == "test"]

        self.logger.info("Extracting features for SVM training...")
        X_train, y_train, _ = self.extract_features_for_split(df_train)
        X_val, y_val, _ = self.extract_features_for_split(df_val)
        test_x, test_y, test_paths = self.extract_features_for_split(df_test)

        # combine train+val for SVM training
        X_trval = np.vstack([X_train, X_val])
        y_trval = np.hstack([y_train, y_val])

        # StandardScaler
        scaler = StandardScaler()
        X_trval_scaled = scaler.fit_transform(X_trval)
        X_val_scaled = scaler.transform(X_val)  # Transform validation set
        test_x_scaled = scaler.transform(test_x)

        # Get class weight configuration
        svm_config = self.cfg.get("svm", {})
        class_weight = svm_config.get("class_weight", None)

        # Feature selection
        selector = None
        if svm_config.get("feature_selection", False):
            selector = SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000), max_features=500)
            X_trval_scaled = selector.fit_transform(X_trval_scaled, y_trval)
            X_val_scaled = selector.transform(X_val_scaled)
            test_x_scaled = selector.transform(test_x_scaled)  # Apply same selection to test
            self.logger.info(f"Feature selection: {X_trval_scaled.shape[1]} features retained.")
        
        # Calculate class weights if requested
        if class_weight == "balanced":
            classes = np.unique(y_trval)
            weights = compute_class_weight('balanced', classes=classes, y=y_trval)
            # Convert numpy types to Python native types for JSON serialization
            class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
            self.logger.info(f"Using balanced class weights: {class_weight}")
        elif class_weight is None:
            self.logger.info("Using default class weights (None)")
        else:
            # Convert any existing class_weight to serializable types
            class_weight = {int(k): float(v) for k, v in class_weight.items()}
            self.logger.info(f"Using custom class weights: {class_weight}")

        # GridSearchCV with class weights
        param_grid = svm_config.get("param_grid", {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        })
        
        svm = SVC(random_state=self.seed, class_weight=class_weight, probability=True)
        grid = GridSearchCV(
            svm,
            param_grid, 
            cv=svm_config.get("cv", 3), 
            n_jobs=svm_config.get("n_jobs", -1), 
            verbose=1, 
            scoring="accuracy"
        )
        
        # Create a custom callback-like logging for SVM
        self.logger.info("Starting SVM GridSearchCV...")
        grid.fit(X_trval_scaled, y_trval)
        self.logger.info("SVM GridSearchCV completed.")
        
        results = pd.DataFrame(grid.cv_results_).sort_values(by="mean_test_score", ascending=False)
        self.logger.info(f"Total SVM GridSearchCV combinations: {len(results)}")
        for _, row in results.iterrows():
            mean_score = row["mean_test_score"]
            std_score = row["std_test_score"]
            rank = int(row["rank_test_score"])
            params = row["params"]
            self.logger.info(
                f"Rank {rank:>2}: mean={mean_score:.4f}, std={std_score:.4f}, params={params}"
            )
        csv_path = self.analysis_dir / "svm_gridsearch_results.csv"
        results.to_csv(csv_path, index=False)
        self.logger.info(f"Full GridSearchCV results saved to {csv_path}")

        self.logger.info(f"Best SVM params: {grid.best_params_}")
        self.logger.info(f"Best SVM CV score: {grid.best_score_:.4f}")

        # save SVM model, scaler, selector
        svm_model_path = self.weights_dir / "svm_model.joblib"
        joblib.dump(grid.best_estimator_, svm_model_path)
        joblib.dump(scaler, self.weights_dir / "svm_scaler.joblib")
        if selector:
            joblib.dump(selector, self.weights_dir / "svm_selector.joblib")
        self.logger.info(f"SVM model saved to {svm_model_path}")

        # evaluate on test set
        pred_y = grid.predict(test_x_scaled)
        test_acc = accuracy_score(test_y, pred_y)
        self.logger.info(f"SVM test accuracy: {test_acc:.4f}")

        # Enhanced evaluation with per-class metrics
        clf_report = classification_report(test_y, pred_y, target_names=self.label_encoder.classes_, output_dict=True)
        clf_report_text = classification_report(test_y, pred_y, target_names=self.label_encoder.classes_)
        cm = confusion_matrix(test_y, pred_y).astype(np.float32)

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

        classes = self.label_encoder.classes_
        num_classes = len(classes)

        # Save detailed results - ensure all data is JSON serializable
        results = {
            "test_accuracy": float(test_acc),
            "best_params": {str(k): (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in grid.best_params_.items()},
            "best_cv_score": float(grid.best_score_),
            "classification_report": clf_report,
            "confusion_matrix": cm.tolist(),
            "class_weights_used": class_weight
        }
        
        with open(self.analysis_dir / "svm_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        with open(self.analysis_dir / "svm_classification_report.txt", "w") as f:
            f.write(clf_report_text)
            f.write(f"\nTest accuracy: {test_acc:.4f}\n")
            f.write(f"Best params: {grid.best_params_}\n")
            f.write(f"CV score: {grid.best_score_:.4f}\n")

        plt.figure(figsize=(10, 8))
        plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
        plt.title(f"SVM Confusion Matrix (Acc={test_acc:.4f})")
        plt.colorbar(label="Percentage (%)")

        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes, rotation=45, ha="right")
        plt.yticks(tick_marks, classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Write NN (PP.PP%) in each cell
        for i in range(num_classes):
            for j in range(num_classes):
                raw_val = int(cm[i, j])
                perc_val = cm_percent[i, j]

                text = f"{perc_val:.2f}%\n({raw_val})"

                # Use white text on dark squares
                color = "white" if perc_val > 50 else "black"

                plt.text(j, i, text,
                        ha="center", va="center",
                        color=color, fontsize=9)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / "svm_confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Save false predictions for test set
        self._save_false_predictions(df_test, test_y, pred_y, test_paths, self.test_false_pred_dir, "test")

        # Also save false predictions for validation set
        # X_val_scaled is already transformed and feature-selected above
        y_val_pred = grid.predict(X_val_scaled)
        self._save_false_predictions(df_val, y_val, y_val_pred, df_val["filepath"].values.tolist(), 
                                self.validation_false_pred_dir, "validation")

        return test_acc, grid.best_estimator_

    def train_softmax(self):
        """
        Continue training from an existing EfficientNetB0 model (final_model.keras)
        using a softmax head instead of SVM. This reuses the same dataset split,
        augmentations, and configurations from the base run.
        """
        self.logger.info(f"Starting SOFTMAX phase continuation from {self.output_dir}")

        if self.base_model_folder is None:
            raise ValueError("Base model folder not specified for softmax training.")
        base_dir = self.outputs_root / self.base_model_folder
        analysis_dir =  base_dir / "analysis"
        weights_dir = base_dir / "weights"

        # Resolve base folder
        base_model_path = weights_dir / "final_model.keras"

        if not base_model_path.exists():
            raise FileNotFoundError(f"Base model not found at {base_model_path}")

        # Load model and label encoder
        self.model = tf.keras.models.load_model(base_model_path)
        le_path = weights_dir / "label_encoder.joblib"
        self.label_encoder = joblib.load(le_path)
        self.logger.info(f"Loaded base model and label encoder from {base_dir}")

        # Use same dataset splits and augmentations as the base run
        split_csv = analysis_dir / "data_splits.csv"
        aug_csv = analysis_dir / "augments.csv"

        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found in {analysis_dir}")
        self.df_splits = pd.read_csv(split_csv)

        use_augments = aug_csv.exists()
        image_size = tuple(self.cfg.get("image_size", [224, 224]))

        # Determine which paths to use for training
        if use_augments:
            df_aug = pd.read_csv(aug_csv)
            train_paths = df_aug["augpath"].tolist()
            train_labels = df_aug["class"].tolist()
        else:
            train_df = self.df_splits[self.df_splits["set"] == "train"]
            train_paths = train_df["filepath"].tolist()
            train_labels = train_df["class"].tolist()

        val_df = self.df_splits[self.df_splits["set"] == "val"]
        test_df = self.df_splits[self.df_splits["set"] == "test"]

        val_paths = val_df["filepath"].tolist()
        test_paths = test_df["filepath"].tolist()
        val_labels = val_df["class"].tolist()
        test_labels = test_df["class"].tolist()

        # Encode labels
        train_y = self.label_encoder.transform(train_labels)
        val_y = self.label_encoder.transform(val_labels)
        test_y = self.label_encoder.transform(test_labels)

        softmax_cfg = self.cfg.get("softmax", {})

        batch_size = int(softmax_cfg.get("batch_size", 32))
        lr = float(softmax_cfg.get("lr", 1e-4))
        epochs = int(self.cfg.get("softmax", {}).get("epochs", 15))

        ds_train = self.paths_to_dataset(train_paths, train_y, batch_size, image_size, shuffle=True)
        ds_val = self.paths_to_dataset(val_paths, val_y, batch_size, image_size)
        ds_test = self.paths_to_dataset(test_paths, test_y, batch_size, image_size)

        # Compile and fit
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        ckpt_path = self.weights_dir / "softmax_best.keras"
        monitor = softmax_cfg.get("monitor", "val_loss")
        earlystop_patience = int(softmax_cfg.get("earlystop_patience", 4))
        reduce_lr_factor = float(softmax_cfg.get("reduce_lr_factor", 0.5))
        reduce_lr_patience = int(softmax_cfg.get("reduce_lr_patience", 3))

        ckpt_path = self.weights_dir / "softmax_best.keras"
        cb = [
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=earlystop_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience
            ),
            callbacks.ModelCheckpoint(
                str(ckpt_path),
                monitor=monitor,
                save_best_only=True
            ),
        ]

        logging_cb = LoggingCallback(self.model, self.logger, "softmax")
        cb.append(logging_cb)

        hist = self.model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=cb, verbose=1)

        # Evaluate
        test_loss, test_acc = self.model.evaluate(ds_test, verbose=1)
        self.logger.info(f"Softmax test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

        pred_y = np.argmax(self.model.predict(ds_test), axis=1)
        cm = confusion_matrix(test_y, pred_y).astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0
        classes = self.label_encoder.classes_
        num_classes = len(classes)
        report = classification_report(test_y, pred_y, target_names=self.label_encoder.classes_, output_dict=True)
        report_text = classification_report(test_y, pred_y, target_names=self.label_encoder.classes_)

        # Save reports & metrics
        with open(self.analysis_dir / "softmax_classification_report.txt", "w") as f:
            f.write(report_text)
            f.write(f"\nTest accuracy: {test_acc:.4f}\n")

        with open(self.analysis_dir / "softmax_metrics.json", "w") as f:
            json.dump({
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
                "classification_report": report,
                "confusion_matrix": cm.tolist()
            }, f, indent=2)

        plt.figure(figsize=(10, 8))
        plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
        plt.title(f"Softmac Confusion Matrix (Acc={test_acc:.4f})")
        plt.colorbar(label="Percentage (%)")

        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes, rotation=45, ha="right")
        plt.yticks(tick_marks, classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Write NN (PP.PP%) in each cell
        for i in range(num_classes):
            for j in range(num_classes):
                raw_val = int(cm[i, j])
                perc_val = cm_percent[i, j]

                text = f"{perc_val:.2f}%\n({raw_val})"

                # Use white text on dark squares
                color = "white" if perc_val > 50 else "black"

                plt.text(j, i, text,
                        ha="center", va="center",
                        color=color, fontsize=9)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / "softmax_confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Losses and accuracies plots
        plt.figure()
        plt.plot(hist.history.get("loss", []), label="train_loss")
        plt.plot(hist.history.get("val_loss", []), label="val_loss")
        plt.title("Softmax Loss")
        plt.legend()
        plt.savefig(self.analysis_dir / "softmax_losses.png")
        plt.close()

        plt.figure()
        plt.plot(hist.history.get("accuracy", []), label="train_accuracy")
        plt.plot(hist.history.get("val_accuracy", []), label="val_accuracy")
        plt.title("Softmax Accuracy")
        plt.legend()
        plt.savefig(self.analysis_dir / "softmax_accuracies.png")
        plt.close()

        # Update training history
        hist_json = self.analysis_dir / "training_history.json"
        try:
            existing = json.load(open(hist_json))
        except Exception:
            existing = {}
        existing["softmax_phase"] = hist.history
        json.dump(existing, open(hist_json, "w"), indent=2)

    def _save_false_predictions(self, df_split, y_true, pred_y, paths, output_dir, split_name):
        """Save false predictions to specified directory"""
        false_indices = np.where(y_true != pred_y)[0]
        
        self.logger.info(f"Saving {len(false_indices)} false {split_name} predictions to {output_dir}")
        
        for idx in false_indices:
            true_label = self.label_encoder.inverse_transform([y_true[idx]])[0]
            pred_label = self.label_encoder.inverse_transform([pred_y[idx]])[0]
            src_path = Path(paths[idx])
            
            # Load and save image with caption
            pil_img = Image.open(src_path).convert("RGB")
            dst_path = output_dir / f"{src_path.stem}_falsepred.jpg"
            image_save_with_text(dst_path, pil_img, true_label, pred_label)

    # ---------------------------
    # Main orchestration
    # ---------------------------
    def run(self):
        if self.base_model_folder == None:
            self.logger.info("Starting PomeloDiseaseTrainer...")
            self.discover_dataset()
            self.create_splits()
            self.do_augmentations()
            self.prepare_feature_extractor()
            self.train_finetune_phases()
            if self.base.startswith("effb0svm"):
                self.train_svm()
            else:
                self.train_softmax()
        elif self.base_model_folder.startswith("effb0svm_"):
            self.train_softmax()
        elif self.base_model_folder.startswith("effb0soft_"):
            print("Not available...")
        else:
            raise ValueError(f"❌ Unsupported base model type: {self.base}")
        self.logger.info("Training completed successfully!")
        

# ---------------------------
# CLI entry point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--base", type=str, default="effb0svm",
                        help="Base model to train")
    args = parser.parse_args()

    trainer = PomeloDiseaseTrainer(args.config, args.base)
    trainer.run()
