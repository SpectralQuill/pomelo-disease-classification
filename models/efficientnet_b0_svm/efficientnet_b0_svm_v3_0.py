#!/usr/bin/env python3
"""
pomelo_effb0_svm_trainer.py
Object-oriented trainer: PomeloEffB0SvmTrainer

Usage:
    python pomelo_effb0_svm_trainer.py --config config.yaml
"""

import os
import sys
import argparse
import yaml
import logging
import shutil
from datetime import datetime
from pathlib import Path
import random
import json

from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# image ops
import cv2

# albumentations
import albumentations as A

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
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
    # os.environ (if you want deterministic TF, set additional envs externally)

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
# Trainer class
# ---------------------------
class PomeloEffB0SvmTrainer:
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Set seed for reproducibility
        self.seed = int(self.cfg.get("seed", 42))
        set_seed(self.seed)

        # Directories
        self.dataset_dir = Path(self.cfg["dataset_dir"]).resolve()
        self.outputs_root = Path(self.cfg["outputs_dir"]).resolve()
        ensure_dir(self.outputs_root)

        # create run-specific output dir
        ts = "effb0svm_" + now_ts()
        self.output_dir = self.outputs_root / ts
        ensure_dir(self.output_dir)
        # create required subfolders
        self.analysis_dir = self.output_dir / "analysis"
        self.augments_dir = self.output_dir / "augments"
        self.false_pred_dir = self.output_dir / "false_predictions"
        self.weights_dir = self.output_dir / "weights"
        for d in [self.analysis_dir, self.augments_dir, self.false_pred_dir, self.weights_dir]:
            ensure_dir(d)

        # Logging
        self.setup_logging()

        # other runtime attributes
        self.df_splits = None  # DataFrame holding filepaths + labels + set
        self.augments_df = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.model = None
        self.feature_extractor = None

        self.logger.info(f"Output dir: {self.output_dir}")

    # ---------------------------
    # Logging / utils
    # ---------------------------
    def setup_logging(self):
        ensure_dir(self.output_dir)
        log_file = self.output_dir / "train.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("PomeloEffB0SvmTrainer")
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
        # save partition pie chart
        counts = df_all_splits["set"].value_counts()
        plt.figure(figsize=(4,4))
        counts.plot.pie(autopct="%1.1f%%")
        plt.title("Dataset partition")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "partition_pie.png")
        plt.close()
        return df_all_splits

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
        For the training set only, generate augmented images under augments_dir preserving structure.
        Follow the infraclass-proportional augmentation method you specified.
        """
        df_train = self.df_splits[self.df_splits["set"] == "train"].copy()
        target_per_class_cfg = int(self.cfg.get("target_images_per_class", 1000))
        augment_boost = float(self.cfg.get("augment_boost", 0.25))
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

        self.logger.info(f"Augmentation target per class set to {target_per_class} (cfg={target_per_class_cfg}, max_class_count={max_class_count})")

        for cls in classes:
            cls_df = df_train[df_train["class"] == cls]
            infraclasses = sorted(cls_df["infraclass"].unique())
            # counts per infraclass
            infracounts = {inf: cls_df[cls_df["infraclass"] == inf].shape[0] for inf in infraclasses}
            original_count = len(cls_df)
            target_augments_total = max(0, target_per_class - original_count)
            if target_augments_total == 0:
                # still copy originals to augments folder with augment_index 0
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
                        "class": cls,
                        "infraclass": r["infraclass"]
                    })
                continue

            # base proportional count = largest infraclass count (per your spec)
            base_count = max(infracounts.values())
            # compute per-infraclass proportions (use infraclass_count / base_count to get values 0..1)
            props = {inf: (cnt / base_count) for inf, cnt in infracounts.items()}
            # add augment_boost
            props = {inf: p + augment_boost for inf, p in props.items()}
            # normalize to sum to 1
            total_prop = sum(props.values())
            norm_props = {inf: p / total_prop for inf, p in props.items()}

            # target per infraclass
            infratargets = {inf: int(round(norm_props[inf] * target_augments_total)) for inf in infraclasses}
            # adjust rounding difference
            s = sum(infratargets.values())
            diff = target_augments_total - s
            inf_sorted = sorted(infratargets.items(), key=lambda x: -x[1])
            idx = 0
            while diff != 0:
                inf_name, _ = inf_sorted[idx % len(inf_sorted)]
                infratargets[inf_name] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
                idx += 1
            self.logger.info(f"Class {cls}: original {original_count}, target total {target_per_class}, augments to create {target_per_class - original_count}")

            # for each infraclass, iterate through files in shuffled loop creating augments
            for inf in infraclasses:
                inf_df = cls_df[cls_df["infraclass"] == inf]
                files = list(inf_df["filepath"].values)
                if not files:
                    continue
                original_count = len(files)
                # ensure starting copy of original images (augment_index 0)
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
                # iterative augmentation
                cur_count = 0
                # prepare an index to cycle through
                idxs = list(range(len(files)))
                random.Random(seed).shuffle(idxs)
                pointer = 0
                augment_index = 1  # start from 1 because 0 reserved for original
                while cur_count < target_count:
                    src_path = Path(files[idxs[pointer]])
                    img = cv2.imread(str(src_path))
                    if img is None:
                        pointer = (pointer + 1) % len(idxs)
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    aug_img = augmenter(image=img)["image"]
                    aug_img = cv2.resize(aug_img, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
                    src = Path(src).resolve()
                    rel = src.relative_to(self.dataset_dir)
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

        # save augments CSV sorted by source name then augment index
        df_aug = pd.DataFrame(aug_rows)
        df_aug = df_aug.sort_values(by=["source", "augment_index"]).reset_index(drop=True)
        aug_csv = self.analysis_dir / "augments.csv"
        df_aug.to_csv(aug_csv, index=False)
        self.augments_df = df_aug
        self.logger.info(f"Augments saved. CSV: {aug_csv}")

        # Save before/after distribution plot
        # before
        before_counts = self.df_splits[self.df_splits["set"] == "train"]["class"].value_counts().sort_index()
        after_counts = df_aug["class"].value_counts().sort_index()
        fig, ax = plt.subplots(1,1,figsize=(8,4))
        x = np.arange(len(before_counts))
        ax.bar(x - 0.2, before_counts.values, width=0.4, label="original")
        ax.bar(x + 0.2, after_counts.reindex(index=before_counts.index).fillna(0).values, width=0.4, label="after aug")
        ax.set_xticks(x)
        ax.set_xticklabels(before_counts.index, rotation=45, ha="right")
        ax.set_ylabel("count")
        ax.set_title("Train counts before and after augmentation")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "augment_distribution.png")
        plt.close()

        # After augmentation, we will treat augments_dir as the training source for images moving forward.
        return df_aug

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
        Build EfficientNetB0 base (notop) and classification head but we'll save the feature extractor after fine-tuning.
        """
        # weights option: either 'imagenet' or a path in configs["base_weights"].
        base_weights = self.cfg.get("base_weights", "imagenet")
        input_shape = tuple(self.cfg.get("image_size", [224, 224])) + (3,)
        base_model = EfficientNetB0(include_top=False, weights=base_weights, input_shape=input_shape)
        num_classes = len(sorted(self.df_splits["class"].unique()))
        model = self.build_model_head(base_model, num_classes)
        self.logger.info(f"Built model with EfficientNetB0 base and head units {self.cfg.get('head_units', 512)}")
        self.model = model
        self.feature_extractor = models.Model(inputs=model.input, outputs=model.get_layer("gap").output)
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
        # Prepare dataset loaders: use augments for train (if provided), else original train paths
        image_size = tuple(self.cfg.get("image_size", [224, 224]))
        # function to load from a dataframe (path list) into tf.data.Dataset
        def paths_to_dataset(paths, labels, batch_size, shuffle=False):
            def load_and_preprocess(p, lab):
                img = tf.io.read_file(p)
                img = tf.image.decode_jpeg(img, channels=3)  # or decode_png if applicable
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

        history_all = {}
        for phase in phases:
            name = phase.get("name", f"phase_{phase}")
            unfreeze_last_n = int(phase.get("unfreeze_last_n", 0))
            epochs = int(phase.get("epochs", 15))
            lr = float(phase.get("lr", 1e-4))
            batch_size = int(phase.get("batch_size", 32))

            self.logger.info(f"Starting fine-tune phase {name}: unfreeze_last_n={unfreeze_last_n}, epochs={epochs}, lr={lr}, batch_size={batch_size}")

            if unfreeze_last_n == 0:
                self.model.trainable = False
            elif unfreeze_last_n > 0:
                self.model.trainable = True
                total_layers = len(self.model.layers)
                layers_to_unfreeze = min(phase.get("unfreeze_last_n", 10), total_layers)
                for layer in self.model.layers[:-layers_to_unfreeze]:
                    layer.trainable = False
                self.logger.info(f"Unfroze last {layers_to_unfreeze} layers for this phase.")

            # compile
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                               loss="sparse_categorical_crossentropy",
                               metrics=["accuracy"])
            # create datasets
            ds_train = paths_to_dataset(train_paths_tf, train_y, batch_size, shuffle=True)
            ds_val = paths_to_dataset(val_paths_tf, val_y, batch_size, shuffle=False)

            # callbacks
            cb = []
            monitor = phase.get("monitor", "val_loss")
            es = callbacks.EarlyStopping(monitor=monitor, patience=phase.get("earlystop_patience", 4), restore_best_weights=True)
            rlr = callbacks.ReduceLROnPlateau(monitor=monitor, factor=phase.get("reduce_lr_factor", 0.5), patience=phase.get("reduce_lr_patience", 3))
            ckpt_path = self.weights_dir / f"{name}_best.keras"
            ckpt = callbacks.ModelCheckpoint(str(ckpt_path), monitor=monitor, save_best_only=True)
            cb.extend([es, rlr, ckpt])

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
        Save models and scaler.
        """
        # prepare dfs for splits
        train_df = self.df_splits[self.df_splits["set"] == "train"].copy()
        val_df = self.df_splits[self.df_splits["set"] == "val"].copy()
        test_df = self.df_splits[self.df_splits["set"] == "test"].copy()

        # If augmentation was applied, we should set the train_df to point to augment images
        aug_csv = self.analysis_dir / "augments.csv"
        if aug_csv.exists():
            df_aug = pd.read_csv(aug_csv)
            # only use augmented images that correspond to classes present in train_df
            # df_aug has columns augpath, source, augment_index, class, infraclass
            # We'll use those augpath entries and their 'class' as label
            # build a pseudo train_df
            train_df = pd.DataFrame({
                "filepath": df_aug["augpath"].values,
                "class": df_aug["class"].values,
                "infraclass": df_aug["infraclass"].values,
                "filename": [Path(p).name for p in df_aug["augpath"].values],
                "set": "train"
            })

        # Extract features
        self.logger.info("Extracting features for train set...")
        X_train, y_train, train_paths = self.extract_features_for_split(train_df)
        self.logger.info("Extracting features for val set...")
        X_val, y_val, val_paths = self.extract_features_for_split(val_df)
        self.logger.info("Extracting features for test set...")
        X_test, y_test, test_paths = self.extract_features_for_split(test_df)

        # scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        joblib.dump(scaler, self.weights_dir / "scaler.joblib")
        self.logger.info("Scaler saved.")

        # Grid search SVM
        svm_cfg = self.cfg.get("svm", {})
        param_grid = svm_cfg.get("param_grid", {
            "C": [0.1, 1, 10],
            "kernel": ["rbf"],
            "gamma": ["scale", "auto"]
        })
        base_svm = SVC(probability=False, class_weight=svm_cfg.get("class_weight", None))
        grid = GridSearchCV(base_svm, param_grid, cv=svm_cfg.get("cv", 3), verbose=2, n_jobs=svm_cfg.get("n_jobs", 1))
        self.logger.info(f"Starting SVM GridSearch with params: {param_grid}")
        grid.fit(X_train_s, y_train)
        self.logger.info(f"GridSearch best params: {grid.best_params_}, best_score: {grid.best_score_}")
        best_svm = grid.best_estimator_

        # Optionally pruning via feature selection
        if self.cfg.get("svm_prune", False):
            # use a LinearSVC to rank features and select
            self.logger.info("Performing feature selection with LinearSVC (SelectFromModel)...")
            lsvc = LinearSVC(C=self.cfg.get("prune_C", 0.01), penalty="l1", dual=False, max_iter=2000).fit(X_train_s, y_train)
            selector = SelectFromModel(lsvc, prefit=True, max_features=self.cfg.get("prune_max_features", None))
            X_train_sel = selector.transform(X_train_s)
            X_val_sel = selector.transform(X_val_s)
            X_test_sel = selector.transform(X_test_s)
            # retrain SVM on selected features with best params
            best_svm.fit(X_train_sel, y_train)
            self.logger.info(f"SVM retrained on selected {X_train_sel.shape[1]} features.")
            # store selector
            joblib.dump(selector, self.weights_dir / "feature_selector.joblib")
            joblib.dump(best_svm, self.weights_dir / "svm.joblib")
            joblib.dump(grid.best_params_, self.weights_dir / "svm_best_params.joblib")
            svm_model_for_pred = best_svm
            X_test_for_pred = X_test_sel
        else:
            joblib.dump(best_svm, self.weights_dir / "svm.joblib")
            joblib.dump(grid.best_params_, self.weights_dir / "svm_best_params.joblib")
            svm_model_for_pred = best_svm
            X_test_for_pred = X_test_s

        # Evaluate on test set
        preds = svm_model_for_pred.predict(X_test_for_pred)
        accuracy = accuracy_score(y_test, preds)
        cls_report = classification_report(y_test, preds, target_names=self.label_encoder.classes_, output_dict=True)
        cm = confusion_matrix(y_test, preds)
        # save metrics json
        metrics = {
            "accuracy": float(accuracy),
            "classification_report": cls_report,
            "confusion_matrix": cm.tolist(),
            "svm_best_params": grid.best_params_
        }
        with open(self.analysis_dir / "svm_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"SVM test accuracy: {accuracy:.4f}. Results saved to svm_results.json")

        # save confusion matrix image
        plt.figure(figsize=(8,6))
        plt.title("Confusion matrix")
        im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im)
        tick_marks = np.arange(len(self.label_encoder.classes_))
        plt.xticks(tick_marks, self.label_encoder.classes_, rotation=45, ha="right")
        plt.yticks(tick_marks, self.label_encoder.classes_)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        # annotate cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "confusion_matrix.png")
        plt.close()

        # Save false predictions images
        self.save_false_predictions(test_paths, y_test, preds)

        return metrics

    def save_false_predictions(self, filepaths, true_y, pred_y):
        ensure_dir(self.false_pred_dir)
        # invert label encoder mapping
        inv_map = {i: lbl for i, lbl in enumerate(self.label_encoder.classes_)}
        count = 0
        for fp, t, p in zip(filepaths, true_y, pred_y):
            if t != p:
                count += 1
                src = Path(fp)
                try:
                    pil = Image.open(src).convert("RGB")
                except Exception as e:
                    self.logger.warning(f"Could not open {fp} to save false pred: {e}")
                    continue
                dst_file = self.false_pred_dir / f"{src.stem}_true_{inv_map[t]}_pred_{inv_map[p]}{src.suffix}"
                image_save_with_text(dst_file, pil, inv_map[t], inv_map[p])
        self.logger.info(f"Saved false predictions. Total count: {count}")

    # ---------------------------
    # Top-level run
    # ---------------------------
    def run_all(self):
        self.logger.info("=== STARTING TRAINING PIPELINE ===")
        self.discover_dataset()
        self.create_splits()
        self.do_augmentations()
        # Build and fine-tune model
        self.prepare_feature_extractor()
        self.train_finetune_phases()
        # Train SVM and evaluate
        metrics = self.train_svm()
        self.logger.info("=== PIPELINE COMPLETE ===")
        return metrics

# ---------------------------
# Main CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    trainer = PomeloEffB0SvmTrainer(args.config)
    metrics = trainer.run_all()

if __name__ == "__main__":
    main()
