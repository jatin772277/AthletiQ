
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets/football/team_identification"
MANIFEST = DATA_DIR / "team_crop_manifest_labeled.csv"
MODEL_DIR = ROOT / "ai-engine/models/team_identification"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
SEED = 42

CLASSES = ["team_a", "team_b"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TeamCropDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = ROOT / row["image_path"]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = CLASS_TO_IDX[row["team_label"]]
        return image, label


def main():
    seed_everything(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Labeled manifest not found: {MANIFEST}\n"
            "Copy the exported CSV to this location first."
        )

    df = pd.read_csv(MANIFEST)
    required = {
        "image_path", "canonical_id", "team_label", "quality"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["team_label"] = df["team_label"].fillna("").str.strip().str.lower()
    df["quality"] = df["quality"].fillna("good").str.strip().str.lower()

    # Only use labeled, good-quality examples from the two teams.
    df = df[
        df["team_label"].isin(CLASSES)
        & df["quality"].eq("good")
    ].copy()

    df["canonical_id"] = pd.to_numeric(
        df["canonical_id"], errors="coerce"
    )
    df = df.dropna(subset=["canonical_id", "image_path"])
    df["canonical_id"] = df["canonical_id"].astype(int)

    # Ensure every image exists before training.
    df = df[
        df["image_path"].map(lambda p: (ROOT / str(p)).is_file())
    ].copy()

    # A temporary ID should not carry conflicting team labels.
    conflicts = df.groupby("canonical_id")["team_label"].nunique()
    conflicts = conflicts[conflicts > 1]

    if len(conflicts):
        raise ValueError(
            "Some temporary IDs have conflicting team labels. "
            f"Resolve these IDs in the manifest: {conflicts.index.tolist()}"
        )

    if df["team_label"].nunique() != 2:
        raise ValueError("Both team_a and team_b need good labeled crops.")

    # Split IDs, not images, to reduce frame-level leakage.
    id_labels = (
        df.groupby("canonical_id")["team_label"]
        .first()
        .reset_index()
    )

    class_id_counts = id_labels["team_label"].value_counts()
    if (class_id_counts < 2).any():
        raise ValueError(
            "Need at least two distinct temporary IDs for each team "
            "to create a train/validation split."
        )

    train_ids, val_ids = train_test_split(
        id_labels["canonical_id"].to_numpy(),
        test_size=0.20,
        random_state=SEED,
        stratify=id_labels["team_label"],
    )

    train_df = df[df["canonical_id"].isin(train_ids)].copy()
    val_df = df[df["canonical_id"].isin(val_ids)].copy()

    if set(train_df["canonical_id"]) & set(val_df["canonical_id"]):
        raise RuntimeError("ID leakage detected between train and validation.")

    print("Dataset summary")
    print(f"Usable images: {len(df)}")
    print(f"Unique temporary IDs: {df['canonical_id'].nunique()}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print("\nTraining class counts:")
    print(train_df["team_label"].value_counts().to_string())
    print("\nValidation class counts:")
    print(val_df["team_label"].value_counts().to_string())

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.10
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = TeamCropDataset(train_df, train_transform)
    val_ds = TeamCropDataset(val_df, val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Download pretrained ImageNet weights on the first run if necessary.
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(CLASSES)
    )
    model = model.to(device)

    # Weighted loss reduces the effect of the class imbalance.
    counts = train_df["team_label"].value_counts()
    class_weights = torch.tensor(
        [
            len(train_df) / (len(CLASSES) * counts.get(name, 1))
            for name in CLASSES
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * len(labels)
                preds = logits.argmax(1)

                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        train_loss = train_loss_sum / max(train_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f}"
        )

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_name": "mobilenet_v3_small",
                    "state_dict": best_state,
                    "classes": CLASSES,
                    "image_size": IMAGE_SIZE,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "best_val_loss": best_val_loss,
                },
                MODEL_DIR / "best_team_classifier.pt",
            )

            best_predictions = list(all_preds)
            best_labels = list(all_labels)

    model.load_state_dict(best_state)
    model.eval()

    metrics = {
        "model": "MobileNetV3-Small",
        "classes": CLASSES,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "usable_images": len(df),
        "train_images": len(train_df),
        "validation_images": len(val_df),
        "train_ids": sorted(map(int, train_ids)),
        "validation_ids": sorted(map(int, val_ids)),
        "best_validation_loss": best_val_loss,
        "history": history,
        "confusion_matrix": confusion_matrix(
            best_labels, best_predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            best_labels,
            best_predictions,
            labels=[0, 1],
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "warning": (
            "Small single-video dataset. Validation metrics are preliminary "
            "and do not establish generalization to other matches."
        ),
    }

    with open(MODEL_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nValidation report (best checkpoint):")
    print(classification_report(
        best_labels,
        best_predictions,
        labels=[0, 1],
        target_names=CLASSES,
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(best_labels, best_predictions, labels=[0, 1]))
    print(f"\nModel: {MODEL_DIR / 'best_team_classifier.pt'}")
    print(f"Metrics: {MODEL_DIR / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
