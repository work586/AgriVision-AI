from pathlib import Path
import json
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = ROOT / "datasets" / "processed" / "disease"
MODEL_ROOT = ROOT / "models" / "disease"

MODEL_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 3

LEARNING_RATE = 0.0005

NUM_WORKERS = 0

SEED = 42

# Coconut has only 24 images.
# We keep it in the dataset but do not train a disease model for it.
CROPS = [
    "black_gram",
    "chilli",
    "cotton",
    "groundnut",
    "guava",
    "lemon",
    "mango",
    "rice",
    "sugarcane",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("DISEASE MODEL TRAINING")
print("=" * 70)
print(f"Device       : {device}")
print(f"Image size   : {IMAGE_SIZE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")
print(f"Dataset root : {DATASET_ROOT}")
print(f"Model root   : {MODEL_ROOT}")
print("=" * 70)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# TRAIN ONE CROP
# ============================================================

def train_crop(crop_name):

    print()
    print("=" * 70)
    print(f"TRAINING CROP: {crop_name.upper()}")
    print("=" * 70)

    crop_root = DATASET_ROOT / crop_name

    train_dir = crop_root / "train"
    val_dir = crop_root / "val"
    test_dir = crop_root / "test"

    if not train_dir.exists():
        print(f"ERROR: Missing {train_dir}")
        return None

    if not val_dir.exists():
        print(f"ERROR: Missing {val_dir}")
        return None

    if not test_dir.exists():
        print(f"ERROR: Missing {test_dir}")
        return None

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform
    )

    classes = train_dataset.classes

    print(f"Classes      : {len(classes)}")
    print(f"Train images : {len(train_dataset)}")
    print(f"Val images   : {len(val_dataset)}")
    print(f"Test images  : {len(test_dataset)}")

    print()
    print("Disease classes:")

    for i, class_name in enumerate(classes):
        print(f"  {i}: {class_name}")

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    weights = EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(
        weights=weights
    )

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        len(classes)
    )

    model = model.to(device)

    # --------------------------------------------------------
    # LOSS / OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    model_dir = MODEL_ROOT / crop_name
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "best_model.pth"

    for epoch in range(EPOCHS):

        start_time = time.time()

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_correct = 0
        train_total = 0
        train_loss_total = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = train_loss_total / train_total

        train_accuracy = (
            train_correct / train_total
        ) * 100

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss_total = 0.0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss_total += (
                    loss.item() * images.size(0)
                )

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_loss_total / val_total

        val_accuracy = (
            val_correct / val_total
        ) * 100

        elapsed = time.time() - start_time

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "crop": crop_name,
                "val_accuracy": val_accuracy,
            }

            torch.save(
                checkpoint,
                best_model_path
            )

            print(
                f"  Saved best model -> "
                f"{best_model_path}"
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ========================================================
    # TEST
    # ========================================================

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            test_correct += (
                predictions == labels
            ).sum().item()

            test_total += labels.size(0)

    test_accuracy = (
        test_correct / test_total
    ) * 100

    print()
    print("-" * 70)
    print(f"{crop_name.upper()} RESULT")
    print("-" * 70)
    print(f"Best validation accuracy : {best_val_accuracy:.2f}%")
    print(f"Test accuracy            : {test_accuracy:.2f}%")
    print(f"Model                    : {best_model_path}")
    print("-" * 70)

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "crop": crop_name,
        "classes": classes,
        "num_classes": len(classes),
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "test_images": len(test_dataset),
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "image_size": IMAGE_SIZE,
    }

    metadata_path = model_dir / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    results = {}

    overall_start = time.time()

    for crop in CROPS:

        result = train_crop(crop)

        if result is not None:
            results[crop] = result

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary_path = MODEL_ROOT / "disease_training_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    elapsed = time.time() - overall_start

    print()
    print()
    print("=" * 70)
    print("DISEASE TRAINING COMPLETE")
    print("=" * 70)

    for crop, result in results.items():

        print(
            f"{crop:15s} | "
            f"Val: {result['best_val_accuracy']:.2f}% | "
            f"Test: {result['test_accuracy']:.2f}%"
        )

    print()
    print(f"Summary: {summary_path}")
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()