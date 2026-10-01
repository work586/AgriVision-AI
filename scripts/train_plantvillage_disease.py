from pathlib import Path
import argparse
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = ROOT / "datasets" / "processed" / "disease" / "plantvillage"
MODEL_ROOT = ROOT / "models" / "disease" / "plantvillage"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 160

# CPU-friendly batch size
BATCH_SIZE = 16

# Start with 10 epochs.
# We can increase later if necessary.
EPOCHS = 10

LEARNING_RATE = 0.0005

NUM_WORKERS = 0

RANDOM_SEED = 42


# ============================================================
# CROP NAME NORMALIZATION
# ============================================================

CROP_FOLDER_MAP = {
    "apple": "Apple",
    "cherry_(including_sour)": "Cherry",
    "corn": "Corn",
    "grape": "Grape",
    "peach": "Peach",
    "pepper_bell": "Pepper_Bell",
    "potato": "Potato",
    "strawberry": "Strawberry",
    "tomato": "Tomato",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TRANSFORMS
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

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
# MODEL
# ============================================================

def create_model(num_classes):

    model = models.efficientnet_b0(
        weights=weights
    )

    # Freeze pretrained feature extractor.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace classifier with the required number of disease classes.
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# TEST
# ============================================================

def test_model(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


# ============================================================
# DATASET INFORMATION
# ============================================================

def print_dataset_info(
    crop_folder,
    train_dataset,
    val_dataset,
    test_dataset
):

    print()
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print("Crop:", crop_folder)

    print("Classes:")
    for index, class_name in enumerate(
        train_dataset.classes
    ):
        print(f"  {index}: {class_name}")

    print()

    print(
        "Train images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    print(
        "Test images:",
        len(test_dataset)
    )

    print("=" * 70)
    print()


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_crop(crop_folder):

    if crop_folder not in CROP_FOLDER_MAP:

        raise ValueError(
            f"Unsupported crop: {crop_folder}\n"
            f"Supported crops: "
            f"{', '.join(CROP_FOLDER_MAP.keys())}"
        )

    crop_display_name = CROP_FOLDER_MAP[crop_folder]

    crop_root = DATASET_ROOT / crop_folder

    train_path = crop_root / "train"
    val_path = crop_root / "val"
    test_path = crop_root / "test"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training folder not found:\n{train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation folder not found:\n{val_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test folder not found:\n{test_path}"
        )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        train_path,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_path,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        test_path,
        transform=eval_transform
    )

    # Make sure all splits have exactly the same classes.
    if train_dataset.classes != val_dataset.classes:
        raise RuntimeError(
            "Train and validation classes do not match.\n"
            f"Train: {train_dataset.classes}\n"
            f"Val:   {val_dataset.classes}"
        )

    if train_dataset.classes != test_dataset.classes:
        raise RuntimeError(
            "Train and test classes do not match.\n"
            f"Train: {train_dataset.classes}\n"
            f"Test:  {test_dataset.classes}"
        )

    print_dataset_info(
        crop_display_name,
        train_dataset,
        val_dataset,
        test_dataset
    )

    # --------------------------------------------------------
    # DATA LOADERS
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

    model = create_model(
        num_classes=len(train_dataset.classes)
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # Only classifier parameters are trainable because
    # the EfficientNet feature extractor is frozen.
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir = MODEL_ROOT / crop_folder

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_model_path = output_dir / "best_model.pth"

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_state = None

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    print("Crop:", crop_display_name)
    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print()

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save the best validation model.
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state = copy.deepcopy(
                model.state_dict()
            )

            checkpoint = {
                "model_state_dict": best_state,
                "classes": train_dataset.classes,
                "image_size": IMAGE_SIZE,
                "crop": crop_folder,
                "val_accuracy": best_val_accuracy,
            }

            torch.save(
                checkpoint,
                best_model_path
            )

            print(
                f"  -> New best model saved: "
                f"{best_val_accuracy:.2f}%"
            )

    total_time = time.time() - start_time

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_accuracy = test_model(
        model,
        test_loader
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("Crop:", crop_display_name)
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Training time: "
        f"{total_time / 60:.1f} minutes"
    )

    print(
        "Model saved to:"
    )

    print(best_model_path)

    print("=" * 70)
    print()

    return {
        "crop": crop_folder,
        "classes": train_dataset.classes,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "test_images": len(test_dataset),
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "model_path": str(best_model_path),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a PlantVillage disease "
            "classifier for one crop."
        )
    )

    parser.add_argument(
        "--crop",
        required=True,
        choices=list(CROP_FOLDER_MAP.keys()),
        help="PlantVillage crop folder to train"
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("PLANTVILLAGE DISEASE MODEL TRAINER")
    print("=" * 70)
    print()
    print("Project root:", ROOT)
    print("Dataset root:", DATASET_ROOT)
    print("Model root:", MODEL_ROOT)
    print("Selected crop:", args.crop)
    print("Device:", DEVICE)
    print()

    train_crop(args.crop)


if __name__ == "__main__":
    main()