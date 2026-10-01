from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "datasets" / "processed" / "disease"
MODEL_ROOT = ROOT / "models" / "disease"
OUTPUT_ROOT = ROOT / "reports" / "disease_evaluation"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

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

IMAGE_SIZE = 160
BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL
# ============================================================

def create_model(num_classes):

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model


# ============================================================
# EVALUATE ONE CROP
# ============================================================

def evaluate_crop(crop):

    print()
    print("=" * 70)
    print(f"EVALUATING: {crop.upper()}")
    print("=" * 70)

    test_dir = (
        DATA_ROOT
        / crop
        / "test"
    )

    model_path = (
        MODEL_ROOT
        / crop
        / "best_model.pth"
    )

    if not test_dir.exists():

        print(
            f"Test folder missing: {test_dir}"
        )

        return None

    if not model_path.exists():

        print(
            f"Model missing: {model_path}"
        )

        return None

    # Dataset
    dataset = datasets.ImageFolder(
        test_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    classes = dataset.classes

    # Load model
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    model = create_model(
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    all_predictions = []
    all_labels = []

    # Prediction
    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # Metrics
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(
        f"Test images : {len(dataset)}"
    )

    print(
        f"Accuracy    : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision   : {precision * 100:.2f}%"
    )

    print(
        f"Recall      : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score    : {f1 * 100:.2f}%"
    )

    print()
    print("CLASSIFICATION REPORT")
    print()

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=classes,
            zero_division=0
        )
    )

    # Save detailed JSON
    result = {
        "crop": crop,
        "test_images": len(dataset),
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "classes": classes,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    output_file = (
        OUTPUT_ROOT
        / f"{crop}_evaluation.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        f"Saved: {output_file}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DISEASE MODEL EVALUATION")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    results = []

    for crop in CROPS:

        result = evaluate_crop(crop)

        if result is not None:

            results.append(result)

    # Summary
    summary = []

    for result in results:

        summary.append({
            "crop": result["crop"],
            "test_images": result["test_images"],
            "accuracy": round(
                result["accuracy"] * 100,
                2
            ),
            "precision": round(
                result["weighted_precision"] * 100,
                2
            ),
            "recall": round(
                result["weighted_recall"] * 100,
                2
            ),
            "f1": round(
                result["weighted_f1"] * 100,
                2
            ),
        })

    summary_file = (
        OUTPUT_ROOT
        / "disease_evaluation_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"{'Crop':<15}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print("-" * 65)

    for row in summary:

        print(
            f"{row['crop']:<15}"
            f"{row['accuracy']:>11.2f}%"
            f"{row['precision']:>11.2f}%"
            f"{row['recall']:>11.2f}%"
            f"{row['f1']:>11.2f}%"
        )

    print()
    print(
        f"Summary saved to: {summary_file}"
    )


if __name__ == "__main__":
    main()