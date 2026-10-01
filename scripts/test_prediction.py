import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CROP_MODEL_PATH = ROOT / "models" / "crop_classifier" / "crop_classifier_best.pth"
DISEASE_MODEL_DIR = ROOT / "models" / "disease"

CROP_TEST_DIR = ROOT / "datasets" / "processed" / "crop" / "test"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):
    print("\nLoading:", model_path)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)

    print("Image size:", image_size)
    print("Number of classes:", len(classes))
    print("Classes:")

    for i, name in enumerate(classes):
        print(f"  {i}: {name}")

    # MobileNetV3-Large
    model = mobilenet_v3_large(
        weights=None,
        num_classes=len(classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)
    model.eval()

    return model, classes, image_size


# ============================================================
# IMAGE TRANSFORM
# ============================================================

def make_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ============================================================
# PREDICT
# ============================================================

def predict(model, classes, image_path, image_size):

    image = Image.open(image_path).convert("RGB")

    transform = make_transform(image_size)

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

    confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    predicted_class = classes[predicted_index]

    return predicted_class, confidence


# ============================================================
# FIND RANDOM TEST IMAGE
# ============================================================

def find_test_image():

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG"
    }

    images = [
        p
        for p in CROP_TEST_DIR.rglob("*")
        if p.is_file() and p.suffix in image_extensions
    ]

    if not images:
        raise FileNotFoundError(
            f"No test images found in:\n{CROP_TEST_DIR}"
        )

    return random.choice(images)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CROP MODEL PREDICTION TEST")
    print("=" * 70)

    print("\nCrop model:")
    print(CROP_MODEL_PATH)

    if not CROP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Crop model not found:\n{CROP_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load crop model
    # --------------------------------------------------------

    model, classes, image_size = load_model(
        CROP_MODEL_PATH
    )

    # --------------------------------------------------------
    # Select test image
    # --------------------------------------------------------

    image_path = find_test_image()

    print("\nTest image:")
    print(image_path)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predicted_class, confidence = predict(
        model,
        classes,
        image_path,
        image_size
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print("Image     :", image_path.name)
    print("Prediction:", predicted_class)
    print("Confidence:", f"{confidence * 100:.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()