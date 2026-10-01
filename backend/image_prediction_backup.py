from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CROP_MODEL = (
    ROOT
    / "models"
    / "crop_classifier"
    / "crop_classifier_best.pth"
)

DISEASE_MODEL_ROOT = (
    ROOT
    / "models"
    / "disease"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 160

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
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
# MODEL BUILDER
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
# LOAD CROP MODEL
# ============================================================

def load_crop_model():

    if not CROP_MODEL.exists():
        raise FileNotFoundError(
            f"Crop model not found:\n{CROP_MODEL}"
        )

    checkpoint = torch.load(
        CROP_MODEL,
        map_location=DEVICE,
        weights_only=False
    )

    classes = checkpoint["classes"]

    model = create_model(
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, classes


# ============================================================
# LOAD DISEASE MODEL
# ============================================================

def load_disease_model(crop):

    model_path = (
        DISEASE_MODEL_ROOT
        / crop
        / "best_model.pth"
    )

    if not model_path.exists():
        return None, None

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    classes = checkpoint["classes"]

    model = create_model(
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, classes


# ============================================================
# PREDICT CROP
# ============================================================

def predict_crop(image):

    model, classes = load_crop_model()

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, index = torch.max(
            probabilities,
            dim=1
        )

    crop = classes[index.item()]

    confidence = confidence.item() * 100

    return crop, confidence


# ============================================================
# PREDICT DISEASE
# ============================================================

def predict_disease(image, crop):

    model, classes = load_disease_model(
        crop
    )

    if model is None:

        return {
            "available": False,
            "disease": "Disease model unavailable",
            "confidence": None,
            "status": "UNKNOWN"
        }

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, index = torch.max(
            probabilities,
            dim=1
        )

    disease = classes[index.item()]

    confidence = confidence.item() * 100

    healthy_words = [
        "healthy",
        "healthy_leaf",
        "healthy_plant",
        "healthy_rice_plant"
    ]

    disease_lower = disease.lower()

    is_healthy = any(
        word in disease_lower
        for word in healthy_words
    )

    status = (
        "HEALTHY"
        if is_healthy
        else "DISEASED"
    )

    return {
        "available": True,
        "disease": disease,
        "confidence": confidence,
        "status": status
    }


# ============================================================
# COMPLETE IMAGE PREDICTION
# ============================================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    crop, crop_confidence = predict_crop(
        image
    )

    disease_result = predict_disease(
        image,
        crop
    )

    return {
        "crop": crop,
        "crop_confidence": crop_confidence,
        "disease": disease_result["disease"],
        "disease_confidence": disease_result["confidence"],
        "status": disease_result["status"],
        "disease_model_available":
            disease_result["available"]
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print("Usage:")

        print(
            "python backend\\image_prediction.py "
            "path_to_image.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    result = predict_image(
        image_path
    )

    print()
    print("=" * 60)
    print("CROP + DISEASE PREDICTION")
    print("=" * 60)

    print(
        f"Crop       : {result['crop']}"
    )

    print(
        f"Crop conf. : "
        f"{result['crop_confidence']:.2f}%"
    )

    print(
        f"Disease    : {result['disease']}"
    )

    if result["disease_confidence"] is not None:

        print(
            f"Disease conf.: "
            f"{result['disease_confidence']:.2f}%"
        )

    print(
        f"Status     : {result['status']}"
    )

    print(
        f"Disease model available: "
        f"{result['disease_model_available']}"
    )

    print("=" * 60)