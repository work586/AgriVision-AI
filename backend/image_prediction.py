import os
import re
from pathlib import Path
from functools import lru_cache

import torch
import torch.nn as nn
from PIL import Image, ImageStat
from torchvision import models, transforms


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT / "models"


# ============================================================
# MODEL PATH
# ============================================================

CROP_MODEL_PATH = MODELS_DIR / "crop_classifier" / "crop_classifier_best.pth"


# ============================================================
# CROP CONFIDENCE SETTINGS
# ============================================================

# A crop prediction is "reliable" only if BOTH conditions hold:
#   - top-1 confidence >= CROP_CONFIDENCE_THRESHOLD (percent)
#   - top-1 minus top-2 confidence >= CROP_MARGIN_THRESHOLD (points)
CROP_CONFIDENCE_THRESHOLD = 60.0
CROP_MARGIN_THRESHOLD = 15.0


# ============================================================
# IMAGE QUALITY
# ============================================================

MIN_IMAGE_WIDTH = 160
MIN_IMAGE_HEIGHT = 160

MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 245.0


# ============================================================
# NORTH KARNATAKA DISEASE MODELS
# ============================================================

NORTH_KARNATAKA_DISEASE_MODELS = {
    "black_gram": "black_gram",
    "chilli": "chilli",
    "cotton": "cotton",
    "groundnut": "groundnut",
    "guava": "guava",
    "lemon": "lemon",
    "mango": "mango",
    "rice": "rice",
    "sugarcane": "sugarcane",
}


# ============================================================
# PLANTVILLAGE CROPS
# ============================================================

PLANTVILLAGE_CROPS = {
    "apple",
    "cherry",
    "corn",
    "grape",
    "peach",
    "pepper_bell",
    "potato",
    "strawberry",
    "tomato",
    "soybean",
    "raspberry",
    "squash",
    "orange",
    "blueberry",
}


# ============================================================
# ACTUAL PLANTVILLAGE DISEASE MODELS
# ============================================================

PLANTVILLAGE_DISEASE_MODELS = {
    "apple": "plantvillage/apple",
    "cherry": "plantvillage/cherry_(including_sour)",
    "corn": "plantvillage/corn",
    "grape": "plantvillage/grape",
    "peach": "plantvillage/peach",
    "pepper_bell": "plantvillage/pepper_bell",
    "potato": "plantvillage/potato",
    "strawberry": "plantvillage/strawberry",
    "tomato": "plantvillage/tomato",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")

torch.set_num_threads(min(4, os.cpu_count() or 1))


# ============================================================
# TRANSFORM
# ============================================================

@lru_cache(maxsize=8)
def create_transform(image_size=160):

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ============================================================
# NORMALIZE CROP NAME
# ============================================================

def normalize_crop_name(crop):

    if crop is None:
        return ""

    value = str(crop).strip().lower()
    value = value.replace(" ", "_")
    value = value.replace("-", "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value


# ============================================================
# DISPLAY CROP NAME
# ============================================================

def display_crop_name(crop):

    if not crop:
        return "Unknown"

    value = str(crop)
    value = value.replace("_", " ")
    value = value.replace("(including sour)", "")

    return value.title().strip()


# ============================================================
# HEALTHY LABEL CHECK
# ============================================================

def is_healthy_label(label):
    """
    True only for genuinely healthy classes.

    "healthy"        -> True
    "Tomato_healthy" -> True
    "unhealthy"      -> False
    "not_healthy"    -> False
    """

    text = re.sub(r"[_\-\s]+", " ", str(label).strip().lower())

    if "unhealthy" in text:
        return False

    if re.search(r"\b(not|non)\s+healthy\b", text):
        return False

    return re.search(r"\bhealthy\b", text) is not None


# ============================================================
# OPEN IMAGE
# ============================================================

def open_image(image):

    if isinstance(image, (str, Path)):

        image_path = Path(image)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found:\n{image_path}")

        with Image.open(image_path) as img:
            return img.convert("RGB")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    raise TypeError("Image input must be either a file path or PIL Image.")


# ============================================================
# IMAGE QUALITY
# ============================================================

def validate_image_quality(image):

    try:
        img = open_image(image)

        width, height = img.size

        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            return {
                "valid": False,
                "reason": "LOW_RESOLUTION",
                "message": (
                    "The uploaded image is too small for reliable analysis. "
                    "Please upload a clearer close-up photo of one leaf."
                ),
            }

        small = img.resize((64, 64))
        grayscale = small.convert("L")
        brightness = ImageStat.Stat(grayscale).mean[0]

        if brightness < MIN_BRIGHTNESS:
            return {
                "valid": False,
                "reason": "TOO_DARK",
                "message": "The image is too dark. Please retake the photo in daylight.",
            }

        if brightness > MAX_BRIGHTNESS:
            return {
                "valid": False,
                "reason": "TOO_BRIGHT",
                "message": (
                    "The image is too bright or overexposed. Please retake "
                    "the photo without strong glare."
                ),
            }

        return {"valid": True, "reason": None, "message": None}

    except Exception as exc:
        return {
            "valid": False,
            "reason": "INVALID_IMAGE",
            "message": f"Unable to validate image: {exc}",
        }


# ============================================================
# BUILD MODEL
# ============================================================

def build_efficientnet(num_classes):

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(path):

    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found:\n{path}")

    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)

    except TypeError:
        return torch.load(path, map_location=DEVICE)


# ============================================================
# LOAD CROP MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_crop_model():

    print(">>> [image_prediction] Loading crop model...")

    checkpoint = load_checkpoint(CROP_MODEL_PATH)

    classes = checkpoint.get("classes")

    if not classes:
        raise ValueError("Crop checkpoint does not contain 'classes'.")

    image_size = checkpoint.get("image_size", 160)

    model = build_efficientnet(len(classes))

    state_dict = checkpoint.get("model_state_dict")

    if state_dict is None:
        raise ValueError("Crop checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, tuple(classes), image_size


# ============================================================
# LOAD DISEASE MODEL
# ============================================================

@lru_cache(maxsize=32)
def load_disease_model(model_folder):

    model_path = MODELS_DIR / "disease" / model_folder / "best_model.pth"

    if not model_path.exists():
        return None, None, None

    checkpoint = load_checkpoint(model_path)

    classes = checkpoint.get("classes")

    if not classes:
        raise ValueError(
            f"Disease checkpoint does not contain 'classes':\n{model_path}"
        )

    image_size = checkpoint.get("image_size", 160)

    model = build_efficientnet(len(classes))

    state_dict = checkpoint.get("model_state_dict")

    if state_dict is None:
        raise ValueError("Disease checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, tuple(classes), image_size


# ============================================================
# FIND DISEASE MODEL
# ============================================================

def find_disease_model(crop):

    normalized_crop = normalize_crop_name(crop)

    if normalized_crop in NORTH_KARNATAKA_DISEASE_MODELS:
        return NORTH_KARNATAKA_DISEASE_MODELS[normalized_crop], "North Karnataka"

    if normalized_crop in PLANTVILLAGE_DISEASE_MODELS:
        return PLANTVILLAGE_DISEASE_MODELS[normalized_crop], "PlantVillage"

    return None, None


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image_tensor(image, image_size):

    image = open_image(image)

    transform = create_transform(int(image_size))

    tensor = transform(image).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# PREDICT CROP
# ============================================================

def _blocked_crop_result(status, message):
    """Shared shape for a crop result that must not be trusted."""

    return {
        "crop": None,
        "crop_confidence": None,
        "top_crop_predictions": [],
        "other_crop_predictions": [],
        "prediction_status": status,
        "prediction_message": message,
        "prediction_reliable": False,
        "confidence_margin": None,
        "low_confidence": True,
        "small_confidence_margin": False,
    }


def predict_crop(image):

    validation = validate_image_quality(image)

    if not validation["valid"]:
        return _blocked_crop_result("INVALID_IMAGE", validation["message"])

    model, classes, image_size = load_crop_model()

    image_tensor = prepare_image_tensor(image, image_size)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    top_k = min(3, len(classes))

    top_probabilities, top_indices = torch.topk(probabilities, top_k, dim=1)

    predictions = []

    for probability, index in zip(top_probabilities[0], top_indices[0]):
        predictions.append(
            {
                "crop": classes[index.item()],
                "confidence": float(probability.item() * 100.0),
            }
        )

    if not predictions:
        return _blocked_crop_result(
            "UNCERTAIN",
            "The AI could not identify a supported crop.",
        )

    best_prediction = predictions[0]

    crop = best_prediction["crop"]
    crop_confidence = best_prediction["confidence"]

    if len(predictions) >= 2:
        confidence_margin = crop_confidence - predictions[1]["confidence"]
    else:
        confidence_margin = None

    low_confidence = crop_confidence < CROP_CONFIDENCE_THRESHOLD

    small_confidence_margin = (
        confidence_margin is not None
        and confidence_margin < CROP_MARGIN_THRESHOLD
    )

    prediction_reliable = not low_confidence and not small_confidence_margin

    if prediction_reliable:
        prediction_status = "RELIABLE"
        prediction_message = (
            "The crop prediction is sufficiently confident for further analysis."
        )
    else:
        prediction_status = "UNCERTAIN"
        prediction_message = (
            "The AI is not sufficiently confident that this image belongs to "
            "one of the supported crop classes. The image may contain an "
            "unsupported crop or image conditions outside the model's training "
            "data. Please retake the photo using one clear leaf in daylight "
            "against a plain background."
        )

    return {
        "crop": crop,
        "crop_confidence": crop_confidence,
        "top_crop_predictions": predictions,
        "other_crop_predictions": predictions[1:],
        "prediction_status": prediction_status,
        "prediction_message": prediction_message,
        "prediction_reliable": prediction_reliable,
        "confidence_margin": confidence_margin,
        "low_confidence": low_confidence,
        "small_confidence_margin": small_confidence_margin,
    }


# ============================================================
# PREDICT DISEASE
# ============================================================

def _no_disease_result(model_source=None):

    return {
        "disease": "Disease model unavailable",
        "disease_confidence": None,
        "status": "UNKNOWN",
        "disease_model_available": False,
        "disease_model_source": model_source,
    }


def predict_disease(image, crop):

    model_folder, model_source = find_disease_model(crop)

    if model_folder is None:
        return _no_disease_result(None)

    model, classes, image_size = load_disease_model(model_folder)

    if model is None:
        return _no_disease_result(model_source)

    image_tensor = prepare_image_tensor(image, image_size)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    probability, index = torch.max(probabilities, dim=1)

    disease = classes[index.item()]

    disease_confidence = float(probability.item() * 100.0)

    status = "HEALTHY" if is_healthy_label(disease) else "DISEASED"

    return {
        "disease": disease,
        "disease_confidence": disease_confidence,
        "status": status,
        "disease_model_available": True,
        "disease_model_source": model_source,
    }


# ============================================================
# COMPLETE IMAGE PREDICTION
# ============================================================

def predict_image(image):

    crop_result = predict_crop(image)

    if not crop_result.get("prediction_reliable", False):
        return {
            **crop_result,
            "disease": "Disease cannot be recognized",
            "disease_confidence": None,
            "status": "UNCERTAIN",
            "disease_model_available": False,
            "disease_model_source": None,
            "analysis_blocked": True,
            "analysis_block_reason": crop_result.get("prediction_message"),
        }

    disease_result = predict_disease(image, crop_result["crop"])

    return {
        **crop_result,
        **disease_result,
        "analysis_blocked": False,
        "analysis_block_reason": None,
    }


# ============================================================
# COMPATIBILITY
# ============================================================

def analyze_image(image):

    return predict_image(image)


# ============================================================
# WARM UP
# ============================================================

def warm_up():

    load_crop_model()


# ============================================================
# CACHE CLEAR
# ============================================================

def clear_model_cache():

    load_crop_model.cache_clear()
    load_disease_model.cache_clear()
    create_transform.cache_clear()


# ============================================================
# CLI
# ============================================================

def print_result(result):

    print()
    print("=" * 60)
    print("CROP + DISEASE PREDICTION")
    print("=" * 60)

    print(f"Crop       : {result.get('crop', 'Unknown')}")

    confidence = result.get("crop_confidence")

    if confidence is None:
        print("Crop conf. : N/A")
    else:
        print(f"Crop conf. : {confidence:.2f}%")

    print(f"Prediction : {result.get('prediction_status', 'UNKNOWN')}")

    predictions = result.get("top_crop_predictions", [])

    if predictions:
        print()
        print("Top 3 crop predictions:")

        for position, item in enumerate(predictions, start=1):
            print(
                f"  {position}. {item.get('crop', 'Unknown')}: "
                f"{item.get('confidence', 0):.2f}%"
            )

    print()
    print(f"Disease    : {result.get('disease', 'Unknown')}")
    print(f"Status     : {result.get('status', 'UNKNOWN')}")
    print(f"Analysis blocked: {result.get('analysis_blocked', False)}")
    print("=" * 60)
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="AgriVision AI crop and disease prediction"
    )

    parser.add_argument("image", nargs="?", help="Path to image")

    args = parser.parse_args()

    if not args.image:
        print("Usage:")
        print('python backend\\image_prediction.py "path\\to\\image.jpg"')
        raise SystemExit(0)

    print_result(analyze_image(args.image))