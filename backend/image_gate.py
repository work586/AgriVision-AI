"""
AgriVision AI - Image Gate

Checks whether an uploaded image appears to contain a plant/leaf
before sending it to the crop classifier.

This is a generic image gate, not a crop classifier.

Flow:

    Uploaded image
          |
    Image quality check
          |
    CLIP plant/non-plant check
          |
       ACCEPT / REJECT

Install:
    pip install transformers

The first run downloads the CLIP model and caches it locally.
"""

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, ImageStat


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"


# ============================================================
# PROMPTS
# ============================================================

PLANT_PROMPTS = [
    "a close-up photograph of a plant leaf",
    "a close-up photograph of a crop leaf",
    "a photograph of a green plant leaf",
    "a photograph of a diseased plant leaf",
    "a photograph of a healthy plant leaf",
    "a photograph of a crop plant",
    "a photograph of leaves on a plant",
    # A leaf held in a hand is a normal field photo.
    "a photograph of a leaf held in a hand",
]


OTHER_PROMPTS = {
    "a person": "a photograph of a person",
    "an animal": "a photograph of an animal",
    "a vehicle": "a photograph of a car or vehicle",
    "a building": "a photograph of a building",
    "food": "a photograph of food",
    "furniture": "a photograph of furniture",
    "a room": "a photograph of a room",
    "a phone or computer": "a photograph of a phone or computer",
    "a screen or screenshot": "a screenshot from a computer or phone",
    "a document": "a photograph of a document",
    "a road": "a photograph of a road",
    "the sky": "a photograph of the sky",
    "an everyday object": "a photograph of an everyday object",
    "a shoe": "a photograph of a shoe",
    "a bag": "a photograph of a bag",
    "a tool": "a photograph of a tool",
    "bare soil": "a photograph of bare soil or dirt",
    "a drawing": "a drawing or cartoon of a plant",
}


# ============================================================
# CLIP THRESHOLDS
# ============================================================

# Starting values. These are NOT universal thresholds:
# tune them using your own field photos.

PLANT_THRESHOLD = 0.62

# Plant probability must beat the strongest non-plant
# category by at least this amount.
PLANT_MARGIN_THRESHOLD = 0.08


# ============================================================
# IMAGE QUALITY
# ============================================================

MIN_IMAGE_WIDTH = 160
MIN_IMAGE_HEIGHT = 160

MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 245.0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# LOAD CLIP
# ============================================================

@lru_cache(maxsize=1)
def _load():
    """Load CLIP only once per running process."""

    from transformers import CLIPModel, CLIPProcessor

    print(">>> [image_gate] Loading CLIP model...")

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    print(">>> [image_gate] CLIP model loaded.")

    return model, processor


def warm_up():
    """Load CLIP before the first user prediction."""

    _load()


# ============================================================
# OPEN IMAGE
# ============================================================

def open_image(image):
    """Accept a PIL Image or an image file path. Return an RGB PIL Image."""

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):

        image_path = Path(image)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found:\n{image_path}")

        with Image.open(image_path) as img:
            return img.convert("RGB")

    raise TypeError("Image must be a PIL Image or file path.")


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_image_quality(image):
    """
    Lightweight checks before CLIP.
    This does NOT determine whether the image is a leaf.
    """

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
                "message": (
                    "The image is too dark for reliable analysis. "
                    "Please retake the photo in daylight."
                ),
            }

        if brightness > MAX_BRIGHTNESS:
            return {
                "valid": False,
                "reason": "TOO_BRIGHT",
                "message": (
                    "The image is too bright or overexposed. Please retake "
                    "the photo in daylight without strong glare."
                ),
            }

        return {"valid": True, "reason": None, "message": None}

    except Exception as exc:
        return {
            "valid": False,
            "reason": "INVALID_IMAGE",
            "message": f"Unable to validate the image: {exc}",
        }


# ============================================================
# CLIP GATE
# ============================================================

def check_is_plant(image):
    """
    Determine whether an image looks like a plant/leaf.

    Returns:
        {
            "ok": bool,
            "plant_score": float,
            "best_nonplant_score": float,
            "plant_margin": float,
            "looks_like": str,
            "reason": str,
            "message": str,
        }

    reason is one of:
        LOW_RESOLUTION, TOO_DARK, TOO_BRIGHT, INVALID_IMAGE,
        NOT_PLANT_LIKELY, AMBIGUOUS_IMAGE, PLANT_DETECTED
    """

    # ---------------- image quality ----------------

    validation = validate_image_quality(image)

    if not validation["valid"]:
        return {
            "ok": False,
            "plant_score": 0.0,
            "best_nonplant_score": 1.0,
            "plant_margin": -1.0,
            "looks_like": "",
            "reason": validation["reason"],
            "message": validation["message"],
        }

    img = open_image(image)

    # ---------------- CLIP ----------------

    model, processor = _load()

    other_names = list(OTHER_PROMPTS.keys())

    prompts = PLANT_PROMPTS + [OTHER_PROMPTS[name] for name in other_names]

    inputs = processor(
        text=prompts,
        images=img,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(DEVICE) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = outputs.logits_per_image[0].softmax(dim=0)

    # ---------------- scores ----------------

    n_plant = len(PLANT_PROMPTS)

    plant_score = float(probabilities[:n_plant].sum().item())

    nonplant_probabilities = probabilities[n_plant:]

    best_nonplant_index = int(nonplant_probabilities.argmax().item())

    best_nonplant_score = float(nonplant_probabilities[best_nonplant_index].item())

    looks_like = other_names[best_nonplant_index]

    plant_margin = plant_score - best_nonplant_score

    # ---------------- decision ----------------

    enough_plant_score = plant_score >= PLANT_THRESHOLD
    enough_margin = plant_margin >= PLANT_MARGIN_THRESHOLD

    ok = enough_plant_score and enough_margin

    if ok:
        reason = "PLANT_DETECTED"
        message = "The image appears to contain a plant or leaf."

    elif not enough_plant_score:
        reason = "NOT_PLANT_LIKELY"
        message = (
            "This image does not appear to contain a plant or leaf. "
            "Please upload one clear leaf photo."
        )

    else:
        reason = "AMBIGUOUS_IMAGE"
        message = (
            "The image is ambiguous. The AI could not confidently determine "
            "that it contains a plant or leaf. Please upload one clear leaf "
            "on a plain background in daylight."
        )

    return {
        "ok": ok,
        "plant_score": plant_score,
        "best_nonplant_score": best_nonplant_score,
        "plant_margin": plant_margin,
        "looks_like": looks_like,
        "reason": reason,
        "message": message,
    }


# ============================================================
# CACHE CONTROL
# ============================================================

def clear_model_cache():

    _load.cache_clear()