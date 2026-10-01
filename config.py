from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
CROP_DATASET = DATASETS / "processed" / "crop"
DISEASE_DATASET = DATASETS / "processed" / "disease"
MODEL_DIR = ROOT / "models"
PRICE_DIR = ROOT / "data" / "price"

CROP_MODEL = MODEL_DIR / "crop_classifier.pt"
CROP_CLASSES = MODEL_DIR / "crop_classes.txt"

# Disease datasets currently prepared and suitable to train.
DISEASE_CROPS = [
    "black_gram", "chilli", "cotton", "groundnut",
    "guava", "lemon", "mango", "rice", "sugarcane"
]

CROP_NAMES = [
    "Apple","Black_Gram","Blueberry","Cherry","Chilli","Coconut",
    "Corn","Cotton","Grape","Groundnut","Guava","Lemon","Mango",
    "Orange","Peach","Pepper_Bell","Potato","Raspberry","Rice",
    "Soybean","Squash","Strawberry","Sugarcane","Tomato"
]
