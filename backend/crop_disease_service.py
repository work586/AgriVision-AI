from pathlib import Path
from backend.image_predictor import ImageModel

def load_disease_model(root, crop):
    path = Path(root) / "models" / "disease" / f"{crop.lower()}.pt"
    if not path.exists():
        return None
    return ImageModel(path)
