"""
Shows, for every crop the app can predict, whether price data exists and
whether a 12-month forecast is possible.

Run from the project root:

    python check_crops.py
"""

from pathlib import Path

from backend.price_prediction import (
    CROP_NAME_MAP,
    MIN_OBSERVED_MONTHS,
    _build_monthly_series,
    find_commodity,
    load_price_data,
)

ROOT = Path(__file__).resolve().parent
CROP_MODEL = ROOT / "models" / "crop_classifier" / "crop_classifier_best.pth"

FALLBACK = [
    "Apple", "Black_Gram", "Blueberry", "Cherry", "Chilli", "Coconut",
    "Corn", "Cotton", "Grape", "Groundnut", "Guava", "Lemon",
    "Mango", "Orange", "Peach", "Pepper_Bell", "Potato", "Raspberry",
    "Rice", "Soybean", "Squash", "Strawberry", "Sugarcane", "Tomato",
]

# Use the exact class names stored in your trained crop model, if possible
crops = FALLBACK

try:
    import torch

    ckpt = torch.load(CROP_MODEL, map_location="cpu", weights_only=False)
    crops = list(ckpt["classes"])
    print(f"Using {len(crops)} class names from the crop model.\n")
except Exception as exc:
    print(f"(Could not read crop model classes: {exc}; using default list.)\n")

df = load_price_data()

print(f"{'CROP':22s} {'SEARCHES FOR':18s} {'ROWS':>8s} {'MONTHS':>7s}  RESULT")
print("-" * 78)

ok = 0

for crop in crops:
    searched = CROP_NAME_MAP.get(crop, crop)
    rows = len(find_commodity(df, crop))

    if rows == 0:
        print(f"{crop:22s} {searched:18s} {0:8d} {0:7d}  NO PRICE DATA")
        continue

    info = _build_monthly_series(df, crop)
    months = info[1] if info else 0

    if months >= MIN_OBSERVED_MONTHS:
        result = "FORECAST OK"
        ok += 1
    else:
        result = f"too little history (need {MIN_OBSERVED_MONTHS} months)"

    print(f"{crop:22s} {searched:18s} {rows:8d} {months:7d}  {result}")

print("-" * 78)
print(f"{ok} of {len(crops)} crops can get a 12-month forecast.")