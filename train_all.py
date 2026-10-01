"""
Pre-train the price-forecast model for every crop, once.

Run from the project root:

    python train_all.py

Then keep models/price/*.pkl with your app (commit / upload them),
so nothing has to be trained while a user clicks "Analyze Image".
Re-run this whenever you add or change CSV files in data/price/.
"""

import time

from backend.price_prediction import CROP_NAME_MAP, load_price_data, train_price_model

print("Loading price data...")
start = time.perf_counter()
df = load_price_data()
print(f"  {len(df):,} rows loaded in {time.perf_counter() - start:.1f}s\n")

for crop in CROP_NAME_MAP:
    t0 = time.perf_counter()

    try:
        info = train_price_model(crop)
    except Exception as exc:
        print(f"{crop:12s} FAILED: {exc}")
        continue

    took = time.perf_counter() - t0

    if info.get("trained"):
        print(
            f"{crop:12s} OK   best={info['best_model']:18s} "
            f"MAE=₹{info['mae']:.2f}  ({took:.1f}s)"
        )
    else:
        print(f"{crop:12s} SKIP {info.get('reason')}")

print("\nDone. Deploy the files in models/price/ together with the app.")