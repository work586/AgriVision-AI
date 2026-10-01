from pathlib import Path
import argparse
import pandas as pd
from backend.price_model import train_price_model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Historical mandi CSV with arrival_date and modal_price columns.")
    p.add_argument("--output", default="models/price_model.joblib")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    train_price_model(df, args.output)
    print("Saved:", args.output)
