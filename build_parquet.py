"""
Builds data/price_data.parquet from the yearly Karnataka price CSVs.

Run from the project root folder:
    python build_parquet.py
"""
import glob

import pandas as pd

from backend.price_prediction import _commodity_pattern   # same crop filter the app uses

FILES = sorted(glob.glob("data/price/karnataka_price_*.csv"))   # NOT the raw daily_price_commodity file
TEXT_COLS = ["Commodity", "District", "Market", "Variety", "Grade"]
PRICE_COLS = ["Min_Price", "Max_Price", "Modal_Price"]
USECOLS = ["State", "Arrival_Date"] + TEXT_COLS + PRICE_COLS
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d %b %Y"]

if not FILES:
    raise SystemExit("No karnataka_price_*.csv files found in data/price/. Run this from the project root.")


def parse_dates(raw):
    s = raw.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        out.loc[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out


pattern = _commodity_pattern()
parts, dropped = [], 0

for f in FILES:
    for chunk in pd.read_csv(f, usecols=USECOLS, dtype=str, chunksize=500_000):
        chunk = chunk[chunk["State"].str.contains("Karnataka", case=False, na=False)]
        chunk = chunk[
            chunk["Commodity"].str.strip().str.lower().str.contains(pattern, na=False, regex=True)
        ].copy()
        if chunk.empty:
            continue

        chunk["Commodity"] = chunk["Commodity"].str.strip()
        chunk["Arrival_Date"] = parse_dates(chunk["Arrival_Date"])
        for c in PRICE_COLS:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("float32")

        before = len(chunk)
        chunk = chunk.dropna(subset=["Arrival_Date", "Modal_Price"])
        dropped += before - len(chunk)

        chunk[TEXT_COLS] = chunk[TEXT_COLS].fillna("")
        parts.append(chunk.drop(columns="State"))
    print("done", f, "| kept so far:", sum(len(p) for p in parts))

df = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)

df["_commodity_lc"] = df["Commodity"].str.lower()
for c in TEXT_COLS + ["_commodity_lc"]:
    df[c] = df[c].astype("category")

df.to_parquet("data/price_data.parquet", compression="zstd", index=False)

print()
print("rows:", len(df), "| dropped (bad date/price):", dropped)
print("dates:", df["Arrival_Date"].min().date(), "->", df["Arrival_Date"].max().date())
print(df["_commodity_lc"].value_counts().head(30))