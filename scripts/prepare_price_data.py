from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/price/daily_price_commodity_2025.csv")
OUTPUT_FILE = Path("data/price/karnataka_price_2025.csv")

print("Loading price dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df):,}")

# Keep only Karnataka
df = df[
    df["State"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "karnataka"
].copy()

print(f"Karnataka rows: {len(df):,}")

# Convert date
df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"],
    errors="coerce"
)

# Convert prices to numeric
for column in ["Min_Price", "Max_Price", "Modal_Price"]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Remove rows with invalid essential information
df = df.dropna(
    subset=[
        "Commodity",
        "Arrival_Date",
        "Modal_Price"
    ]
)

# Remove impossible prices
df = df[
    (df["Min_Price"] >= 0) &
    (df["Max_Price"] >= 0) &
    (df["Modal_Price"] >= 0)
]

# Remove duplicate records
df = df.drop_duplicates()

# Sort
df = df.sort_values(
    ["Commodity", "Arrival_Date", "Market"]
)

# Save
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPrice preparation completed.")
print(f"Output file: {OUTPUT_FILE}")
print(f"Final rows: {len(df):,}")

print("\nKarnataka commodities:")
print(
    df["Commodity"]
    .value_counts()
    .head(30)
)

print("\nDate range:")
print(df["Arrival_Date"].min())
print(df["Arrival_Date"].max())