import sys, pickle
import pandas as pd
import pyarrow as pa

print("python", sys.version)
print("pandas", pd.__version__)
print("pyarrow", pa.__version__)

with open("data/price_data_cache.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))
df = obj["df"] if isinstance(obj, dict) else obj
print(df.shape)
print(df.dtypes)
print(df.memory_usage(deep=True).sort_values(ascending=False))
print(df.head())