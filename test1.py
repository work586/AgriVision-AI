import pandas as pd

obj = pd.read_pickle("data/price_data_cache.pkl")
print(type(obj))
if isinstance(obj, dict):
    print(obj.keys())
    df = obj["df"] if "df" in obj else next(v for v in obj.values() if isinstance(v, pd.DataFrame))
else:
    df = obj

print(df.shape)
print(df.dtypes)
print(df.memory_usage(deep=True).sort_values(ascending=False))
print(df.head())