import pandas as pd, glob
for f in sorted(glob.glob("data/*.csv")):
    d = pd.read_csv(f, nrows=3)
    print(f, "->", list(d.columns))