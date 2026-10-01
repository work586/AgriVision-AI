"""
Replaces the CSV/pickle loader inside backend/price_prediction.py with a
loader that reads data/price_data.parquet.

Run from the project root folder:
    python patch_loader.py

A backup is saved as backend/price_prediction.py.bak first.
"""
from pathlib import Path
import shutil

TARGET = Path("backend/price_prediction.py")
START_MARK = "# LOAD PRICE DATA (all CSVs in data/price/)"
END_MARK = "# FIND COMMODITY (cached)"

NEW_BLOCK = '''# ============================================================
# LOAD PRICE DATA (data/price_data.parquet)
# ============================================================
# The raw CSVs in data/price/ are turned into one small parquet file
# by build_parquet.py. The app only ever reads that file.

PRICE_FILE = ROOT / "data" / "price_data.parquet"

_DATA_CACHE = {"key": None, "df": None}

_EXTRA_COMMODITY_NAMES = [
    "apple", "black gram", "blueberry", "cherry", "chilli", "coconut",
    "corn", "maize", "cotton", "grape", "groundnut", "guava", "lemon",
    "mango", "orange", "peach", "capsicum", "pepper", "potato",
    "raspberry", "rice", "paddy", "soybean", "soyabean", "squash",
    "strawberry", "sugarcane", "tomato",
]


def _commodity_pattern():
    # also used by build_parquet.py
    names = set(_EXTRA_COMMODITY_NAMES)
    for crop, mapped in CROP_NAME_MAP.items():
        names.add(crop.lower().replace("_", " "))
        names.add(mapped.lower())
    return "|".join(re.escape(n) for n in sorted(names))


def load_price_data():
    """Load data/price_data.parquet. Cached in memory until the file changes."""

    if not PRICE_FILE.exists():
        raise FileNotFoundError(
            f"Price data file not found:\\n{PRICE_FILE}\\n"
            "Run `python build_parquet.py` to create it from the CSVs in data/price/."
        )

    stat = PRICE_FILE.stat()
    key = (PRICE_FILE.name, stat.st_mtime, stat.st_size)

    if _DATA_CACHE["key"] == key:
        return _DATA_CACHE["df"]

    df = pd.read_parquet(PRICE_FILE)
    print(f">>> [price] loaded {len(df):,} rows from {PRICE_FILE.name}")

    _DATA_CACHE["key"] = key
    _DATA_CACHE["df"] = df

    # Anything cached against the old data is now stale
    _matches_cached.cache_clear()
    _series_cached.cache_clear()
    _latest_cached.cache_clear()
    _BUNDLE_CACHE.clear()

    return df


'''

if not TARGET.exists():
    raise SystemExit(f"{TARGET} not found. Run this from the project root (G:\\SCDDAPP).")

text = TARGET.read_text(encoding="utf-8")

if "PRICE_FILE" in text and "price_data.parquet" in text:
    raise SystemExit("This file already looks patched. Nothing to do.")

lines = text.splitlines(keepends=True)


def find_header(marker):
    hits = [i for i, l in enumerate(lines) if marker in l]
    if len(hits) != 1:
        raise SystemExit(f"Expected exactly one line containing {marker!r}, found {len(hits)}. "
                         "Not changing anything.")
    i = hits[0]
    # the header comment sits between two '# ====' lines; return the upper one
    if i > 0 and lines[i - 1].lstrip().startswith("# ===="):
        return i - 1
    return i


start = find_header(START_MARK)
end = find_header(END_MARK)

if not start < end:
    raise SystemExit("Section markers are in an unexpected order. Not changing anything.")

shutil.copy(TARGET, TARGET.with_name(TARGET.name + ".bak"))

new_text = "".join(lines[:start]) + NEW_BLOCK + "".join(lines[end:])
TARGET.write_text(new_text, encoding="utf-8")

print(f"Patched {TARGET}. Backup saved as {TARGET.name}.bak")
print(f"Replaced lines {start + 1}-{end} with the parquet loader.")