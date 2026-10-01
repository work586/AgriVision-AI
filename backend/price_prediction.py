import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Put ALL yearly price CSVs (2021, 2022, ... 2025/2026) in this folder.
PRICE_DIR = ROOT / "data" / "price"

MODEL_DIR = ROOT / "models" / "price"

# Bump this number whenever the model/feature format changes;
# older saved .pkl files are then retrained automatically.
MODEL_FORMAT = 3

# Forecasting settings
MIN_OBSERVED_MONTHS = 15      # minimum months of real data needed to train
YEARLY_LAG_MIN_MONTHS = 30    # use the "same month last year" feature from here


# ============================================================
# CROP NAME MAPPING
# ============================================================

CROP_NAME_MAP = {
    "Apple": "Apple",
    "Black_Gram": "Black Gram",
    "Chilli": "Chilli",
    "Coconut": "Coconut",
    "Cotton": "Cotton",
    "Groundnut": "Groundnut",
    "Guava": "Guava",
    "Lemon": "Lemon",
    "Mango": "Mango",
    "Rice": "Rice",
    "Sugarcane": "Sugarcane",
    "Tomato": "Tomato",
    "Corn": "Maize",
    "Soybean": "Soyabean",
    "Pepper_Bell": "Capsicum",   # market data lists bell pepper as Capsicum
}


# ============================================================
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
            f"Price data file not found:\n{PRICE_FILE}\n"
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


# ============================================================
# FIND COMMODITY (cached)
# ============================================================

@lru_cache(maxsize=64)
def _matches_cached(crop, data_key):
    df = _DATA_CACHE["df"]

    search_name = CROP_NAME_MAP.get(
        crop,
        crop
    ).lower()

    # Exact match first
    exact = df[df["_commodity_lc"] == search_name]

    if not exact.empty:
        return exact

    # Partial match as fallback
    return df[
        df["_commodity_lc"].str.contains(
            search_name,
            na=False,
            regex=False,
        )
    ]


def find_commodity(df, crop):
    """
    Rows of `df` for this crop. `df` is accepted for backward
    compatibility; the lookup itself is cached per crop.
    """

    load_price_data()  # makes sure the cache key is current

    return _matches_cached(crop, _DATA_CACHE["key"])


# ============================================================
# GET LATEST PRICE (cached)
# ============================================================

def _get_latest_price_raw(crop):

    df = load_price_data()

    matches = find_commodity(
        df,
        crop
    )

    if matches.empty:

        return {
            "available": False,
            "crop": crop,
            "message": (
                "No market price data found "
                "for this crop."
            )
        }

    # Find latest available date
    latest_date = matches[
        "Arrival_Date"
    ].max()

    latest = matches[
        matches["Arrival_Date"]
        == latest_date
    ].copy()

    # Sort by market name for stable output
    latest = latest.sort_values(
        by=["Market", "Variety"]
    )

    records = []

    for _, row in latest.iterrows():

        records.append({
            "district": str(
                row["District"]
            ),
            "market": str(
                row["Market"]
            ),
            "commodity": str(
                row["Commodity"]
            ),
            "variety": str(
                row["Variety"]
            ),
            "grade": str(
                row["Grade"]
            ),
            "date": row[
                "Arrival_Date"
            ].strftime("%Y-%m-%d"),
            "min_price": float(
                row["Min_Price"]
            )
            if pd.notna(row["Min_Price"])
            else None,
            "max_price": float(
                row["Max_Price"]
            )
            if pd.notna(row["Max_Price"])
            else None,
            "modal_price": float(
                row["Modal_Price"]
            )
            if pd.notna(row["Modal_Price"])
            else None,
        })

    # Overall values for the latest date
    modal_prices = latest[
        "Modal_Price"
    ].dropna()

    min_prices = latest[
        "Min_Price"
    ].dropna()

    max_prices = latest[
        "Max_Price"
    ].dropna()

    return {
        "available": True,
        "crop": crop,
        "date": latest_date.strftime(
            "%Y-%m-%d"
        ),
        "records": records,
        "min_price": (
            float(min_prices.min())
            if not min_prices.empty
            else None
        ),
        "max_price": (
            float(max_prices.max())
            if not max_prices.empty
            else None
        ),
        "modal_price": (
            float(modal_prices.mean())
            if not modal_prices.empty
            else None
        ),
        "market_count": len(records)
    }


@lru_cache(maxsize=64)
def _latest_cached(crop, data_key):
    return _get_latest_price_raw(crop)


def get_latest_price(crop):

    load_price_data()  # makes sure the cache key is current

    return _latest_cached(crop, _DATA_CACHE["key"])


# ============================================================
# MONTHLY PRICE SERIES + FEATURES
# ============================================================
# A year-ahead forecast works much better on MONTHLY averages than on
# daily prices: daily data has gaps and market-mix noise, while monthly
# averages show the real seasonal pattern.

def _build_monthly_series(df, crop):
    """
    Average modal price per month across all Karnataka markets.
    Returns (monthly_series, observed_months) or None if no data.
    Months with no records inside the range are filled by interpolation.
    """

    matches = find_commodity(df, crop)

    if matches.empty:
        return None

    # Average per day first, so busy days/markets don't dominate a month
    daily = matches.groupby("Arrival_Date")["Modal_Price"].mean()

    monthly = daily.groupby(daily.index.to_period("M")).mean()

    observed = len(monthly)

    full_index = pd.period_range(
        monthly.index.min(),
        monthly.index.max(),
        freq="M",
    )

    monthly = monthly.reindex(full_index).interpolate(method="linear")

    return monthly, observed


@lru_cache(maxsize=64)
def _series_cached(crop, data_key):
    return _build_monthly_series(_DATA_CACHE["df"], crop)


def _get_series(crop):
    load_price_data()  # makes sure the cache key is current

    return _series_cached(crop, _DATA_CACHE["key"])


def _make_features(history, period, use_yearly):
    """
    Features for predicting the price of `period`, using only the
    monthly prices in `history` (all months BEFORE `period`).
    The same function is used for training and for forecasting, so
    they can never drift apart.
    """

    m = period.month

    feats = {
        # month as a circle, so December is "next to" January
        "month_sin": float(np.sin(2 * np.pi * m / 12)),
        "month_cos": float(np.cos(2 * np.pi * m / 12)),
        "lag_1": float(history[-1]),
        "lag_2": float(history[-2]),
        "lag_3": float(history[-3]),
        "avg_3": float(np.mean(history[-3:])),
    }

    if use_yearly:
        # price in the same month one year ago
        feats["lag_12"] = float(history[-12])

    return feats


def _signature(monthly):
    """Fingerprint of the training data; changes when the CSVs change."""
    return (
        f"{len(monthly)}|{monthly.index[0]}|{monthly.index[-1]}|"
        f"{float(monthly.sum()):.2f}"
    )


def _model_path(crop):
    return MODEL_DIR / f"{crop.lower().replace(' ', '_')}_price_model.pkl"


def _load_model(crop):
    path = _model_path(crop)

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


# ============================================================
# TRAIN PRICE MODEL
# ============================================================

def train_price_model(crop):
    """
    Train Linear Regression, Decision Tree and Random Forest on the crop's
    monthly price history, compare them on the most recent months (time-based
    holdout) and save the best one.

    NOTE ON SPEED: run `python train_all.py` once so this never has to
    happen during a user's "Analyze Image" click. The result is cached to
    disk in MODEL_DIR and reused until the price CSVs change.
    """

    # Imported here rather than at module level: scikit-learn is one of
    # the slower imports in the Python ecosystem, and this is the ONLY
    # function in the module that needs it.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.tree import DecisionTreeRegressor

    info = _get_series(crop)

    if info is None:
        return {
            "trained": False,
            "crop": crop,
            "reason": "No price data found for this crop.",
        }

    monthly, observed = info

    if observed < MIN_OBSERVED_MONTHS:
        return {
            "trained": False,
            "crop": crop,
            "reason": (
                f"Only {observed} months of price history found; "
                f"at least {MIN_OBSERVED_MONTHS} are needed for a yearly forecast."
            ),
        }

    values = [float(v) for v in monthly.values]
    periods = list(monthly.index)

    use_yearly = len(values) >= YEARLY_LAG_MIN_MONTHS
    start = 12 if use_yearly else 3

    rows, targets = [], []

    for i in range(start, len(values)):
        rows.append(_make_features(values[:i], periods[i], use_yearly))
        targets.append(values[i])

    X = pd.DataFrame(rows)
    y = pd.Series(targets)
    feature_columns = list(X.columns)

    # Time-based split: the most recent months are held out for testing
    test_size = max(3, int(len(X) * 0.2))
    split = len(X) - test_size

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    candidates = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=2, random_state=42
        ),
        # n_jobs=1: on a dataset this small, spawning parallel workers
        # costs more time than it saves.
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=2,
            random_state=42, n_jobs=1,
        ),
    }

    scores = {}
    best_name, best_model, best_mae = None, None, float("inf")

    for name, model in candidates.items():
        model.fit(X_train, y_train)

        mae = mean_absolute_error(y_test, model.predict(X_test))
        scores[name] = round(float(mae), 2)

        if mae < best_mae:
            best_name, best_model, best_mae = name, model, mae

    # Refit the winning model type on ALL data for the saved version
    best_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "format": MODEL_FORMAT,
            "signature": _signature(monthly),
            "model": best_model,
            "model_name": best_name,
            "feature_columns": feature_columns,
            "use_yearly": use_yearly,
            "history": values[-24:],
            "last_period": str(periods[-1]),
            "price_floor": 0.5 * min(values),
            "price_ceiling": 1.5 * max(values),
            "mae": scores[best_name],
        },
        _model_path(crop),
    )

    # A freshly trained model replaces whatever was cached
    _BUNDLE_CACHE.pop(crop, None)

    return {
        "trained": True,
        "crop": crop,
        "best_model": best_name,
        "mae": scores[best_name],
        "all_scores": scores,
        "data_points": len(values),
        "observed_months": observed,
        "uses_last_year_feature": use_yearly,
        "model_path": str(_model_path(crop)),
    }


# In-memory copy of the loaded model bundles, so the .pkl file is
# read from disk at most once per crop per process.
_BUNDLE_CACHE = {}


def _ensure_bundle(crop):
    """
    Load the saved model. Retrain automatically if it is missing, was
    saved by an older version of this file, or the price CSVs changed.
    """

    info = _get_series(crop)

    if info is None:
        return None, "No price data found for this crop."

    monthly, _ = info
    signature = _signature(monthly)

    cached = _BUNDLE_CACHE.get(crop)

    if cached is not None and cached.get("signature") == signature:
        return cached, None

    bundle = _load_model(crop)

    if (
        bundle is None
        or bundle.get("format") != MODEL_FORMAT
        or bundle.get("signature") != signature
    ):
        result = train_price_model(crop)

        if not result.get("trained"):
            return None, result.get(
                "reason", "Could not train a price model for this crop."
            )

        bundle = _load_model(crop)

        if bundle is None:
            return None, "Could not load the trained price model."

    _BUNDLE_CACHE[crop] = bundle

    return bundle, None


# ============================================================
# RECURSIVE MONTHLY FORECAST
# ============================================================

def forecast_monthly(crop, steps):
    """
    Forecast `steps` months beyond the last month with data. Each predicted
    month is fed back in as history to predict the next one.
    """

    bundle, error = _ensure_bundle(crop)

    if bundle is None:
        return {"error": error}

    model = bundle["model"]
    cols = bundle["feature_columns"]
    use_yearly = bundle["use_yearly"]
    history = list(bundle["history"])
    last_period = pd.Period(bundle["last_period"], freq="M")

    floor, ceiling = bundle["price_floor"], bundle["price_ceiling"]

    periods, prices = [], []

    for step in range(1, steps + 1):
        period = last_period + step

        feats = _make_features(history, period, use_yearly)
        X = pd.DataFrame([feats])[cols]

        pred = float(model.predict(X)[0])

        # Safety net: keep predictions in a sane range so a runaway
        # recursive forecast can't produce absurd prices.
        pred = min(max(pred, floor), ceiling)

        history.append(pred)
        periods.append(period)
        prices.append(pred)

    return {
        "forecast": pd.Series(prices, index=pd.PeriodIndex(periods, freq="M")),
        "model_name": bundle["model_name"],
        "mae": bundle["mae"],
        "last_period": last_period,
    }


def predict_future_price(crop, days_ahead=30):
    """
    Kept for backward compatibility. Forecasts are monthly now, so
    `days_ahead` is rounded up to whole months after today.
    """

    months = max(1, int(np.ceil(days_ahead / 30)))
    now_period = pd.Timestamp.today().to_period("M")

    bundle, error = _ensure_bundle(crop)

    if bundle is None:
        return {"available": False, "crop": crop, "message": error}

    last_period = pd.Period(bundle["last_period"], freq="M")
    target = now_period + months
    steps = max(1, (target - last_period).n)

    out = forecast_monthly(crop, steps)

    if "error" in out:
        return {"available": False, "crop": crop, "message": out["error"]}

    price_value = float(out["forecast"].iloc[-1])

    return {
        "available": True,
        "crop": crop,
        "predicted_price": price_value,
        "predicted_date": out["forecast"].index[-1].strftime("%b %Y"),
        "days_ahead": days_ahead,
        "model_used": out["model_name"],
        "based_on_date": str(last_period),
    }


# ============================================================
# YEARLY OUTLOOK + SELLING RECOMMENDATION
# ============================================================

def get_yearly_price_outlook(crop):
    """
    12-month price outlook starting next month: monthly forecast,
    best / worst month to sell, and a plain-language recommendation.
    """

    latest = get_latest_price(crop)

    if not latest.get("available"):
        return {
            "available": False,
            "crop": crop,
            "message": latest.get("message", "No current price data available."),
        }

    bundle, error = _ensure_bundle(crop)

    if bundle is None:
        return {"available": False, "crop": crop, "message": error}

    # ---- how far to forecast: from last data month to 12 months after today ----
    last_period = pd.Period(bundle["last_period"], freq="M")
    now_period = pd.Timestamp.today().to_period("M")

    first_wanted = max(now_period + 1, last_period + 1)
    steps = (first_wanted + 11 - last_period).n

    out = forecast_monthly(crop, steps)

    if "error" in out:
        return {"available": False, "crop": crop, "message": out["error"]}

    # keep only the 12 months we actually want to show
    monthly = out["forecast"][out["forecast"].index >= first_wanted].iloc[:12]

    # ---- "current" price = average of the last 30 days that have data ----
    matches = find_commodity(None, crop)
    last_date = matches["Arrival_Date"].max()

    recent = matches[matches["Arrival_Date"] > last_date - pd.Timedelta(days=30)]
    current_price = float(recent["Modal_Price"].mean())

    data_age_days = max(0, (pd.Timestamp.today().normalize() - last_date).days)

    avg_next_year = float(monthly.mean())

    pct_change = (
        (avg_next_year - current_price) / current_price * 100
        if current_price
        else 0.0
    )

    best_period, worst_period = monthly.idxmax(), monthly.idxmin()
    best_price, worst_price = float(monthly.max()), float(monthly.min())

    if pct_change > 3:
        trend = "rising"
    elif pct_change < -3:
        trend = "falling"
    else:
        trend = "stable"

    action = (
        f"Over the next 12 months the average price is expected to be about "
        f"₹{avg_next_year:,.2f} ({pct_change:+.1f}% vs the recent average). "
        f"The best month to sell is likely {best_period.strftime('%B %Y')} "
        f"(about ₹{best_price:,.2f}); the weakest is "
        f"{worst_period.strftime('%B %Y')} (about ₹{worst_price:,.2f})."
    )

    if data_age_days > 60:
        action += (
            f" Note: the latest market data is from "
            f"{last_date.strftime('%d %b %Y')}, so this outlook is less certain."
        )

    # ---- best market among those reporting on the latest date ----
    best_market = None

    priced = [
        r for r in latest.get("records", []) if r.get("modal_price") is not None
    ]

    if priced:
        b = max(priced, key=lambda r: r["modal_price"])
        best_market = f"{b['market']} ({b['district']}) at ₹{b['modal_price']:.2f}"

    return {
        "available": True,
        "crop": crop,
        "current_price": round(current_price, 2),
        "current_date": last_date.strftime("%Y-%m-%d"),
        "predicted_price": round(best_price, 2),          # peak month price
        "predicted_date": best_period.strftime("%b %Y"),   # peak month
        "worst_price": round(worst_price, 2),
        "worst_date": worst_period.strftime("%b %Y"),
        "average_next_year": round(avg_next_year, 2),
        "days_ahead": 365,
        "percent_change": round(pct_change, 2),
        "trend": trend,
        "model_used": out["model_name"],
        "model_mae": out["mae"],
        "data_age_days": int(data_age_days),
        "best_market_now": best_market,
        "recommendation": action,
        "monthly": [
            {
                "month_key": p.strftime("%Y-%m"),   # sorts correctly in charts
                "month": p.strftime("%b %Y"),
                "price": round(float(v), 2),
            }
            for p, v in monthly.items()
        ],
    }


def get_price_recommendation(crop, days_ahead=None):
    """
    Backward-compatible name used by app.py. It now returns the
    12-month outlook (`days_ahead` is ignored).
    """
    return get_yearly_price_outlook(crop)


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    crop = "Mango"

    df = load_price_data()

    print()
    print("=" * 70)
    print("DATA CHECK")
    print("=" * 70)
    print(f"Rows       : {len(df):,}")
    print(f"Date range : {df['Arrival_Date'].min().date()} -> {df['Arrival_Date'].max().date()}")

    result = get_latest_price(crop)

    print()
    print("=" * 70)
    print("KARNATAKA MARKET PRICE")
    print("=" * 70)

    if not result["available"]:

        print("Price unavailable:")
        print(result["message"])

    else:

        print(f"Crop          : {result['crop']}")
        print(f"Latest date   : {result['date']}")
        print(f"Minimum price : ₹{result['min_price']:.2f}")
        print(f"Maximum price : ₹{result['max_price']:.2f}")
        print(f"Average modal : ₹{result['modal_price']:.2f}")
        print(f"Market records: {result['market_count']}")

    print()
    print("=" * 70)
    print("12-MONTH PRICE OUTLOOK")
    print("=" * 70)

    train_info = train_price_model(crop)
    print(train_info)

    rec = get_yearly_price_outlook(crop)

    if not rec["available"]:
        print(rec["message"])
    else:
        print(f"Recent avg ({rec['current_date']}): ₹{rec['current_price']:.2f}")
        print(f"Peak month  ({rec['predicted_date']}): ₹{rec['predicted_price']:.2f}")
        print(f"Weakest     ({rec['worst_date']}): ₹{rec['worst_price']:.2f}")
        print(f"Trend       : {rec['trend']} ({rec['percent_change']:+.1f}%)")
        print(f"Model used  : {rec['model_used']} (typical monthly error ₹{rec['model_mae']:.2f})")
        print(f"Best market : {rec['best_market_now']}")
        print(f"Advice      : {rec['recommendation']}")
        print()
        for m in rec["monthly"]:
            print(f"  {m['month']}: ₹{m['price']:.2f}")

    print("=" * 70)