# backend/app/model_utils.py
import os
import json
from datetime import timedelta
import warnings

import numpy as np
import pandas as pd

# BASE paths (adjust if your repo layout differs)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ZONES_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "generated", "firozepur_zones_summary.csv")
SENSOR_TIMESERIES_PATH = os.path.join(BASE_DIR, "data", "raw", "sensor_hourly.csv")
OUTPUT_FEATURES_PATH = os.path.join(BASE_DIR, "data", "generated", "features_ready_for_model.csv")
METADATA_PATH = os.path.join(BASE_DIR, "models", "feature_metadata.json")

# Feature order expected by model (keep exact)
FEATURE_ORDER = [
    "timestamp","zone_id","centroid_lat","centroid_lon",
    "rainfall_1h","rainfall_3h","rainfall_6h","rainfall_24h",
    "river_level_m","delta_river_1h","delta_river_3h",
    "soil_moisture","humidity_percent","temperature_c","wind_speed_kmph",
    "elevation_m","drainage_capacity","distance_from_river_km",
    "population_total","people_unable_to_evacuate",
    "forecast_rain_1h","forecast_rain_3h"
]


# ---------- Utilities ----------
def _ensure_dir(path):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)

def _safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

# ---------- Loaders ----------
def load_zone_static(path=ZONES_SUMMARY_PATH):
    """
    Loads zones summary CSV. Returns DataFrame indexed by zone_id.
    Required columns: zone_id, centroid_lat, centroid_lon
    Optional: elevation_m, drainage_capacity, flood_history_score, population_total, people_unable_to_evacuate, distance_from_river_km
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Zone summary not found at {path}")
    df = pd.read_csv(path, dtype={"zone_id": str})
    if "zone_id" not in df.columns or "centroid_lat" not in df.columns or "centroid_lon" not in df.columns:
        raise ValueError("zones summary must include zone_id, centroid_lat, centroid_lon")
    df = df.set_index("zone_id")
    # ensure defaults for optional fields
    optional_defaults = {
        "elevation_m": 186.0,
        "drainage_capacity": 0.6,
        "flood_history_score": 0.0,
        "population_total": 1000,
        "people_unable_to_evacuate": 0,
        "distance_from_river_km": 5.0
    }
    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    return df

def load_sensor_timeseries(path=SENSOR_TIMESERIES_PATH, parse_dates=True):
    """
    Loads sensor timeseries CSV and returns DataFrame with tz-aware timestamps.
    Expects columns: timestamp, zone_id, rainfall_mm, river_level_m, soil_moisture, humidity_percent, temperature_c, wind_speed_kmph
    """
    if not os.path.exists(path):
        # return empty df with expected columns
        cols = ["timestamp","zone_id","rainfall_mm","river_level_m","soil_moisture","humidity_percent","temperature_c","wind_speed_kmph"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("sensor timeseries must include timestamp column")
    # parse timestamp; keep timezone if present
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # ensure zone_id is string
    if "zone_id" in df.columns:
        df["zone_id"] = df["zone_id"].astype(str)
    return df.sort_values("timestamp")

# ---------- Feature computations ----------
def _sum_rainfall(window_hours, now_ts, zrows):
    cutoff = now_ts - pd.Timedelta(hours=window_hours)
    if zrows.empty:
        return 0.0
    s = zrows[zrows["timestamp"] > cutoff]["rainfall_mm"].sum()
    return float(s)

def _last_value(col, zrows):
    if zrows.empty:
        return np.nan
    vals = zrows[col].dropna()
    if vals.empty:
        return np.nan
    return float(vals.iloc[-1])

def build_features_for_zone(zone_id, sensor_df, static_df, now_ts=None, forecast=None):
    """
    Build feature dict for a single zone for the latest time in sensor_df.
    forecast: optional dict with 'forecast_rain_1h','forecast_rain_3h' per zone_id
    """
    if now_ts is None:
        if sensor_df.empty:
            now_ts = pd.Timestamp.utcnow()
        else:
            now_ts = sensor_df["timestamp"].max()
    # select zone rows in last 24h
    zrows = sensor_df[sensor_df["zone_id"] == zone_id]
    # compute rolling sums and latests
    rainfall_1h = _sum_rainfall(1, now_ts, zrows)
    rainfall_3h = _sum_rainfall(3, now_ts, zrows)
    rainfall_6h = _sum_rainfall(6, now_ts, zrows)
    rainfall_24h = _sum_rainfall(24, now_ts, zrows)

    river_now = _last_value("river_level_m", zrows)
    river_1h = np.nan
    river_3h = np.nan
    if not zrows.empty:
        # find last sample before now - 1h and now - 3h
        t1 = now_ts - pd.Timedelta(hours=1)
        t3 = now_ts - pd.Timedelta(hours=3)
        r1 = zrows[zrows["timestamp"] <= t1]["river_level_m"]
        r3 = zrows[zrows["timestamp"] <= t3]["river_level_m"]
        if not r1.empty:
            river_1h = float(r1.iloc[-1])
        if not r3.empty:
            river_3h = float(r3.iloc[-1])
    delta_river_1h = (float(river_now) - float(river_1h)) if (not np.isnan(river_now) and not np.isnan(river_1h)) else 0.0
    delta_river_3h = (float(river_now) - float(river_3h)) if (not np.isnan(river_now) and not np.isnan(river_3h)) else 0.0

    soil_moisture = _last_value("soil_moisture", zrows)
    humidity_percent = _last_value("humidity_percent", zrows)
    temperature_c = _last_value("temperature_c", zrows)
    wind_speed_kmph = _last_value("wind_speed_kmph", zrows)

    # static fields
    if zone_id in static_df.index:
        s = static_df.loc[zone_id]
        centroid_lat = float(s["centroid_lat"])
        centroid_lon = float(s["centroid_lon"])
        elevation_m = float(s["elevation_m"])
        drainage_capacity = float(s["drainage_capacity"])
        distance_from_river_km = float(s["distance_from_river_km"])
        population_total = int(s["population_total"])
        people_unable_to_evacuate = int(s["people_unable_to_evacuate"])
    else:
        # Defaults if zone not found
        centroid_lat = np.nan
        centroid_lon = np.nan
        elevation_m = np.nan
        drainage_capacity = np.nan
        distance_from_river_km = np.nan
        population_total = 0
        people_unable_to_evacuate = 0

    # forecast fields: forecast is a dict keyed by zone_id, optional
    forecast_r1 = None
    forecast_r3 = None
    if isinstance(forecast, dict) and zone_id in forecast:
        fc = forecast[zone_id]
        forecast_r1 = fc.get("forecast_rain_1h", None)
        forecast_r3 = fc.get("forecast_rain_3h", None)

    features = {
        "timestamp": now_ts.isoformat(),
        "zone_id": zone_id,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "rainfall_1h": rainfall_1h,
        "rainfall_3h": rainfall_3h,
        "rainfall_6h": rainfall_6h,
        "rainfall_24h": rainfall_24h,
        "river_level_m": float(river_now) if not np.isnan(river_now) else np.nan,
        "delta_river_1h": delta_river_1h,
        "delta_river_3h": delta_river_3h,
        "soil_moisture": float(soil_moisture) if not np.isnan(soil_moisture) else np.nan,
        "humidity_percent": float(humidity_percent) if not np.isnan(humidity_percent) else np.nan,
        "temperature_c": float(temperature_c) if not np.isnan(temperature_c) else np.nan,
        "wind_speed_kmph": float(wind_speed_kmph) if not np.isnan(wind_speed_kmph) else np.nan,
        "elevation_m": elevation_m,
        "drainage_capacity": drainage_capacity,
        "distance_from_river_km": distance_from_river_km,
        "population_total": population_total,
        "people_unable_to_evacuate": people_unable_to_evacuate,
        "forecast_rain_1h": forecast_r1,
        "forecast_rain_3h": forecast_r3
    }
    return features

def build_features_all_zones(sensor_path=SENSOR_TIMESERIES_PATH, zones_path=ZONES_SUMMARY_PATH,
                              output_path=OUTPUT_FEATURES_PATH, metadata_path=METADATA_PATH, forecast=None):
    """
    Master function: loads sensor + zones CSVs, builds features for all zones at the latest timestamp,
    imputes medians, saves features CSV and metadata (medians + feature order).
    """
    # load
    static_df = load_zone_static(zones_path)
    sensor_df = load_sensor_timeseries(sensor_path)

    # determine 'now' for snapshot; if sensor empty, use current UTC
    if sensor_df.empty:
        now_ts = pd.Timestamp.utcnow()
    else:
        now_ts = sensor_df["timestamp"].max()

    rows = []
    for zone_id in static_df.index:
        feat = build_features_for_zone(zone_id, sensor_df, static_df, now_ts=now_ts, forecast=forecast)
        rows.append(feat)

    feat_df = pd.DataFrame(rows)
    # Reorder columns to match FEATURE_ORDER (add missing columns)
    for col in FEATURE_ORDER:
        if col not in feat_df.columns:
            feat_df[col] = np.nan
    feat_df = feat_df[FEATURE_ORDER]

    # compute medians for numeric columns to store metadata
    numeric_cols = [c for c in FEATURE_ORDER if c not in ("timestamp","zone_id","centroid_lat","centroid_lon")]
    medians = {}
    for col in numeric_cols:
        try:
            med = float(feat_df[col].median(skipna=True))
            if np.isnan(med):
                med = 0.0
        except Exception:
            med = 0.0
        medians[col] = med

    # impute medians
    for col, med in medians.items():
        feat_df[col] = feat_df[col].fillna(med)

    # Save metadata
    metadata = {
        "feature_order": FEATURE_ORDER,
        "medians": medians,
        "created_at": pd.Timestamp.utcnow().isoformat()
    }
    _ensure_dir(metadata_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save features CSV
    _ensure_dir(output_path)
    feat_df.to_csv(output_path, index=False)
    return feat_df

# ---------- Small validation helper ----------
def validate_features_csv(path=OUTPUT_FEATURES_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    # basic sanity
    if df.isnull().sum().sum() > 0:
        warnings.warn("Features CSV contains NaNs after imputation.")
    return True

# ---------- End of module ----------
if __name__ == "__main__":
    # quick run if executed directly
    print("Building features for all zones...")
    df = build_features_all_zones()
    print("Saved features to:", OUTPUT_FEATURES_PATH)
    print(df.head().to_string(index=False))
