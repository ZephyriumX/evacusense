import os
import json
from fastapi import APIRouter, HTTPException
from .schemas import PredictionsResponse, AlertPayload
from .model_utils import build_features_all_zones, OUTPUT_FEATURES_PATH, METADATA_PATH

router = APIRouter()

# Keep a very small in-memory cache to serve quick responses
_latest_predictions_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated", "predictions_latest.json")

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/predictions/latest", response_model=PredictionsResponse)
def get_latest_predictions():
    """
    Returns latest features file as a proxy to predictions.
    Note: at this Phase 1 step we don't have the ETA model yet in place.
    This endpoint returns the built features CSV summarized per zone (as fields).
    """
    # If predictions JSON exists return that, otherwise build features and return features as placeholders
    if os.path.exists(_latest_predictions_path):
        with open(_latest_predictions_path, "r") as f:
            payload = json.load(f)
        return payload

    # build features CSV (this will also write metadata)
    try:
        df = build_features_all_zones()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature building failed: {e}")

    # convert to response shape using minimal placeholders for prediction fields
    zones = []
    for _, row in df.iterrows():
        zones.append({
            "zone_id": row["zone_id"],
            "p_flood_3h": 0.0,
            "eta_minutes": 9999.0,
            "alert_level": None,
            "confirmation_score": 0.0,
            "risk_score": 0.0,
            "rainfall_last_3h": row["rainfall_3h"],
            "river_level_m": row["river_level_m"],
            "soil_moisture": row["soil_moisture"],
            "people_unable_to_evacuate": int(row["people_unable_to_evacuate"]),
            "population_total": int(row["population_total"]),
            "elevation_m": row["elevation_m"],
            "centroid_lat": row["centroid_lat"],
            "centroid_lon": row["centroid_lon"]
        })
    payload = {"timestamp": df["timestamp"].iloc[0], "model_version": None, "zones": zones}

    # cache a JSON for subsequent GETs
    try:
        os.makedirs(os.path.dirname(_latest_predictions_path), exist_ok=True)
        with open(_latest_predictions_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    return payload

@router.post("/alerts/send")
def send_alert(payload: AlertPayload):
    """
    Endpoint to record/send an alert. In production you'd forward this to a notification service.
    For now we save to data/generated/alerts_<id>.json and print a log.
    """
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"alerts_{payload.alert_id}.json")
    with open(out_path, "w") as f:
        json.dump(payload.dict(), f, indent=2)
    print(f"[ALERT SAVED] {out_path}")
    return {"status": "ok", "alert_id": payload.alert_id}
