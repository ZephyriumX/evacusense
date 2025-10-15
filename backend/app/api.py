import os
import json
import pandas as pd
from fastapi import APIRouter, HTTPException
from .schemas import PredictionsResponse, AlertPayload
from .model_utils import build_features_all_zones, OUTPUT_FEATURES_PATH, METADATA_PATH
from .predict_utils import predict_from_features_file

router = APIRouter()

_latest_predictions_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated", "predictions_latest.json")

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/predictions/latest", response_model=PredictionsResponse)
def get_latest_predictions():
    """
    Returns latest features file as a proxy to predictions.
    Note: at Phase 1/2 this returns feature snapshot; after training we also add preds.
    """
    if os.path.exists(_latest_predictions_path):
        with open(_latest_predictions_path, "r") as f:
            payload = json.load(f)
        return payload

    try:
        df = build_features_all_zones()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature building failed: {e}")

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

    try:
        os.makedirs(os.path.dirname(_latest_predictions_path), exist_ok=True)
        with open(_latest_predictions_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    return payload

@router.post("/alerts/send")
def send_alert(payload: AlertPayload):
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"alerts_{payload.alert_id}.json")
    with open(out_path, "w") as f:
        json.dump(payload.dict(), f, indent=2)
    print(f"[ALERT SAVED] {out_path}")
    return {"status": "ok", "alert_id": payload.alert_id}

# New endpoint: run model prediction using latest features snapshot
@router.get("/predict_flood")
def predict_flood():
    """
    Run the trained models on the current features snapshot and return predictions per zone.
    If the models are not yet trained, returns the feature snapshot with placeholder fields.
    """
    try:
        preds_df = predict_from_features_file()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # convert to list of dicts
    results = []
    for _, r in preds_df.iterrows():
        people_unable = r.get("people_unable_to_evacuate")
        # safe-cast: if None or NaN, fallback to 0
        try:
            # handle pandas NaN as well
            if people_unable is None or (isinstance(people_unable, float) and pd.isna(people_unable)):
                people_unable_val = 0
            else:
                people_unable_val = int(people_unable)
        except Exception:
            people_unable_val = 0

        predicted_eta = r.get("predicted_eta_minutes")
        flood_prob = r.get("flood_prob")

        results.append({
            "zone_id": r.get("zone_id"),
            "predicted_eta_minutes": float(predicted_eta) if predicted_eta is not None and not (isinstance(predicted_eta, float) and pd.isna(predicted_eta)) else None,
            "flood_prob": float(flood_prob) if flood_prob is not None and not (isinstance(flood_prob, float) and pd.isna(flood_prob)) else None,
            "centroid_lat": r.get("centroid_lat"),
            "centroid_lon": r.get("centroid_lon"),
            "people_unable_to_evacuate": people_unable_val
        })

    return {"timestamp": preds_df["timestamp"].iloc[0] if "timestamp" in preds_df.columns else None, "predictions": results}
