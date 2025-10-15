# scripts/generate_predictions.py
"""
Simple generator to create data/generated/uttam_flood_predictions.json
Usage:
    python scripts/generate_predictions.py \
        --zones data/zones_def.json \
        --out data/generated/uttam_flood_predictions.json
"""
import json
import argparse
from datetime import datetime, timedelta
import uuid
import os

def load_zones(zones_path):
    with open(zones_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def make_prediction_for_zone(zone, base_time):
    # sample structure — adapt keys to exactly match your mock_uttam_sequence.json if needed
    return {
        "prediction_id": str(uuid.uuid4()),
        "zone_id": zone.get("zone_id") or zone.get("id") or zone.get("name"),
        "zone_name": zone.get("name", ""),
        "timestamp_utc": base_time.isoformat() + "Z",
        "eta_minutes": int(zone.get("eta_minutes", 30)),  # example numeric
        "risk_score": round(float(zone.get("risk_score", 0.3)), 3),
        "recommended_action": zone.get("recommended_action", "evacuate" if zone.get("risk_score", 0.3) > 0.5 else "monitor"),
        "metadata": {
            "source": "uttam_generator_v1",
            "model_files": ["eta_regressor.joblib", "eta_classifier.joblib"]
        }
    }

def main(args):
    zones = load_zones(args.zones)
    # ensure generated directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    base_time = datetime.utcnow()
    sequence = {
        "project": "EvacuSense",
        "owner": "uttam",
        "generated_at": base_time.isoformat() + "Z",
        "predictions": []
    }
    # if zones is a dict with key 'zones', use it
    if isinstance(zones, dict) and "zones" in zones:
        zone_list = zones["zones"]
    elif isinstance(zones, list):
        zone_list = zones
    else:
        # fallback: put the whole object as one zone
        zone_list = [zones]

    for i, zone in enumerate(zone_list):
        # offset timestamps slightly for sequence
        pred_time = base_time + timedelta(minutes=5 * i)
        p = make_prediction_for_zone(zone, pred_time)
        sequence["predictions"].append(p)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(sequence, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(sequence['predictions'])} predictions to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", required=True, help="path to zones_def.json")
    parser.add_argument("--out", default="data/generated/uttam_flood_predictions.json", help="output JSON file")
    args = parser.parse_args()
    main(args)
