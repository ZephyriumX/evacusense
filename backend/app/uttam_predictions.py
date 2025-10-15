# backend/app/uttam_predictions.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uttam_predictions")

app = FastAPI(title="Uttam — EvacuSense Predictions API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo; lock down in prod
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# project-root data folder (repo layout: repo_root/backend/app/uttam_predictions.py)
PREDICTIONS_PATH = Path(__file__).resolve().parents[2].joinpath(
    "data", "generated", "uttam_flood_predictions.json"
)
logger.info(f"Looking for predictions file at: {PREDICTIONS_PATH}")

def _is_cell_keyed_mapping(obj: dict) -> bool:
    """
    Heuristic: object is a mapping where keys look like 'cell_...' and values are dicts.
    """
    if not isinstance(obj, dict) or not obj:
        return False
    # check a few keys
    sample_keys = list(obj.keys())[:10]
    for k in sample_keys:
        if not isinstance(k, str):
            return False
        if not (k.startswith("cell_") or k.startswith("cell") or k.startswith("c")):
            # still allow numeric-looking or other keys, but prefer cell_* pattern
            # if none match 'cell' pattern, treat as not cell-keyed
            return False
    # ensure values are dict-like
    return all(isinstance(obj[k], dict) for k in sample_keys)

def _convert_cell_mapping_to_canonical(obj: dict) -> dict:
    """
    Convert mapping { "cell_1284": {nodes:..., population_total:...}, ... }
    into canonical {"project","owner","generated_at","predictions":[{zone_id:..., ...}, ...]}
    """
    preds = []
    for zone_id, zone_data in obj.items():
        # If zone_data already contains zone_id or name, keep them; otherwise add zone_id
        entry = {}
        if isinstance(zone_data, dict):
            entry.update(zone_data)  # copy fields: nodes, population_total, etc.
        else:
            # If value is not dict (unlikely), wrap it
            entry["value"] = zone_data

        # ensure zone_id field present
        if "zone_id" not in entry and "id" not in entry:
            entry["zone_id"] = zone_id

        # optionally normalize people_unable_to_evacuate -> unable_to_evacuate (keep original though)
        preds.append(entry)

    # try to use file modification time as generated_at, fallback to now
    try:
        mtime = datetime.utcfromtimestamp(PREDICTIONS_PATH.stat().st_mtime).isoformat() + "Z"
    except Exception:
        mtime = datetime.utcnow().isoformat() + "Z"

    canonical = {
        "project": "EvacuSense",
        "owner": "uttam",
        "generated_at": mtime,
        "predictions": preds
    }
    return canonical

@app.get("/uttam/predictions", response_class=JSONResponse)
def uttam_predictions():
    """
    Serve a canonical predictions JSON structure.
    If the underlying file is a cell-keyed mapping, convert it on-the-fly.
    """
    # allow overriding path via env var
    env_path = os.getenv("UTTAM_PREDICTIONS_PATH")
    global PREDICTIONS_PATH
    if env_path:
        PREDICTIONS_PATH = Path(env_path)

    if not PREDICTIONS_PATH.exists():
        logger.error(f"Predictions file not found at: {PREDICTIONS_PATH}")
        raise HTTPException(
            status_code=404,
            detail=(
                f"Predictions file not found at {PREDICTIONS_PATH}. "
                f"Run scripts/generate_predictions.py to create it."
            ),
        )

    try:
        with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in predictions file.")
        raise HTTPException(status_code=500, detail="Invalid JSON format in predictions file.")
    except Exception as e:
        logger.exception("Unexpected error reading predictions file.")
        raise HTTPException(status_code=500, detail=str(e))

    # If the file already matches the canonical format (has 'predictions' list), return as-is
    if isinstance(data, dict) and "predictions" in data and isinstance(data["predictions"], list):
        logger.info("Predictions file is already canonical (has 'predictions' list). Serving directly.")
        return JSONResponse(content=data)

    # If the content is a mapping of cell_* -> dict, convert it
    if isinstance(data, dict) and _is_cell_keyed_mapping(data):
        logger.info("Detected cell-keyed mapping; converting to canonical predictions list.")
        canonical = _convert_cell_mapping_to_canonical(data)
        return JSONResponse(content=canonical)

    # If content is a plain list, assume it's already a predictions list
    if isinstance(data, list):
        logger.info("Detected top-level list — wrapping into canonical structure.")
        canonical = {
            "project": "EvacuSense",
            "owner": "uttam",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "predictions": data
        }
        return JSONResponse(content=canonical)

    # Otherwise, we don't recognize the shape; return it anyway but warn
    logger.warning("Predictions file has an unexpected shape; returning raw content.")
    return JSONResponse(content=data)

@app.get("/")
def root():
    return {"message": "Uttam predictions API running!", "path_checked": str(PREDICTIONS_PATH)}
