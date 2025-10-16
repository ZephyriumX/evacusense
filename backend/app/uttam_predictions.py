# backend/app/uttam_predictions.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import logging
from datetime import datetime
import os
import sys
import sqlite3
import shlex
import asyncio
from typing import Optional, Any
import importlib.util
import inspect

from .logging_setup import logger
from .flood_compute import estimate_flood_start_eta, estimate_zone_eta, compute_risk_score
from .change_tracker import detect_and_log_changes, load_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uttam_predictions")

app = FastAPI(title="Uttam — EvacuSense Predictions API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo; restrict in prod
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ----- Paths & config -----
ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = Path(os.getenv("UTTAM_PREDICTIONS_PATH", ROOT / "data" / "generated" / "uttam_flood_predictions.json"))
SQLITE_HISTORY_PATH = ROOT / "data" / "uttam_history.sqlite3"
GENERATOR_SCRIPT = ROOT / "scripts" / "generate_predictions.py"
ZONES_PATH = ROOT / "data" / "zones_def.json"

REFRESH_MINUTES = int(os.getenv("UTTAM_REFRESH_MINUTES", "10"))
RUN_ON_START = os.getenv("UTTAM_RUN_ON_START", "true").lower() in ("1", "true", "yes")
ADMIN_TOKEN = os.getenv("UTTAM_ADMIN_TOKEN", "").strip()

logger.info(f"Predictions file: {PREDICTIONS_PATH}")
logger.info(f"Generator script: {GENERATOR_SCRIPT}")
logger.info(f"Refresh minutes: {REFRESH_MINUTES}, run_on_start: {RUN_ON_START}")

# Try to discover a persist helper module if user added one
persist_func = None
try:
    spec_ph = importlib.util.find_spec("backend.app.persist_helper")
    if spec_ph:
        import backend.app.persist_helper as persist_helper  # type: ignore
        if hasattr(persist_helper, "persist_significant_changes"):
            persist_func = getattr(persist_helper, "persist_significant_changes")
except Exception:
    persist_func = None


# ----- ENRICHMENT HELPER -----
def _enrich_predictions(data: dict) -> dict:
    """
    Attach flood_start_eta, zone_eta, and risk_score to each zone.
    """
    try:
        # Example baseline hydrologic context (replace with live values)
        current_level = 4.2      # current river level (m)
        critical_level = 5.0     # overflow threshold (m)
        rise_rate = 0.002        # m per min (~12 cm/hour)
        rainfall_rate = 0.3      # mm per min (~18 mm/hour)

        flood_start_eta = estimate_flood_start_eta(
            current_level_m=current_level,
            critical_level_m=critical_level,
            rise_rate_m_per_min=rise_rate,
            rainfall_mm_per_min=rainfall_rate,
        )

        for zone in data.get("predictions", []):
            distance = float(zone.get("distance_to_waterbody_m", 1000.0))
            slope = float(zone.get("slope", 0.0))
            drainage = float(zone.get("drainage_capacity", 1.0))
            depth = float(zone.get("flood_depth_estimate_m", 0.0))
            river_level = float(zone.get("river_level_m", 0.0))
            risk_norm = river_level / critical_level

            zone_eta = estimate_zone_eta(distance, flood_start_eta, slope, risk_norm, drainage)
            risk_score = compute_risk_score(zone_eta, depth, distance, drainage)

            if risk_score < 0.33:
                alert = "green"
            elif risk_score < 0.66:
                alert = "yellow"
            elif risk_score < 0.85:
                alert = "orange"
            else:
                alert = "red"

            zone.update({
                "flood_start_eta_minutes": flood_start_eta,
                "zone_eta_minutes": zone_eta,
                "risk_score": risk_score,
                "alert_level": alert,
            })

        data["flood_start_eta_minutes"] = flood_start_eta
        data["generated_at"] = datetime.utcnow().isoformat() + "Z"
        return data

    except Exception as e:
        logger.error(f"Error enriching predictions: {e}", exc_info=True)
        return data


# ----- GENERATOR -----
async def _run_generator_inprocess() -> tuple[bool, str]:
    """
    Call the generator script.
    """
    try:
        spec = importlib.util.spec_from_file_location("uttam_generator", str(GENERATOR_SCRIPT))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            fn = getattr(mod, "generate", None) or getattr(mod, "generate_predictions", None) or getattr(mod, "main", None)
            if fn:
                fn()
                return True, f"Generated via {GENERATOR_SCRIPT}"
    except Exception as e:
        logger.error(f"Generator error: {e}")
    return False, "generator_failed"


# ----- PERSISTENCE -----
async def _try_persist_changes():
    global persist_func
    if not persist_func:
        try:
            from backend.app.persist_helper import persist_significant_changes  # type: ignore
            persist_func = persist_significant_changes
        except Exception:
            persist_func = None
    if not persist_func:
        return {"ok": False, "msg": "no_persist_helper"}
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, persist_func, PREDICTIONS_PATH)
        return {"ok": True, "result": res}
    except Exception as e:
        logger.exception("persist_significant_changes failed")
        return {"ok": False, "msg": str(e)}


# ----- ENDPOINTS -----
@app.get("/uttam/predictions", response_class=JSONResponse)
def uttam_predictions():
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(status_code=404, detail="Predictions file not found.")
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = _enrich_predictions(data)
    return JSONResponse(content=data)


@app.get("/")
def root():
    return {"message": "Uttam predictions API running!", "path_checked": str(PREDICTIONS_PATH)}


@app.get("/uttam/zones/summary")
def zones_summary():
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(status_code=404, detail="Predictions not found")
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = _enrich_predictions(data)
    zones = data.get("predictions", [])
    summary = [
        {
            "zone_id": z.get("zone_id"),
            "risk_score": z.get("risk_score"),
            "alert_level": z.get("alert_level"),
            "flood_start_eta_minutes": z.get("flood_start_eta_minutes"),
            "zone_eta_minutes": z.get("zone_eta_minutes"),
        }
        for z in zones
    ]
    return {"count": len(summary), "results": summary}


@app.get("/uttam/waterbody/summary")
def waterbody_summary():
    current_level = 4.2
    critical_level = 5.0
    rise_rate = 0.002
    rainfall_rate = 0.3
    flood_start_eta = estimate_flood_start_eta(
        current_level_m=current_level,
        critical_level_m=critical_level,
        rise_rate_m_per_min=rise_rate,
        rainfall_mm_per_min=rainfall_rate,
    )
    return {
        "waterbody_id": "sutlej_main",
        "current_level_m": current_level,
        "critical_level_m": critical_level,
        "rise_rate_m_per_min": rise_rate,
        "rainfall_mm_per_min": rainfall_rate,
        "flood_start_eta_minutes": flood_start_eta,
    }


# ----- SCHEDULER -----
_scheduler_task: Optional[asyncio.Task] = None
_scheduler_stop = False


async def _scheduler_loop():
    if RUN_ON_START:
        logger.info("Scheduler initial run: generating predictions...")
        ok, msg = await _run_generator_inprocess()
        logger.info(f"Initial generator result: ok={ok}, msg={msg}")
        if ok:
            try:
                old_data = load_json(PREDICTIONS_PATH)
                with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
                    new_data = json.load(f)

                new_data = _enrich_predictions(new_data)
                safe_old_data = old_data if isinstance(old_data, dict) else {}
                summary = detect_and_log_changes(safe_old_data, new_data)


                with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
                    json.dump(new_data, f, indent=2)

                logger.info(
                    f"✅ Enriched + compared predictions | "
                    f"changed_zones={summary['changed']} | zones={summary['zones']}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to enrich/compare predictions: {e}")
            await _try_persist_changes()
    else:
        logger.info("RUN_ON_START disabled — skipping initial generation.")

    while not _scheduler_stop:
        logger.info(f"Scheduler sleeping for {REFRESH_MINUTES} minute(s)...")
        await asyncio.sleep(REFRESH_MINUTES * 60)
        if _scheduler_stop:
            break
        logger.info("Scheduler wakeup: generating predictions...")
        ok, msg = await _run_generator_inprocess()
        logger.info(f"Scheduled generator result: ok={ok}, msg={msg}")
        if ok:
            try:
                old_data = load_json(PREDICTIONS_PATH)
                with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
                    new_data = json.load(f)

                new_data = _enrich_predictions(new_data)
                safe_old_data = old_data if isinstance(old_data, dict) else {}
                summary = detect_and_log_changes(safe_old_data, new_data)


                with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
                    json.dump(new_data, f, indent=2)

                logger.info(
                    f"✅ Enriched + compared predictions (scheduled) | "
                    f"changed_zones={summary['changed']} | zones={summary['zones']}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to enrich/compare predictions: {e}")
            await _try_persist_changes()


@app.on_event("startup")
async def _startup():
    global _scheduler_task, _scheduler_stop
    _scheduler_stop = False
    loop = asyncio.get_event_loop()
    _scheduler_task = loop.create_task(_scheduler_loop())
    logger.info("Background scheduler started.")


@app.on_event("shutdown")
async def _shutdown():
    global _scheduler_stop, _scheduler_task
    logger.info("Shutting down scheduler...")
    _scheduler_stop = True
    if _scheduler_task:
        try:
            await _scheduler_task
        except Exception:
            pass
    logger.info("Scheduler stopped.")
