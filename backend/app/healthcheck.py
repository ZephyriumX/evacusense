# backend/app/healthcheck.py
from fastapi import APIRouter
from pathlib import Path
from datetime import datetime, timedelta
import os
from .logging_setup import logger

router = APIRouter()

# Paths (use same env var as uttam_predictions if set)
PREDICTIONS_PATH = Path(os.getenv("UTTAM_PREDICTIONS_PATH", "data/generated/uttam_flood_predictions.json"))
DB_PATH = Path("data/uttam_history.sqlite3")

# Internal health state (these are updated by uttam_predictions)
health_state = {
    "scheduler_alive": False,
    "last_generated": None,   # ISO string or None
    "last_persisted": None,   # ISO string or None
}

# thresholds (in seconds) to judge "freshness"
GENERATOR_FRESH_SEC = int(os.getenv("UTTAM_HEALTH_GEN_FRESH_SEC", "900"))   # 15 min default
PERSIST_FRESH_SEC = int(os.getenv("UTTAM_HEALTH_PERSIST_FRESH_SEC", "900"))  # 15 min default

def _iso_to_dt(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        # strip trailing Z if present
        if iso.endswith("Z"):
            iso = iso[:-1]
        return datetime.fromisoformat(iso)
    except Exception:
        try:
            return datetime.utcfromtimestamp(float(iso))
        except Exception:
            return None

def _friendly_status():
    """
    Compute friendly status string and details based on timestamps and scheduler flag.
    Returns tuple (status_str, details_dict).
    """
    now = datetime.utcnow()
    scheduler_alive = bool(health_state.get("scheduler_alive"))
    last_gen_iso = health_state.get("last_generated")
    last_persist_iso = health_state.get("last_persisted")

    last_gen_dt = _iso_to_dt(last_gen_iso)
    last_persist_dt = _iso_to_dt(last_persist_iso)

    gen_age = (now - last_gen_dt).total_seconds() if last_gen_dt else None
    persist_age = (now - last_persist_dt).total_seconds() if last_persist_dt else None

    ok_gen = (gen_age is not None and gen_age <= GENERATOR_FRESH_SEC)
    ok_persist = (persist_age is not None and persist_age <= PERSIST_FRESH_SEC)

    if scheduler_alive and ok_gen and ok_persist:
        return "🟢 Healthy", {"scheduler_alive": True, "generator_age_sec": gen_age, "persist_age_sec": persist_age}
    if scheduler_alive and (ok_gen or ok_persist):
        return "🟡 Degraded", {"scheduler_alive": scheduler_alive, "generator_age_sec": gen_age, "persist_age_sec": persist_age}
    return "🔴 Inactive", {"scheduler_alive": scheduler_alive, "generator_age_sec": gen_age, "persist_age_sec": persist_age}

@router.get("/uttam/health")
def health_check():
    """
    Returns live backend status for dashboard/monitoring.
    """
    preds_exists = PREDICTIONS_PATH.exists()
    db_exists = DB_PATH.exists()

    # file mtime as fallback for last_generated if runtime state not set
    last_gen_time = None
    if preds_exists:
        try:
            ts = PREDICTIONS_PATH.stat().st_mtime
            last_gen_time = datetime.utcfromtimestamp(ts).isoformat() + "Z"
        except Exception:
            pass

    # Decide which timestamps to show: prefer runtime health_state values
    last_generated = health_state.get("last_generated") or last_gen_time
    last_persisted = health_state.get("last_persisted")

    status_str, status_details = _friendly_status()

    return {
        "status": status_str,
        "status_details": status_details,
        "scheduler_alive": health_state["scheduler_alive"],
        "predictions_file_exists": preds_exists,
        "db_file_exists": db_exists,
        "last_generated": last_generated,
        "last_persisted": last_persisted,
        "generated_path": str(PREDICTIONS_PATH),
        "db_path": str(DB_PATH),
    }

def update_health(event: str):
    """
    Called by uttam_predictions.py whenever generator/persistence runs.
    Events: "scheduler_start", "generator_run", "persist_run", "scheduler_stop".
    """
    now = datetime.utcnow().isoformat() + "Z"
    if event == "scheduler_start":
        health_state["scheduler_alive"] = True
    elif event == "generator_run":
        health_state["last_generated"] = now
    elif event == "persist_run":
        health_state["last_persisted"] = now
    elif event == "scheduler_stop":
        health_state["scheduler_alive"] = False
    logger.info(f"[healthcheck] update: {event} -> {now}")
