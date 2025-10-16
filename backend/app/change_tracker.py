# backend/app/change_tracker.py
"""
Detects significant changes between old and new predictions.
Logs and returns summary of changed zones.
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from .logging_setup import logger
from .db_helpers import insert_change  # uses existing helper

SIGNIFICANT_RISK_DELTA = 0.08   # 8% change threshold
SIGNIFICANT_ETA_DELTA = 20      # minutes
ALERT_CHANGE_WEIGHT = True


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def detect_and_log_changes(old_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare zone-wise predictions and log significant changes.
    """
    if not old_data or "predictions" not in old_data or not new_data:
        return {"changed": 0, "zones": []}

    changed_zones: List[str] = []
    old_map = {z["zone_id"]: z for z in old_data.get("predictions", [])}

    for zone in new_data.get("predictions", []):
        zid = zone.get("zone_id")
        if not zid or zid not in old_map:
            continue

        prev = old_map[zid]
        risk_old = float(prev.get("risk_score", 0))
        risk_new = float(zone.get("risk_score", 0))
        eta_old = float(prev.get("zone_eta_minutes", 0))
        eta_new = float(zone.get("zone_eta_minutes", 0))
        alert_old = prev.get("alert_level", "")
        alert_new = zone.get("alert_level", "")

        risk_delta = abs(risk_new - risk_old)
        eta_delta = abs(eta_new - eta_old)

        if (
            risk_delta >= SIGNIFICANT_RISK_DELTA
            or eta_delta >= SIGNIFICANT_ETA_DELTA
            or (ALERT_CHANGE_WEIGHT and alert_new != alert_old)
        ):
            changed_zones.append(zid)
            logger.info(
                f"⚠️ Zone {zid} changed | risk {risk_old:.2f}→{risk_new:.2f} | "
                f"ETA {eta_old:.0f}→{eta_new:.0f} | alert {alert_old}→{alert_new}"
            )

            insert_change(
                zone_id=zid,
                generated_at=new_data.get("generated_at", datetime.utcnow().isoformat() + "Z"),
                score=risk_new,
                eta=int(eta_new),
                alert_level=alert_new,
                reason=f"riskΔ={risk_delta:.2f}, etaΔ={eta_delta:.0f}, alertΔ={alert_old}->{alert_new}",
                raw_obj=zone,
            )

    logger.info(f"✅ Change tracking complete | {len(changed_zones)} zones updated.")
    return {"changed": len(changed_zones), "zones": changed_zones}
