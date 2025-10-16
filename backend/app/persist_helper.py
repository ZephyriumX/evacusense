# backend/app/persist_helper.py
import json
from pathlib import Path
from datetime import datetime
from .logging_setup import logger
from .change_detector import significant_change_and_save

def persist_significant_changes(predictions_path: Path):
    """
    Reads predictions JSON and persists only meaningful changes.
    """
    try:
        logger.info(f"[persist_helper] Reading predictions from {predictions_path}")
        with open(predictions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        preds = data.get("predictions", [])
        if not preds:
            logger.warning("No predictions found to persist.")
            return {"ok": False, "msg": "no_predictions"}

        total, inserted, skipped = len(preds), 0, 0
        generated_at = data.get("generated_at") or datetime.utcnow().isoformat() + "Z"

        for p in preds:
            zone_id = str(p.get("zone_id", "unknown"))
            score = float(p.get("risk_score", 0))
            eta = int(p.get("eta_minutes", 0))
            alert_level = str(p.get("alert_level", "unknown"))

            ok, msg = significant_change_and_save(zone_id, generated_at, score, eta, alert_level, p)
            if ok:
                inserted += 1
            else:
                skipped += 1

        logger.info(f"[persist_helper] Done → Inserted={inserted}, Skipped={skipped}, Total={total}")
        return {"ok": True, "inserted": inserted, "skipped": skipped, "total": total}

    except Exception as e:
        logger.error(f"[persist_helper] Failed: {e}", exc_info=True)
        return {"ok": False, "msg": str(e)}
