# backend/app/db_helpers.py
from typing import Optional, cast, Dict, Any
from .db_models import init_db, SessionLocal, PredictionHistory
from datetime import datetime
import json
from .logging_setup import logger

def ensure_db():
    init_db()

def insert_change(
    zone_id: str,
    generated_at: str,
    score: float,
    eta: int,
    alert_level: str,
    reason: str,
    raw_obj: Optional[Dict[str, Any]] = None,
) -> bool:
    ensure_db()
    db = SessionLocal()
    try:
        record = PredictionHistory(
            zone_id=zone_id,
            generated_at=datetime.fromisoformat(generated_at.replace("Z", "")),
            saved_at=datetime.utcnow(),
            risk_score=score,
            eta_minutes=eta,
            alert_level=alert_level,
            reason=reason,
            raw_json=json.dumps(raw_obj or {}),
        )
        db.add(record)
        db.commit()
        logger.info(f"[db_helpers] Inserted record for zone_id={zone_id}, alert_level={alert_level}")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"[db_helpers] insert_change failed for {zone_id}: {e}", exc_info=True)
        return False
    finally:
        db.close()

def get_recent_changes(zone_id: Optional[str] = None, limit: int = 100):
    ensure_db()
    db = SessionLocal()
    try:
        q = db.query(PredictionHistory)
        if zone_id:
            q = q.filter(PredictionHistory.zone_id == zone_id)
        q = q.order_by(PredictionHistory.id.desc()).limit(limit)
        rows = q.all()
        results = []
        for r in rows:
            raw_json_value = None
            if getattr(r, "raw_json", None):
                try:
                    raw_json_value = json.loads(cast(str, r.raw_json))
                except Exception:
                    raw_json_value = r.raw_json
            results.append(
                {
                    "id": r.id,
                    "zone_id": r.zone_id,
                    "generated_at": r.generated_at.isoformat() + "Z",
                    "saved_at": r.saved_at.isoformat() + "Z",
                    "risk_score": r.risk_score,
                    "eta_minutes": r.eta_minutes,
                    "alert_level": r.alert_level,
                    "reason": r.reason,
                    "raw_json": raw_json_value,
                }
            )
        return results
    finally:
        db.close()
