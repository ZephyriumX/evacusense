# backend/app/change_detector.py
from typing import Tuple
from .db_helpers import insert_change, get_recent_changes
from .logging_setup import logger

ABS_THRESHOLD = 0.05
REL_THRESHOLD = 0.15

def significant_change_and_save(zone_id: str, generated_at: str, score: float, eta: int, alert_level: str, raw_obj: dict) -> Tuple[bool, str]:
    """
    Determines if a new prediction for a zone is significantly different from the last one.
    Saves only if change is meaningful.
    """
    try:
        recent = get_recent_changes(zone_id, limit=1)
        if not recent:
            insert_change(zone_id, generated_at, score, eta, alert_level, "initial_record", raw_obj)
            return True, "initial_record"

        prev = recent[0]
        old_score = float(prev.get("risk_score", 0))
        abs_diff = abs(score - old_score)
        rel_diff = abs_diff / (abs(old_score) + 1e-9)

        if abs_diff >= ABS_THRESHOLD:
            reason = f"abs_change_{abs_diff:.4f}"
        elif rel_diff >= REL_THRESHOLD:
            reason = f"rel_change_{rel_diff:.4f}"
        elif str(prev.get("alert_level")) != str(alert_level):
            reason = f"alert_level_change_{prev.get('alert_level')}_to_{alert_level}"
        else:
            return False, "no_significant_change"

        insert_change(zone_id, generated_at, score, eta, alert_level, reason, raw_obj)
        logger.info(f"[change_detector] Significant change for {zone_id}: {reason}")
        return True, reason

    except Exception as e:
        logger.error(f"[change_detector] Error in detection: {e}", exc_info=True)
        return False, "error"
