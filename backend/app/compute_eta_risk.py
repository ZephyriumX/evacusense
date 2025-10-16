# backend/app/compute_eta_risk.py
"""
Compute ETA (Estimated Time of Arrival) for flood events and zone-wise risk score.
Used by uttam_predictions.py to enrich prediction data automatically.
"""

import math
from datetime import datetime
from typing import Optional, Dict, Any

# ---- Tunable constants ----
DEFAULT_CRITICAL_RIVER_LEVEL_M = 5.0
EPS_RISE_M_PER_MIN = 1e-6
RAIN_TO_RISE_K = 0.0005   # m rise per mm rainfall per minute
MAX_ETA_MIN = 24 * 60     # 24 hours
BASE_FLOOD_SPEED_M_PER_MIN = 5.0
MIN_FLOOD_SPEED = 0.5
MAX_FLOOD_SPEED = 50.0


# --- 1. Estimate when the waterbody will overflow ---
def estimate_flood_start_eta(
    current_level_m: float,
    critical_level_m: Optional[float] = None,
    rise_rate_m_per_min: Optional[float] = None,
    rainfall_mm_per_min: Optional[float] = None
) -> int:
    """
    Estimate minutes until flood starts (waterbody overflow).
    Returns integer minutes (0 if already overflowing).
    """
    critical_level_m = float(critical_level_m or DEFAULT_CRITICAL_RIVER_LEVEL_M)
    cur = float(current_level_m or 0.0)

    if cur >= critical_level_m:
        return 0  # already overflowing

    # derive rise rate if missing (using rainfall)
    if (not rise_rate_m_per_min or rise_rate_m_per_min <= 0) and rainfall_mm_per_min:
        rise_rate_m_per_min = RAIN_TO_RISE_K * rainfall_mm_per_min

    if not rise_rate_m_per_min or rise_rate_m_per_min <= EPS_RISE_M_PER_MIN:
        return MAX_ETA_MIN  # cannot determine, assume far away

    mins = (critical_level_m - cur) / rise_rate_m_per_min
    mins = max(0, min(MAX_ETA_MIN, int(round(mins))))
    return mins


# --- 2. Estimate ETA for flood to reach a particular zone ---
def estimate_zone_eta(
    distance_m: float,
    flood_start_eta_min: int,
    slope: float = 0.0,
    risk_norm: float = 0.0,
    drainage_capacity: float = 1.0
) -> int:
    """
    Estimate minutes until floodwater reaches the zone.
    """
    severity_factor = 0.5 + 1.0 * risk_norm     # speed multiplier from risk
    slope_factor = 1.0 + 2.0 * slope            # downhill speeds up propagation
    friction = max(0.0, min(0.8, 1.0 - drainage_capacity))  # bad drainage slows down

    speed = BASE_FLOOD_SPEED_M_PER_MIN * severity_factor * max(0.2, slope_factor) * (1.0 - friction)
    speed = max(MIN_FLOOD_SPEED, min(MAX_FLOOD_SPEED, speed))

    if distance_m <= 0:
        return int(max(0, flood_start_eta_min))

    travel_min = distance_m / speed
    eta = int(round(flood_start_eta_min + travel_min))
    eta = max(0, min(MAX_ETA_MIN, eta))
    return eta


# --- 3. Compute risk score (0–1 scale) ---
def compute_risk_score_from_eta(
    eta_min: int,
    depth_m: float = 0.0,
    distance_m: float = 2000.0,
    drainage_capacity: float = 1.0
) -> float:
    """
    Compute normalized flood risk score based on ETA, depth, distance, and drainage.
    """
    MAX_WINDOW = 12 * 60  # 12 hours reference
    time_factor = max(0.0, 1.0 - (eta_min / MAX_WINDOW))
    depth_factor = min(1.0, depth_m / 2.0)
    proximity = max(0.0, 1.0 - (distance_m / 2000.0))
    drainage_penalty = 1.0 - max(0.0, min(1.0, drainage_capacity))

    # Weighted sum
    w1, w2, w3, w4 = 0.45, 0.25, 0.15, 0.15
    raw = w1*time_factor + w2*depth_factor + w3*proximity + w4*drainage_penalty
    return round(max(0.0, min(1.0, raw)), 3)


# --- 4. Enrich a single zone record ---
def enrich_zone(zone: Dict[str, Any], flood_start_eta_min: int) -> Dict[str, Any]:
    """
    Enrich a zone dict with ETA and risk fields.
    Returns updated zone dict.
    """
    distance = float(zone.get("distance_to_waterbody_m", 1000.0))
    slope = float(zone.get("slope", 0.0))
    drainage = float(zone.get("drainage_capacity", 1.0))
    depth = float(zone.get("flood_depth_estimate_m", 0.0))
    river_level = float(zone.get("river_level_m", 0.0))

    risk_norm = river_level / DEFAULT_CRITICAL_RIVER_LEVEL_M
    risk_norm = max(0.0, min(1.0, risk_norm))

    zone_eta = estimate_zone_eta(distance, flood_start_eta_min, slope, risk_norm, drainage)
    risk_score = compute_risk_score_from_eta(zone_eta, depth, distance, drainage)

    if risk_score < 0.33:
        alert = "green"
    elif risk_score < 0.66:
        alert = "yellow"
    elif risk_score < 0.85:
        alert = "orange"
    else:
        alert = "red"

    zone["flood_start_eta_minutes"] = flood_start_eta_min
    zone["zone_eta_minutes"] = zone_eta
    zone["risk_score"] = risk_score
    zone["alert_level"] = alert
    return zone


# --- 5. Enrich entire dataset ---
def enrich_all_zones(
    predictions: Dict[str, Any],
    current_level_m: float,
    critical_level_m: float,
    rise_rate_m_per_min: float,
    rainfall_mm_per_min: float
) -> Dict[str, Any]:
    """
    predictions: dict with key 'predictions' = list[dict(zone data)]
    Adds flood_start_eta, zone_eta, and risk_score to all.
    Returns enriched dict.
    """
    flood_start_eta = estimate_flood_start_eta(
        current_level_m=current_level_m,
        critical_level_m=critical_level_m,
        rise_rate_m_per_min=rise_rate_m_per_min,
        rainfall_mm_per_min=rainfall_mm_per_min
    )

    enriched = []
    for zone in predictions.get("predictions", []):
        enriched.append(enrich_zone(zone, flood_start_eta))

    predictions["flood_start_eta_minutes"] = flood_start_eta
    predictions["predictions"] = enriched
    predictions["generated_at"] = datetime.utcnow().isoformat() + "Z"
    return predictions
