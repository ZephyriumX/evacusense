# backend/app/flood_compute.py
"""
Flood computation utilities — estimate:
1️⃣ Flood Start ETA (time until overflow)
2️⃣ Zone ETA (time until flood reaches zone)
3️⃣ Risk Score (0.0–1.0 severity metric)
"""

from typing import Optional
import math

# ---- Tunable Constants ----
DEFAULT_CRITICAL_RIVER_LEVEL_M = 5.0
EPS_RISE_M_PER_MIN = 1e-6
RAIN_TO_RISE_K = 0.0005  # m rise per mm rainfall per minute
MAX_ETA_MIN = 24 * 60    # 24 hours
BASE_FLOOD_SPEED_M_PER_MIN = 5.0
MIN_FLOOD_SPEED = 0.5
MAX_FLOOD_SPEED = 50.0


# ---- 1️⃣ Flood Start ETA ----
def estimate_flood_start_eta(
    current_level_m: float,
    critical_level_m: float = DEFAULT_CRITICAL_RIVER_LEVEL_M,
    rise_rate_m_per_min: Optional[float] = None,
    rainfall_mm_per_min: Optional[float] = None,
) -> int:
    """
    Estimate minutes until waterbody starts overflowing.
    Returns an integer number of minutes (0 if already overflowing).
    """
    # Already overflowing
    if current_level_m >= critical_level_m:
        return 0

    # Derive rise rate if missing (from rainfall)
    if (not rise_rate_m_per_min or rise_rate_m_per_min <= 0) and rainfall_mm_per_min:
        rise_rate_m_per_min = RAIN_TO_RISE_K * rainfall_mm_per_min

    # No valid rise rate
    if not rise_rate_m_per_min or rise_rate_m_per_min <= EPS_RISE_M_PER_MIN:
        return MAX_ETA_MIN

    # Calculate time to overflow
    minutes = (critical_level_m - current_level_m) / rise_rate_m_per_min
    return int(max(0, min(MAX_ETA_MIN, round(minutes))))


# ---- 2️⃣ Zone ETA ----
def estimate_zone_eta(
    distance_m: float,
    flood_start_eta_min: int,
    slope: float = 0.0,
    risk_norm: float = 0.0,
    drainage_capacity: float = 1.0,
) -> int:
    """
    Estimate minutes for flood to reach a specific zone.

    Parameters:
        distance_m: Distance from waterbody (in meters)
        flood_start_eta_min: ETA for flood to start (in minutes)
        slope: Slope towards zone (positive = downhill)
        risk_norm: Normalized 0–1 severity (river level ratio)
        drainage_capacity: 0–1 efficiency (1.0 = good drainage)
    """
    severity_factor = 0.5 + 1.0 * risk_norm
    slope_factor = 1.0 + 2.0 * slope
    friction = max(0.0, min(0.8, 1.0 - drainage_capacity))

    speed = BASE_FLOOD_SPEED_M_PER_MIN * severity_factor * max(0.2, slope_factor) * (1.0 - friction)
    speed = max(MIN_FLOOD_SPEED, min(MAX_FLOOD_SPEED, speed))

    if distance_m <= 0:
        return int(max(0, flood_start_eta_min))

    travel_min = distance_m / speed
    eta = int(round(flood_start_eta_min + travel_min))
    return max(0, min(MAX_ETA_MIN, eta))


# ---- 3️⃣ Risk Score ----
def compute_risk_score(
    zone_eta_min: int,
    depth_m: float = 0.0,
    distance_m: float = 2000.0,
    drainage_capacity: float = 1.0,
) -> float:
    """
    Simple explainable flood risk metric (0.0–1.0).

    Parameters:
        zone_eta_min: ETA for flood to reach zone
        depth_m: Expected flood depth (m)
        distance_m: Distance from waterbody (m)
        drainage_capacity: 0–1 drainage efficiency
    """
    MAX_WINDOW = 12 * 60  # 12 hours
    time_factor = max(0.0, 1.0 - (zone_eta_min / MAX_WINDOW))
    depth_factor = min(1.0, depth_m / 2.0)
    proximity = max(0.0, 1.0 - (distance_m / 2000.0))
    drainage_penalty = 1.0 - max(0.0, min(1.0, drainage_capacity))

    # Weighted risk components
    w1, w2, w3, w4 = 0.45, 0.25, 0.15, 0.15
    raw = w1 * time_factor + w2 * depth_factor + w3 * proximity + w4 * drainage_penalty
    return round(max(0.0, min(1.0, raw)), 3)
