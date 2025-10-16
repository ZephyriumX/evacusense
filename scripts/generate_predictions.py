# scripts/generate_predictions.py
import json, random
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_predictions")

# paths
LIVE_FEED = Path("data/live/live_feed.json")
ZONES_DEF = Path("data/zones_def.json")
OUT_PATH = Path("data/generated/uttam_flood_predictions.json")

def map_to_alert_level(prob, eta):
    """Convert probability + ETA to alert color level."""
    if prob >= 0.7:
        level = 3  # 🔴 Evacuate
    elif prob >= 0.4:
        level = 2  # 🟠 Prepare
    elif prob >= 0.2:
        level = 1  # 🟡 Monitor
    else:
        level = 0  # 🟢 Safe
    # bump risk if flood is imminent
    if eta <= 30 and prob >= 0.4:
        level = min(3, level + 1)
    if eta <= 10 and prob >= 0.6:
        level = 3
    return level

def rule_based_eta_risk(rain, river, cum24, dist2river=1000):
    """Simple rule model producing risk_score + ETA (min)."""
    rn24 = min(1.0, cum24 / 250.0)
    rriver = min(1.0, river / 7.0)
    pdist = max(0.0, 1.0 - dist2river / 5000.0)
    risk = 0.5 * rn24 + 0.3 * rriver + 0.2 * pdist
    eta = int(max(5, (dist2river / 80.0) - rain * 0.5))
    return round(risk, 3), eta

def main():
    if not ZONES_DEF.exists():
        logger.error(f"Missing {ZONES_DEF}")
        return
    zones = json.load(open(ZONES_DEF))
    try:
        live = json.load(open(LIVE_FEED))
        logger.info(f"✅ Loaded live feed with {len(live)} zones")
    except Exception:
        live = {}
        logger.warning("⚠️  No live feed found, using random fallback values")

    preds = []
    for zid, zdata in zones.items():
        live_z = live.get(zid, {})
        rain = live_z.get("rain_mm_10m", random.uniform(0, 10))
        river = live_z.get("river_level", random.uniform(2.5, 5.5))
        cum24 = live_z.get("cumulative_rain_24h", random.uniform(0, 200))

        risk, eta = rule_based_eta_risk(rain, river, cum24)
        level = map_to_alert_level(risk, eta)

        preds.append({
            "zone_id": zid,
            "risk_score": risk,
            "eta_minutes": eta,
            "alert_level": level,
            "rain_mm_10m": rain,
            "river_level": river,
            "cumulative_rain_24h": cum24,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "recommended_action": ["safe", "monitor", "prepare", "evacuate"][level]
        })

    out = {
        "project": "EvacuSense",
        "owner": "uttam",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "predictions": preds
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    logger.info(f"✅ Generated {len(preds)} predictions → {OUT_PATH}")

if __name__ == "__main__":
    main()
