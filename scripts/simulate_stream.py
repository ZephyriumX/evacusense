# scripts/simulate_stream.py
import json, random, time
from datetime import datetime
from pathlib import Path

# List of zones in your system
ZONES = ["cell_1284", "cell_1205", "cell_1348", "cell_1189", "cell_1188", "cell_1187", "cell_3680"]

LIVE_PATH = Path("data/live/live_feed.json")
LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

def generate_live_data():
    data = {}
    for z in ZONES:
        data[z] = {
            "rain_mm_10m": round(random.uniform(0, 30), 2),        # rainfall intensity
            "river_level": round(random.uniform(2.5, 6.5), 2),     # river height (m)
            "cumulative_rain_6h": round(random.uniform(0, 150), 1),
            "cumulative_rain_24h": round(random.uniform(0, 250), 1),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    return data

def main():
    print(f"[Simulator] 🚀 Writing live data to {LIVE_PATH} every 10 s (1 s = 10 min sim)")
    while True:
        data = generate_live_data()
        with open(LIVE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Updated {len(ZONES)} zones.")
        time.sleep(10)   # 10 seconds = 10 minutes simulated

if __name__ == "__main__":
    main()
