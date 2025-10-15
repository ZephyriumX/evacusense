"""
scripts/run_simulations.py

Hydrologic cellular-automata (CA) flood simulator that produces labels:
  - data/generated/sim_labels.csv   (one row per scenario_id,zone_id)
  - data/generated/scenario_index.json
  - data/generated/sim_timeseries/scenario_{id}.csv  (per-scenario per-zone timeseries)

Usage:
    # run default (500 scenarios)
    python -m scripts.run_simulations

    # run fewer scenarios for quick debugging
    python -m scripts.run_simulations --scenarios 10 --dt 10 --horizon 6

Notes:
- Expects `data/generated/firozepur_zones_summary.csv` to exist (zones summary).
- Outputs to data/generated/
"""

import os
import json
import math
import argparse
import random
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------- Defaults (changeable via CLI) ----------
DEFAULT_DT_MINUTES = 10
DEFAULT_HORIZON_HOURS = 6
DEFAULT_SCENARIOS = 500
DEFAULT_RAIN_RANGE = (10.0, 120.0)     # mm per hour
DEFAULT_DURATION_RANGE = (1.0, 5.0)    # hours
DEFAULT_RIVER_LEVEL_RANGE = (183.0, 187.0)  # meters
DEFAULT_FLOOD_THRESHOLD_MM = 50.0      # standing water mm threshold for inundation
DEFAULT_INFIL_BASE_MM_PER_HR = 5.0

# ---------- Paths ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ZONES_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "generated", "firozepur_zones_summary.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "generated")
TIMESERIES_DIR = os.path.join(OUT_DIR, "sim_timeseries")
SCENARIO_INDEX_PATH = os.path.join(OUT_DIR, "scenario_index.json")
LABELS_PATH = os.path.join(OUT_DIR, "sim_labels.csv")

# ---------- Utilities ----------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

# ---------- Neighbor mapping ----------
def build_downstream_map(zones_df):
    """
    Build a single-neighbor downstream map for each zone.
    For each zone i, find the neighbor j with lower elevation and minimal distance.
    If no neighbor has lower elevation, choose the nearest neighbor (to allow flow).
    Returns dict: {zone_id: downstream_zone_id}
    """
    coords = zones_df[["centroid_lat", "centroid_lon"]].values
    elevations = zones_df["elevation_m"].values
    zone_ids = list(zones_df.index.astype(str))

    # Precompute distances matrix (Euclidean approx)
    n = len(zone_ids)
    coords_rad = np.radians(coords)  # small improvement or keep planar
    dists = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                dists[i, j] = np.inf
            else:
                # planar approx (lat/lon small area)
                dy = (coords[i][0] - coords[j][0]) * 111000.0  # degrees lat -> meters
                dx = (coords[i][1] - coords[j][1]) * 111000.0 * math.cos(math.radians(coords[i][0]))
                dists[i, j] = math.hypot(dx, dy)

    downstream = {}
    for i in range(n):
        cand_idxs = [j for j in range(n) if j != i and elevations[j] <= elevations[i]]
        if not cand_idxs:
            # no lower neighbor: pick nearest neighbor
            j = int(np.argmin(dists[i]))
        else:
            # pick lowest elevation; if many, choose the nearest among them
            min_elev = min(elevations[j] for j in cand_idxs)
            cands = [j for j in cand_idxs if elevations[j] == min_elev]
            if len(cands) == 1:
                j = cands[0]
            else:
                # choose nearest among cands
                j = min(cands, key=lambda x: dists[i, x])
        downstream[zone_ids[i]] = zone_ids[j]
    return downstream, dists

# ---------- Simulation core ----------
def run_single_scenario(scenario_id, zones_df, downstream_map, dists,
                        dt_minutes, horizon_hours,
                        rain_profile, duration_hours,
                        initial_river_level, initial_soil_saturation,
                        flood_threshold_mm, infiltration_base_mm_per_hr,
                        debug=False):
    """
    Runs a single scenario.
    Returns:
      - labels: list of dicts per zone {scenario_id, zone_id, time_to_inundation_min or NaN, flooded_flag}
      - timeseries_df: pandas DataFrame with columns: timestep, zone_id, water_mm, river_level (replicated per zone)
    """

    zone_ids = list(zones_df.index.astype(str))
    n = len(zone_ids)
    steps = int((horizon_hours * 60) / dt_minutes)
    dt_hr = dt_minutes / 60.0  # hours

    # Initialize state arrays (units: mm water depth on cell surface)
    W = np.zeros(n, dtype=float)  # standing water depth (mm)
    # initial soil saturation per zone (0-1)
    if isinstance(initial_soil_saturation, dict):
        S = np.array([initial_soil_saturation.get(zid, 0.2) for zid in zone_ids], dtype=float)
    else:
        S = np.full(n, initial_soil_saturation, dtype=float)

    # drainage capacity (0-1) from zones_df
    drainage = zones_df["drainage_capacity"].values.astype(float)
    elevation = zones_df["elevation_m"].values.astype(float)
    area_factor = 1.0  # placeholder if needed (we use mm units directly)

    # set initial river level (global) and compute river overflow injection mapping
    river_level = initial_river_level
    # bank threshold: zones within distance_from_river_km < 1.5 are susceptible (simple mapping)
    dist_from_river = zones_df["distance_from_river_km"].values.astype(float)
    adjacent_to_river = dist_from_river <= 1.5

    # container for timeseries records
    recs = []
    # keep track of inundation times
    inundation_time = {z: None for z in zone_ids}

    # precompute neighbor slopes factor (normalized)
    slope_factor = np.zeros(n, dtype=float)
    for i, zid in enumerate(zone_ids):
        j = zone_ids.index(downstream_map[zid])
        # slope proxy = max(0, elevation_i - elevation_j) / (distance + small_eps)
        dist_m = dists[i, j] if dists[i, j] > 0 else 1.0
        slope = max(0.0, (elevation[i] - elevation[j]) / dist_m)
        slope_factor[i] = slope

    # normalization of slope_factor so it doesn't blow up; keep between 0..1
    if slope_factor.max() > 0:
        slope_factor = slope_factor / (slope_factor.max() + 1e-9)

    # rain_profile is a function of step index -> mm per hour at that time for the zone (we use same for all zones but could vary)
    # we will add tiny per-zone noise to rainfall to avoid perfect symmetry
    rng = random.Random(scenario_id + 12345)

    for step in range(steps):
        t_minutes = step * dt_minutes
        # rainfall mm/hr at this time
        rain_mm_per_hr = rain_profile(step, dt_minutes, duration_hours)
        # per-zone small variation (±5%)
        noise = 1.0 + (rng.random() - 0.5) * 0.10
        per_zone_rain_mm_per_hr = np.full(n, rain_mm_per_hr * noise, dtype=float)

        # convert to mm in this timestep
        R_dt = per_zone_rain_mm_per_hr * dt_hr  # mm this dt

        # infiltration: base * (1 - saturation) * drainage_capacity * dt_hr
        infiltration_rate_mm_per_hr = infiltration_base_mm_per_hr * (1.0 - S) * (0.5 + 0.5 * drainage)
        infiltrated = np.minimum(R_dt, infiltration_rate_mm_per_hr * dt_hr)

        runoff = R_dt - infiltrated
        runoff = np.maximum(runoff, 0.0)

        # river overflow injection if river_level exceeds a soft bank threshold
        river_overflow = np.zeros(n, dtype=float)
        # approximate bank elevation: mean of nearby zones elevation - 0.5 m
        # if river_level (m) > (elev - 0.5) -> generate overflow mm proportional to difference
        # convert diff in meters to mm of water per timestep using empirical factor
        for i in range(n):
            if adjacent_to_river[i]:
                bank = elevation[i] - 0.5
                if river_level > bank:
                    diff_m = river_level - bank
                    # empirical overflow mm per dt: diff_m * 1000 (to mm), scaled by small factor and dt
                    river_overflow[i] = diff_m * 1000.0 * 0.1 * dt_hr  # configurable factor

        # compute outflow from each cell to its downstream neighbor
        flow_to_down = np.zeros(n, dtype=float)
        for i in range(n):
            # water available = current standing + runoff
            available = W[i] + runoff[i]
            # fraction that moves downstream = k_flow * slope_factor * (1 - drainage_capacity)
            k_flow = 0.4  # tuning constant, controls speed of routing
            transfer = k_flow * slope_factor[i] * available * dt_hr
            # limit transfer to available
            transfer = min(available, transfer)
            flow_to_down[i] = transfer

        # apply transfers
        W_next = W.copy()
        for i, zid in enumerate(zone_ids):
            j = zone_ids.index(downstream_map[zid])
            # remove transferred amount from i, add to j
            W_next[i] = W_next[i] + runoff[i] - flow_to_down[i] - infiltrated[i] + river_overflow[i]
            W_next[j] += flow_to_down[i]

        # apply drainage decay (some water leaves)
        drainage_decay = 0.05 * drainage * dt_hr  # small fraction removed per hour scaled
        W_next = W_next * (1.0 - drainage_decay)
        # floor negative numbers
        W_next = np.maximum(W_next, 0.0)

        # update state
        W = W_next

        # optional: river level may rise slowly if sustained rainfall; simple heuristic
        river_level += 0.01 * (rain_mm_per_hr / 10.0) * dt_hr  # rise proportional to rainfall intensity

        # record timeseries per zone
        for i, zid in enumerate(zone_ids):
            recs.append({
                "scenario_id": scenario_id,
                "step": step,
                "minutes": t_minutes,
                "zone_id": zid,
                "water_mm": float(W[i]),
                "river_level_m": float(river_level)
            })

        # check inundation thresholds and fill inundation_time if not already set
        for i, zid in enumerate(zone_ids):
            if inundation_time[zid] is None:
                if W[i] >= flood_threshold_mm:
                    inundation_time[zid] = t_minutes  # minutes from start

    # build labels list
    labels = []
    for zid in zone_ids:
        t = inundation_time[zid]
        flooded = 1 if t is not None else 0
        labels.append({
            "scenario_id": scenario_id,
            "scenario_type": "auto",
            "start_time": datetime.utcnow().isoformat() + "Z",
            "zone_id": zid,
            "time_to_inundation_min": float(t) if t is not None else float("nan"),
            "flood_within_horizon": flooded,
            "initial_river_level": float(initial_river_level),
            "initial_soil_saturation": float(np.mean(S)),
            "rain_intensity_mm_per_hr": float(rain_profile(0, dt_minutes, duration_hours)),  # peak/initial approx
            "duration_hours": float(duration_hours)
        })

    timeseries_df = pd.DataFrame.from_records(recs)
    return labels, timeseries_df

# ---------- Rainfall profile helpers ----------
def make_step_profile(intensity_mm_per_hr, duration_hours, dt_minutes):
    """Constant intensity for duration_hours, then 0."""
    steps = int((DEFAULT_HORIZON_HOURS * 60) / dt_minutes) if dt_minutes else 36
    duration_steps = int((duration_hours * 60) / dt_minutes)
    def profile(step, dt_minutes_local, duration_hours_local):
        return intensity_mm_per_hr if step < duration_steps else 0.0
    return profile

def make_pulse_profile(peak_mm_per_hr, pulse_hours, dt_minutes):
    steps = int((DEFAULT_HORIZON_HOURS * 60) / dt_minutes)
    pulse_steps = int((pulse_hours * 60) / dt_minutes)
    def profile(step, dt_minutes_local, duration_hours_local):
        # triangular pulse centered in first half
        if step < pulse_steps:
            return peak_mm_per_hr * (1.0 - (step / pulse_steps) * 0.5)
        return 0.0
    return profile

def make_ramp_profile(peak_mm_per_hr, duration_hours, dt_minutes):
    duration_steps = int((duration_hours * 60) / dt_minutes)
    def profile(step, dt_minutes_local, duration_hours_local):
        if step >= duration_steps:
            return 0.0
        # linear ramp 0 -> peak
        return peak_mm_per_hr * (step / max(1, duration_steps))
    return profile

# ---------- Top-level driver ----------
def generate_scenarios(zones_df,
                       scenarios=DEFAULT_SCENARIOS,
                       dt_minutes=DEFAULT_DT_MINUTES,
                       horizon_hours=DEFAULT_HORIZON_HOURS,
                       rain_range=DEFAULT_RAIN_RANGE,
                       duration_range=DEFAULT_DURATION_RANGE,
                       river_level_range=DEFAULT_RIVER_LEVEL_RANGE,
                       flood_threshold_mm=DEFAULT_FLOOD_THRESHOLD_MM,
                       infiltration_base_mm_per_hr=DEFAULT_INFIL_BASE_MM_PER_HR,
                       out_dir=OUT_DIR,
                       timeseries_dir=TIMESERIES_DIR,
                       seed=42):
    _ensure_dir(out_dir)
    _ensure_dir(timeseries_dir)

    # build downstream map
    print("Loading zones from:", ZONES_SUMMARY_PATH)
    zones_df = pd.read_csv(ZONES_SUMMARY_PATH, dtype={"zone_id": str}).set_index("zone_id")
    downstream_map, dists = build_downstream_map(zones_df)

    labels_rows = []
    scenario_index = []

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    for s in range(scenarios):
        sid = s + 1
        # pick rainfall intensity and duration
        intensity = float(rng.uniform(*rain_range))  # mm/hr
        duration = float(rng.uniform(*duration_range))  # hours
        initial_river = float(rng.uniform(*river_level_range))
        soil_sat = float(rng.uniform(0.0, 0.9))

        # choose profile type randomly but weighted
        ptype = rng.choices(["step","ramp","pulse"], weights=[0.6, 0.25, 0.15])[0]
        if ptype == "step":
            profile = make_step_profile(intensity, duration, dt_minutes)
        elif ptype == "ramp":
            profile = make_ramp_profile(intensity, duration, dt_minutes)
        else:
            pulse_hours = min(2.0, duration)
            profile = make_pulse_profile(intensity, pulse_hours, dt_minutes)

        print(f"[Scenario {sid}/{scenarios}] type={ptype} intensity={intensity:.1f}mm/hr dur={duration:.2f}h river={initial_river:.2f} soil_sat={soil_sat:.2f}")

        labels, timeseries_df = run_single_scenario(
            scenario_id=sid,
            zones_df=zones_df,
            downstream_map=downstream_map,
            dists=dists,
            dt_minutes=dt_minutes,
            horizon_hours=horizon_hours,
            rain_profile=profile,
            duration_hours=duration,
            initial_river_level=initial_river,
            initial_soil_saturation=soil_sat,
            flood_threshold_mm=flood_threshold_mm,
            infiltration_base_mm_per_hr=infiltration_base_mm_per_hr
        )

        # append labels
        labels_rows.extend(labels)

        # save per-scenario timeseries
        ts_path = os.path.join(timeseries_dir, f"scenario_{sid}.csv")
        timeseries_df.to_csv(ts_path, index=False)

        scenario_index.append({
            "scenario_id": sid,
            "type": ptype,
            "intensity_mm_per_hr": intensity,
            "duration_hours": duration,
            "initial_river_level": initial_river,
            "initial_soil_saturation": soil_sat,
            "timeseries_path": os.path.relpath(ts_path, start=out_dir)
        })

    # Save combined labels to CSV
    labels_df = pd.DataFrame.from_records(labels_rows)
    labels_df.to_csv(LABELS_PATH, index=False)
    # Save scenario index
    with open(SCENARIO_INDEX_PATH, "w") as f:
        json.dump(scenario_index, f, indent=2)

    print("Simulation complete.")
    print("Labels saved to:", LABELS_PATH)
    print("Scenario index saved to:", SCENARIO_INDEX_PATH)
    return labels_df, scenario_index

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Run hydrologic CA simulations to create training labels.")
    p.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS, help="Number of scenarios to simulate")
    p.add_argument("--dt", type=int, default=DEFAULT_DT_MINUTES, help="Timestep in minutes")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_HOURS, help="Horizon in hours")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out", type=str, default=OUT_DIR, help="Output directory for generated files")
    p.add_argument("--quick", action="store_true", help="Quick small run (5 scenarios) for debugging")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        sc = 5
    else:
        sc = args.scenarios

    # run
    labels_df, scenario_index = generate_scenarios(
        zones_df=None,  # will be loaded internally
        scenarios=sc,
        dt_minutes=args.dt,
        horizon_hours=args.horizon,
        seed=args.seed,
        out_dir=args.out
    )
    # print summary
    print("\nSummary stats (labels):")
    print(labels_df["flood_within_horizon"].value_counts(dropna=False))
    # head
    print("\nSample label rows:")
    print(labels_df.head().to_string(index=False))
