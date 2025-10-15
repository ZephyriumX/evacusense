"""
scripts/train_eta.py

Train ETA regression and flood classification models using simulated labels and current features.

Outputs:
 - models/eta_regressor.joblib
 - models/eta_classifier.joblib
 - models/model_info.json

Usage:
    # quick (small) training (for dev)
    python -m scripts.train_eta --quick

    # full training on all available labels
    python -m scripts.train_eta
"""

import os
import json
import argparse
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "generated", "features_ready_for_model.csv")
LABELS_PATH = os.path.join(BASE_DIR, "data", "generated", "sim_labels.csv")
METADATA_PATH = os.path.join(BASE_DIR, "models", "feature_metadata.json")
OUT_DIR = os.path.join(BASE_DIR, "models")
REG_MODEL_PATH = os.path.join(OUT_DIR, "eta_regressor.joblib")
CLS_MODEL_PATH = os.path.join(OUT_DIR, "eta_classifier.joblib")
MODEL_INFO_PATH = os.path.join(OUT_DIR, "model_info.json")


def try_xgboost():
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def load_data():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features missing: {FEATURES_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels missing: {LABELS_PATH}")
    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    return features, labels


def build_training_table(features, labels, metadata):
    """
    Merge features (single snapshot per zone) with labels (many scenarios per zone).
    Attach scenario-level columns as features: initial_river_level, initial_soil_saturation,
    rain_intensity_mm_per_hr, duration_hours.
    """
    # prepare features: index by zone_id
    if "zone_id" not in features.columns:
        raise ValueError("features must include zone_id column")
    feat_df = features.copy()
    feat_df = feat_df.set_index("zone_id")

    # Select numeric feature columns per metadata
    feature_order = metadata.get("feature_order", [])
    # exclude timestamp + zone_id placeholders
    candidate_cols = [c for c in feature_order if c not in ("timestamp", "zone_id", "centroid_lat", "centroid_lon")]
    # Some columns may not exist; filter
    numeric_cols = [c for c in candidate_cols if c in feat_df.columns]

    # now expand labels: join labels (each scenario-zone row) to static features by zone_id
    labels_copy = labels.copy()
    labels_copy = labels_copy.merge(feat_df.reset_index(), left_on="zone_id", right_on="zone_id", how="left", suffixes=("", "_feat"))

    # features to use: scenario meta + numeric zone features + static coords if needed
    scenario_cols = ["initial_river_level", "initial_soil_saturation", "rain_intensity_mm_per_hr", "duration_hours"]
    use_cols = [c for c in scenario_cols if c in labels_copy.columns] + numeric_cols

    # drop rows where feature merge failed
    labels_copy = labels_copy.dropna(subset=["zone_id"])
    X = labels_copy[use_cols].copy()
    # basic fill
    X = X.fillna(0.0)

    # target: time_to_inundation_min (regression) and flood_within_horizon (classification)
    y_reg = labels_copy["time_to_inundation_min"].copy()
    y_clf = labels_copy["flood_within_horizon"].copy().astype(int)

    return X, y_reg, y_clf, use_cols


def train_models(X_reg_train, X_reg_val, y_reg_train, y_reg_val,
                 X_clf_train, X_clf_val, y_clf_train, y_clf_val):
    """
    Train regressor (on flooded rows only) and classifier (on full dataset rows).
    Returns trained reg, clf and metrics dict.
    """
    use_xgb = try_xgboost()
    print("XGBoost available:", use_xgb)

    # initialize metrics to None (avoid possibly-unbound warnings)
    mae = rmse = r2 = None

    # Regressor (may be skipped if no flooded samples)
    reg = None
    if y_reg_train is None or (hasattr(y_reg_train, "__len__") and len(y_reg_train) == 0):
        print("Warning: No training samples for regressor (no flooded rows). Skipping regressor training.")
    else:
        if use_xgb:
            from xgboost import XGBRegressor
            reg = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42, verbosity=0)
        else:
            reg = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)

        print("Training regression model...")
        # fit
        reg.fit(X_reg_train, y_reg_train)
        # eval
        y_reg_pred = reg.predict(X_reg_val)
        mae = mean_absolute_error(y_reg_val, y_reg_pred)
        # compute RMSE compatibly across sklearn versions
        try:
            rmse = mean_squared_error(y_reg_val, y_reg_pred, squared=False)
        except TypeError:
            rmse = math.sqrt(mean_squared_error(y_reg_val, y_reg_pred))
        r2 = r2_score(y_reg_val, y_reg_pred)
        print(f"Regressor results — MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # Classifier (always train)
    clf = None
    if use_xgb:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)

    print("Training classifier...")
    clf.fit(X_clf_train, y_clf_train)
    y_clf_pred = clf.predict(X_clf_val)

    # Determine probabilities / scores in a safe way (use getattr + callable to satisfy Pylance)
    y_clf_prob = None
    pred_proba_fn = getattr(clf, "predict_proba", None)
    if callable(pred_proba_fn):
        try:
            y_clf_prob = pred_proba_fn(X_clf_val)[:, 1]
        except Exception:
            y_clf_prob = None

    if y_clf_prob is None:
        decision_fn = getattr(clf, "decision_function", None)
        if callable(decision_fn):
            try:
                y_clf_prob = decision_fn(X_clf_val)
            except Exception:
                y_clf_prob = None

    acc = accuracy_score(y_clf_val, y_clf_pred)
    prec = precision_score(y_clf_val, y_clf_pred, zero_division=0)
    rec = recall_score(y_clf_val, y_clf_pred, zero_division=0)
    # Only compute ROC-AUC if we have at least two classes in validation labels
    try:
        if len(np.unique(y_clf_val)) > 1:
            roc = roc_auc_score(y_clf_val, y_clf_prob)
        else:
            roc = 0.0
    except Exception:
        roc = 0.0

    print(f"Classifier results — Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, ROC-AUC: {roc:.3f}")

    reg_metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    clf_metrics = {"acc": acc, "prec": prec, "rec": rec, "roc_auc": roc}

    return reg, clf, {"reg_metrics": reg_metrics, "clf_metrics": clf_metrics}


def save_models(reg, clf, feature_columns, metadata):
    os.makedirs(OUT_DIR, exist_ok=True)
    # Save reg only if trained
    if reg is not None:
        joblib.dump(reg, REG_MODEL_PATH)
    # Save classifier (always)
    joblib.dump(clf, CLS_MODEL_PATH)

    info = {
        "regressor_path": os.path.relpath(REG_MODEL_PATH, start=OUT_DIR) if reg is not None else None,
        "classifier_path": os.path.relpath(CLS_MODEL_PATH, start=OUT_DIR),
        "features": feature_columns,
        "metadata_source": metadata,
        "saved_at": datetime.utcnow().isoformat() + "Z"
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print("Saved models to:", ("<skipped regressor>" if reg is None else REG_MODEL_PATH), "and", CLS_MODEL_PATH)
    return info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Quick small training")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    features, labels = load_data()

    # quick mode: sample small subset for dev (but keep distribution)
    if args.quick:
        # take at most 5 scenarios per zone (if present) to keep dataset small
        try:
            labels = labels.groupby("zone_id").head(5).reset_index(drop=True)
        except Exception:
            # fallback: sample small fraction
            labels = labels.sample(frac=0.1, random_state=args.seed).reset_index(drop=True)

    # load metadata medians + feature order if exists
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
    else:
        metadata = {"feature_order": [], "medians": {}}

    X, y_reg, y_clf, feature_columns = build_training_table(features, labels, metadata)

    # ------- Split for classifier (use full dataset) -------
    # Attempt stratified split on y_clf if both classes present and enough samples
    try:
        unique, counts = np.unique(y_clf, return_counts=True)
        if len(unique) > 1 and min(counts) >= 2:
            X_clf_train, X_clf_val, y_clf_train, y_clf_val = train_test_split(
                X, y_clf, test_size=0.2, random_state=args.seed, stratify=y_clf
            )
        else:
            X_clf_train, X_clf_val, y_clf_train, y_clf_val = train_test_split(
                X, y_clf, test_size=0.2, random_state=args.seed
            )
    except Exception:
        X_clf_train, X_clf_val, y_clf_train, y_clf_val = train_test_split(
            X, y_clf, test_size=0.2, random_state=args.seed
        )

    # ------- Prepare regression training set (only flooded rows) -------
    # flooded_mask True where y_reg is finite (not NaN)
    flooded_mask = ~y_reg.isna()
    X_reg_all = X[flooded_mask].reset_index(drop=True)
    y_reg_all = y_reg[flooded_mask].reset_index(drop=True)

    if len(X_reg_all) >= 10:
        # normal split for regressor
        X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
            X_reg_all, y_reg_all, test_size=0.2, random_state=args.seed
        )
    elif len(X_reg_all) > 0:
        # too small to split well: use all samples for both train and val (degenerate but avoids crash)
        X_reg_train = X_reg_all
        X_reg_val = X_reg_all
        y_reg_train = y_reg_all
        y_reg_val = y_reg_all
        print("Warning: small flooded sample size (<=10). Using same small set for reg train/val.")
    else:
        # no flooded rows at all
        X_reg_train = pd.DataFrame(columns=X.columns)
        X_reg_val = pd.DataFrame(columns=X.columns)
        y_reg_train = pd.Series(dtype=float)
        y_reg_val = pd.Series(dtype=float)
        print("Warning: No flooded rows found in labels. Regressor will be skipped.")

    # Train models
    reg, clf, metrics = train_models(
        X_reg_train, X_reg_val, y_reg_train, y_reg_val,
        X_clf_train, X_clf_val, y_clf_train, y_clf_val
    )

    # Save models and metadata
    info = save_models(reg, clf, feature_columns, METADATA_PATH)
    info["metrics"] = metrics
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    print("Training finished. Model info saved to", MODEL_INFO_PATH)


if __name__ == "__main__":
    main()
