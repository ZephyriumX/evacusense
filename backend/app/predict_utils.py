"""
backend.app.predict_utils

Utilities to load trained models and run predictions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_INFO_PATH = os.path.join(BASE_DIR, "models", "model_info.json")
REG_MODEL_PATH = os.path.join(BASE_DIR, "models", "eta_regressor.joblib")
CLS_MODEL_PATH = os.path.join(BASE_DIR, "models", "eta_classifier.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "generated", "features_ready_for_model.csv")
METADATA_PATH = os.path.join(BASE_DIR, "models", "feature_metadata.json")

class Predictor:
    def __init__(self):
        self.reg = None
        self.clf = None
        self.feature_columns = None
        self.medians = {}
        self._load()

    def _load(self):
        # load metadata medians (for imputation)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                md = json.load(f)
                self.medians = md.get("medians", {})
        # load models if available
        if os.path.exists(REG_MODEL_PATH):
            self.reg = joblib.load(REG_MODEL_PATH)
        if os.path.exists(CLS_MODEL_PATH):
            self.clf = joblib.load(CLS_MODEL_PATH)
        # load feature list from model_info
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH) as f:
                info = json.load(f)
                self.feature_columns = info.get("features", None)

    def preprocess(self, df):
        """
        Accepts df with zone rows and scenario cols (optional).
        Ensures columns self.feature_columns exist and fills missing with medians.
        """
        if self.feature_columns is None:
            # fallback: use all numeric columns
            cols = df.select_dtypes(include=[float, int]).columns.tolist()
        else:
            cols = [c for c in self.feature_columns if c in df.columns]
        proc = df[cols].copy()
        # impute medians
        for c in cols:
            if proc[c].isnull().any():
                med = self.medians.get(c, 0.0)
                proc[c] = proc[c].fillna(med)
        # ensure numeric
        proc = proc.astype(float)
        return proc, cols

    def predict_df(self, df):
        """
        Input: df with zone rows including required feature columns (and optional scenario cols).
        Returns df with added columns: predicted_eta_min, flood_prob
        """
        proc_df, used_cols = self.preprocess(df)
        out = df.copy().reset_index(drop=True)
        if self.reg is not None:
            try:
                pred = self.reg.predict(proc_df[used_cols])
            except Exception:
                pred = np.zeros(len(proc_df))
            out["predicted_eta_minutes"] = pred
        else:
            out["predicted_eta_minutes"] = np.nan

        if self.clf is not None:
            try:
                prob = self.clf.predict_proba(proc_df[used_cols])[:, 1]
            except Exception:
                # classifier with no predict_proba
                prob = self.clf.predict(proc_df[used_cols])
            out["flood_prob"] = prob
        else:
            out["flood_prob"] = np.nan

        return out

# convenience single-instance
_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor

def predict_from_features_file(features_csv_path=None):
    if features_csv_path is None:
        features_csv_path = FEATURES_PATH
    df = pd.read_csv(features_csv_path)
    p = get_predictor()
    return p.predict_df(df)
