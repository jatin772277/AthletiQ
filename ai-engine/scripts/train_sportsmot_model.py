"""Train AthletiQ performance model on SportsMOT football-generated labels.

Reads outputs/sportsmot_ml/sportsmot_ml_merged.csv produced by
generate_sportsmot_ml_data.py and trains a regressor that predicts
within-sequence activity percentile (0-100) from pixel-space
movement features.

Split strategy
--------------
* train sequences (308 rows) -> training set
* val sequences   (309 rows) -> hold-out validation (MAE / R2 reported)
* test sequences  -> held out entirely (not used during model selection)

Usage
-----
  python ai-engine/scripts/train_sportsmot_model.py
  python ai-engine/scripts/train_sportsmot_model.py --merged-csv path/to/file.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MERGED = ROOT / "outputs/sportsmot_ml/sportsmot_ml_merged.csv"

MODEL_DIR  = ROOT / "ai-engine/models/performance_scoring"
MODEL_PATH = MODEL_DIR / "sportsmot_performance_model.joblib"
CANONICAL_MODEL_PATH = MODEL_DIR / "performance_model.joblib"

REPORT_DIR = ROOT / "outputs/sportsmot_ml"
REPORT_PATH = REPORT_DIR / "sportsmot_model_report.json"
PERF_REPORT_PATH = ROOT / "outputs/performance/football_sportsmot_model_report.json"

TARGET = "reviewed_performance_score_0_100"
ID_COL = "canonical_id"

# Columns that must never be used as predictors.
NON_FEATURES = {
    ID_COL,
    "seq_id",
    "split",
    "track_id",
    "first_frame",
    "last_frame",
    "label_source",
    "reviewer_notes",
    TARGET,
    "seq_activity_score_0_100",
    "global_distance_percentile_0_100",
}


def build_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale",   StandardScaler()),
        ("model",   estimator),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=DEFAULT_MERGED,
        help="Path to sportsmot_ml_merged.csv",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="K-fold splits for cross-validation on train set",
    )
    args = parser.parse_args()

    if not args.merged_csv.exists():
        raise SystemExit(
            f"Merged CSV not found: {args.merged_csv}\n"
            "Run generate_sportsmot_ml_data.py first."
        )

    df = pd.read_csv(args.merged_csv)

    if TARGET not in df.columns:
        raise SystemExit(f"Target column missing: {TARGET}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET].between(0, 100)].copy()

    if len(df) < 20:
        raise SystemExit(
            f"Only {len(df)} labeled rows -- need >= 20 to train."
        )

    # Split by sequence split label.
    train_mask = df["split"] == "train" if "split" in df.columns else pd.Series(True, index=df.index)
    val_mask   = df["split"] == "val"   if "split" in df.columns else pd.Series(False, index=df.index)

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()

    print("=" * 66)
    print("ATHLETIQ: Training Football SportsMOT Performance Scoring Model")
    print("=" * 66)
    print(f"Training rows  : {len(train_df)} (from train split sequences)")
    print(f"Validation rows: {len(val_df)} (from val split sequences)")

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"Feature columns ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  - {f}")

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_df[TARGET].astype(float)

    if y_train.nunique() < 2:
        raise SystemExit("Target has fewer than 2 unique values in training set.")

    # ------- Model selection via CV on training set -------
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.8, random_state=42, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.8, random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42,
        ),
    }

    n_folds = min(args.cv_folds, max(2, len(train_df) // 4))
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"\nEvaluating candidate models ({n_folds}-fold CV on train split)...")
    cv_results = {}
    for name, est in candidates.items():
        pipe = build_pipeline(est)
        mae_scores = -cross_val_score(
            pipe, X_train, y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
        )
        r2_scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv,
            scoring="r2",
        )
        cv_results[name] = {
            "cv_mae_mean": float(mae_scores.mean()),
            "cv_mae_std":  float(mae_scores.std()),
            "cv_r2_mean":   float(r2_scores.mean()),
            "folds":       n_folds,
        }
        print(
            f"  {name:18s}: CV MAE = {mae_scores.mean():.2f} +/- {mae_scores.std():.2f} | "
            f"CV R2 = {r2_scores.mean():.3f}"
        )

    winner = min(cv_results, key=lambda n: cv_results[n]["cv_mae_mean"])
    print(f"\nWinning model selected: {winner}")

    final_pipe = build_pipeline(candidates[winner])
    final_pipe.fit(X_train, y_train)

    # Feature importances
    importances = {}
    fitted_model = final_pipe.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        raw_imp = fitted_model.feature_importances_
        # handle indicator columns from SimpleImputer
        named_features = feature_cols
        for i, f in enumerate(named_features):
            if i < len(raw_imp):
                importances[f] = float(raw_imp[i])
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    print("\nTop 5 Feature Importances:")
    for f, imp in list(importances.items())[:5]:
        print(f"  {f:40s}: {imp:.4f}")

    # ------- Validation set evaluation -------
    val_results = {}
    if len(val_df) >= 5:
        X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        y_val = val_df[TARGET].astype(float)
        y_pred_val = final_pipe.predict(X_val)
        val_results = {
            "val_mae":  float(mean_absolute_error(y_val, y_pred_val)),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
            "val_r2":   float(r2_score(y_val, y_pred_val)),
            "val_rows": len(val_df),
        }
        print(
            f"\nHoldout Validation on Football Val Sequences:\n"
            f"  MAE : {val_results['val_mae']:.2f} (out of 100)\n"
            f"  RMSE: {val_results['val_rmse']:.2f}\n"
            f"  R2  : {val_results['val_r2']:.3f}"
        )
    else:
        print("[INFO] Not enough validation rows for hold-out evaluation.")

    # ------- Save model -------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":           final_pipe,
        "feature_columns": feature_cols,
        "target":          TARGET,
        "model_name":      winner,
        "dataset":         "sportsmot_football",
        "training_rows":   len(train_df),
        "validation":      val_results,
        "feature_importances": importances,
        "label_source":    "sportsmot_football_gt_activity_percentile_within_sequence",
    }
    joblib.dump(payload, MODEL_PATH)
    joblib.dump(payload, CANONICAL_MODEL_PATH)
    print(f"\nModel saved to:\n  {MODEL_PATH}\n  {CANONICAL_MODEL_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PERF_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "status":              "trained",
        "dataset":             "sportsmot_football",
        "model":               winner,
        "training_rows":       len(train_df),
        "validation_rows":     len(val_df),
        "feature_columns":     feature_cols,
        "target":              TARGET,
        "cross_validation":    cv_results,
        "hold_out_validation": val_results,
        "feature_importances": importances,
        "artifact":            str(MODEL_PATH.relative_to(ROOT)),
        "canonical_artifact":  str(CANONICAL_MODEL_PATH.relative_to(ROOT)),
        "description": (
            "AthletiQ Performance Scoring Regressor trained on SportsMOT football sequences. "
            "Predicts within-match/sequence activity percentile score (0-100) from tracking kinematics."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    PERF_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nReport written to:\n  " + str(REPORT_PATH) + "\n  " + str(PERF_REPORT_PATH))
    print("\n=== Football SportsMOT Model Training Complete ===")


if __name__ == "__main__":
    main()
