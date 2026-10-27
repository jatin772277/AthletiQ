"""Train a supervised performance regressor only from reviewed labels."""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "outputs/dsa/football2_ml_features.csv"
LABELS = ROOT / "outputs/dsa/player_performance_labels_template.csv"
MODEL_DIR = ROOT / "ai-engine/models/performance_scoring"
MODEL_PATH = MODEL_DIR / "performance_model.joblib"
REPORT_PATH = ROOT / "outputs/performance/football2_supervised_model_report.json"

TARGET = "reviewed_performance_score_0_100"
ID = "canonical_id"

EXCLUDED = {
    ID, "team_label", "team_confidence",
    "reviewed_player_identity", "label_source", "reviewer_notes", TARGET,
}


def main():
    if not FEATURES.exists() or not LABELS.exists():
        raise SystemExit("Feature CSV or reviewed-label template is missing.")

    features = pd.read_csv(FEATURES)
    labels = pd.read_csv(LABELS)

    if TARGET not in labels.columns:
        raise SystemExit(f"Missing reviewed target column: {TARGET}")

    data = features.merge(
        labels[[ID, TARGET, "label_source"]],
        on=ID,
        how="inner",
        validate="one_to_one",
    )
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).copy()
    data = data[data[TARGET].between(0, 100)]

    if data["canonical_id"].duplicated().any():
        raise SystemExit("Duplicate IDs in training rows.")
    if len(data) < 20:
        print(f"Reviewed labeled rows: {len(data)}")
        print("NOT TRAINED: need at least 20 reviewed rows for this initial baseline.")
        print("This is a practical minimum, not a guarantee of model validity.")
        return

    # Avoid ID, team predictions, and reviewer metadata as predictive inputs.
    feature_cols = [
        c for c in features.columns
        if c not in EXCLUDED and pd.api.types.is_numeric_dtype(features[c])
    ]
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = data[TARGET].astype(float)

    if y.nunique() < 2:
        raise SystemExit("Reviewed target must contain more than one score value.")

    cv = KFold(n_splits=min(5, len(data) // 4), shuffle=True, random_state=42)
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.8, random_state=42, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.8, random_state=42, n_jobs=-1,
        ),
    }

    results = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])
        mae = -cross_val_score(
            pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )
        results[name] = {
            "cv_mae_mean": float(mae.mean()),
            "cv_mae_std": float(mae.std()),
            "folds": cv.get_n_splits(),
        }

    winner = min(results, key=lambda n: results[n]["cv_mae_mean"])
    final_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", candidates[winner]),
    ])
    final_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "feature_columns": feature_cols,
        "target": TARGET,
        "model_name": winner,
        "training_rows": len(data),
        "validation": "K-fold cross-validation; report is preliminary",
    }, MODEL_PATH)

    report = {
        "status": "trained_preliminary",
        "model": winner,
        "training_rows": len(data),
        "feature_columns": feature_cols,
        "target": TARGET,
        "cross_validation": results,
        "warning": (
            "Performance depends on label quality, player identity correctness, "
            "and whether validation reflects the intended deployment setting. "
            "Do not treat this as validated until reviewed."
        ),
        "artifact": str(MODEL_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
