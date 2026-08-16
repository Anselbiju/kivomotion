from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = {
    "target_impulse_ns",
    "target_impulse_ns_kg",
    "target_takeoff_velocity_m_s",
}


@dataclass
class ModelResult:
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    best_model: object
    feature_names: list[str]
    validation_method: str
    research_ready: bool


def _models(random_state: int = 42) -> dict[str, object]:
    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    max_features=0.7,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    hist = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(l2_regularization=2.0, random_state=random_state)),
        ]
    )
    return {"ridge": ridge, "random_forest": forest, "hist_gradient_boosting": hist}


def train_and_evaluate(
    features: pd.DataFrame,
    target: str = "target_impulse_ns_kg",
    min_research_events: int = 30,
    model_path: str | Path | None = None,
) -> ModelResult:
    """Compare regressors using out-of-fold predictions.

    Athlete/session grouping is used whenever at least three independent groups
    exist. A single small sample falls back to leave-one-jump-out for pipeline
    demonstration only and is explicitly marked as not research-ready.
    """
    if target not in features:
        raise ValueError(f"Target {target!r} is not present")
    id_cols = {"jump_id", "jump_type", "athlete_id", "session_id"}
    feature_names = [
        col
        for col in features.select_dtypes(include=np.number).columns
        if col not in TARGET_COLUMNS and col not in id_cols
    ]
    X = features[feature_names]
    y = features[target].to_numpy(dtype=float)
    if len(features) < 4:
        raise ValueError("At least four matched jumps are needed even for a demonstration")

    group_col = "athlete_id" if "athlete_id" in features and features["athlete_id"].nunique() >= 3 else "session_id"
    use_groups = group_col in features and features[group_col].nunique() >= 3
    if use_groups:
        groups = features[group_col].astype(str).to_numpy()
        splits = min(5, len(np.unique(groups)))
        cv = GroupKFold(n_splits=splits)
        cv_args = {"groups": groups}
        validation_method = f"{splits}-fold grouped CV by {group_col}"
    else:
        cv = LeaveOneOut()
        cv_args = {}
        validation_method = "leave-one-jump-out DEMO (no independent athlete/session groups)"
        warnings.warn(
            "No independent groups are available. Metrics demonstrate code execution only; "
            "they are not an estimate of new-athlete performance.",
            RuntimeWarning,
        )

    rows, prediction_rows = [], []
    fitted_models: dict[str, object] = {}
    for name, model in _models().items():
        # One process is the safest default in VS Code, notebooks and restricted
        # containers. Increase this locally only after confirming multiprocessing.
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1, **cv_args)
        rows.append(
            {
                "model": name,
                "mae": mean_absolute_error(y, pred),
                "rmse": mean_squared_error(y, pred) ** 0.5,
                "r2": r2_score(y, pred),
            }
        )
        for idx, (actual, predicted) in enumerate(zip(y, pred)):
            prediction_rows.append(
                {
                    "model": name,
                    "row": idx,
                    "jump_id": features.iloc[idx].get("jump_id", idx + 1),
                    "actual": actual,
                    "predicted": predicted,
                    "residual": actual - predicted,
                }
            )
        fitted_models[name] = clone(model).fit(X, y)

    leaderboard = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    best_name = str(leaderboard.iloc[0]["model"])
    best_model = fitted_models[best_name]
    independent_athletes = features["athlete_id"].nunique() if "athlete_id" in features else 0
    # Session-grouped CV is useful for a personalized model, but it does not
    # establish generalization to unseen athletes.
    research_ready = bool(len(features) >= min_research_events and independent_athletes >= 3 and use_groups)
    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": best_model,
                "features": feature_names,
                "target": target,
                "validation_method": validation_method,
                "research_ready": research_ready,
            },
            model_path,
        )
    return ModelResult(
        leaderboard=leaderboard,
        predictions=pd.DataFrame(prediction_rows),
        best_model=best_model,
        feature_names=feature_names,
        validation_method=validation_method,
        research_ready=research_ready,
    )
