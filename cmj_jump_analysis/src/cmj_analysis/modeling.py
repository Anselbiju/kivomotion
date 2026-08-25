from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
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

# These columns describe collection or identity. They must never become model
# inputs, even when pandas reads a date/test value as a numeric dtype.
ADMIN_COLUMNS = {
    "jump_id",
    "jump_type",
    "athlete_id",
    "pseudo_athlete_id",
    "session_id",
    "recording_date",
    "test_number",
    "imu_orientation_group",
}

FEATURE_SETS = {"all_axes", "magnitude_only"}


@dataclass
class ModelResult:
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    best_model: object
    feature_names: list[str]
    validation_method: str
    research_ready: bool
    feature_set: str
    target: str


@dataclass
class FeatureSetComparison:
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    best_result: ModelResult
    best_feature_set: str
    validation_method: str
    research_ready: bool


def select_feature_names(features: pd.DataFrame, feature_set: str = "all_axes") -> list[str]:
    """Select leakage-safe numeric predictors for a documented feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
    candidates = [
        col
        for col in features.select_dtypes(include=np.number).columns
        if col not in TARGET_COLUMNS and col not in ADMIN_COLUMNS
    ]
    if feature_set == "all_axes":
        selected = candidates
    else:
        magnitude_exact = {
            "imu_flight_time_s",
            "acc_mag_area_g_s",
            "gyro_mag_area_deg",
        }
        magnitude_prefixes = ("acc_mag_g_f_", "gyro_mag_dps_f_")
        selected = [
            col
            for col in candidates
            if col in magnitude_exact or col.startswith(magnitude_prefixes)
        ]
    if not selected:
        raise ValueError(f"No numeric predictors are available for feature_set={feature_set!r}")
    return selected


def _save_model_result(result: ModelResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.best_model,
            "features": result.feature_names,
            "target": result.target,
            "feature_set": result.feature_set,
            "validation_method": result.validation_method,
            "research_ready": result.research_ready,
            "training_metrics": result.leaderboard.iloc[0].to_dict(),
            "library_versions": {
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        },
        path,
    )


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
    feature_set: str = "all_axes",
) -> ModelResult:
    """Compare regressors using out-of-fold predictions.

    Athlete/session grouping is used whenever at least three independent groups
    exist. A single small sample falls back to leave-one-jump-out for pipeline
    demonstration only and is explicitly marked as not research-ready.
    """
    if target not in features:
        raise ValueError(f"Target {target!r} is not present")
    feature_names = select_feature_names(features, feature_set)
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
                "feature_set": feature_set,
                "model": name,
                "mae": mean_absolute_error(y, pred),
                "rmse": mean_squared_error(y, pred) ** 0.5,
                "r2": r2_score(y, pred),
            }
        )
        for idx, (actual, predicted) in enumerate(zip(y, pred)):
            prediction_rows.append(
                {
                    "feature_set": feature_set,
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
    result = ModelResult(
        leaderboard=leaderboard,
        predictions=pd.DataFrame(prediction_rows),
        best_model=best_model,
        feature_names=feature_names,
        validation_method=validation_method,
        research_ready=research_ready,
        feature_set=feature_set,
        target=target,
    )
    if model_path:
        _save_model_result(result, model_path)
    return result


def compare_feature_sets(
    features: pd.DataFrame,
    target: str = "target_impulse_ns_kg",
    feature_sets: tuple[str, ...] = ("all_axes", "magnitude_only"),
    min_research_events: int = 30,
    model_path: str | Path | None = None,
    candidate_models_dir: str | Path | None = None,
) -> FeatureSetComparison:
    """Evaluate feature families under the same grouped CV and save the winner."""
    results: dict[str, ModelResult] = {}
    for feature_set in feature_sets:
        candidate_path = None
        if candidate_models_dir:
            candidate_path = Path(candidate_models_dir) / f"best_{feature_set}_model.joblib"
        results[feature_set] = train_and_evaluate(
            features,
            target=target,
            min_research_events=min_research_events,
            model_path=candidate_path,
            feature_set=feature_set,
        )

    leaderboard = (
        pd.concat([result.leaderboard for result in results.values()], ignore_index=True)
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    predictions = pd.concat(
        [result.predictions for result in results.values()], ignore_index=True
    )
    best_feature_set = str(leaderboard.iloc[0]["feature_set"])
    best_result = results[best_feature_set]
    if model_path:
        _save_model_result(best_result, model_path)
    return FeatureSetComparison(
        leaderboard=leaderboard,
        predictions=predictions,
        best_result=best_result,
        best_feature_set=best_feature_set,
        validation_method=best_result.validation_method,
        research_ready=best_result.research_ready,
    )
