from __future__ import annotations

from pathlib import Path
import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_report(
    matched: pd.DataFrame,
    leaderboard: pd.DataFrame,
    predictions: pd.DataFrame,
    validation_method: str,
    research_ready: bool,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_name = str(leaderboard.iloc[0]["model"])
    best_feature_set = (
        str(leaderboard.iloc[0]["feature_set"])
        if "feature_set" in leaderboard.columns
        else "all_axes"
    )
    prediction_mask = predictions["model"] == best_name
    if "feature_set" in predictions.columns:
        prediction_mask &= predictions["feature_set"] == best_feature_set
    best_predictions = predictions[prediction_mask].copy()
    best_label = f"{best_feature_set} / {best_name}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(matched)) + 1
    axes[0, 0].bar(x, matched["net_impulse_ns"] / matched["mass_kg"], color="#3572A5")
    axes[0, 0].set(title="Force-plate net impulse", xlabel="Jump", ylabel="N·s/kg")

    if len(matched) > 30:
        axes[0, 1].scatter(x, matched["takeoff_velocity_m_s"], s=12, label="Impulse method")
        axes[0, 1].scatter(x, matched["flight_velocity_m_s"], s=12, marker="s", label="Flight-time method")
    else:
        axes[0, 1].plot(x, matched["takeoff_velocity_m_s"], "o-", label="Impulse method")
        axes[0, 1].plot(x, matched["flight_velocity_m_s"], "s--", label="Flight-time method")
    axes[0, 1].set(title="Takeoff velocity cross-check", xlabel="Jump", ylabel="m/s")
    axes[0, 1].legend()

    axes[1, 0].scatter(best_predictions["actual"], best_predictions["predicted"], s=55, color="#E07A3F")
    limits = [
        min(best_predictions[["actual", "predicted"]].min()) * 0.98,
        max(best_predictions[["actual", "predicted"]].max()) * 1.02,
    ]
    axes[1, 0].plot(limits, limits, "k--", linewidth=1)
    axes[1, 0].set(title=f"{best_label}: out-of-fold prediction", xlabel="Actual N·s/kg", ylabel="Predicted N·s/kg")

    residual_x = np.arange(len(best_predictions)) + 1
    axes[1, 1].bar(residual_x, best_predictions["residual"], color="#6A9F58")
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set(title="Prediction residual", xlabel="Jump", ylabel="Actual − predicted N·s/kg")

    fig.suptitle("CMJ force plate + IMU analytics", fontsize=16)
    chart_path = output_dir / "prediction_report.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    readiness = "Research-ready validation" if research_ready else "DEMONSTRATION ONLY — collect more independent sessions/athletes"
    metrics_html = leaderboard.round(4).to_html(index=False, border=0)
    jumps_cols = [
        "jump_id", "flight_time_s", "net_impulse_ns", "takeoff_velocity_m_s",
        "flight_velocity_m_s", "sync_error_s", "sync_quality",
    ]
    jumps_html = matched[jumps_cols].round(4).to_html(index=False, border=0)
    orientation_html = ""
    if {"session_id", "imu_orientation_group"}.issubset(matched.columns):
        orientation = (
            matched[["session_id", "imu_orientation_group"]]
            .drop_duplicates()
            .groupby("imu_orientation_group")
            .size()
            .rename("sessions")
            .reset_index()
        )
        orientation_html = (
            "<h2>IMU orientation quality</h2>"
            + orientation.to_html(index=False, border=0)
            + "<p>Orientation is a quality label only; it is excluded from model predictors.</p>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMJ Prediction Report</title><style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;color:#17212b;line-height:1.45}}
.warning{{background:#fff3cd;border-left:5px solid #e0a800;padding:12px 16px}} table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#eef4f8}}img{{width:100%;height:auto}}
</style></head><body><h1>CMJ force plate + IMU prediction report</h1>
<p class="warning"><strong>{html.escape(readiness)}</strong><br>{html.escape(validation_method)}</p>
<p>Target: force-plate net impulse normalized by body mass (N·s/kg). The selected model/feature-set combination is <strong>{html.escape(best_label)}</strong>, based on lowest out-of-fold RMSE.</p>
<img src="prediction_report.png" alt="CMJ analytics charts"><h2>Model comparison</h2>{metrics_html}
{orientation_html}<h2>Matched jumps</h2>{jumps_html}<h2>Interpretation</h2>
<p>Use these metrics only after validation is grouped by athlete (preferred) or independent session. Never randomly split windows from the same jump across train and test data.</p>
</body></html>"""
    html_path = output_dir / "prediction_report.html"
    html_path.write_text(document, encoding="utf-8")
    return html_path, chart_path


def create_imu_prediction_report(
    predictions: pd.DataFrame,
    quality: dict,
    model_metadata: dict,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Create a standalone report for a new IMU-only recording."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        raise ValueError("At least one IMU prediction is required for the report")

    x = predictions["jump_id"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(x, predictions["predicted_normalized_impulse_ns_kg"], color="#3572A5")
    axes[0].set(
        title="Predicted normalized net impulse",
        xlabel="Jump",
        ylabel="N·s/kg (numerically equal to m/s)",
    )
    axes[1].plot(x, predictions["imu_flight_time_s"], "o-", color="#E07A3F")
    axes[1].set(title="IMU-detected flight time", xlabel="Jump", ylabel="Seconds")
    fig.suptitle("CMJ IMU-only prediction")
    chart_path = output_dir / "imu_prediction_report.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    warnings = quality.get("warnings", [])
    warning_text = "<br>".join(html.escape(str(item)) for item in warnings) or "No automatic quality warnings."
    metrics = model_metadata.get("training_metrics") or {}
    metrics_text = (
        f"Grouped-CV MAE: {float(metrics['mae']):.4f}; "
        f"RMSE: {float(metrics['rmse']):.4f}; R²: {float(metrics['r2']):.4f}"
        if {"mae", "rmse", "r2"}.issubset(metrics)
        else "Training metrics are not embedded in this older model artifact."
    )
    prediction_cols = [
        "jump_id", "imu_takeoff_s", "imu_landing_s", "imu_flight_time_s",
        "predicted_normalized_impulse_ns_kg", "predicted_takeoff_velocity_m_s",
    ]
    if "predicted_net_impulse_ns" in predictions:
        prediction_cols.append("predicted_net_impulse_ns")
    predictions_html = predictions[prediction_cols].round(4).to_html(index=False, border=0)
    quality_rows = {
        key: value
        for key, value in quality.items()
        if key != "warnings" and not isinstance(value, (dict, list))
    }
    quality_html = pd.Series(quality_rows, name="value").to_frame().to_html(border=0)
    feature_set = html.escape(str(model_metadata.get("feature_set", "unknown")))
    validation = html.escape(str(model_metadata.get("validation_method", "unknown")))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMJ IMU-only prediction</title><style>
body{{font-family:Arial,sans-serif;max-width:1050px;margin:32px auto;color:#17212b;line-height:1.45}}
.warning{{background:#fff3cd;border-left:5px solid #e0a800;padding:12px 16px}} table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#eef4f8}}img{{width:100%;height:auto}}
</style></head><body><h1>CMJ IMU-only prediction report</h1>
<p class="warning"><strong>Prototype session-level model</strong><br>{warning_text}</p>
<p>Feature set: <strong>{feature_set}</strong><br>Validation: {validation}<br>{html.escape(metrics_text)}</p>
<p>Normalized impulse (N·s/kg) and takeoff velocity (m/s) have the same numeric value under the impulse–momentum formulation used for training. Raw N·s requires body mass.</p>
<img src="imu_prediction_report.png" alt="IMU-only prediction charts">
<h2>Predictions</h2>{predictions_html}<h2>IMU quality</h2>{quality_html}
<p>These predictions are not a substitute for athlete-level validation. Review warnings and detected events before using results.</p>
</body></html>"""
    html_path = output_dir / "imu_prediction_report.html"
    html_path.write_text(document, encoding="utf-8")
    return html_path, chart_path
