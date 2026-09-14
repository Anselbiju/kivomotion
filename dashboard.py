# dashboard.py
# ============================================================
# KIVO — Coach, Athlete and Sport Scientist Dashboard
# FastAPI + Uvicorn + Plotly
#
# Install:
#   pip install fastapi uvicorn pandas numpy plotly
#
# Run:
#   uvicorn dashboard:app --reload
#
# Or:
#   python dashboard.py
#
# Open:
#   http://127.0.0.1:8000
#
# Your preprocessing notebook should create:
#   ml_ready_outputs/clean_force_targets.csv
#   ml_ready_outputs/review_force_jumps_with_quality_status.csv
#   ml_ready_outputs/prediction_review_takeoff_velocity.csv
#   ml_ready_outputs/model_results_summary.csv
#   ml_ready_outputs/imu_force_shift_summary.csv
# ============================================================
from pathlib import Path
from html import escape
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse


# ============================================================
# CONFIG
# ============================================================

# CHANGE THIS ONLY IF YOUR OUTPUT FOLDER IS SOMEWHERE ELSE.
#
# Example:
# DATA_DIR = Path(r"D:\QUT\IndustryProject\ml_ready_outputs")
#
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ml_ready_outputs"

CLEAN_FILE = DATA_DIR / "clean_force_targets.csv"
QUALITY_FILE = DATA_DIR / "review_force_jumps_with_quality_status.csv"
PREDICTION_FILE = DATA_DIR / "prediction_review_takeoff_velocity.csv"
MODEL_FILE = DATA_DIR / "model_results_summary.csv"
SHIFT_FILE = DATA_DIR / "imu_force_shift_summary.csv"

G = 9.81

app = FastAPI(title="KIVO")


# ============================================================
# DATA
# ============================================================

def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def prepare_jump_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean event-level jump data and derive only metrics supported by source columns."""
    if df.empty:
        return df

    df = df.copy()

    # Canonicalise common column-name variants without inventing data.
    aliases = {
        "peak_force_N": ["peak_force_N", "force_peak_N", "max_force_N", "peak_force"],
        "mean_force_N": ["mean_force_N", "force_mean_N", "avg_force_N", "mean_force"],
        "peak_power_W": ["peak_power_W", "power_peak_W", "max_power_W", "peak_power"],
        "mean_power_W": ["mean_power_W", "power_mean_W", "avg_power_W", "mean_power"],
        "peak_acceleration_ms2": ["peak_acceleration_ms2", "imu_peak_acc_y_ms2", "imu_peak_acc_mag_ms2"],
        "mean_acceleration_ms2": ["mean_acceleration_ms2", "imu_mean_acc_y_ms2", "imu_mean_acc_mag_ms2"],
        "peak_velocity_ms": ["peak_velocity_ms", "imu_peak_velocity_ms"],
    }
    for canonical, candidates in aliases.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                df[canonical] = df[candidate]
                break

    numeric_cols = [
        "jump_id", "body_mass_kg", "onset_s", "braking_end_s", "takeoff_s",
        "landing_s", "flight_time_s", "braking_impulse_Ns", "propulsive_impulse_Ns",
        "total_impulse_Ns", "target_takeoff_velocity_ms", "flight_time_velocity_ms",
        "peak_force_N", "mean_force_N", "peak_power_W", "mean_power_W",
        "peak_acceleration_ms2", "mean_acceleration_ms2", "peak_velocity_ms",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean identifiers once so every URL/filter uses the same value.
    if "athlete_id" in df.columns:
        df["athlete_id"] = df["athlete_id"].astype(str).str.strip()
    if "trial_id" in df.columns:
        df["trial_id"] = df["trial_id"].astype(str).str.strip()

    if "target_takeoff_velocity_ms" in df.columns:
        df["jump_height_m"] = (df["target_takeoff_velocity_ms"] ** 2) / (2 * G)
        df["jump_height_cm"] = df["jump_height_m"] * 100

    if {"takeoff_s", "onset_s"}.issubset(df.columns):
        df["time_to_takeoff_s"] = df["takeoff_s"] - df["onset_s"]

    if {"jump_height_m", "time_to_takeoff_s"}.issubset(df.columns):
        valid_time = df["time_to_takeoff_s"].where(df["time_to_takeoff_s"] > 0)
        df["rsi_mod"] = df["jump_height_m"] / valid_time

    if {"braking_end_s", "onset_s"}.issubset(df.columns):
        df["braking_duration_ms"] = (df["braking_end_s"] - df["onset_s"]) * 1000

    if {"takeoff_s", "braking_end_s"}.issubset(df.columns):
        df["propulsive_duration_ms"] = (df["takeoff_s"] - df["braking_end_s"]) * 1000

    if "braking_impulse_Ns" in df.columns:
        df["braking_impulse_display_Ns"] = df["braking_impulse_Ns"].abs()

    # Body-mass-normalised metrics. They are only created when both inputs exist.
    if "body_mass_kg" in df.columns:
        mass = df["body_mass_kg"].where(df["body_mass_kg"] > 0)
        relative_pairs = {
            "peak_force_N": "relative_peak_force_N_kg",
            "mean_force_N": "relative_mean_force_N_kg",
            "total_impulse_Ns": "relative_total_impulse_Ns_kg",
            "braking_impulse_display_Ns": "relative_braking_impulse_Ns_kg",
            "propulsive_impulse_Ns": "relative_propulsive_impulse_Ns_kg",
            "peak_power_W": "relative_peak_power_W_kg",
            "mean_power_W": "relative_mean_power_W_kg",
        }
        for source, target in relative_pairs.items():
            if source in df.columns:
                df[target] = df[source] / mass

        if "relative_total_impulse_Ns_kg" in df.columns:
            df["relative_impulse_Ns_kg"] = df["relative_total_impulse_Ns_kg"]
        elif "relative_propulsive_impulse_Ns_kg" in df.columns:
            df["relative_impulse_Ns_kg"] = df["relative_propulsive_impulse_Ns_kg"]

    if {"braking_impulse_display_Ns", "propulsive_impulse_Ns"}.issubset(df.columns):
        denominator = df["propulsive_impulse_Ns"].replace(0, np.nan).abs()
        df["ecc_con_impulse_ratio"] = df["braking_impulse_display_Ns"] / denominator

    df["_order"] = np.arange(len(df))
    return df


def load_data():
    return {
        "clean": prepare_jump_data(
            read_csv_safe(CLEAN_FILE)
        ),
        "quality": prepare_jump_data(
            read_csv_safe(QUALITY_FILE)
        ),
        "prediction": read_csv_safe(PREDICTION_FILE),
        "models": read_csv_safe(MODEL_FILE),
        "shifts": read_csv_safe(SHIFT_FILE),
    }


# ============================================================
# GENERAL HELPERS
# ============================================================

def as_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def fmt(value, decimals=1, suffix=""):
    value = as_float(value)

    if not np.isfinite(value):
        return "—"

    return f"{value:.{decimals}f}{suffix}"


def normalize_id(value):
    """Return a clean ID for comparisons and URLs."""
    if value is None:
        return ""
    return str(value).strip()


def percent_change(current, previous):
    current = as_float(current)
    previous = as_float(previous)

    if (
        not np.isfinite(current)
        or not np.isfinite(previous)
        or previous == 0
    ):
        return np.nan

    return (
        (current - previous)
        / abs(previous)
        * 100
    )


def trend(change):
    change = as_float(change)

    if not np.isfinite(change):
        return '<span class="trend neutral">—</span>'

    if change > 2:
        return (
            f'<span class="trend positive">'
            f'↑ {abs(change):.1f}%'
            f'</span>'
        )

    if change < -2:
        return (
            f'<span class="trend negative">'
            f'↓ {abs(change):.1f}%'
            f'</span>'
        )

    return (
        f'<span class="trend neutral">'
        f'→ {abs(change):.1f}%'
        f'</span>'
    )


def latest_vs_previous(
    athlete_df: pd.DataFrame,
    metric: str,
):
    if athlete_df.empty:
        return np.nan, np.nan, np.nan

    if metric not in athlete_df.columns:
        return np.nan, np.nan, np.nan

    if "trial_id" not in athlete_df.columns:
        return np.nan, np.nan, np.nan

    d = athlete_df.copy()

    if "_order" in d.columns:
        d = d.sort_values("_order")
    else:
        d["_order"] = np.arange(len(d))

    d[metric] = pd.to_numeric(
        d[metric],
        errors="coerce",
    )

    d = d.dropna(
        subset=[metric]
    )

    if d.empty:
        return np.nan, np.nan, np.nan

    d["trial_id"] = (
        d["trial_id"]
        .astype(str)
        .str.strip()
    )

    trial_order = (
        d.groupby("trial_id")["_order"]
        .max()
        .sort_values()
        .index
        .tolist()
    )

    if not trial_order:
        return np.nan, np.nan, np.nan

    latest_trial = trial_order[-1]

    latest_values = d.loc[
        d["trial_id"] == latest_trial,
        metric,
    ].dropna()

    if latest_values.empty:
        return np.nan, np.nan, np.nan

    latest_value = latest_values.median()

    if len(trial_order) >= 2:

        previous_trial = trial_order[-2]

        previous_values = d.loc[
            d["trial_id"] == previous_trial,
            metric,
        ].dropna()

        if previous_values.empty:
            previous_value = np.nan
        else:
            previous_value = (
                previous_values.median()
            )

    else:
        previous_value = latest_value

    change = percent_change(
        latest_value,
        previous_value,
    )

    return (
        latest_value,
        previous_value,
        change,
    )

def build_athlete_summary(clean):
    rows = []
    if clean.empty:
        return pd.DataFrame()

    def med(g, col):
        return g[col].median() if col in g.columns else np.nan

    for athlete_id, g in clean.groupby("athlete_id"):
        g = g.sort_values("_order")

        jump_h, _, jump_change = latest_vs_previous(g, "jump_height_cm")
        rsi, _, rsi_change = latest_vs_previous(g, "rsi_mod")
        vel, _, vel_change = latest_vs_previous(g, "target_takeoff_velocity_ms")
        braking, _, braking_change = latest_vs_previous(g, "braking_impulse_display_Ns")
        propulsive, _, prop_change = latest_vs_previous(g, "propulsive_impulse_Ns")

        trial_order = g.groupby("trial_id")["_order"].max().sort_values().index.tolist()
        latest_trial = trial_order[-1] if trial_order else "—"
        latest_g = g[g["trial_id"] == latest_trial] if trial_order else g

        reasons = []
        if np.isfinite(jump_change) and jump_change <= -5:
            reasons.append(f"Jump height ↓ {abs(jump_change):.1f}%")
        if np.isfinite(rsi_change) and rsi_change <= -5:
            reasons.append(f"RSI-mod ↓ {abs(rsi_change):.1f}%")
        if np.isfinite(vel_change) and vel_change <= -5:
            reasons.append(f"Take-off velocity ↓ {abs(vel_change):.1f}%")

        latest_jump_values = latest_g["jump_height_cm"].dropna() if "jump_height_cm" in latest_g.columns else pd.Series(dtype=float)
        variability = np.nan
        if len(latest_jump_values) > 1 and latest_jump_values.mean() != 0:
            variability = latest_jump_values.std(ddof=1) / abs(latest_jump_values.mean()) * 100

        rows.append({
            "athlete_id": normalize_id(athlete_id),
            "latest_trial": latest_trial,
            "body_mass_kg": med(latest_g, "body_mass_kg"),
            "jump_height_cm": jump_h,
            "jump_height_change": jump_change,
            "rsi_mod": rsi,
            "rsi_change": rsi_change,
            "velocity_ms": vel,
            "velocity_change": vel_change,
            "peak_force_N": med(latest_g, "peak_force_N"),
            "relative_peak_force_N_kg": med(latest_g, "relative_peak_force_N_kg"),
            "mean_force_N": med(latest_g, "mean_force_N"),
            "total_impulse_Ns": med(latest_g, "total_impulse_Ns"),
            "relative_impulse_Ns_kg": med(latest_g, "relative_impulse_Ns_kg"),
            "peak_power_W": med(latest_g, "peak_power_W"),
            "relative_peak_power_W_kg": med(latest_g, "relative_peak_power_W_kg"),
            "braking_Ns": braking,
            "braking_change": braking_change,
            "propulsive_Ns": propulsive,
            "propulsive_change": prop_change,
            "variability_pct": variability,
            "status": "Needs attention" if reasons else "Stable",
            "reason": " · ".join(reasons),
            "jumps": len(g),
            "trials": g["trial_id"].nunique(),
        })

    return pd.DataFrame(rows)


def missing_data_html():
    return f"""
    <div class="empty">
        <div class="empty-icon">!</div>

        <h2>KIVO data not found</h2>

        <p>
            The dashboard is looking for:
        </p>

        <code>{escape(str(CLEAN_FILE))}</code>

        <p>
            Run your notebook first so that
            <b>clean_force_targets.csv</b>
            is created, or change
            <b>DATA_DIR</b> near the top of
            dashboard.py.
        </p>

        <p>
            Current absolute location:
        </p>

        <code>{escape(str(DATA_DIR.resolve()))}</code>
    </div>
    """



def metric_or_unavailable(df, column, decimals=1, suffix="", absolute=False):
    if df.empty or column not in df.columns:
        return "Unavailable"
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return "Unavailable"
    value = series.median()
    if absolute:
        value = abs(value)
    return fmt(value, decimals, suffix)


def unavailable_panel(title, message):
    return f"""
    <div class="panel unavailable-panel">
        <div class="panel-head"><div><div class="panel-title">{escape(title)}</div></div></div>
        <div class="panel-body"><div class="note">{escape(message)}</div></div>
    </div>
    """


def jump_strategy_text(df):
    if df.empty:
        return "Jump strategy is unavailable for this selection."
    if "braking_duration_ms" not in df.columns or "propulsive_duration_ms" not in df.columns:
        return "Jump strategy requires eccentric/braking and concentric/propulsive phase durations."
    ecc = pd.to_numeric(df["braking_duration_ms"], errors="coerce").median()
    con = pd.to_numeric(df["propulsive_duration_ms"], errors="coerce").median()
    if not np.isfinite(ecc) or not np.isfinite(con) or con <= 0:
        return "Jump strategy could not be calculated for this selection."
    ratio = ecc / con
    if ratio >= 1.15:
        return "Longer eccentric strategy – the athlete uses a longer countermovement period before the propulsive phase. Interpret this together with jump height, RSI-mod, force, impulse and velocity."
    if ratio <= 0.85:
        return "Shorter eccentric strategy – the athlete moves relatively quickly through the countermovement before the propulsive phase. Interpret this together with jump height, RSI-mod, force, impulse and velocity."
    return "Balanced phase-duration strategy – eccentric and concentric durations are relatively similar. This is a movement description, not a good/bad classification."


def statistics_table(df, metric_col="jump_height_cm", label="Jump Height", unit="cm", decimals=1):
    if df.empty or metric_col not in df.columns:
        return '<div class="note">Statistics unavailable for this metric.</div>'
    values = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if values.empty:
        return '<div class="note">Statistics unavailable for this metric.</div>'
    mean = values.mean(); minimum = values.min(); maximum = values.max()
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    variability = (std / abs(mean) * 100) if mean != 0 else np.nan
    best = maximum
    return f"""
    <div class="stats-grid">
        <div class="stat-box"><span>Mean</span><strong>{fmt(mean, decimals, ' '+unit if unit else '')}</strong></div>
        <div class="stat-box"><span>Minimum</span><strong>{fmt(minimum, decimals, ' '+unit if unit else '')}</strong></div>
        <div class="stat-box"><span>Maximum</span><strong>{fmt(maximum, decimals, ' '+unit if unit else '')}</strong></div>
        <div class="stat-box"><span>Std. Deviation</span><strong>{fmt(std, decimals, ' '+unit if unit else '')}</strong></div>
        <div class="stat-box"><span>Best Jump</span><strong>{fmt(best, decimals, ' '+unit if unit else '')}</strong></div>
        <div class="stat-box"><span>Between-jump variability</span><strong>{fmt(variability, 1, '%')}</strong></div>
    </div>
    """


def qc_flag_list(quality):
    if quality.empty:
        return '<div class="note">Data-quality file is unavailable.</div>'
    flag_terms = [
        ("quality_status", "Quality status"), ("braking_impulse_ok", "Braking impulse"),
        ("propulsive_impulse_ok", "Propulsive impulse"), ("flight_time_ok", "Flight time"),
        ("target_velocity_ok", "Take-off velocity"), ("alignment_warning", "Alignment warning"),
        ("noisy_imu", "Noisy IMU signal"), ("missing_phase", "Missing phase"),
        ("event_detection_issue", "Event-detection issue"),
    ]
    rows=[]
    for col,label in flag_terms:
        if col in quality.columns:
            vals=quality[col].astype(str).value_counts(dropna=False).head(5)
            text=" · ".join(f"{escape(str(k))}: {int(v)}" for k,v in vals.items())
            rows.append(f'<div class="summary-row"><span class="summary-label">{escape(label)}</span><span class="summary-value">{text}</span></div>')
    if not rows:
        return '<div class="note">No recognised QC flag columns were found. The component is ready for QC fields when they are exported.</div>'
    return '<div class="summary-list">'+''.join(rows)+'</div>'


def model_error_table(predictions, athlete_id=None):
    if predictions.empty:
        return '<div class="note">Prediction file unavailable.</div>'
    d=predictions.copy()
    if "athlete_id" in d.columns:
        d["athlete_id"]=d["athlete_id"].astype(str).str.strip()
        if athlete_id:
            d=d[d["athlete_id"]==normalize_id(athlete_id)]
    actual="target_takeoff_velocity_ms"; pred="predicted_takeoff_velocity_ms"
    if actual not in d.columns or pred not in d.columns:
        return '<div class="note">Predicted and ground-truth velocity columns are unavailable.</div>'
    d[actual]=pd.to_numeric(d[actual],errors="coerce"); d[pred]=pd.to_numeric(d[pred],errors="coerce")
    d=d.dropna(subset=[actual,pred]).copy()
    if d.empty:
        return '<div class="note">No prediction rows available for this selection.</div>'
    d["abs_error"]=(d[pred]-d[actual]).abs()
    d["pct_error"]=np.where(d[actual].abs()>1e-9,d["abs_error"]/d[actual].abs()*100,np.nan)
    rows=[]
    for _,r in d.tail(12).iterrows():
        athlete=escape(normalize_id(r.get("athlete_id","—")))
        jump=escape(str(r.get("jump_id","—")))
        rows.append(f"<tr><td>{athlete}</td><td>{jump}</td><td>{fmt(r[actual],2,' m/s')}</td><td>{fmt(r[pred],2,' m/s')}</td><td>{fmt(r['abs_error'],2,' m/s')}</td><td>{fmt(r['pct_error'],1,'%')}</td></tr>")
    return f"""<div class="table-wrap"><table><thead><tr><th>Athlete</th><th>Jump</th><th>ForceDeck</th><th>Predicted</th><th>Absolute Error</th><th>% Error</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""


def signal_candidates():
    """Find optional sample-level CSVs without assuming they exist."""
    candidates=[]
    if not DATA_DIR.exists():
        return candidates
    for path in DATA_DIR.glob("*.csv"):
        if path in {CLEAN_FILE,QUALITY_FILE,PREDICTION_FILE,MODEL_FILE,SHIFT_FILE}:
            continue
        try:
            header=pd.read_csv(path,nrows=0)
            cols={str(c).lower() for c in header.columns}
            has_time=any(c in cols for c in ["time","time_s","aligned_time_s","timestamp","elapsed_s"])
            has_signal=any(any(k in c for k in ["force","acc","velocity","power"]) for c in cols)
            if has_time and has_signal:
                candidates.append(path)
        except Exception:
            pass
    return candidates


def load_signal_data():
    frames=[]
    for path in signal_candidates()[:8]:
        try:
            d=pd.read_csv(path)
            d["_source_file"]=path.name
            if "athlete_id" in d.columns:
                d["athlete_id"]=d["athlete_id"].astype(str).str.strip()
            if "trial_id" in d.columns:
                d["trial_id"]=d["trial_id"].astype(str).str.strip()
            frames.append(d)
        except Exception:
            continue
    return frames


def _find_col(df, names, contains=()):
    for n in names:
        if n in df.columns:
            return n
    for c in df.columns:
        low=str(c).lower()
        if any(k in low for k in contains):
            return c
    return None


def signal_chart(athlete_id, trial_id, jump_id, signal_kind, event_row=None):
    frames=load_signal_data()
    kind_map={
        "force": (["net_force_N","force_N","force"], ("force",), "Force-Time Curve", "Force (N)"),
        "acceleration": (["vertical_acc_ms2","acc_filt_ms2","linacc_y_filt_ms2","linacc_y_ms2"], ("acc",), "IMU Acceleration-Time Curve", "Acceleration (m/s²)"),
        "velocity": (["velocity_ms","vertical_velocity_ms","imu_velocity_ms"], ("velocity",), "Velocity-Time Curve", "Velocity (m/s)"),
    }
    names,contains,title,ylabel=kind_map[signal_kind]
    for raw in frames:
        d=raw.copy()
        if "athlete_id" in d.columns:
            d=d[d["athlete_id"]==normalize_id(athlete_id)]
        if trial_id and "trial_id" in d.columns:
            d=d[d["trial_id"]==str(trial_id).strip()]
        if jump_id is not None and "jump_id" in d.columns:
            d=d[pd.to_numeric(d["jump_id"],errors="coerce")==float(jump_id)]
        if d.empty:
            continue
        tcol=_find_col(d,["aligned_time_s","time_s","Time","time","elapsed_s"],("time",))
        ycol=_find_col(d,names,contains)
        if not tcol or not ycol:
            continue
        x=pd.to_numeric(d[tcol],errors="coerce"); y=pd.to_numeric(d[ycol],errors="coerce")
        good=x.notna() & y.notna()
        if good.sum()<3:
            continue
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=x[good],y=y[good],mode="lines",name=ylabel,hovertemplate=f"Time: %{{x:.3f}} s<br>{ylabel}: %{{y:.2f}}<extra></extra>"))
        if event_row is not None:
            marker_info=[("onset_s","Start movement"),("braking_end_s","Start concentric"),("takeoff_s","Take-off"),("landing_s","Landing")]
            for col,label in marker_info:
                if col in event_row.index and np.isfinite(as_float(event_row[col])):
                    fig.add_vline(x=float(event_row[col]),line_dash="dash",annotation_text=label,annotation_position="top")
        fig.update_layout(title=dict(text=title,font=dict(color="white",size=16)),xaxis_title="Time (s)",yaxis_title=ylabel,showlegend=False)
        return plot_html(fig,360)
    return f'<div class="chart-empty">{escape(title)} unavailable. No sample-level CSV with a compatible time and {escape(signal_kind)} column was found in ml_ready_outputs.</div>'


# ============================================================
# PLOTLY
# ============================================================

def plot_html(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0d1626",
        plot_bgcolor="#0d1626",
        font=dict(
            family="Inter, Arial, sans-serif",
            color="#aebdd0",
        ),
        margin=dict(
            l=45,
            r=25,
            t=55,
            b=45,
        ),
        xaxis=dict(
            gridcolor="#1a2940",
            zerolinecolor="#1a2940",
            linecolor="#22334d",
        ),
        yaxis=dict(
            gridcolor="#1a2940",
            zerolinecolor="#1a2940",
            linecolor="#22334d",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#101c30",
            font_color="white",
        ),
    )

    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )


def squad_comparison_chart(
    summary,
    metric="jump_height",
):
    metric_map = {
        "jump_height": (
            "jump_height_cm",
            "Jump Height",
            "cm",
        ),
        "rsi": (
            "rsi_mod",
            "RSI-mod",
            "",
        ),
        "velocity": (
            "velocity_ms",
            "Take-off Velocity",
            "m/s",
        ),
        "braking": (
            "braking_Ns",
            "Braking Impulse",
            "N·s",
        ),
        "propulsive": (
            "propulsive_Ns",
            "Propulsive Impulse",
            "N·s",
        ),
    }

    column, label, unit = metric_map.get(
        metric,
        metric_map["jump_height"],
    )

    d = (
        summary[["athlete_id", column]]
        .dropna()
        .sort_values(column)
    )

    if d.empty:
        return (
            '<div class="chart-empty">'
            'No values available.'
            '</div>'
        )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=d[column],
            y=d["athlete_id"],
            orientation="h",
            marker=dict(
                color=d[column],
                colorscale=[
                    [0, "#705cff"],
                    [0.55, "#1daada"],
                    [1, "#20d7a7"],
                ],
            ),
            text=[
                f"{v:.2f} {unit}".strip()
                for v in d[column]
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + label
                + ": %{x:.2f} "
                + unit
                + "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Squad {label}",
            font=dict(
                color="white",
                size=16,
            ),
        ),
        xaxis_title=unit or label,
        yaxis_title=None,
        showlegend=False,
    )

    height = max(
        330,
        45 * len(d) + 100,
    )

    return plot_html(fig, height)



def athlete_quadrant_chart(clean, athlete_a="", athlete_b=""):
    required={"athlete_id","jump_id","rsi_mod","jump_height_cm"}
    if clean.empty or not required.issubset(clean.columns):
        return '<div class="chart-empty">RSI-mod / jump-height quadrant data unavailable.</div>'
    d=clean[list(required)].copy()
    d["rsi_mod"]=pd.to_numeric(d["rsi_mod"],errors="coerce")
    d["jump_height_cm"]=pd.to_numeric(d["jump_height_cm"],errors="coerce")
    d=d.dropna(subset=["rsi_mod","jump_height_cm"])
    if d.empty:
        return '<div class="chart-empty">No valid RSI-mod / jump-height points.</div>'
    xmid=d["rsi_mod"].median(); ymid=d["jump_height_cm"].median()
    a=normalize_id(athlete_a); b=normalize_id(athlete_b)
    fig=go.Figure()
    others=d[~d["athlete_id"].isin([a,b])]
    if not others.empty:
        fig.add_trace(go.Scatter(x=others["rsi_mod"],y=others["jump_height_cm"],mode="markers",name="Squad",marker=dict(size=8,opacity=.35,color="#71849e"),customdata=np.stack([others["athlete_id"].astype(str),others["jump_id"].astype(str)],axis=-1),hovertemplate="<b>%{customdata[0]}</b> · Jump %{customdata[1]}<br>RSI-mod: %{x:.2f}<br>Jump height: %{y:.1f} cm<extra></extra>"))
    for athlete,label,color in [(a,"Athlete A","#22d3ee"),(b,"Athlete B","#ff9f43")]:
        if athlete:
            g=d[d["athlete_id"]==athlete]
            if not g.empty:
                fig.add_trace(go.Scatter(x=g["rsi_mod"],y=g["jump_height_cm"],mode="markers",name=f"{label}: {athlete}",marker=dict(size=13,color=color,line=dict(width=2,color="#07101d")),customdata=np.stack([g["athlete_id"].astype(str),g["jump_id"].astype(str)],axis=-1),hovertemplate="<b>%{customdata[0]}</b> · Jump %{customdata[1]}<br>RSI-mod: %{x:.2f}<br>Jump height: %{y:.1f} cm<extra></extra>"))
    fig.add_vline(x=xmid,line_dash="dash",line_color="#705cff")
    fig.add_hline(y=ymid,line_dash="dash",line_color="#705cff")
    fig.add_annotation(x=0.99,y=0.98,xref="paper",yref="paper",text="High RSI + High Jump",showarrow=False,font=dict(size=10,color="#20d7a7"))
    fig.add_annotation(x=0.99,y=0.03,xref="paper",yref="paper",text="High RSI + Low Jump",showarrow=False,font=dict(size=10,color="#aebdd0"))
    fig.add_annotation(x=0.01,y=0.98,xref="paper",yref="paper",text="Low RSI + High Jump",showarrow=False,font=dict(size=10,color="#aebdd0"),xanchor="left")
    fig.add_annotation(x=0.01,y=0.03,xref="paper",yref="paper",text="Low RSI + Low Jump",showarrow=False,font=dict(size=10,color="#ff9baf"),xanchor="left")
    fig.update_layout(title=dict(text="Athlete Comparison Quadrant",font=dict(color="white",size=16)),xaxis_title="RSI Modified",yaxis_title="Jump Height (cm)")
    return plot_html(fig,470)


def squad_trend_chart(clean):
    if clean.empty:
        return (
            '<div class="chart-empty">'
            'No trend data.'
            '</div>'
        )

    d = (
        clean.groupby("trial_id")
        .agg(
            jump_height=(
                "jump_height_cm",
                "mean",
            ),
            order=(
                "_order",
                "max",
            ),
        )
        .reset_index()
        .sort_values("order")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d["trial_id"],
            y=d["jump_height"],
            mode="lines+markers",
            line=dict(
                color="#22d3ee",
                width=3,
            ),
            marker=dict(
                color="#22d3ee",
                size=8,
            ),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,.05)",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Squad avg: %{y:.1f} cm"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Squad Performance Trend",
            font=dict(
                color="white",
                size=16,
            ),
        ),
        xaxis_title="Trial",
        yaxis_title="Average jump height (cm)",
        showlegend=False,
    )

    return plot_html(fig, 335)


def athlete_history_chart(
    athlete_df,
    metric,
):
    metric_map = {
        "jump_height": (
            "jump_height_cm",
            "Jump Height",
            "cm",
            1,
        ),
        "rsi": (
            "rsi_mod",
            "RSI-mod",
            "",
            2,
        ),
        "velocity": (
            "target_takeoff_velocity_ms",
            "Take-off Velocity",
            "m/s",
            2,
        ),
        "braking": (
            "braking_impulse_display_Ns",
            "Braking Impulse",
            "N·s",
            1,
        ),
        "propulsive": (
            "propulsive_impulse_Ns",
            "Propulsive Impulse",
            "N·s",
            1,
        ),
        "flight": (
            "flight_time_s",
            "Flight Time",
            "s",
            3,
        ),
    }

    column, label, unit, decimals = metric_map.get(
        metric,
        metric_map["jump_height"],
    )

    if athlete_df.empty:
        return (
            '<div class="chart-empty">'
            'No athlete history available.'
            '</div>'
        )

    if column not in athlete_df.columns:
        return (
            '<div class="chart-empty">'
            'Metric not available.'
            '</div>'
        )

    d = athlete_df.copy()

    if "_order" in d.columns:
        d = d.sort_values("_order")

    d[column] = pd.to_numeric(
        d[column],
        errors="coerce",
    )

    if "jump_id" in d.columns:
        d["jump_id"] = pd.to_numeric(
            d["jump_id"],
            errors="coerce",
        )

    d = d.dropna(
        subset=[column]
    )

    print("ATHLETE HISTORY DEBUG")
    print("Metric:", column)
    print("Number of rows:", len(d))

    if "trial_id" in d.columns:
        print("Trials:", d["trial_id"].tolist())

    if "jump_id" in d.columns:
        print("Jump IDs:", d["jump_id"].tolist())

    print("Values:", d[column].tolist())

    if d.empty:
        return (
            '<div class="chart-empty">'
            'No valid history values available.'
            '</div>'
        )

    labels = []

    for index, row in d.iterrows():

        trial = str(
            row.get(
                "trial_id",
                "Session",
            )
        ).strip()

        jump_value = row.get(
            "jump_id",
            np.nan,
        )

        if np.isfinite(
            as_float(jump_value)
        ):
            jump_text = (
                f"J{int(float(jump_value))}"
            )
        else:
            jump_text = "Jump"

        labels.append(
            f"{trial} · {jump_text}"
        )

    d["jump_label"] = labels

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d["jump_label"],
            y=d[column],
            mode="lines+markers",
            line=dict(
                color="#22d3ee",
                width=3,
            ),
            marker=dict(
                color="#22d3ee",
                size=10,
                line=dict(
                    color="#07101d",
                    width=2,
                ),
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                + label
                + ": %{y:."
                + str(decimals)
                + "f} "
                + unit
                + "<extra></extra>"
            ),
        )
    )

    if len(d) > 1:

        average = d[column].mean()

        if np.isfinite(average):

            fig.add_hline(
                y=average,
                line_dash="dash",
                line_color="#705cff",
                annotation_text=(
                    f"Average "
                    f"{average:.{decimals}f}"
                ),
                annotation_font_color=(
                    "#a99cff"
                ),
            )

    fig.update_layout(
        title=dict(
            text=f"{label} History",
            font=dict(
                color="white",
                size=16,
            ),
        ),
        xaxis_title=(
            "Session / Jump"
        ),
        yaxis_title=(
            f"{label} ({unit})"
            if unit
            else label
        ),
        showlegend=False,
    )

    return plot_html(
        fig,
        370,
    )

def latest_jump_chart(athlete_df):

    if athlete_df.empty:
        return (
            '<div class="chart-empty">'
            'No jumps available.'
            '</div>'
        )

    if "trial_id" not in athlete_df.columns:
        return (
            '<div class="chart-empty">'
            'Trial information unavailable.'
            '</div>'
        )

    if "jump_height_cm" not in athlete_df.columns:
        return (
            '<div class="chart-empty">'
            'Jump height unavailable.'
            '</div>'
        )

    d = athlete_df.copy()

    if "_order" in d.columns:
        trial_order = (
            d.groupby("trial_id")["_order"]
            .max()
            .sort_values()
            .index
            .tolist()
        )
    else:
        trial_order = (
            d["trial_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    if not trial_order:
        return (
            '<div class="chart-empty">'
            'No jumps available.'
            '</div>'
        )

    latest_trial = trial_order[-1]

    d = d[
        d["trial_id"].astype(str)
        == str(latest_trial)
    ].copy()

    if d.empty:
        return (
            '<div class="chart-empty">'
            'No recent jumps available.'
            '</div>'
        )

    d["jump_height_cm"] = pd.to_numeric(
        d["jump_height_cm"],
        errors="coerce",
    )

    if "jump_id" in d.columns:
        d["jump_id"] = pd.to_numeric(
            d["jump_id"],
            errors="coerce",
        )

        d = d.sort_values(
            "jump_id"
        )

    d = d.dropna(
        subset=["jump_height_cm"]
    )

    if d.empty:
        return (
            '<div class="chart-empty">'
            'No valid jump-height values.'
            '</div>'
        )

    jump_labels = []

    for _, row in d.iterrows():

        jump_value = row.get(
            "jump_id",
            np.nan,
        )

        if np.isfinite(
            as_float(jump_value)
        ):
            jump_labels.append(
                f"J{int(float(jump_value))}"
            )
        else:
            jump_labels.append(
                "Jump"
            )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=jump_labels,
            y=d["jump_height_cm"],
            marker=dict(
                color="#705cff",
            ),
            text=[
                f"{value:.1f}"
                for value in d["jump_height_cm"]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Jump Height: %{y:.1f} cm"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Recent Jumps — "
                f"{latest_trial}"
            ),
            font=dict(
                color="white",
                size=16,
            ),
        ),
        xaxis_title="Jump",
        yaxis_title="Jump Height (cm)",
        showlegend=False,
    )

    return plot_html(
        fig,
        300,
    )
def phase_duration_chart(latest_trial_df):
    if (
        "braking_duration_ms"
        not in latest_trial_df.columns
        or "propulsive_duration_ms"
        not in latest_trial_df.columns
    ):
        return (
            '<div class="chart-empty">'
            'Phase duration data unavailable.'
            '</div>'
        )

    d = latest_trial_df.sort_values(
        "jump_id"
    )

    labels = [
        f"Jump {int(v)}"
        for v in d["jump_id"]
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=labels,
            x=d["braking_duration_ms"],
            orientation="h",
            name="Braking",
            marker_color="#ff9f43",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Braking: %{x:.0f} ms"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            y=labels,
            x=d["propulsive_duration_ms"],
            orientation="h",
            name="Propulsive",
            marker_color="#22d3ee",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Propulsive: %{x:.0f} ms"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="Braking & Propulsive Phase Duration",
            font=dict(
                color="white",
                size=16,
            ),
        ),
        xaxis_title="Duration (ms)",
        yaxis_title=None,
        yaxis=dict(
            autorange="reversed",
        ),
    )

    return plot_html(
        fig,
        max(
            310,
            55 * len(d) + 100,
        ),
    )


def quality_chart(quality):
    if (
        quality.empty
        or "quality_status"
        not in quality.columns
    ):
        return (
            '<div class="chart-empty">'
            'Quality file not available.'
            '</div>'
        )

    accepted = int(
        (
            quality["quality_status"]
            == "ACCEPTED"
        ).sum()
    )

    total = len(quality)
    flagged = total - accepted

    fig = go.Figure(
        go.Pie(
            labels=[
                "Accepted",
                "Flagged / rejected",
            ],
            values=[
                accepted,
                flagged,
            ],
            hole=.68,
            marker=dict(
                colors=[
                    "#20d7a7",
                    "#ff6b81",
                ],
            ),
            hovertemplate=(
                "%{label}: %{value}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Data Quality",
            font=dict(
                color="white",
                size=16,
            ),
        ),
        annotations=[
            dict(
                text=(
                    f"<b>{accepted}</b>"
                    f"<br>accepted"
                ),
                x=.5,
                y=.5,
                showarrow=False,
                font=dict(
                    color="white",
                    size=15,
                ),
            )
        ],
    )

    return plot_html(fig, 330)


def prediction_chart(
    predictions,
    athlete_id=None,
):
    if predictions.empty:
        return (
            '<div class="chart-empty">'
            'Prediction file not available.'
            '</div>'
        )

    d = predictions.copy()

    if (
        athlete_id is not None
        and "athlete_id" in d.columns
    ):
        selected_id = normalize_id(athlete_id)
        d = d[
            d["athlete_id"]
            .astype(str)
            .str.strip()
            == selected_id
        ]

    x = "target_takeoff_velocity_ms"
    y = "predicted_takeoff_velocity_ms"

    if (
        d.empty
        or x not in d.columns
        or y not in d.columns
    ):
        return (
            '<div class="chart-empty">'
            'Prediction columns unavailable.'
            '</div>'
        )

    d[x] = pd.to_numeric(
        d[x],
        errors="coerce",
    )

    d[y] = pd.to_numeric(
        d[y],
        errors="coerce",
    )

    d = d.dropna(subset=[x, y])

    if d.empty:
        return (
            '<div class="chart-empty">'
            'No prediction rows.'
            '</div>'
        )

    minimum = min(
        d[x].min(),
        d[y].min(),
    )

    maximum = max(
        d[x].max(),
        d[y].max(),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d[x],
            y=d[y],
            mode="markers",
            name="Jumps",
            marker=dict(
                color="#22d3ee",
                size=9,
                opacity=.82,
            ),
            hovertemplate=(
                "Measured: %{x:.2f} m/s<br>"
                "Predicted: %{y:.2f} m/s"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[
                minimum,
                maximum,
            ],
            y=[
                minimum,
                maximum,
            ],
            mode="lines",
            name="Perfect agreement",
            line=dict(
                color="#705cff",
                dash="dash",
            ),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=dict(
            text="Measured vs Predicted Take-off Velocity",
            font=dict(
                color="white",
                size=16,
            ),
        ),
        xaxis_title="Measured velocity (m/s)",
        yaxis_title="Predicted velocity (m/s)",
    )

    return plot_html(fig, 350)


# ============================================================
# STYLE / APP LAYOUT
# ============================================================

CSS = """
:root {
    --bg:#070d18;
    --sidebar:#08111f;
    --panel:#0d1626;
    --panel2:#101b2d;
    --border:#1a2940;
    --text:#f4f7fb;
    --muted:#71849e;
    --cyan:#22d3ee;
    --purple:#705cff;
    --green:#20d7a7;
    --red:#ff6b81;
    --orange:#ff9f43;
}

* {
    box-sizing:border-box;
}

html,body {
    margin:0;
    min-height:100%;
    background:var(--bg);
    color:var(--text);
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

a {
    color:inherit;
    text-decoration:none;
}

.app {
    min-height:100vh;
    display:grid;
    grid-template-columns:235px minmax(0,1fr);
}

.sidebar {
    position:sticky;
    top:0;
    height:100vh;
    padding:27px 15px;
    background:
        radial-gradient(
            circle at 10% 2%,
            rgba(34,211,238,.08),
            transparent 26%
        ),
        var(--sidebar);
    border-right:1px solid #142139;
}

.brand {
    display:flex;
    align-items:center;
    gap:10px;
    padding:0 10px 29px;
    font-size:23px;
    font-weight:850;
}

.brand-mark {
    width:31px;
    height:31px;
    border-radius:9px;
    display:grid;
    place-items:center;
    color:#07111c;
    background:
        linear-gradient(
            135deg,
            var(--cyan),
            var(--purple)
        );
}

.nav-label {
    padding:0 12px 8px;
    color:#50627c;
    font-size:9px;
    font-weight:850;
    letter-spacing:1.2px;
    text-transform:uppercase;
}

.nav {
    display:flex;
    align-items:center;
    gap:10px;
    margin:4px 0;
    padding:11px 12px;
    border-radius:9px;
    color:#8395ae;
    font-size:12px;
    font-weight:700;
}

.nav:hover {
    color:white;
    background:#101c2f;
}

.nav.active {
    color:#e7fbff;
    border:1px solid rgba(34,211,238,.12);
    background:
        linear-gradient(
            90deg,
            rgba(34,211,238,.13),
            rgba(112,92,255,.06)
        );
}

.nav-icon {
    width:18px;
    text-align:center;
}

.sidebar-foot {
    position:absolute;
    left:17px;
    right:17px;
    bottom:25px;
    border:1px solid #17263c;
    background:#0b1524;
    padding:12px;
    border-radius:9px;
    color:#61748d;
    font-size:10px;
    line-height:1.6;
}

.live-dot {
    display:inline-block;
    width:7px;
    height:7px;
    margin-right:6px;
    border-radius:50%;
    background:var(--green);
    box-shadow:
        0 0 12px rgba(32,215,167,.8);
}

.main {
    min-width:0;
}

.topbar {
    height:75px;
    padding:16px 29px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    position:sticky;
    top:0;
    z-index:10;
    border-bottom:1px solid #162239;
    background:rgba(7,13,24,.92);
    backdrop-filter:blur(10px);
}

.title {
    font-size:20px;
    font-weight:800;
}

.subtitle {
    margin-top:5px;
    color:#71839a;
    font-size:10px;
}

.live {
    color:#71839a;
    font-size:10px;
}

.content {
    max-width:1650px;
    margin:auto;
    padding:25px 29px 50px;
}

.kpis {
    display:grid;
    grid-template-columns:
        repeat(4,minmax(0,1fr));
    gap:13px;
    margin-bottom:15px;
}

.kpi {
    min-height:103px;
    padding:16px;
    position:relative;
    overflow:hidden;
    border:1px solid #1b2a40;
    border-radius:10px;
    background:
        linear-gradient(
            180deg,
            #101b2d,
            #0b1524
        );
}

.kpi:after {
    content:"";
    position:absolute;
    left:0;
    right:0;
    bottom:0;
    height:2px;
    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(34,211,238,.35),
            transparent
        );
}

.kpi-label {
    color:#6d8099;
    font-size:9px;
    font-weight:850;
    letter-spacing:.8px;
    text-transform:uppercase;
}

.kpi-value {
    margin-top:9px;
    font-size:27px;
    font-weight:850;
    letter-spacing:-.8px;
}

.kpi-foot {
    display:flex;
    align-items:center;
    gap:7px;
    margin-top:7px;
    color:#687a91;
    font-size:9px;
}

.trend {
    font-size:9px;
    font-weight:850;
}

.trend.positive {
    color:var(--green);
}

.trend.negative {
    color:var(--red);
}

.trend.neutral {
    color:#91a0b5;
}

.grid-main {
    display:grid;
    grid-template-columns:
        minmax(0,1.55fr)
        minmax(330px,.85fr);
    gap:15px;
}

.grid-2 {
    display:grid;
    grid-template-columns:
        repeat(2,minmax(0,1fr));
    gap:15px;
}

.panel {
    border:1px solid #1a2940;
    border-radius:10px;
    overflow:hidden;
    background:
        linear-gradient(
            180deg,
            #0e1829,
            #0a1321
        );
}

.panel-head {
    padding:14px 16px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:14px;
    border-bottom:1px solid #19273c;
}

.panel-title {
    color:#dfe8f4;
    font-size:11px;
    font-weight:850;
    letter-spacing:.5px;
    text-transform:uppercase;
}

.panel-sub {
    margin-top:4px;
    color:#61748d;
    font-size:9px;
}

.panel-body {
    padding:16px;
}

.tabs {
    display:flex;
    flex-wrap:wrap;
    gap:5px;
}

.tab {
    padding:5px 8px;
    border-radius:7px;
    border:1px solid #21324b;
    color:#788ba4;
    background:#0b1524;
    font-size:9px;
    font-weight:800;
}

.tab:hover {
    color:white;
}

.tab.active {
    color:#dffbff;
    border-color:rgba(34,211,238,.38);
    background:rgba(34,211,238,.08);
}

.table-wrap {
    overflow:auto;
}

table {
    width:100%;
    border-collapse:collapse;
    min-width:780px;
}

th {
    padding:10px 13px;
    text-align:left;
    border-bottom:1px solid #1a2940;
    color:#5f728c;
    background:#0a1423;
    font-size:8px;
    font-weight:850;
    letter-spacing:.6px;
    text-transform:uppercase;
}

td {
    padding:11px 13px;
    border-bottom:1px solid #152338;
    color:#b9c6d7;
    font-size:10px;
}

tr.clickable {
    cursor:pointer;
}

tr.clickable:hover {
    background:rgba(34,211,238,.04);
}

.athlete-name {
    color:white;
    font-weight:800;
}

.meta {
    margin-top:3px;
    color:#60738c;
    font-size:8px;
}

.status {
    display:inline-flex;
    align-items:center;
    gap:5px;
    padding:4px 7px;
    border-radius:99px;
    font-size:8px;
    font-weight:850;
}

.status.good {
    color:#8bf0d2;
    border:1px solid rgba(32,215,167,.18);
    background:rgba(32,215,167,.08);
}

.status.alert {
    color:#ff9baf;
    border:1px solid rgba(255,107,129,.18);
    background:rgba(255,107,129,.08);
}

.attention {
    padding:8px 14px 14px;
}

.attention-item {
    display:block;
    margin-top:8px;
    padding:12px;
    border-radius:9px;
    border:1px solid #1a2a40;
    background:#0a1525;
}

.attention-item:hover {
    border-color:rgba(255,107,129,.35);
}

.attention-top {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:8px;
}

.attention-name {
    color:white;
    font-size:10px;
    font-weight:850;
}

.attention-reason {
    margin-top:6px;
    color:#7c8fa8;
    font-size:9px;
    line-height:1.5;
}

.section {
    display:flex;
    align-items:end;
    justify-content:space-between;
    gap:20px;
    margin:23px 0 12px;
}

.section-title {
    color:#dbe5f2;
    font-size:11px;
    font-weight:850;
    letter-spacing:.6px;
    text-transform:uppercase;
}

.section-note {
    margin-top:4px;
    color:#60738c;
    font-size:9px;
}

.breadcrumb {
    display:flex;
    gap:7px;
    margin-bottom:17px;
    color:#6a7d95;
    font-size:10px;
}

.breadcrumb a:hover {
    color:var(--cyan);
}

.athlete-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    margin-bottom:16px;
}

.athlete-title {
    font-size:24px;
    font-weight:850;
}

.athlete-sub {
    margin-top:4px;
    color:#71849c;
    font-size:10px;
}

.btn {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    padding:7px 11px;
    border:1px solid #24405f;
    border-radius:8px;
    color:#bdebf4;
    background:#0f2037;
    font-size:9px;
    font-weight:800;
}

.btn:hover {
    border-color:rgba(34,211,238,.45);
}

.phase-flow {
    padding:18px 12px;
    display:grid;
    grid-template-columns:
        1fr 20px
        1fr 20px
        1fr 20px
        1fr 20px
        1fr;
    align-items:center;
}

.phase {
    min-height:88px;
    padding:13px 8px;
    text-align:center;
    border:1px solid #1d3049;
    border-radius:9px;
    background:#0a1524;
}

.phase-name {
    color:#8496ad;
    font-size:8px;
    font-weight:850;
    text-transform:uppercase;
}

.phase-value {
    margin-top:9px;
    color:white;
    font-size:15px;
    font-weight:850;
}

.phase-sub {
    margin-top:4px;
    color:#5c7089;
    font-size:8px;
}

.phase-arrow {
    text-align:center;
    color:#4f6481;
    font-weight:900;
}

.summary-list {
    padding:8px 15px 15px;
}

.summary-row {
    padding:10px 0;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:15px;
    border-bottom:1px solid #17263a;
    font-size:10px;
}

.summary-row:last-child {
    border-bottom:none;
}

.summary-label {
    color:#8294ab;
}

.summary-value {
    color:white;
    font-weight:800;
}

.note {
    padding:12px 14px;
    border:1px solid rgba(34,211,238,.15);
    border-radius:9px;
    color:#8498b0;
    background:rgba(34,211,238,.035);
    font-size:9px;
    line-height:1.6;
}

.models {
    display:grid;
    grid-template-columns:
        repeat(3,minmax(0,1fr));
    gap:11px;
}

.model {
    padding:13px;
    border:1px solid #1b2d44;
    border-radius:9px;
    background:#0a1524;
}

.model-name {
    color:white;
    font-size:10px;
    font-weight:850;
}

.model-values {
    margin-top:7px;
    color:#72859e;
    font-size:9px;
    line-height:1.8;
}

.empty {
    max-width:700px;
    margin:75px auto;
    padding:32px;
    text-align:center;
    border:1px solid #1e3049;
    border-radius:12px;
    background:#0c1727;
}

.empty h2 {
    margin:13px 0;
}

.empty p {
    color:#7b8ea6;
    font-size:11px;
    line-height:1.7;
}

.empty code {
    display:block;
    margin:10px 0;
    color:#b9f6ff;
}

.empty-icon {
    width:48px;
    height:48px;
    margin:auto;
    display:grid;
    place-items:center;
    border-radius:50%;
    color:#07111b;
    background:var(--cyan);
    font-size:20px;
    font-weight:900;
}

.chart-empty {
    padding:50px 20px;
    color:#6f829b;
    text-align:center;
    font-size:10px;
}
/* ============================================================
   ATHLETE SELECTION PAGE
   ============================================================ */

.athlete-select-grid {
    display:grid;
    grid-template-columns:
        repeat(3,minmax(0,1fr));
    gap:14px;
}


.athlete-select-card {
    display:block;
    padding:17px;
    border:1px solid #1b2b42;
    border-radius:10px;

    background:
        linear-gradient(
            180deg,
            #101b2d,
            #0a1423
        );

    transition:
        transform .15s ease,
        border-color .15s ease,
        background .15s ease;
}


.athlete-select-card:hover {
    transform:translateY(-2px);

    border-color:
        rgba(34,211,238,.45);

    background:#101d30;
}


.athlete-select-top {
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:10px;
}


.athlete-select-name {
    color:white;
    font-size:19px;
    font-weight:850;
}


.athlete-select-meta {
    color:#687b94;
    font-size:9px;
    margin-top:5px;
}


.athlete-select-metrics {
    display:grid;
    grid-template-columns:
        repeat(2,1fr);
    gap:10px;
    margin-top:18px;
}


.athlete-select-metrics div {
    padding:9px;
    border-radius:7px;
    background:#091320;
    border:1px solid #17283e;
}


.athlete-select-metrics span {
    display:block;
    color:#657a94;
    font-size:8px;
    text-transform:uppercase;
    font-weight:800;
}


.athlete-select-metrics strong {
    display:block;
    margin-top:5px;
    color:#eaf4ff;
    font-size:12px;
}


.athlete-select-open {
    margin-top:16px;
    padding-top:12px;
    border-top:1px solid #17273d;
    color:#22d3ee;
    font-size:9px;
    font-weight:850;
    text-align:right;
}



.compare-form {display:flex;flex-wrap:wrap;gap:10px;align-items:end;padding:14px 16px;border-bottom:1px solid #19273c;background:#0a1423;}
.filter-group {display:flex;flex-direction:column;gap:5px;min-width:150px;}
.filter-group label {color:#647892;font-size:8px;font-weight:850;text-transform:uppercase;letter-spacing:.6px;}
.filter-group select,.filter-group input {padding:8px 9px;border:1px solid #21324b;border-radius:7px;background:#0b1524;color:#dce8f5;font-size:10px;}
.stats-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;}
.stat-box {padding:12px;border:1px solid #1b2d44;border-radius:9px;background:#0a1524;}
.stat-box span {display:block;color:#687b94;font-size:8px;text-transform:uppercase;font-weight:800;}
.stat-box strong {display:block;margin-top:6px;color:white;font-size:14px;}
.metric-matrix {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;}
.metric-mini {padding:12px;border:1px solid #1b2d44;border-radius:9px;background:#0a1524;}
.metric-mini .label {color:#6d8099;font-size:8px;text-transform:uppercase;font-weight:850;}
.metric-mini .value {margin-top:7px;color:white;font-size:16px;font-weight:850;}
details.advanced {border:1px solid #1a2940;border-radius:10px;background:#0c1625;overflow:hidden;}
details.advanced summary {cursor:pointer;padding:14px 16px;color:#dfe8f4;font-size:10px;font-weight:850;text-transform:uppercase;letter-spacing:.5px;}
details.advanced[open] summary {border-bottom:1px solid #19273c;}
.unavailable-panel {opacity:.88;}
.role-badge {display:inline-block;padding:4px 8px;border-radius:99px;border:1px solid rgba(34,211,238,.2);background:rgba(34,211,238,.06);color:#a9f4ff;font-size:8px;font-weight:850;text-transform:uppercase;}
@media(max-width:1100px){.metric-matrix{grid-template-columns:repeat(2,minmax(0,1fr));}.stats-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:720px){.metric-matrix,.stats-grid{grid-template-columns:1fr;}.compare-form{align-items:stretch;}.filter-group{min-width:100%;}}

/* ============================================================
   RESPONSIVE DESIGN
   ============================================================ */
@media(max-width:1100px) {
    .app {
        grid-template-columns:
            76px minmax(0,1fr);
    }

    .brand {
        justify-content:center;
        padding-left:0;
        padding-right:0;
    }

    .brand span:last-child,
    .nav-label,
    .nav span:not(.nav-icon),
    .sidebar-foot {
        display:none;
    }

    .nav {
        justify-content:center;
    }

    .kpis {
        grid-template-columns:
            repeat(2,minmax(0,1fr));
    }

    .grid-main,
    .grid-2 {
        grid-template-columns:1fr;
    }
}

@media(max-width:720px) {
.athlete-select-grid {
    grid-template-columns:1fr;
}
    .app {
        display:block;
    }

    .sidebar {
        position:relative;
        width:100%;
        height:auto;
        padding:10px;
        display:flex;
        align-items:center;
        gap:5px;
        overflow:auto;
        border-right:none;
        border-bottom:1px solid #17253a;
    }

    .brand {
        padding:0 7px 0 0;
        font-size:16px;
    }

    .brand span:last-child,
    .nav span:not(.nav-icon) {
        display:inline;
    }

    .nav-label,
    .sidebar-foot {
        display:none;
    }

    .nav {
        white-space:nowrap;
        margin:0;
    }

    .topbar {
        position:relative;
        padding:14px;
    }

    .content {
        padding:17px 14px 40px;
    }

    .kpis {
        grid-template-columns:1fr;
    }

    .phase-flow {
        grid-template-columns:1fr;
    }

    .phase-arrow {
        transform:rotate(90deg);
    }

    .models {
        grid-template-columns:1fr;
    }
}
"""


def sidebar(active, athlete_id=None):
    coach_class = (
        "nav active"
        if active == "coach"
        else "nav"
    )

    athlete_class = (
        "nav active"
        if active == "athlete"
        else "nav"
    )

    scientist_class = (
        "nav active"
        if active == "scientist"
        else "nav"
    )

    athlete_link = (
        f"/athlete/{quote(normalize_id(athlete_id))}"
        if athlete_id
        else "/athletes"
    )

    scientist_link = (
        f"/scientist/{quote(normalize_id(athlete_id))}"
        if athlete_id
        else "/scientist"
    )

    return f"""
    <aside class="sidebar">

        <a href="/" class="brand">
            <span class="brand-mark">K</span>
            <span>KIVO</span>
        </a>

        <div class="nav-label">Coach</div>

        <a href="/" class="{coach_class}">
            <span class="nav-icon">▦</span>
            <span>Squad Dashboard</span>
        </a>

        <a
            href="{athlete_link}"
            class="{athlete_class}"
        >
            <span class="nav-icon">◎</span>
            <span>Athlete Detail</span>
        </a>

        <div style="height:13px"></div>

        <div class="nav-label">Analysis</div>

        <a
            href="{scientist_link}"
            class="{scientist_class}"
        >
            <span class="nav-icon">◇</span>
            <span>Sport Scientist</span>
        </a>

        <div class="sidebar-foot">
            <div>
                <span class="live-dot"></span>
                Data connected
            </div>
            <div>
                ForceDecks + IMU
            </div>
        </div>
    </aside>
    """


def page(
    title,
    subtitle,
    body,
    active,
    athlete_id=None,
):
    return f"""
    <!DOCTYPE html>

    <html lang="en">

    <head>
        <meta charset="UTF-8">

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
        >

        <title>
            {escape(title)} · KIVO
        </title>

        <style>
            {CSS}
        </style>
    </head>

    <body>

        <div class="app">

            {sidebar(
                active,
                athlete_id,
            )}

            <main class="main">

                <header class="topbar">

                    <div>
                        <div class="title">
                            {escape(title)}
                        </div>

                        <div class="subtitle">
                            {escape(subtitle)}
                        </div>
                    </div>

                    <div class="live">
                        <span class="live-dot"></span>
                        Live data
                    </div>

                </header>

                <section class="content">
                    {body}
                </section>

            </main>

        </div>

    </body>

    </html>
    """


# ============================================================
# COACH PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def coach_dashboard(
    metric: str = "jump_height",
    athlete_a: str = "",
    athlete_b: str = "",
):
    data = load_data()
    clean = data["clean"]

    if clean.empty:
        return page("Coach Dashboard", "Squad overview", missing_data_html(), "coach")

    summary = build_athlete_summary(clean)
    athlete_a = normalize_id(athlete_a)
    athlete_b = normalize_id(athlete_b)
    athletes = summary["athlete_id"].dropna().astype(str).sort_values().tolist()
    if not athlete_a and athletes:
        athlete_a = athletes[0]
    if not athlete_b and len(athletes) > 1:
        athlete_b = athletes[1]

    total_jumps = len(clean)
    athlete_count = clean["athlete_id"].nunique()
    avg_jump = clean["jump_height_cm"].mean() if "jump_height_cm" in clean.columns else np.nan
    avg_rsi = clean["rsi_mod"].mean() if "rsi_mod" in clean.columns else np.nan
    avg_mass = clean["body_mass_kg"].mean() if "body_mass_kg" in clean.columns else np.nan
    avg_rel_impulse = clean["relative_impulse_Ns_kg"].mean() if "relative_impulse_Ns_kg" in clean.columns else np.nan

    attention = summary[summary["status"] == "Needs attention"]
    comparison = squad_comparison_chart(summary, metric)
    quadrant = athlete_quadrant_chart(clean, athlete_a, athlete_b)
    squad_trend = squad_trend_chart(clean)

    tabs = [
        ("jump_height", "Jump Height"),
        ("rsi", "RSI-mod"),
        ("velocity", "Take-off Velocity"),
        ("braking", "Braking Impulse"),
        ("propulsive", "Propulsive Impulse"),
    ]
    tabs_html = "".join(
        f'<a href="/?metric={quote(key)}&athlete_a={quote(athlete_a)}&athlete_b={quote(athlete_b)}" class="tab {"active" if metric == key else ""}">{label}</a>'
        for key, label in tabs
    )

    options_a = "".join(
        f'<option value="{escape(a)}" {"selected" if a == athlete_a else ""}>{escape(a)}</option>'
        for a in athletes
    )
    options_b = "".join(
        f'<option value="{escape(a)}" {"selected" if a == athlete_b else ""}>{escape(a)}</option>'
        for a in athletes
    )
    compare_form = f"""
    <form method="get" action="/" class="compare-form">
        <input type="hidden" name="metric" value="{escape(metric)}">
        <div class="filter-group"><label>Athlete A</label><select name="athlete_a">{options_a}</select></div>
        <div class="filter-group"><label>Athlete B</label><select name="athlete_b">{options_b}</select></div>
        <button class="btn" type="submit">Compare Athletes</button>
    </form>
    """

    if attention.empty:
        attention_html = '<div class="attention"><div class="note">No athlete currently has a greater than 5% decline in jump height, RSI-mod or take-off velocity compared with the previous trial.</div></div>'
    else:
        cards = []
        for _, row in attention.iterrows():
            athlete = normalize_id(row["athlete_id"])
            cards.append(
                f'<a href="/athlete/{quote(athlete)}" class="attention-item">'
                f'<div class="attention-top"><div class="attention-name">{escape(athlete)}</div><span class="status alert">Review</span></div>'
                f'<div class="attention-reason">{escape(str(row["reason"]))}</div></a>'
            )
        attention_html = '<div class="attention">' + "".join(cards) + '</div>'

    rows = []
    for _, row in summary.sort_values("jump_height_cm", ascending=False, na_position="last").iterrows():
        athlete = normalize_id(row["athlete_id"])
        athlete_url = f"/athlete/{quote(athlete)}"
        status_class = "alert" if row["status"] == "Needs attention" else "good"
        rows.append(
            f"""
            <tr class="clickable" onclick="window.location='{athlete_url}'">
                <td><div class="athlete-name">{escape(athlete)}</div><div class="meta">{int(row['jumps'])} jumps · {int(row['trials'])} trial(s)</div></td>
                <td>{fmt(row['body_mass_kg'], 1, ' kg')}</td>
                <td>{fmt(row['jump_height_cm'], 1, ' cm')}<div>{trend(row['jump_height_change'])}</div></td>
                <td>{fmt(row['rsi_mod'], 2)}</td>
                <td>{fmt(row['peak_force_N'], 0, ' N')}</td>
                <td>{fmt(row['relative_peak_force_N_kg'], 2, ' N/kg')}</td>
                <td>{fmt(row['relative_impulse_Ns_kg'], 2, ' N·s/kg')}</td>
                <td>{fmt(row['velocity_ms'], 2, ' m/s')}</td>
                <td>{fmt(row['variability_pct'], 1, '%')}</td>
                <td><span class="status {status_class}">{'Review' if status_class == 'alert' else 'Stable'}</span></td>
                <td style="color:#22d3ee;font-weight:800">View →</td>
            </tr>
            """
        )

    body = f"""
    <div class="kpis">
        <div class="kpi"><div class="kpi-label">Total Accepted Jumps</div><div class="kpi-value">{total_jumps}</div><div class="kpi-foot">Clean ForceDecks targets</div></div>
        <div class="kpi"><div class="kpi-label">Athletes</div><div class="kpi-value">{athlete_count}</div><div class="kpi-foot">Squad athletes</div></div>
        <div class="kpi"><div class="kpi-label">Squad Avg Jump Height</div><div class="kpi-value">{fmt(avg_jump, 1, ' cm')}</div><div class="kpi-foot">Estimated from take-off velocity</div></div>
        <div class="kpi"><div class="kpi-label">Squad Avg RSI-mod</div><div class="kpi-value">{fmt(avg_rsi, 2)}</div><div class="kpi-foot">Jump height / time-to-takeoff</div></div>
    </div>

    <div class="grid-main">
        <div class="panel"><div class="panel-head"><div><div class="panel-title">Athlete Comparison</div><div class="panel-sub">Practical squad comparison</div></div><div class="tabs">{tabs_html}</div></div><div class="panel-body">{comparison}</div></div>
        <div class="panel"><div class="panel-head"><div><div class="panel-title">Needs Attention</div><div class="panel-sub">Simple coaching cues</div></div><span class="status {'alert' if len(attention) else 'good'}">{len(attention)} athlete(s)</span></div>{attention_html}</div>
    </div>

    <div class="section"><div><div class="section-title">Body-Weight Normalised Comparison</div><div class="section-note">Fairer comparison across athletes of different size</div></div></div>
    <div class="metric-matrix">
        <div class="metric-mini"><div class="label">Squad Avg Body Mass</div><div class="value">{fmt(avg_mass, 1, ' kg')}</div></div>
        <div class="metric-mini"><div class="label">Avg Relative Impulse</div><div class="value">{fmt(avg_rel_impulse, 2, ' N·s/kg')}</div></div>
        <div class="metric-mini"><div class="label">Relative Peak Force</div><div class="value">{'Available' if 'relative_peak_force_N_kg' in clean.columns else 'Unavailable'}</div></div>
        <div class="metric-mini"><div class="label">Relative Power</div><div class="value">{'Available' if 'relative_peak_power_W_kg' in clean.columns else 'Unavailable'}</div></div>
    </div>

    <div class="section"><div><div class="section-title">Athlete Comparison Quadrant</div><div class="section-note">RSI Modified vs Jump Height · median lines define the four performance areas</div></div></div>
    <div class="panel">{compare_form}<div class="panel-body">{quadrant}</div></div>

    <div style="height:15px"></div>
    <div class="panel"><div class="panel-body">{squad_trend}</div></div>

    <div class="section"><div><div class="section-title">Squad Athletes</div><div class="section-note">Coach view · practical metrics, normalisation and variability</div></div></div>
    <div class="panel"><div class="table-wrap"><table><thead><tr>
        <th>Athlete</th><th>Body Mass</th><th>Jump Height</th><th>RSI-mod</th><th>Peak Force</th><th>Relative Peak Force</th><th>Relative Impulse</th><th>Take-off Velocity</th><th>Variability</th><th>Status</th><th></th>
    </tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
    """

    return page(
        "Coach Dashboard",
        "Practical squad performance · comparison · normalised metrics",
        body,
        "coach",
    )


# ============================================================
# ATHLETE DETAIL
# ============================================================

# ============================================================
# ATHLETES PAGE
# ============================================================

@app.get(
    "/athletes",
    response_class=HTMLResponse,
)
def athletes_page():

    data = load_data()
    clean = data["clean"]
    
    if "athlete_id" in clean.columns:
        clean = clean.copy()
        clean["athlete_id"] = (
            clean["athlete_id"]
            .astype(str)
            .str.strip()
        )
    if clean.empty:
        return page(
            "Athletes",
            "Select an athlete",
            missing_data_html(),
            "athlete",
        )

    summary = build_athlete_summary(clean)

    athlete_cards = []

    for _, row in summary.sort_values(
        "athlete_id"
    ).iterrows():

        athlete_id = normalize_id(
            row["athlete_id"]
        )

        athlete_url = (
            f"/athlete/{quote(athlete_id)}"
        )

        status_class = (
            "alert"
            if row["status"] == "Needs attention"
            else "good"
        )

        athlete_cards.append(
            f"""
            <a
                href="{athlete_url}"
                class="athlete-select-card"
            >

                <div class="athlete-select-top">

                    <div>

                        <div class="athlete-select-name">
                            {escape(athlete_id)}
                        </div>

                        <div class="athlete-select-meta">
                            {int(row["jumps"])} jumps ·
                            {int(row["trials"])} trial(s)
                        </div>

                    </div>

                    <span
                        class="status {status_class}"
                    >
                        {
                            "Needs attention"
                            if status_class == "alert"
                            else "Stable"
                        }
                    </span>

                </div>


                <div class="athlete-select-metrics">

                    <div>
                        <span>Jump Height</span>

                        <strong>
                            {fmt(
                                row["jump_height_cm"],
                                1,
                                " cm",
                            )}
                        </strong>
                    </div>


                    <div>
                        <span>RSI-mod</span>

                        <strong>
                            {fmt(
                                row["rsi_mod"],
                                2,
                            )}
                        </strong>
                    </div>


                    <div>
                        <span>
                            Take-off Velocity
                        </span>

                        <strong>
                            {fmt(
                                row["velocity_ms"],
                                2,
                                " m/s",
                            )}
                        </strong>
                    </div>


                    <div>
                        <span>
                            Braking Impulse
                        </span>

                        <strong>
                            {fmt(
                                row["braking_Ns"],
                                1,
                                " N·s",
                            )}
                        </strong>
                    </div>

                </div>


                <div class="athlete-select-open">
                    View Athlete →
                </div>

            </a>
            """
        )


    body = f"""
    <div class="section">

        <div>

            <div class="section-title">
                Squad Athletes
            </div>

            <div class="section-note">
                Select an athlete to view
                individual performance
            </div>

        </div>

    </div>


    <div class="athlete-select-grid">

        {''.join(athlete_cards)}

    </div>
    """


    return page(
    "Athletes",
    "Squad athletes · Select an athlete for more detail",
    body,
    "athlete",
)


@app.get(
    "/athlete/{athlete_id}",
    response_class=HTMLResponse,
)
def athlete_detail(
    athlete_id: str,
    metric: str = "jump_height",
):
    data = load_data()
    clean = data["clean"]

    if clean.empty:
        return page(
            "Athlete Detail",
            "Individual performance",
            missing_data_html(),
            "athlete",
            athlete_id,
        )

    athlete_id = normalize_id(athlete_id)
    clean = clean.copy()
    clean["athlete_id"] = clean["athlete_id"].astype(str).str.strip()
    d = clean[clean["athlete_id"] == athlete_id].copy()

    if d.empty:
        raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")

    d = d.sort_values("_order")
    trial_order = d.groupby("trial_id")["_order"].max().sort_values().index.tolist()
    if not trial_order:
        raise HTTPException(status_code=404, detail=f"No trials found for athlete {athlete_id}")

    latest_trial = trial_order[-1]
    latest_trial_df = d[d["trial_id"] == latest_trial].copy()

    jump_h, _, jump_change = latest_vs_previous(d, "jump_height_cm")
    rsi, _, rsi_change = latest_vs_previous(d, "rsi_mod")
    velocity, _, velocity_change = latest_vs_previous(d, "target_takeoff_velocity_ms")
    braking, _, braking_change = latest_vs_previous(d, "braking_impulse_display_Ns")
    propulsive, _, propulsive_change = latest_vs_previous(d, "propulsive_impulse_Ns")

    body_mass = latest_trial_df["body_mass_kg"].median() if "body_mass_kg" in latest_trial_df.columns else np.nan
    personal_best = d["jump_height_cm"].max() if "jump_height_cm" in d.columns else np.nan
    average_jump = d["jump_height_cm"].mean() if "jump_height_cm" in d.columns else np.nan
    latest_braking_duration = latest_trial_df["braking_duration_ms"].median() if "braking_duration_ms" in latest_trial_df.columns else np.nan
    latest_propulsive_duration = latest_trial_df["propulsive_duration_ms"].median() if "propulsive_duration_ms" in latest_trial_df.columns else np.nan
    latest_flight = latest_trial_df["flight_time_s"].median() if "flight_time_s" in latest_trial_df.columns else np.nan
    latest_braking_impulse = latest_trial_df["braking_impulse_display_Ns"].median() if "braking_impulse_display_Ns" in latest_trial_df.columns else np.nan
    latest_propulsive_impulse = latest_trial_df["propulsive_impulse_Ns"].median() if "propulsive_impulse_Ns" in latest_trial_df.columns else np.nan
    latest_ttt = latest_trial_df["time_to_takeoff_s"].median() if "time_to_takeoff_s" in latest_trial_df.columns else np.nan

    history = athlete_history_chart(d, metric)
    recent_jumps = latest_jump_chart(d)
    phase_chart = phase_duration_chart(latest_trial_df)
    strategy = jump_strategy_text(latest_trial_df)

    metric_tabs = [
        ("jump_height", "Jump Height"),
        ("rsi", "RSI-mod"),
        ("velocity", "Take-off Velocity"),
        ("braking", "Braking Impulse"),
        ("propulsive", "Propulsive Impulse"),
        ("flight", "Flight Time"),
    ]
    metric_html = "".join(
        f'<a class="tab {"active" if metric == key else ""}" href="/athlete/{quote(athlete_id)}?metric={quote(key)}">{label}</a>'
        for key, label in metric_tabs
    )

    performance_summary = "Stable overall performance." 
    if np.isfinite(jump_change):
        if jump_change > 2:
            performance_summary = f"Jump height improved by {abs(jump_change):.1f}% compared with the previous session."
        elif jump_change < -2:
            performance_summary = f"Jump height decreased by {abs(jump_change):.1f}% compared with the previous session. Review together with RSI-mod and training context."
        else:
            performance_summary = "Jump height is similar to the previous session."

    recent_rows = []
    for _, row in d.sort_values("_order", ascending=False).head(12).iterrows():
        recent_rows.append(
            f"""
            <tr>
                <td><span class="athlete-name">{escape(str(row['trial_id']))}</span></td>
                <td>J{int(row['jump_id'])}</td>
                <td>{fmt(row.get('jump_height_cm'), 1, ' cm')}</td>
                <td>{fmt(row.get('rsi_mod'), 2)}</td>
                <td>{fmt(row.get('target_takeoff_velocity_ms'), 2, ' m/s')}</td>
            </tr>
            """
        )

    body = f"""
    <div class="breadcrumb">
        <a href="/">Coach Dashboard</a><span>›</span><a href="/athletes">Athletes</a><span>›</span><span>{escape(athlete_id)}</span>
    </div>

    <div class="athlete-head">
        <div>
            <div class="athlete-title">{escape(athlete_id)}</div>
            <div class="athlete-sub">Athlete view · simple performance feedback · Latest trial: {escape(str(latest_trial))}</div>
        </div>
        <div><span class="role-badge">Athlete View</span> <a class="btn" href="/scientist/{quote(athlete_id)}">Advanced Analysis →</a></div>
    </div>

    <div class="kpis">
        <div class="kpi"><div class="kpi-label">Jump Height</div><div class="kpi-value">{fmt(jump_h,1,' cm')}</div><div class="kpi-foot">{trend(jump_change)} vs previous session</div></div>
        <div class="kpi"><div class="kpi-label">RSI Modified</div><div class="kpi-value">{fmt(rsi,2)}</div><div class="kpi-foot">{trend(rsi_change)} vs previous session</div></div>
        <div class="kpi"><div class="kpi-label">Best Jump</div><div class="kpi-value">{fmt(personal_best,1,' cm')}</div><div class="kpi-foot">Personal best in available data</div></div>
        <div class="kpi"><div class="kpi-label">Average Jump</div><div class="kpi-value">{fmt(average_jump,1,' cm')}</div><div class="kpi-foot">Across accepted jumps</div></div>
    </div>

    <div class="panel">
        <div class="panel-head"><div><div class="panel-title">Performance History</div><div class="panel-sub">Progress and session-to-session trend</div></div><div class="tabs">{metric_html}</div></div>
        <div class="panel-body">{history}</div>
    </div>

    <div style="height:15px"></div>
    <div class="grid-2">
        <div class="panel"><div class="panel-head"><div><div class="panel-title">Simple Performance Summary</div><div class="panel-sub">Easy-to-understand feedback</div></div></div><div class="panel-body"><div class="note">{escape(performance_summary)}</div><div style="height:10px"></div><div class="summary-list">
            <div class="summary-row"><span class="summary-label">Body Mass</span><span class="summary-value">{fmt(body_mass,1,' kg')}</span></div>
            <div class="summary-row"><span class="summary-label">Take-off Velocity</span><span class="summary-value">{fmt(velocity,2,' m/s')} {trend(velocity_change)}</span></div>
            <div class="summary-row"><span class="summary-label">Latest Flight Time</span><span class="summary-value">{fmt(latest_flight,3,' s')}</span></div>
        </div></div></div>
        <div class="panel"><div class="panel-body">{recent_jumps}</div></div>
    </div>

    <div class="section"><div><div class="section-title">Recent Jumps</div><div class="section-note">Simple recent performance table</div></div></div>
    <div class="panel"><div class="table-wrap"><table><thead><tr><th>Trial</th><th>Jump</th><th>Jump Height</th><th>RSI-mod</th><th>Take-off Velocity</th></tr></thead><tbody>{''.join(recent_rows)}</tbody></table></div></div>

    <div style="height:15px"></div>
    <details class="advanced">
        <summary>Coach / Scientist phase detail (collapsed by default)</summary>
        <div class="panel-body">
            <div class="note">{escape(strategy)}</div>
            <div style="height:12px"></div>
            <div class="phase-flow">
                <div class="phase"><div class="phase-name">Countermovement</div><div class="phase-value">{fmt(latest_ttt,3,' s')}</div><div class="phase-sub">onset → take-off</div></div>
                <div class="phase-arrow">→</div>
                <div class="phase"><div class="phase-name">Braking</div><div class="phase-value">{fmt(latest_braking_duration,0,' ms')}</div><div class="phase-sub">{fmt(latest_braking_impulse,2,' N·s')}</div></div>
                <div class="phase-arrow">→</div>
                <div class="phase"><div class="phase-name">Propulsive</div><div class="phase-value">{fmt(latest_propulsive_duration,0,' ms')}</div><div class="phase-sub">{fmt(latest_propulsive_impulse,2,' N·s')}</div></div>
                <div class="phase-arrow">→</div>
                <div class="phase"><div class="phase-name">Take-off</div><div class="phase-value">{fmt(velocity,2,' m/s')}</div><div class="phase-sub">target velocity</div></div>
                <div class="phase-arrow">→</div>
                <div class="phase"><div class="phase-name">Flight</div><div class="phase-value">{fmt(latest_flight,3,' s')}</div><div class="phase-sub">take-off → landing</div></div>
            </div>
            {phase_chart}
        </div>
    </details>
    """

    return page(
        f"Athlete · {athlete_id}",
        'Simple performance feedback · "Am I improving?"',
        body,
        "athlete",
        athlete_id,
    )


# ============================================================
# SPORT SCIENTIST
# ============================================================

def model_cards(models):
    if models.empty:
        return """
        <div class="note">
            model_results_summary.csv was not found.
        </div>
        """

    cards = []

    for index, row in models.head(9).iterrows():
        model_name = None

        for candidate in [
            "model",
            "Model",
            "model_name",
            "Model Name",
        ]:
            if candidate in models.columns:
                model_name = row[candidate]
                break

        if model_name is None:
            model_name = f"Model {index + 1}"

        values = []

        for column in models.columns:
            if column in [
                "model",
                "Model",
                "model_name",
                "Model Name",
            ]:
                continue

            value = row[column]

            if pd.isna(value):
                continue

            if isinstance(
                value,
                (
                    float,
                    np.floating,
                ),
            ):
                value_text = f"{value:.4f}"
            else:
                value_text = escape(
                    str(value)
                )

            values.append(
                f"""
                <div>
                    {escape(str(column))}:
                    <strong>
                        {value_text}
                    </strong>
                </div>
                """
            )

        cards.append(
            f"""
            <div class="model">

                <div class="model-name">
                    {escape(str(model_name))}
                </div>

                <div class="model-values">
                    {''.join(values[:6])}
                </div>

            </div>
            """
        )

    return (
        '<div class="models">'
        + "".join(cards)
        + "</div>"
    )


@app.get(
    "/scientist",
    response_class=HTMLResponse,
)
def scientist():
    data = load_data()
    clean = data["clean"]
    if clean.empty:
        return page("Sport Scientist", "Measurement reliability", missing_data_html(), "scientist")

    body = f"""
    <div class="athlete-head"><div><div class="athlete-title">Sport Scientist</div><div class="athlete-sub">Full biomechanical, statistical, signal-processing and model-analysis view</div></div><span class="role-badge">Sport Scientist View</span></div>
    <div class="grid-2">
        <div class="panel"><div class="panel-body">{quality_chart(data['quality'])}</div></div>
        <div class="panel"><div class="panel-body">{prediction_chart(data['prediction'])}</div></div>
    </div>
    <div class="section"><div><div class="section-title">Model Results</div><div class="section-note">Cross-validation and model summary</div></div></div>
    <div class="panel"><div class="panel-body">{model_cards(data['models'])}</div></div>
    <div class="section"><div><div class="section-title">Sample-Level Signal Availability</div><div class="section-note">Force, acceleration and velocity curves are shown only when compatible sample-level CSV files exist</div></div></div>
    <div class="panel"><div class="panel-body"><div class="note">Detected optional signal files: {len(signal_candidates())}. Event-level outputs remain fully usable even when raw/sample-level curves are unavailable. No signal values are fabricated.</div></div></div>
    """
    return page("Sport Scientist", '"Why is the measurement changing, and how reliable is it?"', body, "scientist")


@app.get(
    "/scientist/{athlete_id}",
    response_class=HTMLResponse,
)
def scientist_athlete(
    athlete_id: str,
    session: str = "",
    jump: str = "",
    stat_metric: str = "jump_height",
    date: str = "",
):
    data = load_data()
    clean = data["clean"]
    if clean.empty:
        return page("Sport Scientist", "Advanced athlete analysis", missing_data_html(), "scientist", athlete_id)

    athlete_id = normalize_id(athlete_id)
    clean = clean.copy()
    clean["athlete_id"] = clean["athlete_id"].astype(str).str.strip()
    athlete = clean[clean["athlete_id"] == athlete_id].copy()
    if athlete.empty:
        raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")

    # Filters: athlete is route-selected; session and jump are optional.
    sessions = athlete.groupby("trial_id")["_order"].max().sort_values().index.astype(str).tolist()
    selected_session = str(session).strip() if session else (sessions[-1] if sessions else "")
    filtered = athlete.copy()
    if selected_session:
        filtered = filtered[filtered["trial_id"].astype(str) == selected_session]

    jumps = []
    if "jump_id" in filtered.columns:
        jumps = [str(int(v)) for v in pd.to_numeric(filtered["jump_id"], errors="coerce").dropna().unique()]
    selected_jump = str(jump).strip()
    jump_filtered = filtered.copy()
    if selected_jump and "jump_id" in jump_filtered.columns:
        jump_filtered = jump_filtered[pd.to_numeric(jump_filtered["jump_id"], errors="coerce") == as_float(selected_jump)]

    # Optional date filtering only when a date field is actually available.
    date_col = next((c for c in ["date", "session_date", "test_date", "recording_date"] if c in athlete.columns), None)
    if date and date_col:
        date_values = pd.to_datetime(jump_filtered[date_col], errors="coerce")
        jump_filtered = jump_filtered[date_values.dt.strftime("%Y-%m-%d") == date]

    event_df = jump_filtered if not jump_filtered.empty else filtered
    event_row = event_df.iloc[-1] if not event_df.empty else athlete.iloc[-1]
    selected_jump_num = event_row.get("jump_id", None)

    quality = data["quality"].copy()
    if not quality.empty and "athlete_id" in quality.columns:
        quality["athlete_id"] = quality["athlete_id"].astype(str).str.strip()
        quality = quality[quality["athlete_id"] == athlete_id]
        if selected_session and "trial_id" in quality.columns:
            quality["trial_id"] = quality["trial_id"].astype(str).str.strip()
            quality = quality[quality["trial_id"] == selected_session]

    shifts = data["shifts"].copy()
    if not shifts.empty and "athlete_id" in shifts.columns:
        shifts["athlete_id"] = shifts["athlete_id"].astype(str).str.strip()
        shifts = shifts[shifts["athlete_id"] == athlete_id]

    # Unit-formatted advanced metrics. Missing source columns remain unavailable.
    adv_metrics = [
        ("Body Mass", "body_mass_kg", 1, " kg"),
        ("Jump Height", "jump_height_cm", 1, " cm"),
        ("RSI Modified", "rsi_mod", 2, ""),
        ("Peak Force", "peak_force_N", 0, " N"),
        ("Mean Force", "mean_force_N", 0, " N"),
        ("Relative Force", "relative_peak_force_N_kg", 2, " N/kg"),
        ("Braking Impulse", "braking_impulse_display_Ns", 2, " N·s"),
        ("Propulsive Impulse", "propulsive_impulse_Ns", 2, " N·s"),
        ("Total / Net Impulse", "total_impulse_Ns", 2, " N·s"),
        ("Relative Impulse", "relative_impulse_Ns_kg", 2, " N·s/kg"),
        ("Ecc:Con Impulse Ratio", "ecc_con_impulse_ratio", 2, ""),
        ("Peak Acceleration", "peak_acceleration_ms2", 2, " m/s²"),
        ("Mean Acceleration", "mean_acceleration_ms2", 2, " m/s²"),
        ("Peak Velocity", "peak_velocity_ms", 2, " m/s"),
        ("Take-off Velocity", "target_takeoff_velocity_ms", 2, " m/s"),
        ("Peak Power", "peak_power_W", 0, " W"),
        ("Relative Power", "relative_peak_power_W_kg", 1, " W/kg"),
        ("Eccentric Duration", "braking_duration_ms", 0, " ms"),
        ("Concentric Duration", "propulsive_duration_ms", 0, " ms"),
    ]
    metric_cards = []
    for label, col, dec, suffix in adv_metrics:
        val = metric_or_unavailable(event_df, col, dec, suffix)
        metric_cards.append(f'<div class="metric-mini"><div class="label">{escape(label)}</div><div class="value">{escape(val)}</div></div>')

    # Statistics metric selector.
    stat_map = {
        "jump_height": ("jump_height_cm", "Jump Height", "cm", 1),
        "rsi": ("rsi_mod", "RSI Modified", "", 2),
        "velocity": ("target_takeoff_velocity_ms", "Take-off Velocity", "m/s", 2),
        "braking": ("braking_impulse_display_Ns", "Braking Impulse", "N·s", 2),
        "propulsive": ("propulsive_impulse_Ns", "Propulsive Impulse", "N·s", 2),
        "relative_impulse": ("relative_impulse_Ns_kg", "Relative Impulse", "N·s/kg", 2),
    }
    stat_col, stat_label, stat_unit, stat_dec = stat_map.get(stat_metric, stat_map["jump_height"])
    stats_html = statistics_table(filtered, stat_col, stat_label, stat_unit, stat_dec)

    session_opts = "".join(f'<option value="{escape(x)}" {"selected" if x == selected_session else ""}>{escape(x)}</option>' for x in sessions)
    jump_opts = '<option value="">All jumps</option>' + "".join(f'<option value="{escape(x)}" {"selected" if x == selected_jump else ""}>Jump {escape(x)}</option>' for x in jumps)
    stat_opts = "".join(f'<option value="{k}" {"selected" if k == stat_metric else ""}>{escape(v[1])}</option>' for k, v in stat_map.items())
    date_control = f'<input type="date" name="date" value="{escape(date)}">' if date_col else '<input type="text" value="Date unavailable" disabled>'

    filters_html = f"""
    <form method="get" action="/scientist/{quote(athlete_id)}" class="compare-form">
        <div class="filter-group"><label>Session</label><select name="session">{session_opts}</select></div>
        <div class="filter-group"><label>Jump number</label><select name="jump">{jump_opts}</select></div>
        <div class="filter-group"><label>Date</label>{date_control}</div>
        <div class="filter-group"><label>Statistics metric</label><select name="stat_metric">{stat_opts}</select></div>
        <button class="btn" type="submit">Apply Filters</button>
    </form>
    """

    force_plot = signal_chart(athlete_id, selected_session, selected_jump_num, "force", event_row)
    acceleration_plot = signal_chart(athlete_id, selected_session, selected_jump_num, "acceleration", event_row)
    velocity_plot = signal_chart(athlete_id, selected_session, selected_jump_num, "velocity", event_row)
    strategy = jump_strategy_text(filtered)

    if shifts.empty:
        shift_html = '<div class="panel-body"><div class="note">No athlete-specific alignment row was found in imu_force_shift_summary.csv.</div></div>'
    else:
        columns = shifts.columns[:8].tolist()
        headers = "".join(f"<th>{escape(str(c))}</th>" for c in columns)
        rows = []
        for _, row in shifts.head(20).iterrows():
            cells = "".join(f"<td>{escape(str(row[c]))}</td>" for c in columns)
            rows.append(f"<tr>{cells}</tr>")
        shift_html = f'<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    body = f"""
    <div class="breadcrumb"><a href="/">Coach Dashboard</a><span>›</span><a href="/athlete/{quote(athlete_id)}">{escape(athlete_id)}</a><span>›</span><span>Advanced Analysis</span></div>
    <div class="athlete-head"><div><div class="athlete-title">{escape(athlete_id)}</div><div class="athlete-sub">Full biomechanical and model analysis</div></div><div><span class="role-badge">Sport Scientist View</span> <a class="btn" href="/athlete/{quote(athlete_id)}">← Athlete Profile</a></div></div>

    <div class="panel">{filters_html}</div>

    <div class="section"><div><div class="section-title">Advanced Metrics</div><div class="section-note">Values use available source columns only; unavailable metrics are explicitly marked</div></div></div>
    <div class="metric-matrix">{''.join(metric_cards)}</div>

    <div class="section"><div><div class="section-title">Signal Curves & Phase Markers</div><div class="section-note">Start movement · eccentric/braking · concentric/propulsive · take-off · landing</div></div></div>
    <div class="panel"><div class="panel-body">{force_plot}</div></div>
    <div style="height:15px"></div>
    <div class="grid-2"><div class="panel"><div class="panel-body">{acceleration_plot}</div></div><div class="panel"><div class="panel-body">{velocity_plot}</div></div></div>

    <div class="section"><div><div class="section-title">Jump Strategy</div><div class="section-note">Movement strategy description, not a good/bad classification</div></div></div>
    <div class="panel"><div class="panel-body"><div class="note">{escape(strategy)}</div></div></div>

    <div class="section"><div><div class="section-title">Detailed Statistics</div><div class="section-note">Mean · min · max · standard deviation · best · variability</div></div></div>
    <div class="panel"><div class="panel-head"><div><div class="panel-title">{escape(stat_label)} Statistics</div><div class="panel-sub">Current filtered session / jumps</div></div></div><div class="panel-body">{stats_html}</div></div>

    <div class="section"><div><div class="section-title">Data Quality / QC Flags</div><div class="section-note">Valid/invalid jump, phase, alignment, noise and event-detection information when exported</div></div></div>
    <div class="grid-2"><div class="panel"><div class="panel-body">{quality_chart(quality)}</div></div><div class="panel">{qc_flag_list(quality)}</div></div>

    <div class="section"><div><div class="section-title">ForceDeck vs Model Prediction</div><div class="section-note">Ground truth · predicted value · absolute error · percentage error</div></div></div>
    <div class="grid-2"><div class="panel"><div class="panel-body">{prediction_chart(data['prediction'], athlete_id)}</div></div><div class="panel">{model_error_table(data['prediction'], athlete_id)}</div></div>

    <div class="section"><div><div class="section-title">IMU ↔ ForceDecks Alignment</div><div class="section-note">Alignment and shift information</div></div></div>
    <div class="panel">{shift_html}</div>

    <div class="section"><div><div class="section-title">Model Results</div><div class="section-note">Model validation summary</div></div></div>
    <div class="panel"><div class="panel-body">{model_cards(data['models'])}</div></div>

    <div style="height:15px"></div>
    <div class="note">The dashboard does not fabricate unavailable biomechanics. Force, acceleration, velocity and power components display real data only when matching columns/sample-level files exist in ml_ready_outputs.</div>
    """

    return page(
        f"Sport Scientist · {athlete_id}",
        'Full biomechanical · statistical · signal-processing · model analysis',
        body,
        "scientist",
        athlete_id,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_directory": str(
            DATA_DIR.resolve()
        ),
        "files": {
            "clean_force_targets":
                CLEAN_FILE.exists(),

            "quality_review":
                QUALITY_FILE.exists(),

            "prediction_review":
                PREDICTION_FILE.exists(),

            "model_results":
                MODEL_FILE.exists(),

            "shift_summary":
                SHIFT_FILE.exists(),
        },
    }


# ============================================================
# RUN DIRECTLY WITH:
# python dashboard.py
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
