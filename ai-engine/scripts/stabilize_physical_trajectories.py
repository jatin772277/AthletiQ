from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
INPUT = ROOT / "outputs/physical_metrics/football2_stitched_physical_tracks.csv"
OUTPUT = ROOT / "outputs/physical_metrics/football2_stitched_physical_tracks_stabilized.csv"
METRICS = ROOT / "outputs/physical_metrics/football2_player_physical_metrics_stabilized.csv"

FPS = 25.0
MAX_SPEED_MPS = 9.0
MAX_GAP_SECONDS = 0.20
SMOOTH_WINDOW = 5

df = pd.read_csv(INPUT)
required = {
    "canonical_id", "frame_id", "time_seconds",
    "pitch_x_m", "pitch_y_m"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.sort_values(["canonical_id", "frame_id"]).reset_index(drop=True)

# Preserve original projected coordinates for auditing.
df["pitch_x_m_raw"] = df["pitch_x_m"]
df["pitch_y_m_raw"] = df["pitch_y_m"]
df["trajectory_rejected"] = False
df["trajectory_smoothed"] = False

# Reject isolated implausible jumps; do not bridge long gaps.
for _, indices in df.groupby("canonical_id").groups.items():
    idx = list(indices)
    g = df.loc[idx]
    x = g["pitch_x_m"].to_numpy(dtype=float)
    y = g["pitch_y_m"].to_numpy(dtype=float)
    t = g["time_seconds"].to_numpy(dtype=float)

    dt = np.diff(t)
    dist = np.hypot(np.diff(x), np.diff(y))
    speed = np.divide(
        dist, dt,
        out=np.full_like(dist, np.inf),
        where=dt > 0
    )

    bad = (
        (dt <= 0)
        | (dt > MAX_GAP_SECONDS)
        | (speed > MAX_SPEED_MPS)
    )

    # Mark the destination of each bad step.
    bad_positions = np.where(bad)[0] + 1
    df.loc[np.asarray(idx)[bad_positions], "trajectory_rejected"] = True

# Smooth only short, continuous runs of accepted coordinates.
# Rejected points remain NaN; never interpolate across rejected jumps.
for _, indices in df.groupby("canonical_id").groups.items():
    idx = list(indices)
    g = df.loc[idx].copy()

    accepted = ~g["trajectory_rejected"].to_numpy()
    times = g["time_seconds"].to_numpy(dtype=float)
    x = g["pitch_x_m_raw"].to_numpy(dtype=float)
    y = g["pitch_y_m_raw"].to_numpy(dtype=float)

    # Rolling median is robust to isolated detector jitter.
    sx = pd.Series(x).rolling(
        SMOOTH_WINDOW, center=True, min_periods=3
    ).median().to_numpy()
    sy = pd.Series(y).rolling(
        SMOOTH_WINDOW, center=True, min_periods=3
    ).median().to_numpy()

    sx[~accepted] = np.nan
    sy[~accepted] = np.nan

    df.loc[idx, "pitch_x_m_stabilized"] = sx
    df.loc[idx, "pitch_y_m_stabilized"] = sy
    changed = accepted & (
        np.isfinite(sx) & np.isfinite(sy)
    )
    df.loc[np.asarray(idx)[changed], "trajectory_smoothed"] = True

# Calculate metrics only from consecutive accepted points.
metrics = []
for pid, g in df.groupby("canonical_id"):
    g = g.sort_values("frame_id")
    x = g["pitch_x_m_stabilized"].to_numpy(dtype=float)
    y = g["pitch_y_m_stabilized"].to_numpy(dtype=float)
    t = g["time_seconds"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    dt = np.diff(t)
    dist = np.hypot(np.diff(x), np.diff(y))
    pair_valid = valid[1:] & valid[:-1] & (dt > 0) & (
        dt <= MAX_GAP_SECONDS
    )

    speeds = np.full(len(dt), np.nan)
    speeds[pair_valid] = dist[pair_valid] / dt[pair_valid]
    plausible = pair_valid & (speeds <= MAX_SPEED_MPS)

    metrics.append({
        "canonical_id": pid,
        "observations": len(g),
        "rejected_observations": int(g["trajectory_rejected"].sum()),
        "valid_smoothed_observations": int(valid.sum()),
        "distance_m_lower_bound": float(dist[plausible].sum()),
        "mean_speed_mps_on_valid_steps": (
            float(np.nanmean(speeds[plausible]))
            if plausible.any() else np.nan
        ),
        "p95_speed_mps_on_valid_steps": (
            float(np.nanpercentile(speeds[plausible], 95))
            if plausible.any() else np.nan
        ),
        "valid_speed_steps": int(plausible.sum()),
        "status": "estimate_requires_calibration_validation",
    })

df.to_csv(OUTPUT, index=False)
pd.DataFrame(metrics).to_csv(METRICS, index=False)

print("=== TRAJECTORY STABILIZATION COMPLETE ===")
print(f"Input rows: {len(df)}")
print(f"Rejected observations: {int(df['trajectory_rejected'].sum())}")
print(f"Smoothed observations: {int(df['trajectory_smoothed'].sum())}")
print(f"Saved tracks: {OUTPUT}")
print(f"Saved metrics: {METRICS}")
print("\nTop players by rejected observations:")
print(
    pd.DataFrame(metrics)
    .sort_values("rejected_observations", ascending=False)
    .head(10)
    .to_string(index=False)
)
print("\nIMPORTANT: These remain provisional estimates, not validated physical measurements.")
