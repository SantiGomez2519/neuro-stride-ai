from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .config import ProcessingConfig


@dataclass
class CleaningReport:
    """Summary of cleaning operations applied to a trial."""

    original_missing: int
    hampel_outliers: int
    interpolated_values: int
    final_missing: int
    channel_report: pd.DataFrame


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True runs in a boolean mask."""
    padded = np.r_[False, mask, False].astype(int)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts, ends))


def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Interpolate short internal NaN gaps using PCHIP."""
    x = series.astype(float).copy()
    original_nan = x.isna().to_numpy()

    interpolated = x.interpolate(
        method="pchip",
        limit_direction="both",
    )

    for start, end in _boolean_runs(original_nan):
        gap_length = end - start
        if gap_length > max_gap or start == 0 or end == len(x):
            interpolated.iloc[start:end] = np.nan

    return interpolated


def hampel_to_nan(
    series: pd.Series,
    window: int,
    n_sigma: float = 3.5,
) -> pd.Series:
    """
    Detect local outliers using a Hampel filter.

    This utility is intentionally NOT used by the automatic cleaning
    pipeline. It is kept for QC and diagnostic use.
    """
    x = series.astype(float).copy()
    min_periods = max(3, window // 2)

    median = x.rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    ).median()

    abs_dev = (x - median).abs()

    mad = abs_dev.rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    ).median()

    robust_sigma = 1.4826 * mad

    outlier_mask = (
        abs_dev
        > n_sigma * robust_sigma.replace(0, np.nan)
    )

    x[outlier_mask.fillna(False)] = np.nan
    return x


def filter_valid_segments(
    series: pd.Series,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> pd.Series:
    """Apply a zero-phase Butterworth low-pass filter to valid segments."""
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        raise ValueError(
            "cutoff_hz must be between 0 and Nyquist frequency"
        )

    x = series.astype(float).copy()

    sos = butter(
        order,
        cutoff_hz,
        btype="low",
        fs=fs,
        output="sos",
    )

    valid_mask = x.notna().to_numpy()
    minimum_segment_length = max(18, 3 * order + 3)

    for start, end in _boolean_runs(valid_mask):
        segment_length = end - start
        if segment_length < minimum_segment_length:
            continue

        try:
            x.iloc[start:end] = sosfiltfilt(
                sos,
                x.iloc[start:end].to_numpy(),
            )
        except ValueError:
            pass

    return x


def _clean_channel(
    series: pd.Series,
    fs: float,
    cutoff_hz: float,
    max_gap: int,
    order: int,
) -> tuple[pd.Series, int]:
    """
    Clean one biomechanical signal.

    Processing order:
    1. Short-gap interpolation
    2. Low-pass filtering

    Hampel outlier detection is intentionally excluded from the
    automatic cleaning path.
    """
    missing_before = int(series.isna().sum())

    after_interpolation = interpolate_short_gaps(
        series,
        max_gap=max_gap,
    )

    missing_after = int(after_interpolation.isna().sum())
    interpolated_values = missing_before - missing_after

    cleaned = filter_valid_segments(
        after_interpolation,
        fs=fs,
        cutoff_hz=cutoff_hz,
        order=order,
    )

    return cleaned, interpolated_values


def clean_kinematic_trial(
    df: pd.DataFrame,
    cfg: ProcessingConfig,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean one complete kinematic Vicon trial."""
    required_columns = {"time_s", "frame"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required kinematic columns: "
            f"{sorted(missing_columns)}"
        )

    out = (
        df
        .sort_values(["time_s", "frame"])
        .drop_duplicates(subset=["frame"], keep="first")
        .reset_index(drop=True)
        .copy()
    )

    original_missing = int(out.isna().sum().sum())

    marker_cols = [c for c in out.columns if c.endswith("_mm")]
    angle_cols = [c for c in out.columns if c.endswith("_deg")]
    kinetic_cols = [
        c for c in out.columns
        if c.endswith("_Nm_kg") or c.endswith("_W_kg")
    ]

    max_gap = max(
        1,
        int(round(cfg.max_short_gap_s * cfg.kinematic_fs_hz)),
    )

    total_interpolated = 0
    channel_rows = []

    def process_columns(
        columns: list[str],
        signal_type: str,
        cutoff_hz: float,
    ) -> None:
        nonlocal total_interpolated

        for col in columns:
            original_channel_missing = int(out[col].isna().sum())

            cleaned, n_interpolated = _clean_channel(
                out[col],
                fs=cfg.kinematic_fs_hz,
                cutoff_hz=cutoff_hz,
                max_gap=max_gap,
                order=cfg.filter_order,
            )

            out[col] = cleaned
            final_channel_missing = int(cleaned.isna().sum())
            total_interpolated += n_interpolated

            channel_rows.append(
                {
                    "channel": col,
                    "signal_type": signal_type,
                    "original_missing": original_channel_missing,
                    "hampel_outliers": 0,
                    "interpolated_values": n_interpolated,
                    "final_missing": final_channel_missing,
                }
            )

    process_columns(
        marker_cols,
        signal_type="marker",
        cutoff_hz=cfg.marker_cutoff_hz,
    )

    process_columns(
        angle_cols,
        signal_type="angle",
        cutoff_hz=cfg.angle_cutoff_hz,
    )

    process_columns(
        kinetic_cols,
        signal_type="kinetic",
        cutoff_hz=cfg.kinetic_cutoff_hz,
    )

    final_missing = int(out.isna().sum().sum())
    channel_report = pd.DataFrame(channel_rows)

    report = CleaningReport(
        original_missing=original_missing,
        hampel_outliers=0,
        interpolated_values=total_interpolated,
        final_missing=final_missing,
        channel_report=channel_report,
    )

    return out, report


def clean_analog_trial(
    df: pd.DataFrame,
    cfg: ProcessingConfig,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean analog GRF signals from one Vicon trial."""
    required_columns = {"time_s", "sample"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required analog columns: "
            f"{sorted(missing_columns)}"
        )

    out = (
        df
        .sort_values(["time_s", "sample"])
        .drop_duplicates(subset=["sample"], keep="first")
        .reset_index(drop=True)
        .copy()
    )

    original_missing = int(out.isna().sum().sum())

    grf_cols = [
        c for c in out.columns
        if "_GRF_" in c and c.endswith("_N")
    ]

    max_gap = max(
        1,
        int(round(cfg.analog_max_short_gap_s * cfg.analog_fs_hz)),
    )

    total_interpolated = 0
    channel_rows = []

    for col in grf_cols:
        original_channel_missing = int(out[col].isna().sum())

        interpolated = interpolate_short_gaps(
            out[col],
            max_gap=max_gap,
        )

        missing_after_interpolation = int(interpolated.isna().sum())
        n_interpolated = (
            original_channel_missing
            - missing_after_interpolation
        )

        total_interpolated += n_interpolated

        cleaned = filter_valid_segments(
            interpolated,
            fs=cfg.analog_fs_hz,
            cutoff_hz=cfg.grf_cutoff_hz,
            order=cfg.filter_order,
        )

        out[col] = cleaned
        final_channel_missing = int(cleaned.isna().sum())

        channel_rows.append(
            {
                "channel": col,
                "signal_type": "grf",
                "original_missing": original_channel_missing,
                "hampel_outliers": 0,
                "interpolated_values": n_interpolated,
                "final_missing": final_channel_missing,
            }
        )

    final_missing = int(out.isna().sum().sum())
    channel_report = pd.DataFrame(channel_rows)

    report = CleaningReport(
        original_missing=original_missing,
        hampel_outliers=0,
        interpolated_values=total_interpolated,
        final_missing=final_missing,
        channel_report=channel_report,
    )

    return out, report
