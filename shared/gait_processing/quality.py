from dataclasses import dataclass

import pandas as pd

from .io import TrialData


@dataclass
class SignalIntegrity:
    duplicate_frames: int | None
    non_monotonic_time: bool
    median_dt_s: float
    expected_dt_s: float
    sampling_ok: bool


@dataclass
class TrialQualityResult:
    """Quality-control results for a complete gait trial."""

    subject_id: str
    trial_id: str

    kinematic_integrity: SignalIntegrity
    analog_integrity: SignalIntegrity

    kinematic_channels: pd.DataFrame
    analog_channels: pd.DataFrame


def trial_quality_summary(
    df: pd.DataFrame,
    exclude_columns: tuple[str, ...] = (
    "frame",
    "sample",
    "time_s",
    )
) -> pd.DataFrame:
    """
    Compute basic quality statistics for every signal channel.

    Parameters
    ----------
    df
        Signal dataframe.
    exclude_columns
        Columns that should not be interpreted as signal channels.

    Returns
    -------
    pd.DataFrame
        One row per channel with sample count, missing values,
        minimum, maximum and standard deviation.
    """

    rows = []

    for column in df.columns:

        if column in exclude_columns:
            continue

        series = df[column]

        # Ignore non-numeric columns if they ever appear
        if not pd.api.types.is_numeric_dtype(series):
            continue

        rows.append(
            {
                "channel": column,
                "n_samples": len(series),
                "missing_n": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "std": float(series.std()) if series.notna().any() else None,
            }
        )

    return pd.DataFrame(rows)


def frame_integrity(
    df: pd.DataFrame,
    expected_fs_hz: float,
    sampling_tolerance: float = 0.01,
) -> SignalIntegrity:
    """
    Check temporal integrity and, when available, frame integrity.
    """

    if "time_s" not in df.columns:
        raise ValueError(
            "Missing required column for integrity check: 'time_s'"
        )

    if expected_fs_hz <= 0:
        raise ValueError(
            "expected_fs_hz must be greater than zero."
        )

    # Frame duplicates are checked only when a frame column exists
    duplicate_frames = (
        int(df["frame"].duplicated().sum())
        if "frame" in df.columns
        else None
    )

    time_diff = df["time_s"].diff().dropna()

    if time_diff.empty:
        raise ValueError(
            "At least two samples are required "
            "to evaluate time integrity."
        )

    non_monotonic_time = bool(
        (time_diff <= 0).any()
    )

    median_dt_s = float(
        time_diff.median()
    )

    expected_dt_s = 1.0 / float(expected_fs_hz)

    tolerance_s = (
        expected_dt_s * sampling_tolerance
    )

    sampling_ok = bool(
        abs(median_dt_s - expected_dt_s)
        <= tolerance_s
    )

    return SignalIntegrity(
        duplicate_frames=duplicate_frames,
        non_monotonic_time=non_monotonic_time,
        median_dt_s=median_dt_s,
        expected_dt_s=expected_dt_s,
        sampling_ok=sampling_ok,
    )


def check_trial_quality(
    trial: TrialData,
) -> TrialQualityResult:
    """
    Run basic quality control for a complete gait trial.

    The function does not modify the signals. It only characterizes
    the data as they were loaded from the raw dataset.

    Parameters
    ----------
    trial
        TrialData object containing trajectories, analog signals
        and associated metadata.

    Returns
    -------
    TrialQualityResult
        Quality-control result for kinematic and analog signals.
    """

    kinematic_integrity = frame_integrity(
        df=trial.trajectories,
        expected_fs_hz=trial.kinematic_fs_hz,
    )

    analog_integrity = frame_integrity(
        df=trial.analog,
        expected_fs_hz=trial.analog_fs_hz,
    )

    kinematic_channels = trial_quality_summary(
        trial.trajectories
    )

    analog_channels = trial_quality_summary(
        trial.analog
    )

    return TrialQualityResult(
        subject_id=trial.subject_id,
        trial_id=trial.trial_id,
        kinematic_integrity=kinematic_integrity,
        analog_integrity=analog_integrity,
        kinematic_channels=kinematic_channels,
        analog_channels=analog_channels,
    )