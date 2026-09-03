from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ProcessingConfig


@dataclass
class SegmentationReport:
    """Summary of gait-cycle segmentation for one trial."""

    candidate_cycles: int
    accepted_cycles: int
    rejected_too_short: int
    rejected_too_long: int

    left_cycles: int
    right_cycles: int


def build_gait_cycles(
    events: pd.DataFrame,
    cfg: ProcessingConfig,
) -> tuple[pd.DataFrame, SegmentationReport]:
    """
    Build gait cycles from consecutive same-side Heel Strike events.

    A gait cycle is defined from Heel Strike i to Heel Strike i+1
    for the same side.

    Parameters
    ----------
    events
        Event table containing at least:
        - side
        - event
        - time_s
    cfg
        Processing configuration containing min_cycle_s and max_cycle_s.

    Returns
    -------
    cycles
        One row per candidate gait cycle. Invalid cycles are retained
        with valid=False and a rejection_reason.
    report
        Segmentation summary.
    """
    required_columns = {
        "side",
        "event",
        "time_s",
    }

    missing_columns = (
        required_columns
        - set(events.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required event columns: "
            f"{sorted(missing_columns)}"
        )

    if cfg.min_cycle_s <= 0:
        raise ValueError(
            "min_cycle_s must be greater than zero."
        )

    if cfg.max_cycle_s <= cfg.min_cycle_s:
        raise ValueError(
            "max_cycle_s must be greater than min_cycle_s."
        )

    rows = []

    rejected_short = 0
    rejected_long = 0

    side_cycle_counter = {
        "L": 0,
        "R": 0,
    }

    for side in ("L", "R"):

        heel_strikes = (
            events[
                (events["side"] == side)
                & (events["event"] == "Heel Strike")
            ]
            .sort_values("time_s")
            .reset_index(drop=True)
        )

        hs_times = heel_strikes[
            "time_s"
        ].to_numpy(dtype=float)

        for i in range(
            len(hs_times) - 1
        ):

            start_s = float(
                hs_times[i]
            )

            end_s = float(
                hs_times[i + 1]
            )

            duration_s = (
                end_s - start_s
            )

            valid = True
            rejection_reason = None

            if duration_s < cfg.min_cycle_s:
                valid = False
                rejection_reason = "too_short"
                rejected_short += 1

            elif duration_s > cfg.max_cycle_s:
                valid = False
                rejection_reason = "too_long"
                rejected_long += 1

            side_cycle_counter[side] += 1

            cycle_id = (
                f"{side}_cycle_"
                f"{side_cycle_counter[side]:03d}"
            )

            rows.append(
                {
                    "cycle_id": cycle_id,
                    "side": side,
                    "hs_start_s": start_s,
                    "hs_end_s": end_s,
                    "duration_s": duration_s,
                    "valid": valid,
                    "rejection_reason": rejection_reason,
                }
            )

    cycles = pd.DataFrame(
        rows,
        columns=[
            "cycle_id",
            "side",
            "hs_start_s",
            "hs_end_s",
            "duration_s",
            "valid",
            "rejection_reason",
        ],
    )

    accepted = (
        cycles["valid"].sum()
        if not cycles.empty
        else 0
    )

    left_cycles = (
        (
            (cycles["side"] == "L")
            & cycles["valid"]
        ).sum()
        if not cycles.empty
        else 0
    )

    right_cycles = (
        (
            (cycles["side"] == "R")
            & cycles["valid"]
        ).sum()
        if not cycles.empty
        else 0
    )

    report = SegmentationReport(
        candidate_cycles=int(
            len(cycles)
        ),
        accepted_cycles=int(
            accepted
        ),
        rejected_too_short=int(
            rejected_short
        ),
        rejected_too_long=int(
            rejected_long
        ),
        left_cycles=int(
            left_cycles
        ),
        right_cycles=int(
            right_cycles
        ),
    )

    return cycles, report


def extract_cycle(
    kinematic: pd.DataFrame,
    cycle: pd.Series,
) -> pd.DataFrame:
    """
    Extract one gait-cycle time window from kinematic data.

    The cycle includes samples from hs_start_s through hs_end_s.
    """
    if "time_s" not in kinematic.columns:
        raise ValueError(
            "Kinematic dataframe must contain 'time_s'."
        )

    start_s = float(
        cycle["hs_start_s"]
    )

    end_s = float(
        cycle["hs_end_s"]
    )

    if end_s <= start_s:
        raise ValueError(
            "Cycle end time must be greater than start time."
        )

    segment = (
        kinematic[
            (kinematic["time_s"] >= start_s)
            & (kinematic["time_s"] <= end_s)
        ]
        .copy()
        .reset_index(drop=True)
    )

    return segment


def _interpolate_valid_segments(
    time_s: np.ndarray,
    values: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray:
    """
    Interpolate only within contiguous valid signal segments.

    Missing regions are preserved: interpolation does not bridge
    across NaN gaps.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    time_s = np.asarray(
        time_s,
        dtype=float,
    )

    target_time_s = np.asarray(
        target_time_s,
        dtype=float,
    )

    output = np.full(
        target_time_s.shape,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(values)

    padded = np.r_[
        False,
        valid,
        False,
    ].astype(int)

    edges = np.diff(padded)

    starts = np.flatnonzero(
        edges == 1
    )

    ends = np.flatnonzero(
        edges == -1
    )

    for start, end in zip(
        starts,
        ends,
    ):

        segment_time = (
            time_s[start:end]
        )

        segment_values = (
            values[start:end]
        )

        if len(segment_time) < 2:
            continue

        mask = (
            (target_time_s >= segment_time[0])
            & (target_time_s <= segment_time[-1])
        )

        if not mask.any():
            continue

        output[mask] = np.interp(
            target_time_s[mask],
            segment_time,
            segment_values,
        )

    return output


def time_normalize_cycle(
    cycle_data: pd.DataFrame,
    normalized_points: int = 101,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """
    Time-normalize one gait cycle to 0-100% gait cycle.

    NaN gaps are preserved; interpolation is performed only within
    contiguous valid regions.

    Parameters
    ----------
    cycle_data
        Kinematic samples for one gait cycle.
    normalized_points
        Number of samples in the normalized representation.
    variables
        Signal columns to normalize. If None, all numeric columns except
        frame and time_s are used.

    Returns
    -------
    pd.DataFrame
        Normalized cycle with columns:
        gait_pct + selected variables.
    """
    if normalized_points < 2:
        raise ValueError(
            "normalized_points must be at least 2."
        )

    if "time_s" not in cycle_data.columns:
        raise ValueError(
            "cycle_data must contain 'time_s'."
        )

    if len(cycle_data) < 2:
        raise ValueError(
            "At least two samples are required "
            "for time normalization."
        )

    time_s = cycle_data[
        "time_s"
    ].to_numpy(dtype=float)

    start_s = float(
        time_s[0]
    )

    end_s = float(
        time_s[-1]
    )

    if end_s <= start_s:
        raise ValueError(
            "Cycle duration must be greater than zero."
        )

    target_time_s = np.linspace(
        start_s,
        end_s,
        normalized_points,
    )

    gait_pct = np.linspace(
        0.0,
        100.0,
        normalized_points,
    )

    if variables is None:
        excluded = {
            "frame",
            "time_s",
        }

        variables = [
            column
            for column in cycle_data.columns
            if (
                column not in excluded
                and pd.api.types.is_numeric_dtype(
                    cycle_data[column]
                )
            )
        ]

    missing_variables = [
        variable
        for variable in variables
        if variable not in cycle_data.columns
    ]

    if missing_variables:
        raise ValueError(
            "Missing variables for normalization: "
            f"{missing_variables}"
        )

    normalized = pd.DataFrame(
        {
            "gait_pct": gait_pct,
        }
    )

    for variable in variables:
        normalized[variable] = (
            _interpolate_valid_segments(
                time_s=time_s,
                values=cycle_data[
                    variable
                ].to_numpy(dtype=float),
                target_time_s=target_time_s,
            )
        )

    return normalized


def time_normalize_cycles(
    kinematic: pd.DataFrame,
    cycles: pd.DataFrame,
    cfg: ProcessingConfig,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract and normalize all accepted gait cycles.

    Returns one long dataframe containing cycle_id, side, gait_pct,
    and normalized biomechanical variables.
    """
    required_cycle_columns = {
        "cycle_id",
        "side",
        "hs_start_s",
        "hs_end_s",
        "valid",
    }

    missing_columns = (
        required_cycle_columns
        - set(cycles.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required cycle columns: "
            f"{sorted(missing_columns)}"
        )

    normalized_cycles = []

    valid_cycles = cycles[
        cycles["valid"]
    ]

    for _, cycle in valid_cycles.iterrows():

        segment = extract_cycle(
            kinematic,
            cycle,
        )

        if len(segment) < 2:
            continue

        normalized = time_normalize_cycle(
            segment,
            normalized_points=cfg.normalized_points,
            variables=variables,
        )

        normalized.insert(
            0,
            "side",
            cycle["side"],
        )

        normalized.insert(
            0,
            "cycle_id",
            cycle["cycle_id"],
        )

        normalized_cycles.append(
            normalized
        )

    if not normalized_cycles:
        return pd.DataFrame()

    return pd.concat(
        normalized_cycles,
        ignore_index=True,
    )
