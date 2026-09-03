from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ProcessingConfig


@dataclass
class EventDetectionReport:
    """Summary of gait-event detection for one trial."""

    threshold_on_N: float
    threshold_off_N: float

    n_hs_left: int
    n_to_left: int
    n_hs_right: int
    n_to_right: int

    contact_accuracy_left: float | None
    contact_accuracy_right: float | None


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True runs as (start, end) index pairs."""
    padded = np.r_[False, mask, False].astype(int)
    edges = np.diff(padded)

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    return list(zip(starts, ends))


def _remove_short_runs(
    binary: np.ndarray,
    min_true: int,
    min_false: int,
) -> np.ndarray:
    """
    Remove physiologically implausible short contact/swing runs.

    Short True runs are converted to False and short False runs
    are converted to True.
    """
    x = np.asarray(binary, dtype=bool).copy()

    for value, min_len in (
        (True, min_true),
        (False, min_false),
    ):
        for start, end in _boolean_runs(x == value):
            if end - start < min_len:
                x[start:end] = not value

    return x


def hysteresis_contact(
    fz: np.ndarray,
    threshold_on: float,
    threshold_off: float,
) -> np.ndarray:
    """
    Estimate foot contact from vertical GRF using hysteresis.

    Contact starts when force reaches threshold_on and ends when
    it drops to threshold_off. threshold_off must be lower than
    threshold_on.
    """
    if threshold_off >= threshold_on:
        raise ValueError(
            "threshold_off must be lower than threshold_on"
        )

    state = False
    contact = np.zeros(len(fz), dtype=bool)

    for i, value in enumerate(
        np.nan_to_num(fz, nan=0.0)
    ):
        if not state and value >= threshold_on:
            state = True
        elif state and value <= threshold_off:
            state = False

        contact[i] = state

    return contact


def contact_to_events(
    time_s: np.ndarray,
    sample: np.ndarray,
    contact: np.ndarray,
    side: str,
    source: str,
) -> pd.DataFrame:
    """Convert a binary contact sequence into Heel Strike / Toe Off events."""
    if len(contact) == 0:
        return pd.DataFrame(
            columns=[
                "side",
                "event",
                "time_s",
                "sample",
                "source",
            ]
        )

    contact_int = np.asarray(contact, dtype=int)

    transitions = np.diff(
        contact_int,
        prepend=contact_int[0],
    )

    rows = []

    for idx in np.flatnonzero(transitions == 1):
        rows.append(
            {
                "side": side,
                "event": "Heel Strike",
                "time_s": float(time_s[idx]),
                "sample": int(sample[idx]),
                "source": source,
            }
        )

    for idx in np.flatnonzero(transitions == -1):
        rows.append(
            {
                "side": side,
                "event": "Toe Off",
                "time_s": float(time_s[idx]),
                "sample": int(sample[idx]),
                "source": source,
            }
        )

    return pd.DataFrame(rows)


def _contact_accuracy(
    detected: np.ndarray,
    truth: pd.Series | None,
) -> float | None:
    """Compute sample-level contact accuracy when ground truth exists."""
    if truth is None:
        return None

    truth_array = truth.to_numpy(dtype=int)

    if len(truth_array) != len(detected):
        raise ValueError(
            "Detected and ground-truth contact arrays "
            "must have the same length."
        )

    return float(
        np.mean(
            np.asarray(detected, dtype=int)
            == truth_array
        )
    )


def detect_gait_events(
    analog: pd.DataFrame,
    mass_kg: float,
    cfg: ProcessingConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    EventDetectionReport,
]:
    """
    Detect left/right Heel Strike and Toe Off events from vertical GRF.

    Detection procedure
    -------------------
    1. Compute force thresholds using body weight and absolute minima.
    2. Estimate contact with hysteresis.
    3. Remove unrealistically short stance/swing intervals.
    4. Convert contact transitions into HS/TO events.
    5. If synthetic ground-truth contact labels are present, compute
       sample-level contact accuracy.

    Required analog columns
    -----------------------
    - time_s
    - sample
    - L_GRF_V_N
    - R_GRF_V_N

    Optional synthetic validation columns
    -------------------------------------
    - L_contact_true
    - R_contact_true
    """
    required_columns = {
        "time_s",
        "sample",
        "L_GRF_V_N",
        "R_GRF_V_N",
    }

    missing_columns = (
        required_columns
        - set(analog.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required analog columns for event detection: "
            f"{sorted(missing_columns)}"
        )

    if mass_kg <= 0:
        raise ValueError(
            "mass_kg must be greater than zero."
        )

    body_weight_N = float(mass_kg) * 9.81

    threshold_on = max(
        cfg.contact_on_min_n,
        cfg.contact_on_bw * body_weight_N,
    )

    threshold_off = max(
        cfg.contact_off_min_n,
        cfg.contact_off_bw * body_weight_N,
    )

    if threshold_off >= threshold_on:
        raise ValueError(
            "Configured contact-off threshold must be lower "
            "than contact-on threshold."
        )

    min_stance = max(
        1,
        int(
            round(
                cfg.min_stance_s
                * cfg.analog_fs_hz
            )
        ),
    )

    min_swing = max(
        1,
        int(
            round(
                cfg.min_swing_s
                * cfg.analog_fs_hz
            )
        ),
    )

    time_s = analog["time_s"].to_numpy()
    sample = analog["sample"].to_numpy()

    contacts = {
        "time_s": time_s,
        "sample": sample,
    }

    event_frames = []
    detected_contacts = {}

    for side in ("L", "R"):
        fz = analog[
            f"{side}_GRF_V_N"
        ].to_numpy(dtype=float)

        contact = hysteresis_contact(
            fz,
            threshold_on=threshold_on,
            threshold_off=threshold_off,
        )

        contact = _remove_short_runs(
            contact,
            min_true=min_stance,
            min_false=min_swing,
        )

        detected_contacts[side] = contact
        contacts[f"{side}_contact"] = contact.astype(int)

        side_events = contact_to_events(
            time_s=time_s,
            sample=sample,
            contact=contact,
            side=side,
            source="GRF_hysteresis",
        )

        if not side_events.empty:
            event_frames.append(side_events)

    if event_frames:
        events = (
            pd.concat(
                event_frames,
                ignore_index=True,
            )
            .sort_values(
                ["time_s", "side"],
            )
            .reset_index(drop=True)
        )
    else:
        events = pd.DataFrame(
            columns=[
                "side",
                "event",
                "time_s",
                "sample",
                "source",
            ]
        )

    contact_df = pd.DataFrame(contacts)

    left_truth = (
        analog["L_contact_true"]
        if "L_contact_true" in analog.columns
        else None
    )

    right_truth = (
        analog["R_contact_true"]
        if "R_contact_true" in analog.columns
        else None
    )

    report = EventDetectionReport(
        threshold_on_N=float(threshold_on),
        threshold_off_N=float(threshold_off),

        n_hs_left=int(
            (
                (events["side"] == "L")
                & (events["event"] == "Heel Strike")
            ).sum()
        ),
        n_to_left=int(
            (
                (events["side"] == "L")
                & (events["event"] == "Toe Off")
            ).sum()
        ),
        n_hs_right=int(
            (
                (events["side"] == "R")
                & (events["event"] == "Heel Strike")
            ).sum()
        ),
        n_to_right=int(
            (
                (events["side"] == "R")
                & (events["event"] == "Toe Off")
            ).sum()
        ),

        contact_accuracy_left=_contact_accuracy(
            detected_contacts["L"],
            left_truth,
        ),
        contact_accuracy_right=_contact_accuracy(
            detected_contacts["R"],
            right_truth,
        ),
    )

    return events, contact_df, report


def event_timing_error(
    detected: pd.DataFrame,
    truth: pd.DataFrame,
    max_match_error_s: float = 0.12,
) -> pd.DataFrame:
    """
    Match detected events against reference events and compute timing error.

    Matching is performed independently by side and event type using
    nearest unmatched events within max_match_error_s.
    """
    required = {
        "side",
        "event",
        "time_s",
    }

    for name, df in (
        ("detected", detected),
        ("truth", truth),
    ):
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: "
                f"{sorted(missing)}"
            )

    rows = []

    for side in ("L", "R"):
        for event in (
            "Heel Strike",
            "Toe Off",
        ):
            detected_times = (
                detected.query(
                    "side == @side and event == @event"
                )["time_s"]
                .to_numpy(dtype=float)
            )

            truth_times = (
                truth.query(
                    "side == @side and event == @event"
                )["time_s"]
                .to_numpy(dtype=float)
            )

            used = set()

            for truth_time in truth_times:
                candidates = [
                    (
                        abs(
                            detected_time
                            - truth_time
                        ),
                        i,
                        detected_time,
                    )
                    for i, detected_time
                    in enumerate(detected_times)
                    if i not in used
                ]

                if not candidates:
                    continue

                error_abs, index, detected_time = min(
                    candidates
                )

                if error_abs <= max_match_error_s:
                    used.add(index)

                    rows.append(
                        {
                            "side": side,
                            "event": event,
                            "truth_s": float(
                                truth_time
                            ),
                            "detected_s": float(
                                detected_time
                            ),
                            "error_ms": float(
                                1000
                                * (
                                    detected_time
                                    - truth_time
                                )
                            ),
                        }
                    )

    return pd.DataFrame(rows)
