from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd


@dataclass
class TrialData:
    subject_id: str
    trial_id: str

    trajectories: pd.DataFrame = field(repr=False)
    analog: pd.DataFrame = field(repr=False)
    events: pd.DataFrame | None = field(repr=False)
    artifacts: pd.DataFrame | None = field(repr=False)

    subject: pd.Series = field(repr=False)
    trial: pd.Series = field(repr=False)

    @property
    def kinematic_fs_hz(self) -> float:
        return float(self.trial["kinematic_sampling_hz"])

    @property
    def analog_fs_hz(self) -> float:
        return float(self.trial["analog_sampling_hz"])

    @property
    def duration_s(self) -> float:
        return float(self.trial["duration_s"])

    @property
    def mass_kg(self) -> float:
        return float(self.subject["mass_kg"])

    @property
    def height_m(self) -> float:
        return float(self.subject["height_m"])

    @property
    def age_years(self) -> float:
        return float(self.subject["age_years"])

    @property
    def cohort(self) -> str:
        return str(self.subject["cohort"])

    @property
    def condition(self) -> str:
        return str(self.trial["condition"])


def load_trial(
    root: str | Path,
    subject_id: str,
    trial_id: str,
) -> TrialData:

    root = Path(root)

    # -------------------------
    # Paths
    # -------------------------
    trial_dir = (
        root
        / subject_id
        / "Vicon"
        / trial_id
    )

    metadata_dir = root / "metadata"

    if not trial_dir.exists():
        raise FileNotFoundError(
            f"Trial directory not found: {trial_dir}"
        )

    # -------------------------
    # Signals
    # -------------------------
    trajectories = pd.read_csv(
        trial_dir / "trajectories.csv"
    )

    analog = pd.read_csv(
        trial_dir / "analog.csv"
    )

    # Optional files
    events_path = trial_dir / "events.csv"
    artifacts_path = trial_dir / "artifacts.csv"

    events = (
        pd.read_csv(events_path)
        if events_path.exists()
        else None
    )

    artifacts = (
        pd.read_csv(artifacts_path)
        if artifacts_path.exists()
        else None
    )

    # -------------------------
    # Metadata
    # -------------------------
    subjects = pd.read_csv(
        metadata_dir / "subjects.csv"
    )

    trials = pd.read_csv(
        metadata_dir / "trials.csv"
    )

    subject_rows = subjects[
        subjects["subject_id"] == subject_id
    ]

    if subject_rows.empty:
        raise ValueError(
            f"Subject '{subject_id}' not found in subjects.csv"
        )

    trial_rows = trials[
        (trials["subject_id"] == subject_id)
        & (trials["trial_id"] == trial_id)
    ]

    if trial_rows.empty:
        raise ValueError(
            f"Trial '{subject_id}/{trial_id}' "
            "not found in trials.csv"
        )

    subject_metadata = subject_rows.iloc[0]
    trial_metadata = trial_rows.iloc[0]

    return TrialData(
        subject_id=subject_id,
        trial_id=trial_id,
        trajectories=trajectories,
        analog=analog,
        events=events,
        artifacts=artifacts,
        subject=subject_metadata,
        trial=trial_metadata,
    )