from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProcessingConfig:
    # Sampling
    kinematic_fs_hz: float = 100.0
    analog_fs_hz: float = 1000.0

    # Filtering
    marker_cutoff_hz: float = 6.0
    angle_cutoff_hz: float = 6.0
    kinetic_cutoff_hz: float = 10.0
    grf_cutoff_hz: float = 20.0
    filter_order: int = 4

    # Artifact handling
    max_short_gap_s: float = 0.10
    analog_max_short_gap_s: float = 0.02
    hampel_window_s: float = 0.11
    hampel_n_sigma: float = 3.5


    # Event detection
    min_stance_s: float = 0.25
    min_swing_s: float = 0.18
    contact_on_bw: float = 0.05
    contact_off_bw: float = 0.025
    contact_on_min_n: float = 20.0
    contact_off_min_n: float = 10.0

    # Segmentation
    min_cycle_s: float = 0.65
    max_cycle_s: float = 1.80
    normalized_points: int = 101

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProcessingConfig":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f)

        return cls(
            # Sampling
            kinematic_fs_hz=config["sampling"]["kinematic_fs_hz"],
            analog_fs_hz=config["sampling"]["analog_fs_hz"],

            # Filtering
            marker_cutoff_hz=config["filtering"]["marker_cutoff_hz"],
            angle_cutoff_hz=config["filtering"]["angle_cutoff_hz"],
            kinetic_cutoff_hz=config["filtering"]["kinetic_cutoff_hz"],
            grf_cutoff_hz=config["filtering"]["grf_cutoff_hz"],
            filter_order=config["filtering"]["filter_order"],

            # Artifacts
            max_short_gap_s=config["artifacts"]["max_short_gap_s"],
            hampel_window_s=config["artifacts"]["hampel_window_s"],
            hampel_n_sigma=config["artifacts"]["hampel_n_sigma"],
            analog_max_short_gap_s=config["artifacts"]["analog_max_short_gap_s"],
            
            # Event detection
            min_stance_s=config["events"]["min_stance_s"],
            min_swing_s=config["events"]["min_swing_s"],
            contact_on_bw=config["events"]["contact_on_bw"],
            contact_off_bw=config["events"]["contact_off_bw"],
            contact_on_min_n=config["events"]["contact_on_min_n"],
            contact_off_min_n=config["events"]["contact_off_min_n"],

            # Segmentation
            min_cycle_s=config["segmentation"]["min_cycle_s"],
            max_cycle_s=config["segmentation"]["max_cycle_s"],
            normalized_points=config["segmentation"]["normalized_points"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kinematic_fs_hz": self.kinematic_fs_hz,
            "analog_fs_hz": self.analog_fs_hz,
            "marker_cutoff_hz": self.marker_cutoff_hz,
            "angle_cutoff_hz": self.angle_cutoff_hz,
            "kinetic_cutoff_hz": self.kinetic_cutoff_hz,
            "grf_cutoff_hz": self.grf_cutoff_hz,
            "filter_order": self.filter_order,
            "max_short_gap_s": self.max_short_gap_s,
            "hampel_window_s": self.hampel_window_s,
            "hampel_n_sigma": self.hampel_n_sigma,
            "min_cycle_s": self.min_cycle_s,
            "max_cycle_s": self.max_cycle_s,
            "normalized_points": self.normalized_points,
            "analog_max_short_gap_s": self.analog_max_short_gap_s,
            "min_stance_s": self.min_stance_s,
            "min_swing_s": self.min_swing_s,
            "contact_on_bw": self.contact_on_bw,
            "contact_off_bw": self.contact_off_bw,
            "contact_on_min_n": self.contact_on_min_n,
            "contact_off_min_n": self.contact_off_min_n,
        }