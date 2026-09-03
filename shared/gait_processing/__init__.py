"""Gait preprocessing utilities for Proyecto G8."""

from .config import ProcessingConfig
from .io import TrialData, load_trial

from .quality import (
    SignalIntegrity,
    TrialQualityResult,
    check_trial_quality,
)

