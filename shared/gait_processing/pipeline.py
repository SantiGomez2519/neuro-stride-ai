from pathlib import Path
import pandas as pd

from .cleaning import clean_kinematic_trial, clean_analog_trial
from .events import detect_gait_events
from .segmentation import build_gait_cycles, time_normalize_cycles
from .features import compute_cycle_features


def process_trial(kin, ana, subject, trial_id, cfg, variables):
    kin_clean = clean_kinematic_trial(kin, cfg)
    ana_clean = clean_analog_trial(ana, cfg)
    events, contacts, thresholds = detect_gait_events(ana_clean, float(subject.mass_kg), cfg)
    cycles = build_gait_cycles(events, cfg)
    normalized = time_normalize_cycles(kin_clean, cycles, variables, cfg)
    features = compute_cycle_features(kin_clean, ana_clean, contacts, cycles, float(subject.mass_kg))
    for frame in (events, cycles, normalized, features):
        frame['trial_id'] = trial_id
        frame['subject_id'] = subject.subject_id
        frame['cohort'] = subject.cohort
        frame['affected_side'] = subject.affected_side
    return {
        'kin_clean': kin_clean,
        'ana_clean': ana_clean,
        'events': events,
        'contacts': contacts,
        'cycles': cycles,
        'normalized': normalized,
        'features': features,
        'thresholds': thresholds,
    }
