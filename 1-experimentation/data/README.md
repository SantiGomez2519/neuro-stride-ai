# Vicon G8 dataset — reorganized structure

This version reorganizes the original synthetic dataset into a subject/trial acquisition hierarchy.

## Structure

```text
Dataset_Vicon_G8_realistic_v1_1/
├── metadata/
│   ├── subjects.csv
│   ├── trials.csv
│   └── data_dictionary.csv
├── Sub01/
│   └── Vicon/
│       ├── trial0001/
│       │   ├── trajectories.csv
│       │   ├── analog.csv
│       │   ├── events.csv
│       │   └── artifacts.csv
│       └── trial0002/
│           └── ...
└── SubXX/
    └── Vicon/
        └── ...
```

## Metadata model

- `metadata/subjects.csv`: one row per subject; demographic and clinical descriptors.
- `metadata/trials.csv`: one row per acquisition/trial; sampling rates, condition, duration, gait timing and nominal speed.
- `metadata/data_dictionary.csv`: variable definitions and units.

## Trial files

- `trajectories.csv`: Vicon marker trajectories and biomechanical variables at the kinematic sampling frequency.
- `analog.csv`: analog/force-platform signals at the analog sampling frequency.
- `events.csv`: gait events for that acquisition.
- `artifacts.csv`: synthetic signal-quality artifacts injected into that acquisition.

`subject_id` and `trial_id` were intentionally removed from trial signal files because their identity is encoded by the directory path.

## Identifier convention

Subjects are named `Sub01`, `Sub02`, ... and trials are named `trial0001`, `trial0002`, ... within each subject.

The original synthetic IDs are retained in `source_subject_id` and `source_trial_id` to preserve provenance.

## Important

The data remain synthetic. This reorganization only makes the storage model closer to a real gait-laboratory dataset and improves scalability for future modalities such as IMU and exoskeleton data.
