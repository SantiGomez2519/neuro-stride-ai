# neuro-stride-ai

Gait analysis MLOps pipeline for synthetic and real Vicon motion-capture data.

## Project Structure

```
neuro-stride-ai/
├── shared/                          # Shared Python package
│   ├── pyproject.toml
│   └── gait_processing/             # Core processing modules
│       ├── config.py                # YAML configuration management
│       ├── io.py                    # Data loading (TrialData, load_trial)
│       ├── quality.py               # Signal integrity checks
│       ├── cleaning.py              # Gap interpolation, Hampel filtering
│       ├── events.py                # GRF-based gait event detection
│       ├── segmentation.py          # Gait cycle building, time normalization
│       ├── features.py              # Biomechanical feature extraction
│       ├── normative.py             # Normative reference building
│       ├── pipeline.py              # Full orchestration
│       ├── reporting.py             # Visualization plots
│       └── c3d_adapter.py           # Optional C3D file support
│
├── config/
│   └── processing.yaml              # Processing parameters
│
├── 1-experimentation/               # Phase 1: Notebooks
│   ├── pyproject.toml
│   ├── data/
│   │   ├── raw/                     # Raw Vicon CSV data (gitignored)
│   │   └── README.md
│   └── notebooks/
│       ├── 1_0_setup_configuration.ipynb
│       ├── 1_1_data_raw_profiling_quality.ipynb
│       ├── 1_2_data_raw_preprocessing.ipynb
│       ├── 1_3_data_preprocessed_eda.ipynb
│       ├── 1_4_data_preprocessed_feature_engineering.ipynb
│       ├── 1_5_data_featured_training.ipynb
│       └── 1_6_data_featured_evaluation.ipynb
```

## Quick Start

```bash
cd 1-experimentation
uv sync
uv run jupyter lab
```

Then run notebooks in order: `1_0` -> `1_1` -> `1_2` -> ...
