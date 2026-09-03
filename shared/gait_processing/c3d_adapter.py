"""Optional adapter for real Vicon C3D files.

Install `ezc3d` and adapt the channel map in config/channel_map.yaml. The synthetic
repository uses CSV so the complete example runs without proprietary files.
"""
from pathlib import Path


def load_c3d(path):
    try:
        import ezc3d
    except ImportError as exc:
        raise ImportError('Install optional dependency: pip install ezc3d') from exc
    c3d = ezc3d.c3d(str(Path(path)))
    return c3d
