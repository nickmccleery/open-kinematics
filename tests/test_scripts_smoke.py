"""
Smoke tests for the repo-level helper scripts.

These guard ``scripts/plot_bump_sweep.py`` and the root-level
``visualize_camber_shim.py`` against import rot after IO/schema refactors.
The import checks and the headless camber-shim render are fast and run in
the default suite. The full end-to-end script runs (which need ffmpeg and
write PNG/MP4 artifacts) are marked ``manual``.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pytest

# Force a headless backend before any pyplot import triggered by the scripts.
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_BUMP_SWEEP = REPO_ROOT / "scripts" / "plot_bump_sweep.py"
VISUALIZE_CAMBER_SHIM = REPO_ROOT / "visualize_camber_shim.py"


def _load_script(path: Path) -> Any:
    """
    Import a standalone script by file path and return the loaded module.

    The scripts are not importable as package modules, so they are loaded
    directly from their file location. The return is typed ``Any`` because
    the script's top-level symbols are only known at runtime.
    """
    name = f"_script_smoke_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_plot_bump_sweep_imports() -> None:
    module = _load_script(PLOT_BUMP_SWEEP)
    assert callable(module.main)


def test_visualize_camber_shim_imports() -> None:
    module = _load_script(VISUALIZE_CAMBER_SHIM)
    assert callable(module.main)
    assert callable(module.plot_front_view_comparison)


def test_camber_shim_front_view_renders(tmp_path: Path, test_data_dir: Path) -> None:
    """
    Exercise the camber-shim comparison plot headlessly (no ffmpeg needed).
    """
    from kinematics.io import load_geometry

    module = _load_script(VISUALIZE_CAMBER_SHIM)
    suspension = load_geometry(test_data_dir / "geometry.yaml")

    out = tmp_path / "comparison.png"
    module.plot_front_view_comparison(suspension, suspension, out, shim_delta=0.0)

    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.manual
def test_plot_bump_sweep_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Run the full bump-sweep script, including the MP4 animation.

    Marked ``manual`` because it needs ffmpeg and writes large artifacts;
    outputs are redirected into ``tmp_path`` so nothing lands in the repo.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    module = _load_script(PLOT_BUMP_SWEEP)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    module.main()

    assert (tmp_path / "dashboard.png").exists()
    assert (tmp_path / "bump_sweep.mp4").exists()
