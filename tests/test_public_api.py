# tests/test_public_api.py
import importlib
import sys


def test_public_api_exports():
    km = importlib.import_module("kinematics")
    expected = {
        "load_geometry",
        "solve",
        "solve_sweep",
        "PointID",
        "PointTarget",
        "SweepConfig",
        "Constraint",
        "SolverConfig",
    }
    assert expected.issubset(set(dir(km)))


def test_no_matplotlib_import_on_core():
    # ensure matplotlib is not pulled in by importing kinematics
    sys.modules.pop("matplotlib", None)
    import importlib

    importlib.invalidate_caches()
    importlib.import_module("kinematics")  # noqa: F401
    assert "matplotlib" not in sys.modules
