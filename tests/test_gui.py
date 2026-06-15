import pytest
import numpy as np
from context import *  # noqa: F401, F403
from dataset import Dataset


class TestDatasetValidation:
    def test_valid_dataset_creation(self):
        d = Dataset(
            name="sin(x)", x_start=0, x_end=12.566,
            equations=["Derivative(y(x), x, 2) + y(x) = 0"],
            boundary_conditions=["y0", "dy0 - 1"]
        )
        assert d.name == "sin(x)"

    def test_dataset_defaults(self):
        d = Dataset(
            name="test", x_start=0, x_end=1,
            equations=["eq"], boundary_conditions=[]
        )
        assert d.n_points == 100

    def test_dataset_validation_raises(self):
        with pytest.raises(ValueError):
            Dataset(name="bad", x_start=5, x_end=1, equations=["eq"], boundary_conditions=["y0"])

    def test_empty_boundary_raises(self):
        d = Dataset(name="test", x_start=0, x_end=1, equations=["eq"], boundary_conditions=[])
        with pytest.raises(ValueError):
            from solver import ContinuationSolver
            ContinuationSolver(d)


class TestSolverValidation:
    def test_solve_sin(self):
        from solver import ContinuationSolver
        d = Dataset(
            name="sin(x)", x_start=0, x_end=12.566370614359172,
            equations=["Derivative(y(x), x, 2) + y(x) = 0"],
            boundary_conditions=["y0", "dy0 - 1"], n_points=200
        )
        s = ContinuationSolver(d)
        x, y_all = s.solve(np.array([0.0, 1.0]))
        err = np.max(np.abs(y_all[0] - np.sin(x)))
        assert err < 0.01

    def test_solve_cos(self):
        from solver import ContinuationSolver
        d = Dataset(
            name="cos(x)", x_start=0, x_end=12.566370614359172,
            equations=["Derivative(y(x), x, 2) + y(x) = 0"],
            boundary_conditions=["y0 - 1", "dy0"], n_points=200
        )
        s = ContinuationSolver(d)
        x, y_all = s.solve(np.array([1.0, 0.0]))
        err = np.max(np.abs(y_all[0] - np.cos(x)))
        assert err < 0.01
