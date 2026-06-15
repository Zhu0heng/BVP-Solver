"""
Тесты недопустимого ввода (область «b»):

  * параметры набора данных вне допустимого диапазона отвергаются;
  * пустые / некорректные уравнения и граничные условия отвергаются;
  * уравнение без производной не принимается.
"""

import tokenize
import pytest
from context import *  # noqa: F401, F403
from dataset import Dataset
from solver import ContinuationSolver


_PARSE_ERRORS = (ValueError, SyntaxError, TypeError, tokenize.TokenError)


def _ds(**kw):
    base = dict(
        name="t", x_start=0.0, x_end=1.0,
        equations=["Derivative(y(x), x, 2) = 0"],
        boundary_conditions=["y0", "y1 - 1"],
    )
    base.update(kw)
    return Dataset(**base)


class TestDatasetParameterValidation:
    def test_x_start_not_less_than_x_end(self):
        with pytest.raises(ValueError):
            _ds(x_start=5.0, x_end=1.0)

    def test_x_start_equals_x_end(self):
        with pytest.raises(ValueError):
            _ds(x_start=2.0, x_end=2.0)

    def test_n_points_too_small(self):
        with pytest.raises(ValueError):
            _ds(n_points=1)

    def test_continuation_steps_too_small(self):
        with pytest.raises(ValueError):
            _ds(continuation_steps=0)

    def test_nonpositive_tol(self):
        with pytest.raises(ValueError):
            _ds(tol=0.0)
        with pytest.raises(ValueError):
            _ds(tol=-1e-6)

    def test_empty_equations(self):
        with pytest.raises(ValueError):
            Dataset(name="t", x_start=0.0, x_end=1.0,
                    equations=[], boundary_conditions=["y0", "y1"])


class TestSolverInputValidation:
    def test_empty_boundary_conditions(self):
        d = Dataset(name="t", x_start=0.0, x_end=1.0,
                    equations=["Derivative(y(x), x, 2) = 0"],
                    boundary_conditions=[])
        with pytest.raises(ValueError):
            ContinuationSolver(d)

    def test_malformed_equation(self):
        d = Dataset(name="t", x_start=0.0, x_end=1.0,
                    equations=["this is not @# math"],
                    boundary_conditions=["y0", "y1"])
        with pytest.raises(_PARSE_ERRORS):
            ContinuationSolver(d)

    def test_malformed_boundary(self):
        d = Dataset(name="t", x_start=0.0, x_end=1.0,
                    equations=["Derivative(y(x), x, 2) = 0"],
                    boundary_conditions=["y0", "@@@"])
        with pytest.raises(_PARSE_ERRORS):
            ContinuationSolver(d)

    def test_equation_without_derivative(self):
        # нет производной → невозможно выразить → ValueError
        d = Dataset(name="t", x_start=0.0, x_end=1.0,
                    equations=["y(x) = 0"],
                    boundary_conditions=["y0", "y1"])
        with pytest.raises(_PARSE_ERRORS):
            ContinuationSolver(d)

    def test_too_few_boundary_conditions(self):
        d = Dataset(name="t", x_start=0.0, x_end=1.0,
                    equations=["Derivative(y(x), x, 2) = 0"],
                    boundary_conditions=["y0"])     # 2-й порядок требует 2
        with pytest.raises(ValueError):
            ContinuationSolver(d)
