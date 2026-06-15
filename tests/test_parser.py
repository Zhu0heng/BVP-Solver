import pytest
import sympy as sp
import tokenize
from context import *  # noqa: F401, F403
from dataset import Dataset
from solver import ContinuationSolver


class TestEquationParser:
    def test_parse_sin_equation(self):
        d = Dataset(
            name="sin(x)", x_start=0, x_end=12.566,
            equations=["Derivative(y(x), x, 2) + y(x) = 0"],
            boundary_conditions=["y0", "dy0 - 1"]
        )
        s = ContinuationSolver(d)
        assert s.n_dep == 1
        derivs = s._ode_system(0.0, [1.0, 0.0], 0.0)
        assert abs(derivs[1] - (-1.0)) < 1e-10

    def test_parse_dirichlet(self):
        d = Dataset(
            name="test", x_start=0, x_end=1,
            equations=["Derivative(y(x), x, 2) = 0"],
            boundary_conditions=["y0", "y1 - 1"]
        )
        s = ContinuationSolver(d)
        assert len(s.bc_funcs) == 2

    def test_parse_neumann(self):
        d = Dataset(
            name="test", x_start=0, x_end=1,
            equations=["Derivative(y(x), x, 2) = 0"],
            boundary_conditions=["dy0", "dy1"]
        )
        s = ContinuationSolver(d)
        assert len(s.bc_funcs) == 2

    def test_bad_equation_raises(self):
        d = Dataset(
            name="test", x_start=0, x_end=1,
            equations=["not math !!!"],
            boundary_conditions=["y0", "y1"]
        )
        with pytest.raises((SyntaxError, ValueError, TypeError, tokenize.TokenError)):
            ContinuationSolver(d)

    def test_bad_boundary_raises(self):
        d = Dataset(
            name="test", x_start=0, x_end=1,
            equations=["Derivative(y(x), x, 2) = 0"],
            boundary_conditions=["y0", "!!!"]
        )
        with pytest.raises((SyntaxError, ValueError, TypeError, tokenize.TokenError)):
            ContinuationSolver(d)
