"""
Тесты точности численных методов на задачах с ИЗВЕСТНЫМ аналитическим
решением (область «d»). Полученный результат сравнивается с точным.

Покрыты:
  * первый порядок:  y' = -y           → e^{-x}
  * первый порядок:  y' =  y           → e^{x}
  * система 1-го пор: x1'=x2, x2'=-x1  → sin / cos
  * второй порядок:  y'' = 0           → линейная   y = x
  * второй порядок:  y'' = 6x          → кубическая  y = x^3
  * второй порядок:  y'' - y = 0       → sinh(x)
  * второй порядок:  y'' + y = 0       → sin(x)
"""

import numpy as np
import pytest
from context import *  # noqa: F401, F403
from dataset import Dataset
from solver import ContinuationSolver


def _solve(equations, bcs, guess, x_end=1.0, n=200, x_start=0.0, tol=1e-10):
    d = Dataset(name="acc", x_start=x_start, x_end=x_end,
                equations=equations, boundary_conditions=bcs,
                n_points=n, tol=tol)
    s = ContinuationSolver(d)
    return s.solve(np.array(guess, dtype=float))


class TestFirstOrder:
    def test_exponential_decay(self):
        x, y = _solve(["Derivative(x1(t), t) + x1 = 0"],
                      ["x1(a) = 1"], [1.0], x_end=2.0)
        assert np.max(np.abs(y[0] - np.exp(-x))) < 1e-6

    def test_exponential_growth(self):
        x, y = _solve(["Derivative(x1(t), t) - x1 = 0"],
                      ["x1(a) = 1"], [1.0], x_end=2.0)
        assert np.max(np.abs(y[0] - np.exp(x))) < 1e-6

    def test_harmonic_system_sin_cos(self):
        # x1'=x2, x2'=-x1, x1(0)=0, x2(0)=1 → x1=sin, x2=cos
        x, y = _solve(
            ["Derivative(x1(t), t) - x2 = 0",
             "Derivative(x2(t), t) + x1 = 0"],
            ["x1(a) = 0", "x2(a) = 1"], [0.0, 1.0], x_end=6.0, n=120)
        assert np.max(np.abs(y[0] - np.sin(x))) < 1e-6
        assert np.max(np.abs(y[1] - np.cos(x))) < 1e-6


class TestSecondOrder:
    def test_linear(self):
        # y''=0, y(0)=0, y(1)=1 → y=x
        x, y = _solve(["Derivative(y(x), x, 2) = 0"],
                      ["y0", "y1 - 1"], [0.0, 1.0])
        assert np.max(np.abs(y[0] - x)) < 1e-9

    def test_cubic(self):
        # y''=6x, y(0)=0, y(1)=1 → y=x^3
        x, y = _solve(["Derivative(y(x), x, 2) - 6*x = 0"],
                      ["y0", "y1 - 1"], [0.0, 0.0])
        assert np.max(np.abs(y[0] - x ** 3)) < 1e-9

    def test_sinh(self):
        # y''-y=0, y(0)=0, y'(0)=1 → sinh(x)
        x, y = _solve(["Derivative(y(x), x, 2) - y(x) = 0"],
                      ["y0", "dy0 - 1"], [0.0, 1.0], x_end=2.0)
        assert np.max(np.abs(y[0] - np.sinh(x))) < 1e-6

    def test_sin_full_period(self):
        # y''+y=0, y(0)=0, y'(0)=1 → sin(x) на [0, 2π]
        x, y = _solve(["Derivative(y(x), x, 2) + y(x) = 0"],
                      ["y0", "dy0 - 1"], [0.0, 1.0], x_end=2 * np.pi)
        assert np.max(np.abs(y[0] - np.sin(x))) < 1e-6

    def test_dirichlet_sin_half_wave(self):
        # y''+y=0, y(0)=0, y(π/2)=1 → sin(x); проверяем краевую (не Коши) задачу
        x, y = _solve(["Derivative(y(x), x, 2) + y(x) = 0"],
                      ["y0", "y1 - 1"], [0.0, 1.0], x_end=np.pi / 2)
        assert np.max(np.abs(y[0] - np.sin(x))) < 1e-6
