"""
Проверка математической постановки задачи (область «c»):

  * правильно ли определяется размерность и ПОРЯДОК системы;
  * соответствует ли число неизвестных числу уравнений в примерах;
  * какие критерии сходимости используются (ε передаётся из набора данных
    в solve_ivp и в корректор Левенберга–Марквардта).
"""

import numpy as np
import pytest
from context import *  # noqa: F401, F403
from dataset import Dataset
from solver import ContinuationSolver
from parser import parse_equation_system
from task_io import get_example_tasks


class TestSystemDimension:
    def test_second_order_single(self):
        res = parse_equation_system(["Derivative(y(x), x, 2) + y(x) = 0"])
        dep_vars, indep, order = res[4], res[5], res[7]
        assert order == 2
        assert dep_vars == ['y']
        assert indep == 'x'

    def test_first_order_system(self):
        res = parse_equation_system([
            "Derivative(x1(t), t) - x2 = 0",
            "Derivative(x2(t), t) + x1 = 0",
        ])
        dep_vars, indep, order = res[4], res[5], res[7]
        assert order == 1
        assert dep_vars == ['x1', 'x2']
        assert indep == 't'

    @pytest.mark.parametrize("idx,n_dep", [(0, 4), (1, 4), (2, 6), (3, 2)])
    def test_example_dimensions(self, idx, n_dep):
        ds = get_example_tasks()[idx]
        s = ContinuationSolver(ds)
        assert s.order == 1
        assert s.n_dep == n_dep
        # для системы первого порядка число уравнений = размерности
        assert len(ds.equations) == n_dep

    def test_cannot_mix_orders(self):
        # смешивать 1-й и 2-й порядок нельзя
        with pytest.raises(ValueError):
            parse_equation_system([
                "Derivative(y(x), x, 2) + y(x) = 0",
                "Derivative(z(x), x) - y(x) = 0",
            ])

    def test_bc_var_names_match_order(self):
        # 2-й порядок: на переменную приходится 4 граничных имени
        res2 = parse_equation_system(["Derivative(y(x), x, 2) = 0"])
        assert res2[6] == ['y0', 'dy0', 'y1', 'dy1']
        # 1-й порядок: на переменную — 2 имени (начало/конец)
        res1 = parse_equation_system(["Derivative(x1(t), t) - x1 = 0"])
        assert res1[6] == ['x10', 'x11']


class TestConvergenceCriteria:
    """ε (tol) и метод интегрирования должны браться из набора данных."""

    def test_default_tol(self):
        d = Dataset(name="d", x_start=0, x_end=1,
                    equations=["Derivative(y(x), x, 2) = 0"],
                    boundary_conditions=["y0", "y1 - 1"])
        s = ContinuationSolver(d)
        assert s.tol == 1e-9
        assert s.method == 'RK45'

    def test_tol_propagates(self):
        d = Dataset(name="d", x_start=0, x_end=1,
                    equations=["Derivative(y(x), x, 2) = 0"],
                    boundary_conditions=["y0", "y1 - 1"],
                    tol=1e-6, method='Radau')
        s = ContinuationSolver(d)
        assert s.tol == 1e-6
        assert s.method == 'Radau'

    def test_tighter_tol_is_accurate(self):
        # при жёстком ε решение y''+y=0, y(0)=0, y'(0)=1 близко к sin(x)
        d = Dataset(name="sin", x_start=0.0, x_end=2 * np.pi,
                    equations=["Derivative(y(x), x, 2) + y(x) = 0"],
                    boundary_conditions=["y0", "dy0 - 1"],
                    n_points=200, tol=1e-11)
        s = ContinuationSolver(d)
        x, y = s.solve(np.array([0.0, 1.0]))
        err = np.max(np.abs(y[0] - np.sin(x)))
        assert err < 1e-6

    def test_no_silent_naninf(self):
        # Критерий физичности: при расходимости (здесь — взрыв за конечное
        # время x1' = x1^2, решение 1/(1-t) уходит в бесконечность при t=1)
        # решатель обязан ЛИБО бросить RuntimeError, ЛИБО вернуть конечный
        # массив — но никогда не возвращать NaN/Inf молча.
        d = Dataset(name="blowup", x_start=0.0, x_end=2.0,
                    equations=["Derivative(x1(t), t) - x1**2 = 0"],
                    boundary_conditions=["x1(a) = 1"],
                    n_points=80)
        s = ContinuationSolver(d)
        try:
            x, y = s.solve(np.array([1.0]))
        except RuntimeError:
            return                          # расходимость корректно отклонена
        assert np.all(np.isfinite(y))       # либо результат конечен
