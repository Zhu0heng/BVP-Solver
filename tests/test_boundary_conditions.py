"""
Тесты граничных условий (область «a» и часть «c» из задания):

  * нормализация записи xi(a)/xi(b) в форму остатка;
  * корректность вычисляемых остаточных функций;
  * Дирихле / Неймана / смешанные / связные условия;
  * соответствие ЧИСЛА граничных условий размерности системы.
"""

import pytest
import numpy as np
from context import *  # noqa: F401, F403
from dataset import Dataset
from solver import ContinuationSolver
from parser import (
    parse_equation_system,
    parse_boundary_conditions,
    _normalize_bc,
)
from task_io import get_example_tasks


class TestBCNormalization:
    """Запись 'xi(a) = rhs' преобразуется в остаток 'xi0 - (rhs)'."""

    def test_dirichlet_start_zero(self):
        assert _normalize_bc("x2(a) = 0") == "x20 - (0)"

    def test_dirichlet_start_const(self):
        assert _normalize_bc("x1(a) = 2") == "x10 - (2)"

    def test_dirichlet_end_const(self):
        assert _normalize_bc("x1(b) = 1.5") == "x11 - (1.5)"

    def test_coupled_bc_endpoints(self):
        # связное условие: x1 на конце равен x4 на конце
        assert _normalize_bc("x1(b) = x4(b)") == "x11 - (x41)"

    def test_coupled_bc_start(self):
        assert _normalize_bc("x1(a) = x4(a)") == "x10 - (x40)"

    def test_shorthand_passthrough(self):
        # уже-остаточная краткая форма не меняется
        assert _normalize_bc("y0") == "y0"
        assert _normalize_bc("dy0 - 1") == "dy0 - 1"


class TestBCResiduals:
    """Разобранные ГУ возвращают правильные числовые остатки."""

    def _funcs(self, bc_list, names):
        return parse_boundary_conditions(bc_list, bc_var_names=names)

    def test_second_order_residuals(self):
        names = ['y0', 'dy0', 'y1', 'dy1']
        f = self._funcs(["y0", "dy0 - 1", "y1 - 1", "dy1"], names)
        # сигнатура lambdify: (y0, dy0, y1, dy1, lambda)
        assert f[0](3, 0, 0, 0, 0) == 3        # y0
        assert f[1](0, 5, 0, 0, 0) == 4        # dy0 - 1
        assert f[2](0, 0, 7, 0, 0) == 6        # y1 - 1
        assert f[3](0, 0, 0, 9, 0) == 9        # dy1

    def test_first_order_residuals(self):
        # размерность из самого парсера, чтобы имена совпадали
        res = parse_equation_system([
            "Derivative(x1(t), t) - x2 = 0",
            "Derivative(x2(t), t) + x1 = 0",
        ])
        names = res[6]
        assert names == ['x10', 'x11', 'x20', 'x21']
        f = self._funcs(["x1(a) = 0", "x2(a) = 1"], names)
        # сигнатура: (x10, x11, x20, x21, lambda)
        assert f[0](5, 0, 0, 0, 0) == 5        # x10 - 0
        assert f[1](0, 0, 3, 0, 0) == 2        # x20 - 1

    def test_coupled_residual_value(self):
        names = ['x10', 'x11', 'x40', 'x41']
        f = self._funcs(["x1(b) = x4(b)"], names)
        # x11 - x41
        assert f[0](0, 8, 0, 3, 0) == 5

    def test_empty_bc_rejected(self):
        with pytest.raises(ValueError):
            parse_boundary_conditions([], bc_var_names=['y0', 'dy0', 'y1', 'dy1'])
        with pytest.raises(ValueError):
            parse_boundary_conditions(["   "], bc_var_names=['y0', 'dy0', 'y1', 'dy1'])


class TestBCCountVsDimension:
    """Число граничных условий должно соответствовать размерности задачи."""

    def test_second_order_needs_two_bc(self):
        d = Dataset(
            name="too few", x_start=0, x_end=1,
            equations=["Derivative(y(x), x, 2) = 0"],
            boundary_conditions=["y0"],          # нужно 2, дано 1
        )
        with pytest.raises(ValueError):
            ContinuationSolver(d)

    def test_first_order_system_needs_n_bc(self):
        d = Dataset(
            name="too few", x_start=0, x_end=1,
            equations=[
                "Derivative(x1(t), t) - x2 = 0",
                "Derivative(x2(t), t) + x1 = 0",
            ],
            boundary_conditions=["x1(a) = 0"],   # нужно 2, дано 1
        )
        with pytest.raises(ValueError):
            ContinuationSolver(d)

    def test_exact_count_ok(self):
        d = Dataset(
            name="ok", x_start=0, x_end=1,
            equations=["Derivative(y(x), x, 2) = 0"],
            boundary_conditions=["y0", "y1 - 1"],
        )
        s = ContinuationSolver(d)
        assert len(s.bc_funcs) == 2

    @pytest.mark.parametrize("idx,n_dep", [(0, 4), (1, 4), (2, 6), (3, 2)])
    def test_examples_have_enough_bc(self, idx, n_dep):
        ds = get_example_tasks()[idx]
        s = ContinuationSolver(ds)          # не должно бросать
        assert s.n_dep == n_dep
        assert s.order == 1
        # для системы 1-го порядка нужно >= n_dep условий
        assert len(ds.boundary_conditions) >= n_dep
