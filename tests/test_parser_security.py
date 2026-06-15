"""
Тесты разбора выражений и валидаторов (область «e»):

  * корректные строки превращаются в выражения / лямбда-функции;
  * НЕДОПУСТИМЫЙ ввод отвергается БЕЗ выполнения кода (ключевое требование
    безопасности): полезная нагрузка вида __import__, open(), доступ к
    атрибутам и строковые литералы не должны иметь побочных эффектов.

Замечание: sympy.parse_expr сам по себе НЕ безопасен (внутри использует
eval с доступом к builtins). Защиту обеспечивает «белый список» AST в
parser._assert_safe_expr — эти тесты его и проверяют.
"""

import ast
import os
import math
import tempfile
import pytest
import numpy as np
from context import *  # noqa: F401, F403
import parser as parser_mod
import solver as solver_mod
import dataset as dataset_mod
import task_io as task_io_mod
from parser import parse_equation_system, parse_boundary_conditions


# ─────────────────────────── корректный ввод ───────────────────────────
class TestValidParsing:
    def test_second_order_rhs(self):
        res = parse_equation_system(["Derivative(y(x), x, 2) + y(x) = 0"])
        ode = res[0][0]
        # сигнатура: (t, y, dy, lambda); rhs = -y
        assert abs(ode(0.0, 1.0, 0.0, 0.0) - (-1.0)) < 1e-12

    def test_first_order_rhs(self):
        res = parse_equation_system(["Derivative(x1(t), t) - exp(x1) = 0"])
        ode = res[0][0]
        # сигнатура: (t, x1, lambda); rhs = exp(x1)
        assert abs(ode(0.0, 0.0, 0.0) - 1.0) < 1e-12

    def test_whitelisted_functions(self):
        # sin, cos, sqrt, log, tan допустимы и вычисляются верно
        res = parse_equation_system(["Derivative(y(x), x, 2) + sin(y(x)) = 0"])
        ode = res[0][0]
        assert abs(ode(0.0, math.pi / 2, 0.0, 0.0) - (-1.0)) < 1e-12

        res2 = parse_equation_system(["Derivative(x1(t), t) - sqrt(x1) = 0"])
        ode2 = res2[0][0]
        assert abs(ode2(0.0, 4.0, 0.0) - 2.0) < 1e-12

    def test_boundary_conditions_callable(self):
        f = parse_boundary_conditions(["y0 - 1", "dy0"],
                                      bc_var_names=['y0', 'dy0', 'y1', 'dy1'])
        assert callable(f[0]) and callable(f[1])
        assert f[0](5, 0, 0, 0, 0) == 4
        assert f[1](0, 3, 0, 0, 0) == 3


# ─────────────────── безопасность: нет выполнения кода ───────────────────
class TestNoCodeExecution:
    @staticmethod
    def _fresh_sentinel():
        path = os.path.join(tempfile.gettempdir(),
                            f"_sec_sentinel_{os.getpid()}.txt")
        if os.path.exists(path):
            os.remove(path)
        return path

    def test_import_payload_in_equation(self):
        sentinel = self._fresh_sentinel()
        try:
            eqs = ["Derivative(y(x), x, 2) + __import__('os')"
                   ".system('echo pwned') = 0"]
            with pytest.raises(ValueError):
                parse_equation_system(eqs)
            assert not os.path.exists(sentinel)
        finally:
            if os.path.exists(sentinel):
                os.remove(sentinel)

    def test_open_payload_in_equation(self):
        sentinel = self._fresh_sentinel()
        try:
            payload = "open(r'%s','w')" % sentinel.replace('\\', '\\\\')
            eqs = ["Derivative(y(x), x, 2) + (%s) = 0" % payload]
            with pytest.raises(ValueError):
                parse_equation_system(eqs)
            # САМОЕ ВАЖНОЕ: файл не создан → код не выполнялся
            assert not os.path.exists(sentinel)
        finally:
            if os.path.exists(sentinel):
                os.remove(sentinel)

    def test_open_payload_in_boundary(self):
        sentinel = self._fresh_sentinel()
        try:
            payload = ["open(r'%s','w')" % sentinel.replace('\\', '\\\\')]
            with pytest.raises(ValueError):
                parse_boundary_conditions(
                    payload, bc_var_names=['y0', 'dy0', 'y1', 'dy1'])
            assert not os.path.exists(sentinel)
        finally:
            if os.path.exists(sentinel):
                os.remove(sentinel)

    def test_attribute_access_rejected(self):
        with pytest.raises(ValueError):
            parse_equation_system(
                ["Derivative(y(x), x, 2) + y(x).foo = 0"])

    def test_string_literal_rejected(self):
        with pytest.raises(ValueError):
            parse_boundary_conditions(
                ["'a string'"], bc_var_names=['y0', 'dy0', 'y1', 'dy1'])

    def test_dunder_name_rejected(self):
        with pytest.raises(ValueError):
            parse_boundary_conditions(
                ["__builtins__"], bc_var_names=['y0', 'dy0', 'y1', 'dy1'])

    def test_unknown_function_call_rejected(self):
        with pytest.raises(ValueError):
            parse_equation_system(
                ["Derivative(y(x), x, 2) + eval(y(x)) = 0"])


# ─────────────── статическая проверка: нет eval/exec в коде ───────────────
class TestNoEvalInSource:
    @staticmethod
    def _dangerous_calls(path):
        with open(path, 'r', encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {'eval', 'exec', 'compile', '__import__'}:
                    bad.append(node.func.id)
        return bad

    @pytest.mark.parametrize("mod", [
        parser_mod, solver_mod, dataset_mod, task_io_mod])
    def test_no_dangerous_builtins(self, mod):
        assert self._dangerous_calls(mod.__file__) == []
