"""
Парсер уравнений и граничных условий на основе SymPy.

Безопасный разбор выражений: каждое выражение сначала проверяется по
«белому списку» узлов AST (см. _assert_safe_expr) и лишь затем передаётся
в sympy. Это исключает выполнение произвольного кода (доступ к атрибутам,
строковые литералы, dunder-имена, вызовы небелого списка функций
отвергаются с ValueError). eval() в пользовательском смысле не применяется.
Поддерживает запись в формате Derivative(x1(t), t) - x2 = 0.
Формирует лямбда-функции для численного интегрирования.
"""

import ast
import re
import sympy as sp
import numpy as np


_SYM_FUNCS = {
    'pi': sp.pi, 'E': sp.E,
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
}


# ─────────────────────────────────────────────────────────────────────────
# Безопасный разбор выражений.
#
# ВАЖНО: ``sympy.parse_expr`` внутри вызывает ``eval`` над строкой в окружении,
# куда Python автоматически добавляет встроенные функции (__builtins__).
# Поэтому САМ ПО СЕБЕ он НЕ защищён от вредоносного ввода: строка вида
# "__import__('os').system('...')" или "open('file')" реально ВЫПОЛНИТСЯ.
#
# Чтобы выполнить требование «недопустимый ввод отвергается БЕЗ выполнения
# кода», каждое выражение сначала проходит проверку по «белому списку» узлов
# абстрактного синтаксического дерева (AST): разрешены только числа, имена
# (превращаются в символы), арифметика, степень и ВЫЗОВЫ из белого списка
# функций. Доступ к атрибутам (.system), строковые литералы, индексирование,
# lambda, dunder-имена и любые прочие конструкции отвергаются ДО того, как
# строка попадёт в sympy.
# ─────────────────────────────────────────────────────────────────────────
_ALLOWED_AST_NODES = [
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant,                       # числовые литералы (Python ≥ 3.8)
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
]
if hasattr(ast, 'FloorDiv'):
    _ALLOWED_AST_NODES.append(ast.FloorDiv)
_ALLOWED_AST_NODES = tuple(_ALLOWED_AST_NODES)


def _assert_safe_expr(expr_str, local_dict):
    """Отвергнуть выражение, если оно содержит небезопасные конструкции.

    Разрешённые вызовы — это математические функции из ``_SYM_FUNCS``,
    ``Derivative`` и имена из ``local_dict`` (зависимые переменные и т.п.).
    Любая другая конструкция приводит к ``ValueError`` — БЕЗ выполнения кода.
    """
    allowed_calls = set(local_dict.keys()) | set(_SYM_FUNCS) | {'Derivative'}
    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError as exc:
        raise ValueError(
            f"Синтаксическая ошибка в выражении '{expr_str}': {exc.msg}"
        )
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(
                f"Запрещённая конструкция ({type(node).__name__}) в выражении "
                f"'{expr_str}'. Допустимы только числа, переменные, арифметика "
                f"и функции {sorted(_SYM_FUNCS)}."
            )
        if isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name)
                    or node.func.id not in allowed_calls):
                fname = getattr(node.func, 'id', type(node.func).__name__)
                raise ValueError(
                    f"Недопустимый вызов '{fname}' в выражении '{expr_str}'."
                )
            if getattr(node, 'keywords', None) or getattr(node, 'starargs', None):
                raise ValueError(
                    f"Аргументы-ключи недопустимы в выражении '{expr_str}'."
                )
        elif isinstance(node, ast.Name):
            # dunder / служебные имена (__import__, __builtins__, …) запрещены,
            # кроме внутренних плейсхолдеров, явно присутствующих в local_dict.
            if node.id.startswith('_') and node.id not in local_dict:
                raise ValueError(
                    f"Недопустимое имя '{node.id}' в выражении '{expr_str}'."
                )
        elif hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, complex)):
                raise ValueError(
                    f"Недопустимый литерал {node.value!r} в выражении "
                    f"'{expr_str}'."
                )
    return tree


def _safe_parse_expr(expr_str, local_dict):
    """Проверить выражение по белому списку AST, затем разобрать через sympy."""
    _assert_safe_expr(expr_str, local_dict)
    return sp.parse_expr(expr_str, local_dict=local_dict)


def _detect_order_and_vars(equations):
    if not equations:
        raise ValueError("No equations provided")
    dep_vars = []
    indep_var = None
    has_second = False
    has_first = False
    for eq in equations:
        m2 = re.search(r'Derivative\((\w+)\((\w+)\),\s*(\w+), 2\)', eq)
        if m2:
            has_second = True
            fn_name = m2.group(1)
            if fn_name not in dep_vars:
                dep_vars.append(fn_name)
            if indep_var is None:
                indep_var = m2.group(3)
            continue
        m1 = re.search(r'Derivative\((\w+)\((\w+)\),\s*(\w+)\)', eq)
        if m1:
            has_first = True
            fn_name = m1.group(1)
            if fn_name not in dep_vars:
                dep_vars.append(fn_name)
            if indep_var is None:
                indep_var = m1.group(3)
    if has_second and has_first:
        raise ValueError("Cannot mix first-order and second-order equations")
    if indep_var is None:
        indep_var = 'x'
    if not dep_vars:
        dep_vars = ['y']
        has_second = True
    order = 2 if has_second else 1
    return dep_vars, indep_var, order


def parse_equation_system(equations, param_name='lambda'):
    dep_vars, indep_var, order = _detect_order_and_vars(equations)

    t = sp.symbols(indep_var)
    func_objects = {name: sp.Function(name) for name in dep_vars}
    lmbda = sp.symbols(param_name)
    safe_param = '_param_'

    rhs_list = []
    plain_vars_1st = {}
    if order == 1:
        plain_vars_1st = {name: sp.symbols(name) for name in dep_vars}
    for eq_idx, eq_str in enumerate(equations):
        safe_str = re.sub(r'\b' + re.escape(param_name) + r'\b', safe_param, eq_str)

        if order == 1:
            d1_sym = sp.symbols('_d1_')
            safe_no_d = re.sub(r'Derivative\(\w+\(\w+\),\s*\w+\)', '_d1_', safe_str)
            rhs_local = {**plain_vars_1st, '_d1_': d1_sym, indep_var: t, safe_param: lmbda, **_SYM_FUNCS}
            if '=' in safe_no_d:
                left_str, right_str = safe_no_d.split('=', 1)
                left = _safe_parse_expr(left_str.strip(), rhs_local)
                right = _safe_parse_expr(right_str.strip(), rhs_local)
                eq_sym = sp.Eq(left, right)
            else:
                eq_sym = _safe_parse_expr(safe_no_d, rhs_local)
            sols = sp.solve(eq_sym, d1_sym)
            if not sols:
                raise ValueError(
                    f"Не удалось выразить производную из уравнения №{eq_idx + 1}: "
                    f"'{eq_str}'. Уравнение должно содержать ровно одну первую "
                    f"производную Derivative(x(t), t) и быть разрешимо относительно неё."
                )
            rhs = sols[0]
            rhs_list.append(rhs)
        else:
            local_dict = {
                **{name: func_objects[name] for name in dep_vars},
                indep_var: t,
                safe_param: lmbda,
                **_SYM_FUNCS,
            }
            if '=' in safe_str:
                left_str, right_str = safe_str.split('=', 1)
                left = _safe_parse_expr(left_str.strip(), local_dict)
                right = _safe_parse_expr(right_str.strip(), local_dict)
                eq_sym = sp.Eq(left, right)
            else:
                eq_sym = _safe_parse_expr(safe_str, local_dict)
            funcs_eval = {name: sp.Function(name)(t) for name in dep_vars}
            ddot = sp.diff(funcs_eval[dep_vars[len(rhs_list)]], t, 2)
            sols = sp.solve(eq_sym, ddot)
            if not sols:
                raise ValueError(
                    f"Не удалось выразить вторую производную из уравнения №{eq_idx + 1}: "
                    f"'{eq_str}'. Проверьте структуру уравнения."
                )
            rhs = sols[0]
            rhs_list.append(rhs)

    if order == 1:
        ode_funcs = []
        for rhs in rhs_list:
            fn = sp.lambdify(
                [t] + list(plain_vars_1st.values()) + [lmbda],
                rhs,
                modules=['numpy']
            )
            ode_funcs.append(fn)

        bc_var_names = []
        for name in dep_vars:
            bc_var_names.extend([f'{name}0', f'{name}1'])

        return ode_funcs, t, None, lmbda, dep_vars, indep_var, bc_var_names, order
    else:
        funcs_eval = {name: sp.Function(name)(t) for name in dep_vars}
        rhs_expanded = []
        for rhs in rhs_list:
            r = rhs
            for name in dep_vars:
                r = r.subs({func_objects[name](t): funcs_eval[name]})
            rhs_expanded.append(r)

        ode_funcs = []
        for rhs in rhs_expanded:
            plain_vars = {name: sp.symbols(name) for name in dep_vars}
            plain_dots = {f'd{name}': sp.symbols(f'd{name}') for name in dep_vars}
            substituted = rhs
            for name in dep_vars:
                substituted = substituted.subs({sp.Derivative(func_objects[name](t), t): plain_dots[f'd{name}']})
                substituted = substituted.subs({func_objects[name](t): plain_vars[name]})
            fn = sp.lambdify(
                [t] + list(plain_vars.values()) + list(plain_dots.values()) + [lmbda],
                substituted,
                modules=['numpy']
            )
            ode_funcs.append(fn)

        bc_var_names = []
        for name in dep_vars:
            bc_var_names.extend([f'{name}0', f'd{name}0', f'{name}1', f'd{name}1'])

        return ode_funcs, t, funcs_eval, lmbda, dep_vars, indep_var, bc_var_names, order


def _normalize_bc(bc_str):
    s = bc_str.strip()
    m = re.match(r'x(\d+)\(([ab])\)\s*=\s*(.+)', s)
    if m:
        i = int(m.group(1))
        ep = m.group(2)
        rhs = m.group(3).strip()
        var = f'x{i}{0 if ep == "a" else 1}'
        rhs_normalized = re.sub(
            r'x(\d+)\(([ab])\)',
            lambda mm: f'x{mm.group(1)}{0 if mm.group(2) == "a" else 1}',
            rhs
        )
        return f'{var} - ({rhs_normalized})'
    return s


def parse_boundary_conditions(bc_str_list, param_name='lambda', bc_var_names=None):
    if bc_var_names is None:
        bc_var_names = ['y0', 'dy0', 'y1', 'dy1']

    bc_str_list = [s for s in bc_str_list if s is not None and str(s).strip()]
    if not bc_str_list:
        raise ValueError("No valid boundary conditions provided")

    bc_symbols = {name: sp.symbols(name) for name in bc_var_names}
    lmbda = sp.symbols(param_name)

    bc_funcs = []
    for bc_str in bc_str_list:
        bc_str = str(bc_str).strip()
        if not bc_str:
            raise ValueError(f"Empty boundary condition string")
        bc_str = _normalize_bc(bc_str)
        local_dict = {name: bc_symbols[name] for name in bc_var_names}
        bc_sym = _safe_parse_expr(bc_str, local_dict)
        args = [bc_symbols[name] for name in bc_var_names] + [lmbda]
        fn = sp.lambdify(args, bc_sym, modules=['numpy'])
        bc_funcs.append(fn)

    return bc_funcs
