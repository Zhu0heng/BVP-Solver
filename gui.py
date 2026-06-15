"""
Графический интерфейс на PyQt5 для метода продолжения по параметру.

Основные классы:
  SolverThread  — асинхронное решение краевых задач в отдельном потоке QThread.
  PlotWindow    — окно визуализации с выбором X/Y и многослойным построением.
  MainWindow    — главное окно: ввод уравнений, граничных условий и управление.
"""

import sys
import numpy as np
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QGroupBox, QCheckBox,
    QGridLayout, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QInputDialog, QComboBox, QScrollArea, QMenu, QListWidget, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'font.size': 14})

from dataset import Dataset
from solver import ContinuationSolver
from task_io import save_task, load_task, get_example_tasks

# Parallel to TR['sol_method_items'] — language-independent solver codes.
# Index 0 = auto-detect (legacy behaviour), 1..5 = explicit choices.
_SOLVER_CODES = ['auto', 'custom', 'kepler', 'limit_cycle', 'triple', 'lens']


def _solution_is_physical(y, ref_scale=1.0):
    """Отбраковка нефизических траекторий (расходящихся / переполнённых).

    Пристрелка мультимодальных нелинейных краевых задач иногда «сходится»
    к ветви, уходящей в бесконечность: корректор формально отрабатывает,
    но интеграл раздувается до 1e60 и далее. Такие кривые не имеют
    физического смысла и не должны попадать на график.

    Возвращает False, если массив содержит не-конечные значения (NaN/Inf)
    ИЛИ если максимальная по модулю компонента превышает порог,
    масштабируемый от характерной величины задачи ``ref_scale``.
    """
    arr = np.asarray(y, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return False
    bound = 1e3 * max(1.0, abs(float(ref_scale)))
    return float(np.max(np.abs(arr))) <= bound


def detect_problem_type(ds):
    """Classify into 'kepler','limit_cycle','triple','lens','custom'.

    Detection is conservative — the specialised solver will only be used
    when the equations AND boundary conditions match a known signature.
    Otherwise the generic continuation solver is used.
    """
    if ds is None:
        return 'custom'
    n_eq = len(ds.equations)
    eqs = ' '.join(ds.equations).lower()
    bcs = ' '.join(ds.boundary_conditions).lower()

    if n_eq == 4:
        # Kepler: inverse‑square‑law signature + vx/vy equation pattern
        if (re.search(r'x1\*\*2\s*\+\s*x2\*\*2|\(x1\*\*2\s*\+\s*x2\*\*2\)\s*\*\*\s*[13]', eqs) and
                re.search(r'derivative\(x1\(', eqs) and
                re.search(r'derivative\(x2\(', eqs)):
            return 'kepler'
        # Limit cycles: 4eq with sin term + x2(a)=x2(b)=0
        if ('sin' in eqs and
                re.search(r'x2\(a\)\s*=\s*0', bcs) and
                re.search(r'x2\(b\)\s*=\s*0', bcs)):
            return 'limit_cycle'

    if n_eq == 6 and 'sqrt' in eqs:
        return 'triple'

    if n_eq == 2 and re.search(r'\bu1\b|\bu2\b', eqs):
        return 'lens'

    return 'custom'


class SolverThread(QThread):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, dataset, initial_guess=None, smooth_param_list=None, explicit_type=None, multi_cycle=False):
        super().__init__()
        self.dataset = dataset
        self.initial_guess = initial_guess
        self.smooth_param_list = smooth_param_list
        self.explicit_type = explicit_type  # None / 'auto' → auto-detect; else use directly
        self.multi_cycle = multi_cycle

    def _solve_kepler(self):
        """Задача двух тел (Кеплер): краевая задача методом пристрелки.

        НЕСКОЛЬКО догадок → НЕСКОЛЬКО орбит (как в примере 26.2).
        Поле «Неизвестные» задаёт одну ИЛИ несколько пар начальной скорости
        (vx, vy); каждая пара — отдельная догадка пристрелки.  Задача Ламберта
        (две закреплённые точки за фиксированное время) имеет НЕСКОЛЬКО
        решений, поэтому разные пары сходятся к РАЗНЫМ корректным орбитам, и
        все они накладываются на один график.

        На график выводятся ТОЛЬКО корректные орбиты, проходящие через ОБЕ
        краевые точки, — никаких «промахивающихся» дуг.  Совпавшие орбиты
        (две догадки попали в один «бассейн» притяжения) не дублируются.
        """
        solver = ContinuationSolver(self.dataset)

        # ── Начальное положение из краевых условий ──
        bx10, bx20 = None, None
        for bc in self.dataset.boundary_conditions:
            m = re.match(r'\s*x(\d+)\(a\)\s*=\s*([+-]?[\d.eE+\-]+)', bc)
            if m:
                idx, val = int(m.group(1)), float(m.group(2))
                if idx == 1:
                    bx10 = val
                elif idx == 2:
                    bx20 = val
        if bx10 is None:
            bx10 = 2.0
        if bx20 is None:
            bx20 = 0.0

        # ── Разбор догадок: одна ИЛИ несколько пар (vx, vy) ──
        # initial_guess приходит группами по 4: [x1a, x2a, vx, vy, ...].
        guesses = []  # список (x10, x20, vx, vy)
        if self.initial_guess is not None:
            ug = [float(v) for v in self.initial_guess]
            for k in range(0, len(ug) - 3, 4):
                guesses.append((ug[k], ug[k + 1], ug[k + 2], ug[k + 3]))
            # Хвост из 2 чисел (только vx, vy) — положение из краевых условий.
            if not guesses and len(ug) >= 2:
                guesses.append((bx10, bx20, ug[0], ug[1]))
        if not guesses:
            guesses.append((bx10, bx20, 0.5, 0.8))

        ref = max(1.0, float(np.hypot(bx10, bx20)))

        # ── Пристрелка из каждой догадки → отдельная орбита ──
        # Задача Ламберта МНОГОЗНАЧНА: из одного набора seed пристрелка может
        # сойтись к РАЗНЫМ корректным орбитам.  Поэтому для каждой догадки
        # собираем ВСЕ найденные корректные решения и выбираем то, чьё v0
        # БЛИЖЕ ВСЕГО к самой догадке (vx, vy).  Именно к ближайшему решению
        # ведёт продолжение по параметру из начальной точки; «перескок»
        # обычного метода Ньютона/ЛМ к дальнему корню — численный артефакт.
        # Для учебной догадки (0.5, 0.5) это даёт ans1 ≈ (0, 0.5).
        layers = []
        seen = []  # подписи v0 для дедупликации совпавших орбит
        for (gx10, gx20, vx, vy) in guesses:
            v_scale = max(0.3, ref / 4.0)
            # Широкий набор seed, покрывающий оба «бассейна» решений Ламберта.
            seeds = [(vx, vy), (-vx, -vy), (vy, -vx), (-vy, vx),
                     (0.0, vy), (0.0, -vy),
                     (0.0, v_scale), (0.0, -v_scale),
                     (v_scale, -v_scale), (-v_scale, v_scale)]
            cands = {}  # подпись v0 → (xs, ys) корректного решения
            for svx, svy in seeds:
                try:
                    cand_x, cand_y = solver.solve(
                        np.array([gx10, gx20, svx, svy]))
                except Exception:
                    continue
                if not _solution_is_physical(cand_y, ref):
                    continue
                sig = (round(float(cand_y[2, 0]), 3),
                       round(float(cand_y[3, 0]), 3))
                cands.setdefault(sig, (cand_x, cand_y))
            if not cands:
                continue
            # Выбираем решение, ближайшее к самой догадке (vx, vy).
            best = None  # (расстояние², подпись, xs, ys)
            for sig, (cand_x, cand_y) in cands.items():
                d2 = ((float(cand_y[2, 0]) - vx) ** 2
                      + (float(cand_y[3, 0]) - vy) ** 2)
                if best is None or d2 < best[0]:
                    best = (d2, sig, cand_x, cand_y)
            _, sig, xs, ys = best
            if sig in seen:
                continue
            seen.append(sig)
            try:
                _, full_b = solver.full_orbit(ys[:, 0], n_points=500)
            except Exception:
                full_b = None
            layers.append((xs, ys, full_b,
                           f'Орбита  v0=({ys[2, 0]:.3f}, {ys[3, 0]:.3f})'))

        if not layers:
            raise RuntimeError("Kepler BVP did not converge — "
                               "check BCs and initial guess")
        self.finished.emit(layers, None)

    def _solve_limit_cycles(self):
        """Solve for ONE limit cycle using the user's guess values directly.

        The user provides initial guesses for the unknown components:
          • x1(0) — starting amplitude
          • T    — period estimate (default 2π)
          • x4(0) — phase anchor (default = x1(0))

        These are used as-is (no scanning).  Each guess selects which
        cycle to converge to.  The plot window shows exactly ONE layer
        per solve call.

        If the user-provided guess fails, we fall back to a broad scan
        (80 points from 0.1 to 30) to find any cycles.
        """
        solver = ContinuationSolver(self.dataset)

        # ── Parse user guess into ONE or MULTIPLE guess vectors ──
        ug_raw = [0.5, 0.0, 2 * np.pi, 0.5]
        group_size = 4
        if self.initial_guess is not None:
            ui = list(self.initial_guess)
            if len(ui) >= group_size:
                ug_raw = [float(v) for v in ui]
        # Default: 1 group (4 components).  Multi-cycle checkbox → all groups.
        n_groups = max(1, len(ug_raw) // group_size) if self.multi_cycle else 1

        all_solutions = []

        for gi in range(n_groups):
            base = gi * group_size
            ug = [
                ug_raw[base + 0] if base + 0 < len(ug_raw) else 0.5,
                ug_raw[base + 1] if base + 1 < len(ug_raw) else 0.0,
                ug_raw[base + 2] if base + 2 < len(ug_raw) else 2 * np.pi,
                ug_raw[base + 3] if base + 3 < len(ug_raw) else 0.5,
            ]
            ug[1] = 0.0              # x2(0)=0 always (from BC)
            if ug[2] < 1.0:
                ug[2] = 2 * np.pi
            if abs(ug[3]) < 1e-6:
                ug[3] = ug[0]
            g_direct = np.array(ug, dtype=float)

            try:
                x_eval, y_eval = solver.solve(g_direct)
                if _solution_is_physical(y_eval, max(1.0, abs(g_direct[0]))):
                    x1_val = y_eval[0, 0]
                    T_val = y_eval[2, 0]
                    if abs(x1_val) > 1e-3 and abs(T_val) > 1e-3:
                        all_solutions.append(
                            (x_eval, y_eval, None,
                             f'x1={x1_val:.2f} T={T_val:.2f}  '
                             f'← guess ({g_direct[0]:.3g}, {g_direct[2]:.3g})'))
            except Exception:
                pass

        if all_solutions:
            self.finished.emit(all_solutions, None)
            return

        # ── Phase 2: fallback — broad scan ──
        found_keys = []
        solutions = []
        for x1_guess in np.linspace(0.1, 30.0, 80):
            g = np.array([x1_guess, 0.0, ug[2], x1_guess], dtype=float)
            try:
                x_eval, y_eval = solver.solve(g)
            except Exception:
                continue
            if not _solution_is_physical(y_eval, max(1.0, abs(x1_guess))):
                continue
            x1_val = y_eval[0, 0]
            T_val = y_eval[2, 0]
            if abs(x1_val) < 1e-3 or abs(T_val) < 1e-3:
                continue
            key = (round(abs(x1_val), 1), round(abs(T_val), 2))
            if any(abs(key[0] - k[0]) < 0.1 and abs(key[1] - k[1]) < 0.2
                   for k in found_keys):
                continue
            found_keys.append(key)
            solutions.append((x_eval, y_eval, None,
                             f'x1={x1_val:.2f} T={T_val:.2f}'))

        if not solutions:
            raise RuntimeError("No limit cycle found — try different initial guess")
        self.finished.emit(solutions, None)

    def _solve_triple(self):
        """Solve the triple-integrator BVP by SHOOTING the 3 unknown costates.

        Strategy (универсальный, реагирует на любое изменение уравнений):
          • Initial values x1(a), x2(a), x3(a) — берутся из BC левого края.
          • Unknown initial costates x4(0), x5(0), x6(0) — ищутся через LM.
          • Forward-integrate the USER's equations from self.dataset.
          • Constraint: x1(b)=…, x2(b)=…, x3(b)=… из BC правого края.
          • Control u(t) = RHS уравнения dx3/dt = u, тоже из пользовательских уравнений.
        Так любое изменение коэффициентов / структуры / BC приводит к новой траектории.
        """
        from scipy.integrate import solve_ivp as _solve_ivp
        from scipy.optimize import root as _root

        solver = ContinuationSolver(self.dataset)
        n = solver.n_dep
        T = self.dataset.x_end
        t0 = self.dataset.x_start

        # ── Извлекаем BC левого/правого края ──
        a_vals = {1: 1.0, 2: 0.0, 3: 0.0}      # defaults: original problem
        b_vals = {1: 0.0, 2: 0.0, 3: 0.0}
        for bc in self.dataset.boundary_conditions:
            ma = re.match(r'\s*x(\d+)\(a\)\s*=\s*([+-]?[\d.eE+\-]+)\s*$', bc)
            mb = re.match(r'\s*x(\d+)\(b\)\s*=\s*([+-]?[\d.eE+\-]+)\s*$', bc)
            if ma:
                a_vals[int(ma.group(1))] = float(ma.group(2))
            elif mb:
                b_vals[int(mb.group(1))] = float(mb.group(2))

        def user_ode(t, y):
            # Use current parameter value from dataset, not hardcoded 0.0
            params = self.dataset.parameters or {}
            param_val = float(params.get(self.dataset.continuation_param,
                                         getattr(self.dataset, 'continuation_start', 0.0)))
            return solver._ode_system(t, y, param_val)

        # ── Shooting residual (3-мерный) ──
        # Uses tight ODE tolerances so that finite-difference Jacobians
        # in LM are noise-free.
        def shoot(unknown):
            y0 = np.array([
                a_vals.get(1, 1.0),
                a_vals.get(2, 0.0),
                a_vals.get(3, 0.0),
                unknown[0], unknown[1], unknown[2],
            ])
            try:
                sol = _solve_ivp(user_ode, [t0, T], y0,
                                 dense_output=True,
                                 rtol=1e-12, atol=1e-14, max_step=0.005,
                                 method='DOP853')
                ye = sol.sol(T)
            except Exception:
                return np.full(3, 1e10)
            return np.array([
                ye[0] - b_vals.get(1, 0.0),
                ye[1] - b_vals.get(2, 0.0),
                ye[2] - b_vals.get(3, 0.0),
            ])

        # ── Build list of starting points for shooting ──
        # Strategy (no hardcoded original-problem-specific values):
        #   1. User's guess (highest priority, taken AS-IS from initial_guess[3:6])
        #   2. Backward integration of the USER's equations from a terminal
        #      state derived from b-side BCs.  Unknown terminal components
        #      are scanned over a sign-grid.  This works for ANY modified
        #      ODE, not just the original example.
        #   3. Coarse 3-D Sobol-like grid as final fallback.
        starts = []

        if self.initial_guess is not None and len(self.initial_guess) >= 6:
            starts.append(np.asarray(self.initial_guess[3:6], dtype=float))

        # Terminal state from b-side BCs (constrained components) + sign-grid
        # scan over the unconstrained ones.  We integrate the USER's ODE
        # backwards from each candidate.
        y_terminal_known = np.zeros(6)
        for i in (1, 2, 3):
            y_terminal_known[i - 1] = b_vals.get(i, 0.0)

        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    y_T = y_terminal_known.copy()
                    y_T[3], y_T[4], y_T[5] = sx * 3.0, sy * 5.0, sz * 3.0
                    try:
                        sol_bwd = _solve_ivp(
                            lambda tau, y: [-v for v in user_ode(0.0, y)],
                            [0.0, T], y_T,
                            dense_output=True, rtol=1e-9, atol=1e-12,
                        )
                        starts.append(sol_bwd.sol(T)[3:6])
                    except Exception:
                        continue

        # Coarse random fallback grid (10 quasi-random points)
        rng = np.random.default_rng(42)
        for _ in range(10):
            starts.append(rng.uniform(-10.0, 10.0, size=3))

        # ── Try each start, keep the one with the lowest terminal residual ──
        # Each start gets up to 2 LM rounds (warm-restart from best so far
        # if first LM didn't fully converge).
        best_x, best_r = None, np.inf
        for st in starts:
            cur = np.asarray(st, dtype=float)
            for _round in range(2):
                try:
                    res = _root(shoot, cur, method='lm',
                                options={'ftol': 1e-14, 'xtol': 1e-14,
                                         'maxiter': 500,
                                         'factor': 0.1})
                except Exception:
                    break
                r = float(np.linalg.norm(shoot(res.x)))
                cur = res.x
                if r < best_r:
                    best_x, best_r = res.x.copy(), r
                if r < 1e-9:
                    break
            if best_r < 1e-9:
                break

        if best_x is None:
            raise RuntimeError("Triple integrator shooting failed for all starts")

        y0_final = np.array([
            a_vals.get(1, 1.0), a_vals.get(2, 0.0), a_vals.get(3, 0.0),
            best_x[0], best_x[1], best_x[2],
        ])
        x_eval = np.linspace(t0, T, self.dataset.n_points)
        _tol = getattr(self.dataset, 'tol', 1e-9)
        _method = getattr(self.dataset, 'method', 'RK45')
        sol_final = _solve_ivp(user_ode, [t0, T], y0_final,
                               dense_output=True, method=_method,
                               rtol=_tol, atol=_tol * 1e-3, max_step=0.01)
        y_eval = sol_final.sol(x_eval)

        # Control u(t) = RHS уравнения dx3/dt (user-defined)
        try:
            u_t = np.array([solver.ode_funcs[2](x_eval[j], *y_eval[:, j], 0.0)
                            for j in range(len(x_eval))])
            y_out = np.vstack([y_eval, u_t.reshape(1, -1)])
        except Exception:
            y_out = y_eval

        self.finished.emit(x_eval, y_out)

    def _solve_lens_control(self):
        """Solve the time-optimal lens-shaped control problem.

        Symbolic-aware version: вся правая часть уравнений (включая
        коэффициенты u1, u2 и нелинейные члены по x1, x2) извлекается
        символьно через SymPy и используется напрямую. Сопряжённая
        система строится автоматически через частные производные.
        Любое изменение уравнений в GUI отражается на графиках.
        """
        from scipy.integrate import solve_ivp
        from scipy.optimize import root
        import sympy as sp

        # ── 1. BC левого и правого края ──
        # a1,a2 — начальное состояние; b1,b2 — целевое состояние на правом крае.
        # Раньше правый край был жёстко зашит в начало координат (0,0), из-за
        # чего изменение x1(b)/x2(b) в GUI не влияло на решение.
        a1, a2 = 4.0, 1.0
        b1, b2 = 0.0, 0.0
        for bc in self.dataset.boundary_conditions:
            m = re.match(r'\s*x1\(a\)\s*=\s*([+-]?[\d.eE+\-]+)', bc)
            if m:
                a1 = float(m.group(1))
            m = re.match(r'\s*x2\(a\)\s*=\s*([+-]?[\d.eE+\-]+)', bc)
            if m:
                a2 = float(m.group(1))
            m = re.match(r'\s*x1\(b\)\s*=\s*([+-]?[\d.eE+\-]+)', bc)
            if m:
                b1 = float(m.group(1))
            m = re.match(r'\s*x2\(b\)\s*=\s*([+-]?[\d.eE+\-]+)', bc)
            if m:
                b2 = float(m.group(1))

        # ── 2. Символьный разбор правых частей f1, f2 ──
        x1s, x2s, u1s, u2s = sp.symbols('x1 x2 u1 u2', real=True)
        psi1s, psi2s = sp.symbols('psi1 psi2', real=True)
        d_sym = sp.Symbol('__d__')
        loc = {
            'x1': x1s, 'x2': x2s, 'u1': u1s, 'u2': u2s,
            '__d__': d_sym, 't': sp.Symbol('t'),
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log,
            'pi': sp.pi, 'E': sp.E,
        }

        # Defaults на случай ошибки парсинга
        f1_expr = x2s + u1s
        f2_expr = -sp.Rational(3, 2) * x1s - sp.Rational(1, 4) * x2s + u2s

        try:
            for eq_str in self.dataset.equations:
                if not re.search(r'Derivative\(x[12]\(', eq_str):
                    continue
                safe = re.sub(r'Derivative\(\w+\(\w+\),\s*\w+\)', '__d__', eq_str)
                if '=' in safe:
                    lhs_s, rhs_s = safe.split('=', 1)
                else:
                    lhs_s, rhs_s = safe, '0'
                lhs_e = sp.parse_expr(lhs_s.strip(), local_dict=loc)
                rhs_e = sp.parse_expr(rhs_s.strip(), local_dict=loc)
                sols = sp.solve(sp.Eq(lhs_e, rhs_e), d_sym)
                if not sols:
                    continue
                f_rhs = sp.expand(sols[0])
                if 'Derivative(x1(' in eq_str:
                    f1_expr = f_rhs
                elif 'Derivative(x2(' in eq_str:
                    f2_expr = f_rhs
        except Exception:
            pass  # fall back to defaults

        # ── 3. Гамильтониан и сопряжённые уравнения ──
        # H = 1 + ψ1·f1(x1,x2,u1,u2) + ψ2·f2(x1,x2,u1,u2)
        # dψ1/dt = -∂H/∂x1 = -(ψ1·∂f1/∂x1 + ψ2·∂f2/∂x1)
        # dψ2/dt = -∂H/∂x2 = -(ψ1·∂f1/∂x2 + ψ2·∂f2/∂x2)
        df1_x1 = sp.diff(f1_expr, x1s)
        df1_x2 = sp.diff(f1_expr, x2s)
        df2_x1 = sp.diff(f2_expr, x1s)
        df2_x2 = sp.diff(f2_expr, x2s)

        # Lambdify для быстрых численных вычислений
        f1_n = sp.lambdify((x1s, x2s, u1s, u2s), f1_expr, modules='numpy')
        f2_n = sp.lambdify((x1s, x2s, u1s, u2s), f2_expr, modules='numpy')
        df1_x1_n = sp.lambdify((x1s, x2s, u1s, u2s), df1_x1, modules='numpy')
        df1_x2_n = sp.lambdify((x1s, x2s, u1s, u2s), df1_x2, modules='numpy')
        df2_x1_n = sp.lambdify((x1s, x2s, u1s, u2s), df2_x1, modules='numpy')
        df2_x2_n = sp.lambdify((x1s, x2s, u1s, u2s), df2_x2, modules='numpy')

        # ── 4. Сглаженное лунко-образное управление (зависит от ψ и μ) ──
        # Это математически зафиксировано теоремой Понтрягина для лунки.
        def ctrl_smoothed(psi, mu):
            psi1, psi2 = psi[0], psi[1]
            n2  = psi1**2 + psi2**2
            eps = 1e-14
            A   = max(np.sqrt(mu * n2 + (psi1 + psi2)**2), eps)
            B   = max(np.sqrt(mu * n2 + (psi1 - psi2)**2), eps)
            q1  = 0.5 * (A + B)
            dq1_dpsi1 = 0.5 * ((mu*psi1 + psi1 + psi2) / A +
                                (mu*psi1 + psi1 - psi2) / B)
            dq1_dpsi2 = 0.5 * ((mu*psi2 + psi1 + psi2) / A +
                                (mu*psi2 - psi1 + psi2) / B)
            R  = max(np.sqrt(q1**2 + psi2**2), eps)
            s2 = np.sqrt(2.0)
            u1 = dq1_dpsi1 * (s2 * q1 / R - 1.0)
            u2 = s2 * (q1 * dq1_dpsi2 + psi2) / R - dq1_dpsi2
            return np.array([u1, u2])

        # ── 5. Список μ ──
        mu_values = self.smooth_param_list
        if mu_values is None:
            params = self.dataset.parameters
            if params and 'mu_values' in params:
                mu_values = params['mu_values']
            else:
                mu_values = [1.0, 1e-1, 1e-6]

        layers = []
        # Warm-start: derive from user's initial_guess if available, otherwise
        # use a scale derived from the BC magnitudes — no hardcoded values
        # tuned to the original example.
        bc_scale = max(1.0, np.sqrt(a1 ** 2 + a2 ** 2))
        psi10, psi20, T_opt = -0.5, -0.1, bc_scale  # generic adaptive starts
        if self.initial_guess is not None:
            ug = list(self.initial_guess)
            if len(ug) >= 3:
                try:
                    psi10 = float(ug[-3])
                    psi20 = float(ug[-2])
                    T_opt = max(0.1, float(ug[-1]))
                except Exception:
                    pass

        def make_ode(mu_val):
            def ode_dimless(tau, y):
                x1, x2, psi1, psi2, T = y
                u = ctrl_smoothed(np.array([psi1, psi2]), mu_val)
                u1v, u2v = float(u[0]), float(u[1])
                dx1 = T * float(f1_n(x1, x2, u1v, u2v))
                dx2 = T * float(f2_n(x1, x2, u1v, u2v))
                a11 = float(df1_x1_n(x1, x2, u1v, u2v))
                a12 = float(df1_x2_n(x1, x2, u1v, u2v))
                a21 = float(df2_x1_n(x1, x2, u1v, u2v))
                a22 = float(df2_x2_n(x1, x2, u1v, u2v))
                dp1 = -T * (psi1 * a11 + psi2 * a21)
                dp2 = -T * (psi1 * a12 + psi2 * a22)
                return [dx1, dx2, dp1, dp2, 0.0]
            return ode_dimless

        def make_shoot(ode_fn):
            def shoot_3(v):
                p10, p20, Tv = v
                if Tv <= 0:                      # T must be positive
                    return np.full(3, 1e6)
                try:
                    sol = solve_ivp(ode_fn, [0., 1.], [a1, a2, p10, p20, Tv],
                                    rtol=1e-9, atol=1e-12, max_step=0.005)
                    ye = sol.y[:, -1]
                except Exception:
                    return np.full(3, 1e10)
                # x1(T)=b1, x2(T)=b2 (целевое состояние из BC правого края),
                # |ψ(T)|²=1 — условие трансверсальности (свободное время T).
                return np.array([ye[0] - b1, ye[1] - b2,
                                 ye[2]**2 + ye[3]**2 - 1.0])
            return shoot_3

        def try_root(ode_fn, start):
            shoot_3 = make_shoot(ode_fn)
            try:
                res = root(shoot_3, start, method='lm',
                           options={'ftol': 1e-10, 'xtol': 1e-10, 'maxiter': 300})
                if not res.success:
                    return None, np.inf
                if res.x[2] <= 0:                # reject non-physical T<0
                    return None, np.inf
                rnorm = float(np.linalg.norm(shoot_3(res.x)))
                if rnorm > 1.0:
                    return None, rnorm
                return res.x, rnorm
            except Exception:
                return None, np.inf

        # μ-continuation: refine path between μ values, plus multistart fallbacks
        def refine_mu(mu_list):
            full = []
            for i in range(len(mu_list) - 1):
                full.append(mu_list[i])
                a_, b_ = mu_list[i], mu_list[i + 1]
                if a_ > 0 and b_ > 0 and abs(np.log10(a_) - np.log10(b_)) > 1.5:
                    n_sub = max(2, int(abs(np.log10(a_) - np.log10(b_))))
                    for k in range(1, n_sub):
                        # geometric interpolation
                        full.append(a_ * (b_ / a_) ** (k / n_sub))
            full.append(mu_list[-1])
            return full

        all_mus = refine_mu(list(mu_values))
        # Map refined μ values to their corresponding solutions
        solutions_cache = {}

        for mu in all_mus:
            ode_main = make_ode(mu)
            sol_x, rnorm = try_root(ode_main, [psi10, psi20, T_opt])

            # Multistart fallback if main attempt diverged.
            # Trials are generated adaptively from a problem-scale base, so
            # changing BCs or equations produces a different (scaled) search
            # space — no hardcoded numerical constants.
            if sol_x is None:
                best_x, best_r = None, np.inf
                base_T = bc_scale * 2.0
                trials = []
                for sign_psi in ((-1.0, -1.0), (-1.0, 0.5), (-0.5, -1.0),
                                 (0.5, -1.0), (-1.0, 1.0)):
                    for k in (0.5, 1.0, 1.5):
                        trials.append([sign_psi[0] * 0.5, sign_psi[1] * 0.2,
                                       base_T * k])
                for tr in trials:
                    cand, r = try_root(ode_main, tr)
                    if cand is not None and r < best_r:
                        best_x, best_r = cand, r
                sol_x, rnorm = best_x, best_r

            if sol_x is not None:
                psi10, psi20, T_opt = sol_x
                solutions_cache[mu] = (psi10, psi20, T_opt)

        # Now emit only the originally requested mu_values
        for mu in mu_values:
            if mu in solutions_cache:
                psi10, psi20, T_opt = solutions_cache[mu]
            ode_main = make_ode(mu)
            _tol = getattr(self.dataset, 'tol', 1e-9)
            _method = getattr(self.dataset, 'method', 'RK45')
            sol = solve_ivp(ode_main, [0., 1.], [a1, a2, psi10, psi20, T_opt],
                            dense_output=True, method=_method,
                            rtol=_tol, atol=_tol * 1e-3, max_step=0.002)
            t_dimless = np.linspace(0., 1., 300)
            y_dimless = sol.sol(t_dimless)

            u_vals = np.zeros((2, len(t_dimless)))
            for i in range(len(t_dimless)):
                u_vals[:, i] = ctrl_smoothed(y_dimless[2:4, i], mu)

            y_out = np.vstack([
                y_dimless[0], y_dimless[1],
                y_dimless[2], y_dimless[3],
                np.full(len(t_dimless), T_opt),
                u_vals[0], u_vals[1],
            ])

            # Convert dimensionless τ ∈ [0,1] to real time t = τ·T_opt
            t_real = t_dimless * T_opt
            layers.append((t_real, y_out, f'T={T_opt:.4f}'))

        if len(layers) == 1:
            self.finished.emit(layers[0][0], layers[0][1])
        else:
            self.finished.emit(layers, None)

    def run(self):
        try:
            if self.explicit_type and self.explicit_type != 'auto':
                problem_type = self.explicit_type
            else:
                problem_type = detect_problem_type(self.dataset)
            if problem_type == 'kepler':
                self._solve_kepler()
            elif problem_type == 'limit_cycle':
                self._solve_limit_cycles()
            elif problem_type == 'triple':
                self._solve_triple()
            elif problem_type == 'lens':
                self._solve_lens_control()
            else:
                solver = ContinuationSolver(self.dataset)
                if self.smooth_param_list and len(self.smooth_param_list) > 1:
                    param_name = self.dataset.continuation_param
                    all_vals = sorted(float(v) for v in self.smooth_param_list)
                    dense = []
                    for i in range(len(all_vals) - 1):
                        dense.append(all_vals[i])
                        gap = all_vals[i + 1] - all_vals[i]
                        n_sub = max(3, int(abs(gap) / 0.2))
                        for k in range(1, n_sub):
                            dense.append(all_vals[i] + gap * k / n_sub)
                    dense.append(all_vals[-1])
                    layers = []
                    for pv in all_vals:
                        chain = [v for v in dense if v <= pv + 1e-10]
                        xp, yp = solver.solve(self.initial_guess,
                                              lmbda_values=chain)
                        layers.append((xp, yp, None, f'{param_name}={pv:.4g}'))
                    if layers:
                        self.finished.emit(layers, None)
                    else:
                        raise RuntimeError("No parameter values converged")
                else:
                    x_eval, y_eval = solver.solve(self.initial_guess)
                    self.finished.emit(x_eval, y_eval)
        except Exception as e:
            self.error.emit(str(e))


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=7.5, dpi=120):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#16161A')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#16161A')
        self.axes.tick_params(colors='#9E9EB5', which='both')
        for spine in self.axes.spines.values():
            spine.set_color('#35354A')
        super().__init__(self.fig)
        self.setParent(parent)


class PlotWindow(QMainWindow):
    def __init__(self, t_data, y_data, varnames, title, parent=None, problem_type='custom',
                 color_offset=0):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(150, 150, 1100, 800)
        self.dataset_title = title
        self.varnames = varnames
        self.problem_type = problem_type
        # Сдвиг индекса палитры: при «разделении» окон каждый слой сохраняет
        # тот же цвет, что и в совмещённом графике (см. _toggle_split_view).
        self._color_offset = color_offset

        if isinstance(t_data, list):
            self._has_layers = True
            self._list_data = t_data
            self._list_vars = varnames
            first = t_data[0]
            if len(first) == 4:
                t0, y0 = first[0], first[1]
            else:
                t0, y0 = first[0], first[1]
            self._build_ui(t0, y0, varnames)
        elif isinstance(y_data, list):
            self._has_layers = True
            self._list_data = y_data
            self._list_vars = varnames
            first = y_data[0]
            if len(first) == 4:
                t0, y0 = first[0], first[1]
            else:
                t0, y0 = first[0], first[1]
            self._build_ui(t0, y0, varnames)
        else:
            self._has_layers = False
            self._list_data = []
            self._list_vars = []
            self._build_ui(t_data, y_data, varnames)

        self._redraw()

    def _build_ui(self, t_data, y_data, varnames):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self.canvas = PlotCanvas(self, width=10, height=7.5)
        layout.addWidget(self.canvas)

        def _is_internal(vn):
            if vn == 'T':
                return True
            return False

        bar = QHBoxLayout()
        self.btn_x = QPushButton()
        self.btn_y = QPushButton()
        bar.addWidget(QLabel("X:"))
        bar.addWidget(self.btn_x)
        bar.addSpacing(8)
        bar.addWidget(QLabel("Y:"))
        bar.addWidget(self.btn_y)
        bar.addStretch()

        # Split/Combine toggle — only meaningful when there ARE multiple layers
        self.btn_split = QPushButton("Split solutions")
        self.btn_split.setToolTip(
            "Open each solution in its own window; "
            "click again to recombine into one view."
        )
        self.btn_split.setObjectName("plotToolBtn")
        self.btn_split.clicked.connect(self._toggle_split_view)
        bar.addWidget(self.btn_split)
        # Hide the button if this window only has a single solution
        if not getattr(self, '_has_layers', False) or \
           not isinstance(getattr(self, '_list_data', None), list) or \
           len(getattr(self, '_list_data', [])) < 2:
            self.btn_split.setVisible(False)

        layout.insertLayout(0, bar)

        self.t_data = t_data
        self.y_data = y_data

        self._x_name_to_row = {}
        self._x_items = ["t"]
        if isinstance(y_data, np.ndarray) and y_data.ndim == 2:
            nv = y_data.shape[0]
            if varnames and len(varnames) == nv:
                for vi, vn in enumerate(varnames):
                    if not _is_internal(vn):
                        self._x_name_to_row[len(self._x_items)] = vi
                        self._x_items.append(vn)
            else:
                for vi in range(nv):
                    self._x_name_to_row[vi + 1] = vi
                    self._x_items.append(f"x{vi+1}")

        self._y_items = []
        self._y_row = {-1: -1}
        if isinstance(y_data, np.ndarray) and y_data.ndim == 2:
            nv = y_data.shape[0]
            if varnames and len(varnames) == nv:
                for vi, vn in enumerate(varnames):
                    if not _is_internal(vn):
                        self._y_row[len(self._y_items)] = vi
                        self._y_items.append(vn)
            else:
                for vi in range(nv):
                    self._y_row[len(self._y_items)] = vi
                    self._y_items.append(f"x{vi+1}")

        is_phase = self.problem_type in ('kepler', 'limit_cycle')
        if is_phase:
            self._x_idx = 1
            self._y_indices = [1]
        else:
            self._x_idx = 0
            self._y_indices = [0]

        self.btn_x.setText(self._x_items[self._x_idx])
        self.btn_y.setText(", ".join(self._y_items[i] for i in self._y_indices))

        self.btn_x.clicked.connect(self._pick_x)
        self.btn_y.clicked.connect(self._pick_y)

    def _pick_x(self):
        from PyQt5.QtWidgets import QDialog, QListWidget, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("X")
        dlg.resize(250, 300)
        layout = QVBoxLayout(dlg)
        lst = QListWidget()
        lst.addItems(self._x_items)
        lst.setCurrentRow(self._x_idx)
        layout.addWidget(lst)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec_() and lst.currentRow() >= 0:
            self._x_idx = lst.currentRow()
            self.btn_x.setText(self._x_items[self._x_idx])
            self._redraw()

    def _pick_y(self):
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Y")
        dlg.resize(250, 330)
        layout = QVBoxLayout(dlg)
        checks = []
        for idx, name in enumerate(self._y_items):
            cb = QCheckBox(name)
            cb.setChecked(idx in self._y_indices)
            checks.append(cb)
            layout.addWidget(cb)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec_():
            selected = [i for i, cb in enumerate(checks) if cb.isChecked()]
            if selected:
                self._y_indices = selected
                self.btn_y.setText(", ".join(self._y_items[i] for i in self._y_indices))
                self._redraw()

    def _toggle_split_view(self):
        """Open each layer in its own PlotWindow; close this combined window."""
        if not self._has_layers or len(self._list_data) < 2:
            return
        parent = self.parent()
        # Find MainWindow up the parent chain
        from PyQt5.QtWidgets import QMainWindow
        mw = parent
        while mw is not None and not isinstance(mw, MainWindow):
            mw = mw.parent() if hasattr(mw, 'parent') else None
        if mw is None:
            return
        # Generate per-layer windows
        vn = self.varnames
        layers = self._list_data
        new_windows = []
        for i, layer in enumerate(layers):
            if not isinstance(layer, (list, tuple)):
                continue
            if len(layer) >= 4:
                t_i, y_i, full_i, label_i = layer[0], layer[1], layer[2], layer[3]
            elif len(layer) >= 3:
                t_i, y_i, label_i = layer[0], layer[1], layer[2]
                full_i = None
            else:
                continue
            if t_i is None or y_i is None:
                continue
            title_i = f"{self.dataset_title} — {label_i}" if label_i else \
                      f"{self.dataset_title} #{i+1}"
            # Всегда упаковываем слой как одноэлементный список, чтобы метка
            # (label_i) сохранялась в раздельном окне.  Раньше слои без полной
            # орбиты (full_i is None — предельные циклы) передавались сырыми
            # массивами и теряли подпись «кто есть кто».
            packed = [(t_i, y_i, full_i, label_i or f'#{i+1}')]
            # color_offset=i → слой сохраняет свой цвет из совмещённого графика.
            w = PlotWindow(packed, None, vn, title_i, mw, self.problem_type,
                           color_offset=i)
            w.setGeometry(150 + 40 * i, 150 + 40 * i, 1000, 700)
            w.show()
            new_windows.append(w)
        # Register with the main window so they get tracked, then close self
        if hasattr(mw, '_plot_windows'):
            mw._plot_windows.extend(new_windows)
        self.close()

    def _redraw(self):
        ax = self.canvas.axes
        ax.clear()

        xi = self._x_idx

        if self._has_layers:
            layers = self._list_data
        else:
            layers = [(self.t_data, self.y_data, None, "")]

        x_row = self._x_name_to_row.get(xi, xi - 1) if xi > 0 else -1

        is_phase = self.problem_type in ('kepler', 'limit_cycle')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        off = getattr(self, '_color_offset', 0)

        all_x, all_y = [], []
        x_label = "t" if xi == 0 else self._x_items[xi]
        y_label = ""

        for yi_pos, yi in enumerate(self._y_indices):
            y_row = self._y_row.get(yi, -1)
            y_name = self._y_items[yi]
            if len(self._y_indices) == 1:
                y_label = y_name
            for li, entry in enumerate(layers):
                t_dat = entry[0]
                y_dat = entry[1]
                has_full = len(entry) >= 4 and entry[2] is not None
                label = entry[-1] if isinstance(entry[-1], str) else f"#{li+1}"

                # ── Time-axis rescaling for limit cycles ──
                # The solver normalises time to [0,1]; the true time is t·T.
                # If we are showing a time-series view (xi==0) for a
                # limit-cycle problem, rescale each layer individually.
                t_scale = 1.0
                if self.problem_type == 'limit_cycle' and xi == 0:
                    # T is stored in y_dat[2] (the period variable)
                    t_scale = float(y_dat[2, 0])
                t_plot = t_dat * t_scale

                if len(self._y_indices) == 1:
                    c = colors[(li + off) % len(colors)]
                else:
                    c = colors[(yi_pos * len(layers) + li + off) % len(colors)]

                if has_full and self._has_layers and len(self._y_indices) == 1:
                    full_y = entry[2]
                    n_full = full_y.shape[1]
                    if xi == 0:
                        full_x = np.linspace(t_plot[0], t_plot[-1], n_full)
                    else:
                        full_x = full_y[x_row]
                    if y_row >= 0:
                        full_yv = full_y[y_row]
                    else:
                        full_yv = np.linspace(t_plot[0], t_plot[-1], n_full)
                    all_x.extend(full_x)
                    all_y.extend(full_yv)
                    ax.plot(full_x, full_yv, color=c,
                            lw=1.2, ls='--', alpha=0.55)

                xx = t_plot if xi == 0 else y_dat[x_row]
                yy = t_plot if y_row < 0 else y_dat[y_row]

                all_x.extend(xx); all_y.extend(yy)

                if is_phase:
                    ax.plot(xx, yy, color=c, lw=2, label=label)
                    ax.plot(xx[0], yy[0], 'o', color=c, ms=6)
                elif len(self._y_indices) > 1:
                    ax.plot(xx, yy, color=c, lw=2, label=f'{label}: {y_name}')
                else:
                    ax.plot(xx, yy, color=c, lw=2, label=label)

        if is_phase and xi == 1 and any(self._y_row.get(yi, -1) == 1 for yi in self._y_indices):
            origin_lbl = 'центр (0, 0)' if self.problem_type == 'kepler' else 'начало (0, 0)'
            ax.plot(0, 0, 'o', color='#7AA2F7', markersize=6, label=origin_lbl)
            ax.annotate('(0, 0)', (0, 0), textcoords='offset points',
                        xytext=(6, 6), color='#7AA2F7', fontsize=9)

        if len(self._y_indices) > 1:
            y_label = ", ".join(self._y_items[i] for i in self._y_indices)

        if is_phase:
            ax.set_xlabel(x_label, color='#9E9EB5'); ax.set_ylabel(y_label, color='#9E9EB5')
            ax.set_aspect('equal')
        else:
            ax.set_xlabel(x_label, color='#9E9EB5'); ax.set_ylabel(y_label, color='#9E9EB5')
        if ax.get_legend_handles_labels()[0]:
            leg = ax.legend(prop={'size': 9}, facecolor='#1C1C24',
                           edgecolor='#2E2E3A', labelcolor='#D9D9E3')
            leg.get_frame().set_alpha(0.92)

        ax.axhline(y=0, color='#35354A', linewidth=1.2, zorder=0)
        ax.axvline(x=0, color='#35354A', linewidth=1.2, zorder=0)
        ax.grid(True, alpha=0.18, linewidth=0.5, color='#4A4A65')

        m = 0.15
        if all_x and all_y:
            x_mi, x_ma = min(all_x), max(all_x)
            y_mi, y_ma = min(all_y), max(all_y)
        else:
            x_mi = x_ma = y_mi = y_ma = 0
        if x_ma == x_mi: x_ma = x_mi + 1.0
        if y_ma == y_mi: y_ma = y_mi + 1.0
        ax.set_xlim(x_mi - (x_ma-x_mi)*m, x_ma + (x_ma-x_mi)*m)
        ax.set_ylim(y_mi - (y_ma-y_mi)*m, y_ma + (y_ma-y_mi)*m)

        self.canvas.fig.tight_layout()
        self.canvas.draw()


class MainWindow(QMainWindow):
    TR = {
        'en': {
            'title': 'Parameter Continuation Method',
            'params': 'Problem Parameters',
            'dim': 'Dimension:',
            'a': 'a:',
            'b': 'b:',
            'eps': 'ε:',
            'mu': 'μ:',
            'integr': 'Integr:',
            'method': 'Method:',
            'steps': 'Steps:',
            'points': 'Points:',
            'name': 'Name:',
            'equations': 'Equations',
            'eq_header': 'd/dt[i] = f_i(x)',
            'variables': 'Variables:',
            'bc_label': 'Boundary conditions  (xi0 = xi(a),  xi1 = xi(b))',
            'init_guess': 'Initial Conditions',
            'unknowns': 'Unknowns:',
            'controls': 'Controls',
            'ready': 'Ready',
            'computing': 'Computing...',
            'done': 'Done',
            'error': 'Error',
            'cancelled': 'Cancelled',
            'solve': 'Solve',
            'graph': 'Graph',
            'combined': 'Combined',
            'export': 'Export',
            'clear': 'Clear',
            'load': 'Load',
            'examples': 'Examples',
            'examples_title': 'Task Examples',
            'examples_prompt': 'Select example:',
            'save_success': 'Success',
            'save_msg': 'Task saved',
            'save_error': 'Error',
            'save_error_msg': 'Could not save task:',
            'load_success': 'Success',
            'load_msg': 'Task loaded',
            'load_error': 'Error',
            'load_error_msg': 'Could not load task:',
            'solver_error': 'Solve Error',
            'no_result': 'Information',
            'no_result_msg': 'Compute first.',
            'load_dialog_title': 'Load Task',
            'save_dialog_title': 'Save Task',
            'mu_title': 'Parameter μ',
            'mu_prompt': 'Select values:',
            'mu_items': {'μ = 1': 1.0, 'μ = 10⁻¹': 1e-1, 'μ = 10⁻⁶': 1e-6},
            'eq_placeholder': 'f(t, x0, x1, ...)',
            'bc_placeholder': 'x1(a) = 2\nx2(a) = 0\nx1(b) = 1.0738644361\nx2(b) = -1.0995343576',
            'lang_label': 'Language',
            'about': 'Help',
            'instructions': 'Instructions',
            'instructions_text': '1. Enter equations in the d/dt[i] = f_i(x) format.\n2. Set boundary conditions: xi0 = xi(a), xi1 = xi(b).\n3. Specify known initial values by checking boxes.\n4. Enter initial guesses for unknown parameters.\n5. Click "Solve" to compute.\n6. Click "Graph" to visualize results.\n\nFor the Lens problem (26.4): select μ values in the popup dialog.\nFor custom problems with "mu" in equations: same μ selection dialog appears.',
            'about_author': 'About Author',
            'about_author_text': 'Автор: Ли Чжохэн\n313,\nМосковский государственный университет им. Ломоносова,\nФакультет вычислительной математики и кибернетики,\nКафедра оптимального управления,\nЭлектронная почта: stbc02220010@gse.cs.msu.ru\n\nРуководитель: Аввакумов Сергей Николаевич,\nМосковский государственный университет им. Ломоносова,\nКафедра оптимального управления,\nСтарший преподаватель\n\nОтветственный: Орлов Сергей Михайлович,\nДоцент кафедры ОУ,\nначальник курса\n\nГод создания: 2026',
            'history': 'History',
            'prep_error': 'Preparation error:',
            'continuation_param_label': 'Continuation parameter',
            'guess_warn_title': 'Not enough initial guesses',
            'guess_warn_msg': ('You entered {n} initial-guess value(s), but {m} '
                               'unknown(s) are expected.\nThe missing one(s) will '
                               'default to 0.0, which may give a divergent or '
                               'non-physical trajectory.\n\nContinue anyway?'),
            'sol_method_items': ['Auto', 'Continuation', 'Kepler',
                                 'Limit cycles', 'Triple integrator', 'Lens'],
            'multi_cycle': 'Multiple cycles',
            'multi_cycle_tip': ('Check to split the guess values into groups\n'
                                '— each group yields a separate cycle\n'
                                'overlaid on one plot.'),
        },
        'ru': {
            'title': 'Метод продолжения по параметру',
            'params': 'Параметры задачи',   
            'dim': 'Размерность:',
            'a': 'a:',
            'b': 'b:',
            'eps': 'ε:',
            'mu': 'μ:',
            'integr': 'Интегр:',
            'method': 'Метод:',
            'steps': 'Шаги:',
            'points': 'Точек:',
            'name': 'Название:',
            'equations': 'Уравнения',
            'eq_header': 'd/dt[i] = f_i(x)',
            'variables': 'Переменные:',
            'bc_label': 'Граничные условия  (xi0 = xi(a),  xi1 = xi(b))',
            'init_guess': 'Начальные условия',
            'unknowns': 'Неизвестные:',
            'controls': 'Управление',
            'ready': 'Готов',
            'computing': 'Вычисление...',
            'done': 'Готово',
            'error': 'Ошибка',
            'cancelled': 'Отменено',
            'solve': 'Решить',
            'graph': 'График',
            'combined': 'Объединить',
            'export': 'Экспорт',
            'clear': 'Очистить',
            'load': 'Загрузить',
            'examples': 'Примеры',
            'examples_title': 'Примеры задач',
            'examples_prompt': 'Выберите пример:',
            'save_success': 'Успех',
            'save_msg': 'Задача сохранена',
            'save_error': 'Ошибка',
            'save_error_msg': 'Не удалось сохранить задачу:',
            'load_success': 'Успех',
            'load_msg': 'Задача загружена',
            'load_error': 'Ошибка',
            'load_error_msg': 'Не удалось загрузить задачу:',
            'solver_error': 'Ошибка решения',
            'no_result': 'Информация',
            'no_result_msg': 'Сначала выполните вычисление.',
            'load_dialog_title': 'Загрузить задачу',
            'save_dialog_title': 'Сохранить задачу',
            'mu_title': 'Параметр μ',
            'mu_prompt': 'Выберите значения:',
            'mu_items': {'μ = 1': 1.0, 'μ = 10⁻¹': 1e-1, 'μ = 10⁻⁶': 1e-6},
            'eq_placeholder': 'f(t, x0, x1, ...)',
            'bc_placeholder': 'x1(a) = 2\nx2(a) = 0\nx1(b) = 1.0738644361\nx2(b) = -1.0995343576',
            'lang_label': 'Язык',
            'about': 'Помощь',
            'instructions': 'Инструкция',
            'instructions_text': '1. Введите уравнения в формате d/dt[i] = f_i(x).\n2. Задайте граничные условия: xi0 = xi(a), xi1 = xi(b).\n3. Отметьте известные начальные значения.\n4. Введите начальные приближения для неизвестных.\n5. Нажмите «Решить» для вычислений.\n6. Нажмите «График» для визуализации.\n\nДля задачи с лункой (26.4): выберите значения μ в диалоге.\nДля пользовательских задач с "mu" в уравнениях: появляется тот же диалог.',
            'about_author': 'Об авторе',
            'about_author_text': 'Автор: Ли Чжохэн\n313,\nМосковский государственный университет им. Ломоносова,\nФакультет вычислительной математики и кибернетики,\nКафедра оптимального управления,\nЭлектронная почта: stbc02220010@gse.cs.msu.ru\n\nРуководитель: Аввакумов Сергей Николаевич,\nМосковский государственный университет им. Ломоносова,\nКафедра оптимального управления,\nСтарший преподаватель\n\nОтветственный: Орлов Сергей Михайлович,\nДоцент кафедры ОУ,\nначальник курса\n\nГод создания: 2026',
            'history': 'История',
            'prep_error': 'Ошибка подготовки:',
            'continuation_param_label': 'Параметр продолжения',
            'guess_warn_title': 'Недостаточно начальных приближений',
            'guess_warn_msg': ('Введено {n} начальн(ое/ых) приближени(е/й), но '
                               'требуется {m}.\nНедостающие будут заменены на 0.0, '
                               'что может дать расходящуюся или нефизическую '
                               'траекторию.\n\nПродолжить?'),
            'sol_method_items': ['Авто', 'Продолжение', 'Кеплер',
                                 'Предельные циклы', 'Тройной интегратор', 'Лунка'],
            'multi_cycle': 'Несколько циклов',
            'multi_cycle_tip': ('Включите, чтобы разбить значения на группы —\n'
                                'каждая группа даёт отдельный цикл\n'
                                'на одном графике.'),
        },
    }

    def __init__(self):
        super().__init__()
        self.lang = 'ru'
        self.setGeometry(100, 100, 1180, 980)

        self.current_dataset = None
        self.solver_thread = None
        self._history = []
        self._history_datasets = {}

        self._last_x = None
        self._last_y = None
        self._last_name = ""
        self._last_guess_label = ""
        self._last_problem_type = "custom"
        self._status_state = 'ready'

        self.init_ui()
        self.apply_style()
        self.apply_language()
        self._load_history()

    def _tr(self, key):
        return self.TR.get(self.lang, self.TR['ru']).get(key, key)

    def closeEvent(self, event):
        import os
        path = os.path.join(os.path.dirname(__file__), 'history.json')
        if os.path.exists(path):
            os.remove(path)
        super().closeEvent(event)

    def apply_language(self):
        self.setWindowTitle(self._tr('title'))
        self.header_title.setText(self._tr('title'))
        self.params_group.setTitle(self._tr('params'))
        self.lbl_dim.setText(self._tr('dim'))
        self.lbl_a.setText(self._tr('a'))
        self.lbl_b.setText(self._tr('b'))
        self.lbl_eps.setText(self._tr('eps'))
        self.lbl_integr.setText(self._tr('integr'))
        self.lbl_method.setText(self._tr('method'))
        self.lbl_steps.setText(self._tr('steps'))
        self.lbl_points.setText(self._tr('points'))
        self.lbl_name.setText(self._tr('name'))
        self.eq_group.setTitle(self._tr('equations'))
        self.lbl_eq_header.setText(self._tr('eq_header'))
        self.lbl_variables.setText(self._tr('variables'))
        self.lbl_bc_label.setText(self._tr('bc_label'))
        self.bc_edit.setPlaceholderText(self._tr('bc_placeholder'))
        self.init_group.setTitle(self._tr('init_guess'))
        self.lbl_unknowns.setText(self._tr('unknowns'))
        self.control_group.setTitle(self._tr('controls'))
        self.solve_btn.setText(self._tr('solve'))
        self.graph_btn.setText(self._tr('graph'))
        self.export_btn.setText(self._tr('export'))
        self.clear_btn.setText(self._tr('clear'))
        self.load_btn.setText(self._tr('load'))
        self.example_btn.setText(self._tr('examples'))
        self.sol_method_combo.clear()
        self.sol_method_combo.addItems(self._tr('sol_method_items'))
        for i, ed in enumerate(self.equation_edits):
            ed.setPlaceholderText(self._tr('eq_placeholder'))
        state = getattr(self, '_status_state', 'ready')
        if state == 'computing':
            self._set_status(self._tr('computing'), 'computing')
        else:
            self._set_status(self._tr('ready'), 'ready')
        self.history_group.setTitle(self._tr('history'))
        self.btn_lang.setText(self._tr('lang_label'))
        self.btn_about.setText(self._tr('about'))
        self.multi_cycle_cb.setText(self._tr('multi_cycle'))
        self.multi_cycle_cb.setToolTip(self._tr('multi_cycle_tip'))
        self.about_menu.clear()
        self.about_menu.addAction(self._tr('instructions'), self._on_show_instructions)
        self.about_menu.addAction(self._tr('about_author'), self._on_show_author)

    def _on_lang_selected(self, lang_code):
        self.lang = lang_code
        self.apply_language()

    def _on_show_instructions(self):
        QMessageBox.information(self, self._tr('instructions'), self._tr('instructions_text'))

    def _on_show_author(self):
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        from PyQt5.QtGui import QPixmap
        import os
        dlg = QDialog(self)
        dlg.setWindowTitle(self._tr('about_author'))
        dlg.resize(420, 500)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(8)
        lbl_text = QLabel(self._tr('about_author_text'))
        lbl_text.setWordWrap(True)
        lbl_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(lbl_text)
        photo_path = os.path.join(os.path.dirname(__file__), 'examples', 'photo', '222.jpg')
        if os.path.exists(photo_path):
            lbl_img = QLabel()
            pm = QPixmap(photo_path).scaledToWidth(260, Qt.SmoothTransformation)
            lbl_img.setPixmap(pm)
            lbl_img.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl_img)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.exec_()

    def _parse_varnames(self):
        text = self.var_names_edit.text().strip()
        if not text:
            return []
        return [v.strip() for v in text.split(',')]

    def _derivative_to_ddt(self, eq_str, varnames):
        m = re.match(r'Derivative\(x(\d+)\((\w+)\),\s*(\w+)\)\s*([+-])\s*(.*?)\s*=\s*0$', eq_str)
        if m:
            idx = int(m.group(1)) - 1
            op = m.group(4)
            rhs = m.group(5).strip()
            if op == '+':
                rhs = '-' + rhs
            for i, vn in enumerate(varnames):
                rhs = re.sub(r'\b' + re.escape('x' + str(i+1)) + r'\b', vn, rhs)
            return f'd/dt[{idx}]: {rhs}'
        m2 = re.match(r'Derivative\(x(\d+)\((\w+)\),\s*(\w+)\)\s*=\s*0$', eq_str)
        if m2:
            idx = int(m2.group(1)) - 1
            return f'd/dt[{idx}]: 0'
        return eq_str

    def _ddt_to_derivative(self, line, varnames):
        line = line.strip()
        if not line:
            return line
        m = re.match(r'd/dt\[(\d+)\]\s*:\s*(.*)', line)
        if not m:
            return line
        idx = int(m.group(1))
        rhs = m.group(2).strip()
        if rhs == '0':
            return f'Derivative(x{idx+1}(t), t) = 0'
        rhs_x = rhs
        for i, vn in enumerate(varnames):
            if vn:
                rhs_x = re.sub(r'\b' + re.escape(vn) + r'\b', 'x' + str(i+1), rhs_x)
        if rhs_x.startswith('-'):
            return f'Derivative(x{idx+1}(t), t) + {rhs_x[1:].strip()} = 0'
        return f'Derivative(x{idx+1}(t), t) - {rhs_x} = 0'

    def apply_style(self):
        """Refined dark theme — onyx background, soft sapphire accents, clean geometry."""
        self.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: #16161A;
            }
            QWidget {
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                font-size: 9.5pt;
                color: #D9D9E3;
                background-color: #16161A;
            }

            /* ── Header ── */
            QFrame#headerFrame {
                background-color: #0F0F14;
                border: none;
                border-bottom: 1px solid #2A2A35;
            }
            QLabel#headerTitle {
                color: #E2E2EE;
                font-size: 15pt;
                font-weight: 700;
                background: transparent;
                letter-spacing: 0.3px;
            }

            /* ── GroupBox — card with top accent bar ── */
            QGroupBox {
                background-color: #1C1C24;
                border: 1px solid #2E2E3A;
                border-top: 2px solid #7AA2F7;
                border-radius: 10px;
                margin-top: 18px;
                padding: 16px 14px 14px 14px;
                font-weight: 700;
                font-size: 10pt;
                color: #C5C5D8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 2px 12px;
                color: #7AA2F7;
                background-color: #1C1C24;
                font-size: 10pt;
                font-weight: 700;
                letter-spacing: 0.3px;
            }
            QGroupBox QLabel {
                background-color: transparent;
                color: #9E9EB5;
            }
            QGroupBox QCheckBox {
                background-color: transparent;
            }

            /* ── Inputs ── */
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: #121218;
                border: 1px solid #35354A;
                border-radius: 7px;
                padding: 7px 10px;
                color: #D9D9E3;
                font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 9pt;
                selection-background-color: #3B5F9E;
                selection-color: #E2E8F0;
            }
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
                border-color: #7AA2F7;
                background-color: #121218;
            }
            QLineEdit:hover, QTextEdit:hover {
                border-color: #4A4A65;
            }

            QSpinBox, QDoubleSpinBox {
                background-color: #121218;
                border: 1px solid #35354A;
                border-radius: 7px;
                padding: 4px 7px;
                color: #D9D9E3;
                font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 9pt;
            }
            QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7AA2F7; }
            QSpinBox:hover, QDoubleSpinBox:hover { border-color: #4A4A65; }

            QComboBox {
                background-color: #121218;
                border: 1px solid #35354A;
                border-radius: 7px;
                padding: 5px 10px;
                color: #D9D9E3;
                font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 9pt;
            }
            QComboBox:focus { border-color: #7AA2F7; }
            QComboBox:hover { border-color: #4A4A65; }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right;
                width: 22px;
                border-left: 1px solid #2E2E3A;
                border-top-right-radius: 7px;
                border-bottom-right-radius: 7px;
                background: #24243A;
            }
            QComboBox QAbstractItemView {
                background: #1C1C24;
                border: 1px solid #2E2E3A;
                border-radius: 6px;
                selection-background-color: rgba(122, 162, 247, 0.16);
                selection-color: #C5D5FB;
                color: #D9D9E3;
                padding: 4px;
                outline: none;
            }

            /* ── Primary button ── */
            QPushButton {
                background-color: #3B5F9E;
                color: #E2E8F0;
                border: none;
                border-radius: 7px;
                padding: 8px 22px;
                font-size: 9.5pt;
                font-weight: 700;
                letter-spacing: 0.2px;
            }
            QPushButton:hover { background-color: #4A75C0; }
            QPushButton:pressed { background-color: #2E4D85; }
            QPushButton:disabled {
                background-color: #24242E;
                color: #505065;
            }

            /* ── Solve — warm green ── */
            QPushButton#solveBtn {
                background-color: #3B8C5A;
                color: #E5F5EA;
                border-radius: 8px;
                font-size: 11pt;
                font-weight: 800;
                padding: 11px 30px;
                letter-spacing: 0.6px;
            }
            QPushButton#solveBtn:hover { background-color: #4CA86F; }
            QPushButton#solveBtn:pressed { background-color: #2E7045; }
            QPushButton#solveBtn:disabled {
                background-color: #24242E;
                color: #505065;
            }

            /* ── Secondary / outline ── */
            QPushButton#secondaryBtn {
                background: transparent;
                color: #7AA2F7;
                border: 1.5px solid #35354A;
                border-radius: 7px;
                padding: 6px 16px;
                font-size: 9pt;
                font-weight: 600;
            }
            QPushButton#secondaryBtn:hover {
                background-color: rgba(122, 162, 247, 0.08);
                border-color: #7AA2F7;
                color: #A8C4FB;
            }

            /* ── Header pill buttons ── */
            QPushButton#headerBtn {
                background: transparent;
                color: #7AA2F7;
                border: 1.5px solid #35354A;
                border-radius: 8px;
                padding: 6px 16px;
                font-size: 9.5pt;
                font-weight: 600;
            }
            QPushButton#headerBtn:hover {
                background-color: rgba(122, 162, 247, 0.08);
                border-color: #7AA2F7;
                color: #A8C4FB;
            }

            /* ── Plot toolbar ── */
            QPushButton#plotToolBtn {
                background: transparent;
                color: #7AA2F7;
                border: 1.5px solid #35354A;
                border-radius: 7px;
                padding: 5px 14px;
                font-size: 9pt;
                font-weight: 600;
            }
            QPushButton#plotToolBtn:hover {
                background-color: rgba(122, 162, 247, 0.08);
                border-color: #7AA2F7;
            }

            /* ── CheckBox ── */
            QCheckBox {
                color: #C5C5D8;
                spacing: 9px;
                font-size: 9.5pt;
                background: transparent;
            }
            QCheckBox::indicator {
                width: 17px;
                height: 17px;
                border-radius: 4px;
                border: 1.5px solid #4A4A65;
                background: #121218;
            }
            QCheckBox::indicator:checked {
                background: #7AA2F7;
                border-color: #7AA2F7;
            }
            QCheckBox::indicator:hover { border-color: #7AA2F7; }

            /* ── Tabs ── */
            QTabWidget::pane {
                border: 1px solid #2E2E3A;
                border-radius: 8px;
                background: #16161A;
            }
            QTabBar::tab {
                background: #1C1C24;
                color: #9E9EB5;
                border: 1px solid #2E2E3A;
                padding: 9px 20px;
                margin-right: 3px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }
            QTabBar::tab:selected {
                background: #16161A;
                color: #7AA2F7;
                border-bottom-color: #16161A;
            }
            QTabBar::tab:hover { background: #24243A; }

            /* ── Scrollbar ── */
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: transparent;
                width: 9px;
                border-radius: 4px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #35354A;
                border-radius: 4px;
                min-height: 32px;
            }
            QScrollBar::handle:vertical:hover { background: #4A4A65; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: transparent;
                height: 9px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #35354A;
                border-radius: 4px;
                min-width: 32px;
            }
            QScrollBar::handle:horizontal:hover { background: #4A4A65; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

            /* ── List ── */
            QListWidget {
                background: #121218;
                border: 1px solid #2E2E3A;
                border-radius: 9px;
                font-size: 9pt;
                color: #D9D9E3;
                outline: none;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px 14px;
                border-radius: 5px;
                margin: 1px 0;
            }
            QListWidget::item:hover { background: #24243A; }
            QListWidget::item:selected {
                background: rgba(122, 162, 247, 0.14);
                color: #C5D5FB;
            }

            /* ── Menu ── */
            QMenu {
                background: #1C1C24;
                border: 1px solid #2E2E3A;
                border-radius: 9px;
                padding: 6px;
                color: #D9D9E3;
            }
            QMenu::item {
                padding: 8px 22px;
                border-radius: 5px;
            }
            QMenu::item:selected {
                background: rgba(122, 162, 247, 0.14);
                color: #A8C4FB;
            }
            QMenu::separator {
                height: 1px;
                background: #2E2E3A;
                margin: 5px 10px;
            }

            /* ── Tooltip ── */
            QToolTip {
                background: #24243A;
                color: #D9D9E3;
                border: 1px solid #35354A;
                border-radius: 7px;
                padding: 7px 12px;
                font-size: 9pt;
            }

            /* ── Dialogs ── */
            QDialogButtonBox QPushButton { min-width: 85px; }
            QPlainTextEdit {
                background-color: #121218;
                border: 1px solid #35354A;
                border-radius: 7px;
                padding: 8px;
                color: #D9D9E3;
                font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 9pt;
            }
            QPlainTextEdit:focus { border-color: #7AA2F7; }

            /* ── QTextEdit in group boxes ── */
            QGroupBox QTextEdit {
                background-color: #121218;
                border: 1px solid #35354A;
                border-radius: 7px;
                padding: 6px;
                color: #D9D9E3;
                font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 9pt;
            }
            QGroupBox QTextEdit:focus { border-color: #7AA2F7; }
        """)

    def _create_header(self):
        """Light header bar with title and top-level pill buttons.

        Style: warm-white background, indigo accent, soft bottom border.
        """
        frame = QFrame()
        frame.setObjectName("headerFrame")
        frame.setFixedHeight(64)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(24, 0, 18, 0)
        layout.setSpacing(10)

        # Small decorative dot for visual interest
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet("background: #7AA2F7; border-radius: 5px;")
        layout.addWidget(dot)
        layout.addSpacing(6)

        self.header_title = QLabel()
        self.header_title.setObjectName("headerTitle")
        layout.addWidget(self.header_title)
        layout.addStretch()

        self.btn_lang = QPushButton()
        self.btn_lang.setObjectName("headerBtn")
        self.btn_lang.setFixedWidth(112)
        self.lang_menu = QMenu(self)
        self.lang_menu.addAction("Русский", lambda: self._on_lang_selected('ru'))
        self.lang_menu.addAction("English", lambda: self._on_lang_selected('en'))
        self.btn_lang.setMenu(self.lang_menu)
        layout.addWidget(self.btn_lang)

        self.btn_about = QPushButton()
        self.btn_about.setObjectName("headerBtn")
        self.btn_about.setFixedWidth(124)
        self.about_menu = QMenu(self)
        self.about_menu.addAction(self._tr('instructions'), self._on_show_instructions)
        self.about_menu.addAction(self._tr('about_author'), self._on_show_author)
        self.btn_about.setMenu(self.about_menu)
        layout.addWidget(self.btn_about)

        return frame

    def _set_status(self, text, state='ready'):
        """Pill-shaped status badge tinted for the dark theme."""
        self._status_state = state
        _pal = {
            'ready':     'color:#9E9EB5;background:#1C1C24;'
                         'border:1px solid #35354A;',
            'computing': 'color:#7AA2F7;background:rgba(122,162,247,0.10);'
                         'border:1px solid #7AA2F7;',
            'done':      'color:#6FCF97;background:rgba(111,207,151,0.10);'
                         'border:1px solid #6FCF97;',
            'error':     'color:#F7768E;background:rgba(247,118,142,0.09);'
                         'border:1px solid #F7768E;',
            'cancelled': 'color:#E0AF68;background:rgba(224,175,104,0.09);'
                         'border:1px solid #E0AF68;',
        }
        base = ('border-radius:13px;padding:5px 22px;'
                'font-weight:700;font-size:9.5pt;text-align:center;'
                'letter-spacing:0.4px;')
        self.status_label.setStyleSheet(base + _pal.get(state, _pal['ready']))
        self.status_label.setText(text)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ── Dark header bar (title + lang + help buttons) ──
        outer_layout.addWidget(self._create_header())

        # ── Main body ──
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(10, 8, 10, 8)
        body_layout.setSpacing(0)

        split = QHBoxLayout()
        split.setSpacing(8)
        split.addWidget(self.create_left_panel(), 3)
        split.addWidget(self.create_right_panel(), 2)
        body_layout.addLayout(split)
        outer_layout.addWidget(body)

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 2, 4, 2)
        layout.setSpacing(6)
        layout.addWidget(self.create_params_group())
        layout.addWidget(self.create_equation_group())
        layout.addWidget(self.create_initial_guess_group())
        layout.addWidget(self.create_solve_group())
        layout.addStretch()
        return panel

    def _auto_fit_window(self):
        """Resize the window so that all content is visible without scrolling."""
        n_dim = len(self.equation_edits)
        adv_open = self.adv_widget.isVisible() if hasattr(self, 'adv_widget') else False
        header = 64
        params = 130 + (40 if adv_open else 0)
        eq_group = 70 + 32 * n_dim
        init = 30 + 34 * n_dim + 32
        solve = 160
        margins = 30
        needed_h = header + params + eq_group + init + solve + margins
        clamped = max(680, min(1050, needed_h))
        self.resize(self.width(), clamped)

    def create_right_panel(self):
        self.history_group = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.history_list.setAlternatingRowColors(True)
        self.history_list.setToolTip("Previously solved tasks — click to reload")
        layout.addWidget(self.history_list)
        self.history_group.setLayout(layout)
        return self.history_group

    def create_params_group(self):
        self.params_group = QGroupBox()
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # ── Row 1: dim, a, b ──
        row1 = QHBoxLayout()
        self.lbl_dim = QLabel()
        row1.addWidget(self.lbl_dim)
        self.dim_spin = QSpinBox()
        self.dim_spin.setRange(1, 20)
        self.dim_spin.setValue(4)
        self.dim_spin.setFixedWidth(55)
        self.dim_spin.setToolTip("Number of first‑order ODEs (variables)")
        row1.addWidget(self.dim_spin)
        row1.addSpacing(8)
        self.lbl_a = QLabel()
        row1.addWidget(self.lbl_a)
        self.x_start_spin = QDoubleSpinBox()
        self.x_start_spin.setRange(-1000, 1000)
        self.x_start_spin.setValue(0.0)
        self.x_start_spin.setDecimals(3)
        self.x_start_spin.setFixedWidth(75)
        self.x_start_spin.setToolTip("Start of integration interval")
        row1.addWidget(self.x_start_spin)
        self.lbl_b = QLabel()
        row1.addWidget(self.lbl_b)
        self.x_end_spin = QDoubleSpinBox()
        self.x_end_spin.setRange(-1000, 1000)
        self.x_end_spin.setValue(7.0)
        self.x_end_spin.setDecimals(3)
        self.x_end_spin.setFixedWidth(75)
        self.x_end_spin.setToolTip("End of integration interval (must be > a)")
        row1.addWidget(self.x_end_spin)
        row1.addStretch()
        layout.addLayout(row1)

        # ── Name field ──
        row_name = QHBoxLayout()
        self.lbl_name = QLabel()
        row_name.addWidget(self.lbl_name)
        self.name_edit = QLineEdit("Моя задача")
        self.name_edit.setToolTip("Descriptive task name (shown in window title & history)")
        row_name.addWidget(self.name_edit)
        layout.addLayout(row_name)

        # ── Advanced settings (collapsible) ──
        self.adv_toggle = QPushButton("▼ Advanced")
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setChecked(False)
        self.adv_toggle.setObjectName("secondaryBtn")
        self.adv_toggle.setToolTip("Show/hide solver tolerance, integrator & continuation steps")
        self.adv_toggle.toggled.connect(self._on_advanced_toggled)
        layout.addWidget(self.adv_toggle)

        self.adv_widget = QWidget()
        adv_layout = QVBoxLayout(self.adv_widget)
        adv_layout.setContentsMargins(0, 2, 0, 0)
        adv_layout.setSpacing(4)

        row2 = QHBoxLayout()
        self.lbl_eps = QLabel()
        row2.addWidget(self.lbl_eps)
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(1e-14, 1.0)
        self.eps_spin.setDecimals(12)
        self.eps_spin.setValue(1e-9)
        self.eps_spin.setFixedWidth(105)
        self.eps_spin.setToolTip("ODE solver tolerance (default 1e‑9)")
        row2.addWidget(self.eps_spin)
        row2.addSpacing(8)
        self.lbl_integr = QLabel()
        row2.addWidget(self.lbl_integr)
        self.int_method_combo = QComboBox()
        self.int_method_combo.addItems(
            ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"])
        self.int_method_combo.setFixedWidth(80)
        self.int_method_combo.setToolTip("scipy.integrate.solve_ivp method — RK45 for non‑stiff, BDF for stiff")
        row2.addWidget(self.int_method_combo)
        row2.addStretch()
        adv_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.lbl_steps = QLabel()
        row3.addWidget(self.lbl_steps)
        self.cont_steps_spin = QSpinBox()
        self.cont_steps_spin.setRange(1, 200)
        self.cont_steps_spin.setValue(50)
        self.cont_steps_spin.setFixedWidth(55)
        self.cont_steps_spin.setToolTip("Number of continuation sub‑steps between large parameter jumps")
        row3.addWidget(self.cont_steps_spin)
        self.lbl_points = QLabel()
        row3.addWidget(self.lbl_points)
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(10, 10000)
        self.n_points_spin.setValue(500)
        self.n_points_spin.setFixedWidth(75)
        self.n_points_spin.setToolTip("Number of output points along the solution")
        row3.addWidget(self.n_points_spin)
        self.lbl_method = QLabel()
        row3.addWidget(self.lbl_method)
        self.sol_method_combo = QComboBox()
        self.sol_method_combo.addItems(["Продолжение"])
        self.sol_method_combo.setMinimumWidth(90)
        self.sol_method_combo.setToolTip(
            "Solver strategy:\n"
            "• Auto — detect from equations/BCs\n"
            "• Continuation — generic shooting\n"
            "• Kepler / Limit cycles / Triple / Lens — specialised solvers")
        row3.addWidget(self.sol_method_combo)
        row3.addStretch()
        adv_layout.addLayout(row3)

        self.adv_widget.setVisible(False)
        layout.addWidget(self.adv_widget)

        self.params_group.setLayout(layout)
        return self.params_group

    def _on_advanced_toggled(self, checked):
        self.adv_widget.setVisible(checked)
        self.adv_toggle.setText("▲ Advanced" if checked else "▼ Advanced")
        self._auto_fit_window()

    def create_equation_group(self):
        self.eq_group = QGroupBox()
        layout = QVBoxLayout()
        layout.setSpacing(2)

        self.lbl_eq_header = QLabel()
        layout.addWidget(self.lbl_eq_header)

        self.eq_grid = QGridLayout()
        self.eq_grid.setSpacing(1)
        self.equation_edits = []
        self.eq_labels = []
        n = self.dim_spin.value()
        for i in range(n):
            name = f'x{i+1}'
            lbl = QLabel(f"{name} ' =")
            lbl.setMinimumWidth(36)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setToolTip(f"Right-hand side of d({name})/dt")
            self.eq_labels.append(lbl)
            self.eq_grid.addWidget(lbl, i, 0)

            edit = QLineEdit()
            edit.setToolTip(f"f(t, x1…x{n}) — RHS for d({name})/dt")
            edit.setPlaceholderText("e.g. x2" if i == 0 else "")
            self.equation_edits.append(edit)
            self.eq_grid.addWidget(edit, i, 1)
        self._set_eq_defaults()
        layout.addLayout(self.eq_grid)

        row_vn = QHBoxLayout()
        self.lbl_variables = QLabel()
        row_vn.addWidget(self.lbl_variables)
        self.var_names_edit = QLineEdit("x1, x2, x3, x4")
        self.var_names_edit.setToolTip("Comma‑separated variable names (used in plot legends)")
        row_vn.addWidget(self.var_names_edit)
        layout.addLayout(row_vn)

        self.lbl_bc_label = QLabel()
        layout.addWidget(self.lbl_bc_label)
        self.bc_edit = QTextEdit()
        self.bc_edit.setPlaceholderText("x1(a) = 2\nx2(a) = 0\nx1(b) = 1.07\nx2(b) = -1.10")
        self.bc_edit.setFixedHeight(65)
        self.bc_edit.setToolTip("Boundary conditions in the form  xi(a)=val  or  xi(b)=val, one per line")
        layout.addWidget(self.bc_edit)

        self.dim_spin.valueChanged.connect(self._on_dim_changed)
        self.eq_group.setLayout(layout)
        # Re-evaluate multi-cycle visibility whenever equations or BC change
        for ed in self.equation_edits:
            ed.textChanged.connect(self._update_multi_cycle_visibility)
        self.bc_edit.textChanged.connect(self._update_multi_cycle_visibility)
        return self.eq_group

    def _set_eq_defaults(self):
        defaults = [
            "x2", "-x1/(x1**2 + x3**2)**1.5",
            "x4", "-x3/(x1**2 + x3**2)**1.5",
        ]
        for i, edit in enumerate(self.equation_edits):
            if not edit.text():
                edit.setText(defaults[i] if i < len(defaults) else "")

    def _on_dim_changed(self, new_dim):
        for i in reversed(range(self.eq_grid.count())):
            w = self.eq_grid.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.equation_edits.clear()
        self.eq_labels.clear()

        for i in range(new_dim):
            name = f'x{i+1}'
            lbl = QLabel(f"{name} ' =")
            lbl.setMinimumWidth(36)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setToolTip(f"Right-hand side of d({name})/dt")
            self.eq_labels.append(lbl)
            self.eq_grid.addWidget(lbl, i, 0)

            edit = QLineEdit()
            edit.setToolTip(f"f(t, x1…x{new_dim}) — RHS for d({name})/dt")
            edit.setPlaceholderText("e.g. x2" if i == 0 else "")
            self.equation_edits.append(edit)
            self.eq_grid.addWidget(edit, i, 1)

        cur = [v.strip() for v in self.var_names_edit.text().split(',') if v.strip()]
        needed = ['x' + str(i + 1) for i in range(new_dim)]
        if len(cur) != new_dim:
            self.var_names_edit.setText(', '.join(needed))
        # Re-connect visibility signals to the new editors
        for ed in self.equation_edits:
            ed.textChanged.connect(self._update_multi_cycle_visibility)
        self._update_multi_cycle_visibility()
        self._auto_fit_window()

    def create_initial_guess_group(self):
        self.init_group = QGroupBox()
        layout = QVBoxLayout()
        layout.setSpacing(2)

        self.ic_known = []
        self.ic_value = []
        n = 4
        grid = QGridLayout()
        grid.setSpacing(2)
        for i in range(n):
            name = f'x{i+1}'
            cb = QCheckBox(f"{name}(a) known")
            cb.setChecked(i % 2 == 0)
            cb.setToolTip(f"Checked → use spinbox value; unchecked → pull from guess list below")
            self.ic_known.append(cb)
            grid.addWidget(cb, i, 0)

            sp = QDoubleSpinBox()
            sp.setRange(-1000, 1000)
            sp.setDecimals(6)
            sp.setValue([2.0, -0.5, 0.0, 0.5][i])
            sp.setFixedWidth(85)
            sp.setToolTip(f"Initial value for {name}(a) — used only when checkbox is checked")
            self.ic_value.append(sp)
            grid.addWidget(sp, i, 1)
        layout.addLayout(grid)

        guess_row = QHBoxLayout()
        self.lbl_unknowns = QLabel()
        guess_row.addWidget(self.lbl_unknowns)
        self.guess_edit = QLineEdit("-0.5, 0.5")
        self.guess_edit.setToolTip(
            "Comma‑separated guesses for UNCHECKED unknowns:\n"
            "one value per unknown, in x1, x2, … order.\n"
            "Limit‑cycle problems: enter x1 amplitudes, e.g. 2, 6.5, 9 —\n"
            "with 'Multiple cycles' checked each amplitude yields one cycle.")
        guess_row.addWidget(self.guess_edit)
        layout.addLayout(guess_row)

        self.multi_cycle_cb = QCheckBox("Multiple cycles")
        self.multi_cycle_cb.setToolTip(
            "Check to split the guess values into groups of unknowns "
            "→ each group yields a separate cycle overlaid on one plot.")
        self.multi_cycle_cb.setVisible(False)
        layout.addWidget(self.multi_cycle_cb)

        self.init_group.setLayout(layout)
        return self.init_group

    def _update_multi_cycle_visibility(self):
        """Show the multi-cycle checkbox only for limit‑cycle‑shaped problems."""
        if not hasattr(self, 'multi_cycle_cb'):
            return
        if self.multi_cycle_cb is None:
            return
        import re as _re
        dim = len(self.equation_edits)
        eqs = ' '.join([ed.text() for ed in self.equation_edits]).lower()
        bc = self.bc_edit.toPlainText().lower()
        is_lc = (
            dim >= 4 and 'sin' in eqs and
            _re.search(r'x2\(a\)\s*=\s*0', bc) and
            _re.search(r'x2\(b\)\s*=\s*0', bc) and
            _re.search(r'derivative\(x3', eqs)
        ) or False
        self.multi_cycle_cb.setVisible(is_lc)
        if not is_lc:
            self.multi_cycle_cb.setChecked(False)

    def create_solve_group(self):
        self.control_group = QGroupBox()
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Status badge — styled dynamically by _set_status()
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(26)
        layout.addWidget(self.status_label)

        # Primary row: Solve (green, large) + Graph (blue)
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(6)
        self.solve_btn = QPushButton()
        self.solve_btn.setObjectName("solveBtn")
        self.solve_btn.clicked.connect(self.start_solve)
        self.solve_btn.setMinimumHeight(36)
        self.solve_btn.setToolTip("Run the BVP solver with the current equations and initial guess")
        btn_row1.addWidget(self.solve_btn, 2)

        self.graph_btn = QPushButton()
        self.graph_btn.clicked.connect(self.show_plot_window)
        self.graph_btn.setMinimumHeight(36)
        self.graph_btn.setToolTip("Open plot window (or re‑open the last one)")
        btn_row1.addWidget(self.graph_btn, 1)
        layout.addLayout(btn_row1)

        # Secondary row: outline-style utility buttons
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(5)
        _tips = {
            'export_btn': "Save current task as JSON file",
            'clear_btn': "Clear all fields and reset to defaults",
            'load_btn': "Load a task from a JSON file",
            'example_btn': "Load a built-in example",
        }
        for attr, slot in [
            ('export_btn', self.save_task),
            ('clear_btn',  self.clear_ui),
            ('load_btn',   self.load_task),
            ('example_btn', self.show_examples),
        ]:
            btn = QPushButton()
            btn.setObjectName("secondaryBtn")
            btn.clicked.connect(slot)
            btn.setMinimumHeight(26)
            btn.setToolTip(_tips.get(attr, ''))
            setattr(self, attr, btn)
            btn_row2.addWidget(btn)
        layout.addLayout(btn_row2)

        self.control_group.setLayout(layout)
        return self.control_group

    def get_dataset_from_ui(self):
        varnames = self._parse_varnames()
        lines = [f"d/dt[{i}]: {ed.text().strip()}" for i, ed in enumerate(self.equation_edits) if ed.text().strip()]
        equations = [self._ddt_to_derivative(line, varnames) for line in lines]
        bc_text = self.bc_edit.toPlainText().strip()
        boundary_conditions = [line.strip() for line in bc_text.split('\n') if line.strip()]
        continuation_param = 'lambda'
        continuation_start = 0.0
        continuation_end = 1.0
        parameters = None
        if self.current_dataset is not None:
            continuation_param = self.current_dataset.continuation_param
            continuation_start = self.current_dataset.continuation_start
            continuation_end = self.current_dataset.continuation_end
            parameters = self.current_dataset.parameters
        return Dataset(
            name=self.name_edit.text(),
            x_start=self.x_start_spin.value(),
            x_end=self.x_end_spin.value(),
            equations=equations,
            boundary_conditions=boundary_conditions,
            n_points=self.n_points_spin.value(),
            continuation_steps=self.cont_steps_spin.value(),
            continuation_param=continuation_param,
            continuation_start=continuation_start,
            continuation_end=continuation_end,
            parameters=parameters,
            tol=self.eps_spin.value(),
            method=self.int_method_combo.currentText(),
        )

    def set_dataset_to_ui(self, dataset):
        self.name_edit.setText(dataset.name)
        self.x_start_spin.setValue(dataset.x_start)
        self.x_end_spin.setValue(dataset.x_end)
        self.n_points_spin.setValue(dataset.n_points)
        self.cont_steps_spin.setValue(dataset.continuation_steps)
        # ε и метод интегрирования (getattr — совместимость со старыми JSON)
        self.eps_spin.setValue(getattr(dataset, 'tol', 1e-9))
        method = getattr(dataset, 'method', 'RK45')
        midx = self.int_method_combo.findText(method)
        if midx >= 0:
            self.int_method_combo.setCurrentIndex(midx)
        self.dim_spin.blockSignals(True)
        self.dim_spin.setValue(len(dataset.equations))
        self._on_dim_changed(len(dataset.equations))
        self.dim_spin.blockSignals(False)
        varnames = self._parse_varnames()
        for i, eq in enumerate(dataset.equations):
            if i < len(self.equation_edits):
                ddt = self._derivative_to_ddt(eq, varnames)
                rhs = ddt.split(':', 1)[-1].strip() if ':' in ddt else ddt
                self.equation_edits[i].setText(rhs)
        self.bc_edit.setPlainText('\n'.join(dataset.boundary_conditions))

        # ── Sync initial-guess UI to BCs ────────────────────────────────
        # For each "xi(a) = val" boundary condition, tick the corresponding
        # checkbox and copy val into the spinbox.  Components without an
        # a-side BC are left as "unknown" (checkbox off).
        a_side = {}
        for bc in dataset.boundary_conditions:
            m = re.match(r'\s*x(\d+)\(a\)\s*=\s*([+-]?[\d.eE+\-]+)\s*$', bc)
            if m:
                a_side[int(m.group(1)) - 1] = float(m.group(2))  # 0-indexed
        for i in range(len(self.ic_known)):
            if i in a_side:
                self.ic_known[i].setChecked(True)
                self.ic_value[i].setValue(a_side[i])
            else:
                self.ic_known[i].setChecked(False)
                self.ic_value[i].setValue(0.0)
        prob_type = detect_problem_type(dataset)
        is_lc = prob_type == 'limit_cycle'
        if is_lc:
            # Предельные циклы: поле «Неизвестные» — список амплитуд x1.
            # Пример 26.2 задаёт три приближения p01,p02,p03 (x1 = 2; 6.5; 9)
            # → три предельных цикла, наложенных на один график.
            self.guess_edit.setText('2, 6.5, 9')
        elif prob_type == 'kepler':
            # Кеплер: поле «Неизвестные» — догадки начальной скорости (vx, vy).
            # Пример 26.1 задаёт ДВЕ догадки и приводит ОБЕ траектории на
            # рис. 26.1:
            #   p01 = [2, 0, -0.5,  0.5] → ans1 = (0.0000, 0.5000)  (траект. 1)
            #   p02 = [2, 0,  0.5, -0.5] → ans2 = (0.4511, -0.2994) (траект. 2)
            # Каждая пара (vx, vy) — отдельная догадка пристрелки; обе орбиты
            # накладываются на один график (см. _solve_kepler).
            self.guess_edit.setText('-0.5, 0.5, 0.5, -0.5')
        else:
            n_unknowns = max(0, len(dataset.equations) - len(a_side))
            if n_unknowns > 0:
                defaults = ['0.5'] * max(2, n_unknowns)
                self.guess_edit.setText(', '.join(defaults))
        # Visibility first (it un-checks for non-LC problems), then enable
        # the multi-cycle overlay for the limit-cycle example.
        self._update_multi_cycle_visibility()
        self.multi_cycle_cb.setChecked(is_lc)
        self._auto_fit_window()

    def clear_ui(self):
        for ed in self.equation_edits:
            ed.clear()
        self.bc_edit.clear()
        self.guess_edit.clear()
        self._last_x = None
        self._last_y = None
        self._last_problem_type = "custom"
        self._set_status(self._tr('ready'), 'ready')

    def load_task(self):
        import os
        start_dir = ""
        examples_path = os.path.join(os.path.dirname(__file__), 'examples')
        if os.path.isdir(examples_path):
            start_dir = examples_path
        filepath, _ = QFileDialog.getOpenFileName(
            self, self._tr('load_dialog_title'), start_dir, "JSON Files (*.json)"
        )
        if filepath:
            try:
                dataset = load_task(filepath)
                self.set_dataset_to_ui(dataset)
                self.current_dataset = dataset
                self.setWindowTitle(dataset.name + ' — ' + self._tr('title'))
                QMessageBox.information(self, self._tr('load_success'), self._tr('load_msg'))
            except Exception as e:
                QMessageBox.critical(self, self._tr('load_error'),
                                     f"{self._tr('load_error_msg')} {str(e)}")

    def save_task(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, self._tr('save_dialog_title'), "", "JSON Files (*.json)"
        )
        if filepath:
            try:
                dataset = self.get_dataset_from_ui()
                save_task(dataset, filepath)
                QMessageBox.information(self, self._tr('save_success'), self._tr('save_msg'))
            except Exception as e:
                QMessageBox.critical(self, self._tr('save_error'),
                                     f"{self._tr('save_error_msg')} {str(e)}")

    def show_examples(self):
        try:
            examples = get_example_tasks()
            items = [f"{i+1}. {ex.name}" for i, ex in enumerate(examples)]
            item, ok = QInputDialog.getItem(
                self, self._tr('examples_title'), self._tr('examples_prompt'), items, 0, False
            )
            if ok and item:
                idx = items.index(item)
                self.set_dataset_to_ui(examples[idx])
                self.current_dataset = examples[idx]
                # Оставляем метод на «Авто»: тип задачи определяется во время
                # решения по ТЕКУЩИМ (возможно, отредактированным) уравнениям,
                # поэтому любое изменение в GUI отражается на графиках.
                self.sol_method_combo.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, self._tr('examples_title'),
                                 f"{self._tr('load_error_msg')} {str(e)}")

    def _needs_mu_selection(self, dataset, explicit_type=None):
        effective = (explicit_type if (explicit_type and explicit_type != 'auto')
                     else detect_problem_type(dataset))
        if effective == 'lens':
            return True
        param = dataset.continuation_param
        if param != 'lambda':
            return True
        all_text = ' '.join(dataset.equations + dataset.boundary_conditions).lower()
        if re.search(r'\b' + re.escape(param) + r'\b', all_text):
            return True
        return False

    def _ask_mu_values(self, dataset=None, explicit_type=None):
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.resize(300, 200)
        layout = QVBoxLayout(dlg)

        effective = (explicit_type if (explicit_type and explicit_type != 'auto')
                     else detect_problem_type(dataset))
        if dataset and effective == 'lens':
            dlg.setWindowTitle(self._tr('mu_title'))
            layout.addWidget(QLabel(self._tr('mu_prompt')))
            items = self._tr('mu_items')
            labels_list = list(items.keys())
            checks = {}
            for idx, label in enumerate(labels_list):
                cb = QCheckBox(label)
                cb.setChecked(idx == len(labels_list) - 1)
                checks[label] = cb
                layout.addWidget(cb)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            layout.addWidget(btns)
            if not dlg.exec_():
                return None
            selected = [items[label] for label, cb in checks.items() if cb.isChecked()]
            if not selected:
                selected.append(list(items.values())[0])
            return selected
        else:
            param = dataset.continuation_param if dataset else 'lambda'
            dlg.setWindowTitle(str(param))
            layout.addWidget(QLabel(f"{self._tr('continuation_param_label')} {param}:"))
            from PyQt5.QtWidgets import QLineEdit
            param_edit = QLineEdit("0.0, 0.25, 0.5, 0.75, 1.0")
            layout.addWidget(param_edit)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            layout.addWidget(btns)
            if not dlg.exec_():
                return None
            text = param_edit.text().strip()
            try:
                return [float(x.strip()) for x in text.split(',') if x.strip()]
            except ValueError:
                return [0.0, 0.5, 1.0]

    def start_solve(self):
        try:
            dataset = self.get_dataset_from_ui()
            self.current_dataset = dataset

            # Read explicit solver type from the combo box
            idx = self.sol_method_combo.currentIndex()
            explicit_type = (_SOLVER_CODES[idx]
                             if 0 <= idx < len(_SOLVER_CODES) else 'auto')

            # Determine the problem type up front — it decides how the
            # initial-guess text is interpreted.
            prob_type = (explicit_type if explicit_type != 'auto'
                         else detect_problem_type(dataset))

            guess_text = self.guess_edit.text().strip()
            try:
                unknown_vals = ([float(x.strip()) for x in guess_text.split(',')
                                 if x.strip()] if guess_text else [])
            except ValueError:
                unknown_vals = []

            if prob_type == 'limit_cycle':
                # Для предельных циклов поле «Неизвестные» — это СПИСОК
                # амплитуд x1, например «2, 6.5, 9» (пример 26.2 задаёт три
                # точки p01, p02, p03).  Каждая амплитуда a порождает
                # приближение [x1=a, x2=0, T=2π, x4=a]; решатель сходится к
                # своему предельному циклу.  Флажок «Несколько циклов»
                # накладывает ВСЕ амплитуды на один график; без него берётся
                # только первая.
                amps = unknown_vals if unknown_vals else [0.5]
                if not self.multi_cycle_cb.isChecked():
                    amps = amps[:1]
                two_pi = 2 * np.pi
                combined = []
                for a in amps:
                    combined.extend([float(a), 0.0, two_pi, float(a)])
                initial_guess = combined
            else:
                # Generic builder: per component, use the spinbox value when
                # the "known" checkbox is ticked, otherwise pull the next
                # value from the comma-separated guess list.  Extra values
                # form additional complete groups (multi-solution solves).
                combined = []
                u_idx = 0
                for i in range(len(self.ic_known)):
                    if self.ic_known[i].isChecked():
                        combined.append(float(self.ic_value[i].value()))
                    elif u_idx < len(unknown_vals):
                        combined.append(unknown_vals[u_idx])
                        u_idx += 1
                    else:
                        combined.append(0.0)

                n_comp = len(self.ic_known)
                n_known_per = sum(1 for cb in self.ic_known if cb.isChecked())
                n_unk_per = n_comp - n_known_per
                while n_unk_per > 0 and u_idx + n_unk_per <= len(unknown_vals):
                    for i in range(n_comp):
                        if self.ic_known[i].isChecked():
                            combined.append(float(self.ic_value[i].value()))
                        else:
                            combined.append(unknown_vals[u_idx])
                            u_idx += 1
                while u_idx < len(unknown_vals):
                    combined.append(unknown_vals[u_idx])
                    u_idx += 1

                initial_guess = combined if combined else None

                # Предупреждение о недостатке приближений: несколько значений,
                # но меньше числа неизвестных → раньше молча дополнялось нулями
                # и график «разваливался».  Теперь спрашиваем подтверждение.
                n_unknowns = sum(1 for cb in self.ic_known if not cb.isChecked())
                if 0 < len(unknown_vals) < n_unknowns:
                    reply = QMessageBox.question(
                        self, self._tr('guess_warn_title'),
                        self._tr('guess_warn_msg').format(
                            n=len(unknown_vals), m=n_unknowns),
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.No:
                        self._set_status(self._tr('cancelled'), 'cancelled')
                        return

            mu_list = None
            if self._needs_mu_selection(dataset, explicit_type):
                mu_list = self._ask_mu_values(dataset, explicit_type)
                if mu_list is None:
                    self._set_status(self._tr('cancelled'), 'cancelled')
                    return

            self.solve_btn.setEnabled(False)
            self._set_status(self._tr('computing'), 'computing')

            self._last_problem_type = prob_type

            # Store guess label for plot window title
            if initial_guess:
                vals = [str(v) for v in initial_guess]
                self._pending_guess_label = f'  |  guess: ({", ".join(vals)})'
            else:
                self._pending_guess_label = ''

            self.solver_thread = SolverThread(dataset, initial_guess,
                                              smooth_param_list=mu_list,
                                              explicit_type=explicit_type,
                                              multi_cycle=self.multi_cycle_cb.isChecked())
            self.solver_thread.finished.connect(self.on_solve_finished)
            self.solver_thread.error.connect(self.on_solve_error)
            self.solver_thread.start()
        except Exception as e:
            QMessageBox.critical(self, self._tr('error'),
                                 f"{self._tr('prep_error')} {str(e)}")
            self.solve_btn.setEnabled(True)
            self._set_status(self._tr('error'), 'error')

    def on_solve_finished(self, x_eval, y_eval):
        self.solve_btn.setEnabled(True)
        self._set_status(self._tr('done'), 'done')
        self._last_x = x_eval
        self._last_y = y_eval
        self._last_name = self.current_dataset.name if self.current_dataset else ""
        self._last_guess_label = getattr(self, '_pending_guess_label', '')
        if self.current_dataset:
            name = self.current_dataset.name
            self._add_history(name)
            self._history_datasets[name] = self.current_dataset
            try:
                self._save_history()
            except Exception:
                pass

        # Automatically open one plot window per solution.
        # When the solver emits a list of layers (multiple BVP solutions,
        # limit cycles, or μ-values), every layer gets its OWN window so
        # each solution is shown on a single image.
        self._auto_show_plots()

    def on_solve_error(self, error_msg):
        self.solve_btn.setEnabled(True)
        self._set_status(self._tr('error'), 'error')
        QMessageBox.critical(self, self._tr('solver_error'), error_msg)

    def show_plot_window(self):
        if self._last_x is None:
            QMessageBox.information(self, self._tr('no_result'), self._tr('no_result_msg'))
            return
        self._auto_show_plots()

    def _close_plot_windows(self):
        if hasattr(self, '_plot_windows'):
            for w in self._plot_windows:
                try:
                    w.close()
                except Exception:
                    pass
        self._plot_windows = []

    def _auto_show_plots(self):
        """Open ONE PlotWindow with all solutions overlaid.

        From inside that window the user can click "Split solutions" to
        split into one-window-per-solution view.
        """
        if self._last_x is None:
            return
        self._close_plot_windows()
        vn = self._get_varnames()
        title = (self._last_name or "") + (self._last_guess_label or "")
        win = PlotWindow(self._last_x, self._last_y, vn,
                         title, self, self._last_problem_type)
        win.show()
        self._plot_windows.append(win)

    def _get_varnames(self):
        ny = 0
        if self._last_y is not None and getattr(self._last_y, 'ndim', 0) == 2:
            ny = self._last_y.shape[0]
        elif isinstance(self._last_x, list) and self._last_x:
            first = self._last_x[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                y = first[1]
                if y is not None and getattr(y, 'ndim', 0) == 2:
                    ny = y.shape[0]
        pt = self._last_problem_type
        if pt == 'triple':
            return ["x1", "x2", "x3", "p1", "p2", "p3", "u"][:ny]
        elif pt == 'lens':
            return ["x1", "x2", "psi1", "psi2", "T", "u1", "u2"][:ny]
        elif pt == 'kepler':
            return ["x1", "x2", "x3", "x4"][:ny]
        elif pt == 'limit_cycle':
            return ["x1", "x2", "T", "x3"][:ny]
        return ["x" + str(i + 1) for i in range(ny)]

    def _add_history(self, name):
        for i in range(self.history_list.count() - 1, -1, -1):
            if self.history_list.item(i).text() == name:
                self.history_list.takeItem(i)
        self.history_list.insertItem(0, name)
        if self.history_list.count() > 50:
            self.history_list.takeItem(self.history_list.count() - 1)

    def _save_history(self):
        items = [self.history_list.item(i).text()
                 for i in range(self.history_list.count())]
        import json, os
        path = os.path.join(os.path.dirname(__file__), 'history.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _load_history(self):
        import json, os
        path = os.path.join(os.path.dirname(__file__), 'history.json')
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                for name in items:
                    self.history_list.addItem(name)
            except Exception:
                pass

    def _on_history_clicked(self, item):
        name = item.text()
        ds = self._history_datasets.get(name)
        if ds is not None:
            self.set_dataset_to_ui(ds)
            self.current_dataset = ds
            return
        import os
        fname = (name.replace(' ', '_').replace(':', '')
                 .replace('(', '').replace(')', '')[:80])
        path = os.path.join(os.path.dirname(__file__), fname + '.json')
        if os.path.exists(path):
            try:
                from task_io import load_task
                dataset = load_task(path)
                self.set_dataset_to_ui(dataset)
                self.current_dataset = dataset
                return
            except Exception:
                pass
        QMessageBox.information(self, self._tr('no_result'),
                                self._tr('no_result_msg'))


def plot_results_static(ax, x_eval, y_eval, name):
    ax.clear()
    name_lower = name.lower()

    if isinstance(x_eval, list) and len(x_eval[0]) == 4:
        for i, (x, y, full_y, label) in enumerate(x_eval):
            ax.plot(full_y[0], full_y[1], color=f'C{i}', lw=1,
                    ls='--', alpha=0.4, label=f'{label} (full)')
            ax.plot(y[0], y[1], color=f'C{i}', lw=2, label=label)
            ax.plot(y[0, 0], y[1, 0], 'o', color=f'C{i}', ms=6)
        ax.plot(0, 0, 'ko', markersize=5, label=r'$\mathrm{origin}$')
        ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
        ax.set_aspect('equal'); ax.legend(prop={'size': 10})

        all_x = []; all_y = []
        for x, y, fy, _ in x_eval:
            all_x.extend(fy[0]); all_y.extend(fy[1])
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        m = 0.15
        ax.set_xlim(x_min-(x_max-x_min)*m, x_max+(x_max-x_min)*m)
        ax.set_ylim(y_min-(y_max-y_min)*m, y_max+(y_max-y_min)*m)

    elif isinstance(x_eval, list) and len(x_eval[0]) == 3:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (x, y, label) in enumerate(x_eval):
            T = y[2, -1]
            ax.plot(y[0], y[1], color=colors[i % len(colors)], linewidth=2,
                    label=f'{label}, T={T:.4f}')
            ax.plot(y[0, 0], y[1, 0], 'o', color=colors[i % len(colors)], markersize=8)
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        ax.set_aspect('equal'); ax.legend(prop={'size': 10})

        all_x = []; all_y = []
        for x, y, _ in x_eval:
            all_x.extend(y[0]); all_y.extend(y[1])
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        m = 0.15
        ax.set_xlim(x_min-(x_max-x_min)*m, x_max+(x_max-x_min)*m)
        ax.set_ylim(y_min-(y_max-y_min)*m, y_max+(y_max-y_min)*m)
