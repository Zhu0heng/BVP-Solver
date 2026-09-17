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
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QGroupBox, QCheckBox,
    QGridLayout, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QInputDialog, QComboBox, QScrollArea, QMenu, QListWidget, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QValidator
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
from ui_theme import STYLESHEET, SURFACE, TEXT, MUTED, BORDER


# Generic line-style policy: color distinguishes solutions/layers, while
# line style distinguishes variables or parameter layers.  This keeps plots
# readable when users change equations, variable names, or parameter values.
VARIABLE_LINE_STYLES = ('-', '--', '-.', ':')
LAYER_LINE_STYLES = ('--', ':', '-', '-.')


def _curve_linestyle(variable_index=0, layer_index=0,
                     variable_count=1, layer_count=1,
                     has_reference_curve=False, overlaps_existing=False):
    """Use dashes only to separate a curve that visually overlaps another."""
    return '--' if overlaps_existing else '-'


def _semantic_variable_linestyle(name, selected_names, fallback_index):
    """Variables no longer receive a dash merely because of their meaning."""
    return '-'


def _curve_overlap_mask(x, y, previous_curves):
    """Mark sustained shared segments while ignoring isolated crossings.

    Coordinates are normalised per pair, then each point of the new curve is
    compared with the nearest point of an existing curve.  At least four
    consecutive points (or 8% of the sampled curve) must be close, so curves
    that merely intersect once remain solid.
    """
    x_values = np.asarray(x, dtype=float).ravel()
    y_values = np.asarray(y, dtype=float).ravel()
    current = np.column_stack((x_values, y_values))
    finite = np.isfinite(current).all(axis=1)
    result = np.zeros(len(current), dtype=bool)
    if np.count_nonzero(finite) < 4:
        return result
    for old_x, old_y in previous_curves:
        old = np.column_stack((np.asarray(old_x, dtype=float).ravel(),
                               np.asarray(old_y, dtype=float).ravel()))
        old = old[np.isfinite(old).all(axis=1)]
        if len(old) < 4:
            continue
        joined = np.vstack((current[finite], old))
        scale = np.ptp(joined, axis=0)
        scale[scale < 1e-12] = 1.0
        tree = cKDTree(old / scale)
        nearest, _ = tree.query(current[finite] / scale, k=1)
        close_finite = nearest <= 3e-3
        close = np.zeros(len(current), dtype=bool)
        close[np.flatnonzero(finite)] = close_finite
        required = max(4, int(np.ceil(0.08 * len(close))))
        start = None
        for index, value in enumerate(np.r_[close, False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                if index - start >= required:
                    result[start:index] = True
                start = None
    return result


def _curves_visually_overlap(x, y, previous_curves):
    """Compatibility predicate for callers that only need yes/no."""
    return bool(np.any(_curve_overlap_mask(x, y, previous_curves)))

# Parallel to TR['sol_method_items'] — language-independent solver codes.
# Index 0 = auto-detect (legacy behaviour), 1..5 = explicit choices.
_SOLVER_CODES = ['auto', 'custom', 'kepler', 'limit_cycle', 'triple', 'lens']


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """Display tolerances without hiding their significant digits."""
    def textFromValue(self, value):
        return f'{value:.3g}'

    def valueFromText(self, text):
        return float(text)

    def validate(self, text, pos):
        try:
            value = float(text)
        except ValueError:
            state = QValidator.Intermediate if re.fullmatch(r'[+\-\d.eE]*',text) else QValidator.Invalid
            return state, text, pos
        return (QValidator.Acceptable if np.isfinite(value) and self.minimum() <= value <= self.maximum()
                else QValidator.Intermediate), text, pos


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
        zero_bc = lambda side: any(re.fullmatch(
            rf'\s*x2\({side}\)\s*=\s*[+-]?0+(?:\.0*)?(?:[eE][+-]?\d+)?\s*', bc)
            for bc in ds.boundary_conditions)
        if ('sin' in eqs and zero_bc('a') and zero_bc('b')):
            return 'limit_cycle'

    if n_eq == 6 and 'sqrt' in eqs:
        return 'triple'

    if n_eq == 2 and re.search(r'\bu1\b|\bu2\b', eqs):
        return 'lens'

    return 'custom'


class SolverThread(QThread):
    # Do not shadow QThread.finished(): Qt emits that zero-argument signal when
    # the native thread exits.  Overriding it with a two-argument signal can
    # crash the Windows Qt event dispatcher instead of raising a Python error.
    result_ready = pyqtSignal(object, object)
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
        # First follow the user's seed with the textbook continuation method.
        # That path is the branch-selection mechanism, so scanning ten more
        # seeds after it has already produced a valid orbit is both wasteful
        # and semantically wrong.  A short fallback list is used only when the
        # requested path genuinely fails.
        layers = []
        seen = []  # подписи v0 для дедупликации совпавших орбит
        for (gx10, gx20, vx, vy) in guesses:
            v_scale = max(0.3, ref / 4.0)
            seeds = [(vx, vy), (0.0, vy), (vy, -vx),
                     (-vx, -vy), (0.0, v_scale), (v_scale, -v_scale)]
            selected = None
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
                selected = (sig, cand_x, cand_y)
                break
            if selected is None:
                raise RuntimeError(f'No valid Kepler trajectory for initial velocity ({vx:g}, {vy:g}).')
            sig, xs, ys = selected
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
        self.result_ready.emit(layers, None)

    def _solve_limit_cycles(self):
        """Solve for ONE limit cycle using the user's guess values directly.

        The user provides initial guesses for the unknown components:
          • x1(0) — starting amplitude
          • T    — period estimate (default 2π)
          • x4(0) — phase anchor (default = x1(0))

        These are used as-is (no scanning).  Each guess selects which
        cycle to converge to.  The plot window shows exactly ONE layer
        per solve call.

        Each requested guess must yield a nontrivial validated cycle;
        failures are reported rather than substituted with an unrelated scan.
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
            except Exception as exc:
                raise RuntimeError(f'Limit cycle guess {gi+1} failed: {exc}') from exc
            if len(all_solutions) != gi + 1:
                raise RuntimeError(f'Guess {gi+1} converged to a trivial or invalid cycle.')

        if all_solutions:
            self.result_ready.emit(all_solutions, None)
            return

        raise RuntimeError('No nontrivial limit cycle found for the supplied guesses.')

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

        from control_problems import fixed_boundary_values
        a_fixed, b_fixed = fixed_boundary_values(self.dataset, (1, 2, 3))
        a_vals = dict(enumerate(a_fixed, 1))
        b_vals = dict(enumerate(b_fixed, 1))
        param_val = float((self.dataset.parameters or {}).get(
            self.dataset.continuation_param, self.dataset.continuation_end))

        def user_ode(t, y):
            return solver._ode_system(t, y, param_val)

        # Fast necessary reachability check for the classic rest-to-rest
        # triple integrator.  With x1'=c1*x2, x2'=c2*x3 and |x3'|<=J, the
        # largest displacement over a fixed duration is
        # |c1*c2|*J*T^3/32.  Rejecting only beyond this upper bound avoids a
        # minutes-long multistart search for a mathematically impossible BVP.
        try:
            zero = np.zeros(n)
            base = np.asarray(user_ode(t0, zero), dtype=float)
            x2_one = zero.copy(); x2_one[1] = 1.0
            x2_two = zero.copy(); x2_two[1] = 2.0
            x3_one = zero.copy(); x3_one[2] = 1.0
            x3_two = zero.copy(); x3_two[2] = 2.0
            c1 = float(user_ode(t0, x2_one)[0] - base[0])
            c2 = float(user_ode(t0, x3_one)[1] - base[1])
            linear_chain = (
                abs(base[0]) < 1e-10 and abs(base[1]) < 1e-10
                and abs(user_ode(t0, x2_two)[0] - base[0] - 2*c1) < 1e-8
                and abs(user_ode(t0, x3_two)[1] - base[1] - 2*c2) < 1e-8
            )
            plus = zero.copy(); plus[5] = 1e6
            minus = zero.copy(); minus[5] = -1e6
            jerk_bound = max(abs(float(user_ode(t0, plus)[2])),
                             abs(float(user_ode(t0, minus)[2])))
            rest_to_rest = (max(abs(a_fixed[1]), abs(a_fixed[2]),
                                abs(b_fixed[1]), abs(b_fixed[2])) < 1e-10)
            if linear_chain and rest_to_rest and np.isfinite(jerk_bound):
                duration = T - t0
                displacement_bound = abs(c1*c2) * jerk_bound * duration**3 / 32.0
                requested = abs(float(b_fixed[0] - a_fixed[0]))
                if requested > displacement_bound * (1 + 1e-7) + 1e-9:
                    raise RuntimeError(
                        'Triple-integrator boundary is unreachable for the '
                        f'fixed interval: required displacement {requested:.6g} '
                        f'exceeds the theoretical bound {displacement_bound:.6g}.')
        except RuntimeError:
            raise
        except Exception:
            # If the edited equations are not the classic linear chain, skip
            # this specialised necessary check and use the general solver.
            pass

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
                if not sol.success:
                    return np.full(3, 1e10)
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
        fallback_starts = []
        user_start = None
        if self.initial_guess is not None and len(self.initial_guess) >= 6:
            user_start = np.asarray(self.initial_guess[3:6], dtype=float)
        neutral_start = np.full(3, 0.5, dtype=float)

        def build_fallback_starts():
            """Create expensive backward seeds lazily, only when needed."""
            starts = []
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
                                user_ode, [T, t0], y_T,
                                dense_output=True, rtol=1e-9, atol=1e-12,
                            )
                            if sol_bwd.success:
                                starts.append(sol_bwd.sol(t0)[3:6])
                        except Exception:
                            continue
            rng = np.random.default_rng(42)
            for _ in range(4):
                starts.append(rng.uniform(-10.0, 10.0, size=3))
            return starts

        # ── Tier 1: a bounded quick attempt from the edited user guess ──
        best_x, best_r = None, np.inf
        component_tol = max(1e-7, 100 * solver.tol)
        # best_r is an L2 norm of three endpoint residuals; compare it with
        # sqrt(3) times the per-component tolerance used elsewhere.
        accept_tol = np.sqrt(3.0) * component_tol
        if user_start is not None and not np.allclose(user_start, neutral_start):
            try:
                res = _root(shoot, user_start, method='lm',
                            options={'ftol': 1e-12, 'xtol': 1e-12,
                                     'maxiter': 30, 'factor': 0.1})
                r = float(np.linalg.norm(shoot(res.x)))
                if np.isfinite(r):
                    best_x, best_r = res.x.copy(), r
            except Exception:
                pass

        # ── Tier 2: textbook continuation from a neutral seed ──
        # This solves the common/convex case much faster than hundreds of LM
        # integrations and still uses the actual edited equations and BCs.
        if best_r > accept_tol:
            neutral_full = np.array([
                a_vals.get(1, 1.0), a_vals.get(2, 0.0), a_vals.get(3, 0.0),
                *neutral_start,
            ], dtype=float)
            try:
                _, neutral_y = solver.solve(neutral_full)
                candidate = neutral_y[3:6, 0]
                r = float(np.linalg.norm(shoot(candidate)))
                if r < best_r:
                    best_x, best_r = candidate.copy(), r
            except Exception:
                pass

        # ── Tier 3: bounded fallback for materially changed equations ──
        if best_r > accept_tol:
            fallback_starts = build_fallback_starts()
        for st in fallback_starts:
            if best_r <= accept_tol:
                break
            cur = np.asarray(st, dtype=float)
            for _round in range(2):
                try:
                    res = _root(shoot, cur, method='lm',
                                options={'ftol': 1e-14, 'xtol': 1e-14,
                                         'maxiter': 150,
                                         'factor': 0.1})
                except Exception:
                    break
                r = float(np.linalg.norm(shoot(res.x)))
                cur = res.x
                if r < best_r:
                    best_x, best_r = res.x.copy(), r
                if r <= accept_tol:
                    break
            if best_r <= accept_tol:
                break

        if best_x is None or best_r > accept_tol:
            raise RuntimeError(f"Triple integrator did not satisfy boundary conditions (residual={best_r:.3g})")

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
        if not sol_final.success:
            raise RuntimeError(sol_final.message)
        y_eval = sol_final.sol(x_eval)
        if (not np.isfinite(y_eval).all() or
                np.max(np.abs(y_eval[:3, -1] - b_fixed)) > max(1e-6, 100 * _tol)):
            raise RuntimeError('Final trajectory does not satisfy the boundary conditions.')

        # Control u(t) = RHS уравнения dx3/dt (user-defined)
        try:
            u_t = np.array([solver.ode_funcs[2](x_eval[j], *y_eval[:, j], param_val)
                            for j in range(len(x_eval))])
            y_out = np.vstack([y_eval, u_t.reshape(1, -1)])
        except Exception:
            y_out = y_eval

        self.result_ready.emit(x_eval, y_out)

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
        from control_problems import LensDynamics, fixed_boundary_values

        if self.dataset.x_start != 0 or self.dataset.x_end != 1:
            raise ValueError('Lens: use dimensionless interval [0, 1]; terminal time T is solved, not prescribed.')
        (a1, a2), (b1, b2) = fixed_boundary_values(self.dataset, (1, 2))
        dynamics = LensDynamics(self.dataset)


        # ── 5. Список μ ──
        mu_values = self.smooth_param_list
        if mu_values is None:
            params = self.dataset.parameters
            if params and 'mu_values' in params:
                mu_values = params['mu_values']
            else:
                mu_values = [1.0, 1e-1, 1e-6]

        if not mu_values or any(not np.isfinite(mu) or mu <= 0 for mu in mu_values):
            raise ValueError('Smoothing parameters must be finite and positive.')
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
            return lambda tau, y: dynamics.derivative(y, mu_val)


        def make_shoot(ode_fn):
            def shoot_3(v):
                p10, p20, Tv = v
                if Tv <= 0:                      # T must be positive
                    return np.full(3, 1e6)
                try:
                    sol = solve_ivp(ode_fn, [0., 1.], [a1, a2, p10, p20, Tv],
                                    method='DOP853', rtol=1e-11, atol=1e-13, max_step=0.005)
                    if not sol.success:
                        return np.full(3, 1e10)
                    ye = sol.y[:, -1]
                except Exception:
                    return np.full(3, 1e10)
                # x1(T)=b1, x2(T)=b2 (целевое состояние из BC правого края),
                # |ψ(T)|²=1 — условие трансверсальности (свободное время T).
                return np.array([ye[0] - b1, ye[1] - b2,
                                 ye[2]**2 + ye[3]**2 - 1.0])
            return shoot_3

        def try_root(ode_fn, start, maxiter=160):
            shoot_3 = make_shoot(ode_fn)
            try:
                res = root(shoot_3, start, method='lm',
                           options={'ftol': 1e-11, 'xtol': 1e-11, 'maxiter': maxiter,
                                    'eps': 1e-8})
                if not res.success:
                    return None, np.inf
                if res.x[2] <= 0:                # reject non-physical T<0
                    return None, np.inf
                rnorm = float(np.linalg.norm(shoot_3(res.x)))
                if not np.isfinite(rnorm) or rnorm > max(1e-7, 100*self.dataset.tol):
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
            sol_x, rnorm = try_root(ode_main, [psi10, psi20, T_opt], maxiter=120)

            # Multistart fallback if main attempt diverged.
            # Trials are generated adaptively from a problem-scale base, so
            # changing BCs or equations produces a different (scaled) search
            # space — no hardcoded numerical constants.
            if sol_x is None:
                base_T = bc_scale * 2.0
                trials = []
                for sign_psi in ((-1.0, -1.0), (-1.0, 0.5), (-0.5, -1.0),
                                 (0.5, -1.0), (-1.0, 1.0)):
                    for k in (0.5, 1.0, 1.5):
                        trials.append([sign_psi[0] * 0.5, sign_psi[1] * 0.2,
                                       base_T * k])
                for tr in trials:
                    cand, r = try_root(ode_main, tr, maxiter=300)
                    if cand is not None:
                        # Every accepted candidate already satisfies all three
                        # endpoint conditions.  Minimising residual below the
                        # tolerance does not identify a more time-optimal
                        # branch, so continuing all 15 trials only adds delay.
                        sol_x, rnorm = cand, r
                        break

            if sol_x is not None:
                psi10, psi20, T_opt = sol_x
                solutions_cache[mu] = (psi10, psi20, T_opt)
            else:
                raise RuntimeError(f'Lens solve failed at mu={mu:g}; boundary residual={rnorm:.3g}')

        # Now emit only the originally requested mu_values
        for mu in mu_values:
            psi10, psi20, T_opt = solutions_cache[mu]
            ode_main = make_ode(mu)
            _tol = getattr(self.dataset, 'tol', 1e-9)
            _method = getattr(self.dataset, 'method', 'RK45')
            sol = solve_ivp(ode_main, [0., 1.], [a1, a2, psi10, psi20, T_opt],
                            dense_output=True, method=_method,
                            rtol=_tol, atol=_tol * 1e-3, max_step=0.002)
            if not sol.success:
                raise RuntimeError(sol.message)
            t_dimless = np.linspace(0., 1., self.dataset.n_points)
            y_dimless = sol.sol(t_dimless)
            residual = np.r_[y_dimless[:2,-1]-[b1,b2],
                             np.sum(y_dimless[2:4,-1]**2)-1]
            if not np.isfinite(y_dimless).all() or np.max(np.abs(residual)) > max(1e-6, 100*_tol):
                raise RuntimeError(f'Lens trajectory failed endpoint validation at mu={mu:g}')

            u_vals = np.zeros((2, len(t_dimless)))
            for i in range(len(t_dimless)):
                u_vals[:, i] = dynamics.control(y_dimless[:2, i], y_dimless[2:4, i], mu)

            y_out = np.vstack([
                y_dimless[0], y_dimless[1],
                y_dimless[2], y_dimless[3],
                np.full(len(t_dimless), T_opt),
                u_vals[0], u_vals[1],
            ])

            # Convert dimensionless τ ∈ [0,1] to real time t = τ·T_opt
            t_real = t_dimless * T_opt
            layers.append((t_real, y_out, f'μ={mu:g} · T={T_opt:.6f}'))

        if len(layers) == 1:
            self.result_ready.emit(layers[0][0], layers[0][1])
        else:
            self.result_ready.emit(layers, None)

    def run(self):
        try:
            if self.explicit_type and self.explicit_type != 'auto':
                problem_type = self.explicit_type
            else:
                problem_type = detect_problem_type(self.dataset)
            if problem_type in ('kepler', 'limit_cycle', 'triple') and self.smooth_param_list is not None:
                self._solve_parameter_values(problem_type)
            elif problem_type == 'kepler':
                self._solve_kepler()
            elif problem_type == 'limit_cycle':
                self._solve_limit_cycles()
            elif problem_type == 'triple':
                self._solve_triple()
            elif problem_type == 'lens':
                self._solve_lens_control()
            else:
                solver = ContinuationSolver(self.dataset)
                if self.smooth_param_list:
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
                        self.result_ready.emit(layers, None)
                    else:
                        raise RuntimeError("No parameter values converged")
                else:
                    x_eval, y_eval = solver.solve(self.initial_guess)
                    self.result_ready.emit(x_eval, y_eval)
        except Exception as e:
            self.error.emit(str(e))

    def _solve_parameter_values(self, problem_type):
        """Apply every requested parameter value also to specialised solvers."""
        from dataclasses import replace
        values = self.smooth_param_list
        if not values or not np.isfinite(values).all():
            raise ValueError('Provide finite continuation parameter values.')
        layers = []
        for value in values:
            ds = replace(self.dataset, parameters={**(self.dataset.parameters or {}),
                         self.dataset.continuation_param: float(value)})
            child = SolverThread(ds, self.initial_guess, explicit_type=problem_type,
                                 multi_cycle=self.multi_cycle)
            result = {}
            child.result_ready.connect(lambda t,y: result.update(t=t,y=y))
            child.error.connect(lambda error: result.update(error=error))
            child.run()
            if 'error' in result:
                raise RuntimeError(result['error'])
            entries = result['t'] if isinstance(result['t'],list) else [(result['t'],result['y'],None,'')]
            for entry in entries:
                full = entry[2] if len(entry)==4 else None
                label = entry[-1] if isinstance(entry[-1],str) else ''
                layers.append((entry[0],entry[1],full,
                               f'{self.dataset.continuation_param}={value:g} {label}'.strip()))
        self.result_ready.emit(layers,None)


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=7.5, dpi=120):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=SURFACE)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(SURFACE)
        self.axes.tick_params(colors=MUTED, which='both')
        for spine in self.axes.spines.values():
            spine.set_color(BORDER)
        super().__init__(self.fig)
        self.setParent(parent)


class PlotWindow(QMainWindow):
    def __init__(self, t_data, y_data, varnames, title, parent=None, problem_type='custom',
                 color_offset=0, layer_style_index=None, layer_style_count=None,
                 initial_x_name=None, initial_y_names=None):
        super().__init__(parent)
        self.setStyleSheet(STYLESHEET)
        self.setWindowTitle(title)
        self.setGeometry(150, 150, 1100, 800)
        self.dataset_title = title
        self.varnames = varnames
        self.problem_type = problem_type
        # Сдвиг индекса палитры: при «разделении» окон каждый слой сохраняет
        # тот же цвет, что и в совмещённом графике (см. _toggle_split_view).
        self._color_offset = color_offset
        self._layer_style_index = layer_style_index
        self._layer_style_count = layer_style_count
        self._initial_x_name = initial_x_name
        self._initial_y_names = initial_y_names

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

        if initial_x_name in self._x_items:
            self._x_idx = self._x_items.index(initial_x_name)
            self.btn_x.setText(initial_x_name)
        if initial_y_names:
            selected = [self._y_items.index(name) for name in initial_y_names
                        if name in self._y_items]
            if selected:
                self._y_indices = selected
                self.btn_y.setText(", ".join(self._y_items[i] for i in selected))
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
        self._dimensionless_x_index = None
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
        if self.problem_type == 'lens':
            self._dimensionless_x_index = len(self._x_items)
            self._x_items.append('τ')

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
        elif self.problem_type == 'lens' and self._dimensionless_x_index is not None:
            self._x_idx = self._dimensionless_x_index
            self._y_indices = [0]
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
        selected_x = self._x_items[self._x_idx]
        selected_y = [self._y_items[index] for index in self._y_indices]
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
                           color_offset=i, layer_style_index=i,
                           layer_style_count=len(layers),
                           initial_x_name=selected_x,
                           initial_y_names=selected_y)
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
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        xi = self._x_idx
        is_dimensionless = (self._dimensionless_x_index is not None and
                            xi == self._dimensionless_x_index)

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
        drawn_curves = []
        x_label = 'τ=t/T' if is_dimensionless else ("t" if xi == 0 else self._x_items[xi])
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
                if is_dimensionless:
                    span = float(t_dat[-1] - t_dat[0])
                    t_axis = ((t_dat - t_dat[0]) / span if span else
                              np.zeros_like(t_dat, dtype=float))
                else:
                    t_axis = t_plot

                if len(self._y_indices) == 1:
                    c = colors[(li + off) % len(colors)]
                else:
                    c = colors[(yi_pos * len(layers) + li + off) % len(colors)]

                if has_full and xi > 0 and self._has_layers and len(self._y_indices) == 1:
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

                xx = t_axis if (xi == 0 or is_dimensionless) else y_dat[x_row]
                yy = t_plot if y_row < 0 else y_dat[y_row]

                all_x.extend(xx); all_y.extend(yy)

                # All computed curves are solid unless a sustained segment
                # overlaps a curve already drawn in this same axes.  A single
                # crossing is not an overlap.  Reference/full orbits remain a
                # separate dashed background convention.
                # The textbook examples 26.3 (triple) and 26.4 (lens) use
                # solid computed curves throughout.  Do not split their
                # control/costate or μ layers into dashed overlap segments.
                if self.problem_type in ('triple', 'lens'):
                    overlap_mask = np.zeros(np.asarray(yy).shape, dtype=bool)
                else:
                    overlap_mask = _curve_overlap_mask(xx, yy, drawn_curves)
                curve_label = (label if (is_phase or len(self._y_indices) == 1)
                               else f'{label}: {y_name}')
                solid_mask = ~overlap_mask
                label_used = False
                if np.any(solid_mask):
                    solid_y = np.where(solid_mask, yy, np.nan)
                    ax.plot(xx, solid_y, color=c, lw=2, ls='-',
                            label=curve_label)
                    label_used = True
                if np.any(overlap_mask):
                    dashed_y = np.where(overlap_mask, yy, np.nan)
                    ax.plot(xx, dashed_y, color=c, lw=2, ls='--',
                            label='_nolegend_' if label_used else curve_label)
                if is_phase:
                    ax.plot(xx[0], yy[0], 'o', color=c, ms=6)
                drawn_curves.append((np.asarray(xx), np.asarray(yy)))

        if is_phase and xi == 1 and any(self._y_row.get(yi, -1) == 1 for yi in self._y_indices):
            origin_lbl = 'центр (0, 0)' if self.problem_type == 'kepler' else 'начало (0, 0)'
            ax.plot(0, 0, 'o', color='#7AA2F7', markersize=6, label=origin_lbl)
            ax.annotate('(0, 0)', (0, 0), textcoords='offset points',
                        xytext=(6, 6), color='#7AA2F7', fontsize=9)

        if len(self._y_indices) > 1:
            y_label = ", ".join(self._y_items[i] for i in self._y_indices)

        if is_phase:
            ax.set_xlabel(x_label, color=MUTED); ax.set_ylabel(y_label, color=MUTED)
            ax.set_aspect('equal')
        else:
            ax.set_xlabel(x_label, color=MUTED); ax.set_ylabel(y_label, color=MUTED)
        if ax.get_legend_handles_labels()[0]:
            leg = ax.legend(prop={'size': 9}, facecolor=SURFACE,
                           edgecolor=BORDER, labelcolor=TEXT)
            leg.get_frame().set_alpha(0.92)

        ax.axhline(y=0, color=BORDER, linewidth=0.8, zorder=0)
        ax.axvline(x=0, color=BORDER, linewidth=0.8, zorder=0)
        ax.grid(True, alpha=0.55, linewidth=0.5, color=BORDER)

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
            'advanced': 'Solver settings',
            'changed': 'Inputs changed — solve again',
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
            'about_author_text': 'Автор: Ли Чжохэн\n313,\nМосковский государственный университет им. Ломоносова,\nФакультет вычислительной математики и кибернетики,\nКафедра оптимального управления,\nЭлектронная почта: stbc02220010@gse.cs.msu.ru\n\nРуководитель: Аввакумов Сергей Николаевич,\nМосковский государственный университет им. Ломоносова,\nКафедра оптимального управления,\nСтарший преподаватель\n\nРуководитель: Орлов Сергей Михайлович,\nДоцент кафедры ОУ,\nначальник курса\n\nГод создания: 2026',
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
            'advanced': 'Настройки решателя',
            'changed': 'Данные изменены — решите задачу снова',
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
            'about_author_text': 'Автор: Ли Чжохэн\n313,\nМосковский государственный университет им. Ломоносова,\nФакультет вычислительной математики и кибернетики,\nКафедра оптимального управления,\nЭлектронная почта: stbc02220010@gse.cs.msu.ru\n\nРуководитель: Аввакумов Сергей Николаевич,\nМосковский государственный университет им. Ломоносова,\nКафедра оптимального управления,\nСтарший преподаватель\n\nРуководитель: Орлов Сергей Михайлович,\nДоцент кафедры ОУ,\nначальник курса\n\nГод создания: 2026',
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
        self.setGeometry(100, 100, 1180, 820)
        self._input_revision = 0
        self._pending_revision = None
        self._solving = False

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
        self._connect_input_changes()
        self.set_dataset_to_ui(get_example_tasks()[0])
        self._set_status(self._tr('ready'), 'ready')
        self.graph_btn.setEnabled(False)

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
        self.header_subtitle.setText('Краевые задачи · SymPy + SciPy' if self.lang == 'ru' else 'Boundary-value problems · SymPy + SciPy')
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
        self.adv_toggle.setText(('▲ ' if self.adv_toggle.isChecked() else '▼ ') + self._tr('advanced'))
        self.guess_hint.setText(
            'Граничные условия задают задачу. Начальное приближение может сходиться к тому же решению.'
            if self.lang == 'ru' else
            'Boundary conditions define the problem. Different guesses can converge to the same solution.')
        self.control_group.setTitle(self._tr('controls'))
        self.solve_btn.setText(self._tr('solve'))
        self.graph_btn.setText(self._tr('graph'))
        self.export_btn.setText(self._tr('export'))
        self.clear_btn.setText(self._tr('clear'))
        self.load_btn.setText(self._tr('load'))
        self.example_btn.setText(self._tr('examples'))
        method_index = max(0, self.sol_method_combo.currentIndex())
        self.sol_method_combo.blockSignals(True)
        self.sol_method_combo.clear()
        self.sol_method_combo.addItems(self._tr('sol_method_items'))
        self.sol_method_combo.setCurrentIndex(method_index)
        self.sol_method_combo.blockSignals(False)
        for i, ed in enumerate(self.equation_edits):
            ed.setPlaceholderText(self._tr('eq_placeholder'))
        state = getattr(self, '_status_state', 'ready')
        self._set_status(self._tr(state), state)
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
        direct = re.match(
            r'^\s*Derivative\(x(\d+)\((\w+)\),\s*(\w+)\)\s*=\s*(.*?)\s*$',
            eq_str)
        if direct:
            idx = int(direct.group(1)) - 1
            rhs = direct.group(4).strip()
            mapping = {f'x{i+1}': vn for i, vn in enumerate(varnames)}
            rhs = re.sub(r'\b\w+\b', lambda match: mapping.get(match[0], match[0]), rhs)
            return f'd/dt[{idx}]: {rhs}'
        m = re.match(r'Derivative\(x(\d+)\((\w+)\),\s*(\w+)\)\s*([+-])\s*(.*?)\s*=\s*0$', eq_str)
        if m:
            idx = int(m.group(1)) - 1
            op = m.group(4)
            rhs = m.group(5).strip()
            if op == '+':
                rhs = '-(' + rhs + ')'
            mapping = {f'x{i+1}': vn for i, vn in enumerate(varnames)}
            rhs = re.sub(r'\b\w+\b', lambda m: mapping.get(m[0], m[0]), rhs)
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
        mapping = {vn: f'x{i+1}' for i, vn in enumerate(varnames) if vn}
        rhs_x = re.sub(r'\b\w+\b', lambda m: mapping.get(m[0], m[0]), rhs)
        return f'Derivative(x{idx+1}(t), t) - ({rhs_x}) = 0'

    def apply_style(self):
        self.setStyleSheet(STYLESHEET)

    def _create_header(self):
        """Compact navigation bar with title, context and utility menus."""
        frame = QFrame()
        frame.setObjectName("headerFrame")
        frame.setFixedHeight(70)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(24, 0, 18, 0)
        layout.setSpacing(10)

        title_box = QWidget()
        title_layout = QVBoxLayout(title_box)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)
        self.header_title = QLabel()
        self.header_title.setObjectName("headerTitle")
        title_layout.addWidget(self.header_title)
        self.header_subtitle = QLabel('BVP solver · SymPy + SciPy')
        self.header_subtitle.setObjectName('headerSubtitle')
        title_layout.addWidget(self.header_subtitle)
        layout.addWidget(title_box)
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
        """Small text status; reserve strong emphasis for the solve button."""
        self._status_state = state
        color = {'computing': '#3865D9', 'done': '#24744B',
                 'error': '#B34343', 'changed': '#98651F'}.get(state, MUTED)
        self.status_label.setStyleSheet(f'color:{color};padding:4px;background:transparent;')
        self.status_label.setText(text)

    def _connect_input_changes(self):
        """Connect current editors, including controls created after dimension changes."""
        for cls, signal in ((QLineEdit, 'textChanged'), (QTextEdit, 'textChanged'),
                            (QSpinBox, 'valueChanged'), (QDoubleSpinBox, 'valueChanged'),
                            (QComboBox, 'currentIndexChanged'), (QCheckBox, 'toggled')):
            for widget in self.centralWidget().findChildren(cls):
                if not widget.property('invalidatesResult'):
                    getattr(widget, signal).connect(self._on_input_changed)
                    widget.setProperty('invalidatesResult', True)

    def _clear_result(self):
        self._last_x = self._last_y = None
        self.graph_btn.setEnabled(False)
        self._close_plot_windows()

    def _on_input_changed(self, *args):
        self._input_revision += 1
        self._clear_result()
        if not self._solving:
            self._set_status(self._tr('changed'), 'changed')

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
        body_layout.setContentsMargins(20, 14, 20, 18)
        body_layout.setSpacing(0)

        split = QHBoxLayout()
        split.setSpacing(18)
        split.addWidget(self.create_left_panel(), 3)
        split.addWidget(self.create_right_panel(), 2)
        body_layout.addLayout(split)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer_layout.addWidget(scroll)

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 2, 4, 2)
        layout.setSpacing(12)
        layout.addWidget(self.create_params_group())
        layout.addWidget(self.create_equation_group())
        layout.addStretch()
        return panel

    def _auto_fit_window(self):
        # The scroll area handles long systems without moving/resizing the window.
        self.updateGeometry()

    def create_right_panel(self):
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 2, 0, 2)
        column.setSpacing(12)
        column.addWidget(self.create_initial_guess_group())
        column.addWidget(self.create_solve_group())
        self.history_group = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.history_list.setAlternatingRowColors(True)
        self.history_list.setToolTip("Previously solved tasks — click to reload")
        layout.addWidget(self.history_list)
        self.history_group.setLayout(layout)
        self.history_list.setMaximumHeight(100)
        column.addWidget(self.history_group)
        column.addStretch()
        return panel

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
        self.x_start_spin.setDecimals(12)
        self.x_start_spin.setFixedWidth(75)
        self.x_start_spin.setToolTip("Start of integration interval")
        row1.addWidget(self.x_start_spin)
        self.lbl_b = QLabel()
        row1.addWidget(self.lbl_b)
        self.x_end_spin = QDoubleSpinBox()
        self.x_end_spin.setRange(-1000, 1000)
        self.x_end_spin.setValue(7.0)
        self.x_end_spin.setDecimals(12)
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
        self.eps_spin = ScientificDoubleSpinBox()
        self.eps_spin.setRange(1e-14, 1.0)
        self.eps_spin.setDecimals(14)
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
        self.sol_method_combo.currentIndexChanged.connect(
            self._update_multi_cycle_visibility)
        row3.addWidget(self.sol_method_combo)
        row3.addStretch()
        adv_layout.addLayout(row3)

        self.adv_widget.setVisible(False)
        layout.addWidget(self.adv_widget)

        self.params_group.setLayout(layout)
        return self.params_group

    def _on_advanced_toggled(self, checked):
        self.adv_widget.setVisible(checked)
        self.adv_toggle.setText(('▲ ' if checked else '▼ ') + self._tr('advanced'))
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
        self.bc_edit.setMinimumHeight(130)
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
        if hasattr(self, 'ic_grid'):
            self._rebuild_initial_values(new_dim)
            self._connect_input_changes()
            self._on_input_changed()
        self._update_multi_cycle_visibility()
        self._auto_fit_window()

    def create_initial_guess_group(self):
        self.init_group = QGroupBox()
        layout = QVBoxLayout()
        layout.setSpacing(2)

        self.ic_known = []
        self.ic_value = []
        self.ic_grid = QGridLayout()
        self.ic_grid.setSpacing(5)
        self._rebuild_initial_values(self.dim_spin.value())
        layout.addLayout(self.ic_grid)

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

        # Kept as an internal compatibility object for old code/tests, but it
        # is deliberately not added to the layout.  The number of requested
        # cycles is derived from the amplitude list, never from hidden state.
        self.multi_cycle_cb = QCheckBox("Multiple cycles", self.init_group)
        self.multi_cycle_cb.setToolTip(
            "Check to split the guess values into groups of unknowns "
            "→ each group yields a separate cycle overlaid on one plot. "
            "The option stays visible and is disabled when the current "
            "solver is not a limit-cycle solver.")
        self.multi_cycle_cb.setVisible(False)
        self.multi_cycle_cb.setEnabled(False)
        self.guess_hint = QLabel()
        self.guess_hint.setWordWrap(True)
        self.guess_hint.setStyleSheet(f'color:{MUTED};font-size:9pt;')
        layout.addWidget(self.guess_hint)

        self.init_group.setLayout(layout)
        return self.init_group

    def _rebuild_initial_values(self, n):
        previous = [(cb.isChecked(), sp.value()) for cb, sp in zip(self.ic_known, self.ic_value)]
        while self.ic_grid.count():
            widget = self.ic_grid.takeAt(0).widget()
            widget.setParent(None)
            widget.deleteLater()
        self.ic_known, self.ic_value = [], []
        for i in range(n):
            cb = QCheckBox(f'x{i+1}(a)')
            cb.setToolTip('Known initial component; other components use the guess field.')
            sp = QDoubleSpinBox()
            sp.setRange(-1e9, 1e9)
            sp.setDecimals(8)
            sp.setMinimumWidth(110)
            if i < len(previous):
                cb.setChecked(previous[i][0])
                sp.setValue(previous[i][1])
            sp.setEnabled(cb.isChecked())
            cb.toggled.connect(sp.setEnabled)
            self.ic_known.append(cb)
            self.ic_value.append(sp)
            self.ic_grid.addWidget(cb, i, 0)
            self.ic_grid.addWidget(sp, i, 1)

    def _update_multi_cycle_visibility(self):
        """Show the multi-cycle checkbox only for limit‑cycle‑shaped problems."""
        if not hasattr(self, 'multi_cycle_cb'):
            return
        if self.multi_cycle_cb is None:
            return
        try:
            problem_type = detect_problem_type(self.get_dataset_from_ui())
            if hasattr(self, 'sol_method_combo'):
                index = self.sol_method_combo.currentIndex()
                if 0 <= index < len(_SOLVER_CODES) and _SOLVER_CODES[index] != 'auto':
                    problem_type = _SOLVER_CODES[index]
            is_lc = problem_type == 'limit_cycle'
        except (ValueError, AttributeError):
            problem_type = 'custom'
            is_lc = False
        # Cycle multiplicity is inferred from the amplitude list.  Never let a
        # hidden checkbox influence computation.
        self.multi_cycle_cb.setChecked(False)
        self.multi_cycle_cb.setEnabled(False)
        self.multi_cycle_cb.setVisible(False)
        is_lens = problem_type == 'lens'
        self.x_start_spin.setEnabled(not is_lens)
        self.x_end_spin.setEnabled(not is_lens)
        if is_lens:
            self.x_end_spin.setToolTip('Dimensionless interval [0,1]. Terminal time T is an unknown solved by the algorithm.')

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
        guess_text = self.guess_edit.text().strip()
        initial_guess = None
        if guess_text:
            try:
                initial_guess = [float(value.strip()) for value in guess_text.split(',')
                                 if value.strip()]
            except ValueError as exc:
                raise ValueError('Initial guesses must be comma-separated numbers.') from exc
            if not np.isfinite(initial_guess).all():
                raise ValueError('Initial guesses must be finite.')
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
            initial_guess=initial_guess,
        )

    def set_dataset_to_ui(self, dataset):
        self.current_dataset = dataset
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
        elif prob_type == 'lens':
            self.guess_edit.setText('-0.5, -0.1, 4.0')
            self.guess_edit.setToolTip('Initial guesses: psi1(0), psi2(0), T. T is solved, not fixed.')
        else:
            n_unknowns = max(0, len(dataset.equations) - len(a_side))
            if n_unknowns > 0:
                defaults = ['0.5'] * n_unknowns
                self.guess_edit.setText(', '.join(defaults))
            else:
                # Never retain guesses from the previously opened example.
                # With every initial component fixed, stale values would be
                # appended to the state vector and make its dimension invalid.
                self.guess_edit.clear()
        if dataset.initial_guess is not None:
            self.guess_edit.setText(', '.join(f'{value:g}' for value in dataset.initial_guess))
        self._update_multi_cycle_visibility()
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
                values = [float(x.strip()) for x in text.split(',') if x.strip()]
                if not values or not np.isfinite(values).all():
                    raise ValueError('Provide at least one finite parameter value.')
                return values
            except ValueError as exc:
                raise ValueError('Invalid continuation parameter list; no default values were substituted.') from exc

    def start_solve(self):
        if self._solving:
            return
        self._clear_result()
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
            except ValueError as exc:
                raise ValueError('Initial guesses must be comma-separated numbers.') from exc
            if not np.isfinite(unknown_vals).all():
                raise ValueError('Initial guesses must be finite.')

            multi_cycle_mode = False
            n_components = len(self.ic_known)
            unknown_indices = [i for i, cb in enumerate(self.ic_known)
                               if not cb.isChecked()]

            def build_state(values):
                if len(values) != len(unknown_indices):
                    raise ValueError(
                        f'Expected exactly {len(unknown_indices)} initial guess '
                        f'value(s), got {len(values)}.')
                iterator = iter(values)
                return [float(self.ic_value[i].value()) if self.ic_known[i].isChecked()
                        else float(next(iterator)) for i in range(n_components)]

            if prob_type == 'limit_cycle':
                # Для предельных циклов поле «Неизвестные» — это СПИСОК
                # амплитуд x1, например «2, 6.5, 9» (пример 26.2 задаёт три
                # точки p01, p02, p03).  Каждая амплитуда a порождает
                # приближение [x1=a, x2=0, T=2π, x4=a]; решатель сходится к
                # своему предельному циклу.  One amplitude means one cycle;
                # several amplitudes automatically request several cycles.
                if not unknown_vals:
                    raise ValueError('Enter at least one limit-cycle amplitude.')
                amps = unknown_vals
                two_pi = 2 * np.pi
                combined = []
                for a in amps:
                    combined.extend([float(a), 0.0, two_pi, float(a)])
                initial_guess = combined
                multi_cycle_mode = len(amps) > 1
            elif prob_type == 'lens':
                if len(unknown_vals) != 3:
                    raise ValueError(
                        'Lens requires exactly 3 guesses: psi1(0), psi2(0), T.')
                initial_guess = list(unknown_vals)
            elif prob_type == 'kepler':
                count = len(unknown_indices)
                if count == 0:
                    if unknown_vals:
                        raise ValueError('All initial components are fixed; remove extra guesses.')
                    initial_guess = build_state([])
                else:
                    if not unknown_vals or len(unknown_vals) % count:
                        raise ValueError(
                            f'Kepler guesses must contain complete groups of {count} '
                            f'value(s); got {len(unknown_vals)}.')
                    combined = []
                    for offset in range(0, len(unknown_vals), count):
                        combined.extend(build_state(unknown_vals[offset:offset + count]))
                    initial_guess = combined
            else:
                initial_guess = build_state(unknown_vals)

            mu_list = None
            if self._needs_mu_selection(dataset, explicit_type):
                mu_list = self._ask_mu_values(dataset, explicit_type)
                if mu_list is None:
                    self._set_status(self._tr('cancelled'), 'cancelled')
                    return

            self.solve_btn.setEnabled(False)
            self._pending_revision = self._input_revision
            self._solving = True
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
                                              multi_cycle=multi_cycle_mode)
            self.solver_thread.result_ready.connect(self.on_solve_finished)
            self.solver_thread.error.connect(self.on_solve_error)
            self.solver_thread.start()
        except Exception as e:
            self._solving = False
            QMessageBox.critical(self, self._tr('error'),
                                 f"{self._tr('prep_error')} {str(e)}")
            self.solve_btn.setEnabled(True)
            self._set_status(self._tr('error'), 'error')

    def on_solve_finished(self, x_eval, y_eval):
        self._solving = False
        self.solve_btn.setEnabled(True)
        if self._pending_revision is not None and self._pending_revision != self._input_revision:
            self._clear_result()
            self._set_status(self._tr('changed'), 'changed')
            return
        self._set_status(self._tr('done'), 'done')
        self._last_x = x_eval
        self._last_y = y_eval
        self.graph_btn.setEnabled(True)
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
        self._solving = False
        self._clear_result()
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
            return ["x1", "x2", "T", "x4"][:ny]
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
                    ls=_curve_linestyle(layer_index=i,
                                        layer_count=len(x_eval)),
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
