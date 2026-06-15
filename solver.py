"""
Решатель краевых задач методом продолжения по параметру.

Использует SymPy для безопасного парсинга уравнений и SciPy для численных методов
(пристрелка solve_ivp + коррекция Левенберга-Марквардта).
"""

import numpy as np
from scipy.integrate import solve_ivp

from parser import parse_equation_system, parse_boundary_conditions


class ContinuationSolver:
    """BVP solver using the continuation method (метод продолжения).

    Section 7.26 of the textbook:
      Internal problem (6):  dx/dt = f(t,x),  dX/dt = f'_x·X,  X(t*) = I
      External problem (5): dp/dμ = -[Φ'(p)]⁻¹·Φ(p₀),  p(0)=p₀,  μ∈[0,1]

    where Φ(p) = R(x(a,p), x(b,p)) is the vector of boundary residuals.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        # ε и метод интегрирования берутся из набора данных (управляются из GUI).
        # getattr — для совместимости со старыми JSON без этих полей.
        self.tol = getattr(dataset, 'tol', 1e-9)
        self.method = getattr(dataset, 'method', 'RK45')
        self._validate_dataset()
        self._init_ode()
        self._init_bc()

    def _validate_dataset(self):
        d = self.dataset
        if d.x_start >= d.x_end:
            raise ValueError(f"x_start ({d.x_start}) must be less than x_end ({d.x_end})")
        if d.n_points < 2:
            raise ValueError("n_points must be at least 2")
        if not d.boundary_conditions:
            raise ValueError("boundary_conditions must not be empty")
        if not d.equations:
            raise ValueError("equations must not be empty")

    def _init_ode(self):
        result = parse_equation_system(
            self.dataset.equations, self.dataset.continuation_param
        )
        (self.ode_funcs, self.t_sym, self.funcs, self.lmbda_sym,
         self.dep_vars, self.indep_var, self.bc_var_names, self.order) = result
        self.n_dep = len(self.dep_vars)
        needed = self.n_dep if self.order == 1 else 2 * self.n_dep
        n_bc = len(self.dataset.boundary_conditions)
        if n_bc < needed:
            raise ValueError(
                f"need at least {needed} boundary conditions ({n_bc} given)"
            )

    def _init_bc(self):
        self.bc_funcs = parse_boundary_conditions(
            self.dataset.boundary_conditions,
            self.dataset.continuation_param,
            self.bc_var_names
        )

    def _ode_system(self, t, state, lmbda):
        if self.order == 1:
            vals = {name: state[i] for i, name in enumerate(self.dep_vars)}
            derivs = []
            for i in range(self.n_dep):
                args = [t] + [vals[name] for name in self.dep_vars] + [lmbda]
                derivs.append(self.ode_funcs[i](*args))
            return derivs
        else:
            vals = {}
            for i, name in enumerate(self.dep_vars):
                vals[name] = state[2 * i]
                vals[f'd{name}'] = state[2 * i + 1]
            derivs = []
            for i in range(self.n_dep):
                derivs.append(state[2 * i + 1])
                args = ([t] + [vals[name] for name in self.dep_vars] +
                        [vals[f'd{name}'] for name in self.dep_vars] + [lmbda])
                derivs.append(self.ode_funcs[i](*args))
            return derivs

    def _phi(self, params, lmbda):
        """Compute Φ(p) = vector of boundary residuals."""
        try:
            sol = solve_ivp(
                self._ode_system,
                [self.dataset.x_start, self.dataset.x_end],
                params,
                args=(lmbda,),
                dense_output=True,
                method=self.method,
                rtol=self.tol, atol=self.tol * 1e-3
            )
            y_end = sol.sol(self.dataset.x_end)
        except Exception:
            return np.full(len(self.bc_funcs), 1e10)

        residuals = []
        for bc_func in self.bc_funcs:
            start_vals = list(params)
            end_vals = list(y_end)
            all_vals = []
            if self.order == 1:
                for i in range(self.n_dep):
                    all_vals.extend([start_vals[i], end_vals[i]])
            else:
                for i in range(self.n_dep):
                    all_vals.extend([start_vals[2 * i], start_vals[2 * i + 1],
                                     end_vals[2 * i], end_vals[2 * i + 1]])
            residuals.append(bc_func(*all_vals, lmbda))
        return np.array(residuals)

    def _phi_jacobian(self, params, lmbda):
        """Finite-difference Jacobian of Φ(p)."""
        n = len(params)
        phi0 = self._phi(params, lmbda)
        J = np.zeros((n, n))
        h = 1e-6
        for i in range(n):
            p_h = params.copy()
            p_h[i] += h
            phi_h = self._phi(p_h, lmbda)
            J[:, i] = (phi_h - phi0) / h
        return J

    def solve(self, initial_guess=None, lmbda_values=None):
        """Solve BVP via shooting + Levenberg-Marquardt.

        If lmbda_values is provided, performs parameter continuation:
        solves at each lmbda value in sequence, using the previous
        solution as the initial guess for the next step.
        """
        if self.order == 1:
            need = self.n_dep
        else:
            need = 2 * self.n_dep

        if initial_guess is None or len(initial_guess) != need:
            initial_guess = np.zeros(need)
            for i in range(need):
                if i % 2 == 0:
                    initial_guess[i] = 2.0
            if len(initial_guess) >= 3:
                initial_guess[2] = 6.283
            if len(initial_guess) >= 4:
                initial_guess[3] = 2.0
        p0 = np.asarray(initial_guess, dtype=float)

        from scipy.optimize import root

        if lmbda_values is None or len(lmbda_values) == 0:
            # Use the current parameter value from the dataset if available;
            # otherwise default to the continuation_end (or 0.0).
            param = self.dataset.continuation_param
            params = self.dataset.parameters or {}
            if param in params:
                lmbda_values = [float(params[param])]
            else:
                lmbda_values = [float(getattr(self.dataset, 'continuation_end', 0.0))]

        # Плотность подразбиения больших скачков параметра управляется
        # continuation_steps (поле Steps в GUI): больше шагов → мельче подшаги
        # → устойчивее продолжение по параметру.
        steps = max(1, int(getattr(self.dataset, 'continuation_steps', 10)))
        refined = []
        for i in range(len(lmbda_values) - 1):
            refined.append(lmbda_values[i])
            gap = lmbda_values[i + 1] - lmbda_values[i]
            if abs(gap) > 0.5:
                n_sub = max(2, int(round(abs(gap) * steps / 5.0)))
                for k in range(1, n_sub):
                    refined.append(lmbda_values[i] + gap * k / n_sub)
        refined.append(lmbda_values[-1])

        last_p = p0
        final_lmbda = refined[-1] if refined else 0.0
        for lmbda in refined:
            result = root(lambda p, _l=lmbda: self._phi(p, _l), last_p,
                          method='lm',
                          options={'ftol': self.tol * 1e-3,
                                   'xtol': self.tol * 1e-3})
            if not result.success:
                raise RuntimeError(
                    f"Root finding failed at {self.dataset.continuation_param}={lmbda}: "
                    f"{result.message}"
                )
            last_p = result.x
            final_lmbda = lmbda

        return self._integrate_solution(last_p, final_lmbda)

    def _integrate_solution(self, p, lmbda=0.0):
        """Integrate ODE from x_start to x_end with the given parameter value.

        The lmbda argument must match the actual continuation parameter used
        during solving — previously it was hardcoded to 0.0 which gave wrong
        results whenever the parameter appeared explicitly in the equations.
        """
        x_eval = np.linspace(self.dataset.x_start, self.dataset.x_end,
                             self.dataset.n_points)
        sol = solve_ivp(
            self._ode_system,
            [self.dataset.x_start, self.dataset.x_end],
            p,
            args=(lmbda,),          # use the actual final parameter value
            dense_output=True,
            method=self.method,
            rtol=self.tol, atol=self.tol * 1e-3
        )
        y_eval = sol.sol(x_eval)
        # Защита от нефизических (расходящихся) решений: если интеграл
        # содержит NaN/Inf, корректор «сошёлся» к ветви, уходящей в
        # бесконечность — такое решение нельзя возвращать на график.
        if not np.all(np.isfinite(y_eval)):
            raise RuntimeError(
                "Интегрирование разошлось: решение содержит NaN/Inf — "
                "уточните начальное приближение или ε."
            )
        return x_eval, y_eval

    def full_orbit(self, init_state, n_points=400):
        """Integrate the system from init_state long enough to show a
        representative trajectory.

        Time horizon is chosen ADAPTIVELY using the user's ODE itself
        (no hardcoded Kepler-energy formula):
          • Integrate for a "probe" horizon = max(x_end * 5, 10).
          • If the trajectory returns near init_state during the probe,
            we have a closed orbit — set t_end to the detected period.
          • Otherwise t_end stays at the probe horizon.
        Works for any 2D / 4D system the user defines.
        """
        probe = max(self.dataset.x_end * 5, 10.0)
        try:
            sol_probe = solve_ivp(
                self._ode_system,
                [0, probe],
                init_state,
                args=(0.0,),
                dense_output=True,
                rtol=1e-10, atol=1e-12,
                max_step=probe / 5000,
            )
        except Exception:
            t_eval = np.linspace(0, probe, n_points)
            sol = solve_ivp(self._ode_system, [0, probe], init_state,
                            args=(0.0,), dense_output=True,
                            rtol=1e-9, atol=1e-12, max_step=probe / 2000)
            return t_eval, sol.sol(t_eval)

        # Detect a return to init_state (closed orbit).  We find the first time
        # the FULL state comes back near init_state, then REFINE to the point of
        # CLOSEST approach of that re-entry (the true period) so the drawn orbit
        # closes COMPLETELY.  The previous «first point within tolerance» stopped
        # a little BEFORE the true period and left a visible gap in the ellipse.
        t_dense = np.linspace(0, probe, 8000)
        try:
            ys = sol_probe.sol(t_dense)
            d0 = np.array(init_state).reshape(-1, 1)
            d = np.linalg.norm(ys - d0, axis=0)
            tol = max(0.05 * np.linalg.norm(init_state), 0.05)
            t_end = probe
            left = np.where(d > tol)[0]          # пропускаем «отъезд» от старта
            if len(left) > 0:
                s = left[0]
                back = np.where(d[s:] < tol)[0]  # первое возвращение к старту
                if len(back) > 0:
                    c0 = s + back[0]
                    c1 = c0
                    while c1 + 1 < len(d) and d[c1 + 1] < tol:
                        c1 += 1
                    # точка наибольшего сближения = истинный период → орбита
                    # замыкается полностью, без зазора.
                    t_end = t_dense[c0 + int(np.argmin(d[c0:c1 + 1]))]
        except Exception:
            t_end = probe

        t_eval = np.linspace(0, t_end, n_points)
        sol = solve_ivp(
            self._ode_system,
            [0, t_end],
            init_state,
            args=(0.0,),
            dense_output=True,
            rtol=1e-11, atol=1e-13,
            max_step=t_end / 2000,
        )
        return t_eval, sol.sol(t_eval)

    def continuation(self):
        """Parameter continuation over lambda."""
        lmbda_values = np.linspace(
            self.dataset.continuation_start,
            self.dataset.continuation_end,
            self.dataset.continuation_steps
        )
        solutions = []
        if self.order == 1:
            need = self.n_dep
        else:
            need = 2 * self.n_dep
        initial_guess = np.zeros(need)
        for lmbda in lmbda_values:
            x, y = self.solve(initial_guess)
            solutions.append((x, y))
            dy_start = []
            for i in range(self.n_dep):
                dy_start.append(y[2 * i, 0])
                if len(x) > 1:
                    dy_start.append((y[2 * i, 1] - y[2 * i, 0]) / (x[1] - x[0]))
                else:
                    dy_start.append(y[2 * i + 1, 0])
            initial_guess = np.array(dy_start)
        return lmbda_values, solutions
