"""
Решатель краевых задач методом продолжения по параметру.

Использует SymPy для безопасного парсинга уравнений и SciPy для численных методов.
Метод соответствует §7.26: внутренняя задача интегрирует состояние вместе с
матрицей вариаций, внешняя задача продолжает начальные данные по μ.
"""

import numpy as np
import re
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
        equations = self._with_fixed_parameters(self.dataset.equations)
        result = parse_equation_system(
            equations, self.dataset.continuation_param
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
            self._with_fixed_parameters(self.dataset.boundary_conditions),
            self.dataset.continuation_param,
            self.bc_var_names
        )

    def _with_fixed_parameters(self, expressions):
        """Insert finite numeric constants while keeping the continuation symbol free."""
        result = list(expressions)
        for name, value in (self.dataset.parameters or {}).items():
            if name == self.dataset.continuation_param or not isinstance(value, (int, float)):
                continue
            if not np.isfinite(value) or not re.fullmatch(r'[A-Za-z_]\w*', str(name)):
                raise ValueError(f'Invalid numeric parameter {name!r}')
            result = [re.sub(rf'\b{re.escape(str(name))}\b', f'({float(value)!r})', expression)
                      for expression in result]
        return result

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
            if not sol.success or sol.t[-1] != self.dataset.x_end:
                return np.full(len(self.bc_funcs), 1e10)
            y_end = sol.sol(self.dataset.x_end)
            if not np.isfinite(y_end).all():
                return np.full(len(self.bc_funcs), 1e10)
        except Exception:
            return np.full(len(self.bc_funcs), 1e10)

        return self._boundary_residual(params, y_end, lmbda)

    def _boundary_residual(self, start_state, end_state, lmbda):
        """Evaluate R(x(a), x(b), λ) without integrating the ODE."""
        residuals = []
        for bc_func in self.bc_funcs:
            start_vals = list(start_state)
            end_vals = list(end_state)
            all_vals = []
            if self.order == 1:
                for i in range(self.n_dep):
                    all_vals.extend([start_vals[i], end_vals[i]])
            else:
                for i in range(self.n_dep):
                    all_vals.extend([start_vals[2 * i], start_vals[2 * i + 1],
                                     end_vals[2 * i], end_vals[2 * i + 1]])
            residuals.append(bc_func(*all_vals, lmbda))
        return np.asarray(residuals, dtype=float)

    def _state_jacobian(self, t, state, lmbda):
        """Numerically evaluate f_x for the textbook variational equation."""
        state = np.asarray(state, dtype=float)
        n = len(state)
        jac = np.empty((n, n), dtype=float)
        eps = np.sqrt(np.finfo(float).eps)
        for column in range(n):
            h = eps * max(1.0, abs(state[column]))
            plus = state.copy(); plus[column] += h
            minus = state.copy(); minus[column] -= h
            f_plus = np.asarray(self._ode_system(t, plus, lmbda), dtype=float)
            f_minus = np.asarray(self._ode_system(t, minus, lmbda), dtype=float)
            jac[:, column] = (f_plus - f_minus) / (2.0 * h)
        if not np.isfinite(jac).all():
            raise RuntimeError('The state Jacobian f_x contains NaN/Inf.')
        return jac

    def _integrate_state_and_variational(self, params, lmbda):
        """Solve x'=f and X'=f_x X, X(a)=I (inner problem (6))."""
        params = np.asarray(params, dtype=float)
        n = len(params)
        z0 = np.concatenate([params, np.eye(n).ravel()])

        def augmented(t, z):
            state = z[:n]
            transition = z[n:].reshape(n, n)
            derivative = np.asarray(self._ode_system(t, state, lmbda), dtype=float)
            state_jac = self._state_jacobian(t, state, lmbda)
            return np.concatenate([derivative, (state_jac @ transition).ravel()])

        sol = solve_ivp(
            augmented,
            [self.dataset.x_start, self.dataset.x_end],
            z0,
            method=self.method,
            rtol=self.tol,
            atol=self.tol * 1e-3,
        )
        if not sol.success or sol.t[-1] != self.dataset.x_end:
            raise RuntimeError(f'Variational integration failed: {sol.message}')
        final = sol.y[:, -1]
        if not np.isfinite(final).all():
            raise RuntimeError('Variational integration contains NaN/Inf.')
        return final[:n], final[n:].reshape(n, n)

    def _boundary_partials(self, start_state, end_state, lmbda):
        """Return R_x(a) and R_x(b), used in Φ'=R_a+R_b X(b)."""
        start_state = np.asarray(start_state, dtype=float)
        end_state = np.asarray(end_state, dtype=float)
        rows = len(self.bc_funcs)
        n = len(start_state)
        r_start = np.empty((rows, n), dtype=float)
        r_end = np.empty((rows, n), dtype=float)
        eps = np.sqrt(np.finfo(float).eps)
        for column in range(n):
            h = eps * max(1.0, abs(start_state[column]))
            plus = start_state.copy(); plus[column] += h
            minus = start_state.copy(); minus[column] -= h
            r_start[:, column] = (
                self._boundary_residual(plus, end_state, lmbda)
                - self._boundary_residual(minus, end_state, lmbda)
            ) / (2.0 * h)

            h = eps * max(1.0, abs(end_state[column]))
            plus = end_state.copy(); plus[column] += h
            minus = end_state.copy(); minus[column] -= h
            r_end[:, column] = (
                self._boundary_residual(start_state, plus, lmbda)
                - self._boundary_residual(start_state, minus, lmbda)
            ) / (2.0 * h)
        return r_start, r_end

    def _phi_jacobian(self, params, lmbda):
        """Compute Φ'(p) from the inner variational Cauchy problem."""
        params = np.asarray(params, dtype=float)
        end_state, transition = self._integrate_state_and_variational(params, lmbda)
        r_start, r_end = self._boundary_partials(params, end_state, lmbda)
        jac = r_start + r_end @ transition
        if not np.isfinite(jac).all():
            raise RuntimeError("The boundary map Jacobian Φ'(p) contains NaN/Inf.")
        return jac

    def _phi_jacobian_finite_difference(self, params, lmbda):
        """Independent diagnostic/fallback Jacobian of Φ(p)."""
        n = len(params)
        phi0 = self._phi(params, lmbda)
        J = np.zeros((len(phi0), n))
        h = 1e-6
        for i in range(n):
            p_h = params.copy()
            p_h[i] += h
            phi_h = self._phi(p_h, lmbda)
            J[:, i] = (phi_h - phi0) / h
        return J

    def _continuation_path(self, initial, lmbda):
        """Integrate dp/dμ=-Φ'(p)^+Φ(p0), p(0)=p0, μ∈[0,1].

        The external Cauchy problem (5) is advanced with explicit Euler.  The
        GUI's continuation-step setting is therefore a real numerical control,
        not merely a label.  A variational Newton projection removes the
        Euler discretisation error at μ=1.
        """
        p = np.asarray(initial, dtype=float).copy()
        phi_initial = self._phi(p, lmbda)
        if not np.isfinite(phi_initial).all():
            raise RuntimeError('Initial boundary residual contains NaN/Inf.')
        target_tol = max(1e-8, 100.0 * self.tol)
        if np.max(np.abs(phi_initial)) <= target_tol:
            return p

        steps = max(1, int(getattr(self.dataset, 'continuation_steps', 10)))
        delta_mu = 1.0 / steps
        for _ in range(steps):
            jac = self._phi_jacobian(p, lmbda)
            velocity, *_ = np.linalg.lstsq(jac, -phi_initial, rcond=None)
            p = p + delta_mu * velocity
            if not np.isfinite(p).all():
                raise RuntimeError('External continuation produced NaN/Inf.')

        # Project the endpoint onto Φ(p)=0 using the same variational
        # Jacobian.  Backtracking prevents a Newton step from increasing the
        # boundary defect on strongly nonlinear examples.
        for _ in range(8):
            residual = self._phi(p, lmbda)
            residual_norm = float(np.max(np.abs(residual)))
            if residual_norm <= target_tol:
                return p
            jac = self._phi_jacobian(p, lmbda)
            correction, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
            accepted = False
            scale = 1.0
            for _ in range(10):
                candidate = p + scale * correction
                candidate_residual = self._phi(candidate, lmbda)
                if (np.isfinite(candidate_residual).all()
                        and np.max(np.abs(candidate_residual)) < residual_norm):
                    p = candidate
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break

        # Degenerate starts (for example Φ'(p0)=0) can make continuation
        # locally stationary.  LM is kept only as a robustness fallback; its
        # Jacobian is still obtained from the textbook variational problem.
        from scipy.optimize import root
        result = root(
            lambda q: self._phi(q, lmbda),
            p,
            jac=lambda q: self._phi_jacobian(q, lmbda),
            method='lm',
            options={'ftol': self.tol * 1e-3, 'xtol': self.tol * 1e-3},
        )
        if not result.success:
            raise RuntimeError(f'Continuation correction failed: {result.message}')
        return np.asarray(result.x, dtype=float)

    def solve(self, initial_guess=None, lmbda_values=None):
        """Solve the BVP by the textbook continuation method.

        If lmbda_values is provided, performs parameter continuation:
        solves at each lmbda value in sequence, using the previous
        solution as the initial guess for the next step.
        """
        if self.order == 1:
            need = self.n_dep
        else:
            need = 2 * self.n_dep

        if initial_guess is None:
            initial_guess = np.zeros(need)
            for i in range(need):
                if i % 2 == 0:
                    initial_guess[i] = 2.0
            if len(initial_guess) >= 3:
                initial_guess[2] = 6.283
            if len(initial_guess) >= 4:
                initial_guess[3] = 2.0
        elif len(initial_guess) != need:
            raise ValueError(f'initial_guess must contain exactly {need} values, got {len(initial_guess)}')
        p0 = np.asarray(initial_guess, dtype=float)
        if not np.isfinite(p0).all():
            raise ValueError('initial_guess must contain only finite values')

        if lmbda_values is None or len(lmbda_values) == 0:
            # Use the current parameter value from the dataset if available;
            # otherwise default to the continuation_end (or 0.0).
            param = self.dataset.continuation_param
            params = self.dataset.parameters or {}
            if param in params:
                lmbda_values = [float(params[param])]
            else:
                lmbda_values = [float(getattr(self.dataset, 'continuation_end', 0.0))]
        lmbda_values = [float(value) for value in lmbda_values]
        if not lmbda_values or not np.isfinite(lmbda_values).all():
            raise ValueError('continuation parameter values must be finite')

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
            last_p = self._continuation_path(last_p, lmbda)
            residual = self._phi(last_p, lmbda)
            if not np.isfinite(residual).all() or np.max(np.abs(residual)) > max(1e-7, 100*self.tol):
                raise RuntimeError(f'Boundary conditions not satisfied: max residual={np.max(np.abs(residual)):.3g}')
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
        if not sol.success or sol.t[-1] != self.dataset.x_end:
            raise RuntimeError(f'Integration did not reach the endpoint: {sol.message}')
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
        start = self.dataset.x_start
        probe = max((self.dataset.x_end - start) * 5, 10.0)
        param = float((self.dataset.parameters or {}).get(
            self.dataset.continuation_param, self.dataset.continuation_end))
        try:
            sol_probe = solve_ivp(
                self._ode_system,
                [start, start + probe],
                init_state,
                args=(param,),
                dense_output=True,
                rtol=1e-10, atol=1e-12,
                max_step=probe / 5000,
            )
            if not sol_probe.success:
                raise RuntimeError(sol_probe.message)
        except Exception as exc:
            raise RuntimeError(f'Cannot extend the orbit: {exc}') from exc

        # Detect a return to init_state (closed orbit).  We find the first time
        # the FULL state comes back near init_state, then REFINE to the point of
        # CLOSEST approach of that re-entry (the true period) so the drawn orbit
        # closes COMPLETELY.  The previous «first point within tolerance» stopped
        # a little BEFORE the true period and left a visible gap in the ellipse.
        t_dense = np.linspace(start, start + probe, 8000)
        try:
            ys = sol_probe.sol(t_dense)
            d0 = np.array(init_state).reshape(-1, 1)
            d = np.linalg.norm(ys - d0, axis=0)
            tol = max(0.05 * np.linalg.norm(init_state), 0.05)
            t_end = start + probe
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
            t_end = start + probe

        t_eval = np.linspace(start, t_end, n_points)
        sol = solve_ivp(
            self._ode_system,
            [start, t_end],
            init_state,
            args=(param,),
            dense_output=True,
            rtol=1e-11, atol=1e-13,
            max_step=(t_end - start) / 2000,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
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
            x, y = self.solve(initial_guess, lmbda_values=[lmbda])
            solutions.append((x, y))
            initial_guess = y[:, 0].copy()
        return lmbda_values, solutions
