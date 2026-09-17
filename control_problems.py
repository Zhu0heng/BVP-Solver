"""Validated equations for the specialised optimal-control solvers."""
import re
import numpy as np
import sympy as sp
from parser import _safe_parse_expr, _SYM_FUNCS


def fixed_boundary_values(dataset, indices):
    """Require each requested component at both ends; never substitute default BCs."""
    values = {}
    for condition in dataset.boundary_conditions:
        match = re.fullmatch(r'\s*x(\d+)\(([ab])\)\s*=\s*(.+)\s*', condition)
        if not match:
            raise ValueError('This specialised solver requires fixed endpoint conditions xi(a/b) = number.')
        key = (int(match[1]), match[2])
        val = _safe_parse_expr(match[3], _SYM_FUNCS)
        if key in values or val.free_symbols or not np.isfinite(float(val)):
            raise ValueError(f'Invalid or duplicate boundary condition: {condition}')
        values[key] = float(val)
    expected = {(i, side) for i in indices for side in ('a', 'b')}
    if set(values) != expected:
        raise ValueError('Endpoint conditions do not match the specialised solver; use the general solver.')
    return ([values[i, 'a'] for i in indices], [values[i, 'b'] for i in indices])


class LensDynamics:
    """Autonomous control-affine dynamics; maximise over the book's smoothed lens."""
    def __init__(self, dataset):
        x1, x2, u1, u2 = sp.symbols('x1 x2 u1 u2', real=True)
        d = sp.Symbol('_d_')
        local = dict(_SYM_FUNCS, x1=x1, x2=x2, u1=u1, u2=u2, _d_=d)
        params = dataset.parameters or {}
        for key, value in params.items():
            if key not in local and isinstance(value, (int, float)) and np.isfinite(value):
                local[key] = sp.Float(value)
        exprs = {}
        for equation in dataset.equations:
            matches = list(re.finditer(r'Derivative\(x([12])\(t\),\s*t\)', equation))
            if len(matches) != 1:
                raise ValueError('Lens equations must define x1 and x2 derivatives exactly once.')
            idx = int(matches[0][1])
            if idx in exprs:
                raise ValueError('Duplicate derivative in lens equations.')
            equation = equation[:matches[0].start()] + '_d_' + equation[matches[0].end():]
            left, right = equation.split('=', 1) if '=' in equation else (equation, '0')
            roots = sp.solve(_safe_parse_expr(left.strip(), local) - _safe_parse_expr(right.strip(), local), d)
            if len(roots) != 1:
                raise ValueError('Cannot isolate the derivative in the lens equation.')
            expression = roots[0]
            if expression.free_symbols - {x1,x2,u1,u2}:
                raise ValueError('Undefined symbols or explicit time in lens dynamics; no default equation was substituted.')
            exprs[idx] = expression
        if set(exprs) != {1,2}:
            raise ValueError('Lens dynamics require both x1 and x2 equations.')
        f = sp.Matrix([exprs[1], exprs[2]])
        B = f.jacobian([u1,u2])
        if any(value.has(u1,u2) for value in B):
            raise ValueError('Lens dynamics must be affine in u1 and u2.')
        self._f = sp.lambdify((x1,x2,u1,u2), f, 'numpy')
        self._jac = sp.lambdify((x1,x2,u1,u2), f.jacobian([x1,x2]), 'numpy')
        self._B = sp.lambdify((x1,x2), B, 'numpy')

    @staticmethod
    def support_gradient(sigma, mu):
        p, q = sigma
        norm2 = p*p + q*q
        if norm2 < 1e-28:
            return np.zeros(2)
        a = np.sqrt(mu*norm2 + (p+q)**2)
        b = np.sqrt(mu*norm2 + (p-q)**2)
        q1 = (a+b)/2
        dq1 = np.array([(mu*p+p+q)/a + (mu*p+p-q)/b,
                        (mu*q+p+q)/a + (mu*q-p+q)/b])/2
        radius = np.sqrt(q1*q1+q*q)
        return (np.sqrt(2)*q1/radius-1)*dq1 + np.array([0,np.sqrt(2)*q/radius])

    def control(self, state, psi, mu):
        # The support direction is B(x)^T psi, not psi if control gains change.
        return self.support_gradient(np.asarray(self._B(*state), dtype=float).T @ psi, mu)

    def derivative(self, y, mu):
        state, psi, duration = y[:2], y[2:4], y[4]
        control = self.control(state, psi, mu)
        args = (*state, *control)
        dx = duration*np.asarray(self._f(*args), dtype=float).ravel()
        dp = -duration*np.asarray(self._jac(*args), dtype=float).T @ psi
        return np.r_[dx,dp,0.]
