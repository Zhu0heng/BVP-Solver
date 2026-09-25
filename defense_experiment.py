"""Reproducible defense experiment using the project's actual solver pipeline."""
from pathlib import Path
import json
import re

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from solver import ContinuationSolver
from task_io import get_van_der_pol_tasks, save_task


def validate_experiment_inputs(datasets):
    """Require two distinct IVPs of the same system, before any integration."""
    if len(datasets) != 2:
        raise ValueError('The defense experiment requires exactly two initial points.')
    first, second = datasets
    for field in ('equations', 'parameters', 'continuation_param',
                  'continuation_end', 'x_start', 'x_end'):
        if getattr(first, field) != getattr(second, field):
            raise ValueError(f'Both experiments must use the same system and interval: {field}.')
    points = []
    for dataset in datasets:
        if len(dataset.equations) != 2:
            raise ValueError('The phase experiment requires two first-order state equations.')
        values = {}
        for condition in dataset.boundary_conditions:
            match = re.fullmatch(r'\s*(x[12])\(a\)\s*=\s*(.+)\s*', condition)
            if not match or match[1] in values:
                raise ValueError('Provide each initial condition x1(a), x2(a) exactly once.')
            try:
                values[match[1]] = float(match[2])
            except ValueError as exc:
                raise ValueError('Initial conditions for the experiment must be numeric literals.') from exc
        if set(values) != {'x1', 'x2'} or not np.isfinite(list(values.values())).all():
            raise ValueError('Provide finite initial conditions for both x1 and x2.')
        points.append(np.array([values['x1'], values['x2']]))
    separation = np.linalg.norm(points[0] - points[1])
    scale = max(1.0, *(np.linalg.norm(point) for point in points))
    if separation <= 0.1 * scale:
        raise ValueError('Choose two substantially different initial points (separation > 10% of scale).')
    return points


def analyse_tail(t, states, transient_end=40.0):
    """Compare Poincare crossings after transients; no periodic BVP is solved.

    Sampled x=0 upward crossings are refined on a cubic interpolant. The
    last cycle is phase-aligned at that section, so different time shifts
    do not get mistaken for different limit cycles.
    """
    interpolant = CubicSpline(t, states, axis=1)
    indices = np.flatnonzero((states[0, :-1] < 0) & (states[0, 1:] >= 0))
    crossings = [brentq(lambda s: float(interpolant(s)[0]), t[i], t[i + 1])
                 for i in indices if t[i] >= transient_end]
    if len(crossings) < 3:
        raise RuntimeError('Need at least three upward crossings after the transient.')
    last_three = np.asarray(crossings[-3:])
    periods = np.diff(last_three)
    if np.max(np.diff(t)) > np.min(periods) / 100:
        raise RuntimeError('Output sampling is too coarse for cycle validation; increase n_points.')
    phase = np.linspace(0.0, 1.0, 1001)
    cycle = interpolant(last_three[-2] + phase * periods[-1])
    section_states = interpolant(last_three)
    return {
        'period_estimate': float(periods[-1]),
        'consecutive_period_change': float(abs(periods[-1] - periods[-2])),
        'section_y': float(section_states[1, -1]),
        'consecutive_section_change': float(abs(section_states[1, -1] - section_states[1, -2])),
        'last_cycle_endpoint_distance': float(np.linalg.norm(cycle[:, -1] - cycle[:, 0])),
        'last_three_upward_crossings': last_three.tolist(),
    }, cycle


def compute_experiment(datasets=None):
    """Solve both user datasets through ContinuationSolver, with no saved answers."""
    datasets = get_van_der_pol_tasks() if datasets is None else datasets
    initial_points = validate_experiment_inputs(datasets)
    trajectories, summaries, cycles = [], [], []
    for dataset, initial in zip(datasets, initial_points):
        solver = ContinuationSolver(dataset)
        # An initial guess for the existing boundary solver. Actual initial
        # conditions remain defined by dataset.boundary_conditions.
        if solver.order != 1 or solver.dep_vars != ['x1', 'x2']:
            raise ValueError('The phase experiment requires first-order states x1 and x2.')
        t, states = solver.solve(initial)
        summary, cycle = analyse_tail(t, states)
        diagnostic = solver.last_diagnostics
        summary.update(initial_point=states[:, 0].tolist(),
                       boundary_residuals=list(diagnostic.residuals),
                       max_boundary_residual=diagnostic.max_boundary_residual,
                       boundary_conditions=list(dataset.boundary_conditions),
                       method=dataset.method, tolerance=dataset.tol)
        trajectories.append((t, states))
        summaries.append(summary)
        cycles.append(cycle)
    distance = float(np.max(np.linalg.norm(cycles[0] - cycles[1], axis=0)))
    period_difference = abs(summaries[0]['period_estimate'] - summaries[1]['period_estimate'])
    # This is a numerical convergence diagnostic, not an existence/uniqueness
    # proof or the boundary residual of a periodic boundary-value problem.
    converged = (distance < 1e-3 and period_difference < 1e-4 and all(
        row['consecutive_period_change'] < 1e-4 and
        row['consecutive_section_change'] < 1e-4 for row in summaries))
    report = {
        'experiment': 'Van der Pol initial-value trajectories; not a periodic BVP',
        'equations': list(datasets[0].equations),
        'interval': [datasets[0].x_start, datasets[0].x_end],
        'transient_end': 40.0,
        'runs': summaries,
        'phase_aligned_max_distance': distance,
        'between_run_period_difference': period_difference,
        'numerical_convergence_passed': bool(converged),
        'residual_note': 'Boundary residuals here measure the prescribed initial conditions only.',
    }
    if not converged:
        raise RuntimeError(f'Trajectories have not converged to the same cycle: {report}')
    return trajectories, cycles, report


def save_phase_portrait(trajectories, cycles, report, path):
    """Save transient and phase-aligned late cycles on two clearly labelled axes."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(11.6, 5.3), layout='constrained', facecolor='white')
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)
    for index, ((t, states), cycle, row) in enumerate(zip(trajectories, cycles, report['runs'])):
        color = ['#1864ab', '#d9480f'][index]
        label = f"start {tuple(row['initial_point'])}"
        transient = t <= report['transient_end']
        left.plot(states[0, transient], states[1, transient], color=color, alpha=0.7,
                  linewidth=1.1, label=label)
        left.scatter(*states[:, 0], color=color, s=40, zorder=3)
        # Dashed late curve makes coincidence with the first curve visible.
        right.plot(*cycle, color=color, linestyle='-' if index == 0 else '--',
                   linewidth=2, label=f"{label}; T ~ {row['period_estimate']:.6f}")
    left.set_title('Transient trajectories (0 <= t <= 40)')
    right.set_title('Last full cycles, aligned at x=0, y>0')
    for ax in (left, right):
        ax.set(xlabel='x', ylabel='y')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('Two initial points: transient and late-cycle comparison', fontsize=15)
    fig.supxlabel('Long-time IVP integration; this figure is not a solved periodic BVP', fontsize=10)
    fig.savefig(path, dpi=180)


def run_experiment(output_dir='defense_artifacts'):
    datasets = get_van_der_pol_tasks()
    trajectories, cycles, report = compute_experiment(datasets)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plot_path = directory / 'van_der_pol_phase.png'
    save_phase_portrait(trajectories, cycles, report, plot_path)
    for index, dataset in enumerate(datasets, start=1):
        save_task(dataset, str(directory / f'van_der_pol_{index}.json'))
    (directory / 'diagnostics.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for index, row in enumerate(report['runs'], start=1):
        print(f"Run {index}, start={row['initial_point']}: "
              f"max boundary residual = {row['max_boundary_residual']:.6e}; "
              f"estimated T = {row['period_estimate']:.9f}")
    print(f"Phase-aligned cycle distance = {report['phase_aligned_max_distance']:.6e}")
    print(f'Numerical convergence check: passed. Saved figure: {plot_path.resolve()}')
    print('The displayed boundary residual is for initial conditions, not periodic closure.')
    return report
