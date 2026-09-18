"""Defense requirements: convergence, final residual provenance, GUI lifecycle."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
import numpy as np
import pytest
from dataset import Dataset
from solver import BoundaryDiagnostics, ContinuationSolver
from task_io import get_van_der_pol_tasks
from defense_experiment import compute_experiment


def test_two_initial_points_converge_with_independent_integrators():
    trajectories, cycles, report = compute_experiment()
    assert report['numerical_convergence_passed']
    assert np.linalg.norm(trajectories[0][1][:, 0] - trajectories[1][1][:, 0]) > 3
    assert report['phase_aligned_max_distance'] < 1e-4
    # Compare a different integrator/tolerance to catch plausible-looking but
    # inaccurate curves; no reference trajectory is hardcoded into the solver.
    alternative = [replace(d, method='RK45', tol=1e-9) for d in get_van_der_pol_tasks()]
    _, other_cycles, other_report = compute_experiment(alternative)
    assert other_report['numerical_convergence_passed']
    for actual, other in zip(cycles, other_cycles):
        np.testing.assert_allclose(actual, other, atol=2e-6, rtol=0)
    for row in report['runs']:
        assert 6 < row['period_estimate'] < 7
        assert row['max_boundary_residual'] < 1e-10


def linear_dataset(endpoint=1):
    return Dataset(name='linear BVP', x_start=0, x_end=1,
                   equations=['Derivative(y(t), t, 2) = 0'],
                   boundary_conditions=['y0 = 0', f'y1 = {endpoint}'])


def test_diagnostics_use_final_returned_trajectory_and_reset(monkeypatch):
    solver = ContinuationSolver(linear_dataset())
    _, states = solver.solve([0, 0.5])
    np.testing.assert_allclose(solver.last_diagnostics.residuals,
                               [states[0, 0], states[0, -1] - 1], atol=1e-15)
    assert solver.last_diagnostics.max_boundary_residual < 1e-7
    # Simulate a final reintegration defect after the Newton residual passed.
    integrate = solver._integrate_solution
    def corrupt(*args):
        t, y = integrate(*args)
        y[0, -1] += 0.1
        return t, y
    monkeypatch.setattr(solver, '_integrate_solution', corrupt)
    with pytest.raises(RuntimeError, match='Final trajectory'):
        solver.solve([0, 1])
    assert solver.last_diagnostics is None


@pytest.mark.parametrize('values', [[], [np.nan], [np.inf]])
def test_invalid_diagnostics_rejected(values):
    with pytest.raises(ValueError):
        BoundaryDiagnostics(values)


def test_gui_displays_actual_residual_and_invalidates_it(monkeypatch):
    from PyQt5.QtWidgets import QApplication
    import gui
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui.MainWindow, '_save_history', lambda self: None)
    monkeypatch.setattr(gui.MainWindow, '_auto_show_plots', lambda self: None)
    monkeypatch.setattr(gui.SolverThread, 'start', gui.SolverThread.run)
    window = gui.MainWindow()
    try:
        window.set_dataset_to_ui(get_van_der_pol_tasks()[0])
        window.start_solve()
        assert window._status_state == 'done'
        assert window._last_boundary_diagnostics
        maximum = max(d.max_boundary_residual for d in window._last_boundary_diagnostics)
        assert window.residual_label.text() == f'max boundary residual: {maximum:.6e}'
        window.equation_edits[1].setText('-x1')
        assert window._last_boundary_diagnostics == []
        assert window.residual_label.text() == 'max boundary residual: —'
        assert not window.graph_btn.isEnabled()
    finally:
        window.close()
        app.processEvents()


def test_parameter_layers_keep_separate_diagnostics():
    from PyQt5.QtWidgets import QApplication
    import gui
    app = QApplication.instance() or QApplication([])
    ds = Dataset(name='parameter BVP', x_start=0, x_end=1,
                 equations=['Derivative(y(t), t) = lambda'],
                 boundary_conditions=['y1 = 2'])
    worker = gui.SolverThread(ds, [0], smooth_param_list=[0.5, 1.0])
    result = {}
    worker.result_ready.connect(lambda t, y: result.update(layers=t))
    worker.error.connect(lambda error: result.update(error=error))
    worker.run()
    assert 'error' not in result, result
    assert len(worker.boundary_diagnostics) == 2
    for entry, diagnostic in zip(result['layers'], worker.boundary_diagnostics):
        np.testing.assert_allclose(diagnostic.residuals, [entry[1][0, -1] - 2], atol=1e-15)
    assert worker.boundary_diagnostics[0].label != worker.boundary_diagnostics[1].label
    app.processEvents()
