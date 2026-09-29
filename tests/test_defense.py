"""Defense requirements: convergence, final residual provenance, GUI lifecycle."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
import numpy as np
import pytest
from dataset import Dataset
from solver import BoundaryDiagnostics, ContinuationSolver
from task_io import get_van_der_pol_tasks
from defense_experiment import compute_experiment, validate_experiment_inputs, analyse_tail


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
        assert window.residual_label.text() == f'Максимальная невязка граничных условий: {maximum:.6e}'
        assert window.residual_label.toolTip().startswith('lambda=')
        assert 'невязки R=' in window.residual_label.toolTip()
        assert window.name_edit.text() == 'Ван дер Поль: начальная точка (0.1, 0)'
        window._on_lang_selected('en')
        assert window.residual_label.text() == f'max boundary residual: {maximum:.6e}'
        assert 'residuals R=' in window.residual_label.toolTip()
        assert window.name_edit.text() == 'Van der Pol: initial point (0.1, 0)'
        window._on_lang_selected('ru')
        assert window.residual_label.text() == f'Максимальная невязка граничных условий: {maximum:.6e}'
        assert window.graph_btn.isEnabled()
        window.equation_edits[1].setText('-x1')
        assert window._last_boundary_diagnostics == []
        assert window.residual_label.text() == 'Максимальная невязка граничных условий: —'
        assert not window.graph_btn.isEnabled()
    finally:
        window.close()
        app.processEvents()


def test_russian_ui_tooltips_examples_and_plot_switch_with_language(monkeypatch):
    from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox
    import gui
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui.MainWindow, '_save_history', lambda self: None)
    window = gui.MainWindow()
    plot = None
    try:
        assert window._display_example_name(get_van_der_pol_tasks()[1].name) == \
            'Ван дер Поль: начальная точка (4, 2)'
        assert window.residual_label.text() == 'Максимальная невязка граничных условий: —'
        assert window.solve_btn.toolTip().startswith('Решить задачу')
        assert window.equation_edits[0].toolTip().startswith('f(t, x1…x4): правая часть')
        assert window._localized_error('Initial guesses must be finite.') == \
            'Начальные приближения должны быть конечными числами.'
        monkeypatch.setattr(QMessageBox, 'critical', lambda *args: pytest.fail(str(args[-1])))
        def choose_example(dialog):
            items = dialog.comboBoxItems()
            assert items[4].startswith('5. Ван дер Поль: начальная точка')
            assert dialog.okButtonText() == 'ОК'
            assert dialog.cancelButtonText() == 'Отмена'
            dialog.setTextValue(items[4])
            return 1
        monkeypatch.setattr(QInputDialog, 'exec_', choose_example)
        window.show_examples()
        assert window.name_edit.text() == 'Ван дер Поль: начальная точка (0.1, 0)'
        window.dim_spin.setValue(2)
        assert 'правая часть' in window.equation_edits[0].toolTip()
        plot = gui.PlotWindow(np.linspace(0, 1, 5), np.array([
            np.linspace(0, 1, 5), np.linspace(1, 0, 5)]),
            ['x1', 'x2'], 'Тест', window)
        assert plot.btn_split.text() == 'Разделить решения'
        window._on_lang_selected('en')
        assert window.residual_label.text() == 'max boundary residual: —'
        assert window.solve_btn.toolTip().startswith('Solve the problem')
        assert window._localized_error('Initial guesses must be finite.') == \
            'Initial guesses must be finite.'
        assert plot.btn_split.text() == 'Split solutions'
        window._on_lang_selected('ru')
        assert plot.btn_split.text() == 'Разделить решения'
    finally:
        if plot is not None:
            plot.close()
        window.close()
        app.processEvents()


@pytest.mark.parametrize('filter_name, suffix, signature', [
    ('PNG (*.png)', '.png', b'\x89PNG\r\n\x1a\n'),
    ('SVG (*.svg)', '.svg', b'<svg'),
    ('PDF (*.pdf)', '.pdf', b'%PDF'),
])
def test_plot_window_saves_current_view_in_selected_format(
        tmp_path, monkeypatch, filter_name, suffix, signature):
    from PyQt5.QtWidgets import QApplication
    import gui
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui.MainWindow, '_save_history', lambda self: None)
    window = gui.MainWindow()
    plot = None
    try:
        t = np.linspace(0, 1, 20)
        plot = gui.PlotWindow(t, np.vstack([t, 1 - t]), ['x1', 'x2'],
                              'Тестовый график', window)
        plot._x_idx = 1
        plot._y_indices = [1]
        plot._redraw()
        assert plot.canvas.axes.get_xlabel() == 'x1'
        assert plot.canvas.axes.get_ylabel() == 'x2'
        assert plot.btn_save.text() == 'Сохранить рисунок'
        monkeypatch.setattr(gui.QFileDialog, 'getSaveFileName',
                            lambda *args: (str(tmp_path / 'current_view'), filter_name))
        plot.btn_save.click()
        payload = (tmp_path / ('current_view' + suffix)).read_bytes()
        assert signature in payload[:1000]
        window._on_lang_selected('en')
        assert plot.btn_save.text() == 'Save image'
        window._on_lang_selected('ru')
        assert plot.btn_save.text() == 'Сохранить рисунок'
    finally:
        if plot is not None:
            plot.close()
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


def test_different_systems_and_duplicate_initial_points_are_rejected():
    first, second = get_van_der_pol_tasks()
    faster = replace(second, equations=[
        'Derivative(x1(t), t) - 2*x2 = 0',
        'Derivative(x2(t), t) - 2*((1 - x1**2)*x2 - x1) = 0'])
    with pytest.raises(ValueError, match='same system'):
        compute_experiment([first, faster])
    with pytest.raises(ValueError, match='different initial points'):
        compute_experiment([first, first])


def test_initial_conditions_are_mapped_by_variable_not_line_order():
    datasets = get_van_der_pol_tasks()
    reordered = [replace(d, boundary_conditions=d.boundary_conditions[::-1]) for d in datasets]
    points = validate_experiment_inputs(reordered)
    np.testing.assert_array_equal(points, [[0.1, 0], [4, 2]])
    _, _, report = compute_experiment(reordered)
    assert report['numerical_convergence_passed']
    assert report['between_run_period_difference'] < 1e-4


@pytest.mark.parametrize('conditions', [
    ['x1(a) = 1', 'x2(b) = 2'],
    ['x1(a) = 1', 'x1(a) = 2'],
    ['x1(a) = nan', 'x2(a) = 2'],
])
def test_bad_initial_condition_sets_fail_before_solving(conditions):
    first, second = get_van_der_pol_tasks()
    with pytest.raises(ValueError):
        compute_experiment([replace(first, boundary_conditions=conditions), second])


def test_equilibrium_and_short_observation_are_not_claimed_as_cycles():
    first, second = get_van_der_pol_tasks()
    equilibrium = replace(first, boundary_conditions=['x1(a) = 0', 'x2(a) = 0'])
    with pytest.raises(RuntimeError, match='three upward crossings'):
        compute_experiment([equilibrium, second])
    with pytest.raises(RuntimeError, match='three upward crossings'):
        compute_experiment([replace(d, x_end=5) for d in [first, second]])


def test_coarse_sampling_is_rejected_even_for_an_exact_periodic_curve():
    t = np.linspace(0, 60, 121)
    with pytest.raises(RuntimeError, match='too coarse'):
        analyse_tail(t, np.array([np.sin(t), np.cos(t)]))


@pytest.mark.parametrize('coefficient', [0.8, 1.4])
def test_edited_system_changes_computed_cycle_and_period(coefficient):
    datasets = [replace(d, x_end=80, n_points=8001, equations=[d.equations[0],
                f'Derivative(x2(t), t) - ({coefficient}*(1 - x1**2)*x2 - x1) = 0'])
                for d in get_van_der_pol_tasks()]
    _, cycles, report = compute_experiment(datasets)
    _, reference, _ = compute_experiment()
    assert report['numerical_convergence_passed']
    assert abs(report['runs'][0]['period_estimate'] - 6.66328686) > 0.05
    assert np.max(np.linalg.norm(cycles[0] - reference[0], axis=0)) > 0.05


def test_van_der_pol_variational_matrix_against_perturbed_ivps():
    from scipy.integrate import solve_ivp
    dataset = replace(get_van_der_pol_tasks()[0], x_end=1)
    solver = ContinuationSolver(dataset)
    state = np.array([0.4, -0.7])
    expected_jac = [[0, 1], [-2*state[0]*state[1]-1, 1-state[0]**2]]
    np.testing.assert_allclose(solver._state_jacobian(0, state, 1), expected_jac, atol=2e-8)
    endpoint, transition = solver._integrate_state_and_variational(state, 1)
    def endpoint_at(initial):
        return solve_ivp(lambda t, y: [y[1], (1-y[0]**2)*y[1]-y[0]],
                         [0, 1], initial, method='DOP853', rtol=1e-11,
                         atol=1e-13).y[:, -1]
    h = 1e-5
    independent = np.column_stack([
        (endpoint_at(state + h*direction) - endpoint_at(state - h*direction))/(2*h)
        for direction in np.eye(2)])
    np.testing.assert_allclose(endpoint, endpoint_at(state), atol=1e-9)
    np.testing.assert_allclose(transition, independent, atol=2e-7, rtol=2e-7)
