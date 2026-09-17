"""Regression tests for edited inputs, actual solves, and plotted data."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace
import numpy as np
import pytest
from context import *
from PyQt5.QtWidgets import QApplication, QMessageBox
import gui
from dataset import Dataset
from solver import ContinuationSolver
from task_io import get_example_tasks
from control_problems import LensDynamics

APP = QApplication.instance() or QApplication([])


@pytest.fixture
def editor(monkeypatch):
    # The numerical worker and the chart renderer are real; only dispatch/dialogs are synchronous.
    monkeypatch.setattr(gui.SolverThread, 'start', gui.SolverThread.run)
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a: errors.append(a[-1]))
    monkeypatch.setattr(gui.MainWindow, '_save_history', lambda self: None)
    win = gui.MainWindow()
    win._ask_mu_values = lambda *a, **k: [1.,.1,1e-6]
    win.audit_errors = errors
    yield win
    win._close_plot_windows()
    win.deleteLater()
    APP.processEvents()


def simple_problem(rhs='1', bc='x1(a) = 0', end=1):
    return Dataset(name='edited', x_start=0, x_end=end,
                   equations=[f'Derivative(x1(t), t) - ({rhs}) = 0'],
                   boundary_conditions=[bc], n_points=100)


def solve_editor(win, ds):
    win.set_dataset_to_ui(ds)
    win.start_solve()
    assert not win.audit_errors, win.audit_errors
    assert win._status_state == 'done'
    return win._last_x, win._last_y


def test_gui_coefficient_bc_interval_change_actual_chart(editor):
    t,y = solve_editor(editor, simple_problem())
    old = editor._plot_windows[0]
    np.testing.assert_allclose(y[0], t, atol=1e-7)
    editor.equation_edits[0].setText('2')
    assert editor._last_x is None and not editor.graph_btn.isEnabled()
    assert not old.isVisible()
    editor.start_solve()
    np.testing.assert_allclose(editor._last_y[0], 2*editor._last_x, atol=1e-7)
    np.testing.assert_array_equal(editor._plot_windows[0].canvas.axes.lines[0].get_ydata(), editor._last_y[0])
    editor.bc_edit.setPlainText('x1(a) = 3')
    editor.start_solve()
    np.testing.assert_allclose(editor._last_y[0], 3+2*editor._last_x, atol=1e-7)
    editor.x_end_spin.setValue(2)
    editor.start_solve()
    assert abs(editor._last_y[0,-1]-7) < 1e-7


@pytest.mark.parametrize('rhs', ['x1 + 1','-x1 + 2','-2*(x1-3)'])
def test_expression_roundtrip_preserves_parentheses(editor,rhs):
    ds=simple_problem(rhs)
    editor.set_dataset_to_ui(ds)
    before=ContinuationSolver(ds)
    after=ContinuationSolver(editor.get_dataset_from_ui())
    np.testing.assert_allclose(before._ode_system(.2,[.4],0),after._ode_system(.2,[.4],0))


def test_variable_alias_mapping_is_simultaneous(editor):
    ds=Dataset(name='swap',x_start=0,x_end=1,equations=[
        'Derivative(x1(t), t) - x2 = 0','Derivative(x2(t), t) + x1 = 0'],
        boundary_conditions=['x1(a) = 1','x2(a) = 0'])
    editor.dim_spin.setValue(2)
    editor.var_names_edit.setText('x2, x1')
    editor.set_dataset_to_ui(ds)
    actual=ContinuationSolver(editor.get_dataset_from_ui())._ode_system(.2,[3,5],0)
    np.testing.assert_allclose(actual,[5,-3])


def test_all_six_initial_guesses_are_exposed(editor):
    editor.set_dataset_to_ui(get_example_tasks()[2])
    assert len(editor.ic_known) == len(editor.ic_value) == 6
    assert [c.isChecked() for c in editor.ic_known] == [True]*3+[False]*3
    editor.dim_spin.setValue(2)
    assert len(editor.ic_value)==2


def test_edit_during_solve_cannot_publish_obsolete_result(editor, monkeypatch):
    editor.set_dataset_to_ui(simple_problem())
    monkeypatch.setattr(gui.SolverThread,'start',lambda self: None)
    editor.start_solve()
    editor.equation_edits[0].setText('2')
    editor.on_solve_finished(np.array([0,1]),np.array([[0,1]]))
    assert editor._last_x is None
    assert editor._status_state=='changed'
    assert not editor.graph_btn.isEnabled()


def test_failed_solve_does_not_reuse_old_plot(editor):
    solve_editor(editor,simple_problem())
    editor.equation_edits[0].setText('sin(')
    editor.start_solve()
    assert editor.audit_errors
    assert editor._last_x is None and not editor._plot_windows
    assert not editor.graph_btn.isEnabled()


def test_changed_guesses_may_have_same_unique_solution(editor):
    ds=simple_problem('x1 + 1')
    editor.set_dataset_to_ui(ds)
    editor.ic_known[0].setChecked(False)
    editor.guess_edit.setText('0.2')
    editor.start_solve()
    y=editor._last_y.copy()
    editor.guess_edit.setText('2.0')
    editor.start_solve()
    assert not editor.audit_errors
    np.testing.assert_allclose(editor._last_y,y,atol=1e-7)
    np.testing.assert_allclose(y[0],np.expm1(editor._last_x),atol=1e-7)


def test_lambda_continuation_uses_each_parameter():
    ds=simple_problem('lambda')
    ds.continuation_steps=3
    values,sols=ContinuationSolver(ds).continuation()
    np.testing.assert_allclose([y[0,-1] for t,y in sols],values,atol=1e-7)


def test_single_selected_parameter_is_not_ignored():
    worker=gui.SolverThread(simple_problem('lambda'),[0],smooth_param_list=[2.])
    result={}
    worker.result_ready.connect(lambda t,y: result.update(t=t,y=y))
    worker.error.connect(lambda error: pytest.fail(error))
    worker.run()
    t,y=result['t'][0][:2]
    np.testing.assert_allclose(y[0],2*t,atol=1e-7)


def test_nonzero_phase_boundary_is_not_misclassified_as_zero():
    ds=get_example_tasks()[1]
    ds.boundary_conditions[0]='x2(a) = 0.5'
    assert gui.detect_problem_type(ds)=='custom'


def test_orbit_extension_uses_current_parameter_and_start_time():
    ds=simple_problem('lambda')
    ds.x_start=2; ds.x_end=3
    ds.parameters={'lambda':2.}
    t,y=ContinuationSolver(ds).full_orbit([0],n_points=25)
    np.testing.assert_allclose(y[0],2*(t-2),atol=1e-8)


def test_inconsistent_boundary_conditions_are_not_success():
    ds=simple_problem('0')
    ds.boundary_conditions=['x1(a) = 1','x1(b) = 2']
    with pytest.raises(RuntimeError,match='Boundary'):
        ContinuationSolver(ds).solve([1])


def test_integration_failure_cannot_be_extrapolated_as_solution():
    ds=simple_problem('x1**2','x1(a) = 1',2)
    with pytest.raises(RuntimeError,match='Integration'):
        ContinuationSolver(ds)._integrate_solution(np.array([1.]))


@pytest.mark.parametrize('rhs',['unknown + u1','sin( + u1','u1**2 + x2'])
def test_invalid_lens_equations_never_fall_back_to_defaults(rhs):
    ds=get_example_tasks()[3]
    ds.equations[0]=f'Derivative(x1(t), t) - ({rhs}) = 0'
    with pytest.raises((ValueError,SyntaxError)):
        LensDynamics(ds)


def test_lens_control_gain_changes_support_direction():
    ds=get_example_tasks()[3]
    ds.equations[0]='Derivative(x1(t), t) - (x2 + 2*u1) = 0'
    dynamics=LensDynamics(ds)
    psi=np.array([.3,.8])
    actual=dynamics.control([4,1],psi,.1)
    np.testing.assert_allclose(actual,dynamics.support_gradient([.6,.8],.1),atol=1e-12)
    assert np.max(abs(actual-dynamics.support_gradient(psi,.1)))>.01


def test_smoothing_failure_is_an_error_not_a_reused_layer(monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize,'root',lambda *a,**k: SimpleNamespace(success=False))
    worker=gui.SolverThread(get_example_tasks()[3],smooth_param_list=[1,.1])
    errors,outputs=[],[]
    worker.error.connect(errors.append)
    worker.result_ready.connect(lambda x,y: outputs.append((x,y)))
    worker.run()
    assert errors and not outputs


def test_language_switch_keeps_method_and_result(editor):
    solve_editor(editor,simple_problem())
    editor.sol_method_combo.setCurrentIndex(1)
    editor.start_solve()
    current=editor._last_y
    editor._on_lang_selected('en')
    assert editor.sol_method_combo.currentIndex()==1
    assert editor._last_y is current
    assert editor._status_state=='done'


def test_multi_cycle_control_remains_hidden_during_equation_edit(editor):
    editor.set_dataset_to_ui(get_example_tasks()[1])
    assert editor.multi_cycle_cb.isHidden()
    assert not editor.multi_cycle_cb.isChecked()
    equation = editor.equation_edits[1]
    equation.setText(equation.text().replace('sin', 'cos'))
    assert editor.multi_cycle_cb.isHidden()
    assert not editor.multi_cycle_cb.isChecked()
    equation.setText(equation.text().replace('cos', 'sin'))
    assert editor.multi_cycle_cb.isHidden()
    assert not editor.multi_cycle_cb.isChecked()


def test_curve_styles_are_generic_and_deterministic():
    assert gui._curve_linestyle(variable_index=0, variable_count=2) == '-'
    assert gui._curve_linestyle(variable_index=1, variable_count=2) == '-'
    assert gui._curve_linestyle(variable_index=2, variable_count=3) == '-'
    assert gui._curve_linestyle(layer_index=0, layer_count=3) == '-'
    assert gui._curve_linestyle(layer_index=1, layer_count=3) == '-'
    assert gui._curve_linestyle(layer_index=2, layer_count=3) == '-'
    assert gui._curve_linestyle(overlaps_existing=True) == '--'
