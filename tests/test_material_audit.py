"""Supplemental acceptance checks from the supplied PDFs; failures are findings.

Run with the complete suite: python -m pytest -q
These checks are intentionally not marked xfail.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QEventLoop, QTimer

import gui
from dataset import Dataset
from solver import ContinuationSolver
from task_io import get_example_tasks, save_task, load_task

APP = QApplication.instance() or QApplication([])


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(gui.MainWindow, '_save_history', lambda self: None)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: pytest.fail(str(args[-1])))
    win = gui.MainWindow()
    yield win
    win._close_plot_windows()
    win.close()
    win.deleteLater()
    APP.processEvents()


@pytest.fixture(scope='module')
def book_results():
    results = []
    guesses = [[2,0,-.5,.5,2,0,.5,-.5],
               [2,0,2*np.pi,2,6.5,0,2*np.pi,6.5,9,0,2*np.pi,9],
               [1,0,0,.5,.5,.5], [4,1,-.5,-.1,4]]
    for i, ds in enumerate(get_example_tasks()):
        worker = gui.SolverThread(ds, guesses[i], multi_cycle=(i==1),
                                  smooth_param_list=[1.,.1,1e-6] if i==3 else None)
        result = {}
        worker.result_ready.connect(lambda t,y: result.update(t=t,y=y))
        worker.error.connect(lambda error: result.update(error=error))
        worker.run()
        assert 'error' not in result, result
        layers = result['t'] if isinstance(result['t'],list) else [(result['t'], result['y'])]
        # Diagnostics must describe each accepted, displayed trajectory, including
        # specialised solvers which do not finish through ContinuationSolver.solve.
        assert len(worker.boundary_diagnostics) == len(layers)
        for layer, diagnostic in zip(layers, worker.boundary_diagnostics):
            y = layer[1]
            if i == 3:
                expected = np.r_[y[:2, 0] - [4, 1], y[:2, -1],
                                 np.sum(y[2:4, -1]**2) - 1]
            else:
                expected = ContinuationSolver(ds)._boundary_residual(
                    y[:, 0], y[:, -1], ds.continuation_end)
            np.testing.assert_allclose(diagnostic.residuals, expected, atol=1e-14)
            assert diagnostic.max_boundary_residual <= 1e-6
        results.append(layers)
    return results


def test_book_kepler_initial_velocities(book_results):
    actual = [entry[1][2:4,0] for entry in book_results[0]]
    assert len(actual)==2
    np.testing.assert_allclose(actual, [[.0000004884,.5000000745],[.4510782034,-.2994186665]],atol=2e-5)


def test_book_three_cycle_amplitudes_and_periods(book_results):
    actual = [entry[1][[0,2],0] for entry in book_results[1]]
    np.testing.assert_allclose(actual, [[3.9655467678,6.4661401325],
                                       [7.1078664573,6.3387892836],
                                       [10.2456910360,6.3101121791]],atol=2e-5)


def test_book_triple_terminal_costates_and_control_bound(book_results):
    _,y=book_results[2][0]
    # Book p.256 gives p at t*=T, NOT at t=0.
    np.testing.assert_allclose(y[:6,-1], [0,0,0,-2.9850435834,4.8880088678,-2.9083874537],atol=2e-5)
    assert np.max(abs(y[6])) <= 1+1e-9


def test_book_lens_small_mu_lies_near_original_control_set(book_results):
    t,y,*_=book_results[3][-1]
    # Original U is the intersection of two disks of radius sqrt(2).
    u1,u2=y[5:7]
    violation=np.maximum((u1-1)**2+u2**2-2,(u1+1)**2+u2**2-2)
    assert violation.max()<1e-4
    np.testing.assert_allclose(y[:2,-1],0,atol=2e-6)
    assert t[-1]>0


@pytest.mark.parametrize('index,edit_index,replacement',[
    (0,2,'0'), (1,1,'-x3*(x1-cos(x2))'),
    (2,2,'x6'), (3,0,'x2')])
def test_actual_equation_edit_and_restore_controls(window,index,edit_index,replacement):
    window.set_dataset_to_ui(get_example_tasks()[index])
    before=(window.multi_cycle_cb.isHidden(),window.multi_cycle_cb.isChecked(),
            window.x_start_spin.isEnabled(),window.x_end_spin.isEnabled())
    edit=window.equation_edits[edit_index]
    original=edit.text()
    edit.setText(replacement)
    edit.setText(original)
    after=(window.multi_cycle_cb.isHidden(),window.multi_cycle_cb.isChecked(),
           window.x_start_spin.isEnabled(),window.x_end_spin.isEnabled())
    assert after==before


def test_explicit_limit_cycle_mode_does_not_expose_hidden_switch(window):
    window.set_dataset_to_ui(get_example_tasks()[1])
    edit=window.equation_edits[1]
    edit.setText(edit.text().replace('sin','cos'))
    window.sol_method_combo.setCurrentIndex(gui._SOLVER_CODES.index('limit_cycle'))
    assert window.multi_cycle_cb.isHidden()
    assert not window.multi_cycle_cb.isChecked()


def test_limit_cycle_amplitude_list_automatically_selects_all_cycles(window,monkeypatch):
    captured={}
    monkeypatch.setattr(gui.SolverThread,'start',
                        lambda self:captured.update(guess=self.initial_guess,
                                                    multi=self.multi_cycle))
    window.set_dataset_to_ui(get_example_tasks()[1])
    window.guess_edit.setText('2, 6.5, 9')
    window.start_solve()
    assert window.multi_cycle_cb.isHidden()
    assert captured['multi'] is True
    assert len(captured['guess'])==12


@pytest.mark.parametrize('example_index,guess',[
    (0,'0.1, 0.2, 0.3'),
    (2,'0.1, 0.2'),
    (3,'-0.5, 4'),
    (0,'0.1, 0.2, 0.3, 0.4, 0.5'),
])
def test_problem_specific_guess_groups_are_never_silently_truncated(
        window,monkeypatch,example_index,guess):
    messages=[]
    monkeypatch.setattr(QMessageBox,'critical',
                        lambda *args:messages.append(str(args[-1])))
    window.set_dataset_to_ui(get_example_tasks()[example_index])
    window.guess_edit.setText(guess)
    window.start_solve()
    assert messages
    assert window.solver_thread is None


def test_json_ui_roundtrip_preserves_high_precision_interval(window,tmp_path):
    ds=get_example_tasks()[0]
    ds.x_end=7.123456789
    path=tmp_path/'precise.json'
    save_task(ds,str(path))
    window.set_dataset_to_ui(load_task(str(path)))
    assert abs(window.get_dataset_from_ui().x_end-ds.x_end)<1e-10


def test_json_roundtrip_preserves_custom_guesses(window,tmp_path):
    window.set_dataset_to_ui(get_example_tasks()[1])
    window.guess_edit.setText('4, 8')
    path=tmp_path/'guesses.json'
    save_task(window.get_dataset_from_ui(),str(path))
    window.set_dataset_to_ui(load_task(str(path)))
    assert window.guess_edit.text()=='4, 8'


@pytest.mark.parametrize('field',['x_start','x_end','tol'])
def test_nonfinite_dataset_values_rejected(field):
    data=get_example_tasks()[0].to_dict()
    data[field]=float('nan')
    with pytest.raises(ValueError):
        Dataset.from_dict(data)


def test_incomplete_initial_guess_rejected_instead_of_replaced():
    ds=Dataset(name='branch',x_start=0,x_end=1,
               equations=['Derivative(x1(t), t) = 0'],boundary_conditions=['x10**2 - 1'])
    solver=ContinuationSolver(ds)
    _,valid=solver.solve([-1.])
    np.testing.assert_allclose(valid,-1.,atol=1e-8)
    with pytest.raises(ValueError):
        solver.solve([-1.,-1.])


def test_equivalent_equation_notation_survives_gui_load(window):
    ds=Dataset(name='direct RHS',x_start=0,x_end=1,
               equations=['Derivative(x1(t), t) = 1'],boundary_conditions=['x1(a) = 0'])
    original=ContinuationSolver(ds)
    np.testing.assert_allclose(original._ode_system(0,[0],0),[1])
    window.set_dataset_to_ui(ds)
    loaded=ContinuationSolver(window.get_dataset_from_ui())
    np.testing.assert_allclose(loaded._ode_system(0,[0],0),[1])


def make_layer_plot(window):
    t=np.linspace(0,1,25)
    layers=[(t,np.array([t*i,(t*i)**2]),None,f'parameter={i}') for i in (1,2,3)]
    plot=gui.PlotWindow(layers,None,['x1','x2'],'arbitrary layers',window)
    window._plot_windows.append(plot)
    return plot


def test_split_preserves_line_styles(window):
    plot=make_layer_plot(window)
    before=[l.get_linestyle() for l in plot.canvas.axes.lines[:3]]
    plot._toggle_split_view()
    after=[p.canvas.axes.lines[0].get_linestyle() for p in window._plot_windows[-3:]]
    assert after==before


def test_split_preserves_selected_variables(window):
    plot=make_layer_plot(window)
    plot._y_indices=[1]
    plot._redraw()
    plot._toggle_split_view()
    assert all(p._y_items[p._y_indices[0]]=='x2' for p in window._plot_windows[-3:])


def test_example_26_3_never_uses_dashes_for_computed_curves(window,book_results):
    t,y=book_results[2][0][:2]
    names=['x1','x2','x3','p1','p2','p3','u']
    plot=gui.PlotWindow(t,y,names,'26.4',window,problem_type='triple')
    window._plot_windows.append(plot)
    # The Y picker collects checked items in displayed order, p3 then u.
    plot._y_indices=[5,6]
    plot._redraw()
    lines=plot.canvas.axes.lines
    assert lines[0].get_label().endswith('p3') and lines[0].get_linestyle()=='-'
    assert lines[1].get_label().endswith('u') and lines[1].get_linestyle()=='-'
    assert all(line.get_linestyle()=='-' for line in lines)


def test_dashes_are_used_only_for_sustained_visual_overlap(window):
    t=np.linspace(0,1,80)
    overlapping=[(t,np.array([t]),None,'first'),
                 (t,np.array([t]),None,'same')]
    plot=gui.PlotWindow(overlapping,None,['x1'],'overlap',window)
    window._plot_windows.append(plot)
    assert [line.get_linestyle() for line in plot.canvas.axes.lines[:2]]==['-','--']

    crossing=[(t,np.array([t]),None,'up'),
              (t,np.array([1-t]),None,'down')]
    plot2=gui.PlotWindow(crossing,None,['x1'],'crossing',window)
    window._plot_windows.append(plot2)
    assert [line.get_linestyle() for line in plot2.canvas.axes.lines[:2]]==['-','-']


def test_figure_26_6_supports_dimensionless_time(window,book_results):
    names=['x1','x2','psi1','psi2','T','u1','u2']
    plot=gui.PlotWindow(book_results[3],None,names,'26.6',window,problem_type='lens')
    window._plot_windows.append(plot)
    plot._y_indices=[plot._y_items.index('u1')]
    plot._redraw()
    # Textbook plots have all smoothing layers aligned on tau in [0,1].
    assert any(name in plot._x_items for name in ('tau','τ','t/T')), plot._x_items
    assert all(line.get_linestyle()=='-' for line in plot.canvas.axes.lines)


def test_real_worker_signals_and_gui_event_processing(window):
    ds=Dataset(name='real worker',x_start=0,x_end=1,
               equations=['Derivative(x1(t), t) - 1 = 0'],boundary_conditions=['x1(a) = 0'])
    window.set_dataset_to_ui(ds)
    # This check is about the real QThread and event delivery.  Creating and
    # tearing down a Matplotlib Qt canvas in the same off-screen pytest process
    # is covered separately by the plot tests above and can crash Qt on Windows.
    window._auto_show_plots=lambda: None
    loop=QEventLoop()
    ticks=[]
    timer=QTimer(); timer.setInterval(1); timer.timeout.connect(lambda:ticks.append(1)); timer.start()
    deadline=QTimer(); deadline.setSingleShot(True); deadline.timeout.connect(loop.quit)
    window.start_solve()
    window.solver_thread.finished.connect(loop.quit)
    window.solver_thread.error.connect(loop.quit)
    deadline.start(15000)
    # A tiny problem may finish between start_solve() and the two connections.
    # Re-checking the state prevents a needless nested event loop in that race.
    if window._solving:
        loop.exec_()
    timer.stop(); deadline.stop()
    assert window.solver_thread.wait(15000)
    APP.processEvents()
    assert window._status_state=='done'
    assert ticks
    np.testing.assert_allclose(window._last_y[0],window._last_x,atol=1e-7)


def test_known_constant_parameter_dictionary_is_used():
    ds=Dataset(name='constant k',x_start=0,x_end=1,
               equations=['Derivative(x1(t), t) - k = 0'],
               boundary_conditions=['x1(a) = 0'],parameters={'k':2.0})
    t,y=ContinuationSolver(ds).solve([0.])
    np.testing.assert_allclose(y[0],2*t,atol=1e-7)


def test_textbook_variational_jacobian_matches_exact_boundary_map():
    ds=Dataset(name='variational equation',x_start=0,x_end=1,
               equations=['Derivative(x1(t), t) - lambda*x1 = 0'],
               boundary_conditions=['x1(b) = 3'],
               parameters={'lambda': 0.7})
    solver=ContinuationSolver(ds)
    jac=solver._phi_jacobian(np.array([1.2]),0.7)
    np.testing.assert_allclose(jac,[[np.exp(0.7)]],rtol=2e-7,atol=2e-8)


def test_generic_solver_executes_external_mu_continuation(monkeypatch):
    ds=Dataset(name='external continuation',x_start=0,x_end=1,
               equations=['Derivative(x1(t), t) = 0'],
               boundary_conditions=['x1(b) = 2'],continuation_steps=6)
    solver=ContinuationSolver(ds)
    calls=[]
    original=solver._phi_jacobian
    def observed(params,lmbda):
        calls.append(np.asarray(params).copy())
        return original(params,lmbda)
    monkeypatch.setattr(solver,'_phi_jacobian',observed)
    _,y=solver.solve([0.])
    assert len(calls) >= ds.continuation_steps
    np.testing.assert_allclose(y[0],2.0,atol=1e-8)


def test_valid_kepler_user_guess_does_not_trigger_seed_scan(monkeypatch):
    calls=[]
    def fake_solve(self,guess,lmbda_values=None):
        calls.append(np.asarray(guess).copy())
        t=np.array([self.dataset.x_start,self.dataset.x_end])
        y=np.repeat(np.asarray(guess,dtype=float)[:,None],2,axis=1)
        # This test double must provide the accepted-result metadata as well
        # as the arrays; its purpose remains counting fallback seed attempts.
        from solver import BoundaryDiagnostics
        self.last_diagnostics = BoundaryDiagnostics(
            self._boundary_residual(y[:, 0], y[:, -1], self.dataset.continuation_end))
        return t,y
    monkeypatch.setattr(ContinuationSolver,'solve',fake_solve)
    monkeypatch.setattr(ContinuationSolver,'full_orbit',
                        lambda self,state,n_points=400:(np.array([0.,1.]),
                            np.repeat(np.asarray(state)[:,None],2,axis=1)))
    worker=gui.SolverThread(get_example_tasks()[0],[2,0,-.5,.5],explicit_type='kepler')
    result={}
    worker.result_ready.connect(lambda t,y:result.update(t=t,y=y))
    worker.error.connect(lambda error:result.update(error=error))
    worker.run()
    assert 'error' not in result
    assert len(calls)==1


def test_triple_normal_path_does_not_build_expensive_fallbacks(monkeypatch):
    monkeypatch.setattr(np.random,'default_rng',
                        lambda *args,**kwargs:pytest.fail('unnecessary triple fallback scan'))
    worker=gui.SolverThread(get_example_tasks()[2],[1,0,0,.5,.5,.5],explicit_type='triple')
    result={}
    worker.result_ready.connect(lambda t,y:result.update(t=t,y=y))
    worker.error.connect(lambda error:result.update(error=error))
    worker.run()
    assert 'error' not in result, result
