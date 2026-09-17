"""End-to-end numerical checks after editing all four example families."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from context import *
from PyQt5.QtWidgets import QApplication, QMessageBox
import gui
from task_io import get_example_tasks
from solver import ContinuationSolver

APP=QApplication.instance() or QApplication([])


def run_example(index, change=None):
    ds=get_example_tasks()[index]
    if change:
        change(ds)
    # Select the same defaults as loading the dataset in the real window.
    window=gui.MainWindow()
    window.set_dataset_to_ui(ds)
    errors=[]
    window.on_solve_error=lambda message: errors.append(message)
    window._ask_mu_values=lambda *a,**k: [1.,.1,1e-6]
    window._save_history=lambda: None
    window.start_solve()
    # Tests replace thread dispatch with run(), retaining the genuine solver and signals.
    if errors:
        window._close_plot_windows()
        window.deleteLater()
        raise AssertionError(errors)
    assert window._status_state=='done'
    layers=window._last_x if isinstance(window._last_x,list) else [(window._last_x,window._last_y)]
    layers=[(entry[0].copy(),entry[1].copy()) for entry in layers]
    plot=window._plot_windows[0]
    # Every phase/time curve visible in the default chart must come from this result.
    for t,y in layers:
        row=1 if index in (0,1) else 0
        assert any(np.array_equal(line.get_ydata(),y[row]) for line in plot.canvas.axes.lines)
    window._close_plot_windows()
    window.deleteLater()
    return ds,layers


@pytest.fixture(autouse=True)
def synchronous_workers(monkeypatch):
    monkeypatch.setattr(gui.SolverThread,'start',gui.SolverThread.run)
    monkeypatch.setattr(QMessageBox,'critical',lambda *a: pytest.fail(str(a[-1])))


@pytest.fixture(scope='module')
def baseline_cache():
    return {}


def endpoint_error(ds,layers,index):
    errors=[]
    for t,y in layers:
        if index != 3:
            solver=ContinuationSolver(ds)
            initial=y[:len(ds.equations),0]
            sol=solve_ivp(solver._ode_system,[ds.x_start,ds.x_end],initial,args=(0,),
                          method='DOP853',rtol=2e-12,atol=2e-14,max_step=.01)
            assert sol.success
            ye=sol.y[:,-1]
            # Independent integrator, then evaluate the entered endpoint constraints.
            args=[v for pair in zip(initial,ye) for v in pair]
            errors.append(max(abs(float(bc(*args,0))) for bc in solver.bc_funcs))
        else:
            beta=1.8 if '1.8' in ds.equations[1] else 1.5
            gain=2. if '2*u1' in ds.equations[0] else 1.
            A=np.array([[0.,1.],[-beta,-.25]])
            B=np.diag([gain,1.])
            mu=[1.,.1,1e-6][len(errors)]
            psi0=y[2:4,0]
            # Numerically differentiate the BOOK support function, independently of LensDynamics.
            def support(z):
                p,q=z
                q1=(np.sqrt(mu*(p*p+q*q)+(p+q)**2)+np.sqrt(mu*(p*p+q*q)+(p-q)**2))/2
                return np.sqrt(2*(q1*q1+q*q))-q1
            def ode(time,state):
                sigma=B.T@expm(-A.T*time)@psi0
                u=np.array([np.imag(support(sigma.astype(complex)+1e-25j*np.eye(2)[k]))/1e-25 for k in range(2)])
                return A@state+B@u
            sol=solve_ivp(ode,[0,t[-1]],y[:2,0],method='DOP853',rtol=1e-11,atol=1e-13,max_step=.02)
            assert sol.success
            target=np.array([.1,0]) if 'x1(b) = 0.1' in ds.boundary_conditions else np.zeros(2)
            errors.append(max(np.max(abs(sol.y[:,-1]-target)),abs(np.sum(y[2:4,-1]**2)-1)))
    return max(errors)


CHANGES = [
    (0,'coefficient',lambda d: d.equations.__setitem__(2,d.equations[2].replace('+ x1 /','+ 1.1*x1 /'))),
    (0,'boundary',lambda d: d.boundary_conditions.__setitem__(2,'x1(b) = 1.2')),
    (0,'interval',lambda d: setattr(d,'x_end',6.5)),
    (1,'coefficient',lambda d: d.equations.__setitem__(1,d.equations[1].replace('sin(x2)','0.8*sin(x2)'))),
    (1,'interval',lambda d: setattr(d,'x_end',1.1)),
    (2,'coefficient',lambda d: (d.equations.__setitem__(0,'Derivative(x1(t), t) - 1.1*x2 = 0'),d.equations.__setitem__(4,'Derivative(x5(t), t) + 1.1*x4 = 0'))),
    (2,'boundary',lambda d: d.boundary_conditions.__setitem__(0,'x1(a) = 0.9')),
    (2,'interval',lambda d: setattr(d,'x_end',3.5)),
    (3,'coefficient',lambda d: d.equations.__setitem__(1,d.equations[1].replace('1.5','1.8'))),
    (3,'control_gain',lambda d: d.equations.__setitem__(0,d.equations[0].replace('u1','2*u1'))),
    (3,'boundary',lambda d: d.boundary_conditions.__setitem__(2,'x1(b) = 0.1')),
]


@pytest.mark.parametrize('index,name,change',CHANGES,ids=[f'26.{i+1}-{name}' for i,name,_ in CHANGES])
def test_example_changes_affect_verified_results(index,name,change,baseline_cache):
    if index not in baseline_cache:
        ds,base=run_example(index)
        assert endpoint_error(ds,base,index)<2e-6
        baseline_cache[index]=base
    base=baseline_cache[index]
    ds,layers=run_example(index,change)
    assert endpoint_error(ds,layers,index)<2e-6
    assert len(layers)==len(base)
    delta=max(np.max(abs(y-y0)) for (_,y),(_,y0) in zip(layers,base))
    assert delta>1e-4, (index,name,delta)
    if name=='interval':
        assert layers[0][0][-1] != base[0][0][-1]


def test_lens_sample_count_changes_output_not_the_physical_solution(baseline_cache):
    if 3 not in baseline_cache:
        baseline_cache[3]=run_example(3)[1]
    ds,layers=run_example(3,lambda d:setattr(d,'n_points',120))
    for (t,y),(t0,y0) in zip(layers,baseline_cache[3]):
        assert len(t)==120
        np.testing.assert_allclose(y[:,0],y0[:,0],atol=2e-6)
        np.testing.assert_allclose(y[:,-1],y0[:,-1],atol=2e-6)


def test_lens_interval_is_explicitly_dimensionless():
    win=gui.MainWindow()
    win.set_dataset_to_ui(get_example_tasks()[3])
    assert win.x_start_spin.value()==0 and win.x_end_spin.value()==1
    assert not win.x_start_spin.isEnabled() and not win.x_end_spin.isEnabled()
    win.deleteLater()
    ds=get_example_tasks()[3]
    ds.x_end=5
    worker=gui.SolverThread(ds)
    errors=[]
    worker.error.connect(errors.append)
    worker.run()
    assert errors and '[0, 1]' in errors[0]


def test_triple_parameter_selection_updates_both_trajectory_and_control():
    ds=get_example_tasks()[2]
    ds.continuation_param='mu'
    ds.equations[2]=ds.equations[2].replace('- 0.5*','- mu*0.5*')
    worker=gui.SolverThread(ds,[1,0,0,.5,.5,.5],smooth_param_list=[1.,1.1])
    result={}
    worker.result_ready.connect(lambda t,y: result.update(t=t,y=y))
    worker.error.connect(lambda error: pytest.fail(error))
    worker.run()
    assert len(result['t'])==2
    (t0,y0), (t1,y1)=[entry[:2] for entry in result['t']]
    assert np.max(abs(y0[:3]-y1[:3]))>1e-3
    for value,(t,y,*_) in zip([1.,1.1],result['t']):
        np.testing.assert_allclose(y[:3,-1],0,atol=2e-6)
        u=value*.5*(np.sqrt(1e-10+(y[5]+1)**2)-np.sqrt(1e-10+(y[5]-1)**2))
        np.testing.assert_allclose(y[6],u,atol=1e-12)
