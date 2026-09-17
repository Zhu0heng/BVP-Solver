"""Deterministic robustness sweeps for every user input that changes a solution.

This suite defines the supported numerical domain.  It does not claim that an
arbitrary nonlinear BVP must have a solution; unsolvable data must fail clearly.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

import gui
from dataset import Dataset
from solver import ContinuationSolver
from task_io import get_example_tasks
from control_problems import fixed_boundary_values

APP=QApplication.instance() or QApplication([])


LINEAR_CASES=[
    (-1.5, .7, .4, -2., 15., 'RK45', 1, 31),
    (-.8, -2., 1.7, 3., -20., 'RK23', 4, 73),
    (-.2, 4., 2.3, -1., 30., 'DOP853', 9, 127),
    (0., 1.25, .3, 2., -9., 'Radau', 2, 49),
    (0., -3., 1.2, -4., 11., 'BDF', 12, 101),
    (.15, .2, 2.5, 1., -25., 'LSODA', 5, 64),
    (.5, -1., .8, 4., 0., 'RK45', 20, 200),
    (1.2, .3, .45, -3., 12., 'DOP853', 3, 37),
]

_RNG=np.random.default_rng(20260917)
RANDOM_LINEAR_CASES=[(
    float(_RNG.uniform(-1.25,1.25)),
    float(_RNG.uniform(-3,3)),
    float(_RNG.uniform(.2,2.5)),
    float(_RNG.uniform(-4,4)),
    float(_RNG.uniform(-25,25)),
) for _ in range(12)]


@pytest.mark.parametrize('a,b,length,target,guess,method,steps,n_points',LINEAR_CASES)
def test_first_order_linear_bvp_sweep(a,b,length,target,guess,method,steps,n_points):
    ds=Dataset(name='linear sweep',x_start=-.25,x_end=-.25+length,
               equations=[f'Derivative(x1(t), t) - ({a}*x1 + ({b})) = 0'],
               boundary_conditions=[f'x1(b) = {target}'],method=method,
               continuation_steps=steps,n_points=n_points,tol=2e-9)
    solver=ContinuationSolver(ds)
    t,y=solver.solve([guess])
    if abs(a)<1e-14:
        initial=target-b*length
        exact=initial+b*(t-ds.x_start)
    else:
        initial=(target+b/a)*np.exp(-a*length)-b/a
        exact=(initial+b/a)*np.exp(a*(t-ds.x_start))-b/a
    np.testing.assert_allclose(y[0],exact,rtol=2e-6,atol=3e-7)
    assert abs(solver._phi(y[:,0],ds.continuation_end)[0])<2e-6
    assert len(t)==n_points


@pytest.mark.parametrize('a,b,length,target,guess',RANDOM_LINEAR_CASES)
def test_seeded_random_linear_bvp_perturbations(a,b,length,target,guess):
    ds=Dataset(name='seeded random sweep',x_start=.15,x_end=.15+length,
               equations=[f'Derivative(x1(t), t) - ({a}*x1 + ({b})) = 0'],
               boundary_conditions=[f'x1(b) = {target}'],
               continuation_steps=7,n_points=57,tol=1e-8)
    t,y=ContinuationSolver(ds).solve([guess])
    if abs(a)<1e-12:
        initial=target-b*length
        exact=initial+b*(t-ds.x_start)
    else:
        initial=(target+b/a)*np.exp(-a*length)-b/a
        exact=(initial+b/a)*np.exp(a*(t-ds.x_start))-b/a
    np.testing.assert_allclose(y[0],exact,rtol=4e-6,atol=8e-7)


@pytest.mark.parametrize('dimension',[1,2,4,6])
def test_system_dimension_sweep(dimension):
    rates=np.linspace(-.6,.6,dimension)
    initial=np.linspace(-1.5,2.,dimension)
    equations=[f'Derivative(x{i+1}(t), t) - ({rates[i]})*x{i+1} = 0'
               for i in range(dimension)]
    boundaries=[f'x{i+1}(a) = {initial[i]}' for i in range(dimension)]
    ds=Dataset(name=f'{dimension}D system',x_start=-.2,x_end=.9,
               equations=equations,boundary_conditions=boundaries,
               continuation_steps=1,n_points=67,tol=1e-8)
    t,y=ContinuationSolver(ds).solve(initial+3.)
    exact=initial[:,None]*np.exp(rates[:,None]*(t-ds.x_start))
    np.testing.assert_allclose(y,exact,rtol=4e-6,atol=6e-7)


SECOND_ORDER_CASES=[
    (0.,0.,1.,1.,[8.,-9.],'RK45'),
    (2.,-1.,.5,3.,[-4.,12.],'DOP853'),
    (-3.,2.,2.,-2.,[20.,20.],'Radau'),
    (.75,-2.,1.4,.25,[-30.,4.],'BDF'),
]


@pytest.mark.parametrize('forcing,left,length,right,guess,method',SECOND_ORDER_CASES)
def test_second_order_dirichlet_sweep(forcing,left,length,right,guess,method):
    start=.4
    ds=Dataset(name='second order sweep',x_start=start,x_end=start+length,
               equations=[f'Derivative(y(x), x, 2) - ({forcing}) = 0'],
               boundary_conditions=[f'y0 - ({left})',f'y1 - ({right})'],
               method=method,n_points=91,tol=2e-9)
    t,y=ContinuationSolver(ds).solve(guess)
    z=t-start
    slope=(right-left-.5*forcing*length**2)/length
    exact=left+slope*z+.5*forcing*z**2
    np.testing.assert_allclose(y[0],exact,rtol=2e-6,atol=5e-7)


@pytest.mark.parametrize('constant,target',[(2.,3.),(-1.5,-4.),(.125,2.5)])
def test_fixed_parameter_dictionary_sweep(constant,target):
    ds=Dataset(name='parameter dictionary',x_start=0,x_end=2,
               equations=['Derivative(x1(t), t) - k = 0'],
               boundary_conditions=[f'x1(b) = {target}'],parameters={'k':constant})
    t,y=ContinuationSolver(ds).solve([20.])
    np.testing.assert_allclose(y[0],target+constant*(t-2),atol=2e-7)


@pytest.mark.parametrize('tol',[1e-4,1e-6,1e-8,1e-10])
def test_tolerance_sweep_preserves_verified_solution(tol):
    ds=Dataset(name='tolerance sweep',x_start=0,x_end=1.5,
               equations=['Derivative(x1(t), t) + 0.4*x1 - 0.7 = 0'],
               boundary_conditions=['x1(b) = 2'],tol=tol,n_points=111)
    t,y=ContinuationSolver(ds).solve([-12.])
    initial=(2-0.7/0.4)*np.exp(0.4*1.5)+0.7/0.4
    exact=(initial-0.7/0.4)*np.exp(-0.4*t)+0.7/0.4
    np.testing.assert_allclose(y[0],exact,rtol=max(3e-5,20*tol),atol=max(1e-7,5*tol))


@pytest.mark.parametrize('start,end,steps',[(-2.,2.,5),(.1,1.7,7),(-1.,-.2,3)])
def test_continuation_range_and_step_sweep(start,end,steps):
    ds=Dataset(name='lambda sweep',x_start=0,x_end=.75,
               equations=['Derivative(x1(t), t) - lambda = 0'],
               boundary_conditions=['x1(a) = 0'],
               continuation_start=start,continuation_end=end,
               continuation_steps=steps,n_points=33)
    values,solutions=ContinuationSolver(ds).continuation()
    np.testing.assert_allclose(values,np.linspace(start,end,steps))
    np.testing.assert_allclose([y[0,-1] for _,y in solutions],values*.75,
                               rtol=2e-6,atol=2e-7)


@pytest.mark.parametrize('guess,expected',[(-8.,-2.),(-.2,-2.),(.2,2.),(9.,2.)])
def test_nonlinear_boundary_branch_follows_guess(guess,expected):
    ds=Dataset(name='two branches',x_start=0,x_end=1,
               equations=['Derivative(x1(t), t) = 0'],
               boundary_conditions=['x1(a)**2 - 4 = 0'],continuation_steps=20)
    _,y=ContinuationSolver(ds).solve([guess])
    np.testing.assert_allclose(y[0],expected,atol=2e-7)


def test_inconsistent_boundary_data_fails_instead_of_drawing_answer():
    ds=Dataset(name='no solution',x_start=0,x_end=1,
               equations=['Derivative(x1(t), t) = 0'],
               boundary_conditions=['x1(a) = 0','x1(b) = 1'])
    with pytest.raises(RuntimeError,match='Boundary'):
        ContinuationSolver(ds).solve([0.])


@pytest.mark.parametrize('field,value',[
    ('method','not-a-solver'),
    ('n_points',10.5),
    ('continuation_steps',2.5),
    ('continuation_param','mu;bad'),
    ('parameters',['not','a','dict']),
])
def test_invalid_numerical_configuration_is_rejected_early(field,value):
    kwargs=dict(name='invalid config',x_start=0,x_end=1,
                equations=['Derivative(x1(t), t) = 0'],
                boundary_conditions=['x1(a) = 0'])
    kwargs[field]=value
    with pytest.raises(ValueError):
        Dataset(**kwargs)


def _run_worker(index,guess,**kwargs):
    return _run_dataset_worker(get_example_tasks()[index],guess,**kwargs)


def _run_dataset_worker(dataset,guess,**kwargs):
    worker=gui.SolverThread(dataset,guess,**kwargs)
    result={}
    worker.result_ready.connect(lambda t,y:result.update(t=t,y=y))
    worker.error.connect(lambda error:result.update(error=error))
    worker.run()
    assert 'error' not in result,result
    return result['t'],result['y']


def _layers(result_t,result_y):
    return result_t if isinstance(result_t,list) else [(result_t,result_y)]


def _default_example_run(index,dataset=None,mu=1.):
    dataset=dataset or get_example_tasks()[index]
    guesses={0:[2.,0.,-.5,.5],1:[6.5,0.,2*np.pi,6.5],
             2:[1.,0.,0.,.5,.5,.5],3:[-.5,-.1,4.]}
    kwargs={'explicit_type':('kepler','limit_cycle','triple','lens')[index]}
    if index==3:
        kwargs['smooth_param_list']=[mu]
    return _run_dataset_worker(dataset,guesses[index],**kwargs)


def _validate_example(index,dataset,result_t,result_y):
    layers=_layers(result_t,result_y)
    for entry in layers:
        t,values=entry[:2]
        assert np.isfinite(t).all() and np.isfinite(values).all()
        if index<3:
            solver=ContinuationSolver(dataset)
            state=values[:(solver.n_dep if solver.order==1 else 2*solver.n_dep),0]
            parameter=float((dataset.parameters or {}).get(
                dataset.continuation_param,dataset.continuation_end))
            assert np.max(np.abs(solver._phi(state,parameter)))<3e-6
        else:
            _,target=fixed_boundary_values(dataset,(1,2))
            np.testing.assert_allclose(values[:2,-1],target,atol=3e-6)
            assert abs(np.sum(values[2:4,-1]**2)-1)<3e-6
            assert t[-1]>0
    return layers


def _apply_example_variant(dataset,variant):
    if variant.startswith('gravity_'):
        factor=float(variant.split('_')[1])/100
        dataset.equations[2]=dataset.equations[2].replace('+ x1 /',f'+ {factor}*x1 /')
        dataset.equations[3]=dataset.equations[3].replace('+ x2 /',f'+ {factor}*x2 /')
    elif variant=='kepler_target_plus':
        dataset.boundary_conditions[2]='x1(b) = 1.12'
    elif variant=='kepler_target_minus':
        dataset.boundary_conditions[3]='x2(b) = -1.04'
    elif variant.startswith('kepler_time_'):
        dataset.x_end=float(variant.rsplit('_',1)[1])/10
    elif variant.startswith('sin_'):
        factor=float(variant.split('_')[1])/100
        dataset.equations[1]=dataset.equations[1].replace('sin(x2)',f'{factor}*sin(x2)')
    elif variant.startswith('cycle_time_'):
        dataset.x_end=float(variant.rsplit('_',1)[1])/100
    elif variant.startswith('chain_'):
        factor=float(variant.split('_')[1])/100
        dataset.equations[0]=f'Derivative(x1(t), t) - {factor}*x2 = 0'
        dataset.equations[4]=f'Derivative(x5(t), t) + {factor}*x4 = 0'
    elif variant.startswith('triple_initial_'):
        value=float(variant.rsplit('_',1)[1])/100
        dataset.boundary_conditions[0]=f'x1(a) = {value}'
    elif variant.startswith('triple_time_'):
        dataset.x_end=float(variant.rsplit('_',1)[1])/100
    elif variant.startswith('beta_'):
        value=float(variant.split('_')[1])/100
        dataset.equations[1]=dataset.equations[1].replace('1.5',str(value))
    elif variant.startswith('gain_'):
        value=float(variant.split('_')[1])/100
        dataset.equations[0]=dataset.equations[0].replace('u1',f'{value}*u1')
    elif variant=='lens_target_plus':
        dataset.boundary_conditions[2]='x1(b) = 0.05'
    elif variant=='lens_target_minus':
        dataset.boundary_conditions[2]='x1(b) = -0.05'
    else:
        raise AssertionError(variant)


EXAMPLE_VARIANTS=[
    (0,'gravity_95'),(0,'gravity_105'),
    (0,'kepler_target_plus'),(0,'kepler_target_minus'),
    (0,'kepler_time_68'),(0,'kepler_time_72'),
    (1,'sin_90'),(1,'sin_110'),
    (1,'cycle_time_95'),(1,'cycle_time_105'),
    (2,'chain_105'),(2,'chain_110'),
    (2,'triple_initial_95'),(2,'triple_initial_105'),
    (2,'triple_time_335'),(2,'triple_time_345'),
    (3,'beta_140'),(3,'beta_160'),
    (3,'gain_80'),(3,'gain_120'),
    (3,'lens_target_plus'),(3,'lens_target_minus'),
]


@pytest.fixture(scope='module')
def default_example_layers():
    result={}
    for index in range(4):
        t,y=_default_example_run(index)
        result[index]=_validate_example(index,get_example_tasks()[index],t,y)
    return result


@pytest.mark.parametrize('index,variant',EXAMPLE_VARIANTS,
                         ids=[f'26.{index+1}-{variant}' for index,variant in EXAMPLE_VARIANTS])
def test_each_example_data_family_has_multiple_working_variations(
        index,variant,default_example_layers):
    dataset=get_example_tasks()[index]
    _apply_example_variant(dataset,variant)
    t,y=_default_example_run(index,dataset)
    layers=_validate_example(index,dataset,t,y)
    baseline=default_example_layers[index]
    assert len(layers)==len(baseline)
    difference=max(np.max(np.abs(entry[1]-base[1]))
                   for entry,base in zip(layers,baseline))
    assert difference>1e-5,(index,variant,difference)


@pytest.mark.parametrize('mu',[.5,.1,.01,1e-4])
def test_lens_smoothing_parameter_sweep(mu):
    dataset=get_example_tasks()[3]
    t,y=_default_example_run(3,dataset,mu=mu)
    _validate_example(3,dataset,t,y)


@pytest.mark.parametrize('variant',['chain_90','triple_time_310'])
def test_unreachable_triple_data_fails_fast_with_reason(variant):
    dataset=get_example_tasks()[2]
    _apply_example_variant(dataset,variant)
    worker=gui.SolverThread(dataset,[1.,0.,0.,.5,.5,.5],explicit_type='triple')
    errors=[]
    worker.error.connect(errors.append)
    worker.run()
    assert errors and 'unreachable' in errors[0]


@pytest.mark.parametrize('velocity',[(-.5,.5),(.5,-.5),(0.,.5)])
def test_kepler_multiple_velocity_guesses(velocity):
    t,y=_run_worker(0,[2.,0.,*velocity],explicit_type='kepler')
    layer=t[0] if isinstance(t,list) else (t,y)
    _,values=layer[:2]
    np.testing.assert_allclose(values[:2,0],[2,0],atol=2e-8)
    np.testing.assert_allclose(values[:2,-1],[1.0738644361,-1.0995343576],atol=2e-6)
    assert np.isfinite(values).all()


@pytest.mark.parametrize('amplitude',[2.,6.5,9.])
def test_limit_cycle_multiple_amplitude_guesses(amplitude):
    t,y=_run_worker(1,[amplitude,0.,2*np.pi,amplitude],
                    explicit_type='limit_cycle',multi_cycle=False)
    layer=t[0] if isinstance(t,list) else (t,y)
    _,values=layer[:2]
    assert abs(values[1,0])<2e-6 and abs(values[1,-1])<2e-6
    assert abs(values[0,0]-values[3,0])<2e-6
    assert abs(values[0,-1]-values[3,-1])<2e-6
    assert abs(values[0,0])>1e-3


@pytest.mark.parametrize('costates',[(.5,.5,.5),(-2.,5.,-2.),(-3.,-5.,-3.)])
def test_triple_multiple_costate_guesses(costates):
    _,y=_run_worker(2,[1.,0.,0.,*costates],explicit_type='triple')
    np.testing.assert_allclose(y[:3,0],[1,0,0],atol=2e-8)
    np.testing.assert_allclose(y[:3,-1],[0,0,0],atol=2e-6)
    assert np.max(np.abs(y[6]))<=1+1e-8


@pytest.mark.parametrize('guess',[(-.5,-.1,4.),(0.,0.,1.),(10.,10.,.1)])
def test_lens_multiple_costate_and_time_guesses(guess):
    t,y=_run_worker(3,list(guess),explicit_type='lens',smooth_param_list=[1.])
    np.testing.assert_allclose(y[:2,0],[4,1],atol=2e-8)
    np.testing.assert_allclose(y[:2,-1],[0,0],atol=2e-6)
    assert t[-1]>0 and abs(np.sum(y[2:4,-1]**2)-1)<2e-6


@pytest.mark.parametrize('n_points',[25,100,333])
def test_plot_data_tracks_sampling_changes(n_points):
    t=np.linspace(0,2,n_points)
    y=np.array([np.sin(t),np.cos(t)])
    plot=gui.PlotWindow(t,y,['x1','x2'],f'n={n_points}',problem_type='custom')
    try:
        line=plot.canvas.axes.lines[0]
        np.testing.assert_array_equal(line.get_xdata(),t)
        np.testing.assert_array_equal(line.get_ydata(),y[0])
    finally:
        plot.close()


def test_plot_axis_and_variable_selection_uses_current_solution_arrays():
    t=np.linspace(0,1,75)
    y=np.array([t,2*t+1,np.sin(t)])
    plot=gui.PlotWindow(t,y,['x1','x2','x3'],'axis selection',problem_type='custom')
    try:
        plot._x_idx=plot._x_items.index('x2')
        plot._y_indices=[plot._y_items.index('x3')]
        plot._redraw()
        line=plot.canvas.axes.lines[0]
        np.testing.assert_array_equal(line.get_xdata(),y[1])
        np.testing.assert_array_equal(line.get_ydata(),y[2])

        plot._x_idx=plot._x_items.index('t')
        plot._y_indices=[plot._y_items.index('x1'),plot._y_items.index('x2')]
        plot._redraw()
        np.testing.assert_array_equal(plot.canvas.axes.lines[0].get_ydata(),y[0])
        np.testing.assert_array_equal(plot.canvas.axes.lines[1].get_ydata(),y[1])
    finally:
        plot.close()
