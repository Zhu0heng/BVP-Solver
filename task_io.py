"""
Модуль ввода-вывода: сохранение и загрузка задач в формате JSON,
а также библиотека встроенных примеров (get_example_tasks).
"""

import json
import os
from dataset import Dataset


def save_task(dataset: Dataset, filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(dataset.to_json())


def load_task(filepath: str) -> Dataset:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    required_keys = ['name', 'x_start', 'x_end', 'boundary_conditions']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required field '{key}' in task file")
    if 'equations' not in data and 'equation' not in data:
        raise KeyError("Missing required field 'equations' in task file")
    return Dataset.from_dict(data)


def get_van_der_pol_tasks() -> list:
    """Two initial-value experiments, represented by left-end conditions."""
    return [Dataset(
        name=f'Van der Pol: initial point ({x0:g}, {y0:g})',
        x_start=0.0, x_end=60.0,
        equations=[
            'Derivative(x1(t), t) - x2 = 0',
            'Derivative(x2(t), t) - ((1 - x1**2)*x2 - x1) = 0',
        ],
        boundary_conditions=[f'x1(a) = {x0:g}', f'x2(a) = {y0:g}'],
        n_points=6001, method='DOP853', tol=1e-10,
    ) for x0, y0 in [(0.1, 0.0), (4.0, 2.0)]]


def get_example_tasks() -> list:
    return [
        Dataset(
            name="Example 26.1: Two-body problem (Kepler orbit)",
            x_start=0.0, x_end=7.0,
            equations=[
                "Derivative(x1(t), t) - x3 = 0",
                "Derivative(x2(t), t) - x4 = 0",
                "Derivative(x3(t), t) + x1 / (x1**2 + x2**2)**(3/2) = 0",
                "Derivative(x4(t), t) + x2 / (x1**2 + x2**2)**(3/2) = 0",
            ],
            boundary_conditions=[
                "x1(a) = 2",
                "x2(a) = 0",
                "x1(b) = 1.0738644361",
                "x2(b) = -1.0995343576",
            ],
            n_points=200
        ),
        Dataset(
            name="Example 26.2: Limit cycles (Eckweiler system)",
            x_start=0.0, x_end=1.0,
            equations=[
                "Derivative(x1(t), t) - x3*x2 = 0",
                "Derivative(x2(t), t) + x3*(x1 - sin(x2)) = 0",
                "Derivative(x3(t), t) = 0",
                "Derivative(x4(t), t) = 0",
            ],
            boundary_conditions=[
                "x2(a) = 0",
                "x2(b) = 0",
                "x1(a) = x4(a)",
                "x1(b) = x4(b)",
            ],
            n_points=300
        ),
        Dataset(
            name="Example 26.3: Triple integrator (energy functional)",
            x_start=0.0, x_end=3.275,
            equations=[
                "Derivative(x1(t), t) - x2 = 0",
                "Derivative(x2(t), t) - x3 = 0",
                "Derivative(x3(t), t) - 0.5*(sqrt(1e-10 + (x6 + 1)**2) - sqrt(1e-10 + (x6 - 1)**2)) = 0",
                "Derivative(x4(t), t) = 0",
                "Derivative(x5(t), t) + x4 = 0",
                "Derivative(x6(t), t) + x5 = 0",
            ],
            boundary_conditions=[
                "x1(a) = 1",
                "x2(a) = 0",
                "x3(a) = 0",
                "x1(b) = 0",
                "x2(b) = 0",
                "x3(b) = 0",
            ],
            n_points=500
        ),
        Dataset(
            name="Example 26.4: Управление в форме лунки (Time-optimal, lens-shaped control)",
            x_start=0.0, x_end=1.0,
            equations=[
                "Derivative(x1(t), t) - (x2 + u1) = 0",
                "Derivative(x2(t), t) - (-1.5*x1 - 0.25*x2 + u2) = 0",
            ],
            boundary_conditions=[
                "x1(a) = 4",
                "x2(a) = 1",
                "x1(b) = 0",
                "x2(b) = 0",
            ],
            n_points=500
        ),
    ]
