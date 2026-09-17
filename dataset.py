"""
Класс Dataset для хранения параметров краевой задачи.

Поля: name, x_start, x_end, equations, boundary_conditions, n_points,
continuation_param, continuation_start/end/steps.
Поддерживает сериализацию в/из JSON.
"""

from dataclasses import dataclass, asdict, field, fields
from typing import Optional, List
import math
import json
import re


SUPPORTED_METHODS = {'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'}


@dataclass
class Dataset:
    name: str
    x_start: float
    x_end: float
    equations: List[str] = field(default_factory=list)
    boundary_conditions: List[str] = field(default_factory=list)
    parameters: Optional[dict] = None
    n_points: int = 100
    continuation_param: str = 'lambda'
    continuation_start: float = 0.0
    continuation_end: float = 1.0
    continuation_steps: int = 10
    tol: float = 1e-9          # требуемая точность (ε) интегрирования и коррекции
    method: str = 'RK45'       # метод численного интегрирования solve_ivp
    initial_guess: Optional[List[float]] = None

    def __post_init__(self):
        for name, value in (
                ('x_start', self.x_start), ('x_end', self.x_end),
                ('tol', self.tol), ('continuation_start', self.continuation_start),
                ('continuation_end', self.continuation_end)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.x_start >= self.x_end:
            raise ValueError(f"x_start ({self.x_start}) must be less than x_end ({self.x_end})")
        if isinstance(self.n_points, bool) or not isinstance(self.n_points, int):
            raise ValueError(f"n_points must be an integer, got {self.n_points!r}")
        if self.n_points < 2:
            raise ValueError(f"n_points must be at least 2, got {self.n_points}")
        if (isinstance(self.continuation_steps, bool)
                or not isinstance(self.continuation_steps, int)):
            raise ValueError(
                f"continuation_steps must be an integer, got {self.continuation_steps!r}")
        if self.continuation_steps < 1:
            raise ValueError(f"continuation_steps must be at least 1, got {self.continuation_steps}")
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}")
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {sorted(SUPPORTED_METHODS)}, got {self.method!r}")
        if not isinstance(self.continuation_param, str) or not re.fullmatch(
                r'[A-Za-z_]\w*', self.continuation_param):
            raise ValueError(f"continuation_param must be a valid identifier, got {self.continuation_param!r}")
        if self.parameters is not None and not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a dictionary or None")
        if not self.equations:
            raise ValueError("equations must not be empty")
        if self.initial_guess is not None:
            self.initial_guess = [float(value) for value in self.initial_guess]
            if not all(math.isfinite(value) for value in self.initial_guess):
                raise ValueError("initial_guess values must be finite")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Dataset':
        data = dict(data)
        if 'equation' in data and 'equations' not in data:
            data['equations'] = [data.pop('equation')]
        # Coerce types that JSON may misrepresent
        if 'n_points' in data:
            data['n_points'] = int(data['n_points'])
        if 'x_start' in data:
            data['x_start'] = float(data['x_start'])
        if 'x_end' in data:
            data['x_end'] = float(data['x_end'])
        if 'continuation_steps' in data:
            data['continuation_steps'] = int(data['continuation_steps'])
        if 'continuation_start' in data:
            data['continuation_start'] = float(data['continuation_start'])
        if 'continuation_end' in data:
            data['continuation_end'] = float(data['continuation_end'])
        if 'tol' in data:
            data['tol'] = float(data['tol'])
        if data.get('initial_guess') is not None:
            data['initial_guess'] = [float(value) for value in data['initial_guess']]
        # Drop unknown keys so they don't crash the dataclass constructor
        valid_keys = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)
