"""
Класс Dataset для хранения параметров краевой задачи.

Поля: name, x_start, x_end, equations, boundary_conditions, n_points,
continuation_param, continuation_start/end/steps.
Поддерживает сериализацию в/из JSON.
"""

from dataclasses import dataclass, asdict, field, fields
from typing import Optional, List
import json


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

    def __post_init__(self):
        if self.x_start >= self.x_end:
            raise ValueError(f"x_start ({self.x_start}) must be less than x_end ({self.x_end})")
        if self.n_points < 2:
            raise ValueError(f"n_points must be at least 2, got {self.n_points}")
        if self.continuation_steps < 1:
            raise ValueError(f"continuation_steps must be at least 1, got {self.continuation_steps}")
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}")
        if not self.equations:
            raise ValueError("equations must not be empty")

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
        if 'continuation_steps' in data:
            data['continuation_steps'] = int(data['continuation_steps'])
        if 'continuation_start' in data:
            data['continuation_start'] = float(data['continuation_start'])
        if 'continuation_end' in data:
            data['continuation_end'] = float(data['continuation_end'])
        if 'tol' in data:
            data['tol'] = float(data['tol'])
        # Drop unknown keys so they don't crash the dataclass constructor
        valid_keys = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)
