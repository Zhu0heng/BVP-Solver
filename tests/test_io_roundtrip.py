"""
Тесты ввода-вывода (область «f»):

  * save_task / load_task сохраняют и БЕЗ ПОТЕРЬ восстанавливают задачу;
  * Dataset.to_dict / from_dict / to_json — полный круг;
  * отсутствие обязательных полей и файла обрабатывается корректно;
  * поддерживается псевдоним 'equation' (единственное число).
"""

import os
import json
import tempfile
import pytest
from context import *  # noqa: F401, F403
from dataset import Dataset
from task_io import save_task, load_task, get_example_tasks


@pytest.fixture
def tmp_json():
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


class TestRoundTrip:
    @pytest.mark.parametrize("idx", [0, 1, 2, 3])
    def test_examples_lossless(self, idx, tmp_json):
        original = get_example_tasks()[idx]
        save_task(original, tmp_json)
        loaded = load_task(tmp_json)
        # полное равенство всех полей (без потерь)
        assert loaded.to_dict() == original.to_dict()

    def test_all_fields_preserved(self, tmp_json):
        original = Dataset(
            name="full", x_start=-1.5, x_end=2.5,
            equations=["Derivative(x1(t), t) - x2 = 0",
                       "Derivative(x2(t), t) + x1 = 0"],
            boundary_conditions=["x1(a) = 0", "x2(a) = 1"],
            parameters={"mu": 0.7}, n_points=321,
            continuation_param="mu", continuation_start=0.1,
            continuation_end=0.9, continuation_steps=15,
            tol=1e-7, method="Radau",
        )
        save_task(original, tmp_json)
        loaded = load_task(tmp_json)
        for fld in ["name", "x_start", "x_end", "equations",
                    "boundary_conditions", "parameters", "n_points",
                    "continuation_param", "continuation_start",
                    "continuation_end", "continuation_steps", "tol", "method"]:
            assert getattr(loaded, fld) == getattr(original, fld), fld

    def test_to_dict_from_dict(self):
        original = get_example_tasks()[0]
        clone = Dataset.from_dict(original.to_dict())
        assert clone.to_dict() == original.to_dict()

    def test_to_json_is_valid_json(self):
        original = get_example_tasks()[1]
        text = original.to_json()
        data = json.loads(text)          # должно парситься как JSON
        assert data["name"] == original.name
        assert data["equations"] == original.equations


class TestIOErrors:
    def test_missing_required_key(self, tmp_json):
        with open(tmp_json, 'w', encoding='utf-8') as f:
            json.dump({"name": "x", "x_start": 0.0}, f)   # нет x_end, BC, eqs
        with pytest.raises(KeyError):
            load_task(tmp_json)

    def test_missing_equations_key(self, tmp_json):
        with open(tmp_json, 'w', encoding='utf-8') as f:
            json.dump({"name": "x", "x_start": 0.0, "x_end": 1.0,
                       "boundary_conditions": ["y0", "y1"]}, f)
        with pytest.raises(KeyError):
            load_task(tmp_json)

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_task(os.path.join(tempfile.gettempdir(),
                                   "_definitely_missing_12345.json"))

    def test_equation_singular_alias(self, tmp_json):
        # старый формат: 'equation' (ед.ч.) → должно стать списком equations
        with open(tmp_json, 'w', encoding='utf-8') as f:
            json.dump({"name": "old", "x_start": 0.0, "x_end": 1.0,
                       "equation": "Derivative(y(x), x, 2) = 0",
                       "boundary_conditions": ["y0", "y1 - 1"]}, f)
        loaded = load_task(tmp_json)
        assert loaded.equations == ["Derivative(y(x), x, 2) = 0"]

    def test_unknown_keys_dropped(self):
        d = Dataset.from_dict({
            "name": "x", "x_start": 0.0, "x_end": 1.0,
            "equations": ["Derivative(y(x), x, 2) = 0"],
            "boundary_conditions": ["y0", "y1"],
            "bogus_field": 123, "another": "ignored",
        })
        assert d.name == "x"
        assert not hasattr(d, "bogus_field")
