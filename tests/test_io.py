import unittest
import tempfile
import os
from dataset import Dataset
from task_io import save_task, load_task


class TestIO(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(
            name="sin(x)",
            x_start=0.0, x_end=12.566370614359172,
            equations=["Derivative(y(x), x, 2) + y(x) = 0"],
            boundary_conditions=["y0", "dy0 - 1"],
            n_points=200
        )

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        try:
            save_task(self.dataset, filepath)
            loaded = load_task(filepath)
            self.assertEqual(loaded.name, self.dataset.name)
            self.assertEqual(loaded.equations, self.dataset.equations)
            self.assertEqual(loaded.boundary_conditions, self.dataset.boundary_conditions)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


if __name__ == '__main__':
    unittest.main()
