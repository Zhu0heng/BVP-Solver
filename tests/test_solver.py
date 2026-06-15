import unittest
import numpy as np
from dataset import Dataset
from solver import ContinuationSolver


class TestSolver(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(
            name="sin(x)",
            x_start=0.0, x_end=12.566370614359172,
            equations=["Derivative(y(x), x, 2) + y(x) = 0"],
            boundary_conditions=["y0", "dy0 - 1"],
            n_points=200
        )

    def test_solve_sin(self):
        solver = ContinuationSolver(self.dataset)
        x, y_all = solver.solve(np.array([0.0, 1.0]))
        y = y_all[0]
        err = np.max(np.abs(y - np.sin(x)))
        self.assertLess(err, 0.01)

    def test_solve_cos(self):
        d = Dataset(
            name="cos(x)",
            x_start=0.0, x_end=12.566370614359172,
            equations=["Derivative(y(x), x, 2) + y(x) = 0"],
            boundary_conditions=["y0 - 1", "dy0"],
            n_points=200
        )
        solver = ContinuationSolver(d)
        x, y_all = solver.solve(np.array([1.0, 0.0]))
        y = y_all[0]
        err = np.max(np.abs(y - np.cos(x)))
        self.assertLess(err, 0.01)


if __name__ == '__main__':
    unittest.main()
