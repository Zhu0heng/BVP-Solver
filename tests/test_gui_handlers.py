"""
Тесты логики интерфейса (область «g»):

  * нажатия кнопок вызывают ПРАВИЛЬНЫЕ обработчики (проверка подключений);
  * обработчики результата/ошибки приводят интерфейс в нужное состояние;
  * исключения в start_solve перехватываются (кнопка снова активна, статус
    «error», без падения);
  * вспомогательные функции (detect_problem_type, _solution_is_physical,
    set_dataset_to_ui, _get_varnames) работают корректно.

Qt работает в offscreen-режиме (без дисплея); диалоги QMessageBox заглушены.
"""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from context import *  # noqa: F401, F403
from PyQt5.QtWidgets import QApplication, QMessageBox

_app = QApplication.instance() or QApplication([])

# Заглушаем модальные диалоги, чтобы тесты не блокировались.
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.critical = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)

import gui  # noqa: E402
from task_io import get_example_tasks  # noqa: E402


@pytest.fixture
def win():
    w = gui.MainWindow()
    yield w
    try:
        w.close()
    except Exception:
        pass


# ───────────────────────── чистые функции модуля ─────────────────────────
class TestModuleFunctions:
    def test_detect_problem_type(self):
        ex = get_example_tasks()
        assert gui.detect_problem_type(ex[0]) == 'kepler'
        assert gui.detect_problem_type(ex[1]) == 'limit_cycle'
        assert gui.detect_problem_type(ex[2]) == 'triple'
        assert gui.detect_problem_type(ex[3]) == 'lens'
        assert gui.detect_problem_type(None) == 'custom'

    def test_solution_is_physical(self):
        assert gui._solution_is_physical(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert gui._solution_is_physical(np.array([500.0]), ref_scale=1.0)
        assert not gui._solution_is_physical(np.array([np.inf]))
        assert not gui._solution_is_physical(np.array([np.nan]))
        assert not gui._solution_is_physical(np.array([]))
        assert not gui._solution_is_physical(np.array([1e9]), ref_scale=1.0)


# ─────────────────────── подключения кнопок к слотам ───────────────────────
class TestButtonWiring:
    def test_each_button_calls_its_handler(self):
        handlers = ['start_solve', 'show_plot_window', 'save_task',
                    'clear_ui', 'load_task', 'show_examples']
        calls = {}
        orig = {h: getattr(gui.MainWindow, h) for h in handlers}

        def make(name):
            def stub(self, *a, **k):
                calls[name] = True
            return stub

        for h in handlers:
            setattr(gui.MainWindow, h, make(h))
        try:
            w = gui.MainWindow()
            w.solve_btn.click()
            w.graph_btn.click()
            w.export_btn.click()
            w.clear_btn.click()
            w.load_btn.click()
            w.example_btn.click()
            w.close()
        finally:
            for h in handlers:
                setattr(gui.MainWindow, h, orig[h])

        assert calls.get('start_solve') is True
        assert calls.get('show_plot_window') is True
        assert calls.get('save_task') is True
        assert calls.get('clear_ui') is True
        assert calls.get('load_task') is True
        assert calls.get('show_examples') is True


# ───────────────────────── set_dataset_to_ui ─────────────────────────
class TestSetDatasetToUI:
    def test_limit_cycle_population(self, win):
        win.set_dataset_to_ui(get_example_tasks()[1])
        assert win.guess_edit.text() == '2, 6.5, 9'
        assert win.multi_cycle_cb.isChecked() is True
        bc_lines = [l for l in win.bc_edit.toPlainText().split('\n') if l.strip()]
        assert len(bc_lines) == 4

    def test_non_limit_cycle_unchecks_multi(self, win):
        win.set_dataset_to_ui(get_example_tasks()[0])   # Kepler
        assert win.multi_cycle_cb.isChecked() is False
        assert win.name_edit.text() == get_example_tasks()[0].name


class TestGetVarnames:
    def test_varnames_by_problem_type(self, win):
        win._last_x = None
        win._last_y = np.zeros((4, 10))
        win._last_problem_type = 'limit_cycle'
        assert win._get_varnames() == ['x1', 'x2', 'T', 'x3']
        win._last_problem_type = 'kepler'
        assert win._get_varnames() == ['x1', 'x2', 'x3', 'x4']
        win._last_y = np.zeros((2, 10))
        win._last_problem_type = 'custom'
        assert win._get_varnames() == ['x1', 'x2']


# ─────────────────────── обработчики результата/ошибки ───────────────────────
class TestResultHandlers:
    def test_on_solve_finished_sets_state(self, win):
        win.current_dataset = get_example_tasks()[0]
        win._last_problem_type = 'kepler'
        win.solve_btn.setEnabled(False)
        x = np.linspace(0, 7, 40)
        y = np.zeros((4, 40))
        win.on_solve_finished(x, y)
        assert win.solve_btn.isEnabled() is True
        assert win._status_state == 'done'
        assert win._last_x is x
        assert win._last_y is y

    def test_on_solve_error_sets_state(self, win):
        win.solve_btn.setEnabled(False)
        win.on_solve_error("искусственная ошибка решателя")
        assert win.solve_btn.isEnabled() is True
        assert win._status_state == 'error'

    def test_start_solve_handles_prep_exception(self, win):
        # Заставляем подготовку упасть → исключение должно быть перехвачено.
        def boom():
            raise RuntimeError("prep fail")
        win.get_dataset_from_ui = boom
        win.solve_btn.setEnabled(True)
        win.start_solve()                       # не должно бросать наружу
        assert win.solve_btn.isEnabled() is True
        assert win._status_state == 'error'


# ─────────────────── полный путь «кнопка → решение → результат» ───────────────────
class TestFullSolvePath:
    def test_solve_limit_cycle_through_gui(self, win, monkeypatch):
        # Поток решателя выполняется синхронно (start = run).
        monkeypatch.setattr(gui.SolverThread, 'start', gui.SolverThread.run)
        win._ask_mu_values = lambda *a, **k: [1e-6]
        win.set_dataset_to_ui(get_example_tasks()[1])     # Example 26.2
        win.sol_method_combo.setCurrentIndex(0)           # auto-detect
        win.start_solve()
        assert isinstance(win._last_x, list)
        assert len(win._last_x) == 3                       # три предельных цикла
        assert win.solve_btn.isEnabled() is True
        assert win._status_state == 'done'
