"""
Точка входа приложения. Запускает главное окно PyQt5.
"""

import sys


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Boundary-value problem solver')
    parser.add_argument('--defense', action='store_true',
                        help='Run the Van der Pol experiment and save its phase portrait')
    parser.add_argument('--output-dir', default='defense_artifacts',
                        help='Directory for defense experiment results')
    args, qt_args = parser.parse_known_args()
    if args.defense:
        if qt_args:
            parser.error('Unrecognised arguments: ' + ' '.join(qt_args))
        from defense_experiment import run_experiment
        run_experiment(args.output_dir)
        return
    import matplotlib.pyplot as plt
    from PyQt5.QtWidgets import QApplication
    from gui import MainWindow
    plt.style.use('seaborn-v0_8-darkgrid')
    
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Метод продолжения по параметру")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
