"""
Точка входа приложения. Запускает главное окно PyQt5.
"""

import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from gui import MainWindow


def main():
    plt.style.use('seaborn-v0_8-darkgrid')
    
    app = QApplication(sys.argv)
    app.setApplicationName("Метод продолжения по параметру")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
