"""Shared, restrained light theme for the editor and plot windows."""

BACKGROUND = '#F5F6F8'
SURFACE = '#FFFFFF'
TEXT = '#253047'
MUTED = '#697386'
BORDER = '#DDE2EA'
ACCENT = '#3865D9'

STYLESHEET = """
QWidget {
    font-family: "Segoe UI", "Microsoft YaHei UI";
    font-size: 10pt; color: #253047;
}
QMainWindow, QDialog { background: #F5F6F8; }
QFrame#headerFrame { background: #27344F; border-bottom: 1px solid #1F2B43; }
QLabel#headerTitle { color: white; font-size: 16pt; font-weight: 600; }
QLabel#headerSubtitle { color: #B7C3D9; font-size: 9pt; }
QLabel { background: transparent; }
QGroupBox {
    background: white; border: 1px solid #E2E6ED; border-radius: 8px;
    margin-top: 13px; padding: 16px 14px 12px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 14px; padding: 0 6px;
    color: #48556B; background: #F5F6F8;
}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #FAFBFC; border: 1px solid #DDE2EA; border-radius: 5px;
    padding: 5px 7px; selection-background-color: #DCE6FF;
    selection-color: #253047;
}
QLineEdit, QTextEdit, QPlainTextEdit { font-family: "Consolas"; font-size: 10pt; }
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #3865D9; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background: #F0F2F5; color: #8B94A4;
}
QPushButton {
    background: white; color: #48556B; border: 1px solid #DDE2EA;
    border-radius: 5px; padding: 7px 12px; font-weight: 500;
}
QPushButton:hover { background: #F0F4FF; border-color: #BAC9ED; }
QPushButton:pressed { background: #E4ECFF; }
QPushButton:disabled { background: #F0F2F5; color: #A1A9B7; border-color: #E5E8ED; }
QPushButton#solveBtn { background: #3865D9; color: white; border: none; font-weight: 600; }
QPushButton#solveBtn:hover { background: #2C53BB; }
QPushButton#solveBtn:disabled { background: #C4D0EB; color: white; }
QPushButton#headerBtn { color: #E4EAF5; border: 1px solid #52617B; background: transparent; }
QPushButton#headerBtn:hover { color: white; background: #354564; border-color: #7D8EAC; }
QCheckBox { spacing: 7px; background: transparent; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #CBD2DF; border-radius: 3px; background: white; }
QCheckBox::indicator:checked { background: #3865D9; border-color: #3865D9; }
QComboBox QAbstractItemView, QMenu {
    background: white; border: 1px solid #DDE2EA;
    selection-background-color: #EAF0FF; selection-color: #253047;
}
QMenu { padding: 5px; }
QMenu::item { padding: 7px 18px; }
QMenu::item:selected { background: #EAF0FF; }
QListWidget {
    background: #FAFBFC; alternate-background-color: #FAFBFC;
    border: 1px solid #E6E9EF; border-radius: 5px; outline: none;
}
QListWidget::item { padding: 7px; }
QListWidget::item:selected { background: #EAF0FF; color: #253047; }
QListWidget::item:hover { background: #F0F3F8; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #F5F6F8; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #CBD2DF; min-height: 30px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { background: white; color: #253047; border: 1px solid #DDE2EA; padding: 6px; }
"""
