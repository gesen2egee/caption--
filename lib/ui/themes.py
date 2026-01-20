# -*- coding: utf-8 -*-
"""
UI 主題樣式

定義應用程式的主題 CSS 樣式。
"""

THEME_STYLES = {
    "light": "",  # Use system default
    "dark": """
        QMainWindow, QDialog {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        QWidget {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        QPlainTextEdit, QLineEdit, QTextEdit {
            background-color: #252526;
            color: #cccccc;
            border: 1px solid #3e3e42;
        }
        QPushButton {
            background-color: #333333;
            color: #d4d4d4;
            border: 1px solid #444444;
            padding: 5px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #444444;
            border: 1px solid #666666;
        }
        QPushButton:pressed {
            background-color: #222222;
        }
        QTabWidget::pane {
            border: 1px solid #3e3e42;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #969696;
            padding: 8px 15px;
            border: 1px solid #3e3e42;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        QScrollArea {
            border: 1px solid #3e3e42;
            background-color: #1e1e1e;
        }
        QLabel {
            color: #d4d4d4;
        }
        QGroupBox {
            border: 1px solid #3e3e42;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QStatusBar {
            background-color: #007acc;
            color: white;
        }
        QSplitter::handle {
            background-color: #3e3e42;
        }
    """
}


def get_theme_style(theme_name: str) -> str:
    """取得主題樣式"""
    return THEME_STYLES.get(theme_name, "")


def apply_theme(widget, theme_name: str):
    """套用主題到 widget"""
    style = get_theme_style(theme_name)
    widget.setStyleSheet(style)
