import sys

from PySide6.QtWidgets import QApplication

from attention_monitor.config import Config
from attention_monitor.ui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow(Config())
    window.show()
    window.start_capture()
    sys.exit(app.exec())
