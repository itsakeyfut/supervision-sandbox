import sys

from PySide6.QtWidgets import QApplication

from attention_monitor.config import Config
from attention_monitor.settings import load_into
from attention_monitor.ui import MainWindow


def main():
    app = QApplication(sys.argv)
    config = Config()
    load_into(config)
    window = MainWindow(config)
    window.show()
    window.start_capture()
    sys.exit(app.exec())
