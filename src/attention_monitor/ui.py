from __future__ import annotations

import time

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from attention_monitor import settings
from attention_monitor.capture import WebcamSource
from attention_monitor.config import Config
from attention_monitor.pipeline import Pipeline


def frame_to_qimage(frame):
    """BGR ndarray を、numpy バッファから切り離した RGB888 の QImage に変換する。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return image.copy()


class CaptureThread(QThread):
    """撮影＋推論を回し、処理済みフレームを frameReady で送るワーカースレッド。"""

    frameReady = Signal(QImage)
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._running = True
        self.pipeline = None

    def stop(self):
        self._running = False

    def request_calibration(self):
        if self.pipeline is not None:
            self.pipeline.request_calibration()

    def apply_settings(self):
        if self.pipeline is not None:
            self.pipeline.apply_settings()

    def run(self):
        source = None
        try:
            source = WebcamSource(self._config.camera_index)
            self.pipeline = Pipeline(self._config)
        except Exception as exc:  # カメラ不可・モデル読込失敗など
            if source is not None:
                source.release()
            self.error.emit(str(exc))
            return

        prev = time.perf_counter()
        fps = 0.0
        try:
            while self._running:
                frame = source.read()
                if frame is None:
                    break
                now = time.perf_counter()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
                out = self.pipeline.process(frame, dt, fps)
                self.frameReady.emit(frame_to_qimage(out))
        finally:
            source.release()


class SettingsDialog(QDialog):
    """しきい値を実行時調整するモーダレスダイアログ。閉じる時に保存。"""

    _SPECS = [
        ("yaw_center_threshold", "yaw 閾値 (°)", 0.0, 90.0, 1.0, 1),
        ("pitch_center_threshold", "pitch 閾値 (°)", 0.0, 90.0, 1.0, 1),
        ("gaze_threshold", "視線 閾値", 0.0, 1.0, 0.01, 2),
        ("ema_alpha", "EMA 係数", 0.01, 1.0, 0.05, 2),
        ("commit_seconds", "確定秒数 (s)", 0.0, 5.0, 0.1, 1),
    ]

    def __init__(self, config, on_change, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")
        self._config = config
        self._on_change = on_change
        self._spins = {}

        form = QFormLayout()
        for field, label, lo, hi, step, dec in self._SPECS:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(dec)
            spin.setValue(getattr(config, field))          # connect 前に初期値設定（誤発火防止）
            spin.valueChanged.connect(
                lambda value, f=field: self._on_field_changed(f, value)
            )
            form.addRow(label, spin)
            self._spins[field] = spin

        reset = QPushButton("初期値に戻す")
        reset.clicked.connect(self._reset)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(reset)
        self.setLayout(layout)

    def _on_field_changed(self, field, value):
        setattr(self._config, field, value)
        self._on_change()

    def _reset(self):
        defaults = Config()
        for field, spin in self._spins.items():
            spin.setValue(getattr(defaults, field))        # valueChanged 経由で反映

    def closeEvent(self, event):
        settings.save(self._config)
        event.accept()


class MainWindow(QMainWindow):
    """メニューバー・映像表示・安全終了を担う。"""

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._shutdown_done = False
        self._settings_dialog = None
        self.setWindowTitle(config.window_name)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(640, 480)
        self.setCentralWidget(self._label)

        self._build_menu()

        self._thread = CaptureThread(config)
        self._thread.frameReady.connect(self._on_frame)
        self._thread.error.connect(self._on_error)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("ファイル(&F)")
        quit_action = QAction("終了(&Q)", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        adjust_menu = self.menuBar().addMenu("調整(&A)")
        calibrate_action = QAction("基準を設定(&C)", self)
        calibrate_action.setShortcut(QKeySequence("Ctrl+R"))
        calibrate_action.triggered.connect(self._request_calibration)
        adjust_menu.addAction(calibrate_action)

        overlay_action = QAction("骨格を表示(&S)", self)
        overlay_action.setCheckable(True)
        overlay_action.setChecked(self._config.show_pose_overlay)
        overlay_action.toggled.connect(self._toggle_pose_overlay)
        adjust_menu.addAction(overlay_action)

        settings_action = QAction("設定(&P)...", self)
        settings_action.triggered.connect(self._open_settings)
        adjust_menu.addAction(settings_action)

    def start_capture(self):
        self._thread.start()

    def _request_calibration(self):
        self._thread.request_calibration()

    def _open_settings(self):
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(
                self._config, self._thread.apply_settings, self
            )
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _toggle_pose_overlay(self, checked):
        self._config.show_pose_overlay = checked

    def _on_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def _on_error(self, message):
        QMessageBox.critical(
            self, "エラー", f"カメラまたはモデルの初期化に失敗しました:\n{message}"
        )
        self.close()

    def closeEvent(self, event):
        if not self._shutdown_done:
            self._shutdown_done = True
            self._thread.stop()
            self._thread.wait()
            if self._thread.pipeline is not None:
                print(self._thread.pipeline.stats.summary())
        event.accept()
