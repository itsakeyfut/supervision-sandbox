from __future__ import annotations

import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from attention_monitor.capture import WebcamSource
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
