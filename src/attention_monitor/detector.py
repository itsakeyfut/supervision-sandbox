import supervision as sv
from ultralytics import YOLO


class PoseDetector:
    """Ultralytics YOLO-pose を Supervision の KeyPoints に変換して返す。"""

    def __init__(self, model_path):
        self._model = YOLO(model_path)

    def detect(self, frame):
        results = self._model(frame, verbose=False)[0]
        return sv.KeyPoints.from_ultralytics(results)
