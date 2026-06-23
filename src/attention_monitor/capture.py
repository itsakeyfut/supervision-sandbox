import cv2


class WebcamSource:
    """OpenCV VideoCapture を薄くラップしたフレーム供給源。"""

    def __init__(self, camera_index):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {camera_index} を開けませんでした")

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self._cap.release()
