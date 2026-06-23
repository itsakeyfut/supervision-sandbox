from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_MODEL_PATH = "models/face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass
class FaceResult:
    landmarks: np.ndarray                      # (478, 3) 正規化 x,y,z（虹彩 468-477 含む）
    transformation_matrix: np.ndarray | None   # (4, 4) または None
    blendshapes: dict                          # category_name -> score（全52種）


def _ensure_model(path):
    """モデルが無ければ公式URLから取得する。"""
    if not os.path.exists(path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


class FaceMesh:
    """MediaPipe FaceLandmarker(VIDEO) を薄くラップし、1フレームの顔解析結果を返す。"""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        _ensure_model(model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if not result.face_landmarks:
            return None

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]], dtype=np.float64
        )

        matrix = None
        if result.facial_transformation_matrixes:
            matrix = np.array(
                result.facial_transformation_matrixes[0], dtype=np.float64
            )

        blendshapes = {}
        if result.face_blendshapes:
            blendshapes = {c.category_name: c.score for c in result.face_blendshapes[0]}

        return FaceResult(
            landmarks=landmarks, transformation_matrix=matrix, blendshapes=blendshapes
        )
