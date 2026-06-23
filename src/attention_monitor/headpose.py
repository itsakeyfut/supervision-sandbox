from __future__ import annotations

import cv2
import numpy as np

from attention_monitor.attention import HeadPose, NOSE, L_EYE, R_EYE, L_EAR, R_EAR

# 3D 顔モデル(mm)。OpenCVカメラ規約(X右/Y下/Z奥)。鼻先原点、目・耳は後方。
_MODEL_POINTS = {
    NOSE: (0.0, 0.0, 0.0),
    L_EYE: (32.0, -38.0, 40.0),
    R_EYE: (-32.0, -38.0, 40.0),
    L_EAR: (78.0, 0.0, 100.0),
    R_EAR: (-78.0, 0.0, 100.0),
}
_FACE_INDICES = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


def camera_matrix_from_size(width, height):
    f = float(width)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def estimate_head_pose(subject, camera_matrix, min_confidence):
    """信頼度のある顔キーポイントから solvePnP で頭部姿勢(yaw/pitch度)を出す。"""
    xy = subject.xy
    conf = subject.confidence

    object_points = []
    image_points = []
    for idx in _FACE_INDICES:
        if conf[idx] >= min_confidence:
            object_points.append(_MODEL_POINTS[idx])
            image_points.append((float(xy[idx, 0]), float(xy[idx, 1])))

    if len(object_points) < 4:
        return None

    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    if forward[2] > 0.0:  # 顔が後ろ向き＝退化解
        return None

    yaw = float(np.degrees(np.arctan2(forward[0], -forward[2])))
    pitch = float(np.degrees(np.arctan2(forward[1], -forward[2])))
    return HeadPose(yaw=yaw, pitch=pitch)
