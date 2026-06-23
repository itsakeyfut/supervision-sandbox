import cv2
import numpy as np

from attention_monitor.attention import (
    PrimarySubject, NOSE, L_EYE, R_EYE, L_EAR, R_EAR,
)
from attention_monitor.headpose import (
    camera_matrix_from_size, estimate_head_pose, _MODEL_POINTS,
)

_FACE = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


def _project(rvec, K):
    obj = np.array([_MODEL_POINTS[i] for i in _FACE], dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [1000.0]])
    pts2d, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros((4, 1)))
    return pts2d.reshape(-1, 2)


def _subject(pts2d):
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros((17,), dtype=float)
    for j, i in enumerate(_FACE):
        xy[i] = pts2d[j]
        conf[i] = 1.0
    return PrimarySubject(0, xy, conf)


def test_camera_matrix_shape_and_center():
    K = camera_matrix_from_size(640, 480)
    assert K.shape == (3, 3)
    assert K[0, 0] == 640.0 and K[1, 1] == 640.0
    assert K[0, 2] == 320.0 and K[1, 2] == 240.0


def test_frontal_is_near_zero():
    K = camera_matrix_from_size(640, 480)
    pts = _project(np.zeros((3, 1)), K)
    pose = estimate_head_pose(_subject(pts), K, 0.5)
    assert pose is not None
    assert abs(pose.yaw) < 2.0
    assert abs(pose.pitch) < 2.0


def test_recovers_known_yaw():
    K = camera_matrix_from_size(640, 480)
    t = np.radians(25.0)
    R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    rvec, _ = cv2.Rodrigues(R)
    pose = estimate_head_pose(_subject(_project(rvec, K)), K, 0.5)
    assert pose is not None
    assert abs(abs(pose.yaw) - 25.0) < 3.0
    assert abs(pose.pitch) < 3.0


def test_recovers_known_pitch():
    K = camera_matrix_from_size(640, 480)
    p = np.radians(15.0)
    R = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rvec, _ = cv2.Rodrigues(R)
    pose = estimate_head_pose(_subject(_project(rvec, K)), K, 0.5)
    assert pose is not None
    assert abs(abs(pose.pitch) - 15.0) < 3.0
    assert abs(pose.yaw) < 3.0


def test_too_few_points_returns_none():
    K = camera_matrix_from_size(640, 480)
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros((17,), dtype=float)
    for i in (NOSE, L_EYE, R_EYE):  # 3点のみ
        xy[i] = (300 + i, 200)
        conf[i] = 1.0
    assert estimate_head_pose(PrimarySubject(0, xy, conf), K, 0.5) is None
