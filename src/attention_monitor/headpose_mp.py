from __future__ import annotations

import numpy as np

from attention_monitor.attention import HeadPose


def head_pose_from_matrix(matrix):
    """MediaPipe の顔変換行列(4x4)から頭部姿勢(yaw/pitch 度)を出す。推定不能なら None。"""
    if matrix is None:
        return None

    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    if forward[2] > 0.0:  # 顔が後ろ向き＝退化解
        return None

    yaw = float(np.degrees(np.arctan2(forward[0], -forward[2])))
    pitch = float(np.degrees(np.arctan2(forward[1], -forward[2])))
    return HeadPose(yaw=yaw, pitch=pitch)
