from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attention_monitor.status import Status

# COCO keypoint indices
NOSE = 0
L_EYE = 1
R_EYE = 2
L_EAR = 3
R_EAR = 4


@dataclass(frozen=True)
class PrimarySubject:
    index: int
    xy: np.ndarray          # shape (K, 2)
    confidence: np.ndarray  # shape (K,)


@dataclass(frozen=True)
class HeadPose:
    yaw: float    # degrees, 0 = facing camera
    pitch: float  # degrees, 0 = level, + = looking down


def select_primary(xy, confidence, min_confidence):
    """最も手前（キーポイント外接矩形が最大）の人物を返す。なければ None。"""
    if xy is None or confidence is None or len(xy) == 0:
        return None

    best_index = -1
    best_area = -1.0
    for i in range(len(xy)):
        valid = confidence[i] >= min_confidence
        if int(valid.sum()) < 3:
            continue
        pts = xy[i][valid]
        width = float(pts[:, 0].max() - pts[:, 0].min())
        height = float(pts[:, 1].max() - pts[:, 1].min())
        area = max(width, 1.0) * max(height, 1.0)
        if area > best_area:
            best_area = area
            best_index = i

    if best_index < 0:
        return None
    return PrimarySubject(best_index, xy[best_index], confidence[best_index])


def estimate_head_pose(subject, min_confidence, pitch_neutral_ratio):
    """主被写体の顔キーポイントから頭の向き(yaw/pitch)を近似。推定不能なら None。"""
    xy = subject.xy
    conf = subject.confidence

    def ok(i):
        return conf[i] >= min_confidence

    if not ok(NOSE):
        return None

    # yaw: 両耳優先、なければ両目
    if ok(L_EAR) and ok(R_EAR):
        left_x, right_x = float(xy[L_EAR, 0]), float(xy[R_EAR, 0])
    elif ok(L_EYE) and ok(R_EYE):
        left_x, right_x = float(xy[L_EYE, 0]), float(xy[R_EYE, 0])
    else:
        return None

    span = abs(right_x - left_x)
    if span < 1e-6:
        return None

    mid_x = (left_x + right_x) / 2.0
    ratio = (float(xy[NOSE, 0]) - mid_x) / (span / 2.0)
    yaw = float(np.clip(ratio, -1.0, 1.0)) * 90.0

    # pitch: 両目があれば鼻の縦オフセットから近似
    if ok(L_EYE) and ok(R_EYE):
        eye_mid_y = (float(xy[L_EYE, 1]) + float(xy[R_EYE, 1])) / 2.0
        pitch_ratio = (float(xy[NOSE, 1]) - eye_mid_y) / span
        pitch = (pitch_ratio - pitch_neutral_ratio) * 90.0
    else:
        pitch = 0.0

    return HeadPose(yaw=yaw, pitch=float(pitch))


def classify(subject_present, head_pose, yaw_threshold, pitch_threshold):
    """在席フラグと頭の向きからステータスを決める（生判定）。"""
    if not subject_present:
        return Status.AWAY
    if head_pose is None:
        return Status.DISTRACTED
    if abs(head_pose.yaw) <= yaw_threshold and abs(head_pose.pitch) <= pitch_threshold:
        return Status.FOCUSED
    return Status.DISTRACTED
