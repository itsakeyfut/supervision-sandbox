from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
