from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GazeOffset:
    x: float  # + = 視線が画面右寄り
    y: float  # + = 下向き（pitch と符号整合）


def gaze_offset_from_blendshapes(blendshapes):
    """MediaPipe の eyeLook blendshapes から視線オフセット(x,y)を出す。なければ None。"""
    if not blendshapes:
        return None

    g = blendshapes.get
    x = (g("eyeLookOutRight", 0.0) + g("eyeLookInLeft", 0.0)) / 2.0 - (
        g("eyeLookInRight", 0.0) + g("eyeLookOutLeft", 0.0)
    ) / 2.0
    y = (g("eyeLookDownLeft", 0.0) + g("eyeLookDownRight", 0.0)) / 2.0 - (
        g("eyeLookUpLeft", 0.0) + g("eyeLookUpRight", 0.0)
    ) / 2.0
    return GazeOffset(x=float(x), y=float(y))
