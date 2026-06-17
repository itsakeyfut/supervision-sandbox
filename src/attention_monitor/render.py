import cv2
import numpy as np
import supervision as sv

from attention_monitor.status import Status

_STATUS_COLOR = {
    Status.FOCUSED: (0, 180, 0),       # 緑 (BGR)
    Status.DISTRACTED: (0, 165, 255),  # 橙
    Status.AWAY: (0, 0, 220),          # 赤
}
_STATUS_LABEL = {
    Status.FOCUSED: "FOCUSED",
    Status.DISTRACTED: "DISTRACTED",
    Status.AWAY: "AWAY",
}


class Renderer:
    def __init__(self, config):
        self._config = config
        self._vertex = sv.VertexAnnotator(radius=4)
        self._edge = sv.EdgeAnnotator(thickness=2)

    def draw(self, frame, keypoints, primary, raw, committed, stats, fps):
        out = frame.copy()

        # スケルトン（検出があれば）
        if keypoints is not None and len(keypoints) > 0:
            out = self._edge.annotate(out, keypoints)
            out = self._vertex.annotate(out, keypoints)

        h, w = out.shape[:2]
        color = _STATUS_COLOR[committed]

        # 上部ステータスバナー
        cv2.rectangle(out, (0, 0), (w, 48), color, thickness=-1)
        cv2.putText(
            out, _STATUS_LABEL[committed], (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # 統計 HUD（左下）
        lines = [
            f"FPS: {fps:4.1f}   raw: {_STATUS_LABEL[raw]}",
            f"focused:    {stats.durations[Status.FOCUSED]:6.1f}s",
            f"distracted: {stats.durations[Status.DISTRACTED]:6.1f}s",
            f"away:       {stats.durations[Status.AWAY]:6.1f}s",
            f"focus rate: {stats.focus_rate:5.1f}%",
        ]
        y0 = h - 16 - 22 * (len(lines) - 1)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, y0 - 22), (320, h), (0, 0, 0), thickness=-1)
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
        for i, line in enumerate(lines):
            cv2.putText(
                out, line, (12, y0 + 22 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return out
