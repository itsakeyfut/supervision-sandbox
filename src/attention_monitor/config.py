from dataclasses import dataclass


@dataclass
class Config:
    camera_index: int = 0
    model_path: str = "yolo11n-pose.pt"
    yaw_center_threshold: float = 20.0      # |yaw| <= これなら正面とみなす（度）
    pitch_center_threshold: float = 20.0    # |pitch| <= これなら水平とみなす（度）
    pitch_neutral_ratio: float = 0.5        # 正面時の「鼻が目より下にある量 / 顔幅」の基準値
    commit_seconds: float = 0.7             # ステータス確定までの継続秒数
    keypoint_confidence_min: float = 0.5    # これ未満のキーポイントは欠損扱い
    window_name: str = "Attention Monitor"
