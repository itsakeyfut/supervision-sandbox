from dataclasses import dataclass


@dataclass
class Config:
    camera_index: int = 0
    model_path: str = "yolo11s-pose.pt"
    yaw_center_threshold: float = 25.0      # |yaw-基準| <= これなら正面とみなす（度）
    pitch_center_threshold: float = 20.0    # |pitch-基準| <= これなら水平とみなす（度）
    commit_seconds: float = 0.7             # ステータス確定までの継続秒数
    keypoint_confidence_min: float = 0.5    # これ未満のキーポイントは欠損扱い
    window_name: str = "Attention Monitor"
    show_pose_overlay: bool = False         # True で骨格を重畳表示（デバッグ用）
    ema_alpha: float = 0.4                  # 頭部姿勢の EMA 平滑化係数（0<a<=1、大きいほど追従）
    auto_calibrate: bool = True             # 起動時に最初の有効姿勢を基準化
