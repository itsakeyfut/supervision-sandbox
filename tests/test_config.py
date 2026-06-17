from attention_monitor.status import Status
from attention_monitor.config import Config


def test_status_members():
    assert {s.value for s in Status} == {"away", "focused", "distracted"}


def test_config_defaults():
    cfg = Config()
    assert cfg.camera_index == 0
    assert cfg.model_path == "yolo11n-pose.pt"
    assert cfg.yaw_center_threshold == 20.0
    assert cfg.pitch_center_threshold == 20.0
    assert cfg.pitch_neutral_ratio == 0.5
    assert cfg.commit_seconds == 0.7
    assert cfg.keypoint_confidence_min == 0.5
    assert cfg.window_name == "Attention Monitor"
