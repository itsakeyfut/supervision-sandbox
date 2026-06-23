from __future__ import annotations

import json
from pathlib import Path

from attention_monitor.config import Config

SETTINGS_PATH = Path.home() / ".attention_monitor" / "settings.json"

ADJUSTABLE = (
    "yaw_center_threshold",
    "pitch_center_threshold",
    "gaze_threshold",
    "ema_alpha",
    "commit_seconds",
)


def load_into(config, path=SETTINGS_PATH):
    """設定ファイルがあれば ADJUSTABLE のキーだけ Config に上書きする。"""
    path = Path(path)
    if not path.exists():
        return config
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return config
    if not isinstance(data, dict):
        return config
    for key in ADJUSTABLE:
        if key in data:
            try:
                setattr(config, key, float(data[key]))
            except (TypeError, ValueError):
                pass
    return config


def save(config, path=SETTINGS_PATH):
    """ADJUSTABLE の現在値を JSON に保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: getattr(config, key) for key in ADJUSTABLE}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
