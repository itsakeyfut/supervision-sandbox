import json

from attention_monitor.config import Config
from attention_monitor.settings import ADJUSTABLE, load_into, save


def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    c = Config()
    c.yaw_center_threshold = 33.0
    c.gaze_threshold = 0.5
    c.ema_alpha = 0.6
    save(c, p)
    c2 = Config()
    load_into(c2, p)
    assert c2.yaw_center_threshold == 33.0
    assert c2.gaze_threshold == 0.5
    assert c2.ema_alpha == 0.6


def test_missing_file_keeps_defaults(tmp_path):
    c = Config()
    load_into(c, tmp_path / "nope.json")
    assert c.yaw_center_threshold == Config().yaw_center_threshold


def test_partial_json_overrides_only_present(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"pitch_center_threshold": 12.0}), encoding="utf-8")
    c = Config()
    load_into(c, p)
    assert c.pitch_center_threshold == 12.0
    assert c.yaw_center_threshold == Config().yaw_center_threshold


def test_unknown_and_nonadjustable_keys_ignored(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps({"foo": 1, "model_path": "evil.pt", "yaw_center_threshold": 40.0}),
        encoding="utf-8",
    )
    c = Config()
    load_into(c, p)
    assert c.yaw_center_threshold == 40.0
    assert c.model_path == Config().model_path


def test_save_writes_only_adjustable(tmp_path):
    p = tmp_path / "s.json"
    save(Config(), p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data.keys()) == set(ADJUSTABLE)


def test_save_creates_parent_dir(tmp_path):
    p = tmp_path / "sub" / "dir" / "s.json"
    save(Config(), p)
    assert p.exists()
