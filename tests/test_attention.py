import numpy as np

from attention_monitor.attention import select_primary, PrimarySubject, HeadPose, classify
from attention_monitor.gaze import GazeOffset
from attention_monitor.status import Status


def _person(points):
    """points: list of (x, y, conf) 長さ K=17。足りない分は conf=0 で埋める。"""
    arr = np.zeros((17, 3), dtype=float)
    for i, (x, y, c) in enumerate(points):
        arr[i] = (x, y, c)
    return arr[:, :2], arr[:, 2]


def test_returns_none_when_empty():
    xy = np.zeros((0, 17, 2))
    conf = np.zeros((0, 17))
    assert select_primary(xy, conf, 0.5) is None


def test_skips_person_with_too_few_valid_keypoints():
    # 有効点が 2 点だけ（>=3 が必要）
    xy0, c0 = _person([(0, 0, 0.9), (10, 0, 0.9)])
    xy = xy0[None, ...]
    conf = c0[None, ...]
    assert select_primary(xy, conf, 0.5) is None


def test_picks_largest_person():
    small_xy, small_c = _person([(0, 0, 0.9), (10, 0, 0.9), (5, 10, 0.9)])
    big_xy, big_c = _person([(0, 0, 0.9), (100, 0, 0.9), (50, 100, 0.9)])
    xy = np.stack([small_xy, big_xy])
    conf = np.stack([small_c, big_c])
    result = select_primary(xy, conf, 0.5)
    assert isinstance(result, PrimarySubject)
    assert result.index == 1  # big


def test_classify_away_when_absent():
    assert classify(False, None, 20.0, 20.0) is Status.AWAY


def test_classify_distracted_when_present_but_no_pose():
    assert classify(True, None, 20.0, 20.0) is Status.DISTRACTED


def test_classify_focused_when_centered():
    pose = HeadPose(yaw=5.0, pitch=-3.0)
    assert classify(True, pose, 20.0, 20.0) is Status.FOCUSED


def test_classify_distracted_when_yaw_exceeds():
    pose = HeadPose(yaw=45.0, pitch=0.0)
    assert classify(True, pose, 20.0, 20.0) is Status.DISTRACTED


def test_classify_distracted_when_pitch_exceeds():
    pose = HeadPose(yaw=0.0, pitch=27.0)
    assert classify(True, pose, 20.0, 20.0) is Status.DISTRACTED


def test_classify_focused_with_gaze_within():
    pose = HeadPose(5.0, -3.0)
    assert classify(True, pose, 20.0, 20.0, GazeOffset(0.1, 0.05), 0.3) is Status.FOCUSED


def test_classify_distracted_when_gaze_exceeds():
    pose = HeadPose(5.0, -3.0)
    assert classify(True, pose, 20.0, 20.0, GazeOffset(0.5, 0.0), 0.3) is Status.DISTRACTED


def test_classify_gaze_optional_backward_compat():
    pose = HeadPose(5.0, -3.0)
    assert classify(True, pose, 20.0, 20.0) is Status.FOCUSED
