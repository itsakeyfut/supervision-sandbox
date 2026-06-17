import numpy as np

from attention_monitor.attention import select_primary, PrimarySubject


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
