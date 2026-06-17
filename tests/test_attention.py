import numpy as np

from attention_monitor.attention import select_primary, PrimarySubject, estimate_head_pose, HeadPose


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


def _subject_with(nose, l_eye, r_eye, l_ear, r_ear):
    """各引数は (x, y, conf)。"""
    arr = np.zeros((17, 3), dtype=float)
    arr[0], arr[1], arr[2], arr[3], arr[4] = nose, l_eye, r_eye, l_ear, r_ear
    return PrimarySubject(0, arr[:, :2], arr[:, 2])


def test_head_pose_forward_is_centered():
    # 鼻が両耳の中点、目は鼻の 0.5*span 上 → yaw=0, pitch=0
    subj = _subject_with(
        nose=(0, 50, 0.9),
        l_eye=(-20, 0, 0.9), r_eye=(20, 0, 0.9),
        l_ear=(-50, 0, 0.9), r_ear=(50, 0, 0.9),
    )
    pose = estimate_head_pose(subj, 0.5, pitch_neutral_ratio=0.5)
    assert isinstance(pose, HeadPose)
    assert abs(pose.yaw) < 1e-6
    assert abs(pose.pitch) < 1e-6


def test_head_pose_turned_gives_large_yaw():
    # 鼻が右耳に寄る → |yaw| 大
    subj = _subject_with(
        nose=(40, 50, 0.9),
        l_eye=(-20, 0, 0.9), r_eye=(20, 0, 0.9),
        l_ear=(-50, 0, 0.9), r_ear=(50, 0, 0.9),
    )
    pose = estimate_head_pose(subj, 0.5, pitch_neutral_ratio=0.5)
    assert abs(pose.yaw) > 30.0


def test_head_pose_returns_none_without_nose():
    subj = _subject_with(
        nose=(0, 50, 0.0),  # 無効
        l_eye=(-20, 0, 0.9), r_eye=(20, 0, 0.9),
        l_ear=(-50, 0, 0.9), r_ear=(50, 0, 0.9),
    )
    assert estimate_head_pose(subj, 0.5, pitch_neutral_ratio=0.5) is None


def test_head_pose_pitch_down():
    # 鼻が目より 0.8*span 下 → pitch=(0.8-0.5)*90=27 度
    subj = _subject_with(
        nose=(0, 80, 0.9),
        l_eye=(-20, 0, 0.9), r_eye=(20, 0, 0.9),
        l_ear=(-50, 0, 0.9), r_ear=(50, 0, 0.9),
    )
    pose = estimate_head_pose(subj, 0.5, pitch_neutral_ratio=0.5)
    assert abs(pose.pitch - 27.0) < 1e-6  # (0.8-0.5)*90、浮動小数なので許容誤差で比較
