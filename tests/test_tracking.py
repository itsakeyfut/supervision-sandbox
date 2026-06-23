from attention_monitor.attention import HeadPose
from attention_monitor.gaze import GazeOffset
from attention_monitor.tracking import AttentionTracker


def test_attention_tracker_first_values_passthrough():
    t = AttentionTracker(ema_alpha=0.5)
    head, gaze = t.update(HeadPose(10.0, 4.0), GazeOffset(0.2, -0.1))
    assert head == HeadPose(10.0, 4.0)
    assert gaze == GazeOffset(0.2, -0.1)


def test_attention_tracker_ema_each_channel():
    t = AttentionTracker(ema_alpha=0.5)
    t.update(HeadPose(10.0, 0.0), GazeOffset(0.4, 0.0))
    head, gaze = t.update(HeadPose(20.0, 0.0), GazeOffset(0.8, 0.0))
    assert abs(head.yaw - 15.0) < 1e-9
    assert abs(gaze.x - 0.6) < 1e-9


def test_attention_tracker_single_calibration_both():
    t = AttentionTracker(ema_alpha=1.0)
    t.request_calibration()
    head, gaze = t.update(HeadPose(30.0, 10.0), GazeOffset(0.5, -0.3))
    assert head == HeadPose(0.0, 0.0)
    assert gaze == GazeOffset(0.0, 0.0)
    head2, gaze2 = t.update(HeadPose(45.0, 10.0), GazeOffset(0.7, -0.3))
    assert abs(head2.yaw - 15.0) < 1e-9
    assert abs(gaze2.x - 0.2) < 1e-9


def test_attention_tracker_independent_none():
    t = AttentionTracker(ema_alpha=0.5)
    head, gaze = t.update(None, GazeOffset(0.4, 0.0))
    assert head is None
    assert gaze == GazeOffset(0.4, 0.0)
    head2, gaze2 = t.update(HeadPose(10.0, 0.0), None)
    assert head2 == HeadPose(10.0, 0.0)
    assert gaze2 is None


def test_set_ema_alpha_updates_both_channels():
    t = AttentionTracker(ema_alpha=0.5)
    t.set_ema_alpha(1.0)  # 平滑化なし
    t.update(HeadPose(10.0, 0.0), GazeOffset(0.4, 0.0))
    head, gaze = t.update(HeadPose(20.0, 0.0), GazeOffset(0.8, 0.0))
    assert abs(head.yaw - 20.0) < 1e-9
    assert abs(gaze.x - 0.8) < 1e-9
