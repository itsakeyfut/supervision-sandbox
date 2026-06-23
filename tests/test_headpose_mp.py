import numpy as np

from attention_monitor.headpose_mp import head_pose_from_matrix


def _mat4(rotation):
    m = np.eye(4)
    m[:3, :3] = rotation
    return m


def _ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_frontal_identity_is_zero():
    pose = head_pose_from_matrix(np.eye(4))
    assert pose is not None
    assert abs(pose.yaw) < 1e-6
    assert abs(pose.pitch) < 1e-6


def test_recovers_known_yaw():
    pose = head_pose_from_matrix(_mat4(_ry(np.radians(25.0))))
    assert pose is not None
    assert abs(abs(pose.yaw) - 25.0) < 1e-6
    assert abs(pose.pitch) < 1e-6


def test_recovers_known_pitch():
    pose = head_pose_from_matrix(_mat4(_rx(np.radians(15.0))))
    assert pose is not None
    assert abs(abs(pose.pitch) - 15.0) < 1e-6
    assert abs(pose.yaw) < 1e-6


def test_facing_away_returns_none():
    assert head_pose_from_matrix(_mat4(_ry(np.radians(180.0)))) is None


def test_none_returns_none():
    assert head_pose_from_matrix(None) is None
