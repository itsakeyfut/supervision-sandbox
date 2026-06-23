from attention_monitor.gaze import GazeOffset, gaze_offset_from_blendshapes

_KEYS = [
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
]


def _bs(**overrides):
    d = {k: 0.0 for k in _KEYS}
    d.update(overrides)
    return d


def test_centered_is_zero():
    assert gaze_offset_from_blendshapes(_bs()) == GazeOffset(0.0, 0.0)


def test_looking_right_positive_x():
    g = gaze_offset_from_blendshapes(_bs(eyeLookOutRight=0.8, eyeLookInLeft=0.8))
    assert g.x > 0.0
    assert abs(g.y) < 1e-9


def test_looking_left_negative_x():
    g = gaze_offset_from_blendshapes(_bs(eyeLookInRight=0.8, eyeLookOutLeft=0.8))
    assert g.x < 0.0


def test_looking_down_positive_y():
    g = gaze_offset_from_blendshapes(_bs(eyeLookDownLeft=0.8, eyeLookDownRight=0.8))
    assert g.y > 0.0
    assert abs(g.x) < 1e-9


def test_looking_up_negative_y():
    g = gaze_offset_from_blendshapes(_bs(eyeLookUpLeft=0.8, eyeLookUpRight=0.8))
    assert g.y < 0.0


def test_empty_returns_none():
    assert gaze_offset_from_blendshapes({}) is None


def test_none_returns_none():
    assert gaze_offset_from_blendshapes(None) is None
