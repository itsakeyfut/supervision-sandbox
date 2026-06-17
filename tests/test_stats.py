from attention_monitor.stats import SessionStats
from attention_monitor.status import Status


def test_accumulates_durations():
    s = SessionStats()
    s.update(Status.FOCUSED, 1.0)
    s.update(Status.FOCUSED, 0.5)
    s.update(Status.AWAY, 2.0)
    assert s.durations[Status.FOCUSED] == 1.5
    assert s.durations[Status.AWAY] == 2.0
    assert s.total == 3.5


def test_focus_rate():
    s = SessionStats()
    s.update(Status.FOCUSED, 3.0)
    s.update(Status.DISTRACTED, 1.0)
    s.update(Status.AWAY, 10.0)  # present に含めない
    assert s.present == 4.0
    assert s.focus_rate == 75.0


def test_focus_rate_zero_when_no_presence():
    s = SessionStats()
    s.update(Status.AWAY, 5.0)
    assert s.focus_rate == 0.0


def test_summary_is_string_with_values():
    s = SessionStats()
    s.update(Status.FOCUSED, 2.0)
    text = s.summary()
    assert isinstance(text, str)
    assert "focused" in text
    assert "%" in text
