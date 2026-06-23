from attention_monitor.state import StateMachine
from attention_monitor.status import Status


def test_starts_away():
    sm = StateMachine(commit_seconds=1.0)
    assert sm.committed is Status.AWAY


def test_no_switch_before_commit_seconds():
    sm = StateMachine(commit_seconds=1.0)
    assert sm.update(Status.FOCUSED, 0.5) is Status.AWAY
    assert sm.update(Status.FOCUSED, 0.4) is Status.AWAY  # 累計 0.9 < 1.0


def test_switch_after_commit_seconds():
    sm = StateMachine(commit_seconds=1.0)
    sm.update(Status.FOCUSED, 0.5)
    sm.update(Status.FOCUSED, 0.4)
    assert sm.update(Status.FOCUSED, 0.2) is Status.FOCUSED  # 累計 1.1 >= 1.0


def test_candidate_change_resets_timer():
    sm = StateMachine(commit_seconds=1.0)
    sm.update(Status.FOCUSED, 0.6)
    # 候補が変わると計時リセット（DISTRACTED を 0.6 から数え直し）
    assert sm.update(Status.DISTRACTED, 0.6) is Status.AWAY
    assert sm.update(Status.DISTRACTED, 0.5) is Status.DISTRACTED  # 0.6+0.5=1.1


def test_raw_equal_committed_clears_candidate():
    sm = StateMachine(commit_seconds=1.0)
    sm.update(Status.FOCUSED, 0.9)        # 候補 FOCUSED, 経過 0.9
    sm.update(Status.AWAY, 0.5)           # raw == committed(AWAY) → 候補クリア
    assert sm.update(Status.FOCUSED, 0.5) is Status.AWAY  # 0.5 のみ → まだ確定しない
