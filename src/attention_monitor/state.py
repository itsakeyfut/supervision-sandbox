from attention_monitor.status import Status


class StateMachine:
    """生判定(raw)が commit_seconds 継続して初めて確定ステータスを切り替える。"""

    def __init__(self, commit_seconds):
        self.commit_seconds = commit_seconds
        self.committed = Status.AWAY
        self._candidate = None
        self._elapsed = 0.0

    def update(self, raw, dt):
        if raw == self.committed:
            self._candidate = None
            self._elapsed = 0.0
            return self.committed

        if raw == self._candidate:
            self._elapsed += dt
        else:
            self._candidate = raw
            self._elapsed = dt

        if self._elapsed >= self.commit_seconds:
            self.committed = raw
            self._candidate = None
            self._elapsed = 0.0

        return self.committed
