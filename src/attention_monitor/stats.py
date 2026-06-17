from attention_monitor.status import Status


class SessionStats:
    def __init__(self):
        self.durations = {
            Status.AWAY: 0.0,
            Status.FOCUSED: 0.0,
            Status.DISTRACTED: 0.0,
        }

    def update(self, status, dt):
        self.durations[status] += dt

    @property
    def total(self):
        return sum(self.durations.values())

    @property
    def present(self):
        return self.durations[Status.FOCUSED] + self.durations[Status.DISTRACTED]

    @property
    def focus_rate(self):
        present = self.present
        if present <= 0.0:
            return 0.0
        return 100.0 * self.durations[Status.FOCUSED] / present

    def summary(self):
        lines = ["=== Session Summary ==="]
        for status in (Status.FOCUSED, Status.DISTRACTED, Status.AWAY):
            lines.append(f"{status.value}: {self.durations[status]:.1f}s")
        lines.append(f"focus rate: {self.focus_rate:.1f}%")
        lines.append(f"total: {self.total:.1f}s")
        return "\n".join(lines)
