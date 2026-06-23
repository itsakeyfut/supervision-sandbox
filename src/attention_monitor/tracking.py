from attention_monitor.attention import HeadPose
from attention_monitor.gaze import GazeOffset


class _Tracker2D:
    """2成分値(a,b)を EMA 平滑化し、基準からのズレを返す。基準は calibration で登録。"""

    def __init__(self, ema_alpha):
        self._alpha = ema_alpha
        self._smoothed = None
        self._neutral = (0.0, 0.0)
        self._calibrate_pending = False

    def request_calibration(self):
        self._calibrate_pending = True

    def set_alpha(self, alpha):
        self._alpha = alpha

    def update(self, value):
        if value is None:
            self._smoothed = None
            return None
        a, b = value
        if self._smoothed is None:
            sa, sb = a, b
        else:
            al = self._alpha
            sa = al * a + (1.0 - al) * self._smoothed[0]
            sb = al * b + (1.0 - al) * self._smoothed[1]
        self._smoothed = (sa, sb)
        if self._calibrate_pending:
            self._neutral = (sa, sb)
            self._calibrate_pending = False
        return (sa - self._neutral[0], sb - self._neutral[1])


class AttentionTracker:
    """頭部姿勢と視線オフセットを EMA 平滑化し、単一点較正からのズレを返す。"""

    def __init__(self, ema_alpha):
        self._head = _Tracker2D(ema_alpha)
        self._gaze = _Tracker2D(ema_alpha)

    def request_calibration(self):
        self._head.request_calibration()
        self._gaze.request_calibration()

    def set_ema_alpha(self, alpha):
        self._head.set_alpha(alpha)
        self._gaze.set_alpha(alpha)

    def update(self, head_pose, gaze):
        head_in = None if head_pose is None else (head_pose.yaw, head_pose.pitch)
        gaze_in = None if gaze is None else (gaze.x, gaze.y)
        h = self._head.update(head_in)
        g = self._gaze.update(gaze_in)
        head_dev = None if h is None else HeadPose(h[0], h[1])
        gaze_dev = None if g is None else GazeOffset(g[0], g[1])
        return head_dev, gaze_dev
