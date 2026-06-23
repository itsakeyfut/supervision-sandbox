from attention_monitor.attention import HeadPose


class HeadPoseTracker:
    """頭部姿勢を EMA 平滑化し、基準角からのズレを返す。基準は calibration で登録。"""

    def __init__(self, ema_alpha):
        self._alpha = ema_alpha
        self._smoothed = None
        self._neutral_yaw = 0.0
        self._neutral_pitch = 0.0
        self._calibrate_pending = False

    def request_calibration(self):
        self._calibrate_pending = True

    def update(self, headpose):
        if headpose is None:
            self._smoothed = None
            return None

        if self._smoothed is None:
            sy, sp = headpose.yaw, headpose.pitch
        else:
            a = self._alpha
            sy = a * headpose.yaw + (1.0 - a) * self._smoothed.yaw
            sp = a * headpose.pitch + (1.0 - a) * self._smoothed.pitch
        self._smoothed = HeadPose(sy, sp)

        if self._calibrate_pending:
            self._neutral_yaw = sy
            self._neutral_pitch = sp
            self._calibrate_pending = False

        return HeadPose(sy - self._neutral_yaw, sp - self._neutral_pitch)
