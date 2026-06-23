from attention_monitor.attention import classify, select_primary
from attention_monitor.detector import PoseDetector
from attention_monitor.headpose import camera_matrix_from_size, estimate_head_pose
from attention_monitor.render import Renderer
from attention_monitor.state import StateMachine
from attention_monitor.stats import SessionStats
from attention_monitor.tracking import HeadPoseTracker


class Pipeline:
    def __init__(self, config):
        self._config = config
        self._detector = PoseDetector(config.model_path)
        self._renderer = Renderer(config)
        self._tracker = HeadPoseTracker(config.ema_alpha)
        if config.auto_calibrate:
            self._tracker.request_calibration()
        self._camera_matrix = None
        self.state = StateMachine(config.commit_seconds)
        self.stats = SessionStats()

    def request_calibration(self):
        self._tracker.request_calibration()

    def process(self, frame, dt, fps):
        cfg = self._config
        if self._camera_matrix is None:
            h, w = frame.shape[:2]
            self._camera_matrix = camera_matrix_from_size(w, h)

        keypoints = self._detector.detect(frame)
        primary = select_primary(
            keypoints.xy, keypoints.keypoint_confidence, cfg.keypoint_confidence_min
        )
        if primary is not None:
            raw_pose = estimate_head_pose(
                primary, self._camera_matrix, cfg.keypoint_confidence_min
            )
        else:
            raw_pose = None

        tracked = self._tracker.update(raw_pose)
        raw = classify(
            primary is not None, tracked,
            cfg.yaw_center_threshold, cfg.pitch_center_threshold,
        )
        committed = self.state.update(raw, dt)
        self.stats.update(committed, dt)

        return self._renderer.draw(
            frame, keypoints, primary, raw, committed, self.stats, fps
        )
