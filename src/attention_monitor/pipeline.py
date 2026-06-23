from attention_monitor.attention import classify, select_primary
from attention_monitor.detector import PoseDetector
from attention_monitor.facemesh import FaceMesh
from attention_monitor.gaze import gaze_offset_from_blendshapes
from attention_monitor.headpose_mp import head_pose_from_matrix
from attention_monitor.render import Renderer
from attention_monitor.state import StateMachine
from attention_monitor.stats import SessionStats
from attention_monitor.tracking import AttentionTracker


class Pipeline:
    def __init__(self, config):
        self._config = config
        self._detector = PoseDetector(config.model_path)
        self._facemesh = FaceMesh()
        self._renderer = Renderer(config)
        self._tracker = AttentionTracker(config.ema_alpha)
        if config.auto_calibrate:
            self._tracker.request_calibration()
        self._ts_ms = 0
        self.state = StateMachine(config.commit_seconds)
        self.stats = SessionStats()

    def request_calibration(self):
        self._tracker.request_calibration()

    def process(self, frame, dt, fps):
        cfg = self._config
        self._ts_ms += max(1, round(dt * 1000))

        keypoints = self._detector.detect(frame)
        primary = select_primary(
            keypoints.xy, keypoints.keypoint_confidence, cfg.keypoint_confidence_min
        )

        face = self._facemesh.detect(frame, self._ts_ms)
        if face is not None:
            raw_head = head_pose_from_matrix(face.transformation_matrix)
            raw_gaze = gaze_offset_from_blendshapes(face.blendshapes)
        else:
            raw_head = None
            raw_gaze = None

        head_dev, gaze_dev = self._tracker.update(raw_head, raw_gaze)
        raw = classify(
            primary is not None, head_dev,
            cfg.yaw_center_threshold, cfg.pitch_center_threshold,
            gaze_dev, cfg.gaze_threshold,
        )
        committed = self.state.update(raw, dt)
        self.stats.update(committed, dt)

        return self._renderer.draw(
            frame, keypoints, primary, raw, committed, self.stats, fps
        )
