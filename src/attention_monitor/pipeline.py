from attention_monitor.attention import classify, estimate_head_pose, select_primary
from attention_monitor.detector import PoseDetector
from attention_monitor.render import Renderer
from attention_monitor.state import StateMachine
from attention_monitor.stats import SessionStats


class Pipeline:
    def __init__(self, config):
        self._config = config
        self._detector = PoseDetector(config.model_path)
        self._renderer = Renderer(config)
        self.state = StateMachine(config.commit_seconds)
        self.stats = SessionStats()

    def process(self, frame, dt, fps):
        cfg = self._config
        keypoints = self._detector.detect(frame)

        primary = select_primary(
            keypoints.xy, keypoints.confidence, cfg.keypoint_confidence_min
        )
        if primary is not None:
            head_pose = estimate_head_pose(
                primary, cfg.keypoint_confidence_min, cfg.pitch_neutral_ratio
            )
        else:
            head_pose = None

        raw = classify(
            primary is not None, head_pose,
            cfg.yaw_center_threshold, cfg.pitch_center_threshold,
        )
        committed = self.state.update(raw, dt)
        self.stats.update(committed, dt)

        return self._renderer.draw(
            frame, keypoints, primary, raw, committed, self.stats, fps
        )
