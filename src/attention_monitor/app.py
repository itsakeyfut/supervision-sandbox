import time

import cv2

from attention_monitor.capture import WebcamSource
from attention_monitor.config import Config
from attention_monitor.pipeline import Pipeline


def main():
    config = Config()
    source = WebcamSource(config.camera_index)
    pipeline = Pipeline(config)

    prev = time.perf_counter()
    fps = 0.0
    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            out = pipeline.process(frame, dt, fps)
            cv2.imshow(config.window_name, out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
        print(pipeline.stats.summary())
