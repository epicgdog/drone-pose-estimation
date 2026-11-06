import logging
import time
from pathlib import Path

import cv2
from djitellopy import Tello

from inference import VideoFrameInference

logger = logging.getLogger(__name__)

VIDEO_DIR = Path(__file__).parent / "videos"
RECORD_FPS = 30
INTERVAL = 1 / RECORD_FPS
PLACEHOLDER_SHAPE = (
    300,
    400,
)  # tello has a black image 300x400 while waiting for frames.
FRAME_TIMEOUT = 10  # in seconds


# arg is frame_read of type BackgroundFrameRead
# you could pass the tello and do tello.get_frame_read().
# however, the tello api auto-refreshes the frame, so passing in the object
# means we have the stream going. we only ever need to do tello.get_frame_read() once.
def wait_for_real_frame(frame_read):
    deadline = time.perf_counter() + FRAME_TIMEOUT
    while time.perf_counter() < deadline:
        frame = frame_read.frame
        if frame is not None and frame.shape[:2] != PLACEHOLDER_SHAPE:
            return frame
        time.sleep(0.05)
    raise RuntimeError(f"No video frames within {FRAME_TIMEOUT}s")


def stream(frame_read, inference, show=True):

    target_time = time.perf_counter() + INTERVAL
    fps_frames = 0
    fps_start = time.perf_counter()

    total_elapsed = 0
    total_frames = 0

    avg_fps = 0

    try:
        while True:
            frame = frame_read.frame
            if frame is not None:
                # rgb to bgr since opencv renders in bgr.
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                inference.infer(bgr)

                # benchmark: frames processed per wall-clock second
                fps_frames += 1
                elapsed = time.perf_counter() - fps_start
                if elapsed >= 1.0:
                    fps = fps_frames / elapsed
                    logger.info("%.2f fps", fps)

                    total_elapsed += elapsed
                    total_frames += fps_frames
                    avg_fps = total_frames / total_elapsed

                    fps_frames = 0
                    fps_start = time.perf_counter()

                if show:
                    cv2.imshow("Tello Stream", bgr)

            # quit on q
            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break

            target_time += INTERVAL
            diff = target_time - time.perf_counter()
            if diff > 0:
                time.sleep(diff)

            # else, processing took too long and just grab the next frame.
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("avg fps: %.2f", avg_fps)


def main():
    logger.info("Starting up!")
    inference = VideoFrameInference(device="mps")
    tello = Tello()
    tello.connect()
    logger.info("Battery: %s", tello.get_battery())

    tello.streamon()
    frame_read = tello.get_frame_read()

    try:
        wait_for_real_frame(frame_read)
        stream(frame_read, inference)
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
