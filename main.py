import contextlib
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


@contextlib.contextmanager
def filtered_stderr():
    """Drop the objc duplicate-class warnings that appear because cv2 and av
    each link their own bundled copy of ffmpeg. All other stderr is kept."""
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+") as buf:
        try:
            sys.stderr.flush()
            os.dup2(buf.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            buf.seek(0)
            for line in buf:
                if line.startswith("objc[") and (
                    "is implemented in both" in line or "One of the duplicates" in line
                ):
                    continue
                sys.stderr.write(line)


with filtered_stderr():
    import cv2
    from djitellopy import Tello

VIDEO_DIR = Path(__file__).parent / "videos"
RECORD_FPS = 30
# djitellopy hands out this black placeholder until the decoder produces a real
# frame, so wait it out before sizing the VideoWriter.
PLACEHOLDER_SHAPE = (300, 400)


def wait_for_real_frame(frame_read, timeout=10):
    """Block until the stream delivers a frame that isn't the placeholder."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        frame = frame_read.frame
        if frame is not None and frame.shape[:2] != PLACEHOLDER_SHAPE:
            return frame
        time.sleep(0.05)
    raise RuntimeError(f"No video frames within {timeout}s")


def record(frame_read, writer, fps=RECORD_FPS):
    """Write the latest frame at a fixed cadence until interrupted.

    Fixed cadence keeps playback matched to wall clock: a stalled stream
    repeats the last frame rather than compressing the timeline.

    Returns the number of frames written.
    """
    interval = 1 / fps
    next_write = time.perf_counter()
    written = 0
    try:
        while True:
            frame = frame_read.frame
            if frame is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                written += 1
            next_write += interval
            time.sleep(max(0.0, next_write - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    return written


def main():
    print("Starting up!")
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}")

    tello.streamon()
    frame_read = tello.get_frame_read()

    first_frame = wait_for_real_frame(frame_read)
    height, width = first_frame.shape[:2]

    VIDEO_DIR.mkdir(exist_ok=True)
    out_path = VIDEO_DIR / f"tello-{datetime.now():%Y%m%d-%H%M%S}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")
    print(f"Recording {width}x{height} @ {RECORD_FPS}fps -> {out_path}")
    print("Press Ctrl-C to stop.")

    try:
        frames = record(frame_read, writer)
    finally:
        writer.release()
        tello.streamoff()

    print(f"Saved {out_path} ({frames} frames, {frames / RECORD_FPS:.1f}s)")


if __name__ == "__main__":
    main()
