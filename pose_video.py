"""Run 2D -> 3D pose on a recorded video and write it with the 3D pose next to it.

    uv run python3 pose_video.py videos/tello-track-20261119-180000.mp4 [more.mp4 ...]

writes <name>-pose.mp4 next to each input: camera frame on the left, 3d skeleton on the right.
tracker.py runs this on its recording after landing.
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from euro_filter import EuroFilter
from inference import BONES, JOINTS_LEFT, JOINTS_RIGHT, VideoFrameInference, YOLOTestCoco

logger = logging.getLogger(__name__)

# bgr. match the old pose videos: left blue, right orange.
COLOR_LEFT, COLOR_RIGHT, COLOR_MID = (255, 160, 0), (0, 160, 255), (200, 200, 200)


def render_pose(pose, size, azim_deg=-35.0, elev_deg=15.0, span_m=2.2):
    # (17, 3) h36m pose, meters, z up -> size x size bgr image. orthographic view,
    # rotated around z by azim and tilted by elev, centered on the hip.
    img = np.full((size, size, 3), 30, dtype=np.uint8)
    if pose is None:
        cv2.putText(img, "no 3d pose", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (160, 160, 160), 2, cv2.LINE_AA)
        return img
    p = np.asarray(pose, dtype=np.float64) - np.asarray(pose[0], dtype=np.float64)
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    rot_z = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    p = p @ rot_z.T @ rot_x.T
    scale = size / span_m
    # screen: x right, y down. world z up -> screen up.
    uv = np.stack([size / 2 + p[:, 0] * scale, size / 2 - p[:, 2] * scale], axis=1)
    uv = np.round(uv).astype(int)

    # floor line under the lowest joint, for a sense of up
    floor = int(np.clip(uv[:, 1].max() + 10, 0, size - 1))
    cv2.line(img, (20, floor), (size - 20, floor), (70, 70, 70), 1, cv2.LINE_AA)

    for j1, j2 in BONES:
        if j1 in JOINTS_LEFT or j2 in JOINTS_LEFT:
            color = COLOR_LEFT
        elif j1 in JOINTS_RIGHT or j2 in JOINTS_RIGHT:
            color = COLOR_RIGHT
        else:
            color = COLOR_MID
        cv2.line(img, tuple(uv[j1]), tuple(uv[j2]), color, 3, cv2.LINE_AA)
    for x, y in uv:
        cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img, "3d pose", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (200, 200, 200), 2, cv2.LINE_AA)
    return img


def render_video(src, dst=None, inference=None, device="mps") -> Path:
    """Every frame of src -> yolo -> euro filter -> 3d lift -> side-by-side video at dst."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.stem + "-pose.mp4")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if inference is None:
        inference = VideoFrameInference(device=device)
    yolo = YOLOTestCoco(device=device)
    yolo.setup_runtime()
    euro = EuroFilter()
    euro.set_rate(fps)  # offline, so frames are exactly 1/fps apart

    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w + h, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {dst}")

    logger.info("rendering 3d pose for %s (%d frames)", src, total)
    start = last_log = time.monotonic()
    done = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            pose = None
            det = yolo._detect_best_person(bgr)
            if det is None:
                euro = EuroFilter()  # person gone, don't smooth across the gap
                euro.set_rate(fps)
            else:
                kpts = euro.filter(det.keypoints).astype(np.float32)
                pose = inference.lift(kpts, det.scores, bgr.shape[:2])
            writer.write(np.hstack([bgr, render_pose(pose, h)]))
            done += 1
            if time.monotonic() - last_log >= 5:
                last_log = time.monotonic()
                logger.info("  %d/%d frames", done, total)
    finally:
        cap.release()
        writer.release()
    logger.info("saved %s (%d frames in %.0f s)", dst, done, time.monotonic() - start)
    return dst


def main():
    parser = argparse.ArgumentParser(description="render recorded videos with their 3d pose")
    parser.add_argument("videos", nargs="+", help="input video files")
    args = parser.parse_args()
    inference = VideoFrameInference(device="mps")  # load the models once for all videos
    for path in args.videos:
        render_video(path, inference=inference)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
