"""Run FMPose3D on a video file and render the 2D and 3D poses side by side.

Usage:
    uv run python3 test_video.py videos/some-flight.mp4
"""

import argparse
import contextlib
import os
import sys
import tempfile
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
    import matplotlib
    matplotlib.use("Agg")
    import cv2
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from fmpose3d import FMPose3DInference

# H36M 17-joint topology: 0 hip, 1-3 right leg, 4-6 left leg, 7 spine,
# 8 thorax, 9 nose, 10 head, 11-13 left arm, 14-16 right arm.
BONES = [
    (0, 1), (1, 2), (2, 3),        # right leg
    (0, 4), (4, 5), (5, 6),        # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine -> head
    (8, 11), (11, 12), (12, 13),   # left arm
    (8, 14), (14, 15), (15, 16),   # right arm
]
JOINTS_LEFT = {4, 5, 6, 11, 12, 13}
JOINTS_RIGHT = {1, 2, 3, 14, 15, 16}

# BGR for cv2, RGB (0-1) for matplotlib
COLOR_LEFT_BGR, COLOR_RIGHT_BGR, COLOR_MID_BGR = (255, 160, 0), (0, 160, 255), (200, 200, 200)
COLOR_LEFT_RGB, COLOR_RIGHT_RGB, COLOR_MID_RGB = "#00a0ff", "#ff8000", "#909090"


def bone_side(a, b):
    """Which limb side a bone belongs to, for colouring."""
    if a in JOINTS_LEFT or b in JOINTS_LEFT:
        return "left"
    if a in JOINTS_RIGHT or b in JOINTS_RIGHT:
        return "right"
    return "mid"


def read_frames(path, stride=1, max_frames=None):
    """Decode a video into a list of BGR frames (what FMPose3D expects)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, fps / stride


def draw_pose_2d(frame, keypoints, scores=None, min_score=0.0):
    """Draw the 2D skeleton onto a copy of frame."""
    img = frame.copy()
    for a, b in BONES:
        if scores is not None and min(scores[a], scores[b]) < min_score:
            continue
        colors = {"left": COLOR_LEFT_BGR, "right": COLOR_RIGHT_BGR, "mid": COLOR_MID_BGR}
        pa = tuple(np.round(keypoints[a]).astype(int))
        pb = tuple(np.round(keypoints[b]).astype(int))
        cv2.line(img, pa, pb, colors[bone_side(a, b)], 2, cv2.LINE_AA)
    for j, (x, y) in enumerate(keypoints):
        if scores is not None and scores[j] < min_score:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def render_pose_3d(pose, size, radius=1.0, elev=15, azim=70):
    """Render one 3D pose to a BGR image of the given (height, width)."""
    height, width = size
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")

    # centre the plot on the root joint so the skeleton stays in frame
    root = pose[0]
    x, y, z = pose[:, 0] - root[0], pose[:, 1] - root[1], pose[:, 2] - root[2]

    colors = {"left": COLOR_LEFT_RGB, "right": COLOR_RIGHT_RGB, "mid": COLOR_MID_RGB}
    for a, b in BONES:
        ax.plot(
            [x[a], x[b]], [y[a], y[b]], [z[a], z[b]],
            color=colors[bone_side(a, b)], linewidth=2,
        )
    ax.scatter(x, y, z, c="white", edgecolors="black", s=12, depthshade=False)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.2)

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def label(img, text):
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Run FMPose3D on a video, show 2D + 3D pose")
    ap.add_argument("video", type=Path, help="input video file")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output mp4 (default: <video>-pose.mp4)")
    ap.add_argument("--max-frames", type=int, default=300, help="cap frames processed (default 300)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--device", default=None, help="torch device, e.g. cpu / mps / cuda")
    ap.add_argument("--weights", default=None, help="3D model .pth (default: download from HuggingFace)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the flow-matching sampler")
    ap.add_argument("--radius", type=float, default=1.0, help="half-extent of the 3D axes")
    ap.add_argument("--show", action="store_true", help="also display a live window while rendering")
    args = ap.parse_args()

    frames, fps = read_frames(args.video, stride=args.stride, max_frames=args.max_frames)
    height, width = frames[0].shape[:2]
    print(f"Decoded {len(frames)} frames @ {width}x{height}, {fps:.1f}fps")

    api = FMPose3DInference(model_weights_path=args.weights, device=args.device)
    api.setup_runtime()

    print("Running 2D pose estimation...")
    result_2d = api.prepare_2d(np.stack(frames))
    print(f"  2D: {result_2d.status.value} - {result_2d.status_message}")
    if result_2d.status.value in {"empty", "invalid"}:
        sys.exit(f"Unusable 2D pose, nothing to lift: {result_2d.status_message}")

    print("Lifting to 3D...")
    result_3d = api.pose_3d(result_2d, seed=args.seed)
    print(f"  3D: {result_3d.status.value} - {result_3d.status_message}")

    kpts_2d = result_2d.keypoints[0]      # first person -> (N, J, 2)
    scores_2d = result_2d.scores[0]       # (N, J)
    poses_3d = result_3d.poses_3d_world   # (N, J, 3)
    valid = result_3d.valid_frames_mask

    out_path = args.out or args.video.with_name(f"{args.video.stem}-pose.mp4")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    n = min(len(frames), len(kpts_2d), len(poses_3d))
    for i in range(n):
        panel_2d = label(draw_pose_2d(frames[i], kpts_2d[i], scores_2d[i]), "2D pose")
        if valid is not None and not valid[i]:
            panel_3d = label(np.zeros_like(frames[i]), "3D pose (no detection)")
        else:
            panel_3d = label(render_pose_3d(poses_3d[i], (height, width), args.radius), "3D pose")
        combined = np.hstack([panel_2d, panel_3d])
        writer.write(combined)

        if args.show:
            cv2.imshow("FMPose3D: 2D | 3D", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  rendered {i + 1}/{n}")

    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
