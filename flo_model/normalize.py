"""2D keypoint normalisation, copied verbatim from Flo checker/datagen.py (normalize_2d).
COCO-17 order. Must match training exactly."""
import numpy as np

ROOT_JOINTS = (11, 12)  # left hip, right hip


def normalize_2d(uv: np.ndarray, mask: np.ndarray, focal: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """uv (n,17,2) px, mask (n,17), focal (n,) -> kp2d (n,17,2), frame (n,3).

    kp2d: minus the 2D root (mean of visible hips; mean of visible joints if both hips are hidden), divided by
    the bounding-box diagonal of visible joints, hidden joints zeroed.
    frame: where the person is in the image, in focal units: ((root_u - W/2)/f, (root_v - H/2)/f, diag/f)."""
    hip_mask = mask[:, list(ROOT_JOINTS)]
    hip_count = hip_mask.sum(axis=1)
    hip_root = (uv[:, list(ROOT_JOINTS)] * hip_mask[..., None]).sum(axis=1) / np.maximum(hip_count, 1)[:, None]
    vis_count = np.maximum(mask.sum(axis=1), 1)
    vis_mean = (uv * mask[..., None]).sum(axis=1) / vis_count[:, None]
    root2d = np.where((hip_count > 0)[:, None], hip_root, vis_mean)
    big = 1e9
    uv_min = np.where(mask[..., None] > 0, uv, big).min(axis=1)
    uv_max = np.where(mask[..., None] > 0, uv, -big).max(axis=1)
    diag = np.linalg.norm(uv_max - uv_min, axis=1)
    diag = np.where(np.isfinite(diag) & (diag > 1e-6), diag, 1.0)
    kp2d = (uv - root2d[:, None, :]) / diag[:, None, None] * mask[..., None]
    frame = np.stack([(root2d[:, 0] - W / 2) / focal, (root2d[:, 1] - H / 2) / focal, diag / focal], axis=1)
    return kp2d, frame
