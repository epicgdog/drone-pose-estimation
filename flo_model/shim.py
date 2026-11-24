"""Drop-in stand-in for fmpose3d.FMPose3DInference using the Flo flow-matching lifter.

Same surface as the parts drone-pose-estimation uses:
    api = FloInference(estimator_2d=..., seed=0)
    api.setup_runtime()
    r2 = api.prepare_2d(frames)        # frames (n,H,W,3) BGR -> keypoints (1,n,17,2), scores (1,n,17), status
    r3 = api.pose_3d(r2, seed=0)       # -> poses_3d (n,17,3), valid_frames_mask (n,)

Differences from FMPose3D: COCO-17 joint order (not H36M), meters, pelvis (mean of hips 11/12) relative, camera axes
(x right, y down, z forward). The model also takes a per-joint visibility mask, built from 2D scores > mask_thr.
Camera: trained on a 960x720 sensor with focal 683 px; focal is scaled with image width for other sizes.
Runs on CPU (the sampler uses a CPU generator); the net is ~3M params, a few ms per frame.
"""
import enum
import importlib.util
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .normalize import normalize_2d
from .sampler import sample_hypotheses

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_W, TRAIN_FOCAL = 960, 683.0
K_HYPOTHESES, EULER_STEPS = 10, 10      # from Flo config.yaml `sample:`
MIN_VISIBLE = 6                          # fewer confident joints than this -> frame is skipped

COCO_JOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
                    "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
COCO_BONES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COCO_LEFT = {1, 3, 5, 7, 9, 11, 13, 15}


class Status(enum.Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY = "empty"
    INVALID = "invalid"


@dataclass
class Result2D:
    keypoints: np.ndarray                 # (1, n, 17, 2) pixels, COCO order
    scores: np.ndarray                    # (1, n, 17)
    image_size: tuple                     # (H, W)
    valid_frames_mask: np.ndarray         # (n,) bool
    status: Status = Status.SUCCESS


@dataclass
class Result3D:
    poses_3d: np.ndarray                  # (n, 17, 3) meters, pelvis-relative, COCO order
    valid_frames_mask: np.ndarray         # (n,) bool
    hypotheses: np.ndarray | None = None  # (n, K, 17, 3)
    status: Status = Status.SUCCESS
    poses_3d_world: np.ndarray = field(default=None)  # alias of poses_3d (no camera-to-world transform)


class YOLOCoco:
    """ultralytics pose model -> raw COCO-17 keypoints for the most confident person per frame."""

    def __init__(self, weights="yolo11s-pose.mlpackage", device="mps"):
        self.weights, self.device, self._model = weights, device, None

    def setup_runtime(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights, task="pose")

    def predict(self, frames):
        self.setup_runtime()
        n = frames.shape[0]
        kp = np.zeros((1, n, 17, 2), np.float32)
        sc = np.zeros((1, n, 17), np.float32)
        for i, f in enumerate(frames):
            r = self._model.predict(f, device=self.device, verbose=False)[0]
            if len(r.boxes) == 0:
                continue
            b = int(r.boxes.conf.argmax())
            kp[0, i] = r.keypoints.xy[b].cpu().numpy()
            sc[0, i] = r.keypoints.conf[b].cpu().numpy()
        return kp, sc, (sc[0] > 0).any(axis=1)


class FloInference:
    def __init__(self, model_dir=HERE, estimator_2d=None, device=None, seed=0, mask_thr=0.3):
        self.model_dir = model_dir
        self.estimator_2d = estimator_2d if estimator_2d is not None else YOLOCoco()
        self.seed = seed
        self.mask_thr = mask_thr
        self.model = None                 # `device` is accepted for API parity; sampling runs on CPU

    def setup_runtime(self):
        if self.model is None:
            spec = importlib.util.spec_from_file_location("flo_model_def", os.path.join(self.model_dir, "model_def.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.model = mod.load(os.path.join(self.model_dir, "model.pt")).eval()
        if hasattr(self.estimator_2d, "setup_runtime"):
            self.estimator_2d.setup_runtime()

    def prepare_2d(self, frames):
        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[None]
        self.setup_runtime()
        kp, sc, valid = self.estimator_2d.predict(frames)
        status = Status.SUCCESS if valid.all() else (Status.PARTIAL if valid.any() else Status.EMPTY)
        return Result2D(kp, sc, frames.shape[1:3], valid, status)

    def pose_3d(self, result_2d, seed=None, image_size=None):
        self.setup_runtime()
        kp = np.asarray(result_2d.keypoints, np.float64)[0]               # (n,17,2)
        sc = np.asarray(result_2d.scores, np.float64)[0]
        H, W = image_size or result_2d.image_size
        n = kp.shape[0]
        mask = (sc > self.mask_thr).astype(np.float64)
        ok = mask.sum(axis=1) >= MIN_VISIBLE
        poses = np.zeros((n, 17, 3), np.float64)
        hyps = np.zeros((n, K_HYPOTHESES, 17, 3), np.float64)
        if ok.any():
            focal = np.full(ok.sum(), TRAIN_FOCAL * W / TRAIN_W)
            k2, fr = normalize_2d(kp[ok], mask[ok], focal, W, H)
            cond = torch.from_numpy(np.concatenate([k2.reshape(-1, 34), mask[ok], fr], axis=1).astype(np.float32))
            with torch.no_grad():
                h = sample_hypotheses(self.model, cond, K_HYPOTHESES, EULER_STEPS,
                                      self.seed if seed is None else seed)
            h = h.reshape(-1, K_HYPOTHESES, 17, 3).double().numpy()
            hyps[ok] = h
            poses[ok] = h.mean(axis=1)
        status = Status.SUCCESS if ok.all() else (Status.PARTIAL if ok.any() else Status.EMPTY)
        return Result3D(poses, ok, hyps, status, poses)

    def predict(self, frames, seed=None):
        return self.pose_3d(self.prepare_2d(frames), seed=seed)
