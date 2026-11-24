import os

# fmpose3d fetches its weights through hugging face, which checks online on every start.
# the tello's wifi has no internet, so use the cached copy (~/.cache/huggingface) only.
# set before anything imports huggingface_hub. to download the weights the first time:
#   HF_HUB_OFFLINE=0 uv run python3 main.py --video <some.mp4>
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402  (after the env var on purpose)
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from flo_model.shim import FloInference, Result2D  # noqa: E402


NUM_JOINTS = 17

class COCOSkeleton:
    """COCO 17-joint layout: what YOLO outputs and what Flo predicts (no spine/thorax/head joints).
    0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
    11-12 hips, 13-14 knees, 15-16 ankles. odd = person's left, even = right."""

    NAME = "coco"
    BONES = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # face, ears to shoulders (coco has no neck)
        (5, 6), (5, 11), (6, 12), (11, 12),  # torso
        (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    ]
    JOINT_NAMES = [
        "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle",
    ]
    JOINTS_LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
    JOINTS_RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
    ROOT = (11, 12)  # pelvis = mean of the hips
    FOREARMS = ((7, 9), (8, 10))  # (left elbow, wrist), (right elbow, wrist)


class H36MSkeleton:
    """H36M 17-joint layout: what FMPose3D predicts.
    0 hip, 1-3 right leg, 4-6 left leg, 7 spine, 8 thorax, 9 nose, 10 head, 11-13 left arm, 14-16 right arm."""

    NAME = "h36m"
    BONES = [
        (0, 1), (1, 2), (2, 3),  # hip - right leg
        (0, 4), (4, 5), (5, 6),  # hip - left leg
        (0, 7), (7, 8), (8, 9), (9, 10),  # hip - spine - thorax - nose - head
        (8, 11), (11, 12), (12, 13),  # thorax - left arm
        (8, 14), (14, 15), (15, 16),  # thorax - right arm
    ]
    JOINT_NAMES = [
        "hip", "r_hip", "r_knee", "r_foot", "l_hip", "l_knee", "l_foot", "spine", "thorax",
        "nose", "head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
    ]
    JOINTS_LEFT = {4, 5, 6, 11, 12, 13}
    JOINTS_RIGHT = {1, 2, 3, 14, 15, 16}
    ROOT = 0  # hip joint
    FOREARMS = ((12, 13), (15, 16))


SKELETONS = {"flo": COCOSkeleton, "fmpose3d": H36MSkeleton}  # backend -> its 3d joint layout


# result of a yolo predict type, assuming one person is detected
class PersonDetection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    box_conf: float  # confidence of the person's bounding box
    keypoints: NDArray[np.float32]  # (17, 2): x, y in source-frame pixels
    scores: NDArray[np.float32]  # (17,): confidence per joint


# class to load in our own custom 2d pose estimator.
# these classes need a setup_runtime method and a predict method so that
# the Flo lifter (flo_model) can use it as its 2d estimator.
class YOLOTestCoco:
    # default is the CoreML FP16 export (runs on the Neural Engine, ~1.6x faster
    # than the .pt on mps). create it once with:
    #   uv run --with "numpy<=2.3.5" --with coremltools python3 -c \
    #     "from ultralytics import YOLO; YOLO('yolo11s-pose.pt').export(format='coreml', half=True)"
    def __init__(self, weights="yolo11s-pose.mlpackage", device="mps"):
        self.weights = weights  # ultralytics pose weights, .pt or CoreML .mlpackage
        self.device = device  # torch device, e.g. cpu / mps
        self._model = None

    def setup_runtime(self):
        if self._model is None:
            # task must be given for exported models, they can't infer it
            self._model = YOLO(self.weights, task="pose")

    def _detect_best_person(self, frame) -> PersonDetection | None:
        result = self._model.predict(frame, device=self.device, verbose=False)[0]
        if len(result.boxes) == 0:
            return None
        best = int(result.boxes.conf.argmax())
        return PersonDetection(
            box_conf=float(result.boxes.conf[best]),
            keypoints=result.keypoints.xy[best].cpu().numpy(),
            scores=result.keypoints.conf[best].cpu().numpy(),
        )

    def predict(self, frames):

        # frames: the frames to predict
        # the use case for the drone we only care about 1 frame at a time.
        # so when we loop, we only ever just check one frame.
        # this probably will be a point of optimization, as this is a blocker.
        # inference takes longer than 1/30th of a second, so we won't be able to
        # achieve the stream throughput.

        self.setup_runtime()
        n = frames.shape[0]  # this is the batch number.

        # the dimensions are (1, n, 17, 2)
        # 1 = number of people
        # n = batch number
        # 17 = pose joints?
        # 2 = coordiantes, x and y.
        coco_kpts = np.zeros((1, n, NUM_JOINTS, 2), dtype=np.float32)
        coco_scores = np.zeros((1, n, NUM_JOINTS), dtype=np.float32)
        for i, frame in enumerate(frames):
            person = self._detect_best_person(frame)
            if person is None:
                continue  # leave this frame as zeros
            coco_kpts[0, i] = person.keypoints
            coco_scores[0, i] = person.scores

        # this mask basically is of every frame and if they have valid scores or not.
        valid_frames_mask = np.any(coco_scores > 0, axis=(0, 2))
        return coco_kpts, coco_scores, valid_frames_mask


class VideoFrameInference:
    """3d lifting. backend="flo" (default): Flo flow-matching lifter in flo_model/, cpu, coco-17 out.
    backend="fmpose3d": FMPose3D, h36m-17 out. self.skeleton says which joint layout lift() returns."""

    def __init__(self, device=None, seed=0, backend="flo"):
        self.device = device  # torch device for fmpose3d, flo always samples on cpu
        self.seed = seed  # seed for the flow-matching sampler
        self.backend = backend

        if backend == "flo":
            self.skeleton = COCOSkeleton
            self.api = FloInference(estimator_2d=YOLOTestCoco(), device="cpu", seed=seed)
        elif backend == "fmpose3d":
            from fmpose3d import FMPose3DInference

            self.skeleton = H36MSkeleton
            self.api = FMPose3DInference(
                model_weights_path=None, device=device, estimator_2d=YOLOTestCoco()
            )
        else:
            raise ValueError(f"unknown backend {backend!r}, expected 'flo' or 'fmpose3d'")
        # load the models once, here. pose_3d calls setup_runtime() on every frame, and
        # fmpose3d reloads its weights from disk each time (~40 ms). make it a no-op.
        self.api.setup_runtime()
        self.api.setup_runtime = lambda: None

    def lift(self, coco_kpts, coco_scores, frame_shape):
        # coco_kpts: (17, 2) pixels in the source frame, coco_scores: (17,)
        # frame_shape: (h, w) of that frame, needed to normalize the 2d points.
        # returns (17, 3) pose in self.skeleton's joint order, meters, root at the origin, z up
        # (x right, y away from the camera, z up), or None if too few joints are visible.
        kpts = np.asarray(coco_kpts, dtype=np.float32).reshape(1, 1, NUM_JOINTS, 2)
        scores = np.asarray(coco_scores, dtype=np.float32).reshape(1, 1, NUM_JOINTS)
        if not np.any(scores > 0):
            return None
        image_size = (int(frame_shape[0]), int(frame_shape[1]))
        if self.backend == "flo":
            return self._lift_flo(kpts, scores, image_size)
        return self._lift_fmpose3d(kpts, scores, image_size)

    def _lift_flo(self, kpts, scores, image_size):
        result_2d = Result2D(
            keypoints=kpts, scores=scores, image_size=image_size,
            valid_frames_mask=np.array([True]),
        )
        result_3d = self.api.pose_3d(result_2d, seed=self.seed)
        valid = result_3d.valid_frames_mask
        if valid is not None and not valid[0]:
            return None
        # flo gives camera axes (x right, y down, z forward), pelvis relative
        cam = np.asarray(result_3d.poses_3d[0], dtype=np.float64)
        return np.stack([cam[:, 0], cam[:, 2], -cam[:, 1]], axis=1)

    def _lift_fmpose3d(self, kpts, scores, image_size):
        from fmpose3d import Pose2DResult
        from fmpose3d.lib.preprocess import h36m_coco_format

        # fmpose3d wants h36m 2d. h36m_coco_format drops the person when all keypoints are zero
        h36m_kpts, h36m_scores, _ = h36m_coco_format(kpts, scores)
        if h36m_kpts.shape[0] == 0:
            return None
        result_2d = Pose2DResult(
            keypoints=h36m_kpts, scores=h36m_scores, image_size=image_size,
            valid_frames_mask=np.any(h36m_scores > 0, axis=(0, 2)),
        )
        result_3d = self.api.pose_3d(result_2d, seed=self.seed)
        valid = result_3d.valid_frames_mask
        if valid is not None and not valid[0]:
            return None
        # turned 180 deg about z from flo's frame. rotate (not mirror) so left/right stay right.
        pose = np.asarray(result_3d.poses_3d_world[0], dtype=np.float64)
        pose = pose - pose[H36MSkeleton.ROOT]  # hip relative, like flo's output
        pose[:, :2] *= -1
        return pose
