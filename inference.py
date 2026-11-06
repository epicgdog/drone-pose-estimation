import time

import numpy as np
from fmpose3d import FMPose3DInference
from fmpose3d.lib.preprocess import h36m_coco_format
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from ultralytics import YOLO

from euro_filter import EuroFilter

NUM_JOINTS = 17  # COCO and H36M both use 17 joints

# H36M 17-joint topology: 0 hip, 1-3 right leg, 4-6 left leg, 7 spine,
# 8 thorax, 9 nose, 10 head, 11-13 left arm, 14-16 right arm.
BONES = [
    (0, 1),  # hip - r_hip
    (1, 2),  # r_hip - r_knee
    (2, 3),  # r_knee - r_foot
    (0, 4),  # hip - l_hip
    (4, 5),  # l_hip - l_knee
    (5, 6),  # l_knee - l_foot
    (0, 7),  # hip - spine
    (7, 8),  # spine - thorax
    (8, 9),  # thorax - nose
    (9, 10),  # nose - head
    (8, 11),  # thorax - l_shoulder
    (11, 12),  # l_shoulder - l_elbow
    (12, 13),  # l_elbow - l_wrist
    (8, 14),  # thorax - r_shoulder
    (14, 15),  # r_shoulder - r_elbow
    (15, 16),  # r_elbow - r_wrist
]

# names of the joints, line up with indices of bones.
# EX: bones[0] = hip joint, connecting 0 (hip) and 1 (lower right leg)
JOINT_NAMES = [
    "hip",
    "r_hip",
    "r_knee",
    "r_foot",
    "l_hip",
    "l_knee",
    "l_foot",
    "spine",
    "thorax",
    "nose",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
]

# indices for which joints are left and right sides.
# JOINT_NAMES[1] -> r_hip
# JOINT_NAMES[4] -> l_hip
JOINTS_LEFT = {4, 5, 6, 11, 12, 13}
JOINTS_RIGHT = {1, 2, 3, 14, 15, 16}


# result of a yolo predict type, assuming one person is detected
class PersonDetection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    box_conf: float  # confidence of the person's bounding box
    keypoints: NDArray[np.float32]  # (17, 2): x, y in source-frame pixels
    scores: NDArray[np.float32]  # (17,): confidence per joint


# class to load in our own custom 2d pose estimator.
# these classes need a setup_runtime method and a predict method so that
# fmpose3d api can use it.
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

        # h36m_coco_format drops the person entirely when all frames are empty
        h36m_kpts, h36m_scores, _ = h36m_coco_format(coco_kpts, coco_scores)
        if h36m_kpts.shape[0] == 0:
            h36m_kpts = np.zeros((1, n, NUM_JOINTS, 2), dtype=np.float32)
            h36m_scores = np.zeros((1, n, NUM_JOINTS), dtype=np.float32)

        # this mask basically is of every frame and if they have valid scores or not.
        valid_frames_mask = np.any(h36m_scores > 0, axis=(0, 2))
        return h36m_kpts, h36m_scores, valid_frames_mask


class VideoFrameInference:
    def __init__(self, device=None, weights=None, seed=0):
        self.device = device  # torch device, e.g. cpu / mps / cuda
        self.weights = weights  # 3D model .pth, None downloads from HuggingFace
        self.seed = seed  # seed for the flow-matching sampler

        self.api = FMPose3DInference(
            model_weights_path=self.weights,
            device=self.device,
            estimator_2d=YOLOTestCoco(),
        )
        # load the models once, here.
        self.api.setup_runtime()
        # prepare_2d and pose_3d call setup_runtime() on every frame, and it
        # reloads the 3D weights from disk each time (~40 ms). make it a no-op.
        self.api.setup_runtime = lambda: None

        # euro filter
        self.euro_filter = EuroFilter()
        self.old_time = 0

    def infer(self, frame):

        result_2d = self.api.prepare_2d(frame[np.newaxis])
        if result_2d.status.value in {"empty", "invalid"}:
            return None

        if self.old_time == 0:
            # use default for the first frame
            self.old_time = time.monotonic()
        else:
            curr_time = time.monotonic()
            dt = curr_time - self.old_time
            self.euro_filter.set_rate(1.0 / dt)
            self.old_time = curr_time
        result_2d.keypoints = self.euro_filter.filter(result_2d.keypoints)

        result_3d = self.api.pose_3d(result_2d, seed=self.seed)
        valid = result_3d.valid_frames_mask
        if valid is not None and not valid[0]:
            return None
