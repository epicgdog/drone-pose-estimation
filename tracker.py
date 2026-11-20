import logging
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel

from euro_filter import EuroFilter

logger = logging.getLogger(__name__)

# coco-17 indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


class TrackerConfig(BaseModel):
    # measurement
    kpt_conf: float = 0.3  # joints below this score are ignored
    target_size: float = 0.25  # wanted torso size, fraction of frame height
    outlier_jump: float = 0.4  # reject size jumps bigger than this fraction

    # deadzones, errors inside these are treated as zero
    dz_x: float = 0.05
    dz_y: float = 0.08
    dz_size: float = 0.03

    # gains, command units per unit error
    gain_yaw: float = 120
    gain_strafe: float = 80
    gain_ud: float = 100
    gain_fb: float = 200

    # caps, tello rc range is -100..100
    cap_yaw: float = 60
    cap_strafe: float = 40
    cap_ud: float = 40
    cap_fb: float = 40

    slew: float = 15  # max command change per step
    lateral_mode: Literal["yaw", "strafe", "mix"] = "mix"
    mix_split: float = 0.2  # |x error| where the blend is half yaw, half strafe

    # lost person handling
    lost_freeze_s: float = 1.0  # hover this long before searching
    search_yaw: float = 25
    stale_s: float = 0.3  # no measurement this old counts as lost

    # altitude, cm
    min_alt_cm: float = 50
    max_alt_cm: float = 250
    airborne_cm: float = 30


class Measurement(BaseModel):
    cx: float  # person center x, fraction of frame width
    cy: float  # person center y, fraction of frame height
    size: float  # torso length, fraction of frame height
    t: float  # time.monotonic()


def measure(kpts, scores, frame_shape, cfg: TrackerConfig) -> Measurement | None:
    # kpts: (17, 2) pixels, scores: (17,), frame_shape: (h, w)
    h, w = frame_shape[:2]

    def midpoint(indices):
        pts = [kpts[i] for i in indices if scores[i] >= cfg.kpt_conf]
        if not pts:
            return None
        return np.mean(np.stack(pts), axis=0)

    sh_mid = midpoint([L_SHOULDER, R_SHOULDER])
    hip_mid = midpoint([L_HIP, R_HIP])
    if sh_mid is None and hip_mid is None:
        return None

    # cx from hips, fallback shoulders
    anchor_x = hip_mid if hip_mid is not None else sh_mid
    cx = anchor_x[0] / w

    # cy is the torso middle, fallback shoulder mid
    if sh_mid is not None and hip_mid is not None:
        cy = (sh_mid[1] + hip_mid[1]) / 2 / h
    else:
        cy = (sh_mid if sh_mid is not None else hip_mid)[1] / h

    # size from torso length, fallback shoulder width
    if sh_mid is not None and hip_mid is not None:
        size = np.linalg.norm(sh_mid - hip_mid) / h
    elif (
        scores[L_SHOULDER] >= cfg.kpt_conf and scores[R_SHOULDER] >= cfg.kpt_conf
    ):
        shoulder_width = np.linalg.norm(kpts[L_SHOULDER] - kpts[R_SHOULDER])
        size = shoulder_width * 1.6 / h
    else:
        # hips only, no way to get a size
        return None

    return Measurement(cx=float(cx), cy=float(cy), size=float(size), t=time.monotonic())


class MeasurementFilter:
    MAX_REJECTS = 5  # after this many rejects in a row, accept the jump

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.euro = EuroFilter()
        self.prev_size = None  # size of the last accepted measurement
        self.prev_t = None  # time of the last accepted measurement
        self.rejects = 0

    def reset(self):
        self.euro = EuroFilter()
        self.prev_size = None
        self.prev_t = None
        self.rejects = 0

    def update(self, m: Measurement | None) -> Measurement | None:
        if m is None:
            return None

        # reject size jumps, unless it keeps happening (person really moved)
        if self.prev_size is not None and self.prev_size > 0:
            jump = abs(m.size - self.prev_size) / self.prev_size
            if jump > self.cfg.outlier_jump and self.rejects < self.MAX_REJECTS:
                self.rejects += 1
                logger.debug("rejected size jump %.2f", jump)
                return None
        self.rejects = 0
        self.prev_size = m.size

        # one euro filter, sample rate from the time between accepted measurements
        if self.prev_t is not None:
            dt = m.t - self.prev_t
            if dt > 0:
                self.euro.set_rate(1.0 / dt)
        self.prev_t = m.t

        cx, cy, size = self.euro.filter(np.array([m.cx, m.cy, m.size], dtype=np.float64))
        return Measurement(cx=float(cx), cy=float(cy), size=float(size), t=m.t)
