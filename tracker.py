import argparse
import logging
import threading
import time
from typing import Literal

import cv2
import numpy as np
from djitellopy import Tello
from pydantic import BaseModel

from euro_filter import EuroFilter
from inference import VideoFrameInference, YOLOTestCoco
from main import wait_for_real_frame

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


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Axis:
    def __init__(self, gain: float, cap: float, deadzone: float):
        self.gain = gain
        self.cap = cap
        self.deadzone = deadzone

    def update(self, err: float) -> int:
        if abs(err) < self.deadzone:
            cmd = 0.0
        else:
            cmd = _clamp(self.gain * err, -self.cap, self.cap)
        return int(round(cmd))


class PersonTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.yaw_axis = Axis(cfg.gain_yaw, cfg.cap_yaw, 0.0)
        self.strafe_axis = Axis(cfg.gain_strafe, cfg.cap_strafe, 0.0)
        self.ud_axis = Axis(cfg.gain_ud, cfg.cap_ud, cfg.dz_y)
        self.fb_axis = Axis(cfg.gain_fb, cfg.cap_fb, cfg.dz_size)
        self.state = "idle"
        self._last_seen = None  # time of last fresh measurement
        self._last_ex = 0.0  # x error of last fresh measurement, for search dir

    def reset(self):
        self._last_seen = None
        self._last_ex = 0.0
        self.state = "idle"

    def update(self, m: Measurement | None, now: float, height_cm: float):
        cfg = self.cfg
        fresh = m is not None and now - m.t <= cfg.stale_s

        if not fresh:
            if self._last_seen is None:
                self._last_seen = now
            lost_for = now - self._last_seen
            if lost_for < cfg.lost_freeze_s:
                self.state = "frozen"
                return 0, 0, 0, 0
            self.state = "searching"
            direction = 1 if self._last_ex >= 0 else -1
            return 0, 0, 0, int(round(direction * cfg.search_yaw))

        self.state = "tracking"
        self._last_seen = m.t

        ex = m.cx - 0.5  # positive: person right of center
        ey = 0.5 - m.cy  # positive: person above center
        es = cfg.target_size - m.size  # positive: person too far away
        self._last_ex = ex

        # x errors inside the deadzone are zero, then split by lateral mode
        if abs(ex) < cfg.dz_x:
            ex = 0.0
        if cfg.lateral_mode == "yaw":
            yaw_err, strafe_err = ex, 0.0
        elif cfg.lateral_mode == "strafe":
            yaw_err, strafe_err = 0.0, ex
        else:
            w = _clamp(abs(ex) / (2 * cfg.mix_split), 0.0, 1.0)
            yaw_err, strafe_err = w * ex, (1 - w) * ex

        left_right = self.strafe_axis.update(strafe_err)
        yaw = self.yaw_axis.update(yaw_err)
        forward_backward = self.fb_axis.update(es)
        up_down = self.ud_axis.update(ey)

        # keep the drone in the altitude band
        if height_cm <= cfg.min_alt_cm:
            up_down = max(up_down, 0)
        if height_cm >= cfg.max_alt_cm:
            up_down = min(up_down, 0)

        return left_right, forward_backward, up_down, yaw


class LatestSlot:
    # holds only the newest value. writers overwrite, readers take the latest.
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def set(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


class LiftWorker:
    def __init__(self, inference: VideoFrameInference):
        self.inference = inference
        self.input = LatestSlot()  # (kpts, scores, frame_shape)
        self.output = LatestSlot()  # newest 3d pose, (17, 3)
        self.count = 0  # successful lifts, for fps logging
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lift", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        last = None
        while not self._stop.is_set():
            item = self.input.get()
            if item is None or item is last:
                time.sleep(0.005)
                continue
            last = item
            kpts, scores, frame_shape = item
            try:
                pose = self.inference.lift(kpts, scores, frame_shape)
            except Exception:
                # never let lifting take down tracking
                logger.exception("lift failed")
                continue
            self.count += 1
            if pose is not None:
                self.output.set(pose)


def describe(cmd) -> str:
    # rc command -> what the drone is doing, in words
    left_right, forward_backward, up_down, yaw = cmd
    moves = []
    if yaw:
        moves.append("turn right" if yaw > 0 else "turn left")
    if left_right:
        moves.append("strafe right" if left_right > 0 else "strafe left")
    if forward_backward:
        moves.append("closer" if forward_backward > 0 else "back off")
    if up_down:
        moves.append("up" if up_down > 0 else "down")
    return ", ".join(moves) or "holding"


class EventLog:
    # logs only tracking events: state changes and changes in what the drone is doing.
    STABLE_FRAMES = 3  # an adjustment must hold this many frames before it's logged

    def __init__(self):
        self.state = "idle"
        self.adjust = None  # last logged adjustment
        self._candidate = None
        self._count = 0

    def update(self, state, m: Measurement | None, cmd):
        if state != self.state:
            self._log_state(self.state, state, m, cmd)
            self.state = state
            self.adjust = None
            self._candidate = None
        if state != "tracking":
            return
        adjust = describe(cmd)
        if adjust == self.adjust:
            self._candidate = None
            return
        # debounce: near a deadzone edge the command can flicker frame to frame
        if adjust != self._candidate:
            self._candidate, self._count = adjust, 0
        self._count += 1
        if self._count >= self.STABLE_FRAMES:
            logger.info("adjusting: %s", adjust)
            self.adjust = adjust
            self._candidate = None

    @staticmethod
    def _log_state(old, new, m, cmd):
        if new == "tracking":
            verb = "locked on" if old == "idle" else "reacquired"
            logger.info("%s person at x=%.2f size=%.2f", verb, m.cx, m.size)
        elif new == "frozen":
            if old == "idle":
                logger.info("airborne, looking for person")
            else:
                logger.info("lost person, holding position")
        elif new == "searching":
            logger.info("searching: %s", describe(cmd))
        elif new == "idle":
            logger.info("on the ground, tracking idle")


def track(tello, frame_read, args, worker: LiftWorker):
    cfg = TrackerConfig(lateral_mode=args.lateral)
    yolo = YOLOTestCoco(device="mps")
    yolo.setup_runtime()
    mfilter = MeasurementFilter(cfg)
    tracker = PersonTracker(cfg)
    events = EventLog()

    last_frame = None

    while True:
        frame = frame_read.frame
        if frame is None or frame is last_frame:
            time.sleep(0.005)
            continue
        last_frame = frame
        # rgb to bgr since opencv and yolo expect bgr.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        det = yolo._detect_best_person(bgr)
        m = None
        if det is None:
            mfilter.reset()
        else:
            raw = measure(det.keypoints, det.scores, bgr.shape, cfg)
            m = mfilter.update(raw)
            if m is not None:
                worker.input.set((det.keypoints, det.scores, bgr.shape[:2]))

        height = tello.get_height()
        now = time.monotonic()
        if height > cfg.airborne_cm:
            cmd = tracker.update(m, now, height)
            if not args.dry_run:
                tello.send_rc_control(*cmd)
            state = tracker.state
        else:
            # on the ground: send nothing, start fresh once airborne
            tracker.reset()
            mfilter.reset()
            cmd = None
            state = "idle"

        events.update(state, m, cmd)


def main():
    parser = argparse.ArgumentParser(description="follow a person with a tello drone")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and log commands, never send them to the drone",
    )
    parser.add_argument(
        "--lateral",
        choices=["yaw", "strafe", "mix"],
        default="mix",
        help="how x error is handled: turn, strafe, or a blend (default mix)",
    )
    args = parser.parse_args()

    logger.info("Starting up!")
    lift_inference = VideoFrameInference(device="mps")
    worker = LiftWorker(lift_inference)

    tello = Tello()
    tello.connect()
    logger.info("Battery: %s", tello.get_battery())

    tello.streamon()
    frame_read = tello.get_frame_read()

    try:
        wait_for_real_frame(frame_read)
        worker.start()
        track(tello, frame_read, args, worker)
    except KeyboardInterrupt:
        logger.info("interrupted, stopping")
    finally:
        worker.stop()
        if not args.dry_run:
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                logger.exception("failed to send stop command")
        tello.streamoff()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # djitellopy logs every rc command at info; keep only its warnings and errors
    Tello.LOGGER.setLevel(logging.WARNING)
    main()
