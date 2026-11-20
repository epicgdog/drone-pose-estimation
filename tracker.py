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
from main import RECORD_FPS, VIDEO_DIR, wait_for_real_frame
from pose_video import render_video

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
        self.output = LatestSlot()  # (newest 3d pose (17, 3), time.monotonic() it was made)
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
                self.output.set((pose, time.monotonic()))



class Recorder:
    # own thread: writes the raw camera frames at a fixed fps so the video plays back in
    # real time (frames repeat if the stream lags). the 3d pose is rendered after landing.
    def __init__(self, frame_read, fps: float = RECORD_FPS):
        self.frame_read = frame_read
        self.fps = fps
        h, w = frame_read.frame.shape[:2]
        self.size = (w, h)
        VIDEO_DIR.mkdir(exist_ok=True)
        self.path = VIDEO_DIR / time.strftime("tello-track-%Y%m%d-%H%M%S.mp4")
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {self.path}")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="record", daemon=True)

    def start(self):
        logger.info("recording to %s", self.path)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.writer.release()
        logger.info("saved recording %s", self.path)

    def _run(self):
        interval = 1.0 / self.fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                frame = self.frame_read.frame
                if frame is not None and (frame.shape[1], frame.shape[0]) == self.size:
                    self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception:
                # never let recording take down tracking
                logger.exception("recording frame failed")
            next_t += interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # fell behind, don't try to catch up


def track(tello, frame_read, args, worker: LiftWorker | None, until: float | None = None):
    # until: time.monotonic() deadline to stop tracking at, None runs until ctrl+c
    cfg = TrackerConfig(lateral_mode=args.lateral)
    yolo = YOLOTestCoco(device="mps")
    yolo.setup_runtime()
    mfilter = MeasurementFilter(cfg)
    tracker = PersonTracker(cfg)
    events = EventLog()

    last_frame = None

    while until is None or time.monotonic() < until:
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
            if m is not None and worker is not None:
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
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="don't save the flight video to videos/",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="don't run the 3d pose on the recording after landing",
    )
    parser.add_argument(
        "--live-lift",
        action="store_true",
        help="also run the 3d lift live during flight (slows tracking, output unused for now)",
    )
    parser.add_argument(
        "--flight-s",
        type=float,
        default=60,
        help="seconds to track after takeoff before landing (default 60, 0 = until ctrl+c)",
    )
    args = parser.parse_args()

    logger.info("Starting up!")
    worker = LiftWorker(VideoFrameInference(device="mps")) if args.live_lift else None

    tello = Tello()
    tello.connect()
    logger.info("Battery: %s", tello.get_battery())

    tello.streamon()
    frame_read = tello.get_frame_read()

    # the laptop is the only controller: it takes off and lands. keep the phone app
    # disconnected, its stick packets fight the rc commands sent here.
    took_off = False
    recorder = None
    try:
        wait_for_real_frame(frame_read)
        if worker is not None:
            worker.start()
        if not args.no_record:
            recorder = Recorder(frame_read)
            recorder.start()
        if not args.dry_run:
            logger.info("taking off, ctrl+c to land")
            took_off = True  # set first so ctrl+c mid takeoff still lands
            tello.takeoff()
        until = None
        if not args.dry_run and args.flight_s > 0:
            until = time.monotonic() + args.flight_s
            logger.info("landing in %.0f s", args.flight_s)
        track(tello, frame_read, args, worker, until)
        if until is not None:
            logger.info("flight time up, landing")
    except KeyboardInterrupt:
        logger.info("interrupted, %s", "landing" if took_off else "stopping")
    finally:
        if worker is not None:
            worker.stop()
        if not args.dry_run:
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                logger.exception("failed to send stop command")
        if took_off:
            try:
                tello.land()
            except Exception:
                logger.exception("land failed, land it from the phone app or catch it")
        if recorder is not None:
            recorder.stop()
        tello.streamoff()

    # drone is down: now there's time for the full 3d pass on every recorded frame
    if recorder is not None and not args.no_render:
        logger.info("rendering 3d pose, ctrl+c to skip (run pose_video.py on it later)")
        try:
            render_video(recorder.path, inference=worker.inference if worker else None)
        except KeyboardInterrupt:
            logger.info("skipped render: uv run python3 pose_video.py %s", recorder.path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # djitellopy logs every rc command at info; keep only its warnings and errors
    Tello.LOGGER.setLevel(logging.WARNING)
    main()
