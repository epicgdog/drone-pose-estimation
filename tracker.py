import argparse
import csv
import logging
import threading
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from djitellopy import Tello
from pydantic import BaseModel

from inference import VideoFrameInference, YOLOTestCoco
from main import wait_for_real_frame

logger = logging.getLogger(__name__)

# coco-17 indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

RC_MAX = 100  # tello rc range is -100..100
LOG_DIR = Path("logs")  # per-frame tracking logs


class TrackerConfig(BaseModel):
    # measurement
    kpt_conf: float = 0.3  # joints below this score are ignored
    target_size: float = 0.25  # wanted torso size, fraction of frame height

    # deadzones, errors inside these are treated as zero
    dz_x: float = 0.05
    dz_y: float = 0.08
    dz_size: float = 0.03

    # gains, command units per unit error
    gain_yaw: float = 360
    gain_strafe: float = 120
    gain_ud: float = 150
    gain_fb: float = 300

    # d gains, command units per (unit error per second). about 0.1 s of
    # lookahead: d_gain / gain. no filter, so raw keypoint noise shows up here.
    d_gain_yaw: float = 36
    d_gain_strafe: float = 12
    d_gain_ud: float = 15
    d_gain_fb: float = 30

    # caps on the risky axes, yaw and strafe only limited by the rc range
    cap_fb: float = 40
    cap_ud: float = 40

    lateral_mode: Literal["yaw", "strafe", "mix"] = "mix"
    mix_split: float = 0.2  # |x error| where the blend is half yaw, half strafe

    # lost person handling
    hold_s: float = 0.3  # on a dropout, repeat the last command this long
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


def measure(kpts, scores, frame_shape, cfg: TrackerConfig, t: float | None = None) -> Measurement | None:
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

    t = time.monotonic() if t is None else t
    return Measurement(cx=float(cx), cy=float(cy), size=float(size), t=t)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Axis:
    def __init__(self, gain: float, deadzone: float, d_gain: float = 0.0, cap: float = RC_MAX):
        self.gain = gain
        self.d_gain = d_gain
        self.deadzone = deadzone
        self.cap = min(cap, RC_MAX)

    def update(self, err: float, derr: float = 0.0) -> int:
        # d term opposes how fast the error changes, so it brakes when closing in
        if abs(err) < self.deadzone:
            cmd = 0.0
        else:
            cmd = max(-self.cap, min(self.cap, self.gain * err + self.d_gain * derr))
        return int(round(cmd))


class PersonTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.yaw_axis = Axis(cfg.gain_yaw, 0.0, cfg.d_gain_yaw)
        self.strafe_axis = Axis(cfg.gain_strafe, 0.0, cfg.d_gain_strafe)
        self.ud_axis = Axis(cfg.gain_ud, cfg.dz_y, cfg.d_gain_ud, cfg.cap_ud)
        self.fb_axis = Axis(cfg.gain_fb, cfg.dz_size, cfg.d_gain_fb, cfg.cap_fb)
        self.state = "idle"
        self._last_seen = None  # time of last fresh measurement
        self._last_cmd = None  # last tracking command, repeated on short dropouts
        self._prev_err = None  # (t, errors) of the last measurement, for the d terms

    def reset(self):
        self._last_seen = None
        self._last_cmd = None
        self._prev_err = None
        self.state = "idle"

    def _alt_band(self, up_down: int, height_cm: float) -> int:
        # keep the drone in the altitude band
        if height_cm <= self.cfg.min_alt_cm:
            up_down = max(up_down, 0)
        if height_cm >= self.cfg.max_alt_cm:
            up_down = min(up_down, 0)
        return up_down

    def _rates(self, t: float, errors: tuple[float, ...]) -> tuple[float, ...]:
        # how fast each error changes, per second. 0 on the first frame and after a gap
        # longer than stale_s, so there are no slopes across a dropout.
        rates = tuple(0.0 for _ in errors)
        if self._prev_err is not None:
            prev_t, prev_errors = self._prev_err
            dt = t - prev_t
            if 0 < dt <= self.cfg.stale_s:
                rates = tuple((e - p) / dt for e, p in zip(errors, prev_errors))
        self._prev_err = (t, errors)
        return rates

    def update(self, m: Measurement | None, now: float, height_cm: float):
        cfg = self.cfg
        fresh = m is not None and now - m.t <= cfg.stale_s

        if not fresh:
            if self._last_seen is None:
                self._last_seen = now
            lost_for = now - self._last_seen

            # short dropout: keep the last command instead of stopping the turn
            if self._last_cmd is not None and lost_for < cfg.hold_s:
                self.state = "holding"
                lr, fb, ud, yaw = self._last_cmd
                return lr, fb, self._alt_band(ud, height_cm), yaw

            self._last_cmd = None
            self._prev_err = None
            # no person: hover in place, never spin looking for one
            self.state = "frozen"
            return 0, 0, 0, 0

        self.state = "tracking"
        self._last_seen = m.t

        ex = m.cx - 0.5  # positive: person right of center
        ey = 0.5 - m.cy  # positive: person above center
        es = cfg.target_size - m.size  # positive: person too far away

        # rates from the raw errors, before the deadzones zero them out
        dex, dey, des = self._rates(m.t, (ex, ey, es))

        # x errors inside the deadzone are zero, then split by lateral mode.
        # the x rate is split with the same weight as the x error.
        if abs(ex) < cfg.dz_x:
            ex = 0.0
        if cfg.lateral_mode == "yaw":
            w = 1.0
        elif cfg.lateral_mode == "strafe":
            w = 0.0
        else:
            w = _clamp(abs(ex) / (2 * cfg.mix_split), 0.0, 1.0)

        left_right = self.strafe_axis.update((1 - w) * ex, (1 - w) * dex)
        yaw = self.yaw_axis.update(w * ex, w * dex)
        forward_backward = self.fb_axis.update(es, des)
        up_down = self._alt_band(self.ud_axis.update(ey, dey), height_cm)

        self._last_cmd = (left_right, forward_backward, up_down, yaw)
        return self._last_cmd


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
        elif new == "idle":
            logger.info("on the ground, tracking idle")


class FrameLog:
    COLUMNS = ["t", "frame_ms", "infer_ms", "reason", "state", "cx", "size", "ex",
               "lr", "fb", "ud", "yaw", "height_cm"]

    def __init__(self):
        LOG_DIR.mkdir(exist_ok=True)
        self.path = LOG_DIR / time.strftime("track_%Y%m%d_%H%M%S.csv")
        self._file = open(self.path, "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(self.COLUMNS)
        self._t0 = time.monotonic()
        self._last_t = None
        self._window_start = self._t0
        self._window = []  # rows since the last summary
        logger.info("frame log: %s", self.path)

    def row(self, now, infer_ms, reason, state, m, cmd, height):
        frame_ms = (now - self._last_t) * 1000 if self._last_t is not None else 0.0
        self._last_t = now
        lr, fb, ud, yaw = cmd if cmd is not None else (0, 0, 0, 0)
        ex = m.cx - 0.5 if m is not None else None
        self._csv.writerow([
            f"{now - self._t0:.3f}", f"{frame_ms:.1f}", f"{infer_ms:.1f}", reason, state,
            f"{m.cx:.3f}" if m else "", f"{m.size:.3f}" if m else "",
            f"{ex:.3f}" if ex is not None else "", lr, fb, ud, yaw, height,
        ])
        self._window.append((infer_ms, reason, ex, yaw))
        if now - self._window_start >= 1.0:
            self._summary(now)

    def _summary(self, now):
        rows = self._window
        fps = len(rows) / (now - self._window_start)
        infer = sum(r[0] for r in rows) / len(rows)
        counts = {}
        for r in rows:
            counts[r[1]] = counts.get(r[1], 0) + 1
        exs = [abs(r[2]) for r in rows if r[2] is not None]
        mean_ex = f"{sum(exs) / len(exs):.2f}" if exs else "-"
        mean_yaw = sum(abs(r[3]) for r in rows) / len(rows)
        reasons = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        logger.info("%.1f fps  infer %.0f ms  |ex| %s  |yaw| %.0f  %s",
                    fps, infer, mean_ex, mean_yaw, reasons)
        self._window = []
        self._window_start = now

    def close(self):
        self._file.close()


def track(tello, frame_read, args, worker: LiftWorker | None, until: float | None = None):
    # until: time.monotonic() deadline to stop tracking at, None runs until ctrl+c
    cfg = TrackerConfig(lateral_mode=args.lateral)
    yolo = YOLOTestCoco(device="mps")
    yolo.setup_runtime()
    tracker = PersonTracker(cfg)
    events = EventLog()
    frames = FrameLog()
    try:
        _track_frames(tello, frame_read, args, worker, until, cfg, yolo, tracker, events, frames)
    finally:
        frames.close()


def _track_frames(tello, frame_read, args, worker, until, cfg, yolo, tracker, events, frames):
    last_frame = None

    while until is None or time.monotonic() < until:
        frame = frame_read.frame
        if frame is None or frame is last_frame:
            time.sleep(0.005)
            continue
        last_frame = frame
        # rgb to bgr since opencv and yolo expect bgr.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        t_frame = time.monotonic()  # when we got the frame, before inference
        det = yolo._detect_best_person(bgr)
        infer_ms = (time.monotonic() - t_frame) * 1000
        m = None
        if det is None:
            reason = "no_person"
        else:
            # raw measurement straight to the controller, no smoothing
            m = measure(det.keypoints, det.scores, bgr.shape, cfg, t=t_frame)
            if m is None:
                reason = "no_torso"  # person found, but shoulders/hips too unsure
            else:
                reason = "ok"
                if worker is not None:
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
            cmd = None
            state = "idle"

        frames.row(now, infer_ms, reason, state, m, cmd, height)
        # a hold is a blip inside tracking, not a state change worth logging
        events.update("tracking" if state == "holding" else state, m, cmd)


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
    try:
        wait_for_real_frame(frame_read)
        if worker is not None:
            worker.start()
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
        tello.streamoff()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # djitellopy logs every rc command at info; keep only its warnings and errors
    Tello.LOGGER.setLevel(logging.WARNING)
    main()
