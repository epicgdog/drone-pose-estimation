import argparse
import csv
import logging
import math
import threading
import time
from pathlib import Path

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

RC_MAX = 100  # tello rc range is -100..100
LOG_DIR = Path("logs")  # per-frame tracking logs


class TrackerConfig(BaseModel):
    # measurement
    kpt_conf: float = 0.3  # joints below this score are ignored
    target_size: float = 0.25  # wanted torso size, fraction of frame height

    # deadzones, errors inside these are treated as zero
    dz_x: float = 0.05  # yaw: no centering push inside this, see _ff_yaw
    dz_y: float = 0.08
    dz_size: float = 0.03

    # pd gains for forward/back and up/down, command units per unit error
    gain_ud: float = 150
    gain_fb: float = 300

    # d gains, command units per (unit error per second). about 0.1 s of
    # lookahead: d_gain / gain. no filter, so raw keypoint noise shows up here.
    d_gain_ud: float = 15
    d_gain_fb: float = 30

    # caps on forward/back and up/down, yaw only limited by the rc range
    cap_fb: float = 40
    cap_ud: float = 40

    # yaw feedforward. constants measured with calibrate.py on 2026-11-21.
    video_latency_s: float = 0.085  # video frame behind the heading state
    hfov_deg: float = 53.0  # camera horizontal field of view
    rc_deadband: float = 12  # rc yaw below this barely turns the drone
    rc_per_dps: float = 1.0  # rc yaw per deg/s above the deadband
    ff_window_s: float = 0.2  # person rate is the bearing slope over this window
    ff_gain: float = 0.8  # under 1 so the drone overshoots less when the person stops
    ff_kp: float = 4.0  # deg/s per deg the person is past the deadzone edge
    ff_min_dps: float = 3.0  # wanted turn rates below this are zero (rate noise)

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


class HeadingBuffer:
    # recent drone headings, unwrapped, so we can ask what the heading was when an
    # older video frame was captured. get_yaw() is whole degrees and only changes when
    # a state packet arrives, so only the moments it changed are kept.
    KEEP_S = 2.0

    def __init__(self):
        self.samples = []  # (t, heading deg)
        self._offset = 0.0
        self._last_raw = None

    def add(self, t: float, raw_deg: float):
        if self._last_raw is not None:
            d = raw_deg - self._last_raw
            if d > 180:
                self._offset -= 360
            elif d < -180:
                self._offset += 360
        if raw_deg != self._last_raw or not self.samples:
            self.samples.append((t, raw_deg + self._offset))
        self._last_raw = raw_deg
        while len(self.samples) > 2 and self.samples[1][0] < t - self.KEEP_S:
            self.samples.pop(0)

    def at(self, t: float) -> float | None:
        if not self.samples:
            return None
        ts = [s[0] for s in self.samples]
        hs = [s[1] for s in self.samples]
        return float(np.interp(t, ts, hs))


class PersonRate:
    # person's bearing in the world (drone heading + angle in the image) and how fast
    # it changes, deg/s. works whether the drone or the person is the one moving.
    def __init__(self, window_s: float):
        self.window_s = window_s
        self.points = []  # (t, bearing deg)

    def reset(self):
        self.points = []

    def update(self, t: float, bearing: float) -> float:
        self.points.append((t, bearing))
        self.points = [p for p in self.points if p[0] >= t - self.window_s]
        if len(self.points) < 4 or self.points[-1][0] - self.points[0][0] < 0.5 * self.window_s:
            return 0.0  # not enough history yet
        ts = np.array([p[0] for p in self.points])
        bs = np.array([p[1] for p in self.points])
        return float(np.polyfit(ts - ts[0], bs, 1)[0])


class PersonTracker:
    # forward/back and up/down: pd on the size and height errors. yaw: feedforward.
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.ud_axis = Axis(cfg.gain_ud, cfg.dz_y, cfg.d_gain_ud, cfg.cap_ud)
        self.fb_axis = Axis(cfg.gain_fb, cfg.dz_size, cfg.d_gain_fb, cfg.cap_fb)
        self.heading = HeadingBuffer()
        self.person_rate = PersonRate(cfg.ff_window_s)
        self.debug = {}  # feedforward internals for the frame log
        self.state = "idle"
        self._last_seen = None  # time of last fresh measurement
        self._last_cmd = None  # last tracking command, repeated on short dropouts
        self._prev_err = None  # (t, ey, es) of the last measurement, for the d terms

    def reset(self):
        self._last_seen = None
        self._last_cmd = None
        self._prev_err = None
        self.person_rate.reset()
        self.state = "idle"

    def _alt_band(self, up_down: int, height_cm: float) -> int:
        # keep the drone in the altitude band
        if height_cm <= self.cfg.min_alt_cm:
            up_down = max(up_down, 0)
        if height_cm >= self.cfg.max_alt_cm:
            up_down = min(up_down, 0)
        return up_down

    def _rates(self, t: float, ey: float, es: float) -> tuple[float, float]:
        # per second. 0 after a gap longer than stale_s, so no slopes across a dropout.
        dey = des = 0.0
        if self._prev_err is not None:
            prev_t, prev_ey, prev_es = self._prev_err
            dt = t - prev_t
            if 0 < dt <= self.cfg.stale_s:
                dey, des = (ey - prev_ey) / dt, (es - prev_es) / dt
        self._prev_err = (t, ey, es)
        return dey, des

    def update(self, m: Measurement | None, now: float, height_cm: float,
               heading_deg: float | None = None):
        cfg = self.cfg
        self.debug = {}
        if heading_deg is not None:
            self.heading.add(now, heading_deg)
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
            self.person_rate.reset()
            # no person: hover in place, never spin looking for one
            self.state = "frozen"
            return 0, 0, 0, 0

        self.state = "tracking"
        self._last_seen = m.t

        ey = 0.5 - m.cy  # positive: person above center
        es = cfg.target_size - m.size  # positive: person too far away
        # rates from the raw errors, before the deadzones zero them out
        dey, des = self._rates(m.t, ey, es)

        yaw = self._ff_yaw(m)
        forward_backward = self.fb_axis.update(es, des)
        up_down = self._alt_band(self.ud_axis.update(ey, dey), height_cm)

        self._last_cmd = (0, forward_backward, up_down, yaw)
        return self._last_cmd

    def _ff_yaw(self, m: Measurement) -> int:
        # wanted turn rate = person's own rate (feedforward) + a push back toward the
        # center when they're past the deadzone edge. then deg/s -> rc with the deadband.
        cfg = self.cfg
        half = math.tan(math.radians(cfg.hfov_deg / 2))
        angle = math.degrees(math.atan((m.cx - 0.5) * 2 * half))  # + is right of center

        head = self.heading.at(m.t - cfg.video_latency_s)  # heading when frame was shot
        rate = 0.0
        if head is not None:
            bearing = head + angle
            rate = self.person_rate.update(m.t, bearing)
            self.debug.update(heading=head, bearing=bearing)

        dz_deg = cfg.dz_x * cfg.hfov_deg
        past = math.copysign(max(0.0, abs(angle) - dz_deg), angle)  # 0 inside, ramps outside
        want = cfg.ff_gain * rate + cfg.ff_kp * past

        if abs(want) < cfg.ff_min_dps:
            yaw = 0
        else:
            rc = cfg.rc_deadband + abs(want) * cfg.rc_per_dps
            yaw = int(round(math.copysign(min(rc, RC_MAX), want)))
        self.debug.update(person_rate=rate, want_dps=want)
        return yaw


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
    def __init__(self, inference: VideoFrameInference, name: str = "lift"):
        self.inference = inference
        self.name = name  # backend name, e.g. flo / fmpose3d
        self.input = LatestSlot()  # (kpts, scores, frame_shape)
        self.output = LatestSlot()  # (newest 3d pose (17, 3), time.monotonic() it was made)
        self.count = 0  # successful lifts, for fps logging
        self.lift_ms = 0.0  # duration of the last lift
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"lift-{name}", daemon=True)

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
            t = time.monotonic()
            try:
                pose = self.inference.lift(kpts, scores, frame_shape)
            except Exception:
                # never let lifting take down tracking
                logger.exception("%s lift failed", self.name)
                continue
            self.lift_ms = (time.monotonic() - t) * 1000
            self.count += 1
            if pose is not None:
                self.output.set((pose, time.monotonic()))



LIFT_STALE_S = 0.5  # 3d poses older than this aren't shown or logged


class LiftStats:
    # per-lifter jitter (m) and forearm lengths (m) for the frame log, to compare lifters
    def __init__(self, worker: LiftWorker):
        self.worker = worker
        self.forearms = worker.inference.skeleton.FOREARMS
        self._prev = None  # previous pose
        self._prev_t = None  # its timestamp

    def update(self, now) -> tuple[str, str, str]:
        out = self.worker.output.get()
        if out is None or out[1] == self._prev_t or now - out[1] >= LIFT_STALE_S:
            return "", "", ""  # nothing new this frame
        pose, t = out
        jitter = ""
        if self._prev is not None and t - self._prev_t < LIFT_STALE_S:
            jitter = f"{np.linalg.norm(pose - self._prev, axis=1).mean():.4f}"
        self._prev, self._prev_t = pose, t
        (le, lw), (re, rw) = self.forearms
        return (jitter, f"{np.linalg.norm(pose[le] - pose[lw]):.4f}",
                f"{np.linalg.norm(pose[re] - pose[rw]):.4f}")


def describe(cmd) -> str:
    # rc command -> what the drone is doing, in words
    _, forward_backward, up_down, yaw = cmd
    moves = []
    if yaw:
        moves.append("turn right" if yaw > 0 else "turn left")
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
               "lr", "fb", "ud", "yaw", "height_cm",
               "heading", "bearing", "person_rate", "want_dps"]

    def __init__(self, lifters=()):
        self.lift_columns = [f"{n}_{c}" for n in lifters for c in ("jitter", "l_forearm", "r_forearm")]
        LOG_DIR.mkdir(exist_ok=True)
        self.path = LOG_DIR / time.strftime("track_%Y%m%d_%H%M%S.csv")
        self._file = open(self.path, "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(self.COLUMNS + self.lift_columns)
        self._t0 = time.monotonic()
        self._last_t = None
        self._window_start = self._t0
        self._window = []  # rows since the last summary
        logger.info("frame log: %s", self.path)

    def row(self, now, infer_ms, reason, state, m, cmd, height, debug=None, lift=()):
        frame_ms = (now - self._last_t) * 1000 if self._last_t is not None else 0.0
        self._last_t = now
        lr, fb, ud, yaw = cmd if cmd is not None else (0, 0, 0, 0)
        ex = m.cx - 0.5 if m is not None else None
        self._csv.writerow([
            f"{now - self._t0:.3f}", f"{frame_ms:.1f}", f"{infer_ms:.1f}", reason, state,
            f"{m.cx:.3f}" if m else "", f"{m.size:.3f}" if m else "",
            f"{ex:.3f}" if ex is not None else "", lr, fb, ud, yaw, height,
            *(f"{debug[k]:.2f}" if debug and k in debug else ""
              for k in ("heading", "bearing", "person_rate", "want_dps")),
            *lift,
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


def track(tello, frame_read, args, workers: dict[str, LiftWorker] | None = None,
          until: float | None = None):
    # until: time.monotonic() deadline to stop tracking at, None runs until ctrl+c
    workers = workers or {}
    cfg = TrackerConfig()
    yolo = YOLOTestCoco(device="mps")
    yolo.setup_runtime()
    tracker = PersonTracker(cfg)
    events = EventLog()
    frames = FrameLog(lifters=list(workers))
    try:
        _track_frames(tello, frame_read, args, workers, until, cfg, yolo, tracker, events, frames)
    finally:
        frames.close()


def _track_frames(tello, frame_read, args, workers, until, cfg, yolo, tracker, events, frames):
    # one euro filter on the 2d keypoints for the 3d lift only.
    # control (measure -> tracker) keeps using the raw keypoints, no added lag.
    euro2d, last_t2d = EuroFilter(), None
    stats = [LiftStats(w) for w in workers.values()]
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
        k2d = None  # smoothed keypoints for the lift
        if det is None:
            euro2d, last_t2d = EuroFilter(), None  # person gone, don't smooth across the gap
        else:
            if last_t2d is not None and t_frame > last_t2d:
                euro2d.set_rate(1.0 / (t_frame - last_t2d))
            last_t2d = t_frame
            k2d = euro2d.filter(det.keypoints.astype(np.float64)).astype(np.float32)
        m = None
        if det is None:
            reason = "no_person"
        else:
            m = measure(det.keypoints, det.scores, bgr.shape, cfg, t=t_frame)
            if m is None:
                reason = "no_torso"  # person found, but shoulders/hips too unsure
            else:
                reason = "ok"
                item = (k2d, det.scores, bgr.shape[:2])  # same 2d for every lifter
                for w in workers.values():
                    w.input.set(item)

        height = tello.get_height()
        now = time.monotonic()
        if height > cfg.airborne_cm:
            cmd = tracker.update(m, now, height, heading_deg=tello.get_yaw())
            if not args.dry_run:
                tello.send_rc_control(*cmd)
            state = tracker.state
        else:
            # on the ground: send nothing, start fresh once airborne
            tracker.reset()
            cmd = None
            state = "idle"

        lift = [v for s in stats for v in s.update(now)]
        frames.row(now, infer_ms, reason, state, m, cmd, height, tracker.debug, lift)
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
        "--live-lift",
        action="store_true",
        help="also run the 3d lift live during flight (slows tracking, output unused for now)",
    )
    parser.add_argument(
        "--lifters",
        choices=["flo", "fmpose3d", "both"],
        default="flo",
        help="3d lifter(s) for --live-lift: flo, fmpose3d, or both side by side (default flo)",
    )
    parser.add_argument(
        "--flight-s",
        type=float,
        default=60,
        help="seconds to track after takeoff before landing (default 60, 0 = until ctrl+c)",
    )
    args = parser.parse_args()

    logger.info("Starting up!")
    workers = {}
    if args.live_lift:
        names = ["flo", "fmpose3d"] if args.lifters == "both" else [args.lifters]
        for name in names:
            # flo samples on cpu, fmpose3d on mps (yolo is on the neural engine)
            workers[name] = LiftWorker(VideoFrameInference(device="mps", backend=name), name)

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
        for w in workers.values():
            w.start()
        if not args.dry_run:
            logger.info("taking off, ctrl+c to land")
            took_off = True  # set first so ctrl+c mid takeoff still lands
            tello.takeoff()
        until = None
        if not args.dry_run and args.flight_s > 0:
            until = time.monotonic() + args.flight_s
            logger.info("landing in %.0f s", args.flight_s)
        track(tello, frame_read, args, workers, until)
        if until is not None:
            logger.info("flight time up, landing")
    except KeyboardInterrupt:
        logger.info("interrupted, %s", "landing" if took_off else "stopping")
    finally:
        for w in workers.values():
            w.stop()
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
