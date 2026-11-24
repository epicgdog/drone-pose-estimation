"""Follow a person with a Tello drone.

    uv run python3 main.py                       # take off and track until ctrl+c
    uv run python3 main.py --flight-s 60         # land after 60 s
    uv run python3 main.py --video videos/x.mp4  # recorded video instead of the tello, never flies
    uv run python3 main.py --lifters both        # start with flo and fmpose3d side by side

The dashboard always runs at http://127.0.0.1:8000: camera, 2d + 3d pose, commands, and
controls to switch lifters and change gains. ctrl+c in the terminal lands.

Each frame: yolo 2d -> measure the person -> tracker picks an rc command -> send it.
Forward/back and up/down are pd controlled, yaw is feedforward. The pieces live in tracker.py.
With --video there's no real heading, so yaw stays 0 and only forward/back and up/down run.
"""

import argparse
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello

from euro_filter import EuroFilter
from inference import VideoFrameInference, YOLOTestCoco
from live_view import Controls, LiveServer
from tracker import (
    LIFT_STALE_S,
    TUNABLES,
    EventLog,
    FrameLog,
    LiftStats,
    LiftWorker,
    PersonTracker,
    TrackerConfig,
    measure,
    wait_for_real_frame,
)

logger = logging.getLogger(__name__)

LIFTERS = ["flo", "fmpose3d"]  # every 3d backend, all loaded at startup


def parse_args():
    parser = argparse.ArgumentParser(description="follow a person with a tello drone")
    parser.add_argument(
        "--video",
        type=Path,
        help="play this recorded video instead of the tello camera. never flies, "
             "commands are computed and logged only, yaw stays 0 (no real heading)",
    )
    parser.add_argument(
        "--lifters",
        choices=["flo", "fmpose3d", "both"],
        default="flo",
        help="3d lifter(s) running at start, switchable on the dashboard (default flo)",
    )
    parser.add_argument(
        "--flight-s",
        type=float,
        default=0,
        help="seconds to track before landing (default 0 = fly until ctrl+c)",
    )
    return parser.parse_args()


class VideoFrameRead:
    # plays a video at its own fps, same interface as djitellopy's BackgroundFrameRead
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"could not open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame = None
        self.stopped = False
        self._thread = threading.Thread(target=self._run, name="video", daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        next_t = time.monotonic()
        while not self.stopped:
            ok, bgr = self.cap.read()
            if not ok:
                logger.info("video ended")
                break
            self.frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            next_t += 1.0 / self.fps
            time.sleep(max(0.0, next_t - time.monotonic()))
        self.stopped = True
        self.cap.release()

    def stop(self):
        self.stopped = True


class VideoDrone:
    # fake Tello backed by a video: never moves, fixed height and heading, rc goes nowhere
    HEIGHT_CM = 120  # inside the altitude band

    def __init__(self, path: Path):
        self.path = path
        self.reader = VideoFrameRead(path)
        self._height = 0

    def connect(self):
        logger.info("video source %s at %.0f fps", self.path, self.reader.fps)

    def get_battery(self):
        return 100

    def streamon(self):
        self.reader.start()

    def get_frame_read(self):
        return self.reader

    def takeoff(self):
        self._height = self.HEIGHT_CM

    def land(self):
        self._height = 0

    def get_height(self):
        return self._height

    def get_yaw(self):
        return 0

    def send_rc_control(self, *cmd):
        pass

    def streamoff(self):
        self.reader.stop()


def make_workers() -> dict[str, LiftWorker]:
    # flo samples on cpu, fmpose3d on mps (yolo is on the neural engine)
    return {name: LiftWorker(VideoFrameInference(device="mps", backend=name), name)
            for name in LIFTERS}


def track(drone, frame_read, args, workers: dict[str, LiftWorker], until: float | None):
    cfg = TrackerConfig()
    yolo = YOLOTestCoco(device="mps")
    yolo.setup_runtime()
    tracker = PersonTracker(cfg, use_ff=args.video is None)
    events = EventLog()
    frames = FrameLog(lifters=list(workers))
    stats = [LiftStats(w) for w in workers.values()]
    start = LIFTERS if args.lifters == "both" else [args.lifters]
    controls = Controls(list(workers), start, TUNABLES, cfg.model_dump())
    # one euro filter on the 2d keypoints for the dashboard and the 3d lift only.
    # control (measure -> tracker) keeps using the raw keypoints, no added lag.
    euro2d, last_t2d = EuroFilter(), None
    last_frame = None
    live = None
    try:
        live = LiveServer(controls, {n: w.inference.skeleton for n, w in workers.items()},
                          open_browser=True)

        while until is None or time.monotonic() < until:
            if frame_read.stopped:
                break  # video ended, or the tello stream stopped
            frame = frame_read.frame
            if frame is None or frame is last_frame:
                time.sleep(0.005)
                continue
            last_frame = frame
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 0. gain changes from the dashboard
            note = ""
            pending = controls.take_pending()
            if pending:
                cfg = TrackerConfig(**{**cfg.model_dump(), **pending})
                tracker.set_config(cfg)
                note = " ".join(f"{k}={v:g}" for k, v in pending.items())
                logger.info("gains changed: %s", note)
            active = controls.active()

            # 1. 2d pose
            t_frame = time.monotonic()  # when we got the frame, before inference
            det = yolo._detect_best_person(bgr)
            infer_ms = (time.monotonic() - t_frame) * 1000

            # 2. smoothed 2d for display + lift
            k2d = None
            if det is None:
                euro2d, last_t2d = EuroFilter(), None  # person gone, don't smooth across the gap
            else:
                if last_t2d is not None and t_frame > last_t2d:
                    euro2d.set_rate(1.0 / (t_frame - last_t2d))
                last_t2d = t_frame
                k2d = euro2d.filter(det.keypoints.astype(np.float64)).astype(np.float32)

            # 3. measure the person from the raw 2d, hand the smoothed 2d to the active lifters
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
                    for name in active:
                        workers[name].input.set(item)

            # 4. pick and send the rc command
            height = drone.get_height()
            now = time.monotonic()
            if height > cfg.airborne_cm:
                cmd = tracker.update(m, now, height, heading_deg=drone.get_yaw())
                drone.send_rc_control(*cmd)
                state = tracker.state
            else:
                # on the ground: send nothing, start fresh once airborne
                tracker.reset()
                cmd = None
                state = "idle"

            # 5. logs and dashboard
            lift = [v for s in stats for v in s.update(now)]
            frames.row(now, infer_ms, reason, state, m, cmd, height, tracker.debug, lift, note)
            publish_live(live, workers, active, now, bgr, k2d, det, infer_ms, state, cmd,
                         tracker, args)
            # a hold is a blip inside tracking, not a state change worth logging
            events.update("tracking" if state == "holding" else state, m, cmd)
    finally:
        frames.close()
        if live is not None:
            live.close()


def publish_live(live, workers, active, now, bgr, k2d, det, infer_ms, state, cmd, tracker, args):
    poses3d = {}
    for name in active:
        out = workers[name].output.get()
        fresh = out is not None and now - out[1] < LIFT_STALE_S  # drop stale 3d poses
        poses3d[name] = out[0] if fresh else None
    live.publish(
        bgr,
        kpts=k2d,
        scores=det.scores if det is not None else None,
        poses3d=poses3d,
        active=active,
        lift_ms={n: round(workers[n].lift_ms, 1) for n in active},
        detect_ms=round(infer_ms, 1),
        state=state,
        cmd=list(cmd) if cmd is not None else None,
        person_rate=round(tracker.debug.get("person_rate", 0.0), 1),
        source="video, no yaw" if args.video else "tello",
    )


def main():
    args = parse_args()
    logger.info("Starting up!")
    workers = make_workers()

    drone = VideoDrone(args.video) if args.video else Tello()
    drone.connect()
    logger.info("Battery: %s", drone.get_battery())

    drone.streamon()
    frame_read = drone.get_frame_read()

    # the laptop is the only controller: it takes off and lands. keep the phone app
    # disconnected, its stick packets fight the rc commands sent here.
    took_off = False
    try:
        wait_for_real_frame(frame_read)
        for w in workers.values():
            w.start()
        logger.info("taking off, ctrl+c to land")
        took_off = True  # set first so ctrl+c mid takeoff still lands
        drone.takeoff()
        until = None
        if args.flight_s > 0:
            until = time.monotonic() + args.flight_s
            logger.info("landing in %.0f s", args.flight_s)
        track(drone, frame_read, args, workers, until)
        if until is not None and time.monotonic() >= until:
            logger.info("flight time up, landing")
    except KeyboardInterrupt:
        logger.info("interrupted, %s", "landing" if took_off else "stopping")
    finally:
        for w in workers.values():
            w.stop()
        try:
            drone.send_rc_control(0, 0, 0, 0)
        except Exception:
            logger.exception("failed to send stop command")
        if took_off:
            try:
                drone.land()
            except Exception:
                logger.exception("land failed, land it from the phone app or catch it")
        drone.streamoff()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # djitellopy logs every rc command at info; keep only its warnings and errors
    Tello.LOGGER.setLevel(logging.WARNING)
    main()
