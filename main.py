"""Follow a person with a Tello drone.

    uv run python3 main.py                       # take off and track until ctrl+c
    uv run python3 main.py --flight-s 60         # land after 60 s
    uv run python3 main.py --dry-run             # never fly, just watch the dashboard
    uv run python3 main.py --lifters fmpose3d    # 3d lift with fmpose3d instead of flo

The dashboard (camera, 2d + 3d pose, commands) always runs at http://127.0.0.1:8000.

Each frame: yolo 2d -> measure the person -> tracker picks an rc command -> send it.
Forward/back and up/down are pd controlled, yaw is feedforward. The pieces live in tracker.py.
"""

import argparse
import logging
import time

import cv2
import numpy as np
from djitellopy import Tello

from euro_filter import EuroFilter
from inference import VideoFrameInference, YOLOTestCoco
from live_view import LiveServer
from tracker import (
    LIFT_STALE_S,
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


def parse_args():
    parser = argparse.ArgumentParser(description="follow a person with a tello drone")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and log commands, never send them to the drone",
    )
    parser.add_argument(
        "--lifters",
        choices=["flo", "fmpose3d", "both"],
        default="flo",
        help="3d lifter(s): flo, fmpose3d, or both side by side (default flo)",
    )
    parser.add_argument(
        "--flight-s",
        type=float,
        default=0,
        help="seconds to track before landing (default 0 = fly until ctrl+c)",
    )
    return parser.parse_args()


def make_workers(args) -> dict[str, LiftWorker]:
    names = ["flo", "fmpose3d"] if args.lifters == "both" else [args.lifters]
    # flo samples on cpu, fmpose3d on mps (yolo is on the neural engine)
    return {name: LiftWorker(VideoFrameInference(device="mps", backend=name), name)
            for name in names}


def track(tello, frame_read, args, workers: dict[str, LiftWorker], until: float | None):
    cfg = TrackerConfig()
    yolo = YOLOTestCoco(device="mps")
    yolo.setup_runtime()
    tracker = PersonTracker(cfg)
    events = EventLog()
    frames = FrameLog(lifters=list(workers))
    stats = [LiftStats(w) for w in workers.values()]
    # one euro filter on the 2d keypoints for the dashboard and the 3d lift only.
    # control (measure -> tracker) keeps using the raw keypoints, no added lag.
    euro2d, last_t2d = EuroFilter(), None
    last_frame = None
    live = None
    try:
        live = LiveServer({n: w.inference.skeleton for n, w in workers.items()},
                          open_browser=True)

        while until is None or time.monotonic() < until:
            frame = frame_read.frame
            if frame is None or frame is last_frame:
                time.sleep(0.005)
                continue
            last_frame = frame
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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

            # 3. measure the person from the raw 2d, hand the smoothed 2d to the lifters
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

            # 4. pick and send the rc command
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

            # 5. logs and dashboard
            lift = [v for s in stats for v in s.update(now)]
            frames.row(now, infer_ms, reason, state, m, cmd, height, tracker.debug, lift)
            publish_live(live, workers, now, bgr, k2d, det, infer_ms, state, cmd, tracker)
            # a hold is a blip inside tracking, not a state change worth logging
            events.update("tracking" if state == "holding" else state, m, cmd)
    finally:
        frames.close()
        if live is not None:
            live.close()


def publish_live(live, workers, now, bgr, k2d, det, infer_ms, state, cmd, tracker):
    poses3d = {}
    for name, w in workers.items():
        out = w.output.get()
        fresh = out is not None and now - out[1] < LIFT_STALE_S  # drop stale 3d poses
        poses3d[name] = out[0] if fresh else None
    live.publish(
        bgr,
        kpts=k2d,
        scores=det.scores if det is not None else None,
        poses3d=poses3d,
        lift_ms={n: round(w.lift_ms, 1) for n, w in workers.items()},
        detect_ms=round(infer_ms, 1),
        state=state,
        cmd=list(cmd) if cmd is not None else None,
        person_rate=round(tracker.debug.get("person_rate", 0.0), 1),
    )


def main():
    args = parse_args()
    logger.info("Starting up!")
    workers = make_workers(args)

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
