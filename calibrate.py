"""Measure the constants yaw feedforward needs.

    uv run python3 calibrate.py rate            # spin test: rc yaw -> deg/s  (gives k_ff)
    uv run python3 calibrate.py sweep           # stand still in front, drone swings left/right
                                                # (gives video latency L and camera hfov)
    uv run python3 calibrate.py fit logs/calib_*.csv   # re-run the analysis on saved logs

Both flights take off, run the test, land, save raw samples to logs/ and print the fit.
ctrl+c lands at any time.
"""

import argparse
import csv
import logging
import math
import threading
import time

import cv2
import numpy as np
from djitellopy import Tello

from inference import YOLOTestCoco
from main import wait_for_real_frame
from tracker import LOG_DIR, TrackerConfig, measure

logger = logging.getLogger(__name__)

MIN_BATTERY = 30  # percent, fast spins drain it quickly
RATE_LEVELS = [20, 40, 60, 80, 100]  # rc yaw values for the spin test
RATE_HOLD_S = 3.0  # seconds per level
RATE_SETTLE_S = 0.8  # ignore this much of each level while the turn spins up


# ---------- heading sampling ----------


class HeadingLog:
    # samples get_yaw() in a thread at ~200 hz, keeps (t, unwrapped heading deg).
    # get_yaw() only changes when a state packet arrives, so samples come in steps.
    def __init__(self, tello):
        self.tello = tello
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._offset = 0.0
        self._last_raw = None

    def _run(self):
        while not self._stop.is_set():
            raw = float(self.tello.get_yaw())
            if self._last_raw is not None:
                d = raw - self._last_raw
                if d > 180:
                    self._offset -= 360
                elif d < -180:
                    self._offset += 360
            self._last_raw = raw
            self.samples.append((time.monotonic(), raw + self._offset))
            time.sleep(0.005)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)


def change_points(hs):
    # heading samples arrive as steps (integer degrees, one value per state packet).
    # keep only the moments the value changed, so interpolation runs between real updates.
    keep = np.concatenate([[True], np.diff(hs[:, 1]) != 0])
    return hs[keep]


# ---------- flight helpers ----------


def connect():
    tello = Tello()
    tello.connect()
    battery = tello.get_battery()
    logger.info("battery %d%%", battery)
    if battery < MIN_BATTERY:
        raise SystemExit(f"battery {battery}% is below {MIN_BATTERY}%, charge it first")
    return tello


def fly(tello, test):
    # takes off, runs test(), always stops and lands
    took_off = False
    try:
        logger.info("taking off, ctrl+c to land")
        took_off = True
        tello.takeoff()
        time.sleep(2)  # let it settle into a hover
        test()
    except KeyboardInterrupt:
        logger.info("interrupted, landing")
    finally:
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            logger.exception("failed to send stop")
        if took_off:
            try:
                tello.land()
            except Exception:
                logger.exception("land failed, land it from the phone app or catch it")


def hold_rc(tello, yaw, seconds, marks=None, label=None):
    # sends rc yaw repeatedly (the tello stops if rc goes quiet), records when it started
    start = time.monotonic()
    if marks is not None:
        marks.append((start, label if label is not None else yaw))
    while time.monotonic() - start < seconds:
        tello.send_rc_control(0, 0, 0, int(yaw))
        time.sleep(0.02)


def save_csv(path, header, rows):
    LOG_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    logger.info("saved %s", path)


# ---------- test 1: rc yaw -> deg/s ----------


def run_rate():
    tello = connect()
    heading = HeadingLog(tello)
    marks = []  # (t, rc yaw) at the start of each segment

    def test():
        heading.start()
        hold_rc(tello, 0, 1.0, marks)
        for level in RATE_LEVELS:
            logger.info("yaw %+d for %.0f s", level, RATE_HOLD_S)
            hold_rc(tello, level, RATE_HOLD_S, marks)
            hold_rc(tello, 0, 1.0, marks)
            # spin back the other way so it doesn't drift through a full circle
            logger.info("yaw %+d for %.0f s", -level, RATE_HOLD_S)
            hold_rc(tello, -level, RATE_HOLD_S, marks)
            hold_rc(tello, 0, 1.0, marks)

    try:
        fly(tello, test)
    finally:
        heading.stop()
    path = LOG_DIR / time.strftime("calib_rate_%Y%m%d_%H%M%S.csv")
    rows = [("h", f"{t:.4f}", f"{h:.1f}") for t, h in heading.samples]
    rows += [("m", f"{t:.4f}", yaw) for t, yaw in marks]
    save_csv(path, ["kind", "t", "value"], rows)
    fit_rate(path)


def fit_rate(path):
    hs, marks = [], []
    for r in csv.DictReader(open(path)):
        (hs if r["kind"] == "h" else marks).append((float(r["t"]), float(r["value"])))
    hs = change_points(np.array(hs))
    marks.append((hs[-1, 0], 0.0))  # end of the last segment

    print(f"\nyaw rate test: {path}")
    print(f"{'rc':>5} {'deg/s':>8} {'spin-up s':>10}")
    rcs, rates = [], []
    for (t0, rc), (t1, _) in zip(marks, marks[1:]):
        if rc == 0:
            continue
        seg = hs[(hs[:, 0] >= t0) & (hs[:, 0] < t1)]
        steady = seg[seg[:, 0] >= t0 + RATE_SETTLE_S]
        if len(steady) < 5:
            continue
        rate = np.polyfit(steady[:, 0], steady[:, 1], 1)[0]
        # spin-up: first time the rate over a 0.3 s window reaches 63% of steady
        spin_up = float("nan")
        for tau in np.arange(t0, t0 + RATE_SETTLE_S + 0.5, 0.02):
            win = seg[(seg[:, 0] >= tau) & (seg[:, 0] < tau + 0.3)]
            if len(win) >= 3 and abs(np.polyfit(win[:, 0], win[:, 1], 1)[0]) >= 0.63 * abs(rate):
                spin_up = tau + 0.15 - t0
                break
        print(f"{int(rc):>+5} {rate:>8.1f} {spin_up:>10.2f}")
        rcs.append(rc)
        rates.append(rate)

    rcs, rates = np.array(rcs), np.array(rates)
    if len(rcs) == 0:
        print("no usable segments")
        return
    # rc = k_ff * deg/s, least squares through the origin
    k_ff = float(np.sum(rcs * rates) / np.sum(rates * rates))
    print(f"\nk_ff = {k_ff:.3f} rc per deg/s  (1 rc ≈ {1 / k_ff:.2f} deg/s)")
    print("check the table for a dead band at low rc or flattening near 100")


# ---------- test 2: video latency and hfov ----------


def run_sweep(amp_rc, freq_hz, seconds):
    tello = connect()
    tello.streamon()
    frame_read = tello.get_frame_read()
    wait_for_real_frame(frame_read)
    yolo = YOLOTestCoco(device="mps")
    yolo.setup_runtime()
    cfg = TrackerConfig()
    heading = HeadingLog(tello)
    frames = []  # (t_frame, cx)

    def test():
        heading.start()
        logger.info("stand still in front of the drone. sweeping %.0f s", seconds)
        start = time.monotonic()
        last_frame = None
        while time.monotonic() - start < seconds:
            t = time.monotonic() - start
            # ramp the amplitude in over the first 2 s so it doesn't jerk
            amp = amp_rc * min(1.0, t / 2.0)
            tello.send_rc_control(0, 0, 0, int(round(amp * math.sin(2 * math.pi * freq_hz * t))))
            frame = frame_read.frame
            if frame is None or frame is last_frame:
                time.sleep(0.003)
                continue
            last_frame = frame
            t_frame = time.monotonic()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            det = yolo._detect_best_person(bgr)
            if det is None:
                continue
            m = measure(det.keypoints, det.scores, bgr.shape, cfg)
            if m is not None:
                frames.append((t_frame, m.cx))

    try:
        fly(tello, test)
    finally:
        heading.stop()
        tello.streamoff()
    path = LOG_DIR / time.strftime("calib_sweep_%Y%m%d_%H%M%S.csv")
    rows = [("h", f"{t:.4f}", f"{h:.1f}") for t, h in heading.samples]
    rows += [("f", f"{t:.4f}", f"{cx:.4f}") for t, cx in frames]
    save_csv(path, ["kind", "t", "value"], rows)
    fit_sweep(path)


def fit_sweep(path):
    hs, fs = [], []
    for r in csv.DictReader(open(path)):
        (hs if r["kind"] == "h" else fs).append((float(r["t"]), float(r["value"])))
    hs, fs = change_points(np.array(hs)), np.array(fs)
    print(f"\nsweep test: {path}  ({len(fs)} frames with a person)")
    if len(fs) < 30:
        print("too few frames with a person, stand in view and try again")
        return
    h_smooth = hs[:, 1]

    # person stood still, so heading(t - L) + angle(cx) should be constant.
    # grid search L and hfov for the flattest bearing.
    best = None
    for lat in np.arange(0.0, 0.601, 0.005):
        head = np.interp(fs[:, 0] - lat, hs[:, 0], h_smooth)
        for hfov in np.arange(40.0, 100.01, 0.5):
            angle = np.degrees(np.arctan((fs[:, 1] - 0.5) * 2 * math.tan(math.radians(hfov / 2))))
            spread = np.std(head + angle)
            if best is None or spread < best[0]:
                best = (spread, lat, hfov)
    spread, lat, hfov = best
    head0 = np.interp(fs[:, 0], hs[:, 0], h_smooth)
    angle0 = np.degrees(np.arctan((fs[:, 1] - 0.5) * 2 * math.tan(math.radians(hfov / 2))))
    print(f"heading swing: {np.ptp(hs[:, 1]):.0f} deg, person cx range {fs[:, 1].min():.2f}-{fs[:, 1].max():.2f}")
    print(f"bearing spread without latency fix: {np.std(head0 + angle0):.2f} deg")
    print(f"bearing spread with best fit:       {spread:.2f} deg")
    print(f"\nL    = {lat * 1000:.0f} ms  (video behind heading)")
    print(f"hfov = {hfov:.1f} deg")
    if lat < 0.01 or lat > 0.59 or hfov < 41 or hfov > 99:
        print("warning: fit hit the edge of the search range, don't trust it")


# ---------- cli ----------


def main():
    parser = argparse.ArgumentParser(description="measure yaw feedforward constants")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rate", help="spin test: rc yaw -> deg/s")
    sweep = sub.add_parser("sweep", help="latency + hfov, stand still in front")
    sweep.add_argument("--amp", type=float, default=30, help="rc yaw amplitude (default 30)")
    sweep.add_argument("--freq", type=float, default=0.4, help="sweep frequency hz (default 0.4)")
    sweep.add_argument("--seconds", type=float, default=15, help="sweep length (default 15)")
    fit = sub.add_parser("fit", help="re-run the analysis on saved logs")
    fit.add_argument("paths", nargs="+")
    args = parser.parse_args()

    if args.cmd == "rate":
        run_rate()
    elif args.cmd == "sweep":
        run_sweep(args.amp, args.freq, args.seconds)
    else:
        for p in args.paths:
            fit_sweep(p) if "sweep" in p else fit_rate(p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Tello.LOGGER.setLevel(logging.WARNING)
