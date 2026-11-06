import math

import numpy as np


class LowPassFilter:
    def __init__(self):
        self.prev = None
        self.first_time = True

    def filter(self, kpts, alpha):
        if self.first_time:
            self.first_time = False
            self.prev = kpts
            return kpts
        else:
            curr_pos = (alpha * kpts) + ((1 - alpha) * self.prev)
            self.prev = curr_pos
            return curr_pos


class EuroFilter:
    def _alpha(self, rate, cut_off):
        tau = 1.0 / (2 * math.pi * cut_off)
        elapsed = 1.0 / rate
        return 1.0 / (1.0 + tau / elapsed)

    def __init__(self):
        self.first_time = True
        self.min_cut_off = 1
        self.rate = 30  # how many samples per second
        self.beta = 0.007  # just based on the sim i was playing
        self.main_filter = LowPassFilter()
        self.d_cut_off = 1.0
        self.d_main_filter = LowPassFilter()

    def set_rate(self, new_rate):
        self.rate = max(1e-3, new_rate)

    def filter(self, kpts):
        dx = 0
        if self.first_time:
            self.first_time = False
        else:
            dx = (kpts - self.main_filter.prev) * self.rate

        edx = self.d_main_filter.filter(dx, self._alpha(self.rate, self.d_cut_off))
        cut_off = self.min_cut_off + self.beta * np.abs(edx)
        return self.main_filter.filter(kpts, self._alpha(self.rate, cut_off))
