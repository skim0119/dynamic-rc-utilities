import os

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

from miv.core.datatype import Spikestamps
from miv.statistics import decay_spike_counts, spike_counts_with_kernel
from miv.core.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call


@dataclass
class BaseParseInput(OperatorMixin):
    binsize: float = 0.001  # sec

    TTL_state: int = 1
    tag: str = "input parsing"

    def __post_init__(self):
        super().__init__()

    def plot_ttl(self, outputs, inputs, show=False, save_path=None):
        y, time = outputs
        ttl_signal = inputs

        states = ttl_signal[0]
        timestamps = ttl_signal.timestamps

        interval = 10
        stime = timestamps.min()
        etime = timestamps.max()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(timestamps, states)
        ax2.plot(time, y, "or")

        ax1.set_xlabel("time (sec)")
        ax1.set_ylabel("TTL state")
        ax2.set_ylabel("Pattern")
        idx = 0
        while stime < etime:
            plt.xlim(stime, stime + interval)
            plt.savefig(os.path.join(save_path, f"ttl_state_{idx:04d}.png"))
            stime += interval
            idx += 1
        plt.close("all")

    def plot_data_distribution(self, outputs, inputs, show=False, save_path=None):
        data, time = outputs

        plt.figure()
        plt.hist(data, bins=20)
        plt.title(f"Data distribution (mean={data.mean():.2f}, std={data.std():.2f})")
        plt.xlabel("Pattern Indes")
        plt.ylabel("Count")
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "data_distribution.png"))
        if show:
            plt.show()
        plt.close("all")
