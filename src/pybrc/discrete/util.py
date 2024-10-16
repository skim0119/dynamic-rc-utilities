from typing import List, Tuple, Optional

import os, sys
import numpy as np
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt

from miv.core.datatype import Spikestamps
from miv.visualization.event import plot_spiketrain_raster

from tqdm import tqdm

from miv.core.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call

@dataclass
class ExcludeSpike(OperatorMixin):
    """
    The module is used to eliminate false-spikes of some duration.
    The deadtime can be found using PSTH plot with varying bin-size.
    """
    deadtime: float # in s

    TTL_state:int

    pre_deadtime: float = 0.01# in s

    tag:str = "exclude spike"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, spikestamps, ttl):
        timestamps = ttl.timestamps
        ttl_on = timestamps[ttl[0] == self.TTL_state]

        intervals = [(p-self.pre_deadtime, p+self.deadtime) for p in ttl_on]
        self.logger.info(f"{intervals=}")

        # TODO: Try to optimize
        delays = []
        return_spikestamps = Spikestamps()
        for spikestamp in spikestamps:
            new_spikestamp = []
            for t in spikestamp:
                is_in_interval, delay = self.in_interval(t, intervals)
                if not is_in_interval:
                    new_spikestamp.append(t)
                else:
                    delays.append(delay)
            return_spikestamps.append(new_spikestamp)
            self.logger.info(f"{new_spikestamp=}")

        plt.figure()
        plt.hist(delays, bins=30)
        plt.xlabel('delay (s)')
        plt.ylabel('count')
        plt.savefig(os.path.join(self.analysis_path, "excluded.png"))
        plt.close('all')
        return return_spikestamps

    def in_interval(self, value:int, intervals:List[Tuple[int,int]]):
        for idx, interval in enumerate(intervals):
            if interval[0] < value and value < interval[1]:
                return True, value - interval[0]
            if value < interval[0]:
                break
        return False, 0

        #for idx, interval in enumerate(intervals[start_interval:]):
        #    if value > interval[1]: break
        #    elif interval[1] > value and value > interval[0]:
        #        return True, idx + start_interval
        #return False, idx + start_interval

    def plot_spiketrain(
        self,
        spikestamps,
        inputs,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Axes:
        """
        Plot spike train in raster
        """
        t0 = spikestamps.get_first_spikestamp()
        tf = spikestamps.get_last_spikestamp()

        # TODO: REFACTOR. Make single plot, and change xlim
        term = 10
        n_terms = int(np.ceil((tf - t0) / term))
        if n_terms == 0:
            # TODO: Warning message
            return None
        for idx in range(n_terms):
            fig, ax = plot_spiketrain_raster(
                spikestamps, idx * term + t0, min((idx + 1) * term + t0, tf)
            )
            if save_path is not None:
                plt.savefig(os.path.join(save_path, f"spiketrain_raster_{idx:03d}.png"))
            if not show:
                plt.close("all")
        if show:
            plt.show()
            plt.close("all")
        return ax


