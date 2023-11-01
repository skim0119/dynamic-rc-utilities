import os, sys
import multiprocessing as mp
import numpy as np
from dataclasses import dataclass

from functools import partial

import matplotlib.pyplot as plt

from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff
from miv.core.datatype import Spikestamps
from miv.core.pipeline import Pipeline
from miv.statistics import decay_spike_counts, spike_counts_with_kernel

import pickle as pkl

from tqdm import tqdm

from pybrc.utils import get_nearest

from miv.core.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call


@dataclass
class ParseInput(OperatorMixin):
    binsize: float

    TTL_state: int = 1
    binsize_threshold: float = 0.001  # sec
    tag: str = "input parsing"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, ttl_signal):
        states = ttl_signal[0]
        timestamps = ttl_signal.timestamps

        on = timestamps[states == self.TTL_state]
        off = timestamps[states == -self.TTL_state]

        self.logger.info(f"{len(on)=} should be same as {len(off)=}")
        self.logger.info(f"TTL states in this recording: {np.unique(states, return_counts=True)=}")

        off_probe = 0
        time = []
        data = []
        front_bar = on[0]
        while front_bar < off[-1]:
            next_bar = front_bar + self.binsize

            # Correct "next_bar" for little offset
            _nearest_next_on, delta, _ = get_nearest(on, next_bar)
            if delta < self.binsize_threshold:
                next_bar = _nearest_next_on
                
            count = 0
            while off_probe < len(off) and off[off_probe] < next_bar:
                off_probe += 1
                count += 1
            data.append(count)
            time.append(next_bar)
            front_bar = next_bar

            #if count == 1:
            #    plt.eventplot(on-front_bar, color='black', lineoffsets=pp)
            #    plt.eventplot(off-front_bar, color='red', lineoffsets=pp)
            #    pp += 1

        return np.array(data), np.array(time)
    # TODO: plot distribution of data
    # TODO: some other debugging plots


@dataclass
class ParseTemporalDecoding(OperatorMixin):
    tag: str = "temporal decoding"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, spikestamps, input_parse):
        _, probe_times = input_parse

        _N = 3
        _Extra = 1
        Xs = np.zeros((probe_times.shape[0], len(spikestamps) * _N + _Extra))

        # Time
        Xs[:, 0] = np.linspace(0, 1, probe_times.shape[0])
        for idx, spiketrain in enumerate(spikestamps):
            idx_N = idx * _N + _Extra

            # Decay spike count
            #Xs[:, idx_N + 0] = decay_spike_counts(
            #    np.asarray(spiketrain), probe_times, decay_rate=1
            #)

            # Firing Rante
            Xs[:, idx_N + 0] = spike_counts_with_kernel(
                np.asarray(spiketrain), probe_times, lambda x: np.logical_and(x>0, x<1).astype(np.float_)
            )
            Xs[:, idx_N + 1] = spike_counts_with_kernel(
                np.asarray(spiketrain), probe_times, lambda x: np.logical_and(x>0, x<0.5).astype(np.float_)
            )
            Xs[:, idx_N + 2] = spike_counts_with_kernel(
                np.asarray(spiketrain), probe_times, lambda x: np.logical_and(x>0, x<0.25).astype(np.float_)
            )

            # Vector of tanhs
            # Xs[:, idx_N + 0] = spike_counts_with_kernel(
            #     np.asarray(spiketrain), probe_times, lambda x: 1.0 - np.tanh(x)
            # )
            # Xs[:, idx_N + 1] = spike_counts_with_kernel(
            #     np.asarray(spiketrain), probe_times, lambda x: np.tanh(x)
            # )
            # Xs[:, idx_N + 2] = spike_counts_with_kernel(
            #     np.asarray(spiketrain), probe_times, lambda x: np.tanh(-x)
            # )
            # Xs[:, idx_N + 3] = spike_counts_with_kernel(
            #     np.asarray(spiketrain), probe_times, lambda x: -1.0 - np.tanh(-x)
            # )
        self.logger.info(f"    {Xs.mean()=}, {Xs.std()=}")
        self.logger.info(f"    {Xs.shape=}")

        return Xs


