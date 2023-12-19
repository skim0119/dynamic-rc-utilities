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
        os.makedirs(self.analysis_path, exist_ok=True)

        states = ttl_signal[0]
        timestamps = ttl_signal.timestamps

        stime = timestamps.min()
        etime = timestamps.max()

        # DEBUG: plotting ttl -- maybe add in sequence
        interval = 10
        plt.figure()
        plt.plot(timestamps, states)
        plt.xlabel('time (sec)')
        plt.ylabel('TTL state')
        idx = 0
        while stime < etime:
            plt.xlim(stime, stime+interval)
            plt.savefig(os.path.join(self.analysis_path, f"ttl_state_{idx:04d}.png"))
            stime += interval
            idx += 1
        plt.close('all')

        time = []
        data = []
        on = timestamps[states == self.TTL_state]

        self.logger.info(f"{len(on)=}")
        self.logger.info(f"TTL states in this recording: {np.unique(states, return_counts=True)=}")

        if len(on) == 0:
            return np.array(data), np.array(time)

        #off_probe = 0
        deltas = []
        front_bar = next_bar = on[0]
        index = 0
        while next_bar < on[-1]:
            next_bar = front_bar + self.binsize

            # Correct "next_bar" for little offset
            _nearest_next_on, delta, next_index = get_nearest(on, next_bar)
            deltas.append(delta)
            if delta < self.binsize_threshold:
                next_bar = _nearest_next_on

            data.append(next_index - index)
            time.append(next_bar)
            front_bar, index = next_bar, next_index

        # Plot histogram of delta
        plt.figure()
        plt.hist(deltas, bins=50)
        plt.title("(should be close to zero, ideally all smaller than pulse length)")
        plt.savefig(os.path.join(self.analysis_path, 'parsing_precision.png'))
        plt.close('all')

        return np.array(data), np.array(time)

    def plot_data_distribution(self, outputs, inputs, show=False, save_path=None):
        data, time = outputs
        plt.figure()
        plt.hist(data, bins=20)
        plt.title(f"Data distribution (mean={data.mean():.2f}, std={data.std():.2f})")
        plt.xlabel("Pattern Indes")
        plt.ylabel("Count")
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'data_distribution.png'))
        if show:
            plt.show()
        plt.close('all')


@dataclass
class ParseInputConstPulse(OperatorMixin):
    """
    Used for constant pulse input
    """
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

        time = []
        data = []
        on = timestamps[states == self.TTL_state]

        self.logger.info(f"{len(on)=}")
        self.logger.info(f"TTL states in this recording: {np.unique(states, return_counts=True)=}")

        if len(on) <= 2:
            return np.array(data), np.array(time)

        time = on[::2]

        val = on[1::2] - on[::2]
        num_patterns = int(np.round(val.max() / val.min()))
        self.logger.info(f"{num_patterns=}")
        data = np.round(val / val.min()).astype(int)
        return data, time

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
        ax2.plot(time, y, 'or')

        ax1.set_xlabel('time (sec)')
        ax1.set_ylabel('TTL state')
        ax2.set_ylabel('Pattern')
        idx = 0
        while stime < etime:
            plt.xlim(stime, stime+interval)
            plt.savefig(os.path.join(save_path, f"ttl_state_{idx:04d}.png"))
            stime += interval
            idx += 1
        plt.close('all')

    def plot_data_distribution(self, outputs, inputs, show=False, save_path=None):
        data, time = outputs
        plt.figure()
        plt.hist(data, bins=20)
        plt.title(f"Data distribution (mean={data.mean():.2f}, std={data.std():.2f})")
        plt.xlabel("Pattern Indes")
        plt.ylabel("Count")
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'data_distribution.png'))
        if show:
            plt.show()
        plt.close('all')


@dataclass
class ParseTemporalDecoding(OperatorMixin):
    tag: str = "temporal decoding"

    _N = 3
    _Extra = 1

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, spikestamps, input_parse):
        _, probe_times = input_parse

        _N = self._N
        _Extra = self._Extra
        Xs = np.zeros((probe_times.shape[0], len(spikestamps) * _N + _Extra))

        # Time
        Xs[:, 0] = np.linspace(0, 1, probe_times.shape[0])
        for idx, spiketrain in enumerate(spikestamps):
            idx_N = idx * _N + _Extra

            # Decay spike count
            #Xs[:, idx_N + 0] = decay_spike_counts(
            #    np.asarray(spiketrain), probe_times, decay_rate=1
            #)

            # Firing Rates
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

    def plot_output(self, outputs, inputs, show=False, save_path=None):
        Xs = outputs
        spikestamps, (y, probe_times) = inputs
        _N = self._N
        _Extra = self._Extra
        n_channels = len(spikestamps)

        ids = np.sort(np.unique(y))
        npatterns = ids.size

        for channel in range(n_channels):
            fig, axes = plt.subplots(1, _N, figsize=(16,12), sharex=True, sharey=True)
            for n in range(_N):
                X = []
                for pattern in ids:
                    indices = y==pattern
                    X.append(Xs[indices, _Extra+n+_N*channel])

                axes[n].boxplot(X, positions=ids)
            axes[0].set_ylabel('val')

            for n in range(_N):
                axes[n].set_xlabel('patterns')

            if save_path is not None:
                plt.savefig(os.path.join(save_path, f'states_{channel}.png'))
            plt.close('all')
