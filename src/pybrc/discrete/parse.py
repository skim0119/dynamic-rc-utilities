import os, sys
import multiprocessing as mp
import numpy as np
from dataclasses import dataclass

import matplotlib.pyplot as plt

from miv.core.datatype import Spikestamps
from miv.statistics import decay_spike_counts, spike_counts_with_kernel

import pickle as pkl

from tqdm import tqdm

from miv.core.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.visualization.event import plot_spiketrain_raster

from pybrc.utils import get_nearest
from pybrc.discrete.base import BaseParseInput


@dataclass
class ParseInput(BaseParseInput):
    binsize_threshold: float = 0.001  # sec

    @cache_call
    def __call__(self, ttl_signal):
        states = ttl_signal[0]
        timestamps = ttl_signal.timestamps

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


@dataclass
class ParseInputConstPulse(BaseParseInput):

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
        binsize = probe_times[1] - probe_times[0]

        _N = self._N
        _Extra = self._Extra
        Xs = np.zeros((probe_times.shape[0], len(spikestamps) * _N + _Extra))

        # Time
        Xs[:, 0] = np.linspace(0, 1, probe_times.shape[0])
        for idx, spiketrain in enumerate(spikestamps):
            idx_N = idx * _N + _Extra

            # Decay spike count
            Xs[:, idx_N + 0] = decay_spike_counts(
                np.asarray(spiketrain), probe_times, decay_rate=5
            )
            Xs[:, idx_N + 1] = decay_spike_counts(
                np.asarray(spiketrain), probe_times - (binsize / 3), decay_rate=5
            )
            Xs[:, idx_N + 2] = decay_spike_counts(
                np.asarray(spiketrain), probe_times - 2 * (binsize / 3), decay_rate=5
            )

            # Firing Rates
            #Xs[:, idx_N + 0] = spike_counts_with_kernel(
            #    np.asarray(spiketrain), probe_times, lambda x: np.logical_and(x>0, x<1).astype(np.float_)
            #)
            #Xs[:, idx_N + 1] = spike_counts_with_kernel(
            #    np.asarray(spiketrain), probe_times, lambda x: np.logical_and(x>0, x<0.5).astype(np.float_)
            #)
            #Xs[:, idx_N + 2] = spike_counts_with_kernel(
            #    np.asarray(spiketrain), probe_times, lambda x: np.logical_and(x>0, x<0.25).astype(np.float_)
            #)

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

    def plot_spiketrain_for_each_label(self, outputs, inputs, show=False, save_path=None):
        # Export
        Xs = outputs
        spikestamps, (y, probe_times) = inputs
        binsize = probe_times[1] - probe_times[0]

        first_spiketime = spikestamps.get_first_spikestamp()
        for pattern in np.sort(np.unique(y)):
            indices = y == pattern
            path_name = os.path.join(self.analysis_path, f"pattern_{pattern}")
            os.makedirs(path_name, exist_ok=True)

            for idx, stime in enumerate(probe_times[indices]):
                stime += first_spiketime
                fig, ax = plot_spiketrain_raster(
                        spikestamps,
                        stime,
                        stime + binsize
                        )
                plt.savefig(os.path.join(path_name, f"occurance_{idx}.png"))
                plt.close('all')


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

    def plot_output_in_csv(self, outputs, inputs, show=False, save_path=None):
        # Export
        Xs = outputs
        spikestamps, (y, probe_times) = inputs
        _N = self._N
        _Extra = self._Extra
        n_channels = len(spikestamps)

        header = ['time']
        for ch in range(n_channels):
            header.extend([f"ch{ch}_{n}" for n in range(_N)])
        np.savetxt(
            os.path.join(save_path, "Xs.csv"),
            Xs,
            delimiter=",",
            header=",".join(header),
        )
        np.savetxt(
            os.path.join(save_path, "ys.csv"),
            np.vstack([probe_times, y]).T,
            delimiter=",",
            header=",".join(header),
        )


@dataclass
class ParseDecodingUniform(OperatorMixin):
    tag: str = "temporal decoding"

    _N = 3
    _Extra = 1

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, spikestamps):
        input_pkl_path = self.path

        with open(self.path, "rb") as f:
            data = pkl.load(f)
            y, _ = data
            self.logger.info(f"loaded: {data}")
            self.logger.info(f"{y.shape=}")

        N = y.shape[0]

        probe_times = np.arange(N) * self.binsize + self.delay_start
        _N = self._N
        _Extra = self._Extra

        if len(probe_times) < 2:
            raise RuntimeError("Input TTL signal not found. Check if it is in the data")

        Xs = np.zeros((probe_times.shape[0], len(spikestamps) * _N + _Extra))

        # Time
        Xs[:, 0] = np.linspace(0, 1, probe_times.shape[0])
        first_spikestamp = spikestamps.get_first_spikestamp()
        for idx, spiketrain in enumerate(spikestamps):
            spiketrain = np.asarray(spiketrain) - first_spikestamp
            idx_N = idx * _N + _Extra

            # Decay spike count
            Xs[:, idx_N + 0] = decay_spike_counts(
                spiketrain, probe_times, decay_rate=5
            )
            Xs[:, idx_N + 1] = decay_spike_counts(
                spiketrain, probe_times - (self.binsize / 3), decay_rate=5
            )
            Xs[:, idx_N + 2] = decay_spike_counts(
                spiketrain, probe_times - 2 * (self.binsize / 3), decay_rate=5
            )

            # Firing Rates
            #Xs[:, idx_N + 0] = spike_counts_with_kernel(
            #    spiketrain, probe_times, lambda x: np.logical_and(x>0, x<1).astype(np.float_)
            #)
            #Xs[:, idx_N + 1] = spike_counts_with_kernel(
            #    spiketrain, probe_times, lambda x: np.logical_and(x>0, x<0.5).astype(np.float_)
            #)
            #Xs[:, idx_N + 2] = spike_counts_with_kernel(
            #    spiketrain, probe_times, lambda x: np.logical_and(x>0, x<0.25).astype(np.float_)
            #)

            # Vector of tanhs
            # Xs[:, idx_N + 0] = spike_counts_with_kernel(
            #     spiketrain, probe_times, lambda x: 1.0 - np.tanh(x)
            # )
            # Xs[:, idx_N + 1] = spike_counts_with_kernel(
            #     spiketrain, probe_times, lambda x: np.tanh(x)
            # )
            # Xs[:, idx_N + 2] = spike_counts_with_kernel(
            #     spiketrain, probe_times, lambda x: np.tanh(-x)
            # )
            # Xs[:, idx_N + 3] = spike_counts_with_kernel(
            #     spiketrain, probe_times, lambda x: -1.0 - np.tanh(-x)
            # )
        self.logger.info(f"    {Xs.mean()=}, {Xs.std()=}")
        self.logger.info(f"    {Xs.shape=}")

        return Xs

    def plot_spiketrain_for_each_label(self, outputs, inputs, show=False, save_path=None):
        # Export
        Xs = outputs
        spikestamps = inputs
        with open(self.path, "rb") as f:
            data = pkl.load(f)
            y, _ = data
            self.logger.info(f"loaded: {data}")
            self.logger.info(f"{y.shape=}")
        N = y.shape[0]
        probe_times = np.arange(N) * self.binsize + self.delay_start

        first_spiketime= spikestamps.get_first_spikestamp()

        for pattern in np.sort(np.unique(y)):
            indices = y == pattern
            path_name = os.path.join(self.analysis_path, f"pattern_{pattern}")
            os.makedirs(path_name, exist_ok=True)

            for idx, stime in enumerate(probe_times[indices]):
                stime += first_spiketime
                fig, ax = plot_spiketrain_raster(
                        spikestamps,
                        stime,
                        stime + self.binsize
                        )
                plt.savefig(os.path.join(path_name, f"occurance_{idx}.png"))
                plt.close('all')

    def plot_output(self, outputs, inputs, show=False, save_path=None):
        Xs = outputs
        spikestamps = inputs

        with open(self.path, "rb") as f:
            data = pkl.load(f)
            y, _ = data
            self.logger.info(f"loaded: {data}")
            self.logger.info(f"{y.shape=}")

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

    def plot_output_in_csv(self, outputs, inputs, show=False, save_path=None):
        # Export
        Xs = outputs
        spikestamps = inputs
        with open(self.path, "rb") as f:
            data = pkl.load(f)
            y, _ = data
            self.logger.info(f"loaded: {data}")
            self.logger.info(f"{y.shape=}")
        N = y.shape[0]
        probe_times = np.arange(N) * self.binsize + self.delay_start
        _N = self._N
        _Extra = self._Extra
        n_channels = len(spikestamps)

        header = ['time']
        for ch in range(n_channels):
            header.extend([f"ch{ch}_{n}" for n in range(_N)])
        np.savetxt(
            os.path.join(save_path, "Xs.csv"),
            Xs,
            delimiter=",",
            header=",".join(header),
        )
        np.savetxt(
            os.path.join(save_path, "ys.csv"),
            np.vstack([probe_times, y]).T,
            delimiter=",",
            header=",".join(header),
        )
