__all__ = ["parse_event_data", "parse_spiketrain", "spike_decoding"]

import os, sys
import multiprocessing as mp
import numpy as np

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


def parse_event_data(data, binsize, path, verbose:bool=True, force:bool=False, binsize_threshold:float=0.001):
    if not verbose:
        vprint = lambda x: x
    else:
        vprint = print
    if not force and os.path.exists(path):
        vprint(f"\t[-] parse_event_data: The path ({path}) already exists for {data=}.")
        data = np.load(path)
        return data["data"], data["time"]
    assert data is not None

    ttl_signal = data.load_ttl_event()
    states = ttl_signal[0]
    timestamps = ttl_signal.timestamps

    on = timestamps[states == 1]
    off = timestamps[states == -1]
    print(f"{len(on)=}")
    print(f"{len(off)=}")
    print(f"{np.unique(states, return_counts=True)=}")

    off_probe = 0
    time = []
    data = []
    front_bar = on[0]
    while front_bar < off[-1]:
        next_bar = front_bar + binsize

        # Correct "next_bar" for little offset
        _nearest_next_on, delta, _ = get_nearest(on, next_bar)
        if delta < binsize_threshold:
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

    np.savez(path, data=np.array(data), time=np.array(time))
    vprint(f"\t[+] parse_event_data: Data saved in ({path}).")
    return np.array(data), np.array(time)


def parse_spiketrain(data, path, verbose:bool=True, force:bool=False, impedances=None):
    if not verbose:
        vprint = lambda x: x
    else:
        vprint = print
    if not force and os.path.exists(path):
        vprint(f"\t[-] parse_spiketrain: The path ({path}) already exists for {data=}.")
        with open(path, "rb") as handle:
            total_spikestamps = pkl.load(handle)
        return total_spikestamps
    assert data is not None

    bandpass_filter = ButterBandpass(lowcut=400, highcut=1500, order=4)
    spike_detection = ThresholdCutoff()
    data >> bandpass_filter >> spike_detection
    Pipeline(spike_detection).run(data.analysis_path, verbose=True, skip_plot=True)

    total_spikestamps = spike_detection.output()
    if impedances is not None:
        channels_with_impedances_in_range = list(impedances.keys())
        total_spikestamps = total_spikestamps.select(channels_with_impedances_in_range)
    if path is not None:
        with open(path, "wb") as handle:
            pkl.dump(total_spikestamps, handle, protocol=pkl.HIGHEST_PROTOCOL)
        vprint(f"\t[+] parse_spiketrain: Data saved in ({path}).")

    return total_spikestamps

def spike_decoding(spikestamps, probe_times, path:str=None, verbose:bool=True, force:bool=False, progress_bar:bool=False):
    if not verbose:
        vprint = lambda x: x
    else:
        vprint = print
    if not force and path is not None and os.path.exists(path):
        vprint(f"\t[-] spike decoding: The path ({path}) already exists.")
        data = np.load(path)
        return data["X"]

    _N = 1
    Xs = np.zeros((probe_times.shape[0], len(spikestamps) * _N))
    for idx, spiketrain in tqdm(enumerate(spikestamps), disable=not progress_bar, total=len(spikestamps)):
        idx_N = idx * _N
        # Decay spike count
        #Xs[:, idx_N + 0] = decay_spike_counts(
        #    np.asarray(spiketrain), probe_times, decay_rate=1
        #)
        # Firing Rante
        Xs[:, idx_N + 0] = spike_counts_with_kernel(
            np.asarray(spiketrain), probe_times, lambda x: np.logical_and(x>0, x<1).astype(np.float_)
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
    print("    ", Xs.mean(), Xs.std())
    if path is not None:
        np.savez(path, X=np.array(Xs))
        vprint(f"\t[+] spike decoding: Data saved in ({path}).")
    return Xs

def parse_spiketrain_intan(data, path, preprocess=None, verbose:bool=True, force:bool=False):
    if not verbose:
        vprint = lambda x: x
    else:
        vprint = print
    if not force and os.path.exists(path):
        vprint(f"\t[-] parse_spiketrain: The path ({path}) already exists for {data=}.")
        with open(path, "rb") as handle:
            total_spikestamps = pkl.load(handle)
        return total_spikestamps

    assert data is not None
    bandpass_filter = ButterBandpass(lowcut=300, highcut=3000, order=4)
    spike_detection = ThresholdCutoff()
    data >> bandpass_filter >> spike_detection
    Pipeline(spike_detection).run(data.analysis_path)
    total_spikestamps = spike_detection.output()

    with open(path, "wb") as handle:
        pkl.dump(total_spikestamps, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    return total_spikestamps
