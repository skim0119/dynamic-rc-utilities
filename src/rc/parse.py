__all__ = ["parse_event_data", "parse_spiketrain", "spike_decoding"]

import os, sys
import multiprocessing as mp
import numpy as np

from functools import partial

import matplotlib.pyplot as plt

from miv.io.openephys import Data, DataManager
from miv.signal.filter import FilterCollection, ButterBandpass, FilterProtocol
from miv.signal.spike import ThresholdCutoff, SpikeDetectionProtocol
from miv.core.datatype import Spikestamps
from miv.core.pipeline import Pipeline
from miv.statistics import decay_spike_counts, spike_counts_with_kernel

import pickle as pkl

from tqdm import tqdm

from rc.utils import get_nearest


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

def _preprocess(data, filter: FilterProtocol, detector: SpikeDetectionProtocol):
    signal, timestamps, sampling_rate = data
    # DEBUG
    #for ch in range(signal.shape[1]):
    #    print(ch, np.mean(signal[:,ch]), signal[:,ch].max(), signal[:,ch].min(), np.median(signal[:,ch]), np.std(signal[:,ch]))
    #input('pause')
    filtered_signal = filter(signal, sampling_rate)
    spiketrains = detector(
        filtered_signal,
        timestamps,
        sampling_rate,
        return_neotype=False,
        progress_bar=False,
    )
    return spiketrains


def parse_spiketrain(data, path, verbose:bool=True, force:bool=False):
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

    total_spikestamps = spike_detection.output.data
    if path is not None:
        with open(path, "wb") as handle:
            pkl.dump(total_spikestamps, handle, protocol=pkl.HIGHEST_PROTOCOL)
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
    Xs = np.empty((probe_times.shape[0], len(spikestamps) * _N))
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
    if path is not None:
        np.savez(path, X=np.array(Xs))
        vprint(f"\t[+] spike decoding: Data saved in ({path}).")
    return Xs

def parse_spiketrain_intan(data, path, delay=0, preprocess=None, verbose:bool=True, force:bool=False):
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
    if preprocess is None:
        preprocess = _preprocess
    pre_filter = ButterBandpass(lowcut=300, highcut=3000, order=4)
    spike_detection = ThresholdCutoff()

    #stim, timestamps, sampling_rate = data.get_stimulation()
    #stimulated_channels = np.where(np.abs(stim).sum(axis=0))[0]
    #stimulated_channel = stimulated_channels[0]
    #stim = stim[:, stimulated_channel]
    
    #events = ~np.isclose(stim, 0)
    #eventstrain = timestamps[np.where(events)[0]]
    #ref = np.concatenate([[True], np.diff(eventstrain) > refractory])
    #eventstrain = eventstrain[ref]
    
    total_spikestamps = Spikestamps([])
    for signal, timestamps, sampling_rate in tqdm(data.load(), total=len(data.get_recording_files())):
        filtered_signal = pre_filter(signal, sampling_rate)
        spikestamp = spike_detection(filtered_signal, timestamps, sampling_rate, return_neotype=False, progress_bar=False)
        total_spikestamps.extend(spikestamp)
    for i in range(len(total_spikestamps)):
        total_spikestamps[i] -= delay

    with open(path, "wb") as handle:
        pkl.dump(total_spikestamps, handle, protocol=pkl.HIGHEST_PROTOCOL)
    fig = plt.figure(figsize=(12, 12))
    plt.eventplot(total_spikestamps)
    plt.xlabel("time (sec)")
    plt.ylabel("channels")
    plt.title("spiketrain")
    plt.savefig(path + ".png")
    plt.close(fig)
    return total_spikestamps
