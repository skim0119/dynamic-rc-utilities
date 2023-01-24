__all__ = ["parse_event_data", "parse_spiketrain", "spike_decoding"]

import os, sys
import multiprocessing as mp
import numpy as np

from functools import partial

import matplotlib.pyplot as plt

from miv.io.data import Data, DataManager
from miv.signal.filter import FilterCollection, ButterBandpass, FilterProtocol
from miv.signal.spike import ThresholdCutoff, SpikeDetectionProtocol
from miv.core import Spikestamps
from miv.statistics import decay_spike_counts, spike_counts_with_kernel

import pickle as pkl

from tqdm import tqdm

from rc.utils import get_nearest


def parse_event_data(data, binsize, path, verbose:bool=True, force:bool=False):
    if not verbose:
        vprint = lambda x: x
    else:
        vprint = print
    if not force and os.path.exists(path):
        vprint(f"\t[-] parse_event_data: The path ({path}) already exists for {data=}.")
        data = np.load(path)
        return data["data"], data["time"]
    assert data is not None
    states, full_words, timestamps, sampling_rate, initial_state = data.load_ttl_event()

    on = timestamps[states == 1]
    off = timestamps[states == -1]

    off_probe = 0
    time = []
    data = []
    front_bar = on[0]
    #pp = 1
    while front_bar < off[-1]:
        next_bar = front_bar + binsize
        _nearest_next_on, delta = get_nearest(on, next_bar)
        if delta < 0.001:
            next_bar = _nearest_next_on
        count = 0
        # TEMP
        if off_probe+1 >= len(off):
            break
        d = off[off_probe+1] - off[off_probe]
        while off_probe < len(off) and off[off_probe] < next_bar:
            off_probe += 1
            count += 1
        data.append(count)
        time.append(next_bar)

        #if count == 1:
        #    plt.eventplot(on-front_bar, color='black', lineoffsets=pp)
        #    plt.eventplot(off-front_bar, color='red', lineoffsets=pp)
        #    pp += 1
        
        # DEBUG
        if count == 7 or count == 8:
            if np.abs(d-0.1125) > np.abs(d-0.128566):
                data[-1] = 7

        front_bar = next_bar

    #plt.xlim([-0.1, 1.1])
    #plt.show()
    #sys.exit()

    np.savez(path, data=np.array(data), time=np.array(time))
    vprint(f"\t[+] parse_event_data: Data saved in ({path}).")
    return np.array(data), np.array(time)

def _preprocess(data, filter: FilterProtocol, detector: SpikeDetectionProtocol):
    signal, timestamps, sampling_rate = data
    timestamps *= sampling_rate  # Still not sure about this part...
    filtered_signal = filter(signal, sampling_rate)
    spiketrains = detector(
        filtered_signal,
        timestamps,
        sampling_rate,
        return_neotype=False,
        progress_bar=False,
    )
    return spiketrains


def parse_spiketrain(data, path, preprocess=None, verbose:bool=True, force:bool=False):
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
    pre_filter = FilterCollection(tag="Filter Example").append(
        ButterBandpass(lowcut=300, highcut=3000, order=4)
    )
    spike_detection = ThresholdCutoff()
    # Apply filter to `dataset[0]`

    num_fragments = 100
    total_spikestamps = Spikestamps([])
    with mp.Pool(8) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(preprocess, filter=pre_filter, detector=spike_detection),
                    data.load_fragments(num_fragments=num_fragments),
                ),
                total=num_fragments,
            )
        )
        for spikestamp in results:
            total_spikestamps.extend(spikestamp)
    if path is not None:
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
