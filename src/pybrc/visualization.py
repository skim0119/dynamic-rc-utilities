from typing import Tuple, Sequence, Union, List, Optional, Callable

import os, sys
import time
import numpy as np
import pathlib
import datetime
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from tqdm import tqdm

from miv.core import Signal, Spikestamps, OperatorMixin


@dataclass
class PatternVisualization(OperatorMixin):
    training_interval: float = 1.0  # Seconds
    input_ttl_index:int = 1  # Typically used 1 for input, 5 for recording

    tag: str = "pattern visualization"
    progress_bar: bool = False

    def __post_init__(self):
        super().__init__()
        self.cacher.policy = "OFF"

    def __call__(self, spikestamps: Spikestamps, input_signal: Signal, patterns: List[int]):
        num_intervals = len(patterns)
        stim_times = input_signal.timestamps[input_signal[0]==self.input_ttl_index]
        indices = np.concatenate([[0], np.cumsum(patterns)])
        pattern_interval_bins = stim_times[indices]

        if stim_times.size == 0:
            return None
        if pattern_interval_bins.size != len(patterns) + 1:
            raise ValueError("Number of pattern_interval_bins does not match number of patterns")

        stacked_spikes_for_each_pattern = {}
        for nch in tqdm(range(spikestamps.number_of_channels)):
            spikes = spikestamps[nch]
            indices = np.digitize(spikes, bins=pattern_interval_bins)  # sorted order
            stack = [[] for _ in range(num_intervals)]
            for v, i in zip(spikes, indices):
                if i == 0 or i == len(pattern_interval_bins):
                    continue  # out of bounds
                pattern_start_time = pattern_interval_bins[i - 1]
                stack[i-1].append(v - pattern_start_time)
            stacked_spikes_for_each_pattern[nch] = stack

        return stacked_spikes_for_each_pattern, pattern_interval_bins
    
    def plot_all_patterns_per_channels(
        self,
        outputs,
        inputs,
        show:bool=False,
        save_path:pathlib.Path | None=None,
    ):
        stacked_spikes_for_each_pattern, pattern_interval_bins = outputs
        spikestamps, input_signal, patterns = inputs

        from joblib import Parallel, delayed
        Parallel(n_jobs=-1, verbose=100)(delayed(psth_plot)(
            stacked_spikes_for_each_pattern[nch],
            title=f"Channel {nch}",
            save_path=os.path.join(save_path, f"psth_channel_{nch}.png"),
        ) for nch in range(spikestamps.number_of_channels))


def psth_plot(data, title, save_path, xlim=(-0.05, 1.05)):
    fig, axes = plt.subplots(1,1, figsize=(10, 8))

    _ = axes.eventplot(data)
    axes.set_xlim(*xlim)
    axes.set_title(title)
    plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close(fig)