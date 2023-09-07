import os
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
from dataclasses import dataclass

from miv.core.operator import OperatorMixin
from miv.core.datatype import Signal
from miv.core.wrapper import cache_call

from miv.statistics import spike_counts_with_kernel
from miv.utils.progress_logging import pbar

@dataclass
class RateState(OperatorMixin):
    augmented_N: int = 1

    tag:str = "rate state"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, spikestamps, stimulus_signal):
        probe_times = stimulus_signal.timestamps
 
        Xs = np.zeros((probe_times.shape[0], spikestamps.number_of_channels * self.augmented_N))
        for idx, spiketrain in enumerate(pbar(spikestamps, logger=self.logger)):
            idx_N = idx * self.augmented_N
            Xs[:, idx_N + 0] = spike_counts_with_kernel(
                 np.asarray(spiketrain),
                 probe_times,
                 lambda x: np.logical_and(x>0, x<1).astype(np.float_)
             )
        self.logger.info(f"state shape: {Xs.shape=}")

        return Xs
