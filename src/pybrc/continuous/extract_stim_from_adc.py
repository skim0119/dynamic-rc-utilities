import os
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
from dataclasses import dataclass

from miv.core.operator import OperatorMixin
from miv.core.datatype import Signal
from miv.core.wrapper import cache_call


@dataclass
class ExtractStimulusFromADC(OperatorMixin):
    num_channels: int
    adc_index: int

    target_sampling: int

    tag:str = "stimulus channel"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, data, spikestamps):
        s = []
        t = []
        rate = None
        index = self.num_channels + self.adc_index - 1
        for idx, signal in enumerate(data):
            s.extend(signal[index])
            t.extend(signal.timestamps)
            rate = signal.rate

        tstart, tend = min(t), max(t)
        q = rate // self.target_sampling
        #new_s = sps.decimate(s, q, ftype='fir')
        #new_t = np.linspace(tstart, tend, new_s.size)
        new_s = np.array(s)[::q] / 1.0e6
        new_t = np.array(t)[::q]

        signal = Signal(new_s[:,None], new_t, self.target_sampling)

        #DEBUG
        os.makedirs(self.analysis_path, exist_ok=True)
        fig, axes = plt.subplots(2,1,figsize=(16,10),gridspec_kw={'height_ratios': [1, 4]}, sharex=True)
        axes[0].plot(new_t, new_s)

        axes[1].eventplot(spikestamps.data)

        interval = 20
        tstart -= interval * 0.1
        tend += interval * 0.1
        _start, _end = tstart, tstart + interval
        while _start < tend:
            axes[0].set_xlim(_start, _end)
            axes[1].set_xlim(_start, _end)
            plt.savefig(os.path.join(self.analysis_path, f"window_{_start:.2f}_{_end:.2f}.png"))
            _start += interval
            _end += interval

        plt.close('all')

        return signal
