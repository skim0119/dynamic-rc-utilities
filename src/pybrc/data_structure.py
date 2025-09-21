from typing import TypeAlias, Callable, Iterable
from numpy.typing import NDArray

import os
import pickle as pkl
from dataclasses import dataclass, field, asdict
from functools import partial

import numpy as np
from tqdm import tqdm

from miv.core import Spikestamps
from miv.statistics import decay_spike_counts

Spikestamp: TypeAlias = Iterable[float]
InputTime: TypeAlias = Iterable[float]
FeatureArray: TypeAlias = NDArray[np.float64]
KernelType: TypeAlias = Callable[[Spikestamp, InputTime], FeatureArray]

_default_kernel = partial(decay_spike_counts, decay_rate=5.0)


@dataclass(eq=False, kw_only=True)
class ExperimentData:
    spikestamps: Spikestamps
    labels: NDArray[np.int32]
    input_time: NDArray[np.float64]

    tag: str = "temp"

    metadata: dict = field(default_factory=dict)

    # Derived properties
    feature_set: NDArray[np.float64] | None = None
    pattern_size: float = field(init=False)  # (s)
    pattern_rest: float = 0.1  # (s), pattern rest for refractory
    pattern_start_offset: float = 0.0  # (s), pattern start offset

    # Alias
    t: NDArray[np.float64] = field(init=False)
    y: NDArray[np.int32] = field(init=False)
    X: NDArray[np.float64] = field(init=False)

    def __len__(self):
        return self.labels.size

    @property
    def num_features(self):
        return self.feature_set.shape[1]

    def __post_init__(self):
        if self._is_cached():
            self._cache_load()
            return

        # Derive
        if self.feature_set is None:
            self.feature_set = self._compute_feature_set()
        self.pattern_size = np.median(np.diff(self.input_time))

        # Alias
        self.X = self.feature_set
        self.t = self.input_time
        self.y = self.labels

        assert self.X.shape[0] == self.t.shape[0]
        assert self.y.shape[0] == self.t.shape[0]

        self._cache_save()

    # ---------------- Derive -------------------
    def _compute_feature_set(self, kernel: KernelType | None = None, progbar=True):
        if kernel is None:
            kernel = _default_kernel

        X = np.ones((self.input_time.shape[0], len(self.spikestamps)))
        for idx, spiketrain in tqdm(
            enumerate(self.spikestamps),
            total=len(self.spikestamps),
            disable=not progbar,
            desc="Computing feature set",
        ):
            array_spiketrain = np.asarray(spiketrain)
            rate = kernel(array_spiketrain, self.input_time)
            X[:, idx] = rate
        return X

    # ---------------- Cache -------------------
    def _is_cached(self, directory="."):
        return os.path.exists(os.path.join(directory, f"{self.tag}.pkl"))

    def _cache_save(self, directory="."):
        path = os.path.join(directory, f"{self.tag}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pkl.dump(self, f)

    def _cache_load(self, directory="."):
        with open(os.path.join(directory, f"{self.tag}.pkl"), "rb") as f:
            data = pkl.load(f)
            for k, v in asdict(data).items():
                self.__dict__[k] = v

    # ---------------- Diagnostics -------------------
    def summary(self):
        print("Experiment data summary")
        print(f"number of channels: {self.spikestamps.number_of_channels}")
        print(f"pattern binsize  : {self.pattern_size}s")

        patterns, counts = np.unique(self.y, return_counts=True)
        print(f"Total number of patterns: {len(self.y)}")
        print(f"unique patterns : {patterns.tolist()}")
        print(f"pattern counts  : {counts.tolist()}")

    # ---------------- Getter -------------------
    def get_pattern_ids(self, pattern: int) -> NDArray[np.int32]:
        """
        Return array of pattern-ids that match the given pattern
        """
        return np.where(self.y == pattern)[0]

    def get_pattern_spikestamps(
        self, pattern_id: int, reset_start: bool = False
    ) -> Spikestamp:
        """
        Return spikestamps for the given pattern
        """
        T = self.input_time[pattern_id]
        return self.spikestamps.get_view(
            T - self.pattern_size,
            T,
            reset_start=reset_start,
        )

    def get_fine_feature(
        self, pattern_id, kernel: KernelType | None = None, N=100
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return feature for the given pattern in fine resolution
        """
        if kernel is None:
            kernel = _default_kernel

        probe_time = np.linspace(0, self.pattern_size, N)
        X = np.empty((len(self.spikestamps), N))
        spikestamps = self.get_pattern_spikestamps(pattern_id, reset_start=True)
        for idx, spiketrain in enumerate(spikestamps):
            array_spiketrain = np.asarray(spiketrain)
            rate = kernel(array_spiketrain, probe_time)
            X[idx] = rate
        return probe_time, X

    def get_smooth_pattern(self, pattern_id, bin_size=0.001, weight=0.67):
        """
        Return smooth shape of the given pattern.
        It uses bin-size and exponential moving average

        (number_of_channels, size)
        """
        from .utils import ema_smooth

        spikestamps = self.get_pattern_spikestamps(pattern_id, reset_start=True)
        binned_spikestamps = spikestamps.binning(
            bin_size, return_count=True, t_start=0, t_end=self.pattern_size
        )
        Xs = binned_spikestamps.data.astype(float)  # (size, num_channels)
        for channel in tqdm(range(spikestamps.number_of_channels)):
            Xs[:, channel] = ema_smooth(Xs[:, channel], weight=weight)
        return Xs.T

    def get_input_times(self, pattern_id: int) -> NDArray[np.float64]:
        T = self.input_time[pattern_id]
        pattern = self.y[pattern_id]
        return (
            np.linspace(T - self.pattern_size, T - self.pattern_rest, pattern + 1)[:-1]
            + self.pattern_start_offset
        )
