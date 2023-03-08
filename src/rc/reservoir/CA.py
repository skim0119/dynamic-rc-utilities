__all__ = ["CA"]

from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

import cellpylib as cpl

class CA:
    """
    Cellular Automata
    """

    def __init__(
        self,
        in_size,
        out_size,
        reservoir_size,
        rule_index,
        subsample_size:Optional[int]=None,
        seed=None,
    ):
        """
        Echo-state network constructed with random connection, sparsity, and leaky rate.

        Parameters
        ----------
        in_size : int
            dimension of inputs
        out_size : int
            dimension of outputs
        reservoir_size : int
            size of reservoir
        alpha : float
            leaking rate
        sparsity : float
            portion of connections != 0
        rule_index : int
            Wolfram rule index

        """
        # Random 
        # TODO: Maybe share the rng with other modules
        self.rng = np.random.default_rng(seed)

        self.rule = rule_index
        self.ca_iteration = 4

        self.in_size = in_size
        self.out_size = out_size
        self.reservoir_size = reservoir_size
        if subsample_size is None:
            subsample_size = reservoir_size * self.ca_iteration
        self.subsample_size = subsample_size
        self.sample_idx = self.rng.choice(reservoir_size * self.ca_iteration, size=subsample_size, replace=False)

        # Input weight
        self.Win = np.zeros((in_size, reservoir_size)) 
        ind = self.rng.choice(np.arange(4), reservoir_size)
        self.Win[ind, np.arange(reservoir_size)] = 1

        # Output weight (pretrained)
        #self.state_size = 1 + in_size + reservoir_size
        self.state_size = self.subsample_size
        self.Wout = self.rng.normal(size=(self.state_size, out_size))

    def _validate_input_data(self, data):
        assert len(data.shape) == 2, f"The shape of the input data must be [timesteps, input_size]. (Given {data.shape=})"

    def _reset_reservoir_state(self):
        return np.zeros((1, self.reservoir_size))

    def train(self, X, y, new_start:bool=True, progbar=False):
        self._validate_input_data(X)
        dm,_ = self.predict(X, new_start, progbar=False)
        self.Wout = np.dot(np.linalg.pinv(dm), y)
        return dm

    def predict(self, data, new_start=False, progbar=False):
        """
        Parameters
        ----------
        data :
            input data points [cycles , in_size]
        new_start :
            reset reservoir state
        """
        self._validate_input_data(data)
        if new_start:
            self.R = self._reset_reservoir_state()

        # Reservoir stepper
        num_steps = data.shape[0]

        dm = np.zeros((num_steps, self.state_size))

        for t in tqdm(range(data.shape[0]), disable=not progbar):
            u = np.logical_or(data[t] @ self.Win, self.R[-1])[None, :]
            
            self.R = cpl.evolve(u, timesteps=self.ca_iteration, memoize=True,
                                            apply_rule=lambda n, c, t: cpl.nks_rule(n, self.rule))

            # put bias, input & reservoir activation into one row
            state = self.R.ravel()[self.sample_idx]
            dm[t] = state
        y = np.dot(dm, self.Wout)
        return dm, y
