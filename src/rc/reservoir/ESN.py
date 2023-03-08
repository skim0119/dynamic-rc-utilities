__all__ = ["ESN"]

from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

# TODO: Implement subsampling

class ESN:
    """
    Echo State Network Class

    Attributes
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
    Win : np.array
        input to reservoir matrix [in_size+1 , reservoir_size]
    Wout : np.array
        output to reservoir [reservoir-size, out_size]
    W : np.array
        reservoir to reservoir matrix [reservoir_size , reservoir_size]
    R : np.array
        reservoir activation [1 , reservoir_size]
    """

    def __init__(
        self,
        in_size,
        out_size,
        reservoir_size,
        alpha,
        subsample_size:Optional[int]=None,
        sparsity=0.2,
        spectral_radius=0.99,
        seed=None,
        non_linearity:Callable[[float], float]=np.tanh
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

        """
        # Random 
        # TODO: Maybe share the rng with other modules
        self.rng = np.random.default_rng(seed)

        self.in_size = in_size
        self.reservoir_size = reservoir_size
        self.out_size = out_size
        self.alpha = alpha
        self.sparsity = sparsity
        self.non_linearity = non_linearity
        if subsample_size is None:
            subsample_size = reservoir_size
        self.subsample_size = subsample_size
        assert subsample_size <= reservoir_size
        self.sample_idx = self.rng.choice(reservoir_size, size=subsample_size, replace=False)

        # Input weight
        self.Win = (
            self.rng.normal(size=(in_size + 1, reservoir_size)) # +1 for bias
        )  
        # Reservoir weights
        self.W = (
            self.rng.normal(size=(reservoir_size, reservoir_size))
        ) 

        self.W[self.rng.random((reservoir_size, reservoir_size)) > self.sparsity] = 0 # Sparsity
        #force_grid_structure(self.W, self.rng, sparsity)
        #force_torus_structure(self.W, self.rng, sparsity)
        #force_cube_structure(self.W, self.rng, sparsity)

        self.W /= (self._get_spectral_radius() / spectral_radius)

        # Reservoir State
        self.R = self._reset_reservoir_state()

        # Output weight (pretrained)
        #self.state_size = 1 + in_size + reservoir_size
        self.state_size = self.subsample_size * 2
        self.Wout = self.rng.normal(size=(self.state_size, out_size+1))

    def _get_spectral_radius(self):
        return np.max(np.abs(np.linalg.eig(self.W)[0]))

    def _validate_input_data(self, data):
        assert len(data.shape) == 2, f"The shape of the input data must be [timesteps, input_size]. (Given {data.shape=})"

    def _reset_reservoir_state(self):
        return 0.1 * (np.ones((1, self.reservoir_size)) - 0.5)

    def train(self, X, y, new_start:bool=True, progbar=False):
        self._validate_input_data(X)
        dm,_ = self.predict(X, new_start, progbar=False)
        self.Wout = np.dot(np.linalg.pinv(dm), y)
        return dm

    def predict(self, data, new_start=True, progbar=False):
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
            u = data[t] 

            # reservoir stepper
            # first summand: influence of last reservoir state (same neuron)
            # second summand:
            #   first dot product: influence of input
            #   second dot product: influence of of last reservoir
            #                       state (other neurons)
            #self.R = (1 - self.alpha) * self.R + self.alpha * self.non_linearity(
            #    np.dot(np.hstack((1, u)), self.Win) + np.dot(self.R, self.W)
            self.R = (1 - self.alpha) * self.R + self.alpha * (
                np.dot(np.hstack((1, u)), self.Win) + np.dot(self.R, self.W)
                #np.dot(np.hstack((1, u)), self.Win) # No recurrency
            )

            # put bias, input & reservoir activation into one row
            #state = np.append(np.append(1, u), self.R)
            subR = self.R[:,self.sample_idx]
            state = np.concatenate([subR, subR**2], axis=1)
            dm[t] = state
        y = np.dot(dm, self.Wout)
        self.dm = dm
        return dm, y

    def plot_reservoir(
        self,
        path="images/",
        name="Plot",
        nr_neurons=20,
        max_plot_cycles=100,
        plot_show=False,
    ):
        """
        Plotting reservoir states and their inputs from last use of .reservoir()
        are saved in a figure (and optionally displayed)

        Args:
            path (string): path of the saved plot png
            name (string): title of plot and name of saved .png
            nr_neurons (int): nr of neurons to be plotted
            max_plot_cycles (int): max number of cycles to be plotted
            plot_show (boolean): display plot?
        """
        # for plotting, separate bias, input, and real reservoir activations
        # which were saved all together in the res_history
        R = self.dm[:, -self.reservoir_size :]
        # remove bias and get input
        #R_input = self.dm[:, 1 : -self.reservoir_size]

        # check if we are below max_plot_cycles
        #if R_input.shape[0] > max_plot_cycles:
        #    limit = max_plot_cycles
        #else:
        #    limit = R_input.shape[0]

        plt.figure("Reservoir Activity", figsize=(20, 10)).clear()
        plt.title("Reservoir Activity")
        #plt.plot(R_input[:limit], color="k", label="Input Signals", linewidth=4)
        plt.plot(R[:, :nr_neurons], linewidth=2)
        plt.legend(loc="upper right")

        plt.savefig(path + name + "_ReservoirActivity" + ".png")
        print("\t[+]Plot saved in", path + name + "_ReservoirActivity" + ".png")

        if plot_show:
            plt.show()

    def save_dm(self, path="csv_files/", name="ESN"):
        """
        Saves current design matrix in a csv file

        Args:
            path (string): path of the saved csv file
            name (string): name of saved csv file
        """
        f = open(path + name + ".csv", "w")
        writer = csv.writer(f)

        # create header
        # get shape of input by subracting reservoir_size and bias
        input_shape = self.dm.shape[1] - (self.reservoir_size + 1)
        header = ["Bias"]
        for i in range(input_shape):
            header.append("Input" + str(i + 1))
        for i in range(self.reservoir_size):
            header.append("Neuron" + str(i + 1))
        writer.writerow(header)
        writer.writerows(self.dm)
        print("\t[+]CSV file saved in", path + name + ".csv")

def force_grid_structure(adj, rng, threshold):
    rank = adj.shape[0]
    x_coord = rng.uniform(0,1,size=rank)
    y_coord = rng.uniform(0,1,size=rank)

    for i in range(rank):
        x = x_coord[i]
        y = y_coord[i]

        mask = np.logical_or(np.abs(x - x_coord) > threshold, np.abs(y-y_coord) > threshold)
        adj[i, mask] = 0
        adj[mask, i] = 0
    print(f"{(~np.isclose(adj, 0.0)).sum()=}")

def force_cube_structure(adj, rng, threshold):
    rank = adj.shape[0]

    x_coord = rng.uniform(0,1,size=rank)
    y_coord = rng.uniform(0,1,size=rank)
    z_coord = rng.uniform(0,1,size=rank)

    for i in range(rank):
        x = x_coord[i]
        y = y_coord[i]
        z = z_coord[i]

        r = np.sqrt((z - z_coord) ** 2 + (y - y_coord) ** 2 + (x - x_coord) ** 2)
        mask = r > threshold
        adj[i, mask] = 0
        adj[mask, i] = 0
    print(f"{(~np.isclose(adj, 0.0)).sum()=}")

def force_torus_structure(adj, rng, threshold, R=2, r=1):
    rank = adj.shape[0]

    phi = rng.uniform(0,2*np.pi,size=rank)
    psi = rng.uniform(0,2*np.pi,size=rank)

    x_coord = (R + r*np.cos(phi))*np.cos(psi)
    y_coord = (R + r*np.cos(phi))*np.sin(psi)
    z_coord = r*np.sin(phi)

    for i in range(rank):
        x = x_coord[i]
        y = y_coord[i]
        z = z_coord[i]

        r = np.sqrt((z - z_coord) ** 2 + (y - y_coord) ** 2 + (x - x_coord) ** 2)
        mask = r > threshold
        adj[i, mask] = 0
        adj[mask, i] = 0
    print(f"{(~np.isclose(adj, 0.0)).sum()=}")
