import os
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from dataclasses import dataclass
from collections import OrderedDict

import json

from miv.core.operator import OperatorMixin
from miv.core.datatype import Signal
from miv.core.wrapper import cache_call

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error

from pybrc.kernel_rank import kernel_rank
from pybrc.memory_capacity import memory_capacity

@dataclass
class AmplitudeRCAnalysis(OperatorMixin):
    """
    Amplitude encoding RC
    """
    # Memory Capacity Configuration
    tau_max:int = 100

    tag:str = "rc amplitude"

    def __post_init__(self):
        super().__init__()

    #@cache_call
    def __call__(self, input_signal, states):
        os.makedirs(self.analysis_path, exist_ok=True)
        X = states
        y = input_signal[0]

        log = OrderedDict([])
        self.statistics(log, X, y)

        # Kernel rank
        log["kernel_rank"] = int(kernel_rank(X))
        self.logger.info("Computed kernel rank")

        # Find regularization factor
        alpha = self.find_regularization(X, y)
        log["regression regularization (alpha)"] = float(alpha)

        # Memory capacity
        mc = self.compute_memory_capacity(X, y, alpha)
        self.logger.info("Computed memory capacity")

        # Train
        clf = self.train(X, y, alpha)
        log["coefficient of determination (R2)"] = clf.score(X, y)
        y_pred = clf.predict(X)
        log["mean absolute error"] = mean_absolute_error(y, y_pred)

        # Save log
        log_json = json.dumps(log, indent=4)
        with open(os.path.join(self.analysis_path, "output.json"), "w") as outfile:
            outfile.write(log_json)

    def statistics(self, log, X, y):
        # Dataset Construction
        log["X shape"] = X.shape
        log["y shape"] = y.shape
        self.logger.info(f"X shape: {X.shape}")
        self.logger.info(f"y shape: {y.shape}")
        
        # Plot distribution
        plt.figure()
        plt.hist(y, bins=50, density=True)
        plt.xlabel("input")
        plt.ylabel("count (normalized)")
        plt.title("Input signal distribution")
        plt.close('all')

    def find_regularization(self, X, y):
        data = []
        cv_score_max = -99999
        c_pick = 0  # Selected factor
        reg_range = np.power(10.0, np.arange(-4,1,0.5))
        for c in reg_range:
            clf = Ridge(alpha=c)
            clf.fit(X,y)
            comp_value = clf.score(X, y)
            data.append(comp_value)
            if cv_score_max < comp_value:
                cv_score_max = comp_value
                c_pick = c

        plt.figure()
        plt.plot(reg_range, data)
        plt.axvline(c_pick)
        plt.xticks(rotation=90)
        plt.xlabel("Regularization Factor")
        plt.ylabel("validation score")
        plt.title("hyper-parameter tuning")
        plt.savefig(os.path.join(self.analysis_path, "regularization_tuning.png"))
        plt.close()

        return c_pick

    def compute_memory_capacity(self, X, y, alpha):
        mc = memory_capacity(X,y,tau_max=self.tau_max,discrete=False,regularization_factor=alpha)
        plt.figure()
        plt.plot(mc)
        plt.xlabel("Tau")
        plt.ylabel("Correlation")
        plt.title("Memory Capacity")
        plt.savefig(os.path.join(self.analysis_path, "memory_capacity.png"))
        plt.close()
        return mc

    def train(self, X, y, alpha, skip_percentage=0.0):
        skip_length = int(skip_percentage * X.shape[0])
        clf = Ridge(alpha)
        clf.fit(X[skip_length:],y[skip_length:])
        return clf

    def tuning_curve(self, X, y, alpha):
        from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
        #TODO
        # Tunign curve
        reg = LinearRegression()
        fig, axes = plt.subplots(1,2,figsize=(16,8))
        for channel in np.arange(X.shape[1]):
            tune = np.zeros(len(patterns))
            dev = np.zeros(len(patterns))
            for idx, pattern in enumerate(patterns):
                m = y==pattern
                v = X[m,channel]
                tune[idx] = v.mean()
                dev[idx] = v.std()
            reg.fit(patterns[:,None], tune[:,None])    
            if reg.coef_ > 0:
                axes[0].plot(patterns, tune, marker='o', label=f"c{channel}")
            else:
                axes[1].plot(patterns, tune, marker='o', label=f"c{channel}")
        axes[1].legend()
        axes[0].set_xlabel("frequency")
        axes[1].set_xlabel("frequency")
        axes[0].set_ylabel("firing rate")
        axes[1].set_ylabel("firing rate")
        axes[0].set_title("Excitatory")
        axes[1].set_title("Inhibitory")
        plt.savefig(os.path.join(self.analysis_path,"tuning_curve.png"))
        plt.close()
        print("done: tuning curve")

        # Learning Curve
        size, train_scores, test_scores = learning_curve(clf, X, y, train_sizes=np.linspace(0.1, 1.0, 20), cv=10)
        plt.figure()
        plt.plot(size, train_scores.mean(axis=1), label="train", marker='o')
        plt.fill_between(size,
                         train_scores.mean(axis=1)-train_scores.std(axis=1),
                         train_scores.mean(axis=1)+train_scores.std(axis=1),
                         alpha=0.2)
        plt.plot(size, test_scores.mean(axis=1), label="test", marker='o')
        plt.fill_between(size,
                         test_scores.mean(axis=1)-test_scores.std(axis=1),
                         test_scores.mean(axis=1)+test_scores.std(axis=1),
                         alpha=0.2)
        plt.legend()
        plt.xlabel("Number of Samples")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.savefig(os.path.join(self.analysis_path, "learning_curve.png"))
        plt.close()
        print("done: learning curve")
