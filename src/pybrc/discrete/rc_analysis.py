from typing import List, Tuple, Callable

import os, sys
import multiprocessing as mp
import numpy as np
import itertools
from functools import partial
from collections import OrderedDict

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, f1_score, mean_absolute_error, balanced_accuracy_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import scipy.stats as spst
import scipy

from miv.core.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.datatype import Signal

import json

from tqdm import tqdm

from pybrc.kernel_rank import kernel_rank
from pybrc.memory_capacity import memory_capacity


@dataclass
class DiscreteTemporalRCAnalysis(OperatorMixin):
    rests: List
    tau_max: int = 100

    target_function: Callable[[np.ndarray], np.ndarray] | None = None

    tag: str = "rc discrete analysis"

    def __post_init__(self):
        super().__init__()

    def __call__(self, training_states, input_signal, *test_input_states_tuples):
        """
        test_input_state_tuples contain remaining tuples as:
        ((test_X1, test_y1), (test_X2, test_y2), ...)
        """
        log = OrderedDict([])

        os.makedirs(self.analysis_path, exist_ok=True)
        self.logger.info(f"created: analysis path = {self.analysis_path}")
        X = training_states
        y, _ = input_signal
        if len(X) == 0 and len(y) == 0:
            self.logger.info(f"missing data: training_states shape {X.shape=}, input_signal shape {y.shape=}")
            return 0
        log["X shape"] = X.shape
        log["y shape"] = y.shape

        # Kernel rank
        log['kernel rank'] = int(kernel_rank(X))
        self.logger.info("done: computed kernel rank")

        # ------------------- RC Targer Functional ----------------
        if self.target_function is not None:
            y = self.target_function(y)

        # ------------------- Training phase ----------------------
        c_opt = self.optimize_regularization(X, y, log, cv=10)
        self.logger.info(f"regularization factor: {c_opt}")
        clf = SVC(C=c_opt)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        log["coefficient of determination (R2)"] = clf.score(X, y)
        log["mean absolute error"] = mean_absolute_error(y, y_pred)

        # Memory Capacity
        memory = memory_capacity(X, y, discrete=False, regularization_factor=c_opt)
        self.inplot_memory_capacity(memory)
        self.logger.info("done: memory capacity")

        # ------------------- Test phase ----------------------
        self.statistics(X, y, log)

        num_tests = len(test_input_states_tuples) // 2
        X_tests, y_tests = [], []
        for test_index in range(num_tests):
            X_test = test_input_states_tuples[2*test_index]
            y_test, _ = test_input_states_tuples[2*test_index+1]
            X_tests.append(X_test)
            y_tests.append(y_test)
        self.reconstruction_statistics_with_test(clf, X, y, X_tests, y_tests, log)
        self.logger.info(f"done: reconstruction statistics plot")

        # ------------------- Debug and other statistics ---------
        self.tuning_curve(clf, X, y, log)
        self.logger.info(f"done: tuning plot")
        self.firing_rate_plot(X, y, log)
        self.logger.info(f"done: firing rate plot")

        self.save_log(log)

        return 1

    def save_log(self, log):
        log_json = json.dumps(log, indent=4)
        with open(os.path.join(self.analysis_path, "output.json"), "w") as outfile:
            outfile.write(log_json)

    def reconstruction_statistics_with_test(self, clf, X, y, X_tests, y_tests, log):
        patterns, cc = np.unique(y, return_counts=1)
        log["patterns"] = patterns.tolist()
        log["patterns count"] = cc.tolist()

        # Reconstruction Statistics
        f1s = [ ]
        recalls = []
        precisions = []
        accs = []
        baccs = []

        # Statistics from training
        _, val_X, _,val_y = train_test_split(X, y, test_size=0.33)
        val_label = clf.predict(val_X)
        precision, recall, f1, support = precision_recall_fscore_support(val_y, val_label, average="weighted", zero_division=0)
        acc = accuracy_score(val_y, val_label)
        bacc = balanced_accuracy_score(val_y, val_label)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        accs.append(acc)
        baccs.append(bacc)

        # Statistics from testing
        for test_index, (X_test, y_test) in enumerate(zip(X_tests, y_tests)):
            test_index += 1  # First index is training
            rest = self.rests[test_index]
            test_predicted_label = clf.predict(X_test)

            # Plot: reconstruction confusion matrix for each test cases
            fig, ax = plt.subplots(1,1,figsize=(8,6))
            cmatrix = confusion_matrix(y_test, test_predicted_label, labels=patterns, normalize='true')
            cmatrix_display = ConfusionMatrixDisplay(confusion_matrix=cmatrix)
            cmatrix_display.plot(ax=ax)
            ax.set_title(f"rest: {rest} min")
            save_path = os.path.join(self.analysis_path, f"reconstruction_after_rest_{rest}.png")
            plt.savefig(save_path)
            self.logger.info(f"done: plot reconstruction after reset: {save_path}")
            plt.close()
            
            precision, recall, f1, support = precision_recall_fscore_support(y_test, test_predicted_label, average="weighted", zero_division=0)
            acc = accuracy_score(y_test, test_predicted_label)
            bacc = balanced_accuracy_score(y_test, test_predicted_label)
            f1s.append(f1)
            recalls.append(recall)
            precisions.append(precision)
            accs.append(acc)
            baccs.append(bacc)

        # Plot: f1/recall/precision progression
        plt.figure()
        plt.plot(self.rests, f1s, 'o-', label="f1")
        plt.plot(self.rests, recalls, 'o-', label="recall")
        plt.plot(self.rests, precisions, 'o-', label="precision")
        plt.plot(self.rests, accs, 'o-', label="acc")
        plt.plot(self.rests, baccs, 'o-', label="bacc")
        plt.xlabel("rest (min)")
        plt.ylabel("score (weighted average)")
        plt.ylim([-0.1,1.1])
        plt.legend()
        plt.savefig(os.path.join(self.analysis_path, "reconstruction_test.png"))
        plt.close()
        np.savez(os.path.join(self.analysis_path, "reconstruction_test.npz"), rests=self.rests, f1s=f1s)

        # Plot: F1 Throughout the experiments
        indices = np.arange(X.shape[0])
        scores = []
        for tasks in np.array_split(indices, 100):
            score = f1_score(y[indices[tasks]], clf.predict(X[indices[tasks]]), average='weighted')
            scores.append(score)
        plt.figure()
        plt.plot(scores)
        plt.xlabel("Segment")
        plt.ylabel("F1 Score")
        plt.title("Reconstruction score over time")
        plt.savefig(os.path.join(self.analysis_path, "reconstruction_test_over_time.png"))
        plt.close()
        self.logger.info("done: plot reconstruction test over time")

    def inplot_memory_capacity(self, memory):
        plt.figure()
        plt.plot(memory)
        plt.xlabel("Tau")
        plt.ylabel("Correlation")
        plt.title("Memory Capacity")
        plt.savefig(os.path.join(self.analysis_path, "memory_capacity.png"))
        plt.close('all')
        self.logger.info("done: plot memory capacity")

    def optimize_regularization(self, X, y, log, cv=10):
        # Find Regularization Factor
        scores = []
        best_value = -99999
        best_c = 0 # Selected factor
        reg_range = np.power(10.0, np.linspace(-4,1,8))
        for c in reg_range:
            clf_test = SVC(C=c)
            score = cross_val_score(clf_test, X, y, cv=cv)
            scores.append(score)
            
            # Early stopping criteria
            comp_value = 0.8 * score.mean() + 0.2 * score.max()
            if best_value < comp_value:
                best_value = comp_value
                best_c = c
            if comp_value < best_value -0.05:
                break
        log["regression regularization (alpha)"] = float(best_c)

        # Plot
        plt.figure()
        plt.boxplot(scores, labels=reg_range[:len(scores)])
        plt.xticks(rotation=90)
        plt.xlabel("Regularization Factor")
        plt.ylabel("10-fold cross validation score")
        plt.title("hyper-parameter tuning")
        plt.savefig(os.path.join(self.analysis_path, "regularization_tuning.png"))
        plt.close('all')
        self.logger.info("done: plot regularization tuning")

        return best_c

    def statistics(self, X, y, log):
        # Dataset Construction
        log["X shape"] = X.shape
        log["y shape"] = y.shape
        
        # Plot distribution
        plt.figure()
        plt.hist(y, bins=50)
        plt.xlabel("input")
        plt.ylabel("count")
        plt.title("Input signal distribution")
        plt.close('all')

    def tuning_curve(self, clf, X, y, log):
        patterns = np.unique(y)

        # Tuning curve
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
        self.logger.info("done: plot tuning curve")

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
        self.logger.info(f"  {size=}")
        self.logger.info(f"  {train_scores.mean(axis=1)=}")
        self.logger.info(f"  {train_scores.std(axis=1)=}")
        self.logger.info(f"  {test_scores.mean(axis=1)=}")
        self.logger.info(f"  {test_scores.std(axis=1)=}")
        self.logger.info(f"  {size=}")
        self.logger.info("done: learning curve")

    def firing_rate_plot(self, X, y, log):
        # Firing rate through experiments
        patterns = np.unique(y)
        for pattern in patterns:
            m = y==pattern
            
            fig, axes = plt.subplots(1,1, figsize=(16,8))
            axes.plot(X[m].mean(axis=1))
            axes.set_xlabel("occurance")
            axes.set_ylabel("firing rate")
            fig.suptitle(f"{pattern=}")
            plt.savefig(os.path.join(self.analysis_path, f"mean_firing_rate_per_pattern_{pattern}.png"))
            plt.close()
        self.logger.info("done: firng rate for each patterns")

@dataclass
class DiscreteTemporalRCAnalysisRankOnly(OperatorMixin):
    tag: str = "rc rank analysis"

    def __post_init__(self):
        super().__init__()

    def __call__(self, *states):
        """
        test_input_state_tuples contain remaining tuples as:
        ((test_X1, test_y1), (test_X2, test_y2), ...)
        """
        log = OrderedDict([])

        os.makedirs(self.analysis_path, exist_ok=True)
        self.logger.info(f"created: analysis path = {self.analysis_path}")
        for sidx, state in enumerate(states):
            X = state
            if len(X) == 0:
                self.logger.info(f"missing data: states shape {X.shape=}")
                continue
            log[f"X shape {sidx}"] = X.shape

            # rank
            log[f'rank for state {sidx}'] = int(kernel_rank(X))
        self.logger.info("done: computed kernel rank")

        self.save_log(log)

        return 1

    def save_log(self, log):
        log_json = json.dumps(log, indent=4)
        with open(os.path.join(self.analysis_path, "output.json"), "w") as outfile:
            outfile.write(log_json)
