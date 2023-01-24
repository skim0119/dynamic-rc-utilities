from typing import Optional, Literal

import os, sys
import numpy as np

import itertools

from collections import defaultdict

from sklearn.linear_model import Ridge
from sklearn.svm import SVC
import scipy.stats as spst

from config import *

ALLOWED_CORRELATION_METHOD = Literal["pearson", "spearman"]


def memory_capacity(
    X,
    y,
    tau_max=20,
    residual=True,
    correlation_method: ALLOWED_CORRELATION_METHOD = "pearson",
    keep_monotonic=True,
    discrete=False,
    regularization_factor=0.1,
):
    mc = np.zeros(tau_max)
    mc[:] = np.NaN
    for tau in range(0, tau_max):
        _s = y if tau == 0 else y[:-tau]
        if discrete:  # No normalization for discrete classification
            s = _s
        else:
            s = (_s - np.mean(_s)) / np.linalg.norm(_s)  # Normalize

        if residual:
            data = np.concatenate([X[tau:], y[tau:, None]], axis=1)
        else:
            data = X[tau:]

        if discrete:  # Classification memory capacity
            clf = SVC(C=regularization_factor, kernel='linear')
            patterns = np.unique(y)
            clf.fit(data, s)
            h = clf.predict(data)
        else:  # Continuous memory capacity
            # Output (Continuous)
            clf = Ridge(alpha=regularization_factor)
            clf.fit(data, s)
            h = clf.predict(data)

        if correlation_method == "pearson":
            # Pearson correlation
            corr = np.corrcoef(h, s, rowvar=False)  # Pearson correlation coefficient
            _mc = corr[0, 1] ** 2
        elif correlation_method == "spearman":
            # Spearman correlation
            _mc = spst.spearmanr(h, s).correlation ** 2
        else:
            raise NotImplementedError

        if keep_monotonic and tau > 0 and _mc > mc[tau - 1]:
            break
        else:
            mc[tau] = _mc
    return mc
