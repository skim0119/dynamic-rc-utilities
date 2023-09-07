import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report

def classification(X, y, patterns, tau=0, test_X=None, alpha=0.1):
    s = y if tau == 0 else y[:-tau]
    X = X[tau:]
    X = np.concatenate([X, X**2], axis=1)
    if test_X is None:
        test_X = X
    else:
        test_X = np.concatenate([test_X, test_X**2], axis=1)
    scores = []

    _score = []
    for pattern in patterns:
        clf = Ridge(alpha=alpha)
        clf.fit(X, (s == pattern).astype(np.float_))
        _score.append(clf.predict(test_X))
    #score = log_softmax(_score, axis=0)
    predicted_label = patterns[np.argmax(_score, axis=0)]
    return predicted_label, s

def reconstruction_score(X, y, patterns, tau=0):
    h, s = classification(X, y, patterns)
    report = classification_report(s, h, output_dict=True, labels=patterns, zero_division=0)
    return report
