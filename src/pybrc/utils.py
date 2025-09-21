from typing import Optional
import numpy as np
import pickle as pkl
import math


def get_nearest(array, value, idx_guess: int = 0):
    """
    Assume array is ascending array, get nearest
    """
    array = np.asarray(array)

    # Corner cases
    if value < array[0]:
        return array[0], np.abs(array[0] - value), 0
    elif array[-1] < value:
        return array[-1], np.abs(array[-1] - value), array.shape[0] - 1
    # Binary Search
    left = 0
    right = array.shape[0] - 1
    min_diff = np.abs(array[left] - value)
    mid_index = left
    while left <= right:
        mid = (left + right) // 2
        diff = np.abs(array[mid] - value)
        if diff < min_diff:
            min_diff = diff
            mid_index = mid

        if array[mid] < value:
            left = mid + 1
        elif array[mid] > value:
            right = mid - 1
        else:
            break

    return array[mid_index], np.abs(array[mid_index] - value), mid_index


def rand_jitter(arr):
    stdev = 0.01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def get_data(sample_id, binsize, date):
    event_path = f"{PATTERN_PATH}/{date}_sample_{sample_id}_{binsize}_inputs.npz"
    data_path = f"{RESERVOIR_STATE_PATH}/{date}_sample_{sample_id}_{binsize}_states.npz"

    try:
        event = np.load(event_path)
        data = np.load(data_path)
    except FileNotFoundError:
        return None
    return data["X"], event["data"], event["time"]


def get_spiketrain_data(sample_id, date, spiketrain_path, binsize=None):
    if binsize is None:
        path = f"{spiketrain_path}/{date}_sample_{sample_id}_spiketrain.pkl"
    else:
        path = f"{spiketrain_path}/{date}_sample_{sample_id}_{binsize}_spiketrain.pkl"

    try:
        with open(path, "rb") as f:
            data = pkl.load(f)
    except FileNotFoundError:
        return None
    return data


def moving_average(a, n=3, axis=None):
    ret = np.cumsum(a, dtype=np.float_, axis=axis)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def ema_smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed
