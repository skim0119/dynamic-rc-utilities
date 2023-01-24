import numpy as np
import pickle as pkl

def get_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], np.abs(array[idx] - value)

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
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
        with open(path, 'rb') as f:
            data = pkl.load(f)
    except FileNotFoundError:
        return None
    return data

def moving_average(a, n=3, axis=None) :
    ret = np.cumsum(a, dtype=np.float_, axis=axis)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
