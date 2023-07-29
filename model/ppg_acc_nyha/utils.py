from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d
import os
import pandas as pd

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]
    
def normalize(arr: np.ndarray):
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr


def scale_data(data, lower=0, upper=1024):
    rng = np.max(data) - np.min(data)
    minimum = np.min(data)
    data = (upper - lower) * ((data - minimum) / rng) + lower
    return data


def window_scaling(data, sample_rate, windowsize=2.5, lower=0, upper=1024):
    window_dimension = int(windowsize * sample_rate)

    data_start = 0
    data_end = window_dimension

    output = np.empty(len(data))

    while data_end <= len(data):
        sliced = data[data_start:data_end]
        sliced = np.power(sliced, 2)
        scaled = scale_data(sliced, lower, upper)

        output[data_start:data_end] = scaled
        data_start += window_dimension
        data_end += window_dimension

    return np.array(output[0:data_start])
    

def Pythagorean(*args):
    s = 0
    for x in args:
        for i in x:
            s += i**2
    return s ** (1/2)


def upsample(arr: np.ndarray, length: int):
    x = np.arange(arr.shape[0])
    f = interp1d(x, arr, bounds_error=False, fill_value="extrapolate")
    xnew = np.arange(0, arr.shape[0], arr.shape[0]/length)
    return f(xnew)

def try_mkdir(foler_name):
    try:
        os.mkdir(foler_name)
    except Exception as e:
        pass

def ppg_to_csv(data: np.ndarray, path, idx):
    output_path = os.path.join(path, f'{idx}.csv')
    np.savetxt(output_path, data, delimiter=",")