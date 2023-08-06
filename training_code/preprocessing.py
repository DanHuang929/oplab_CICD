
import antropy as ent
from utils import *
import os

import logging
import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

col_name = ['datetime', 'ppg_ts', 'ppg', 'acc_ts', 'x', 'y', 'z']
MA_WINDOW_SIZE = 3
SAMPLE_RATE = 64.0
AVG_WINDOW_SIZE = 2
LOWER_BOUND = 0
UPPER_BOUND = 1

RAW_DATA_FOLDER = "../聯發科PPG Raw Data"
IMG_FOLDER = "./hp_csv/long_exception_bp_0.7-3.5_BL_Notch_ma_2"
ALL_FOLDER = "./hp_all/long_exception_bp_0.7-3.5_BL_Notch_ma_2"
TIME_FORMAT = '%Y%m%d%H%M%S'

CSV_HEAD = generate_csv_header(freq=True, acc_entropy=True, ppg_entropy=True)

TIME_RANGE_LIST = [('05:00', '07:00'), ('07:00', '09:00'), ('09:00', '11:00'),
                   ('11:00', '13:00'), ('13:00', '15:00'), ('15:00', '17:00'),
                   ('17:00', '19:00'), ('19:00', '21:00'), ('21:00', '23:00'),
                   ('23:00', '1:00'), ('1:00', '3:00'), ('3:00', '5:00')]
TIME_NAME_LIST = ["morning1", "morning2", "morning3",
                  "afternoon1", "afternoon2", "afternoon3",
                  "evening1", "evening2", "evening3",
                  "night1", "night2", "night3"]


def df_preprocess(fpath):
    ppg_df = pd.read_csv(fpath, names=col_name, header=None)

    #  data preprocess
    try:
        # timestsamp preprocessing
        ppg_df['datetime'] = ppg_df['datetime'].replace(
            to_replace=0, method='bfill')
        ppg_df = ppg_df[ppg_df['datetime'] > 0]
        ppg_df.index = pd.to_datetime(ppg_df['datetime'], format=TIME_FORMAT)

        # interpolate
        ppg_df = interpolate_acc(ppg_df)
        ppg_df = ppg_df.dropna()

        # get trixial
        trixial = ppg_df[['x_new', 'y_new', 'z_new']].to_numpy()
        ab = Pythagorean_3d(ppg_df[['x_new', 'y_new', 'z_new']].to_numpy())
        ppg_df['tri_acc'] = np.array(ab)
        return ppg_df

    except Exception as e:
        logging.error(f"df_preprocess error: {e}, fpath={fpath}")
        


def interpolate_acc(df):
    acc = df[['acc_ts', 'x', 'y', 'z']].copy()
    ppg = df[['datetime', 'ppg_ts', 'ppg']].copy()

    acc.dropna(inplace=True)
    ppg.dropna(inplace=True)

    x = np.arange(acc.shape[0])
    y_x = np.array(acc['x'])
    y_y = np.array(acc['y'])
    y_z = np.array(acc['z'])

    xnew = np.arange(0, acc.shape[0], acc.shape[0]/ppg.shape[0])
    f_x = interp1d(x, y_x, bounds_error=False, fill_value="extrapolate")
    f_y = interp1d(x, y_y, bounds_error=False, fill_value="extrapolate")
    f_z = interp1d(x, y_z, bounds_error=False, fill_value="extrapolate")

    x_xnew, x_ynew, x_znew = f_x(xnew), f_y(xnew), f_z(xnew)

    df['x_new'], df['y_new'], df['z_new'] = [x_xnew, x_ynew, x_znew]
    return df


def calc_baseline(signal):
    # https://github.com/spebern/py-bwr
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True and iterations < 10:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for idx in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]


def baseline_remove(data):
    return data - calc_baseline(data)


def denoising(data):

    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    # Baseline Wander Removal with Wavelet Transform
    # data = baseline_remove(data)

    # Baseline Wander Removal with Notch Filter
    # data = hp.remove_baseline_wander(
    #     data, SAMPLE_RATE, cutoff=0.005)

    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)

    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')

    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=40, bpmmax=180, reject_segmentwise=False,
                                        high_precision=True, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=True, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures


def denoising_BL_Notch_WS_BP(data):

    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    # Baseline Wander Removal with Wavelet Transform
    data = baseline_remove(data)

    # scale the data
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=3, lower=0, upper=1000)

    # Baseline Wander Removal with Notch Filter
    data = hp.remove_baseline_wander(
        data, SAMPLE_RATE, cutoff=0.005)

    # We'll take out frequencies below 0.7Hz (42 BPM) and above 3.5 Hz (210 BPM).
    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')

    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures


def calc_entropy(data, prefix="", dict_like=False):
    head_list = [f"en_{prefix}_perm", f"en_{prefix}_spectral", f"en_{prefix}_svd", f"en_{prefix}_app", f"en_{prefix}_sample", f"en_{prefix}_hjorth_l", f"en_{prefix}_hjorth_h",
                 f"en_{prefix}_zerocross", f"en_{prefix}_lziv", f"en_{prefix}_petrosian", f"en_{prefix}_katz", f"en_{prefix}_higuchi", f"en_{prefix}_detrend"]
    entropy = []
    # Permutation entropy
    try:
        entropy.append(ent.perm_entropy(data, normalize=True))
    except Exception as e:
        print(ent.perm_entropy)
        getexception(e)
    # Spectral entropy
    try:
        entropy.append(ent.spectral_entropy(data, sf=100, method='welch', normalize=True))
    except Exception as e:
        print(ent.spectral_entropy)
        getexception(e)
    # Singular value decomposition entropy
    try:
        entropy.append(ent.svd_entropy(data, normalize=True))
    except Exception as e:
        print(ent.svd_entropy)
        getexception(e)
    # Approximate entropy
    try:
        entropy.append(ent.app_entropy(data))
    except Exception as e:
        print(ent.app_entropy)
        getexception(e)
    # Sample entropy
    try:
        entropy.append(ent.sample_entropy(data))
    except Exception as e:
        print(ent.sample_entropy)
        getexception(e)
    # Hjorth mobility and complexity
    try:
        hjorth = ent.hjorth_params(data)
        entropy.append(hjorth[0])
        entropy.append(hjorth[1])
    except Exception as e:
        print(ent.hjorth_params)
        getexception(e)
    # Number of zero-crossings
    try:
        entropy.append(ent.num_zerocross(data))
    except Exception as e:
        print(ent.num_zerocross)
        getexception(e)
    # Lempel-Ziv complexity
    try:
        entropy.append(ent.lziv_complexity('01111000011001', normalize=True))
    except Exception as e:
        print(ent.lziv_complexity)
        getexception(e)

    # Petrosian fractal dimension
    try:
        entropy.append(ent.petrosian_fd(data))
    except Exception as e:
        print(ent.petrosian_fd)
        getexception(e)
    # Katz fractal dimension
    try:
        entropy.append(ent.katz_fd(data))
    except Exception as e:
        print(ent.katz_fd)
        getexception(e)
    # # Higuchi fractal dimension
    # try:
    #     entropy.append(ent.higuchi_fd(data))
    # except Exception as e:
    #     print(ent.higuchi_fd)
    #     getexception(e)
    # Detrended fluctuation analysis
    try:
        entropy.append(ent.detrended_fluctuation(data))
    except Exception as e:
        print(ent.detrended_fluctuation)
        getexception(e)

    if dict_like:
        return dict(zip(head_list, entropy))
    return entropy


def calc_acc_stats(acc) -> dict:
    min_max_scaler = MinMaxScaler()

    acc_stat_col = ['acc_q01', 'acc_q05', 'acc_q10',
                    'acc_q50', 'acc_q90', 'acc_q95', 'acc_q99']
    q = np.quantile(acc, [0.01, 0.05, 0.1, 0.5,
                    0.9, 0.95, 0.99]).reshape(-1, 1)
    q = min_max_scaler.fit_transform(q).reshape(-1)
    data = dict(zip(acc_stat_col, q))
    data['acc_max'] = np.array(acc).max()
    data['acc_min'] = np.array(acc).min()
    data['acc_mean'] = np.array(acc).mean()

    return data

def ppg_to_csv(data: np.ndarray, path, idx):
    output_path = os.path.join(path, f'{idx}.csv')
    np.savetxt(output_path, data, delimiter=",")


if __name__ == "__main__":
    print('main')