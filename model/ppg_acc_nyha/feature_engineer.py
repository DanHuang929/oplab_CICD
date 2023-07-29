
import antropy as ent
import heartpy as hp
import numpy as np
import pandas as pd
import pywt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

from .utils import *

col_name = ['datetime', 'ppg_ts', 'ppg', 'acc_ts', 'x', 'y', 'z']
MA_WINDOW_SIZE = 3
SAMPLE_RATE = 64.0
AVG_WINDOW_SIZE = 2
LOWER_BOUND = 0
UPPER_BOUND = 1
TIME_FORMAT = '%Y%m%d%H%M%S'



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
        ab = [Pythagorean(obj) for obj in trixial]
        ppg_df['tri_acc'] = np.array(ab)
        return ppg_df

    except Exception as e:
        print(e)
        raise RuntimeError


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
    entropy.append(ent.perm_entropy(data, normalize=True))
    # Spectral entropy
    entropy.append(ent.spectral_entropy(
        data, sf=100, method='welch', normalize=True))
    # Singular value decomposition entropy
    entropy.append(ent.svd_entropy(data, normalize=True))
    # Approximate entropy
    entropy.append(ent.app_entropy(data))
    # Sample entropy
    entropy.append(ent.sample_entropy(data))
    # Hjorth mobility and complexity
    hjorth = ent.hjorth_params(data)
    entropy.append(hjorth[0])
    entropy.append(hjorth[1])
    # Number of zero-crossings
    entropy.append(ent.num_zerocross(data))
    # Lempel-Ziv complexity
    entropy.append(ent.lziv_complexity('01111000011001', normalize=True))

    # Petrosian fractal dimension
    entropy.append(ent.petrosian_fd(data))
    # Katz fractal dimension
    entropy.append(ent.katz_fd(data))
    # Higuchi fractal dimension
    entropy.append(ent.higuchi_fd(data))
    # Detrended fluctuation analysis
    entropy.append(ent.detrended_fluctuation(data))

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


if __name__ == "__main__":
    print('main')
