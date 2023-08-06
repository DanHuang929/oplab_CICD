
from preprocessing import *
import heartpy as hp
from stress_ppg_utils import *
from utils import *
import os
import itertools
import antropy as ent
import heartpy as hp
import numpy as np
import pandas as pd
import pywt
from scipy import signal, sparse, stats
from scipy.fft import fftshift
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve
from stress_ppg_utils import *
from utils import *
from preprocess_utils import change_logging_config
import neurokit2 as nk

def do_nothing(data):

    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=180, reject_segmentwise=False,
                                        high_precision=True, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=True, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures


def de_elgendi(data):
    
    data = nk.ppg_clean(data, sampling_rate=SAMPLE_RATE)
    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=180, reject_segmentwise=False,
                                        high_precision=True, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=True, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures

def de_MA_BP(data):
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
                                        hampel_correct=False, bpmmin=42, bpmmax=180, reject_segmentwise=False,
                                        high_precision=True, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=True, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures

def de_BP_MA(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')
    
    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)

    
    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=180, reject_segmentwise=False,
                                        high_precision=True, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=True, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures



def de_MA(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)

    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=180, reject_segmentwise=False,
                                        high_precision=True, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=True, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures

def de_BP(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')

    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=180, reject_segmentwise=False,
                                        high_precision=True, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=True, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures

def de_WT(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)
    
    # Baseline Wander Removal with Wavelet Transform
    data = baseline_remove(data)

    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures

def de_NF(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)
    
    # Baseline Wander Removal with Notch Filter
    data = hp.remove_baseline_wander(
        data, SAMPLE_RATE, cutoff=0.005)

    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures


def de_SC(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)
    
    # scale the data
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=3, lower=0, upper=1000)

    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures


def de_MA_WT_SC_NF_BP(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)

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



def de_MA_BP_WT(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)

    # We'll take out frequencies below 0.7Hz (42 BPM) and above 3.5 Hz (210 BPM).
    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')
    
    # Baseline Wander Removal with Wavelet Transform
    data = baseline_remove(data)


    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures



def de_BP_MA_WT(data):

    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)
    
    # We'll take out frequencies below 0.7Hz (42 BPM) and above 3.5 Hz (210 BPM).
    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')

    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)

    # Baseline Wander Removal with Wavelet Transform
    data = baseline_remove(data)


    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures


def de_MA_WT_BP(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)

    # Baseline Wander Removal with Wavelet Transform
    data = baseline_remove(data)

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

def de_WT_MA_SC_NF_BP(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    # Baseline Wander Removal with Wavelet Transform
    data = baseline_remove(data)

    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)

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

def de_WT_SC_NF_BP(data):
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

def de_WT_NF_BP(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    # Baseline Wander Removal with Wavelet Transform
    data = baseline_remove(data)


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

def de_WT_BP(data):
    data = hp.preprocessing.scale_data(
        data, lower=LOWER_BOUND, upper=UPPER_BOUND)

    # Baseline Wander Removal with Wavelet Transform
    data = baseline_remove(data)


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




def mix_denoise(data):
    try:
        data, working_data, measures = de_WT_NF_BP(data)
        return data, working_data, measures
    except:
        try:
            data, working_data, measures = de_with_ST_BP_BL(data)
            return data, working_data, measures
        except:
            pass


# def de_with_ST(data):
#     data = hp.preprocessing.scale_data(data, lower=0, upper=1)

#     clean_signal = np.loadtxt('clean_signal.csv', delimiter=',')
#     temp_ths = [1, 2, 1.8, 1.5]
#     cycle = int(19200 / 64 / 120)
#     ths = statistic_threshold(clean_signal, 64, temp_ths)
#     len_before, len_after, time_signal_index = eliminate_noise_in_time(
#         data, 64, ths, cycle)

#     data = np.take(data, time_signal_index, 0)

#     # calculate bpm and other data
#     working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
#                                         calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
#                                         interp_clipping=True, clipping_scale=True, interp_threshold=1020,
#                                         hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
#                                         high_precision=False, high_precision_fs=128.0, breathing_method='welch',
#                                         clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

#     return data, working_data, measures


# def de_with_ST_BP_BL(data):
#     data = hp.preprocessing.scale_data(data, lower=0, upper=1)
#     data = hp.remove_baseline_wander(data, 64, cutoff=0.005)
#     data = hp.filter_signal(
#         data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')

#     # clean_signal = data[1920 * 9 + 1270 : 1920 * 9 + 1510]
#     clean_signal = np.loadtxt('clean_signal.csv', delimiter=',')
#     temp_ths = [1, 2, 1.8, 1.5]
#     cycle = int(19200 / 64 / 120)
#     ths = statistic_threshold(clean_signal, 64, temp_ths)
#     len_before, len_after, time_signal_index = eliminate_noise_in_time(
#         data, 64, ths, cycle)

#     data = np.take(data, time_signal_index, 0)

#     # calculate bpm and other data
#     working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
#                                         calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
#                                         interp_clipping=True, clipping_scale=True, interp_threshold=1020,
#                                         hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
#                                         high_precision=False, high_precision_fs=128.0, breathing_method='welch',
#                                         clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

#     return data, working_data, measures


# def de_with_ST_BP(data):
#     data = hp.preprocessing.scale_data(data, lower=0, upper=1)
#     data = hp.filter_signal(
#         data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')

#     # clean_signal = data[1920 * 9 + 1270 : 1920 * 9 + 1510]
#     clean_signal = np.loadtxt('clean_signal.csv', delimiter=',')
#     temp_ths = [1, 2, 1.8, 1.5]
#     cycle = int(19200 / 64 / 120)
#     ths = statistic_threshold(clean_signal, 64, temp_ths)
#     len_before, len_after, time_signal_index = eliminate_noise_in_time(
#         data, 64, ths, cycle)

#     data = np.take(data, time_signal_index, 0)

#     # calculate bpm and other data
#     working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
#                                         calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
#                                         interp_clipping=True, clipping_scale=True, interp_threshold=1020,
#                                         hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
#                                         high_precision=False, high_precision_fs=128.0, breathing_method='welch',
#                                         clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

#     return data, working_data, measures



clean_signal_org = np.loadtxt('clean_signal.csv', delimiter=',')
def statistic_denoise(data):
    if clean_signal_org is None:
        clean_signal = np.loadtxt('clean_signal.csv', delimiter=',')
    else:
        clean_signal = clean_signal_org.copy()

    temp_ths = [1, 2, 1.8, 1.5] 
    cycle = int(19200 / 64 / 120)
    ths = statistic_threshold(clean_signal, 64, temp_ths)
    len_before, len_after, time_signal_index = eliminate_noise_in_time(data, 64, ths, cycle)

    first_data = np.take(data, time_signal_index, 0)
    return first_data


def de_ST(data):
    data = hp.preprocessing.scale_data(data, lower=0, upper=1)
    
    # Statistical denoising
    data = statistic_denoise(data)

    # calculate bpm and other data
    working_data, measures = hp.process(data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)

    return data, working_data, measures


def de_BP_ST(data):
    data = hp.preprocessing.scale_data(data, lower=0, upper=1)

    # We'll take out frequencies below 0.7Hz (42 BPM) and above 3.5 Hz (210 BPM).
    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')
    
    # Statistical denoising
    first_data = statistic_denoise(data)

    #  calculate bpm and other data
    working_data, measures = hp.process(first_data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                    calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                    interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                    hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                    high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                    clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)
    return data, working_data, measures


def de_MA_BP_ST(data):
    data = hp.preprocessing.scale_data(data, lower=0, upper=1)

    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)
    
    # We'll take out frequencies below 0.7Hz (42 BPM) and above 3.5 Hz (210 BPM).
    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')
    
    # Statistical denoising
    first_data = statistic_denoise(data)

    try:
    #  calculate bpm and other data
        working_data, measures = hp.process(first_data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)
    except:
        return de_MA_BP_WT(data)
    return data, working_data, measures


def de_BP_ST_ADA_WT(data):
    data = hp.preprocessing.scale_data(data, lower=0, upper=1)

    # We'll take out frequencies below 0.7Hz (42 BPM) and above 3.5 Hz (210 BPM).
    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')
    
    # Statistical denoising
    first_data = statistic_denoise(data)

    try:
    #  calculate bpm and other data
        working_data, measures = hp.process(first_data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)
    except:
        return de_MA_BP_WT(data)
    return data, working_data, measures

    
def de_MA_BP_ST_ADA_WT(data):
    data = hp.preprocessing.scale_data(data, lower=0, upper=1)

    # Moving average
    data = window_scaling(
        data, SAMPLE_RATE, windowsize=MA_WINDOW_SIZE, lower=0, upper=1000)

    # We'll take out frequencies below 0.7Hz (42 BPM) and above 3.5 Hz (210 BPM).
    data = hp.filter_signal(
        data, cutoff=[0.7, 3.5], sample_rate=SAMPLE_RATE, order=3, filtertype='bandpass')
    
    # Statistical denoising
    first_data = statistic_denoise(data)

    try:
    #  calculate bpm and other data
        working_data, measures = hp.process(first_data, sample_rate=SAMPLE_RATE, windowsize=AVG_WINDOW_SIZE, report_time=False,
                                        calc_freq=True, freq_method='fft', welch_wsize=240, freq_square=False,
                                        interp_clipping=True, clipping_scale=True, interp_threshold=1020,
                                        hampel_correct=False, bpmmin=42, bpmmax=210, reject_segmentwise=False,
                                        high_precision=False, high_precision_fs=128.0, breathing_method='welch',
                                        clean_rr=False, clean_rr_method='z-score', measures=None, working_data=None)
    except:
        return de_MA_BP_WT(data)
    return data, working_data, measures