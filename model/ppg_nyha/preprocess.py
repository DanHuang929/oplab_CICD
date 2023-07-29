import math

import heartpy as hp
import matplotlib.pyplot as plt
import nolds
import numpy as np
import pandas as pd
from hrvanalysis import (get_frequency_domain_features, get_sampen,
                         get_time_domain_features)
from scipy import signal

from model.preprocess_exceptions import (AbnormalBPM, BadSignalWarning,
                                         ShapeNotCorrect)


def add_time_domain_features(data):
    keep = ['median_nni', 'range_nni', 'cvsd', 'cvnni', 'max_hr', 'min_hr', 'std_hr']
    new_dict = {}
    for i in keep:
        new_dict[i] = data[i]
    return new_dict


def get_feature(data):
    try:
        wd, m = hp.process(data, sample_rate=64)
    except Exception:
        raise BadSignalWarning

    if(len(m) == 0 or math.isnan(m['bpm'])):
        raise BadSignalWarning

    else:
        if(np.array(m['bpm']) > 180 or np.array(m['bpm']) < 40):
            raise AbnormalBPM(m['bpm'])

        for i in m.keys():
            m[i] = float(m[i])

        time_domain_features = get_time_domain_features(np.array(wd['RR_list_cor']))
        other_feature = add_time_domain_features(time_domain_features)
        m.update(other_feature)

        frequency_domain_features = get_frequency_domain_features(np.array(wd['RR_list_cor']))
        m.update(frequency_domain_features)

        sampen = get_sampen(np.array(wd['RR_list_cor']))
        m.update(sampen)
        m['dfa'] = nolds.dfa(np.array(wd['RR_list_cor']))
        m['lyap'] = nolds.lyap_r(np.array(wd['RR_list_cor']), min_tsep=10)

        return(m, 1)


def feature_5_min(ppg_data, NYHA, gender, age):
    all_feature, hr_bool = get_feature(ppg_data)

    if(hr_bool == 0):
        return 0
    else:
        all_feature['NYHA'] = NYHA
        all_feature['gender'] = gender
        all_feature['age'] = age
        df1 = pd.DataFrame([all_feature])
        return df1.values


def main(file, NYHA, gender, age):
    df = pd.read_csv(file, header=None)
    age = 6 if age > 60 else 3
    ppgs = np.array(df[[2]].values.tolist()).T[0]
    x_test = []
    size = 19200
    if ppgs.shape[0] > size:
        for i in range(0, ppgs.shape[0], size):
            try:
                ppg = ppgs[i:i+size]
                b, a = signal.butter(3, [0.5, 8], btype="bandpass", output="ba", fs=64)
                filtered = signal.filtfilt(b, a, ppg, method="gust")[3*64:-3*64]
                x = feature_5_min(filtered, NYHA, gender, age)
                x_test.append(x)
            except Exception:
                pass
    elif ppgs.shape[0] == size:
        ppg = ppgs
        b, a = signal.butter(3, [0.5, 8], btype="bandpass", output="ba", fs=64)
        filtered = signal.filtfilt(b, a, ppg, method="gust")[3*64:-3*64]
        x = feature_5_min(filtered, NYHA, gender, age)
        x_test.append(x)
    else:
        raise ShapeNotCorrect(ppgs.shape[0])
    return np.array(x_test)
