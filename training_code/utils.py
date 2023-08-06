from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d
import os
import pandas as pd
from scipy import stats, signal
import sys
import traceback

def rm_outlier(data, threshold = 3):
    z_scores = np.abs(stats.zscore(data))  # outlier
    filtered_entries = z_scores < threshold
    data = np.array(data[filtered_entries])
    return data

def smoothing(data, idx):
    smooth_data = pd.Series(data).rolling(window=640).mean()
    return smooth_data

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
    

def timer(start_time=None, title="", display=True, return_str=True):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)

        if display:
            print('\n' + title + ' Time taken: %i hours %i minutes and %s seconds.' %
                (thour, tmin, round(tsec, 2)))

        if return_str:
            return title + f' Time taken: {thour} hours {tmin} minutes and {round(tsec, 2)} seconds.'


def Pythagorean(*args):
    s = 0
    for x in args:
        for i in x:
            s += i**2
    return s ** (1/2)

def Pythagorean_3d(arr):
    """
    ab = Pythagorean_3d(ppg_df[['x_new', 'y_new', 'z_new']].to_numpy())
    ppg_df['tri_acc'] = np.array(ab)
    """
    a,b,c = np.hsplit(arr, 3)
    return np.around(((a**2 + b**2 + c**2) ** (1/2)), 5)

def upsample(arr: np.ndarray, length: int):
    x = np.arange(arr.shape[0])
    f = interp1d(x, arr, bounds_error=False, fill_value="extrapolate")
    xnew = np.arange(0, arr.shape[0], arr.shape[0]/length)
    return f(xnew)


def get_all_dict_values_into_list(feature_map:dict):
    col = []
    for arr in feature_map.values():
        col += arr
    return col

    
def generate_csv_header(freq=False, ppg_entropy=False, acc_entropy=False, dict_format=False):
    range_list = np.arange(0, 19200)
    time_head_list = ["timestamp", "timerange", "timename", "begin", "end"]
    ppg_head_list = ["ppg_"+str(x) for x in range_list]
    acc_head_list = ["acc_"+str(x) for x in range_list]
    # breath_head_list = ["breath_"+str(x) for x in range_list]
    measures_head_list = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20',
                          'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate']
    freq_head_list = ['vlf', 'lf', 'hf', 'lf/hf', 'p_total',
                      'vlf_perc', 'lf_perc', 'hf_perc', 'lf_nu', 'hf_nu']
    ppg_entropy_head_list = ["en_ppg_perm", "en_ppg_spectral", "en_ppg_svd", "en_ppg_app", "en_ppg_sample", 'en_ppg_hjorth_l', 'en_ppg_hjorth_h', "en_ppg_zerocross", "en_ppg_lziv", "en_ppg_petrosian", "en_ppg_katz", "en_ppg_detrend"]
    acc_entropy_head_list = ["en_acc_perm", "en_acc_spectral", "en_acc_svd", "en_acc_app", "en_acc_sample", 'en_acc_hjorth_l', 'en_acc_hjorth_h', "en_acc_zerocross", "en_acc_lziv", "en_acc_petrosian", "en_acc_katz", "en_acc_detrend"]
    labels_head_list = ['patient_id', 'csv num',
                        'Age ', 'Gender', 'NYHA', 'Event label']

    # if no need for each featrues, append nothing to csv_head    
    if not freq:
        freq_head_list = []
    if not ppg_entropy:
        ppg_entropy_head_list = []
    if not acc_entropy:
        acc_entropy_head_list = []
    if dict_format:
        return {
        'time_col': time_head_list,
        'ppg_col': ppg_head_list,
        'acc_col': acc_head_list,
        # 'breath_col': breath_head_list,
        'measures_col': measures_head_list,
        'freq_col': freq_head_list,
        'ppg_entropy_col': ppg_entropy_head_list,
        'acc_entropy_col': acc_entropy_head_list,
        'labels_col': labels_head_list}
    csv_head = time_head_list + ppg_head_list + acc_head_list + measures_head_list + freq_head_list + \
            ppg_entropy_head_list + acc_entropy_head_list + labels_head_list

    return csv_head


def get_patient_data(extend=False):
    # read patient_data.csv
    if extend:
        label_df = pd.read_csv("patient_data_extend.csv", index_col=0)
    else:
        label_df = pd.read_csv("patient_data.csv", index_col=0)
        
    label_df = label_df.rename(columns={"編號id": "num"})
    return label_df

def get_patient_id_list(raw_data_folder):
    # get patient_id_list
    patient_folders = next(os.walk(raw_data_folder), (None, None, []))[1]
    patient_id_list = sorted([int(f[4:]) for f in patient_folders if (
        f.find('TVGH') != -1 and not f.endswith("(x)"))])
    return patient_id_list


def pid_to_path(pid, folder_path):
    patient_num = 'TVGH%03d' % pid
    path = os.path.join(folder_path, patient_num)
    return path 

def get_all_csv_filenames(folder_path) -> list:
    filenames = next(os.walk(folder_path), (None, None, []))[2]
    filenames = [f for f in filenames if f.find('.csv') != -1]
    return filenames

def get_all_h5_filenames(folder_path) -> list:
    filenames = next(os.walk(folder_path), (None, None, []))[2]
    filenames = [f for f in filenames if f.find('.h5') != -1]
    return filenames

def get_all_files_with_filename(folder_path, filename) ->list:
    filenames = next(os.walk(folder_path), (None, None, []))[2]
    filenames = [f for f in filenames if f.find(filename) != -1]
    return filenames

def get_all_patient_data(h5_path_list: list, label_df: pd.DataFrame) -> pd.DataFrame:
    # read each patient's h5 file and concat together
    df_list = []
    for idx, path in enumerate(h5_path_list):
        print(f"-> {idx}: {path}")

        allf_df = pd.read_hdf(path+".h5", key="df")

        labels_list = label_df.iloc[idx][['patient_id', 'csv num',
                                          'Age ', 'Gender', 'NYHA', 'Event label']]
        allf_df[['patient_id', 'csv num', 'Age ',
                'Gender', 'NYHA', 'Event label']] = labels_list

        df_list.append(allf_df)

        try:
            del allf_df
            del labels_list
            gc.collect()
        except Exception as e:
            logging.error(f"get_all_patient_data gc error: {e}")

    alldf = pd.concat(df_list, axis=0, ignore_index=True)
    return alldf


def try_mkdir(foler_name):
    try:
        os.mkdir(foler_name)
    except Exception as e:
        pass

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    # >>> list(split(range(11), 3))
    # [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]

def get_psutil_status(str=False):
    cpu = cpu_percent()
    mem = virtual_memory().percent

    if str:
        return f"cpu: {cpu}, mem: {mem}"
    return cpu, mem

def getexception(e, logger=None):
    error_class = e.__class__.__name__ #取得錯誤類型
    detail = e.args[0] #取得詳細內容
    cl, exc, tb = sys.exc_info() #取得Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
    fileName = lastCallStack[0] #取得發生的檔案名稱
    lineNum = lastCallStack[1] #取得發生的行號
    funcName = lastCallStack[2] #取得發生的函數名稱
    
    errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
    print(errMsg)
    if logger:
        logger.error(f"errMsg: {errMsg}")


# logging config
import gc
import logging
from psutil import cpu_percent, virtual_memory


class PsutilFilter(logging.Filter):
    """psutil logging filter."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add contextual information about the currently used CPU and virtual memory percentages into the given log record."""
        record.psutil = f"c{cpu_percent():02.0f}m{virtual_memory().percent:02.0f}"  # type: ignore
        return True


import logging.config
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'filters': {
        "psutil": {"()": PsutilFilter}
    },
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        "detailed": {
            "format": "%(asctime)s %(levelname)s %(psutil)s %(process)x:%(threadName)s:%(name)s:%(lineno)d:%(funcName)s: %(message)s"
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': '/home/bobo/Desktop/bobo/P4_FTP_Data/bobo/log_test.txt',  # Default is stderr
            'filters': ["psutil"],
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'my.packg': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        '__main__': {  # if __name__ == '__main__'
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}
