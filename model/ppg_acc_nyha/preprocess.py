import gc
import numpy as np
import logging
import warnings
from collections import OrderedDict

from model.preprocess_exceptions import (BadSignalWarning, EntropyError,
                                         ParsingError, ShapeNotCorrect, AccStatsCalcError)

from .feature_engineer import (calc_acc_stats, calc_entropy,
                               denoising_BL_Notch_WS_BP)

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True


def init_data() -> OrderedDict:

    # define col
    ppg_time_col = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad',
                    'sd1', 'sd2', 's', 'sd1/sd2']
    ppg_freq_col = ['vlf', 'lf', 'hf', 'lf/hf', 'p_total',
                    'vlf_perc', 'lf_perc', 'hf_perc', 'lf_nu', 'hf_nu']
    acc_col_col = ['acc_q01', 'acc_q05', 'acc_q10',
                   'acc_q50', 'acc_q90', 'acc_q95', 'acc_q99', 'acc_max', 'acc_min', 'acc_mean']
    breath_col = ['breathingrate']
    entropy_col = ['en_ppg_sample', 'en_ppg_zerocross', 'en_ppg_lziv', 'en_ppg_petrosian',
                   'en_ppg_katz', 'en_ppg_higuchi', 'en_ppg_detrend', 'en_acc_perm',
                   'en_acc_spectral', 'en_acc_svd', 'en_acc_app', 'en_acc_sample',
                   'en_acc_zerocross', 'en_acc_lziv', 'en_acc_petrosian', 'en_acc_katz',
                   'en_acc_higuchi', 'en_acc_detrend', 'en_ppg_hjorth_l', 'en_ppg_hjorth_h',
                   'en_acc_hjorth_l', 'en_acc_hjorth_h']
    nyha_col = ['NYHA']

    col = ppg_time_col + ppg_freq_col + \
        acc_col_col + breath_col + entropy_col + nyha_col
    return OrderedDict((k, None) for k in col)


def main(
    ppg: list,
    acc: list,
    nyha: int,
):
    
    # check shape
    try:
        if len(ppg) != 19200:
            logger.warning("status: {'PPG shape not 19200!'}")
        if len(acc) != 19200:
            logger.warning("status: {'ACC shape not 19200!'}")
    except Exception as exc:
        logger.warning(f"status: {'PPG or ACC data error!'}, detail: {exc}")
        raise ShapeNotCorrect from exc

    # ppg denoise and analysis
    try:
        # hr, working_data, measures = denoising(ppg)
        hr, working_data, measures = denoising_BL_Notch_WS_BP(ppg)
    except Exception as exc:
        logger.warning(f"status: {'Denoising error!'}, detail: {exc}")
        raise BadSignalWarning from exc

    # calc hr entropy
    try:
        ppg_entropy_dict = calc_entropy(hr, prefix="ppg", dict_like=True)
    except Exception as exc:
        print("ppg entropy error: ", exc)
        logger.warning(f"{'PPG Entropy error!'}, detail: {exc}")
        raise EntropyError from exc

    # calc acc entropy
    try:
        acc_entropy_dict = calc_entropy(acc, prefix="acc", dict_like=True)
    except Exception as exc:
        print("acc entropy error: ", exc)
        logger.warning(f"{'ACC Entropy error!'}, detail: {exc}")
        raise EntropyError from exc

    # calc acc statistics
    try:
        acc_stats_dict = calc_acc_stats(acc)
    except Exception as exc:
        print("acc statistics error: ", exc)
        logger.warning(f"{'ACC statistics error!'}, detail: {exc}")
        raise AccStatsCalcError from exc

    # parse result
    try:
        data = init_data()
        data['NYHA'] = nyha
        data.update(measures)
        data.update(ppg_entropy_dict)
        data.update(acc_entropy_dict)
        data.update(acc_stats_dict)

    except Exception as exc:
        logger.warning(
            f"status: {'Result parsing error!'}, data dict: {data}")
        raise ParsingError from exc

    data = {k: v for k, v in data.items() if k in init_data().keys()}
    data =  np.array(list(data.values())).reshape(1, -1)
    print("training data shape: ", data.shape)
    return data
