import argparse
from loguru import logger
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, make_scorer, matthews_corrcoef, mean_absolute_percentage_error)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from copy import deepcopy
import mlflow
from mlflow_utils import log_results
from utils import *

warnings.filterwarnings('ignore')

RANDOM_SEED = 40
np.random.seed(RANDOM_SEED)

mlflow.set_tracking_uri("http://140.112.106.237:16894")

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity

def main(args):

    model_class = models[args.model_name]
    model_params = model_parameters[args.model_name]

    cust_params = {'nyha_type' : 'no', 'patient': 'all_patient_optimal_denoise', "random_seed":RANDOM_SEED} | vars(args)
    tags = {'nyha_type' : 'no', "random_seed":RANDOM_SEED} | vars(args)

    print(cust_params)
    print(tags)
    # define feature names
    feature_col_map = {
        # 'ppg_time': ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad',
        #              'sd1', 'sd2', 's', 'sd1/sd2'],
        # 'ppg_freq': ['vlf', 'lf', 'hf', 'lf/hf', 'p_total',
        #              'vlf_perc', 'lf_perc', 'hf_perc', 'lf_nu', 'hf_nu'],
        'acc_quantile': ['acc_q01', 'acc_q05', 'acc_q10',
                         'acc_q50', 'acc_q90', 'acc_q95', 'acc_q99'],
        'acc_agg': ['acc_max', 'acc_min', 'acc_mean'],
        # 'breath': ['breathingrate'],
        # 'ppg_entropy': ["en_ppg_perm", "en_ppg_spectral", "en_ppg_svd", "en_ppg_app", "en_ppg_sample", 'en_ppg_hjorth_l', 'en_ppg_hjorth_h',
        #                 "en_ppg_zerocross", "en_ppg_lziv", "en_ppg_petrosian", "en_ppg_katz", "en_ppg_higuchi", "en_ppg_detrend"],
        # 'acc_entropy': ["en_acc_perm", "en_acc_spectral", "en_acc_svd", "en_acc_app", "en_acc_sample", 'en_acc_hjorth_l', 'en_acc_hjorth_h',
        #                 "en_acc_zerocross", "en_acc_lziv", "en_acc_petrosian", "en_acc_katz", "en_acc_higuchi", "en_acc_detrend"],
                        
        # 'nyha': ['NYHA']
    }

    # read data
    ALL_FOLDER = "./hp_all/all_segments_with_optimal_denoise/" + args.method
    data = pd.read_hdf(os.path.join(ALL_FOLDER, "all_optimal_denoise.h5"), key='data')
    
    col = get_all_dict_values_into_list(feature_col_map)
    print(col)

    if args.undersample:
        # Sample 3000 rows for class 0
        df_class0 = data[data['Event label'] == 0].sample(n=args.undersample, random_state=RANDOM_SEED, replace=False)

        # Sample 3000 rows for class 1
        df_class1 = data[data['Event label'] == 1].sample(n=args.undersample, random_state=RANDOM_SEED, replace=False)
        
        sample_df = pd.concat([df_class0, df_class1], axis=0, ignore_index=True)

        # Reset the index of the sampled DataFrame
        sample_df.reset_index(drop=True, inplace=True)
        data = sample_df

    X = data[col]
    Y = data['Event label']

    if args.balanced:
        rus = RandomUnderSampler(random_state=RANDOM_SEED)
        X, Y = rus.fit_resample(X, Y)
    

    
    # parameter grid for XGBoost
    param_grid = {
        
    }

    # define pipeline
    skf = StratifiedKFold(n_splits=args.cv_fold, shuffle=True, random_state=RANDOM_SEED)
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", model_class(**model_params))
    ])

    # define gridsearch
    # accuracy, specificity, precision, recall, $F_{\beta}$-score, Area Under the Curve (AUC) of a Receiver Operating Characteristic Curve, Geometric-mean (GM), and Matthews correlation coefficient (MCC)
    scoring = {
        "accuracy_score": 'accuracy',
        "specificity_score": make_scorer(specificity_score),
        "precision_score": 'precision',
        "recall_score": 'recall',
        "f1_score": 'f1',
        "roc_auc_score": 'roc_auc',
        "gm_score": make_scorer(geometric_mean_score),
        "mcc": make_scorer(matthews_corrcoef)
    }
    search = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring,
                          refit="roc_auc_score", n_jobs=4, cv=skf.split(X, Y), verbose=10)
    # search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=param_comb,
    #                                    scoring=scoring, refit="roc_auc_score", n_jobs=2, cv=skf.split(X, Y), verbose=0, random_state=1)

    # fit grid search
    search.fit(X, Y)

    # update mlflow details
    tags.update({'features': col})
    tags.update({'feature type': feature_col_map.keys()})
    tags.update({'x_columns': X.columns.tolist()})
    tags.update({'y_columns': Y.name})

    cust_params.update({'feature type': list(feature_col_map.keys())})
    
    cust_params.update({'neg event': Y.shape[0] - Y.sum()})
    cust_params.update({'pos event': Y.sum()})
    for feat_type in ['ppg_time', 'ppg_freq', 'acc_quantile', 'acc_agg', 'breath', 'ppg_entropy', 'acc_entropy', 'nyha']:
        if feat_type in feature_col_map.keys():
            cust_params.update({feat_type: 'yes'})
        else :
            cust_params.update({feat_type: 'no'})


    # log result to mlflow
    log_results(search, experiment_name=args.exp_name, model_name=args.model_name, tags=tags,
                complete_log=True, description=args.description, cust_params=cust_params)


if __name__ == '__main__':
    
    test = False
    UNDER_SAMPLE = 33000 # int, means the number of samples for each class, if zero, don't balance the data
    BALANCED = True
    EXP_NAME = "all_data_different_models_on_optimal_denoise"
    # EXP_NAME = "long_bp_0.7-3.5_ma_30"
    # EXP_NAME = "long_exception_bp_0.7-3.5_BL_Notch_ma_2"

    # argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='experiment name', type=str, default=EXP_NAME)
    parser.add_argument('--model_name', help='model name', type=str, default='lgbm')
    parser.add_argument('--description', help="description string", type=str, default="try different models on the optimal denoise methods, with all patient optimal denoise data")
    parser.add_argument('--cv_fold', help='cv fold', type=int, default=10)
    parser.add_argument("--undersample", help="whether to under sample the data", default=UNDER_SAMPLE, type=int)
    parser.add_argument("--balanced", help="whether to balanced the data", default=BALANCED, type=bool)
    
    args = parser.parse_args()

    models = {
        "lgbm": LGBMClassifier,
        # "knn": KNeighborsClassifier,
        # "xgb": XGBClassifier,
        # "rf": RandomForestClassifier,
        # "lr": LogisticRegression,
    }

    model_parameters = {
        "lgbm": {"n_jobs": 1, "verbosity": 0},
        "knn": {},
        "xgb": {"n_jobs": 1, "verbosity": 0, "use_label_encoder": False},
        "rf": {"n_jobs": 1},
        "lr": {"n_jobs": 1},
        "svm": {"probability": True},
        
    }

    methods = [
        "de_BP",
        "de_MA_BP_ST",
        "de_BP_ST_ADA_WT",
        # "de_MA_BP_ST_ADA_WT",
        # "de_WT_NF_BP",
        # "de_MA_BP_WT"
    ]

    list_of_args = []

    if test:
        args.cv_fold = 2
        methods = ["de_BP"]
        
    else:
        pass
    

    
    for model_name in models.keys():
        for method in methods:
            new_arg = deepcopy(args)
            new_arg.method = method
            new_arg.model_name = model_name
            list_of_args.append(new_arg)
    
    print(models.keys())
    print(len(list_of_args))
    
    # for args in list_of_args:
    #     print(args)
    for arg in list_of_args:
        main(arg)
