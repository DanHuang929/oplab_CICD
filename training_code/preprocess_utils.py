import logging
from utils import LOGGING_CONFIG
import pandas as pd
import numpy as np
from scipy import stats
import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def drop_too_many_na(alldf, na_amount=5):
    # data drop too many na
    return alldf[(alldf.isna().sum(axis=1) < na_amount)]


def simple_imputor(all_df, imp_float, imp_category):
    # data impute
    data_impute = all_df 
    print("shape before impute: ", data_impute.shape)
    data_float = data_impute.select_dtypes(include=[float, int])
    data_category = data_impute.select_dtypes(exclude=[float, int])

    #remove rows with any values that are not finite
    data_float = data_float[np.isfinite(data_float).all(1)]
    
    # print("try to fit float")
    imp_float.fit(data_float)
    # print("try to fit category")
    imp_category.fit(data_category)

    # print("data trans")
    data_float_trans = pd.DataFrame(imp_float.transform(
        data_float), columns=data_float.columns)
    data_category_trans = pd.DataFrame(imp_category.transform(
        data_category), columns=data_category.columns)

    # print("data concat")
    data_simple_impute = pd.concat([data_float_trans, data_category_trans], axis=1)
    print("shape after impute: ", data_simple_impute.shape)
    
    # data_simple_impute.isna().sum(axis=1).value_counts().sort_index().plot(kind="bar", use_index=True,
    #                                                                     rot=45, title="missing values of each row", xlabel="missingvalue", ylabel="count", figsize=(7, 4))

    return data_simple_impute


def change_type(data, column:list, type:str):
    # change type
    try:
        data[column] = data[column].astype(type)
    except Exception as e:
        print(f"change type error: {e}")
        raise ValueError
    return data


def agg_data(data, col_startwith:str):

    # agg acc
    acc_data = data.loc[:, data.columns.str.startswith(col_startwith)]
    acc_data.T.max().head(), acc_data[(np.abs(stats.zscore(acc_data)) < 4).all(axis=1)].T.max().head()

    min_max_scaler = MinMaxScaler()
    data[['acc_q01', 'acc_q05', 'acc_q10', 'acc_q50', 'acc_q90', 'acc_q95', 'acc_q99']] = min_max_scaler.fit_transform( acc_data.T.quantile([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).T )

    data['acc_max'] = acc_data.max(axis=1)
    data['acc_min'] = acc_data.min(axis=1)
    data['acc_mean'] = acc_data.mean(axis=1)
    return data


def separate_df_col(alldf, origin_col, separate_col):
    for n, col in enumerate(separate_col):
        alldf[col] = alldf[origin_col].apply(lambda d: d[n])
    alldf = alldf.drop([origin_col], axis=1)
    return alldf

def change_logging_config(EXP_NAME):

    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # logging.basicConfig(
    #     format=FORMAT, filename="/home/bobo/Desktop/bobo/P4_FTP_Data/bobo/log_"+EXP_NAME+".txt", level=0)
    LOGGING_CONFIG['handlers']['default']['filename'] = "/home/bobo/Desktop/bobo/P4_FTP_Data/bobo/logs/log_"+EXP_NAME+".txt"
    logging.config.dictConfig(LOGGING_CONFIG)