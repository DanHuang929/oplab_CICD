from model.ppg_acc_nyha.feature_engineer import \
    df_preprocess as pn_df_preprocess
from model.ppg_acc_nyha.model import prediction as pan_prediction
from model.ppg_acc_nyha.preprocess import main as pan_preprocess
import pandas as pd
import numpy as np



data_path = "user_data.csv"
dataframe = pn_df_preprocess(data_path)
dataframe.to_csv('user_data_1.csv', index=False)
ppg = dataframe.iloc[:19200]['ppg'].to_list()
acc = dataframe.iloc[:19200]['tri_acc'].to_list()
x_test = pan_preprocess(ppg=ppg, acc=acc, nyha=1)
result = pan_prediction(x_test, model_path='./model/ppg_acc_nyha/')
prediction = {"prediction":[result]}
print(prediction)
result = pd.DataFrame(prediction)
result.to_csv('result.csv', index=False)