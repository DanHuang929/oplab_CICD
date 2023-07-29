import logging
from datetime import datetime, timedelta
from typing import Literal, Union

from fastapi import Depends, FastAPI, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from model.ppg_acc_nyha.feature_engineer import \
    df_preprocess as pn_df_preprocess
from model.ppg_acc_nyha.model import prediction as pan_prediction
from model.ppg_acc_nyha.preprocess import main as pan_preprocess
from model.ppg_nyha.model import prediction as pn_prediction
from model.ppg_nyha.preprocess import main as pn_preprocess
from model.preprocess_exceptions import (AbnormalBPM, BadSignalWarning,
                                         NoOutputError)
from model.preprocess_exceptions import AbnormalBPM, BadSignalWarning
from utils import getexception, raise_warning
import pandas as pd




# dataframe = pn_df_preprocess(ppg_acc.file)
# ppg = dataframe.iloc[:19200]['ppg'].to_list()
# acc = dataframe.iloc[:19200]['tri_acc'].to_list()
# x_test = pan_preprocess(ppg=ppg, acc=acc, nyha=nyha)
result = {}
result['result'] = [0,1,0,0]
result = pd.DataFrame(result)
result.to_csv('result.csv', index=False)
# result = pan_prediction(x_test, model_path='./model/ppg_acc_nyha/')