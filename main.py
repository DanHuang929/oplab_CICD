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

logger = logging.getLogger(__name__)

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "6ee6352efdf30038cc492cf67ca1248e9524c4ff4b61453f77774115ea6ad6d0"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$u0esEtqrBSVTZaX7RNCVPOXlIxGulM3gfojyx10E069k8p/YhKuN6",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/v1/ppg/upload/ppg_nyha")
async def ppg_nyha_upload(
    ppg: UploadFile,
    current_user: User = Depends(get_current_active_user),
    nyha: int = 1,
    gender: bool = 0,
    age: int = 20,
    method: Literal['avg', 'max', 'min', 'best'] = 'avg'
):

    try:
        print('start preprocess')
        x_test = pn_preprocess(ppg.file, NYHA=nyha, gender=int(gender), age=age)
    except BadSignalWarning as exc:
        logger.warning("bad signal")
        raise HTTPException(status_code=400, detail="bad signal") from exc
    except AbnormalBPM as exc:
        logger.warning("AbnormalBPM")
        raise HTTPException(status_code=400, detail="abnormal bpm") from exc
    except Exception as exc:
        logger.warning(f'other exceptions: {exc}')
        logger.debug(getexception(exc))
        raise HTTPException(status_code=400, detail=f"other exceptions: {exc}") from exc

    try:
        if x_test is None:
            raise HTTPException(status_code=400, detail="ppg length to short")
        print(f'predict {x_test.shape}')
        result = []
        for _, data in enumerate(x_test):
            try:
                result.append(pn_prediction(data, model_path='./model/ppg_nyha/model.txt'))
            except Exception as exc:
                pass

        warning = raise_warning(result, method)
        return {
            'warning': int(warning),
            'avg': round(sum(result) / len(result), 3),
            'max': round(max(result), 3),
            'min': round(min(result), 3)
        }
    except Exception as exc:
        logger.warning(exc)
        raise HTTPException(status_code=400, detail="Error file-type") from exc


@app.post("/v1/ppg/upload/ppg_acc_nyha")
async def ppg_acc_nyha_upload(
    ppg_acc: UploadFile,
    nyha: int = 1,
    method: Literal['avg', 'max', 'min', 'best'] = 'avg'
):
    try:
        dataframe = pn_df_preprocess(ppg_acc.file)
        ppg = dataframe.iloc[:19200]['ppg'].to_list()
        acc = dataframe.iloc[:19200]['tri_acc'].to_list()
    except Exception as exc:
        logger.error(f"df preproces_error, detail: {getexception(exc)}")
        raise HTTPException(status_code=400, detail="file format not correct") from exc

    try:
        print('start preprocess')
        x_test = pan_preprocess(ppg=ppg, acc=acc, nyha=nyha)
    except BadSignalWarning as exc:
        logger.warning("bad signal")
        raise HTTPException(status_code=400, detail="bad signal") from exc
    except AbnormalBPM as exc:
        logger.warning("AbnormalBPM")
        raise HTTPException(status_code=400, detail="abnormal bpm") from exc
    except Exception as exc:
        logger.warning(f'other exceptions: {exc}')
        logger.debug(getexception(exc))
        raise HTTPException(status_code=400, detail=f"other exceptions: {exc}") from exc

    try:
        if x_test is None:
            raise HTTPException(status_code=400, detail="ppg length to short")

        result = pan_prediction(x_test, model_path='./model/ppg_acc_nyha/')
        
        return {
            'warning': int(result)
        }
        
    except Exception as exc:
        logger.error(f'post processing error, detail: {exc}')
        raise HTTPException(status_code=400, detail=f"post processing error, {exc}") from exc
