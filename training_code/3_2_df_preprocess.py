from utils import *
from preprocessing import *
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import loguru

RAW_DATA_FOLDER = "C:/Users/oplab/Desktop/聯發科PPG Raw Data"
CLEAN_DATA_FOLDER = "./hp_all/clean_dataframe"
patient_id_list = get_patient_id_list(RAW_DATA_FOLDER)


def generate_df(pid):
    patient_num = 'TVGH%03d' % pid
    root = os.path.join(RAW_DATA_FOLDER, patient_num)

    filenames = next(os.walk(root), (None, None, []))[2]
    filenames = [f for f in filenames if f.find('.csv') != -1]
    abs_filenames = [os.path.join(
        *[RAW_DATA_FOLDER, patient_num, f]) for f in filenames]

    try_mkdir(os.path.join(*[CLEAN_DATA_FOLDER, patient_num]))
    for fpath in abs_filenames:
        try:
            # print(fpath)
            loguru.logger.info(fpath)
            df = df_preprocess(fpath)
            filename = os.path.basename(fpath).replace(".csv", "")
            new_path = os.path.join(*[CLEAN_DATA_FOLDER, patient_num, f"{filename}.h5"])
            # print(new_path)
            df.to_hdf(new_path, key='data')
        except Exception as e:
            loguru.logger.error(f"{fpath}: {e}")

if __name__ == "__main__":

    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            for out1 in executor.map(generate_df, patient_id_list):
                pass
    except KeyboardInterrupt as e:
        loguru.logger.warning("Keyboard Interrrupt !!!")
    except Exception as e:
        loguru.logger.error(f"Main Exception : {e}")
        getexception(e)
