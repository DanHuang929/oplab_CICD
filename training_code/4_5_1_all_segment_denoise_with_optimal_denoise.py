import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from preprocess_utils import change_logging_config
from time import sleep, time
import logging
from denoise_function import (
    de_MA,
    de_BP,
    de_SC,
    de_NF,
    de_WT,
    de_elgendi,
    de_BP_MA,
    de_MA_BP,
    de_WT_BP,
    de_WT_NF_BP,
    de_MA_BP_WT,
    de_MA_WT_BP,
    de_WT_SC_NF_BP,
    de_WT_MA_SC_NF_BP,
    de_MA_WT_SC_NF_BP,
    de_ST,
    de_BP_ST,
    de_MA_BP_ST,
    de_BP_ST_ADA_WT,
    de_MA_BP_ST_ADA_WT
)
import itertools
from utils import getexception, try_mkdir, upsample, generate_csv_header, getexception
from preprocessing import calc_entropy
from loguru import logger

import warnings
warnings.filterwarnings("ignore")

logger.remove()
logger.add("./logs/4_4_5.log", rotation="500 MB", compression="zip", level="DEBUG", enqueue=True)
logger.add("./logs/4_4_5-info.log", level="INFO", filter=lambda record: record["extra"].get("is_count", True))
logger.add("./logs/4_4_5-count.log", filter=lambda record: record["extra"].get("is_count", False))



EXP_NAME = "all_segments_with_optimal_denoise"
ALL_FOLDER = "./hp_all/" + EXP_NAME
CLEAN_FOLDER = "./hp_all/clean_dataframe/"
CSV_HEAD = generate_csv_header(freq=True, acc_entropy=True, ppg_entropy=True)
CSV_HEAD = [x for x in CSV_HEAD if x not in ['timerange','timename','begin','end']]


label_df = pd.read_csv("patient_data.csv", index_col=0)
label_df = label_df.rename(columns={"編號id": "num"})
test = False
m = 5
base_folder = "./hp_all/" + EXP_NAME
try_mkdir(base_folder)

def procedure(input):
    
    # init 
    data_list = []    
    all_signal_count, good_signal_count = 0, 0
    
    # parse input
    pid_filename = input[0]
    denoise_function = input[1]
    denoise_name = denoise_function.__name__
    full_pid = pid_filename.split("/")[0]
    pid = pid_filename.split("/")[0][4:7]
    filename = pid_filename.split("/")[1]
    
    # derive path
    denoise_folder = os.path.join(base_folder, denoise_name)
    pid_folder = os.path.join(denoise_folder, full_pid)
    out_filename = os.path.join(pid_folder, filename)
    try_mkdir(denoise_folder)
    try_mkdir(pid_folder)


    logger.info(f"start {filename} with {denoise_name}")
    
    df = pd.read_hdf(os.path.join(*[CLEAN_FOLDER, full_pid, filename]), key='data')
    
    if test:
        df = df.iloc[:192000]
    
    
    for x in range(0, len(df)//64//60, 5):
        print(f"{full_pid} -> {denoise_name}: {x}")
        try:
            sample = df.iloc[x*60*64:(x+m)*60*64]['ppg']
            acc = df.iloc[x*60 * 64:(x+m)*60*64]['tri_acc'].to_list()[:19200]
            if sample.shape[0] != 19200:
                raise "shape not 19200"
        except Exception as e:
            logger.warning(
                f"PID={pid}, DF={denoise_name}, filename : {out_filename}, status: {'Sample error!'}")
            continue
        
        try:
            all_signal_count += 1
            hr, working_data, measures = denoise_function(sample)
            
        except Exception as e:
            logger.debug(
                f"PID={pid}, DF={denoise_name}, filename : {out_filename}, status: {'Denoising error!'}")
            continue
        

        # calc entropy
        try:
            ppg_entropy = calc_entropy(hr)
        except Exception as e:
            print(f"ppg entropy {denoise_name} error: {e}")
            logging.warning(
                f"PID={pid}, DF={denoise_name}, filename : {out_filename}, status: {'PPG Entropy error!'}, detail: {e}")
            continue

        try:
            acc_entropy = calc_entropy(acc)
        except Exception as e:
            print(f"acc entropy {denoise_name} error: {e}")
            logging.warning(
                f"PID={pid}, DF={denoise_name}, filename : {out_filename}, status: {'ACC Entropy error!'}, detail: {e}")
            continue

        good_signal_count += 1

        # parse result
        try:
            abs_time = df.iloc[x*60*64]["datetime"]
            time_list = [abs_time]
            acc_list = df.iloc[x*60 * 64:(x+m)*60*64]['tri_acc'].to_list()[:19200]
            logger.debug(f"Parse: workding_data.keys(): {working_data.keys()}")
            # breath_list = np.zeros(19200)
            measures_list = list(measures.values())
            
            # remove leading zeros in string number,  example: turn 012 to 12, turn 003 to 3
            pid = int(pid)

            labels_list = label_df.loc[pid][['patient_id', 'csv num',
                                                'Age ', 'Gender', 'NYHA', 'Event label']].to_list()

            append_list = time_list + list(hr)[:19200] + acc_list + measures_list + ppg_entropy + acc_entropy + labels_list

        except Exception as e:
            logger.error(f"PID={pid}, DF={denoise_name}, filename: {out_filename}, status: {'Parsing error!'}, detail: {e}")
            logger.error(getexception(e))
            try:
                logger.error(f"PID={pid}, DF={denoise_name}, filename: {out_filename}, status: {'Parsing error!'}, detail: {e}")
                logger.error(f"Lens: {len(time_list), len(list(hr)[:19200]), len(acc_list), len(measures_list), len(labels_list)}")
            except Exception as e2:
                logger.error(f"PID={pid}, DF={denoise_name}, filename: {out_filename}, status: {'Parsing error!'}, detail: {e2}")
                logger.error(f"denoise result: {working_data.keys()}, {measures.keys()}")
            continue

        try:
            logger.debug("data_list append")
            if len(append_list) != len(CSV_HEAD):
                # print(len(CSV_HEAD), len(append_list), len(time_list) , len(list(hr)[:19200]) , len(acc_list), len(breath_list), len(ppg_entropy), len(acc_entropy), len(measures_list) , len(labels_list))
                logger.error(f"PID={pid}, DF={denoise_name}, filename : {out_filename}, status: {'Write size different!'}")
            # print("append_list ", append_list)
            data_list.append(append_list)
        except Exception as e:
            logger.error(f"data_list append error: {e}")
            continue
    
    # print("data_list ", data_list)
    try:
        logger.debug("start write file")
        temp_df = pd.DataFrame(data_list, columns=CSV_HEAD)
        temp_df.to_hdf(out_filename, 'df', mode='w')   
    except Exception as e:
        logger.error(f"write file error: {e}")

    logger.info(f"PID={pid}, signal count: {good_signal_count} / {all_signal_count} @ {denoise_name}, filename : {out_filename},", is_count=True)


    # try:
    #     del sample, hr, temp_df, data_list, acc_list,\
    #         breath_list, measures_list, labels_list
    #     gc.collect()
    # except Exception as e:
    #     logger.warning(f"PID={pid}, DF={denoise_name},status: gc failed, detail:{e}")
    #     pass

def main(PoolExecutor):
    change_logging_config("denoise_exp")

    # define patient folders
    patient_folder = sorted(os.listdir(CLEAN_FOLDER))
    # patient_folder = ["TVGH040"]

    # iterate folders in patient_folder and get all data ends with ".h5" in and append to input_files
    input_files = []
    for folder in patient_folder:
        for file in sorted(os.listdir(os.path.join(CLEAN_FOLDER, folder))):
            if file.endswith(".h5"):
                input_files.append(os.path.join(folder, file))

   

    # define denoise functinos
    denoise_function = [
        # de_MA,
        de_BP,
        # de_SC,
        # de_NF,
        # de_WT,
        # de_elgendi,
        # de_BP_MA,
        # de_MA_BP,
        # de_WT_BP,
        # de_WT_NF_BP,
        # de_MA_BP_WT,
        # de_MA_WT_BP,
        # de_WT_SC_NF_BP,
        # de_WT_MA_SC_NF_BP,
        # de_MA_WT_SC_NF_BP,
        # de_ST,
        # de_BP_ST,
        de_MA_BP_ST,
        de_BP_ST_ADA_WT,
        # de_MA_BP_ST_ADA_WT
    ]
    
     # test flag for input_files
    if test:
        input_files = input_files[:2]
        denoise_function = denoise_function
    # logger.info(input_files, denoise_function)

    # generate input combinations
    input_combinations = itertools.product(input_files, denoise_function)
    exec_start = time()
    # with ProcessPoolExecutor(mp_context=mp.get_context('fork'), max_workers=4) as executor:
    try:
        with ProcessPoolExecutor(max_workers=30) as executor:
            for out1 in executor.map(procedure, input_combinations):
                pass
    except KeyboardInterrupt as e:
        logger.warning("Keyboard Interrrupt !!!")
    except Exception as e:
        logger.error(f"Main Exception : {e}")
        getexception(e)

    exec_finish = time()
    print(f'time : {(exec_finish-exec_start)}')
    logger.info(f'PoolExecutor: {PoolExecutor}')
    logger.info(f'time : {(exec_finish-exec_start)}')

main(ProcessPoolExecutor)

