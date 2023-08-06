import os
import tempfile
import warnings
from utils import *
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from datetime import datetime

warnings.filterwarnings('ignore')


def get_experiment_id_from_name(experiment_name):
    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id

def log_run(gridsearch: GridSearchCV, experiment_name: str, model_name: str, run_index: int, conda_env, tags={}, exp_type="", cust_params={}):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        run_index (int): Index of the run (in Gridsearch)
        conda_env (str): A dictionary that describes the conda environment (MLFlow Format)
        tags (dict): Dictionary of extra data and tags (usually features)
    """

    cv_results = gridsearch.cv_results_
    with mlflow.start_run(run_name=str(run_index)) as run:
        mlflow.log_param("folds", gridsearch.cv)
        for k, v in cust_params:
            mlflow.log_param(k, v)

        print("Logging parameters")
        params = list(gridsearch.param_grid.keys())
        for param in params:
            mlflow.log_param(param, cv_results["param_%s" % param][run_index])
            
        print("Logging metrics")
        for score_name in [score for score in cv_results if "mean_test" in score]:
            mlflow.log_metric(score_name, cv_results[score_name][run_index])
            mlflow.log_metric(score_name.replace(
                "mean", "std"), cv_results[score_name.replace("mean", "std")][run_index])
            mlflow.log_metric(
                score_name.replace("mean", "rank"),
                cv_results[score_name.replace("mean", "rank")][run_index]
            )

        print("Logging model")
        mlflow.sklearn.log_model(
            gridsearch.best_estimator_, model_name, conda_env=conda_env)

        print("Logging CV results matrix")
        tempdir = tempfile.TemporaryDirectory().name
        os.mkdir(tempdir)
        timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")
        filename = "%s-%s-cv_results.csv" % (model_name, timestamp)
        csv = os.path.join(tempdir, filename)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.DataFrame(cv_results).to_csv(csv, index=False)

        mlflow.log_artifact(csv, "cv_results")

        print("Logging extra data related to the experiment")
        mlflow.set_tags(tags)

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        print(mlflow.get_artifact_uri())
        print("runID: %s" % run_id)
        # mlflow.end_run() # prevent additinal run creation

        return run



def log_results(gridsearch: GridSearchCV, experiment_name, model_name, tags={}, log_only_best=False, complete_log=False, description="", cust_params={}):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        tags (dict): Dictionary of extra tags
        log_only_best (bool): Whether to log only the best model in the gridsearch or all the other models as well
    """
    conda_env = {
        'name': 'mlflow-env',
        'channels': ['defaults'],
        'dependencies': [
                'python>=3.7.0',
                'scikit-learn>=0.21.3',
                {'pip': ['xgboost>=1.0.1']}
        ]
    }

    best = gridsearch.best_index_

    mlflow.set_experiment(experiment_name)

    if(log_only_best):
        log_run(gridsearch, experiment_name, model_name, best, conda_env, tags, cust_params)

    # create a parent run for and record all cross fold result in as child
    elif complete_log:
        fold_num = gridsearch.n_splits_
        RAW_DATA_FOLDER = "/home/bobo/Desktop/bobo/P4_FTP_Data/bobo/hp_all/clean_dataframe"
        patient_id_list = get_patient_id_list(RAW_DATA_FOLDER)
        tags = tags | {"child": False, "split_num": fold_num}
        # iterate each search
        for run_idx in range(len(gridsearch.cv_results_['params'])):
            parent_run = log_run(gridsearch, experiment_name,
                                 model_name, run_idx, conda_env, tags, cust_params)
            parent_id = parent_run.info.run_id

            with mlflow.start_run(run_id=parent_id, run_name=str(run_idx)):
                # iterate each split and log as child run
                if cust_params:
                    for k, v in cust_params.items():
                        mlflow.log_param(k, v)
                for child_idx in range(fold_num):
                    print(f"Logging split {child_idx} metrics")
                    cv_results = gridsearch.cv_results_
                    
                    # run a nested child run to log cross fold result
                    with mlflow.start_run(run_name=f"child_{run_idx}_{child_idx}", nested=True) as run:
                        for score_name in [score for score in cv_results if f"split{str(child_idx)}_test" in score]:
                            mlflow.log_metric(
                                score_name.replace(f"split{str(child_idx)}_", ""), cv_results[score_name][run_idx])
                            temp_tags = tags | {"child": True, "split_num": child_idx, "parentRunId": parent_id}
                            mlflow.set_tags(temp_tags)
                            mlflow.log_param("pid", 'TVGH%03d' % patient_id_list[child_idx])
                            mlflow.log_param("child_idx", child_idx)
                            if cust_params:
                                for k, v in cust_params.items():
                                    mlflow.log_param(k, v)
                        for score_name in [score for score in cv_results if "metric" in score]:
                            mlflow.log_metric(score_name, cv_results[score_name][run_idx])
                            
    else:                   
        for i in range(len(gridsearch.cv_results_['params'])):
            log_run(gridsearch, experiment_name,
                    model_name, i, conda_env, tags, cust_params)

    
def temp_get_estimator_by_abs_model_path(logged_model, model_name):
    # # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(
        logged_model, dst_path="/home/bobo/Desktop/bobo/P4_FTP_Data/bobo/mlflow-artifact-temp")

    loaded_model = pickle.load(open(
        f"/home/bobo/Desktop/bobo/P4_FTP_Data/bobo/mlflow-artifact-temp/{model_name}/model.pkl", 'rb'))
    return loaded_model
