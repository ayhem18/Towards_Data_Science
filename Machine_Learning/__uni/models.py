import os, pickle, re
import pandas as pd, numpy as np

from pathlib import Path
from typing import Union, Optional, Dict, Callable
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, f1_score, accuracy_score, roc_auc_score


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


def _extract_model_name_from_cls(model: BaseEstimator):
    cls_str = str(type(model))
    # sub any non-alpha numerical characters at the start of the string
    cls_str = re.sub("^[^a-zA-Z]+", "", cls_str)
    # sub any non-alpha numerical charcs at the end of the string
    cls_str = re.sub("[^a-zA-Z]+$", "", cls_str)
    return cls_str.split(".")[-1]


def evaluate_reg_model(
                    x_test: pd.DataFrame | np.ndarray, 
                    y_test: pd.DataFrame | np.ndarray,
                    model: Optional[BaseEstimator]=None, 
                    y_pred: Optional[pd.DataFrame | np.ndarray]=None) -> Dict[str, float]:

    if model is None and y_pred is None:
        raise TypeError(f"At least one of the arguments 'model', 'y_pred' must be passed")

    if model is not None:
        y_pred = model.predict(x_test)

    # calculate the mse, mae and r^2 scores
    res = {"rmse": root_mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)}

    # round the values to '4' decimals
    for k, v in res.items():
        res[k] = round(v, 4)
    
    return res


def evaluate_classification_model(
                    x_test: pd.DataFrame | np.ndarray, 
                    y_test: pd.DataFrame | np.ndarray,
                    model: Optional[BaseEstimator]=None, 
                    y_pred: Optional[pd.DataFrame | np.ndarray]=None) -> Dict[str, float]:
    if model is None and y_pred is None:
        raise TypeError(f"At least one of the arguments 'model', 'y_pred' must be passed")


    if model is not None:
        y_pred = model.predict(x_test)

    # calculate the mse, mae and r^2 scores
    res = {"accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred)
            }

    # round the values to '4' decimals
    for k, v in res.items():
        res[k] = round(v, 4)
    
    return res


def pipeline(X_train: pd.DataFrame | np.ndarray, 
            y_train: pd.DataFrame | np.ndarray, 
            X_test: pd.DataFrame | np.ndarray,
            y_test: pd.DataFrame | np.ndarray,
            model:BaseEstimator,
            task:str,
            save_model_path: Optional[Union[str, Path]]=None,
            ):

    if task not in ['regression', 'classification']:
        raise NotImplementedError(f"The function expectes the task argument to be either 'regresion' or 'classification'. Found: {task}")

    
    # create a pipeline with a standard scaler and the model
    pip = make_pipeline(model)

    # fit 
    pip.fit(X_train, y_train)

    # extract the training metrics
    model_name, _ = pip.steps[-1]
    
    eval_funcion = evaluate_reg_model if task == 'regression' else evaluate_classification_model

    train_metrics = eval_funcion(X_train, y_train, model=model)    
    
    items = list(train_metrics.items())
    for k, v in items:
        train_metrics[f"train_{k}"] = v
        del(train_metrics[k])
    

    test_metrics = eval_funcion(X_test, y_test, model=model)    
    items = list(test_metrics.items())
    for k, v in items:
        test_metrics[f"test_{k}"] = v
        del(test_metrics[k])

    # save everything
    # :the model
    if save_model_path is None:
        save_dir = os.path.join(DATA_FOLDER, 'models', task, model_name, )        
        os.makedirs(save_dir, exist_ok=True)
        save_model_path = os.path.join(save_dir, f'{model_name}.ob')


    train_metrics_save_path, test_metrics_save_path = (os.path.join(Path(save_model_path).parent, f'{model_name}_train_metrics.ob'), 
                                                        os.path.join(Path(save_model_path).parent, f'{model_name}_test_metrics.ob'))

    for obj, path in [(model, save_model_path), (train_metrics, train_metrics_save_path), (test_metrics, test_metrics_save_path)]:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    return model, train_metrics, test_metrics


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune(X_train: pd.DataFrame | np.ndarray, 
        y_train: pd.DataFrame | np.ndarray,
        model: BaseEstimator, 
        param_grid: Dict,
        score: str | Callable,
        strategy: str
        ) -> BaseEstimator:

    if strategy == 'grid':
        search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, scoring=score)
    else:
        search = RandomizedSearchCV(estimator=model, 
                                    param_distributions=param_grid, 
                                    n_jobs=-1, 
                                    cv=5, 
                                    scoring=score, 
                                    n_iter=10)

    cv = search.fit(X_train, y_train)    

    return cv.best_estimator_
    
