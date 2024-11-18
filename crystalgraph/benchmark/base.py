import glob
import os.path
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split

np.float = float
np.int = int
np.object = object
np.bool = bool
from skopt import BayesSearchCV


class TrainingParams(BaseModel):
    """ parameters used in training """

    dataset_split_random_state: int = 42
    """ random state used in train-test split """

    cv_random_state: int = 43
    """ random state used in cross-validation """

    random_state_model: int = 44
    """ the random state for the model """

    bscv_random_state: int = 45
    """ random state used in bs cross-validation """


class BenchmarkDataset(BaseModel):
    """ dataset used in benchmarks """

    dataset_name: str

    dataset_path: str

    structure_extension: str

    structure_folder_name: str

    lqg_feat_csv: str

    target_csv: str

    target_name: str

    @property
    def structure_filepaths(self) -> list[str]:
        pattern = os.path.join(self.dataset_path, self.structure_folder_name, f"*.{self.structure_extension}")
        return sorted(glob.glob(pattern))


class BenchmarkResults(BaseModel):
    test_mae: float

    train_cv_mae: list[float]


class BenchmarkModel(BaseModel):
    dataset: BenchmarkDataset

    benchmark_model_name: str

    benchmark_model_params: TrainingParams

    benchmark_model_results: BenchmarkResults

    work_dir: str

    def model_post_init(self, __context: Any) -> None:
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)

    def load_lqg_xy(self):
        df_feat_path = os.path.join(self.dataset.dataset_path, self.dataset.lqg_feat_csv)
        df_feat = pd.read_csv(df_feat_path)
        df_target_path = os.path.join(self.dataset.dataset_path, self.dataset.target_csv)
        df_target = pd.read_csv(df_target_path)

        # indexed by structure identifier
        assert [i for i in df_feat.index] == [j for j in df_target.index]

        # first column is structure identifier
        X = df_feat.iloc[:, 1:]
        y = df_target[self.dataset.target_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=self.benchmark_model_params.dataset_split_random_state)
        return X, y, X_train, X_test, y_train, y_test


def plot_rfecv(rfecv: RFECV, min_features_to_select: int, work_dir: str):
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("Mean test accuracy")
    ax.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    ax.set_title("Recursive Feature Elimination \nwith correlated features")
    fig.savefig(f"{work_dir}/feature_selection.png")


def run_bscv(optimizer: BayesSearchCV, X, y, callbacks=None, **kwargs):
    start = time.perf_counter()

    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks, **kwargs)
    else:
        optimizer.fit(X, y, **kwargs)

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    bscv_result = {
        "best_parameters": best_params,
        "best_score": best_score,
        "best_score_std": best_score_std,
        "time_cost": time.perf_counter() - start,
    }
    return bscv_result
