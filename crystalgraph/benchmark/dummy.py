import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from .base import TrainingParams, BenchmarkModel, BenchmarkResults
from ..utils import json_dump


class DummyModelParams(TrainingParams):
    """ LQG-XGBoost with feature selected using rfecv and bscv hyperparam opt """

    strategy: str = 'mean'

    scoring: str = "neg_mean_absolute_error"


class DummyModel(BenchmarkModel):
    """ dummy baseline """

    benchmark_model_name: str = 'Dummy'

    benchmark_model_params: DummyModelParams

    benchmark_model_results: Optional[BenchmarkResults] = None

    def get_reg(self, **kwargs):
        return DummyRegressor(
            strategy=self.benchmark_model_params.strategy
        )

    def train_and_eval(self, verbose=10):
        X, y, X_train, X_test, y_train, y_test = self.load_lqg_xy()
        cv = KFold(5, random_state=self.benchmark_model_params.cv_random_state, shuffle=True)
        reg = self.get_reg()
        final_scores = cross_val_score(
            reg, X_train, y_train,
            scoring=self.benchmark_model_params.scoring,
            cv=cv, n_jobs=-1, verbose=verbose
        )
        final_scores = np.absolute(final_scores)  # return MAE if scoring was NMAE

        scorer = get_scorer(self.benchmark_model_params.scoring)
        reg.fit(X_train, y_train)
        test_score = scorer(reg, X_test, y_test)
        test_score = np.absolute(test_score)

        self.benchmark_model_results = BenchmarkResults(
            test_mae=test_score,
            train_cv_mae=final_scores,
        )

        y_train_pred = pd.DataFrame(reg.predict(X_train))
        y_test_pred = pd.DataFrame(reg.predict(X_test))

        json_dump(self.model_dump(), os.path.join(self.work_dir, "trained_model.json"))
        X_train.to_csv(os.path.join(self.work_dir, "X_train.csv"), index=False)
        y_train.to_csv(os.path.join(self.work_dir, "y_train.csv"), index=False)
        y_train_pred.to_csv(os.path.join(self.work_dir, "y_train_pred.csv"), index=False)
        X_test.to_csv(os.path.join(self.work_dir, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(self.work_dir, "y_test.csv"), index=False)
        y_test_pred.to_csv(os.path.join(self.work_dir, "y_test_pred.csv"), index=False)
