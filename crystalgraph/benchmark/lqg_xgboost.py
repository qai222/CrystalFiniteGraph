import os.path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Integer

from .base import TrainingParams, BenchmarkModel, plot_rfecv, run_bscv, BenchmarkResults
from ..utils import json_dump


class LqgXGBoostParams(TrainingParams):
    """ LQG-XGBoost with feature selected using rfecv and bscv hyperparam opt """

    booster: str = 'gbtree'
    """ booster choice """

    objective: str = 'reg:absoluteerror'
    """ optimization objective """

    n_estimators: int = 200
    """ number of boosting rounds """

    min_feature_selected: int = 30
    """ minimum number of features selected """

    scoring: str = "neg_mean_absolute_error"

    bscv_iter: int = 50


class LqgXGBoostResults(BenchmarkResults):
    selected_features: list[str]

    bscv_data: dict


class LqgXGBoost(BenchmarkModel):
    """ the model for LQG-XGBoost """

    # TODO early stop https://stackoverflow.com/questions/60231559/how-to-set-eval-metrics-for-xgboost-train

    benchmark_model_name: str = 'LQG-XGBoost'

    benchmark_model_params: LqgXGBoostParams

    benchmark_model_results: Optional[LqgXGBoostResults] = None

    def get_reg(self, **kwargs) -> xgb.XGBRegressor:
        reg = xgb.XGBRegressor(
            random_state=self.benchmark_model_params.random_state_model,
            n_estimators=self.benchmark_model_params.n_estimators,
            booster=self.benchmark_model_params.booster,
            objective=self.benchmark_model_params.objective,
            **kwargs,
        )
        return reg

    def train_and_eval(self, verbose=10):
        X, y, X_train, X_test, y_train, y_test = self.load_lqg_xy()

        rfecv = self.select_features(X_train, y_train, verbose=verbose)
        plot_rfecv(rfecv, self.benchmark_model_params.min_feature_selected, self.work_dir)

        assert [*rfecv.feature_names_in_] == X.columns.tolist()
        assert X.shape[1] == len(rfecv.ranking_)
        features_used = []
        for name, rank in sorted(zip(rfecv.feature_names_in_, rfecv.ranking_), key=lambda x: x[1]):
            if rank == 1:
                features_used.append(name)
        X_train_selected = X_train[features_used]

        cv = KFold(5, random_state=self.benchmark_model_params.cv_random_state, shuffle=True)
        reg = self.get_reg()
        search_spaces = {
            'learning_rate': Real(0.01, 1.0, 'log-uniform'),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.1, 1.0, 'uniform'),
            'colsample_bytree': Real(0.1, 1.0, 'uniform'),  # subsample ratio of columns by tree
            'reg_lambda': Real(1e-6, 1000, 'log-uniform'),
            'reg_alpha': Real(1e-6, 1.0, 'log-uniform'),
        }
        opt = BayesSearchCV(
            estimator=reg,
            search_spaces=search_spaces,
            scoring=self.benchmark_model_params.scoring,
            cv=cv,
            n_iter=self.benchmark_model_params.bscv_iter,  # max number of trials
            n_points=1,  # number of hyperparameter sets evaluated at the same time
            n_jobs=-1,  # number of jobs
            verbose=verbose,
            return_train_score=False,
            refit=False,
            optimizer_kwargs={'base_estimator': 'GP'},  # optmizer parameters: we use Gaussian Process (GP)
            random_state=self.benchmark_model_params.bscv_random_state
        )
        overdone_control = DeltaYStopper(delta=0.0001)  # We stop if the gain of the optimization becomes too small
        time_limit_control = DeadlineStopper(total_time=60 * 60 * 1)  # We impose a time limit
        bscv_data = run_bscv(
            opt, X_train_selected, y_train,
            callbacks=[overdone_control, time_limit_control],
        )

        reg_final = self.get_reg(**bscv_data['best_parameters'])
        cv = KFold(5, random_state=self.benchmark_model_params.cv_random_state, shuffle=True)
        final_scores = cross_val_score(
            reg_final, X_train_selected, y_train,
            scoring=self.benchmark_model_params.scoring,
            cv=cv, n_jobs=-1, verbose=verbose
        )
        final_scores = np.absolute(final_scores)  # return MAE if scoring was NMAE

        reg_final.save_model(os.path.join(self.work_dir, "saved_model.json"))

        scorer = get_scorer(self.benchmark_model_params.scoring)
        reg_final.fit(X_train_selected, y_train)
        test_score = scorer(reg_final, X_test[features_used], y_test)
        test_score = np.absolute(test_score)

        self.benchmark_model_results = LqgXGBoostResults(
            test_mae=test_score,
            train_cv_mae=final_scores,
            selected_features=features_used,
            bscv_data=bscv_data,
        )

        y_train_pred = pd.DataFrame(reg_final.predict(X_train_selected))
        y_test_pred = pd.DataFrame(reg_final.predict(X_test[features_used]))

        json_dump(self.model_dump(), os.path.join(self.work_dir, "trained_model.json"))
        X_train.to_csv(os.path.join(self.work_dir, "X_train.csv"), index=False)
        y_train.to_csv(os.path.join(self.work_dir, "y_train.csv"), index=False)
        y_train_pred.to_csv(os.path.join(self.work_dir, "y_train_pred.csv"), index=False)
        X_test.to_csv(os.path.join(self.work_dir, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(self.work_dir, "y_test.csv"), index=False)
        y_test_pred.to_csv(os.path.join(self.work_dir, "y_test_pred.csv"), index=False)

    def select_features(self, X, y, verbose=10):
        reg = self.get_reg()
        cv = KFold(5, random_state=self.benchmark_model_params.cv_random_state, shuffle=True)
        rfecv = RFECV(
            estimator=reg,
            step=1,
            cv=cv,
            scoring=self.benchmark_model_params.scoring,
            min_features_to_select=self.benchmark_model_params.min_feature_selected,
            n_jobs=-1,
            verbose=verbose,
        )
        rfecv.fit(X, y)
        return rfecv
