# training the data

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting Training and Test Input Data")

            X_train, Y_train, X_test, Y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {

                "Random Forest": RandomForestRegressor(),

                "Decision Tree": DecisionTreeRegressor(),

                "Gradient Boosting": GradientBoostingRegressor(),

                "Linear Regression": LinearRegression(),

                "K-Neighbors Regressor": KNeighborsRegressor(),

                "XGBRegressor": XGBRegressor(),

                "CatBoostRegressor": CatBoostRegressor(verbose=False),

                "AdaBoostRegressor": AdaBoostRegressor()
            }

            param = {

                "Decision Tree": {
                "criterion": ["squared_error", "absolute_error"],
                "splitter": ["best", "random"]
                },

                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [8, 16, 32, 64]
                },

                "Linear Regression": {},

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9]
                },

                "XGBRegressor": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [8, 16, 32, 64]
                },

                "CatBoostRegressor": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1]
                },

                "AdaBoostRegressor": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [8, 16, 32, 64]
                }
            }

            model_report,best_models = evaluate_models(
                X_train,
                Y_train,
                X_test,
                Y_test,
                models,
                param
                )

            best_model_score = max(model_report.values())

            best_model_name = max(model_report, key=model_report.get)

            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found", sys)

            logging.info(f"Best Model Found : {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(Y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)