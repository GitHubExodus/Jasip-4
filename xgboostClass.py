import os
import json
import boto3
import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class XGBoostLibrary:

    def __init__(
        self,
        train_file="data/training.parquet",
        validation_file="data/validation.parquet",
        test_file="data/testing.parquet",
        stats_dir="data/xgboost"
    ):
        self.train = pd.read_parquet(train_file)
        self.validation = pd.read_parquet(validation_file)
        self.test = pd.read_parquet(test_file)

        self.stats_dir = stats_dir

        os.makedirs(stats_dir, exist_ok=True)

        self.model = None
        self.params = {}

    def _x(self, data):
        return data.drop(
            columns=["target", "stock"],
            errors="ignore"
        )

    def train_model(self, **params):

        self.params = params

        x_train = self._x(self.train)
        y_train = self.train["target"]

        x_val = self._x(self.validation)
        y_val = self.validation["target"]

        self.model = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            eval_metric="rmse"
        )

        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False
        )

        self._save_stats()

        return self.model

    def _save_stats(self):

        x_test = self._x(self.test)
        y_test = self.test["target"]

        predictions = self.model.predict(x_test)

        stats = pd.DataFrame([{
            "mae": mean_absolute_error(
                y_test,
                predictions
            ),

            "mse": mean_squared_error(
                y_test,
                predictions
            ),

            "rmse": np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            ),

            "r2": r2_score(
                y_test,
                predictions
            ),

            "actual_average_profit":
                y_test.mean(),

            "predicted_average_profit":
                predictions.mean(),

            "actual_total_profit":
                y_test.sum(),

            "predicted_total_profit":
                predictions.sum()
        }])

        stats.to_parquet(
            f"{self.stats_dir}/model_stats.parquet",
            index=False
        )

        booster = self.model.get_booster()

        gain = booster.get_score(
            importance_type="gain"
        )

        weight = booster.get_score(
            importance_type="weight"
        )

        cover = booster.get_score(
            importance_type="cover"
        )

        importance = pd.DataFrame({
            "feature": x_test.columns,

            "gain": [
                gain.get(f, 0)
                for f in x_test.columns
            ],

            "weight": [
                weight.get(f, 0)
                for f in x_test.columns
            ],

            "cover": [
                cover.get(f, 0)
                for f in x_test.columns
            ]
        })

        total_gain = importance["gain"].sum()

        if total_gain > 0:
            importance["gain_percent"] = (
                importance["gain"]
                / total_gain
                * 100
            )
        else:
            importance["gain_percent"] = 0

        importance = importance.sort_values(
            "gain",
            ascending=False
        )

        importance.to_parquet(
            f"{self.stats_dir}/feature_stats.parquet",
            index=False
        )

        with open(
            f"{self.stats_dir}/model_params.json",
            "w"
        ) as f:
            json.dump(
                self.params,
                f,
                indent=4
            )

    def predict(self, data=None):

        if data is None:
            data = self.test

        return self.model.predict(
            self._x(data)
        )

    def predict_profit(self, data=None):

        return self.predict(data)

    def threshold_stats(self):

        predictions = self.predict(self.test)
        actual = self.test["target"].to_numpy()

        thresholds = [
            -2.0,
            -1.0,
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
            5.0
        ]

        rows = []

        for threshold in thresholds:

            selected = predictions >= threshold

            count = selected.sum()

            if count > 0:
                profits = actual[selected]

                rows.append({
                    "threshold": threshold,
                    "trades": count,
                    "total_profit": profits.sum(),
                    "average_profit": profits.mean(),
                    "median_profit": np.median(profits),
                    "win_rate": (
                        (profits > 0).mean() * 100
                    )
                })

            else:
                rows.append({
                    "threshold": threshold,
                    "trades": 0,
                    "total_profit": 0,
                    "average_profit": 0,
                    "median_profit": 0,
                    "win_rate": 0
                })

        result = pd.DataFrame(rows)

        result.to_parquet(
            f"{self.stats_dir}/threshold_stats.parquet",
            index=False
        )

        return result

    def evaluate(self):

        stats = pd.read_parquet(
            f"{self.stats_dir}/model_stats.parquet"
        )

        return stats.iloc[0].to_dict()

    def feature_usefulness(self):

        return pd.read_parquet(
            f"{self.stats_dir}/feature_stats.parquet"
        )

    def save(
        self,
        file="data/xgboost/xgboost_model.json"
    ):

        self.model.save_model(file)

    def load(
        self,
        file="data/xgboost/xgboost_model.json"
    ):

        self.model = xgb.XGBRegressor()

        self.model.load_model(file)

        return self.model

    def upload_all(
        self,
        bucket,
        cloud_folder
    ):

        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"]
        )

        files = [
            "xgboost_model.json",
            "model_stats.parquet",
            "feature_stats.parquet",
            "model_params.json",
            "threshold_stats.parquet"
        ]

        for file in files:

            if file == "xgboost_model.json":
                local_file = (
                    "data/xgboost/"
                    + file
                )
            else:
                local_file = (
                    self.stats_dir
                    + "/"
                    + file
                )

            with open(
                local_file,
                "rb"
            ) as f:

                s3.put_object(
                    Bucket=bucket,
                    Key=f"{cloud_folder}/{file}",
                    Body=f.read()
                )