import os
import json
import xgboost as xgb
import boto3
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
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

    def train_model(self, **params):
        self.params = params

        x_train = self.train.drop(columns=["target", "stock"], errors="ignore")
        y_train = self.train["target"]

        x_val = self.validation.drop(columns=["target", "stock"], errors="ignore")
        y_val = self.validation["target"]

        self.model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss"
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
        x_test = self.test.drop(columns=["target", "stock"], errors="ignore")
        y_test = self.test["target"]

        predictions = self.model.predict(x_test)
        probabilities = self.model.predict_proba(x_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions
        ).ravel()

        stats = pd.DataFrame([{
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "log_loss": log_loss(y_test, probabilities),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp
        }])

        stats.to_parquet(
            f"{self.stats_dir}/model_stats.parquet",
            index=False
        )

        importance = pd.DataFrame({
            "feature": x_test.columns,
            "gain": [
                self.model.get_booster().get_score(
                    importance_type="gain"
                ).get(f, 0)
                for f in x_test.columns
            ],
            "weight": [
                self.model.get_booster().get_score(
                    importance_type="weight"
                ).get(f, 0)
                for f in x_test.columns
            ],
            "cover": [
                self.model.get_booster().get_score(
                    importance_type="cover"
                ).get(f, 0)
                for f in x_test.columns
            ]
        })

        importance["gain_percent"] = (
            importance["gain"] /
            importance["gain"].sum() * 100
        )

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
            json.dump(self.params, f, indent=4)

    def predict(self, data=None):
        if data is None:
            data = self.test

        x = data.drop(
            columns=["target", "stock"],
            errors="ignore"
        )

        return self.model.predict(x)

    def predict_probability(self, data=None):
        if data is None:
            data = self.test

        x = data.drop(
            columns=["target", "stock"],
            errors="ignore"
        )

        return self.model.predict_proba(x)[:, 1]

    def evaluate(self):
        stats = pd.read_parquet(
            f"{self.stats_dir}/model_stats.parquet"
        )

        return stats.iloc[0].to_dict()

    def feature_usefulness(self):
        return pd.read_parquet(
            f"{self.stats_dir}/feature_stats.parquet"
        )

    def save(self, file="data/xgboost/xgboost_model.json"):
        self.model.save_model(file)

    def load(self, file="data/xgboost/xgboost_model.json"):
        self.model = xgb.XGBClassifier()
        self.model.load_model(file)
        return self.model


    def upload_all(self, bucket, cloud_folder):
        s3 = boto3.client(
            "s3",
            endpoint_url="https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com",
            aws_access_key_id="f47f48ce0d129b1a69bb36da1d64bad1",
            aws_secret_access_key="3e92e25062abc6fe86c13455712967444aa1ffa3492d1d81258f7f4ecd5923aa"
        )

        files = [
            "xgboost_model.json",
            "model_stats.parquet",
            "feature_stats.parquet",
            "model_params.json"
        ]

        for file in files:
            local_file = (
                self.stats_dir + "/" + file
                if file != "xgboost_model.json"
                else "data/xgboost/" + file
            )

            with open(local_file, "rb") as f:
                s3.put_object(
                    Bucket=bucket,
                    Key=f"{cloud_folder}/{file}",
                    Body=f.read()
                )