import os
import boto3
import numpy as np
import pandas as pd


class SplitRanking:
    def __init__(
        self,
        train,
        validation,
        testing,
        local_dir="data/splits",
        bucket="stocks-data",
        cloud_folder="jasip4/splits"
    ):
        self.train = train.copy()
        self.validation = validation.copy()
        self.testing = testing.copy()

        self.local_dir = local_dir
        self.bucket = bucket
        self.cloud_folder = cloud_folder

        os.makedirs(local_dir, exist_ok=True)

        self.s3 = boto3.client(
            "s3",
            endpoint_url="https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com",
            aws_access_key_id="f47f48ce0d129b1a69bb36da1d64bad1",
            aws_secret_access_key="3e92e25062abc6fe86c13455712967444aa1ffa3492d1d81258f7f4ecd5923aa"
        )

    def find_splits(self):
        target = self.train["target"].to_numpy(dtype=np.float64)

        features = self.train.drop(
            columns=["target", "stock"],
            errors="ignore"
        )

        splits = []

        for feature in features.columns:
            values = features[feature].to_numpy(dtype=np.float64)

            valid = np.isfinite(values) & np.isfinite(target)

            values = values[valid]

            if len(values) < 2:
                continue

            values = np.sort(values)

            changes = values[1:] != values[:-1]

            indexes = np.flatnonzero(changes)

            for i in indexes:
                split = (values[i] + values[i + 1]) / 2

                splits.append({
                    "feature": feature,
                    "condition": "<=",
                    "split": split
                })

                splits.append({
                    "feature": feature,
                    "condition": ">",
                    "split": split
                })

        return pd.DataFrame(splits)

    def evaluate(self, data, splits):
        results = []

        for _, row in splits.iterrows():
            feature = row["feature"]
            condition = row["condition"]
            split = row["split"]

            values = data[feature].to_numpy(dtype=np.float64)
            target = data["target"].to_numpy(dtype=np.float64)

            valid = np.isfinite(values) & np.isfinite(target)

            values = values[valid]
            target = target[valid]

            if condition == "<=":
                yes = values <= split
            else:
                yes = values > split

            yes_profit = target[yes].sum()
            yes_bars = yes.sum()

            results.append({
                "feature": feature,
                "condition": condition,
                "split": split,
                "bars": yes_bars,
                "total_profit": yes_profit,
                "average_profit": (
                    yes_profit / yes_bars
                    if yes_bars > 0
                    else np.nan
                )
            })

        ranking = pd.DataFrame(results)

        return ranking.sort_values(
            "total_profit",
            ascending=False
        ).reset_index(drop=True)

    def run(self):
        splits = self.find_splits()

        self.training = self.evaluate(
            self.train,
            splits
        )

        self.validation = self.evaluate(
            self.validation,
            splits
        )

        self.testing = self.evaluate(
            self.testing,
            splits
        )

        self.training["rank"] = np.arange(
            1, len(self.training) + 1
        )

        self.validation["rank"] = np.arange(
            1, len(self.validation) + 1
        )

        self.testing["rank"] = np.arange(
            1, len(self.testing) + 1
        )

        return (
            self.training,
            self.validation,
            self.testing
        )

    def save(self):
        files = {
            "training_split_ranking.parquet": self.training,
            "validation_split_ranking.parquet": self.validation,
            "testing_split_ranking.parquet": self.testing
        }

        for name, data in files.items():
            local_file = f"{self.local_dir}/{name}"

            data.to_parquet(
                local_file,
                index=False
            )

            with open(local_file, "rb") as f:
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=f"{self.cloud_folder}/{name}",
                    Body=f.read()
                )