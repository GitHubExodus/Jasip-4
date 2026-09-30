import os
import boto3
import numpy as np
import pandas as pd


class SplitRanking:
    def __init__(
        self,
        data,
        local_dir="data/splits",
        bucket="stocks-data",
        cloud_folder="jasip4/splits"
    ):
        self.data = data.copy()
        self.local_dir = local_dir
        self.bucket = bucket
        self.cloud_folder = cloud_folder

        os.makedirs(local_dir, exist_ok=True)

        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"]
        )

    def run(self):
        target = self.data["target"].to_numpy(dtype=np.float64)

        features = self.data.drop(
            columns=["target", "stock"],
            errors="ignore"
        )

        results = []

        for feature in features.columns:
            values = features[feature].to_numpy(dtype=np.float64)

            valid = np.isfinite(values) & np.isfinite(target)

            values = values[valid]
            profits = target[valid]

            if len(values) < 2:
                continue

            order = np.argsort(values)
            values = values[order]
            profits = profits[order]

            # Only test splits where the feature value actually changes.
            changes = np.r_[values[1:] != values[:-1], True]

            split_indexes = np.flatnonzero(changes)

            cumulative = np.cumsum(profits)
            total_profit = cumulative[-1]

            for i in split_indexes[:-1]:
                split = (values[i] + values[i + 1]) / 2

                yes_profit = cumulative[i]
                no_profit = total_profit - yes_profit

                yes_count = i + 1
                no_count = len(values) - yes_count

                results.append({
                    "feature": feature,
                    "condition": "<=",
                    "split": split,
                    "bars": yes_count,
                    "total_profit": yes_profit,
                    "average_profit": yes_profit / yes_count
                })

                results.append({
                    "feature": feature,
                    "condition": ">",
                    "split": split,
                    "bars": no_count,
                    "total_profit": no_profit,
                    "average_profit": no_profit / no_count
                })

        ranking = pd.DataFrame(results)

        ranking = ranking.sort_values(
            "total_profit",
            ascending=False
        ).reset_index(drop=True)

        ranking["rank"] = np.arange(1, len(ranking) + 1)

        self.ranking = ranking

        return ranking

    def save(self):
        file = f"{self.local_dir}/split_ranking.parquet"

        self.ranking.to_parquet(
            file,
            index=False
        )

        with open(file, "rb") as f:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"{self.cloud_folder}/split_ranking.parquet",
                Body=f.read()
            )

        return file