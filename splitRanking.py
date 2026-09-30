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

    def _find_training_splits(self):
        target = self.train["target"].to_numpy(dtype=np.float64)

        features = self.train.drop(
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

            changes = np.r_[
                values[1:] != values[:-1],
                True
            ]

            split_indexes = np.flatnonzero(changes)

            cumulative = np.cumsum(profits)
            total_profit = cumulative[-1]

            for i in split_indexes[:-1]:
                split = (
                    values[i] + values[i + 1]
                ) / 2

                yes_profit = cumulative[i]
                no_profit = total_profit - yes_profit

                results.append({
                    "feature": feature,
                    "condition": "<=",
                    "split": split,
                    "training_bars": i + 1,
                    "training_profit": yes_profit
                })

                results.append({
                    "feature": feature,
                    "condition": ">",
                    "split": split,
                    "training_bars": (
                        len(values) - i - 1
                    ),
                    "training_profit": no_profit
                })

        return pd.DataFrame(results)

    def _test_splits(self, ranking, data, prefix):
        target = data["target"].to_numpy(
            dtype=np.float64
        )

        features = data.drop(
            columns=["target", "stock"],
            errors="ignore"
        )

        for feature in ranking["feature"].unique():

            if feature not in features.columns:
                continue

            values = features[feature].to_numpy(
                dtype=np.float64
            )

            feature_rows = ranking.index[
                ranking["feature"] == feature
            ]

            for index in feature_rows:
                condition = ranking.at[
                    index,
                    "condition"
                ]

                split = ranking.at[
                    index,
                    "split"
                ]

                valid = (
                    np.isfinite(values)
                    & np.isfinite(target)
                )

                if condition == "<=":
                    selected = (
                        valid
                        & (values <= split)
                    )
                else:
                    selected = (
                        valid
                        & (values > split)
                    )

                profits = target[selected]

                ranking.at[
                    index,
                    f"{prefix}_bars"
                ] = len(profits)

                ranking.at[
                    index,
                    f"{prefix}_profit"
                ] = (
                    profits.sum()
                    if len(profits)
                    else 0.0
                )

                ranking.at[
                    index,
                    f"{prefix}_average_profit"
                ] = (
                    profits.mean()
                    if len(profits)
                    else np.nan
                )

        ranking[f"{prefix}_profit"] = (
            ranking[f"{prefix}_profit"]
            .fillna(0.0)
        )

        ranking[f"{prefix}_bars"] = (
            ranking[f"{prefix}_bars"]
            .fillna(0)
            .astype(int)
        )

        ranking[f"{prefix}_rank"] = (
            ranking[f"{prefix}_profit"]
            .rank(
                ascending=False,
                method="min"
            )
            .astype(int)
        )

        return ranking

    def run(self):
        # Find the actual splits using training only.
        ranking = self._find_training_splits()

        # Test those exact same splits everywhere.
        ranking = self._test_splits(
            ranking,
            self.train,
            "training"
        )

        ranking = self._test_splits(
            ranking,
            self.validation,
            "validation"
        )

        ranking = self._test_splits(
            ranking,
            self.testing,
            "testing"
        )

        # The worst rank that this split received
        # across the three datasets.
        ranking["worst_rank"] = ranking[
            [
                "training_rank",
                "validation_rank",
                "testing_rank"
            ]
        ].max(axis=1)

        # Best worst-case split first.
        ranking = ranking.sort_values(
            [
                "worst_rank",
                "training_rank"
            ]
        ).reset_index(drop=True)

        # Overall ranking based on best worst-case rank.
        ranking["overall_rank"] = np.arange(
            1,
            len(ranking) + 1
        )

        self.ranking = ranking

        return ranking

    def print_rankings(self, n=30):
        print("\nTRAINING RANKING")

        print(
            self.ranking
            .sort_values("training_rank")
            .head(n)
            .to_string(index=False)
        )

        print("\nVALIDATION RANKING")

        print(
            self.ranking
            .sort_values("validation_rank")
            .head(n)
            .to_string(index=False)
        )

        print("\nTESTING RANKING")

        print(
            self.ranking
            .sort_values("testing_rank")
            .head(n)
            .to_string(index=False)
        )

        print("\nBEST WORST-RANK RANKING")

        print(
            self.ranking
            .sort_values("overall_rank")
            .head(n)
            .to_string(index=False)
        )

    def save(self):
        file = (
            f"{self.local_dir}/"
            "split_ranking.parquet"
        )

        self.ranking.to_parquet(
            file,
            index=False
        )

        with open(file, "rb") as f:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=(
                    f"{self.cloud_folder}/"
                    "split_ranking.parquet"
                ),
                Body=f.read()
            )

        return file