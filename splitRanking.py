# splitRanking.py

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

    def _candidate_splits(self, data, feature):
        values = data[feature].to_numpy(dtype=np.float64)
        profits = data["target"].to_numpy(dtype=np.float64)

        valid = np.isfinite(values) & np.isfinite(profits)

        values = values[valid]
        profits = profits[valid]

        if len(values) < 2:
            return pd.DataFrame()

        order = np.argsort(values)

        values = values[order]
        profits = profits[order]

        changes = values[1:] != values[:-1]

        indexes = np.flatnonzero(changes)

        if len(indexes) == 0:
            return pd.DataFrame()

        cumulative = np.cumsum(profits)
        total_profit = cumulative[-1]

        results = []

        for i in indexes:
            split = (values[i] + values[i + 1]) / 2

            yes_profit = cumulative[i]
            no_profit = total_profit - yes_profit

            results.append({
                "condition": "<=",
                "split": split,
                "profit": yes_profit
            })

            results.append({
                "condition": ">",
                "split": split,
                "profit": no_profit
            })

        return pd.DataFrame(results)

    def _weighted_split(self, candidates):
        if candidates.empty:
            return None

        profits = candidates["profit"].to_numpy(dtype=np.float64)

        minimum = np.min(profits)

        weights = profits - minimum + 1e-12

        if np.sum(weights) <= 0:
            weights = np.ones(len(profits))

        splits = candidates["split"].to_numpy(dtype=np.float64)

        return np.sum(splits * weights) / np.sum(weights)

    def _score_split(self, data, feature, condition, split):
        values = data[feature].to_numpy(dtype=np.float64)
        profits = data["target"].to_numpy(dtype=np.float64)

        valid = np.isfinite(values) & np.isfinite(profits)

        if condition == "<=":
            selected = valid & (values <= split)
        else:
            selected = valid & (values > split)

        selected_profit = profits[selected]

        if len(selected_profit) == 0:
            return {
                "bars": 0,
                "profit": 0.0,
                "average_profit": np.nan
            }

        return {
            "bars": len(selected_profit),
            "profit": selected_profit.sum(),
            "average_profit": selected_profit.mean()
        }

    def _find_best_input_split(self, feature):
        candidates = self._candidate_splits(
            self.train,
            feature
        )

        if candidates.empty:
            return None

        results = []

        for condition in ["<=", ">"]:
            condition_candidates = candidates[
                candidates["condition"] == condition
            ]

            if condition_candidates.empty:
                continue

            weighted_split = self._weighted_split(
                condition_candidates
            )

            score = self._score_split(
                self.train,
                feature,
                condition,
                weighted_split
            )

            results.append({
                "feature": feature,
                "condition": condition,
                "split": weighted_split,
                "training_bars": score["bars"],
                "training_profit": score["profit"],
                "training_average_profit": score["average_profit"]
            })

        if not results:
            return None

        results = pd.DataFrame(results)

        # The weighted-average split is recalculated on the
        # actual training data, then the better condition is kept.
        best = results.loc[
            results["training_profit"].idxmax()
        ]

        return best.to_dict()

    def _evaluate(self, ranking, data, prefix):
        for i in ranking.index:
            feature = ranking.at[i, "feature"]
            condition = ranking.at[i, "condition"]
            split = ranking.at[i, "split"]

            score = self._score_split(
                data,
                feature,
                condition,
                split
            )

            ranking.at[i, f"{prefix}_bars"] = score["bars"]
            ranking.at[i, f"{prefix}_profit"] = score["profit"]
            ranking.at[i, f"{prefix}_average_profit"] = (
                score["average_profit"]
            )

        ranking[f"{prefix}_bars"] = (
            ranking[f"{prefix}_bars"]
            .fillna(0)
            .astype(int)
        )

        ranking[f"{prefix}_profit"] = (
            ranking[f"{prefix}_profit"]
            .fillna(0.0)
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
        features = self.train.drop(
            columns=["target", "stock"],
            errors="ignore"
        ).columns

        results = []

        # Find exactly ONE representative split per input.
        # Only training is used to find it.
        for feature in features:
            result = self._find_best_input_split(feature)

            if result is not None:
                results.append(result)

        ranking = pd.DataFrame(results)

        if ranking.empty:
            self.ranking = ranking
            return ranking

        # The same exact input/split is now tested
        # against all three datasets.
        ranking = self._evaluate(
            ranking,
            self.train,
            "training"
        )

        ranking = self._evaluate(
            ranking,
            self.validation,
            "validation"
        )

        ranking = self._evaluate(
            ranking,
            self.testing,
            "testing"
        )

        # Master table is always stored in training order.
        ranking = ranking.sort_values(
            "training_profit",
            ascending=False
        ).reset_index(drop=True)

        ranking["training_rank"] = (
            np.arange(len(ranking)) + 1
        )

        self.ranking = ranking

        return ranking

    def print_rankings(self, n=30):
        columns = [
            "feature",
            "condition",
            "split",
            "training_profit",
            "validation_profit",
            "testing_profit"
        ]

        print("\nTRAINING RANKING")
        print(
            self.ranking
            .sort_values("training_rank")
            .head(n)[columns]
            .to_string(index=False)
        )

        print("\nVALIDATION RANKING")
        print(
            self.ranking
            .sort_values("validation_rank")
            .head(n)[columns]
            .to_string(index=False)
        )

        print("\nTESTING RANKING")
        print(
            self.ranking
            .sort_values("testing_rank")
            .head(n)[columns]
            .to_string(index=False)
        )

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