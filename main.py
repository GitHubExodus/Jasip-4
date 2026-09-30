import os
import io
import boto3
import numpy as np
import pandas as pd

from simulation import Simulation


class StockLibrary:
    def __init__(
        self,
        bucket,
        start_date,
        end_date,
        train_percent=0.70,
        validation_percent=0.15,
        local_dir="data"
    ):
        self.bucket = bucket
        self.start_date = start_date
        self.end_date = end_date
        self.train_percent = train_percent
        self.validation_percent = validation_percent
        self.local_dir = local_dir

        self.s3 = boto3.client(
            "s3",
            endpoint_url="https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com",
            aws_access_key_id="f47f48ce0d129b1a69bb36da1d64bad1",
            aws_secret_access_key="3e92e25062abc6fe86c13455712967444aa1ffa3492d1d81258f7f4ecd5923aa"
        )

        os.makedirs(local_dir, exist_ok=True)

    def _rsi(self, x, n):
        d = x.diff()
        up = d.clip(lower=0).rolling(n).mean()
        down = -d.clip(upper=0).rolling(n).mean()
        return 100 - (100 / (1 + up / down))

    def _atr(self, df, n):
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(n).mean()

    def _process(self, stock, data):
        data = data.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"], utc=True
        )

        data["timestamp"] = data["timestamp"].dt.tz_convert(
            "America/New_York"
        )

        data = data.set_index("timestamp").sort_index()
        data = data.loc[self.start_date:self.end_date]

        data = data.between_time("09:30", "16:00")
        data = data.dropna(
            subset=["open", "high", "low", "close", "volume"]
        )

        data = data.resample("1D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        dollar_vol = data["close"] * data["volume"]

        x = pd.DataFrame(index=data.index)

        x["close"] = data["close"]
        x["vol"] = data["volume"]
        x["dollar_vol"] = dollar_vol
        x["avg_90_vol"] = data["volume"].rolling(90).mean()
        x["avg_90_dollar_vol"] = dollar_vol.rolling(90).mean()

        for n in [3, 5, 9, 21, 50, 100, 200]:
            ema = data["close"].ewm(span=n, adjust=False).mean()

            x[f"ema_{n}_pct"] = (
                data["close"] / ema - 1
            ) * 100

            x[f"rsi_{n}"] = self._rsi(data["close"], n)
            x[f"roc_{n}"] = data["close"].pct_change(n) * 100

        for n in [9, 50, 100, 200]:
            x[f"atr_{n}_pct"] = (
                self._atr(data, n) / data["close"]
            ) * 100

            x[f"std_{n}_pct"] = (
                data["close"].rolling(n).std()
                / data["close"]
            ) * 100

        x["avg_day_vol_90"] = data["volume"].rolling(90).mean()
        x["avg_price_x_vol_90"] = dollar_vol.rolling(90).mean()

        # Remove bars below $100k dollar volume
        valid = dollar_vol > 100_000

        data = data[valid]
        x = x[valid]

        # Run simulation
        result = Simulation(data).run()

        sim_cols = [
            "close_to_high",
            "close_to_low",
            "ema_200_high",
            "ema_200_low"
        ]

        x = x.join(result[sim_cols])

        # Current bar only gets previous-bar information.
        x = x.shift(1)

        # Profit belongs to the current bar.
        profit = result["profit"]

        # Align and remove every row with missing input/profit.
        valid = x.notna().all(axis=1) & profit.notna()

        x = x[valid]
        profit = profit[valid]

        x["target"] = profit

        return x, profit

    def upload(self, local_file, cloud_file):
        with open(local_file, "rb") as f:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=cloud_file,
                Body=f.read()
            )

    def run(self):
        objects = self.s3.list_objects_v2(
            Bucket=self.bucket
        ).get("Contents", [])

        stocks = [
            obj["Key"][:-8]
            for obj in objects
            if "/" not in obj["Key"]
            and obj["Key"].endswith(".parquet")
        ]

        # stocks = ["AAPL"]

        train_parts = []
        validation_parts = []
        testing_parts = []
        profit_parts = []

        for stock in stocks:
            obj = self.s3.get_object(
                Bucket=self.bucket,
                Key=stock + ".parquet"
            )

            data = pd.read_parquet(
                io.BytesIO(obj["Body"].read())
            )

            inputs, profit = self._process(stock, data)

            n = len(inputs)

            train_end = int(n * self.train_percent)

            validation_end = (
                train_end
                + int(n * self.validation_percent)
            )

            train_parts.append(inputs.iloc[:train_end])
            validation_parts.append(
                inputs.iloc[train_end:validation_end]
            )
            testing_parts.append(
                inputs.iloc[validation_end:]
            )

            profit_parts.append(
                pd.DataFrame(
                    {stock: profit},
                    index=profit.index
                )
            )

        train = pd.concat(train_parts)
        validation = pd.concat(validation_parts)
        testing = pd.concat(testing_parts)

        profits = pd.concat(profit_parts, axis=1)
        profits = profits.sort_index()

        train.to_parquet(
            f"{self.local_dir}/training.parquet"
        )

        validation.to_parquet(
            f"{self.local_dir}/validation.parquet"
        )

        testing.to_parquet(
            f"{self.local_dir}/testing.parquet"
        )

        profits.to_parquet(
            f"{self.local_dir}/profits.parquet"
        )

        return train, validation, testing, profits