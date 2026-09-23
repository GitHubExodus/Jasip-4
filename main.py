# ============================================================
# 1. CONFIGURATION
# ============================================================

import os
import io
import gc

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME = "stocks-data"

R2_ENDPOINT = (
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
)

TIMEZONE = "America/New_York"

DATA_START = pd.Timestamp("2025-01-01", tz=TIMEZONE)
DATA_END = pd.Timestamp.now(tz=TIMEZONE)

WARMUP_DAYS = 20

RAW_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

CACHE_DIR = "data"

os.makedirs(CACHE_DIR, exist_ok=True)



# ============================================================
# 2. R2 CONNECTION
# ============================================================

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


# # ============================================================
# # 3. FILE DISCOVERY
# # ============================================================

# stock_keys = []

# response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME)

# for obj in response.get("Contents", []):
#     key = obj["Key"]

#     if "/" not in key and key.lower().endswith(".parquet"):
#         stock_keys.append(key)

# print(f"Found {len(stock_keys):,} stock files")


# ============================================================
# 3. FILE DISCOVERY
# ============================================================

MAX_STOCKS = 3
# Set to None or a large number to process all available stocks.

stock_keys = []

continuation_token = None

while True:

    params = {
        "Bucket": R2_BUCKET_NAME,
    }

    if continuation_token is not None:
        params["ContinuationToken"] = continuation_token

    response = s3.list_objects_v2(**params)

    for obj in response.get("Contents", []):
        key = obj["Key"]

        if "/" not in key and key.lower().endswith(".parquet"):
            stock_keys.append(key)

    if not response.get("IsTruncated"):
        break

    continuation_token = response["NextContinuationToken"]

if MAX_STOCKS is not None:
    stock_keys = stock_keys[:MAX_STOCKS]

print(f"Found {len(stock_keys):,} stock files to process")



# ============================================================
# 4. PROCESS EACH STOCK
# ============================================================

for key in stock_keys:

    obj = s3.get_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
    )

    df = pd.read_parquet(
        io.BytesIO(obj["Body"].read()),
        columns=RAW_COLUMNS,
    )

    symbol = key.rsplit("/", 1)[-1].removesuffix(".parquet").upper()

    print(f"Processing {symbol}...")

    # ============================================================
    # 5. CLEAN DATA + TIMEZONE
    # ============================================================

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df["open"] = df["open"].astype(np.float64)
    df["high"] = df["high"].astype(np.float64)
    df["low"] = df["low"].astype(np.float64)
    df["close"] = df["close"].astype(np.float64)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = (
        df.dropna(subset=RAW_COLUMNS)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df["timestamp"] = df["timestamp"].dt.tz_convert(TIMEZONE)

    # ============================================================
    # 6. WARM-UP + DATE RANGE
    # ============================================================

    warmup_start = DATA_START - pd.Timedelta(days=WARMUP_DAYS)

    df = df[
        (df["timestamp"] >= warmup_start) &
        (df["timestamp"] <= DATA_END)
    ].copy()

    if df.empty or df["timestamp"].max() < DATA_START:
        print(f"Skipping {symbol}: no usable data")
        continue

    # ============================================================
    # 7. SESSION FEATURES
    # ============================================================

    ts = df["timestamp"]

    minutes = ts.dt.hour * 60 + ts.dt.minute
    session_start = 4 * 60

    df["session_elapsed_minutes"] = minutes - session_start

    df["session"] = np.select(
        [
            (minutes >= 240) & (minutes < 570),
            (minutes >= 570) & (minutes < 960),
            (minutes >= 960) & (minutes < 1200),
        ],
        [
            "PREMARKET",
            "REGULAR_MARKET",
            "AFTER_HOURS",
        ],
        default=None,
    )

    session_id = (
        ts.dt.normalize()
        + pd.to_timedelta(
            np.where(minutes < session_start, -1, 0), unit="D"
        )
    )

    df["session_open"] = (
        df.groupby(session_id)["close"]
        .transform("first")
    )

    df["session_pct_change"] = (
        (df["close"] - df["session_open"])
        / df["session_open"]
        * 100
    )

    # ============================================================
    # 8. VOLUME FEATURES
    # ============================================================

    df["dollar_volume"] = df["close"] * df["volume"]

    df["_date"] = df["timestamp"].dt.date
    df["_time"] = df["timestamp"].dt.time

    df["avg_volume_10d"] = (
        df.groupby("_time")["volume"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=10).mean())
    )

    df["avg_dollar_volume_10d"] = (
        df.groupby("_time")["dollar_volume"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=10).mean())
    )

    df["rvol"] = df["volume"] / df["avg_volume_10d"]

    df.drop(columns=["_date", "_time"], inplace=True)


    # ============================================================
    # 9. ATR / ROC / RSI
    # ============================================================

    df["_date"] = df["timestamp"].dt.normalize()

    daily = (
        df.groupby("_date", sort=True)
        .agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
    )

    prev_close = daily["close"].shift(1)

    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    daily["atr_pct"] = (
        true_range.rolling(14, min_periods=14).mean()
        / daily["close"]
        * 100
    )

    delta = daily["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss

    daily["rsi"] = 100 - (100 / (1 + rs))

    daily["roc_pct"] = daily["close"].pct_change(9) * 100

    # Use only completed previous-day indicators.
    daily = daily[["atr_pct", "roc_pct", "rsi"]].shift(1)

    df = df.join(daily, on="_date")

    df.drop(columns="_date", inplace=True)


    # ============================================================
    # 10. PREVIOUS 30-MINUTE HIGH
    # ============================================================

    df["_30m"] = df["timestamp"].dt.floor("30min")

    block_high = (
        df.groupby("_30m", sort=True)["high"]
        .max()
    )

    previous_30m_high = block_high.shift(1)

    df["previous_30m_high"] = df["_30m"].map(previous_30m_high)

    df.drop(columns="_30m", inplace=True)


    # ============================================================
    # 11. FINAL FILTER + REMOVE INVALID ROWS
    # ============================================================

    FEATURE_COLUMNS = [
        "close",
        "dollar_volume",
        "avg_volume_10d",
        "avg_dollar_volume_10d",
        "session_elapsed_minutes",
        "session_open",
        "session_pct_change",
        "rvol",
        "atr_pct",
        "roc_pct",
        "rsi",
        "previous_30m_high",
    ]

    df = df.dropna(subset=FEATURE_COLUMNS)

    df = df[
        (df["timestamp"] >= DATA_START) &
        (df["timestamp"] <= DATA_END)
    ].copy()

    if df.empty:
        print(f"Skipping {symbol}: no valid rows after indicators")
        continue

    df.insert(0, "symbol", symbol)

    # ============================================================
    # 12. APPEND TO DAILY PARQUET CACHE
    # ============================================================

    df["_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    for date, day_df in df.groupby("_date", sort=False):

        path = os.path.join(CACHE_DIR, f"{date}.parquet")

        day_df = day_df.drop(columns="_date")

        if os.path.exists(path):
            existing = pd.read_parquet(path)
            day_df = pd.concat([existing, day_df], ignore_index=True)

        day_df.to_parquet(
            path,
            index=False,
            engine="pyarrow",
        )

    # ============================================================
    # 13. MEMORY CLEANUP
    # ============================================================

    del df
    gc.collect()

    print(f"Finished {symbol}")







# ============================================================
# 14. OPTUNA SETUP
# ============================================================

import optuna

from optuna.samplers import TPESampler

# ============================================================
# 15. TRAINING DATA BOUNDARIES
# ============================================================

TRAIN_START = pd.Timestamp("2025-01-01", tz=TIMEZONE)
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz=TIMEZONE)


# ============================================================
# 16. OPTUNA STUDY
# ============================================================

study = optuna.create_study(
    study_name="jasip_sortino_optimization",
    direction="maximize",
    sampler=TPESampler(seed=42),
)


# ============================================================
# 17. PARAMETER SUGGESTION
# ============================================================

def suggest_filter_params(trial):

    p = {}

    # Close
    p["use_close_filter"] = trial.suggest_categorical(
        "use_close_filter", [True, False]
    )

    if p["use_close_filter"]:
        p["close_operator"] = trial.suggest_categorical(
            "close_operator",
            ["GREATER_THAN", "LESS_THAN", "IN_RANGE"],
        )
        p["close_min"] = trial.suggest_float(
            "close_min", 0.50, 100.0, step=0.50
        )

        if p["close_operator"] == "IN_RANGE":
            p["close_max"] = trial.suggest_float(
                "close_max", 100.0, 500.0, step=5.0
            )

    # Time
    p["use_time_filter"] = trial.suggest_categorical(
        "use_time_filter", [True, False]
    )

    if p["use_time_filter"]:
        p["time_operator"] = trial.suggest_categorical(
            "time_operator",
            ["IN_RANGE", "OUTSIDE_RANGE"],
        )
        p["time_min"] = trial.suggest_int(
            "time_min", 0, 960, step=15
        )
        p["time_max"] = trial.suggest_int(
            "time_max", 15, 960, step=15
        )

    # RVOL
    p["use_rvol_filter"] = trial.suggest_categorical(
        "use_rvol_filter", [True, False]
    )

    if p["use_rvol_filter"]:
        p["rvol_operator"] = trial.suggest_categorical(
            "rvol_operator",
            [
                "GREATER_THAN",
                "LESS_THAN",
                "IN_RANGE",
                "OUTSIDE_RANGE",
            ],
        )
        p["rvol_min"] = trial.suggest_float(
            "rvol_min", 0.5, 10.0, step=0.1
        )

        if p["rvol_operator"] in ("IN_RANGE", "OUTSIDE_RANGE"):
            p["rvol_max"] = trial.suggest_float(
                "rvol_max", 1.0, 100.0, step=0.5
            )

    # Average Volume
    p["use_avg_vol_filter"] = trial.suggest_categorical(
        "use_avg_vol_filter", [True, False]
    )

    if p["use_avg_vol_filter"]:
        p["avg_vol_operator"] = trial.suggest_categorical(
            "avg_vol_operator",
            ["GREATER_THAN", "LESS_THAN"],
        )
        p["avg_vol_min"] = trial.suggest_float(
            "avg_vol_min",
            100_000,
            500_000_000,
            step=100_000,
        )

    # Dollar Volume
    p["use_dollar_vol_filter"] = trial.suggest_categorical(
        "use_dollar_vol_filter", [True, False]
    )

    if p["use_dollar_vol_filter"]:
        p["bar_dollar_vol_min"] = trial.suggest_float(
            "bar_dollar_vol_min",
            100_000,
            100_000_000,
            step=100_000,
        )
        p["exp_dollar_vol_min"] = trial.suggest_float(
            "exp_dollar_vol_min",
            100_000,
            500_000_000,
            step=100_000,
        )

    # ATR
    p["use_atr_filter"] = trial.suggest_categorical(
        "use_atr_filter", [True, False]
    )

    if p["use_atr_filter"]:
        p["atr_operator"] = trial.suggest_categorical(
            "atr_operator",
            [
                "GREATER_THAN",
                "LESS_THAN",
                "IN_RANGE",
                "OUTSIDE_RANGE",
            ],
        )
        p["atr_min"] = trial.suggest_float(
            "atr_min", 0.5, 10.0, step=0.1
        )

        if p["atr_operator"] in ("IN_RANGE", "OUTSIDE_RANGE"):
            p["atr_max"] = trial.suggest_float(
                "atr_max", 2.0, 25.0, step=0.5
            )

    # ROC
    p["use_roc_filter"] = trial.suggest_categorical(
        "use_roc_filter", [True, False]
    )

    if p["use_roc_filter"]:
        p["roc_operator"] = trial.suggest_categorical(
            "roc_operator",
            [
                "GREATER_THAN",
                "LESS_THAN",
                "IN_RANGE",
                "OUTSIDE_RANGE",
            ],
        )
        p["roc_min"] = trial.suggest_float(
            "roc_min", -20.0, 20.0, step=0.25
        )

        if p["roc_operator"] in ("IN_RANGE", "OUTSIDE_RANGE"):
            p["roc_max"] = trial.suggest_float(
                "roc_max", 0.0, 100.0, step=0.5
            )

    # RSI
    p["use_rsi_filter"] = trial.suggest_categorical(
        "use_rsi_filter", [True, False]
    )

    if p["use_rsi_filter"]:
        p["rsi_operator"] = trial.suggest_categorical(
            "rsi_operator",
            [
                "GREATER_THAN",
                "LESS_THAN",
                "IN_RANGE",
                "OUTSIDE_RANGE",
            ],
        )
        p["rsi_min"] = trial.suggest_int(
            "rsi_min", 10, 60, step=1
        )

        if p["rsi_operator"] in ("IN_RANGE", "OUTSIDE_RANGE"):
            p["rsi_max"] = trial.suggest_int(
                "rsi_max", 40, 90, step=1
            )

    return p



# ============================================================
# 18. SINGLE SCREENER PARAMETERS
# ============================================================

def suggest_params(trial):

    p = suggest_filter_params(trial)

    p["tp_pct"] = trial.suggest_float(
        "tp_pct", 0.01, 0.20, step=0.005
    )

    p["tsl_pct"] = trial.suggest_float(
        "tsl_pct", 0.01, 0.20, step=0.005
    )

    return p



# ============================================================
# 19. SCREENER EVALUATION
# ============================================================

def passes_screener(row, p):

    checks = []

    if p["use_close_filter"]:
        if p["close_operator"] == "GREATER_THAN":
            checks.append(row.close > p["close_min"])
        elif p["close_operator"] == "LESS_THAN":
            checks.append(row.close < p["close_min"])
        else:
            checks.append(p["close_min"] <= row.close <= p["close_max"])

    if p["use_time_filter"]:
        if p["time_operator"] == "IN_RANGE":
            checks.append(p["time_min"] <= row.session_elapsed_minutes <= p["time_max"])
        else:
            checks.append(
                row.session_elapsed_minutes < p["time_min"]
                or row.session_elapsed_minutes > p["time_max"]
            )

    if p["use_rvol_filter"]:
        if p["rvol_operator"] == "GREATER_THAN":
            checks.append(row.rvol > p["rvol_min"])
        elif p["rvol_operator"] == "LESS_THAN":
            checks.append(row.rvol < p["rvol_min"])
        elif p["rvol_operator"] == "IN_RANGE":
            checks.append(p["rvol_min"] <= row.rvol <= p["rvol_max"])
        else:
            checks.append(row.rvol < p["rvol_min"] or row.rvol > p["rvol_max"])

    if p["use_avg_vol_filter"]:
        if p["avg_vol_operator"] == "GREATER_THAN":
            checks.append(row.avg_volume_10d > p["avg_vol_min"])
        else:
            checks.append(row.avg_volume_10d < p["avg_vol_min"])

    if p["use_dollar_vol_filter"]:
        checks.append(row.dollar_volume >= p["bar_dollar_vol_min"])
        checks.append(
            row.avg_dollar_volume_10d >= p["exp_dollar_vol_min"]
        )

    if p["use_atr_filter"]:
        if p["atr_operator"] == "GREATER_THAN":
            checks.append(row.atr_pct > p["atr_min"])
        elif p["atr_operator"] == "LESS_THAN":
            checks.append(row.atr_pct < p["atr_min"])
        elif p["atr_operator"] == "IN_RANGE":
            checks.append(p["atr_min"] <= row.atr_pct <= p["atr_max"])
        else:
            checks.append(row.atr_pct < p["atr_min"] or row.atr_pct > p["atr_max"])

    if p["use_roc_filter"]:
        if p["roc_operator"] == "GREATER_THAN":
            checks.append(row.roc_pct > p["roc_min"])
        elif p["roc_operator"] == "LESS_THAN":
            checks.append(row.roc_pct < p["roc_min"])
        elif p["roc_operator"] == "IN_RANGE":
            checks.append(p["roc_min"] <= row.roc_pct <= p["roc_max"])
        else:
            checks.append(row.roc_pct < p["roc_min"] or row.roc_pct > p["roc_max"])

    if p["use_rsi_filter"]:
        if p["rsi_operator"] == "GREATER_THAN":
            checks.append(row.rsi > p["rsi_min"])
        elif p["rsi_operator"] == "LESS_THAN":
            checks.append(row.rsi < p["rsi_min"])
        elif p["rsi_operator"] == "IN_RANGE":
            checks.append(p["rsi_min"] <= row.rsi <= p["rsi_max"])
        else:
            checks.append(row.rsi < p["rsi_min"] or row.rsi > p["rsi_max"])

    return all(checks)



# ============================================================
# 20. SIMULATION ENGINE
# ============================================================

class Simulation:

    def __init__(self, params, start_date, end_date):

        self.params = params
        self.start_date = start_date
        self.end_date = end_date

        self.initial_capital = 10_000.0
        self.buying_power = self.initial_capital

        self.open_trades = {}
        self.completed_trades = []

        self.equity_curve = []

        self.daily_equity = []
        self.daily_returns = []

        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0

        self.current_day = None
        self.session = None

        self.session_anchors = {}
        self.last_bar_high = {}

        self.entry_windows = {}

        self.current_timestamp = None

        self.portfolio_balance = self.initial_capital
        self.realized_pnl = 0.0


    ######
    def run(self):

        dates = pd.date_range(
            self.start_date.normalize(),
            self.end_date.normalize(),
            freq="D",
        )

        for day in dates:

            path = f"{CACHE_DIR}/{day.strftime('%Y-%m-%d')}.parquet"

            if not os.path.exists(path):
                continue

            df = pd.read_parquet(path)

            df = df[
                (df["timestamp"] >= self.start_date) &
                (df["timestamp"] <= self.end_date)
            ]

            if df.empty:
                del df
                continue

            df.sort_values("timestamp", inplace=True)

            self._process_day(df)

            del df
            gc.collect()

        return self.stats()


    #######
    def _process_day(self, df):
        self.buying_power = self.portfolio_balance

        for timestamp, bars in df.groupby("timestamp", sort=True):

            self.current_timestamp = timestamp


            ######
            for symbol, trade in list(self.open_trades.items()):

                row = bars[bars["symbol"] == symbol]

                if row.empty:
                    continue

                row = row.iloc[0]

                # Peak high
                trade["peak_high"] = max(trade["peak_high"], row.high)

                tp_price = trade["entry_price"] * (1 + self.params["tp_pct"])
                tsl_price = trade["peak_high"] * (1 - self.params["tsl_pct"])

                exit_price = None
                exit_reason = None

                # Gap-aware TP
                if row.open >= tp_price:
                    exit_price = row.open
                    exit_reason = "TAKE_PROFIT"
                elif row.high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TAKE_PROFIT"

                # Gap-aware trailing stop
                elif row.open <= tsl_price:
                    exit_price = row.open
                    exit_reason = "TRAILING_STOP_LOSS"
                elif row.low <= tsl_price:
                    exit_price = tsl_price
                    exit_reason = "TRAILING_STOP_LOSS"

                if exit_price is not None:
                    trade["exit_price"] = exit_price
                    trade["exit_timestamp"] = timestamp
                    trade["exit_reason"] = exit_reason

                    trade["profit"] = (
                        trade["exit_price"] - trade["entry_price"]
                    ) * trade["shares"]

                    trade["return_pct"] = (
                        trade["exit_price"] / trade["entry_price"] - 1
                    ) * 100

                    trade["length_minutes"] = (
                        trade["exit_timestamp"] - trade["entry_timestamp"]
                    ).total_seconds() / 60

                    self.portfolio_balance += trade["profit"]
                    
                    self.completed_trades.append(trade)
                    del self.open_trades[symbol]



            #####
            for symbol, trade in list(self.open_trades.items()):

                row = bars[bars["symbol"] == symbol]

                if row.empty:
                    continue

                row = row.iloc[0]

                session_end = (
                    timestamp.time() >= pd.Timestamp(
                        {"PREMARKET": "09:29:00",
                         "REGULAR": "15:59:00",
                         "AFTER_HOURS": "19:59:00"}[row["session"]]
                    ).time()
                )

                if session_end:

                    trade["exit_price"] = row["close"]
                    trade["exit_timestamp"] = timestamp
                    trade["exit_reason"] = "SESSION_END_LIQUIDATION"

                    trade["profit"] = (
                        trade["exit_price"] - trade["entry_price"]
                    ) * trade["shares"]

                    trade["return_pct"] = (
                        trade["exit_price"] / trade["entry_price"] - 1
                    ) * 100

                    trade["length_minutes"] = (
                        trade["exit_timestamp"] - trade["entry_timestamp"]
                    ).total_seconds() / 60

                    self.portfolio_balance += trade["profit"]

                    self.completed_trades.append(trade)
                    del self.open_trades[symbol]


            #####

            exited = [
                trade for trade in self.completed_trades
                if trade["exit_timestamp"] == timestamp
            ]

            if exited:
                self.equity_curve.append({
                    "timestamp": timestamp,
                    "equity": self.portfolio_balance,
                })

            candidates = []

            for _, row in bars.iterrows():

                symbol = row["symbol"]
                previous_high = self.last_bar_high.get(symbol, -np.inf)
                level = row["previous_30m_high"]

                if (
                    symbol not in self.open_trades
                    and passes_screener(row, self.params)
                    and not pd.isna(level)
                    and previous_high <= level
                    and row["high"] > level
                    and self.entry_windows.get(symbol) != row["timestamp"].floor("30min")
                ):
                    candidates.append(row)
                    self.entry_windows[symbol] = row["timestamp"].floor("30min")

                self.last_bar_high[symbol] = row["high"]

            #####
            if not candidates:
                continue

            pooled_pct = sum(
                row["session_pct_change"]
                for row in candidates
                if row["session_pct_change"] > 0
            )

            if pooled_pct <= 0:
                continue

            buying_power_before = self.buying_power

            for row in candidates:

                if row["session_pct_change"] <= 0:
                    continue

                contribution_pct = row["session_pct_change"] / pooled_pct * 100
                allocation = buying_power_before * contribution_pct / 100
                shares = int(allocation // row["close"])

                if shares <= 0:
                    continue

                cost = shares * row["close"]

                self.open_trades[row["symbol"]] = {
                    "entry_timestamp": timestamp,
                    "entry_price": row["close"],
                    "entry_session": row["session"],
                    "shares": shares,
                    "position_size": cost,
                    "contribution_pct": contribution_pct,
                    "pooled_session_pct": pooled_pct,
                    "buying_power_before": buying_power_before,
                    "peak_high": row["high"],
                    "entry_data": row.to_dict(),
                }

                self.buying_power -= cost


        self.daily_equity.append({
            "date": df["timestamp"].dt.date.iloc[0],
            "equity": self.portfolio_balance,
        })

        if len(self.daily_equity) > 1:
            prev = self.daily_equity[-2]["equity"]
            self.daily_returns.append(
                self.portfolio_balance / prev - 1
            )



    def stats(self):

        trades = pd.DataFrame(self.completed_trades)

        if trades.empty:
            return {}

        wins = trades["profit"] > 0
        losses = trades["profit"] < 0

        win_rate = wins.mean()
        avg_win = trades.loc[wins, "return_pct"].mean()
        avg_loss = trades.loc[losses, "return_pct"].mean()

        #####
        returns = np.asarray(self.daily_returns, dtype=np.float64)

        downside = returns[returns < 0]

        sortino = (
            returns.mean() / np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
            if downside.size else np.inf
        )

        sharpe = (
            returns.mean() / returns.std(ddof=1) * np.sqrt(252)
            if returns.size > 1 and returns.std(ddof=1) > 0 else np.inf
        )


        #####
        expectancy_pct = (
            win_rate * (0.0 if pd.isna(avg_win) else avg_win)
            - (1 - win_rate) * abs(0.0 if pd.isna(avg_loss) else avg_loss)
        )

        gross_profit = trades.loc[wins, "profit"].sum()
        gross_loss = -trades.loc[losses, "profit"].sum()

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0 else np.inf
        )

        equity = np.asarray(
            [x["equity"] for x in self.daily_equity],
            dtype=np.float64,
        )

        peaks = np.maximum.accumulate(equity)
        drawdowns = equity / peaks - 1

        max_drawdown = drawdowns.min() * 100


        #####
        days = max(
            (self.end_date - self.start_date).total_seconds() / 86400,
            1,
        )

        annualized_return = (
            (self.portfolio_balance / self.initial_capital)
            ** (365 / days) - 1
        ) * 100

        dd = drawdowns < 0
        max_dd_duration = 0
        current_dd = 0

        for x in dd:
            current_dd = current_dd + 1 if x else 0
            max_dd_duration = max(max_dd_duration, current_dd)
        return {
            "total_return_pct": (
                self.portfolio_balance / self.initial_capital - 1
            ) * 100,
            "total_trades": len(trades),
            "win_rate_pct": win_rate * 100,
            "wins": int(wins.sum()),
            "losses": int(losses.sum()),
            "avg_win_pct": 0.0 if pd.isna(avg_win) else avg_win,
            "avg_loss_pct": 0.0 if pd.isna(avg_loss) else avg_loss,
            "avg_trade_length_minutes": trades["length_minutes"].mean(),
            "sortino_ratio": sortino,
            "sharpe_ratio": sharpe,
            "trade_expectancy_pct": expectancy_pct,
            "trade_expectancy_dollar": trades["profit"].mean(),
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown,
            "annualized_return_pct": annualized_return,
            "max_drawdown_duration_days": max_dd_duration,
        }