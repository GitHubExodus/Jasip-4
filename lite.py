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


# ============================================================
# 3. FILE DISCOVERY
# ============================================================

BATCH_SIZE = 3

LEVEL = int(os.environ["LEVEL"])

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


# ============================================================
# SELECT STOCK BATCH
# ============================================================

stock_keys.sort()

start_index = (LEVEL - 1) * BATCH_SIZE
end_index = start_index + BATCH_SIZE

stock_keys = stock_keys[start_index:end_index]

print(
    f"LEVEL {LEVEL}: "
    f"processing stocks {start_index + 1:,} "
    f"to {min(end_index, start_index + len(stock_keys)):,}"
)

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
    # 7. CALCULATE SCREENER INPUTS
    # ============================================================

    # Trading-session date in New York time.
    # This prevents overnight/timezone differences from affecting
    # the daily calculations.
    df["session_date"] = df["timestamp"].dt.date

    # ------------------------------------------------------------
    # Daily volume
    # ------------------------------------------------------------
    #
    # Calculate total volume for each trading session, then create
    # the 10-session average used by the screener.
    #
    # min_periods=10 means a stock needs 10 completed sessions
    # before it can qualify. The warm-up period above supplies those
    # sessions without allowing pre-2025 data into the simulation.
    #

    daily_volume = (
        df.groupby("session_date", sort=True)["volume"]
        .sum()
        .rename("daily_volume")
    )

    daily_volume_df = daily_volume.to_frame()

    daily_volume_df["avg_10d_volume"] = (
        daily_volume_df["daily_volume"]
        .rolling(window=10, min_periods=10)
        .mean()
    )

    # ------------------------------------------------------------
    # Map the 10-day average volume back to every 1-minute bar
    # ------------------------------------------------------------

    df = df.merge(
        daily_volume_df[["avg_10d_volume"]],
        left_on="session_date",
        right_index=True,
        how="left",
    )

    # ------------------------------------------------------------
    # Current session cumulative volume
    # ------------------------------------------------------------

    df["session_volume"] = (
        df.groupby("session_date")["volume"]
        .cumsum()
    )

    # ------------------------------------------------------------
    # Relative volume
    # ------------------------------------------------------------
    #
    # Compare the current cumulative session volume against the
    # expected volume at the same point in the trading session.
    #
    # The calculation below uses the 10-session average daily
    # volume as the denominator and scales it by elapsed regular-
    # market minutes.
    #

    market_open_minutes = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
        - (9 * 60 + 30)
        + 1
    )

    market_open_minutes = market_open_minutes.clip(lower=1, upper=390)

    expected_volume = (
        df["avg_10d_volume"]
        * (market_open_minutes / 390.0)
    )

    df["relative_volume"] = (
        df["session_volume"] / expected_volume
    )

    # ------------------------------------------------------------
    # Previous candle high
    # ------------------------------------------------------------

    df["previous_high"] = df["high"].shift(1)

    # ------------------------------------------------------------
    # Keep only the actual 2025+ simulation period.
    #
    # The warm-up rows were needed to calculate the rolling inputs,
    # but they are not eligible to generate trades or be saved.
    # ------------------------------------------------------------

    df = df[df["timestamp"] >= DATA_START].copy()

    if df.empty:
        print(f"Skipping {symbol}: no data from 2025 onward")
        continue


    # ============================================================
    # 8. APPLY HARD-CODED SCREENER
    # ============================================================

    # Screener requirements:
    #
    # 1. Average 10-day volume > 1,000,000
    # 2. Close × average 10-day volume > 1,000,000
    # 3. Relative volume > 2
    # 4. Close price between $0.50 and $20.00 inclusive
    #
    # Every bar is evaluated independently.
    # Only bars passing ALL conditions are qualified.

    df["qualified"] = (
        (df["avg_10d_volume"] > 1_000_000)
        & (
            df["close"] * df["avg_10d_volume"]
            > 1_000_000
        )
        & (df["relative_volume"] > 2)
        & (df["close"] >= 0.50)
        & (df["close"] <= 20.00)
    )

    # ------------------------------------------------------------
    # Entry condition
    # ------------------------------------------------------------
    #
    # A trade opportunity exists only when:
    #
    #   - the current bar passes the screener
    #   - current high > previous 1-minute high
    #
    # The actual entry price is calculated later.
    #

    df["entry_signal"] = (
        df["qualified"]
        & df["previous_high"].notna()
        & (df["high"] > df["previous_high"])
    )

    qualified_count = int(df["qualified"].sum())
    entry_count = int(df["entry_signal"].sum())

    print(
        f"{symbol}: "
        f"{qualified_count:,} qualified bars, "
        f"{entry_count:,} entry signals"
    )




        # ============================================================
    # 9. IDENTIFY TRADING SESSIONS + SESSION FINAL BARS
    # ============================================================

    # The data has already been converted to New York time.
    # Therefore each session is determined from the New York date.

    # ------------------------------------------------------------
    # Identify the first and last bar of every session
    # ------------------------------------------------------------

    session_group = df.groupby("session_date", sort=False)

    df["session_first_timestamp"] = (
        session_group["timestamp"]
        .transform("first")
    )

    df["session_last_timestamp"] = (
        session_group["timestamp"]
        .transform("last")
    )

    # True only on the actual final bar available for that stock
    # on that trading session.
    df["is_session_final_bar"] = (
        df["timestamp"] == df["session_last_timestamp"]
    )

    # ------------------------------------------------------------
    # Session-open price
    # ------------------------------------------------------------
    #
    # Used later to calculate:
    #
    #   undelayed percentage change
    #
    # from the session's first bar open to the current bar close.
    #

    df["session_open"] = (
        session_group["open"]
        .transform("first")
    )

    # ------------------------------------------------------------
    # Highest price reached after an entry
    # ------------------------------------------------------------
    #
    # This column is NOT used to calculate the trailing stop yet.
    # It is created here as a placeholder for the trade simulation,
    # where the high reached after entry will be tracked bar by bar.
    #
    # Each trade will maintain its own independent highest price.
    # Therefore there is no portfolio-level state.
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Verify that session boundaries are valid
    # ------------------------------------------------------------

    invalid_sessions = df[
        df["session_first_timestamp"].isna()
        | df["session_last_timestamp"].isna()
        | df["session_open"].isna()
    ]

    if not invalid_sessions.empty:
        print(
            f"Warning: {symbol} has "
            f"{invalid_sessions['session_date'].nunique():,} "
            f"invalid session(s)"
        )

    session_count = df["session_date"].nunique()

    print(
        f"{symbol}: "
        f"{session_count:,} trading sessions identified"
    )


    # ============================================================
    # 10. TP / TSL CONFIGURATIONS
    # ============================================================

    # Every combination is simulated independently.
    #
    # There is:
    #   - no shared position
    #   - no shared capital
    #   - no buying power
    #   - no portfolio balance
    #   - no interaction between simulations
    #
    # TP and TSL values are stored as decimal percentages.

    TP_TSL_CONFIGS = [
        (0.05, 0.10),  # TP 5%,  TSL 10%
        (0.05, 0.05),  # TP 5%,  TSL 5%
        (0.10, 0.05),  # TP 10%, TSL 5%
        (0.10, 0.10),  # TP 10%, TSL 10%
        (0.20, 0.10),  # TP 20%, TSL 10%
        (0.20, 0.15),  # TP 20%, TSL 15%
        (0.30, 0.05),  # TP 30%, TSL 5%
        (0.30, 0.10),  # TP 30%, TSL 10%
        (0.30, 0.20),  # TP 30%, TSL 20%
    ]

    # ------------------------------------------------------------
    # History output directory
    # ------------------------------------------------------------

    HISTORY_DIR = "history"

    os.makedirs(HISTORY_DIR, exist_ok=True)

    print(
        f"{symbol}: "
        f"{len(TP_TSL_CONFIGS)} independent TP/TSL simulations"
    )


    # ============================================================
    # 11. RUN INDEPENDENT TP / TSL SIMULATIONS
    # ============================================================

    for tp_pct, tsl_pct in TP_TSL_CONFIGS:

        print(
            f"{symbol}: "
            f"running TP {tp_pct:.0%} / TSL {tsl_pct:.0%}"
        )

        # --------------------------------------------------------
        # Create history columns.
        #
        # Every screener-qualified bar is retained.
        # Trade columns remain empty unless that bar generates
        # an entry.
        # --------------------------------------------------------

        history = df[df["qualified"]].copy()

        history["trade_created"] = False
        history["entry_price"] = np.nan
        history["take_profit_price"] = np.nan
        history["trailing_stop_price"] = np.nan
        history["highest_price_after_entry"] = np.nan
        history["exit_price"] = np.nan
        history["trade_profit_pct"] = np.nan
        history["trade_hold_minutes"] = np.nan
        history["exit_reason"] = pd.NA

        # --------------------------------------------------------
        # Every qualifying entry is an independent trade.
        #
        # There is intentionally NO position dictionary, balance,
        # buying power, allocation, or portfolio state.
        # --------------------------------------------------------

        trade_records = []

        entry_indices = df.index[
            df["entry_signal"]
        ].tolist()

        # --------------------------------------------------------
        # Process each entry independently.
        # --------------------------------------------------------

        for entry_idx in entry_indices:

            entry_row = df.loc[entry_idx]

            entry_price = (
                entry_row["high"] * 1.0025
            )

            take_profit_price = (
                entry_price * (1.0 + tp_pct)
            )

            entry_session = entry_row["session_date"]

            # Only subsequent bars from the SAME trading session
            # are considered. A trade cannot continue into another
            # session.
            
            # ----------------------------------------------------
            # Only bars AFTER the entry candle can manage the trade.
            #
            # The entry occurs at:
            #
            #     current candle high × 1.0025
            #
            # Therefore the entry candle itself cannot trigger an
            # exit after the entry.
            # ----------------------------------------------------

            subsequent = df.loc[
                (df.index > entry_idx)
                & (df["session_date"] == entry_session)
            ].copy()

            highest_price = entry_price

            exit_price = np.nan
            exit_reason = None
            exit_timestamp = None

            # ----------------------------------------------------
            # Track the trade bar by bar.
            # ----------------------------------------------------

            for future_idx, future_row in subsequent.iterrows():

                bar_high = future_row["high"]
                bar_low = future_row["low"]

                # ------------------------------------------------
                # 1. If this is the actual final bar of the
                # session, liquidate at that bar's CLOSE.
                # ------------------------------------------------

                if future_row["is_session_final_bar"]:

                    exit_price = future_row["close"]
                    exit_reason = "SESSION_END_LIQUIDATION"
                    exit_timestamp = future_row["timestamp"]

                    break

                # ------------------------------------------------
                # 2. Update highest price reached after entry.
                # ------------------------------------------------

                if bar_high > highest_price:
                    highest_price = bar_high

                trailing_stop_price = (
                    highest_price * (1.0 - tsl_pct)
                )

                # ------------------------------------------------
                # 3. Take-profit
                # ------------------------------------------------

                if bar_high >= take_profit_price:

                    exit_price = take_profit_price
                    exit_reason = "TAKE_PROFIT"
                    exit_timestamp = future_row["timestamp"]

                    break

                # ------------------------------------------------
                # 4. Trailing stop
                # ------------------------------------------------

                if bar_low <= trailing_stop_price:

                    exit_price = trailing_stop_price
                    exit_reason = "TRAILING_STOP"
                    exit_timestamp = future_row["timestamp"]

                    break

            # ----------------------------------------------------
            # Safety fallback.
            #
            # This should only occur if the entry itself happens
            # on the final available bar, leaving no subsequent
            # bar to process.
            # ----------------------------------------------------

            if exit_reason is None:

                if entry_row["is_session_final_bar"]:

                    exit_price = entry_row["close"]
                    exit_reason = "SESSION_END_LIQUIDATION"
                    exit_timestamp = entry_row["timestamp"]

                else:
                    print(
                        f"Warning: {symbol} entry at "
                        f"{entry_row['timestamp']} "
                        f"did not receive an exit"
                    )

                    continue

            # ----------------------------------------------------
            # Trade calculations
            # ----------------------------------------------------

            trade_profit_pct = (
                (exit_price - entry_price)
                / entry_price
                * 100.0
            )

            hold_minutes = (
                (
                    exit_timestamp
                    - entry_row["timestamp"]
                ).total_seconds()
                / 60.0
            )

            # ----------------------------------------------------
            # Undelayed session performance
            #
            # From the actual session open to the current/exit
            # bar's close.
            # ----------------------------------------------------

            session_open = entry_row["session_open"]

            undelayed_pct_change = (
                (entry_row["close"] - session_open)
                / session_open
                * 100.0
            )

            # ----------------------------------------------------
            # Store the complete trade record.
            # ----------------------------------------------------

            trade_records.append(
                {
                    "entry_index": entry_idx,
                    "entry_timestamp": entry_row["timestamp"],
                    "entry_price": entry_price,
                    "take_profit_price": take_profit_price,
                    "trailing_stop_price": (
                        highest_price * (1.0 - tsl_pct)
                    ),
                    "highest_price_after_entry": highest_price,
                    "exit_timestamp": exit_timestamp,
                    "exit_price": exit_price,
                    "trade_profit_pct": trade_profit_pct,
                    "trade_hold_minutes": hold_minutes,
                    "exit_reason": exit_reason,
                    "undelayed_pct_change": (
                        undelayed_pct_change
                    ),
                }
            )

        # --------------------------------------------------------
        # Attach trade information to the qualified entry bars.
        # --------------------------------------------------------

        for trade in trade_records:

            idx = trade["entry_index"]

            history.loc[idx, "trade_created"] = True
            history.loc[idx, "entry_price"] = trade["entry_price"]
            history.loc[idx, "take_profit_price"] = (
                trade["take_profit_price"]
            )
            history.loc[idx, "trailing_stop_price"] = (
                trade["trailing_stop_price"]
            )
            history.loc[idx, "highest_price_after_entry"] = (
                trade["highest_price_after_entry"]
            )
            history.loc[idx, "exit_price"] = trade["exit_price"]
            history.loc[idx, "trade_profit_pct"] = (
                trade["trade_profit_pct"]
            )
            history.loc[idx, "trade_hold_minutes"] = (
                trade["trade_hold_minutes"]
            )
            history.loc[idx, "exit_reason"] = (
                trade["exit_reason"]
            )

        print(
            f"{symbol}: "
            f"TP {tp_pct:.0%} / TSL {tsl_pct:.0%} -> "
            f"{len(trade_records):,} trades"
        )


        # ========================================================
        # 12. FINALIZE + SAVE HISTORY FILE
        # ========================================================

        # --------------------------------------------------------
        # Add stock and simulation identifiers.
        # --------------------------------------------------------

        history["symbol"] = symbol
        history["tp_pct"] = tp_pct * 100.0
        history["tsl_pct"] = tsl_pct * 100.0

        # --------------------------------------------------------
        # Undelayed percentage change for EVERY qualified bar.
        #
        # This is based on:
        #
        #     session open -> current bar close
        #
        # It is independent of whether the bar generated a trade.
        # --------------------------------------------------------

        history["undelayed_pct_change"] = (
            (
                history["close"]
                - history["session_open"]
            )
            / history["session_open"]
            * 100.0
        )

        # --------------------------------------------------------
        # Explicitly identify whether the qualified bar generated
        # an entry.
        # --------------------------------------------------------

        history["entry_signal"] = history["entry_signal"].astype(bool)

        # --------------------------------------------------------
        # Select the columns that belong in the history.
        #
        # Each row represents one screener-qualified 1-minute bar.
        # --------------------------------------------------------

        HISTORY_COLUMNS = [
            "symbol",
            "timestamp",

            "open",
            "high",
            "low",
            "close",
            "volume",

            "session_date",
            "session_open",
            "session_first_timestamp",
            "session_last_timestamp",
            "is_session_final_bar",

            "avg_10d_volume",
            "session_volume",
            "relative_volume",

            "previous_high",
            "qualified",
            "entry_signal",

            "tp_pct",
            "tsl_pct",

            "trade_created",
            "entry_price",
            "take_profit_price",
            "trailing_stop_price",
            "highest_price_after_entry",
            "exit_price",
            "trade_profit_pct",
            "trade_hold_minutes",
            "exit_reason",

            "undelayed_pct_change",
        ]

        history = history[HISTORY_COLUMNS].copy()

        # --------------------------------------------------------
        # Sort chronologically.
        # --------------------------------------------------------

        history = history.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # --------------------------------------------------------
        # Construct the required filename.
        #
        # Examples:
        #
        # AAPL_tp5_tsl10.parquet
        # AAPL_tp20_tsl15.parquet
        # --------------------------------------------------------

        tp_label = int(tp_pct * 100)
        tsl_label = int(tsl_pct * 100)

        output_filename = (
            f"{symbol}_tp{tp_label}_tsl{tsl_label}.parquet"
        )

        output_path = os.path.join(
            HISTORY_DIR,
            output_filename,
        )

        # --------------------------------------------------------
        # Save the independent history.
        # --------------------------------------------------------

        history.to_parquet(
            output_path,
            engine="pyarrow",
            index=False,
        )

        print(
            f"Saved: {output_path} "
            f"({len(history):,} qualified bars)"
        )

        # --------------------------------------------------------
        # Release the history dataframe before moving to the next
        # TP/TSL configuration.
        # --------------------------------------------------------

        del history
        gc.collect()



    # ============================================================
    # 13. FINISH STOCK PROCESSING
    # ============================================================

    # Release the stock dataframe before moving to the next stock.
    del df
    gc.collect()

    print(
        f"Finished {symbol}"
    )
    print("-" * 60)

print("All stocks processed.")