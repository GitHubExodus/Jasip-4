# ============================================================
# 1. CONFIGURATION
# ============================================================

import os
import io
import gc

import boto3
import numpy as np
import pandas as pd
import numba


R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]

R2_BUCKET_NAME = "stocks-data"

R2_ENDPOINT = (
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
)

TIMEZONE = "America/New_York"

DATA_START = pd.Timestamp(
    "2025-01-01",
    tz=TIMEZONE,
)

DATA_END = pd.Timestamp.now(
    tz=TIMEZONE,
)

WARMUP_DAYS = 20

MAX_STOCKS = 3
# Set to None to process all available stocks.

HISTORY_PREFIX = "history"

RAW_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

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
# 3. NUMBA TRADE SIMULATION
# ============================================================

@numba.njit(cache=True)
def simulate_trades(
    high,
    low,
    close,
    entry_signal,
    session_id,
    session_final,
    session_open,
    tp_pct,
    tsl_pct,
):
    """
    Simulate every entry independently.

    There is no:
        - buying power
        - balance
        - allocation
        - portfolio
        - equity curve
        - interaction between trades

    Each qualifying entry is its own independent trade.

    Exit reason codes:

        1 = TAKE_PROFIT
        2 = TRAILING_STOP
        3 = SESSION_END_LIQUIDATION
        0 = no trade
    """

    n = len(high)

    trade_created = np.zeros(n, dtype=np.bool_)

    entry_price_out = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    take_profit_price_out = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    trailing_stop_price_out = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    highest_price_out = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    exit_price_out = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    trade_profit_pct_out = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    trade_hold_minutes_out = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    exit_reason_out = np.zeros(
        n,
        dtype=np.int8,
    )

    exit_index_out = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    n_trades = 0

    for i in range(n):

        if not entry_signal[i]:
            continue

        # --------------------------------------------------------
        # Entry
        # --------------------------------------------------------

        entry_price = high[i] * 1.0025

        take_profit_price = (
            entry_price * (1.0 + tp_pct)
        )

        highest_price = entry_price

        exit_price = np.nan
        exit_index = -1
        exit_reason = 0

        # --------------------------------------------------------
        # The entry candle itself is NOT used to manage the trade.
        #
        # Management starts with the next bar.
        # --------------------------------------------------------

        j = i + 1

        while j < n:

            # A trade cannot cross into another session.
            if session_id[j] != session_id[i]:
                break

            # ----------------------------------------------------
            # SESSION END LIQUIDATION
            #
            # If this is the actual final bar available for this
            # stock/session, liquidate at THIS BAR'S CLOSE.
            #
            # This takes precedence over TP/TSL because the trade
            # has reached the end of its trading session.
            # ----------------------------------------------------

            if session_final[j]:

                exit_price = close[j]
                exit_index = j
                exit_reason = 3

                break

            bar_high = high[j]
            bar_low = low[j]

            # ----------------------------------------------------
            # Update highest price reached after entry.
            # ----------------------------------------------------

            if bar_high > highest_price:
                highest_price = bar_high

            trailing_stop_price = (
                highest_price * (1.0 - tsl_pct)
            )

            # ----------------------------------------------------
            # Take profit
            # ----------------------------------------------------

            if bar_high >= take_profit_price:

                exit_price = take_profit_price
                exit_index = j
                exit_reason = 1

                break

            # ----------------------------------------------------
            # Trailing stop
            #
            # HIGH updates the trailing stop.
            # LOW triggers the trailing stop.
            # ----------------------------------------------------

            if bar_low <= trailing_stop_price:

                exit_price = trailing_stop_price
                exit_index = j
                exit_reason = 2

                break

            j += 1

        # --------------------------------------------------------
        # Entry on the actual final bar.
        #
        # There is no subsequent bar, so the trade is liquidated
        # using that final bar's close.
        # --------------------------------------------------------

        if exit_reason == 0 and session_final[i]:

            exit_price = close[i]
            exit_index = i
            exit_reason = 3

        # --------------------------------------------------------
        # Record trade
        # --------------------------------------------------------

        if exit_reason != 0:

            trade_created[i] = True

            entry_price_out[i] = entry_price

            take_profit_price_out[i] = (
                take_profit_price
            )

            highest_price_out[i] = highest_price

            if exit_reason == 1:

                trailing_stop_price_out[i] = (
                    highest_price
                    * (1.0 - tsl_pct)
                )

            elif exit_reason == 2:

                trailing_stop_price_out[i] = (
                    exit_price
                )

            else:

                trailing_stop_price_out[i] = (
                    highest_price
                    * (1.0 - tsl_pct)
                )

            exit_price_out[i] = exit_price

            trade_profit_pct_out[i] = (
                (exit_price - entry_price)
                / entry_price
                * 100.0
            )

            trade_hold_minutes_out[i] = (
                (exit_index - i)
                # Timestamp differences are calculated outside
                # Numba because timestamps are timezone-aware.
            )

            exit_reason_out[i] = exit_reason
            exit_index_out[i] = exit_index

            n_trades += 1

    return (
        trade_created,
        entry_price_out,
        take_profit_price_out,
        trailing_stop_price_out,
        highest_price_out,
        exit_price_out,
        trade_profit_pct_out,
        trade_hold_minutes_out,
        exit_reason_out,
        exit_index_out,
        n_trades,
    )


# ============================================================
# 4. FILE DISCOVERY
# ============================================================

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

        if (
            "/" not in key
            and key.lower().endswith(".parquet")
        ):
            stock_keys.append(key)

    if not response.get("IsTruncated"):
        break

    continuation_token = (
        response["NextContinuationToken"]
    )

if MAX_STOCKS is not None:

    stock_keys = stock_keys[:MAX_STOCKS]

print(
    f"Found {len(stock_keys):,} stock files to process"
)


# ============================================================
# 5. PROCESS EACH STOCK
# ============================================================

for key in stock_keys:

    symbol = (
        key
        .rsplit("/", 1)[-1]
        .replace(".parquet", "")
        .upper()
    )

    print()
    print("=" * 60)
    print(f"Processing {symbol}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load stock data directly from R2
    # --------------------------------------------------------

    obj = s3.get_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
    )

    raw_bytes = obj["Body"].read()

    df = pd.read_parquet(
        io.BytesIO(raw_bytes),
        columns=RAW_COLUMNS,
    )

    del raw_bytes
    gc.collect()

    # ========================================================
    # 6. CLEAN DATA + NEW YORK TIME
    # ========================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df["open"] = df["open"].astype(
        np.float64
    )

    df["high"] = df["high"].astype(
        np.float64
    )

    df["low"] = df["low"].astype(
        np.float64
    )

    df["close"] = df["close"].astype(
        np.float64
    )

    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=RAW_COLUMNS)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Convert all timestamps to New York time.
    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_convert(TIMEZONE)
    )

    # ========================================================
    # 7. WARM-UP + AVAILABLE DATE RANGE
    # ========================================================

    warmup_start = (
        DATA_START
        - pd.Timedelta(days=WARMUP_DAYS)
    )

    df = df[
        (df["timestamp"] >= warmup_start)
        & (df["timestamp"] <= DATA_END)
    ].copy()

    if df.empty:

        print(
            f"Skipping {symbol}: no usable data"
        )

        del df
        gc.collect()

        continue

    # ========================================================
    # 8. SESSION INFORMATION
    # ========================================================

    # Session date is based entirely on New York time.

    df["session_date"] = (
        df["timestamp"].dt.date
    )

    # Create a numeric session ID for Numba.

    session_codes, _ = pd.factorize(
        df["session_date"],
        sort=False,
    )

    df["session_id"] = (
        session_codes.astype(np.int64)
    )

    session_group = df.groupby(
        "session_date",
        sort=False,
    )

    # --------------------------------------------------------
    # Session first/last timestamps
    # --------------------------------------------------------

    df["session_first_timestamp"] = (
        session_group["timestamp"]
        .transform("first")
    )

    df["session_last_timestamp"] = (
        session_group["timestamp"]
        .transform("last")
    )

    # --------------------------------------------------------
    # Actual final bar of each session
    # --------------------------------------------------------

    df["is_session_final_bar"] = (
        df["timestamp"]
        == df["session_last_timestamp"]
    )

    # --------------------------------------------------------
    # Session open
    # --------------------------------------------------------

    df["session_open"] = (
        session_group["open"]
        .transform("first")
    )

    # ========================================================
    # 9. DAILY VOLUME + 10-DAY AVERAGE
    # ========================================================

    daily_volume = (
        df.groupby(
            "session_date",
            sort=True,
        )["volume"]
        .sum()
        .rename("daily_volume")
    )

    daily_volume_df = (
        daily_volume.to_frame()
    )

    # This is an estimation using the rolling 10-session
    # volume, including the current session.

    daily_volume_df[
        "avg_10d_volume"
    ] = (
        daily_volume_df["daily_volume"]
        .rolling(
            window=10,
            min_periods=10,
        )
        .mean()
    )

    # Map the daily estimate back to every minute.

    df = df.merge(
        daily_volume_df[
            ["avg_10d_volume"]
        ],
        left_on="session_date",
        right_index=True,
        how="left",
    )

    # ========================================================
    # 10. CURRENT SESSION VOLUME
    # ========================================================

    df["session_volume"] = (
        df.groupby("session_date")["volume"]
        .cumsum()
    )

    # ========================================================
    # 11. ESTIMATED RELATIVE VOLUME
    # ========================================================

    market_open_minutes = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
        - (9 * 60 + 30)
        + 1
    )

    market_open_minutes = (
        market_open_minutes
        .clip(lower=1, upper=390)
    )

    expected_volume = (
        df["avg_10d_volume"]
        * (market_open_minutes / 390.0)
    )

    df["relative_volume"] = (
        df["session_volume"]
        / expected_volume
    )

    # ========================================================
    # 12. PREVIOUS 1-MINUTE HIGH
    # ========================================================

    # Previous high is calculated within each stock's session,
    # so the first bar of a session does not compare against the
    # previous day's final candle.

    df["previous_high"] = (
        df.groupby("session_date")["high"]
        .shift(1)
    )

    # ========================================================
    # 13. KEEP ONLY 2025+ DATA FOR SIMULATION
    # ========================================================

    df = df[
        df["timestamp"] >= DATA_START
    ].copy()

    if df.empty:

        print(
            f"Skipping {symbol}: "
            f"no data from 2025 onward"
        )

        del df
        gc.collect()

        continue

    # Recreate contiguous index after filtering.

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ========================================================
    # 14. APPLY HARD-CODED SCREENER
    # ========================================================

    df["qualified"] = (
        (df["avg_10d_volume"] > 1_000_000)
        &
        (
            df["close"]
            * df["avg_10d_volume"]
            > 1_000_000
        )
        &
        (df["relative_volume"] > 2)
        &
        (df["close"] >= 0.50)
        &
        (df["close"] <= 20.00)
    )

    # ========================================================
    # 15. ENTRY CONDITION
    # ========================================================

    df["entry_signal"] = (
        df["qualified"]
        &
        df["previous_high"].notna()
        &
        (
            df["high"]
            > df["previous_high"]
        )
    )

    qualified_count = int(
        df["qualified"].sum()
    )

    entry_count = int(
        df["entry_signal"].sum()
    )

    print(
        f"{symbol}: "
        f"{qualified_count:,} qualified bars, "
        f"{entry_count:,} entry signals"
    )

    # ========================================================
    # 16. SKIP STOCKS WITH NOTHING TO SIMULATE
    # ========================================================

    if qualified_count == 0:

        print(
            f"Skipping {symbol}: "
            f"0 qualified bars"
        )

        del df
        gc.collect()

        continue

    if entry_count == 0:

        print(
            f"Skipping {symbol}: "
            f"0 entry signals"
        )

        del df
        gc.collect()

        continue

    # ========================================================
    # 17. PREPARE NUMBA ARRAYS
    # ========================================================

    high_array = (
        df["high"]
        .to_numpy(dtype=np.float64)
    )

    low_array = (
        df["low"]
        .to_numpy(dtype=np.float64)
    )

    close_array = (
        df["close"]
        .to_numpy(dtype=np.float64)
    )

    entry_signal_array = (
        df["entry_signal"]
        .to_numpy(dtype=np.bool_)
    )

    session_id_array = (
        df["session_id"]
        .to_numpy(dtype=np.int64)
    )

    session_final_array = (
        df["is_session_final_bar"]
        .to_numpy(dtype=np.bool_)
    )

    session_open_array = (
        df["session_open"]
        .to_numpy(dtype=np.float64)
    )

    timestamps = (
        df["timestamp"]
        .reset_index(drop=True)
    )

    # ========================================================
    # 18. CREATE HISTORY BASE
    # ========================================================

    history_base = df[
        df["qualified"]
    ].copy()

    history_base = (
        history_base
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Keep the qualified rows' original dataframe indices
    # so Numba results can be attached to the correct bars.

    qualified_indices = (
        df.index[
            df["qualified"]
        ]
        .to_numpy(dtype=np.int64)
    )

    # ========================================================
    # 19. RUN ALL NINE INDEPENDENT TP / TSL SIMULATIONS
    # ========================================================

    for tp_pct, tsl_pct in TP_TSL_CONFIGS:

        print(
            f"{symbol}: "
            f"running TP {tp_pct:.0%} / "
            f"TSL {tsl_pct:.0%}"
        )

        (
            trade_created,
            entry_price,
            take_profit_price,
            trailing_stop_price,
            highest_price,
            exit_price,
            trade_profit_pct,
            trade_hold_minutes_index,
            exit_reason_code,
            exit_index,
            n_trades,
        ) = simulate_trades(
            high_array,
            low_array,
            close_array,
            entry_signal_array,
            session_id_array,
            session_final_array,
            session_open_array,
            tp_pct,
            tsl_pct,
        )

        # ----------------------------------------------------
        # Create history from EVERY qualified bar.
        # ----------------------------------------------------

        history = history_base.copy()

        # ----------------------------------------------------
        # Add stock/simulation identifiers.
        # ----------------------------------------------------

        history["symbol"] = symbol

        history["tp_pct"] = (
            tp_pct * 100.0
        )

        history["tsl_pct"] = (
            tsl_pct * 100.0
        )

        # ----------------------------------------------------
        # Trade fields for qualified bars.
        #
        # First create full-length arrays, then select the
        # qualified rows.
        # ----------------------------------------------------

        history["trade_created"] = (
            trade_created[
                qualified_indices
            ]
        )

        history["entry_price"] = (
            entry_price[
                qualified_indices
            ]
        )

        history["take_profit_price"] = (
            take_profit_price[
                qualified_indices
            ]
        )

        history["trailing_stop_price"] = (
            trailing_stop_price[
                qualified_indices
            ]
        )

        history["highest_price_after_entry"] = (
            highest_price[
                qualified_indices
            ]
        )

        history["exit_price"] = (
            exit_price[
                qualified_indices
            ]
        )

        history["trade_profit_pct"] = (
            trade_profit_pct[
                qualified_indices
            ]
        )

        # ----------------------------------------------------
        # Calculate actual holding minutes from timestamps.
        #
        # Numba stores the exit bar index, then pandas calculates
        # the real timestamp difference.
        # ----------------------------------------------------

        qualified_exit_indices = (
            exit_index[
                qualified_indices
            ]
        )

        hold_minutes = np.full(
            len(history),
            np.nan,
            dtype=np.float64,
        )

        entry_timestamps = (
            history["timestamp"]
            .reset_index(drop=True)
        )

        for row_number, exit_idx in enumerate(
            qualified_exit_indices
        ):

            if exit_idx >= 0:

                exit_timestamp = (
                    timestamps.iloc[exit_idx]
                )

                entry_timestamp = (
                    entry_timestamps.iloc[row_number]
                )

                hold_minutes[row_number] = (
                    (
                        exit_timestamp
                        - entry_timestamp
                    ).total_seconds()
                    / 60.0
                )

        history["trade_hold_minutes"] = (
            hold_minutes
        )

        # ----------------------------------------------------
        # Exit reason
        # ----------------------------------------------------

        reason_values = (
            exit_reason_code[
                qualified_indices
            ]
        )

        exit_reason = np.full(
            len(history),
            None,
            dtype=object,
        )

        exit_reason[
            reason_values == 1
        ] = "TAKE_PROFIT"

        exit_reason[
            reason_values == 2
        ] = "TRAILING_STOP"

        exit_reason[
            reason_values == 3
        ] = "SESSION_END_LIQUIDATION"

        history["exit_reason"] = (
            exit_reason
        )

        # ----------------------------------------------------
        # Undelayed percentage change.
        #
        # Session open -> CURRENT BAR CLOSE.
        #
        # This is calculated for every qualified bar, regardless
        # of whether it generated a trade.
        # ----------------------------------------------------

        history["undelayed_pct_change"] = (
            (
                history["close"]
                - history["session_open"]
            )
            / history["session_open"]
            * 100.0
        )

        # ----------------------------------------------------
        # Select final history columns.
        # ----------------------------------------------------

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

        history = (
            history[HISTORY_COLUMNS]
            .copy()
        )

        history = (
            history
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # ====================================================
        # 20. SAVE HISTORY DIRECTLY TO R2
        # ====================================================

        tp_label = int(
            tp_pct * 100
        )

        tsl_label = int(
            tsl_pct * 100
        )

        output_filename = (
            f"{symbol}_tp{tp_label}_tsl{tsl_label}.parquet"
        )

        output_key = (
            f"{HISTORY_PREFIX}/"
            f"{output_filename}"
        )

        # Serialize Parquet directly into memory.
        # Nothing is written to the laptop's disk.

        parquet_buffer = io.BytesIO()

        history.to_parquet(
            parquet_buffer,
            engine="pyarrow",
            index=False,
        )

        parquet_buffer.seek(0)

        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=output_key,
            Body=parquet_buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        print(
            f"Saved to R2: "
            f"{output_key} "
            f"({len(history):,} qualified bars, "
            f"{n_trades:,} trades)"
        )

        # ----------------------------------------------------
        # Release this simulation's history.
        # ----------------------------------------------------

        del history
        del parquet_buffer

        gc.collect()

    # ========================================================
    # 21. FINISH STOCK
    # ========================================================

    del history_base
    del high_array
    del low_array
    del close_array
    del entry_signal_array
    del session_id_array
    del session_final_array
    del session_open_array
    del qualified_indices
    del timestamps
    del df

    gc.collect()

    print(
        f"Finished {symbol}"
    )


# ============================================================
# 22. FINISH
# ============================================================

print()
print("All stocks processed.")