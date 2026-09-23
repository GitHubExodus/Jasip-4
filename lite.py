# ============================================================
# 1. CONFIGURATION
# ============================================================

import os
import io
import gc

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit


R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]

R2_BUCKET_NAME = "stocks-data"

R2_ENDPOINT = (
    "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"
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

RAW_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

BATCH_SIZE = 3

LEVEL = int(os.environ["LEVEL"])

HISTORY_PREFIX = "history"

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

@njit(cache=True)
def simulate_trades(
    entry_indices,
    timestamps_ns,
    session_ids,
    session_final_flags,
    close_prices,
    high_prices,
    low_prices,
    session_open_prices,
    tp_pct,
    tsl_pct,
):
    """
    Simulates every entry independently.

    There is no portfolio state.

    Each entry gets its own:
        - entry price
        - TP
        - highest price
        - trailing stop
        - exit
        - profit
        - holding time

    Exit priority:

        1. SESSION_END_LIQUIDATION
        2. TAKE_PROFIT
        3. TRAILING_STOP

    The session final bar is therefore always liquidated
    at that bar's CLOSE if the trade survives until it.
    """

    n_entries = len(entry_indices)

    entry_price_out = np.full(
        n_entries,
        np.nan,
        dtype=np.float64,
    )

    take_profit_out = np.full(
        n_entries,
        np.nan,
        dtype=np.float64,
    )

    trailing_stop_out = np.full(
        n_entries,
        np.nan,
        dtype=np.float64,
    )

    highest_price_out = np.full(
        n_entries,
        np.nan,
        dtype=np.float64,
    )

    exit_timestamp_out = np.full(
        n_entries,
        np.int64(-1),
        dtype=np.int64,
    )

    exit_price_out = np.full(
        n_entries,
        np.nan,
        dtype=np.float64,
    )

    profit_pct_out = np.full(
        n_entries,
        np.nan,
        dtype=np.float64,
    )

    hold_minutes_out = np.full(
        n_entries,
        np.nan,
        dtype=np.float64,
    )

    exit_reason_out = np.full(
        n_entries,
        np.int8(-1),
        dtype=np.int8,
    )

    undelayed_pct_out = np.full(
        n_entries,
        np.nan,
        dtype=np.float64,
    )

    for trade_number in range(n_entries):

        entry_idx = entry_indices[trade_number]

        entry_price = (
            high_prices[entry_idx] * 1.0025
        )

        take_profit_price = (
            entry_price * (1.0 + tp_pct)
        )

        entry_session = session_ids[entry_idx]

        highest_price = entry_price

        exited = False

        for future_idx in range(
            entry_idx + 1,
            len(high_prices),
        ):

            # A trade never crosses into another session.
            if session_ids[future_idx] != entry_session:
                break

            bar_high = high_prices[future_idx]
            bar_low = low_prices[future_idx]

            # ----------------------------------------------------
            # SESSION-END LIQUIDATION
            #
            # The actual final bar of the session always closes
            # the trade at that bar's CLOSE.
            # ----------------------------------------------------

            if session_final_flags[future_idx]:

                exit_price = close_prices[future_idx]

                entry_price_out[trade_number] = entry_price
                take_profit_out[trade_number] = take_profit_price
                trailing_stop_out[trade_number] = (
                    highest_price * (1.0 - tsl_pct)
                )
                highest_price_out[trade_number] = highest_price

                exit_timestamp_out[trade_number] = (
                    timestamps_ns[future_idx]
                )

                exit_price_out[trade_number] = exit_price

                profit_pct_out[trade_number] = (
                    (exit_price - entry_price)
                    / entry_price
                    * 100.0
                )

                hold_minutes_out[trade_number] = (
                    (
                        timestamps_ns[future_idx]
                        - timestamps_ns[entry_idx]
                    )
                    / 60_000_000_000.0
                )

                exit_reason_out[trade_number] = 2

                undelayed_pct_out[trade_number] = (
                    (
                        close_prices[entry_idx]
                        - session_open_prices[entry_idx]
                    )
                    / session_open_prices[entry_idx]
                    * 100.0
                )

                exited = True
                break

            # ----------------------------------------------------
            # UPDATE HIGHEST PRICE
            # ----------------------------------------------------

            if bar_high > highest_price:
                highest_price = bar_high

            trailing_stop_price = (
                highest_price * (1.0 - tsl_pct)
            )

            # ----------------------------------------------------
            # TAKE PROFIT
            # ----------------------------------------------------

            if bar_high >= take_profit_price:

                exit_price = take_profit_price

                entry_price_out[trade_number] = entry_price
                take_profit_out[trade_number] = take_profit_price
                trailing_stop_out[trade_number] = (
                    trailing_stop_price
                )
                highest_price_out[trade_number] = highest_price

                exit_timestamp_out[trade_number] = (
                    timestamps_ns[future_idx]
                )

                exit_price_out[trade_number] = exit_price

                profit_pct_out[trade_number] = (
                    (exit_price - entry_price)
                    / entry_price
                    * 100.0
                )

                hold_minutes_out[trade_number] = (
                    (
                        timestamps_ns[future_idx]
                        - timestamps_ns[entry_idx]
                    )
                    / 60_000_000_000.0
                )

                exit_reason_out[trade_number] = 0

                undelayed_pct_out[trade_number] = (
                    (
                        close_prices[entry_idx]
                        - session_open_prices[entry_idx]
                    )
                    / session_open_prices[entry_idx]
                    * 100.0
                )

                exited = True
                break

            # ----------------------------------------------------
            # TRAILING STOP
            #
            # HIGH updates the trailing-stop level.
            # LOW triggers the trailing stop.
            # ----------------------------------------------------

            if bar_low <= trailing_stop_price:

                exit_price = trailing_stop_price

                entry_price_out[trade_number] = entry_price
                take_profit_out[trade_number] = take_profit_price
                trailing_stop_out[trade_number] = (
                    trailing_stop_price
                )
                highest_price_out[trade_number] = highest_price

                exit_timestamp_out[trade_number] = (
                    timestamps_ns[future_idx]
                )

                exit_price_out[trade_number] = exit_price

                profit_pct_out[trade_number] = (
                    (exit_price - entry_price)
                    / entry_price
                    * 100.0
                )

                hold_minutes_out[trade_number] = (
                    (
                        timestamps_ns[future_idx]
                        - timestamps_ns[entry_idx]
                    )
                    / 60_000_000_000.0
                )

                exit_reason_out[trade_number] = 1

                undelayed_pct_out[trade_number] = (
                    (
                        close_prices[entry_idx]
                        - session_open_prices[entry_idx]
                    )
                    / session_open_prices[entry_idx]
                    * 100.0
                )

                exited = True
                break

        # --------------------------------------------------------
        # ENTRY OCCURS ON THE FINAL SESSION BAR
        #
        # There is no subsequent bar, so the trade is liquidated
        # at that bar's close.
        # --------------------------------------------------------

        if not exited:

            if session_final_flags[entry_idx]:

                exit_price = close_prices[entry_idx]

                entry_price_out[trade_number] = entry_price
                take_profit_out[trade_number] = take_profit_price
                trailing_stop_out[trade_number] = (
                    entry_price * (1.0 - tsl_pct)
                )
                highest_price_out[trade_number] = entry_price

                exit_timestamp_out[trade_number] = (
                    timestamps_ns[entry_idx]
                )

                exit_price_out[trade_number] = exit_price

                profit_pct_out[trade_number] = (
                    (exit_price - entry_price)
                    / entry_price
                    * 100.0
                )

                hold_minutes_out[trade_number] = 0.0

                exit_reason_out[trade_number] = 2

                undelayed_pct_out[trade_number] = (
                    (
                        close_prices[entry_idx]
                        - session_open_prices[entry_idx]
                    )
                    / session_open_prices[entry_idx]
                    * 100.0
                )

    return (
        entry_price_out,
        take_profit_out,
        trailing_stop_out,
        highest_price_out,
        exit_timestamp_out,
        exit_price_out,
        profit_pct_out,
        hold_minutes_out,
        exit_reason_out,
        undelayed_pct_out,
    )


# ============================================================
# 4. STOCK DISCOVERY + BATCHING
# ============================================================

response = s3.get_object(
    Bucket=R2_BUCKET_NAME,
    Key="misc/symbols_full.txt",
)

text = response["Body"].read().decode("utf-8")

stock_symbols = [
    line.strip().upper()
    for line in text.splitlines()
    if line.strip()
]

start_index = (
    LEVEL - 1
) * BATCH_SIZE

end_index = (
    start_index + BATCH_SIZE
)

stock_symbols = stock_symbols[
    start_index:end_index
]

print(
    f"LEVEL {LEVEL}: "
    f"processing stocks "
    f"{start_index + 1:,} "
    f"to "
    f"{start_index + len(stock_symbols):,}"
)

print(
    f"Found {len(stock_symbols):,} "
    f"stock files to process"
)


# ============================================================
# 5. PROCESS EACH STOCK
# ============================================================

for symbol in stock_symbols:

    print("=" * 70)
    print(f"Processing {symbol}...")

    # --------------------------------------------------------
    # Load stock data directly from R2.
    # --------------------------------------------------------

    try:

        obj = s3.get_object(
            Bucket=R2_BUCKET_NAME,
            Key=f"{symbol}.parquet",
        )

    except Exception as exc:

        print(
            f"Skipping {symbol}: "
            f"could not load stock data: {exc}"
        )

        continue

    df = pd.read_parquet(
        io.BytesIO(obj["Body"].read()),
        columns=RAW_COLUMNS,
    )

    del obj
    gc.collect()

    # ========================================================
    # 6. CLEAN DATA + NEW YORK TIMEZONE
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

    df["timestamp"] = (
        df["timestamp"]
        .dt
        .tz_convert(TIMEZONE)
    )

    # ========================================================
    # 7. WARM-UP + DATE RANGE
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
            f"Skipping {symbol}: "
            f"no usable data"
        )

        del df
        gc.collect()

        continue

    if df["timestamp"].max() < DATA_START:

        print(
            f"Skipping {symbol}: "
            f"no data from 2025 onward"
        )

        del df
        gc.collect()

        continue

    # ========================================================
    # 8. SESSION IDENTIFICATION
    # ========================================================

    df["session_date"] = (
        df["timestamp"].dt.date
    )

    # Numeric session IDs are much more efficient for Numba.

    session_codes, _ = pd.factorize(
        df["session_date"],
        sort=True,
    )

    df["session_id"] = (
        session_codes.astype(np.int64)
    )

    # ========================================================
    # 9. DAILY VOLUME
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

    daily_volume_df["avg_10d_volume"] = (
        daily_volume_df["daily_volume"]
        .rolling(
            window=10,
            min_periods=10,
        )
        .mean()
    )

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
        df.groupby(
            "session_date"
        )["volume"]
        .cumsum()
    )

    # ========================================================
    # 11. RELATIVE VOLUME ESTIMATION
    # ========================================================

    market_open_minutes = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
        - (9 * 60 + 30)
        + 1
    )

    market_open_minutes = (
        market_open_minutes
        .clip(
            lower=1,
            upper=390,
        )
    )

    expected_volume = (
        df["avg_10d_volume"]
        * (
            market_open_minutes
            / 390.0
        )
    )

    df["relative_volume"] = (
        df["session_volume"]
        / expected_volume
    )

    # ========================================================
    # 12. PREVIOUS 1-MINUTE HIGH
    # ========================================================
    #
    # Previous candle must belong to the same trading session.
    #

    previous_high = (
        df["high"]
        .shift(1)
    )

    previous_session = (
        df["session_id"]
        .shift(1)
    )

    df["previous_high"] = np.where(
        df["session_id"].to_numpy()
        == previous_session.to_numpy(),
        previous_high.to_numpy(),
        np.nan,
    )

    # ========================================================
    # 13. SESSION OPEN
    # ========================================================

    session_group = (
        df.groupby(
            "session_date",
            sort=False,
        )
    )

    df["session_open"] = (
        session_group["open"]
        .transform("first")
    )

    df["session_first_timestamp"] = (
        session_group["timestamp"]
        .transform("first")
    )

    df["session_last_timestamp"] = (
        session_group["timestamp"]
        .transform("last")
    )

    df["is_session_final_bar"] = (
        df["timestamp"]
        == df["session_last_timestamp"]
    )

    # ========================================================
    # 14. APPLY HARD-CODED SCREENER
    # ========================================================

    df["qualified"] = (
        (df["avg_10d_volume"] > 1_000_000)
        & (
            df["close"]
            * df["avg_10d_volume"]
            > 1_000_000
        )
        & (
            df["relative_volume"] > 2
        )
        & (
            df["close"] >= 0.50
        )
        & (
            df["close"] <= 20.00
        )
    )

    # ========================================================
    # 15. REMOVE WARM-UP FROM SIMULATION
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

    # ========================================================
    # 16. ENTRY SIGNAL
    # ========================================================

    df["entry_signal"] = (
        df["qualified"]
        & df["previous_high"].notna()
        & (
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
    # 17. SKIP STOCKS WITH NO QUALIFYING OPPORTUNITIES
    # ========================================================

    if (
        qualified_count == 0
        or entry_count == 0
    ):

        print(
            f"Skipping {symbol}: "
            f"no qualified bars or no entry signals"
        )

        del df
        gc.collect()

        continue

    # ========================================================
    # 18. PREPARE NUMBA ARRAYS
    # ========================================================

    timestamps_ns = (
        df["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    session_ids = (
        df["session_id"]
        .to_numpy(
            dtype=np.int64
        )
    )

    session_final_flags = (
        df["is_session_final_bar"]
        .to_numpy(
            dtype=np.bool_
        )
    )

    high_prices = (
        df["high"]
        .to_numpy(
            dtype=np.float64
        )
    )

    low_prices = (
        df["low"]
        .to_numpy(
            dtype=np.float64
        )

    )

    close_prices = (
        df["close"]
        .to_numpy(
            dtype=np.float64
        )
    )

    session_open_prices = (
        df["session_open"]
        .to_numpy(
            dtype=np.float64
        )
    )

    entry_indices = np.flatnonzero(
        df["entry_signal"].to_numpy(
            dtype=np.bool_
        )
    ).astype(np.int64)

    # ========================================================
    # 19. HISTORY BASE DATA
    # ========================================================

    history_base = df[
        df["qualified"]
    ].copy()

    history_base["symbol"] = symbol

    history_base["entry_signal"] = (
        history_base["entry_signal"]
        .astype(bool)
    )

    # ========================================================
    # 20. RUN ALL TP / TSL COMBINATIONS
    # ========================================================

    for tp_pct, tsl_pct in TP_TSL_CONFIGS:

        print(
            f"{symbol}: "
            f"TP {tp_pct:.0%} / "
            f"TSL {tsl_pct:.0%}"
        )

        # ----------------------------------------------------
        # Run independent simulation.
        # ----------------------------------------------------

        (
            entry_price_array,
            take_profit_array,
            trailing_stop_array,
            highest_price_array,
            exit_timestamp_array,
            exit_price_array,
            profit_pct_array,
            hold_minutes_array,
            exit_reason_array,
            undelayed_trade_array,
        ) = simulate_trades(
            entry_indices,
            timestamps_ns,
            session_ids,
            session_final_flags,
            close_prices,
            high_prices,
            low_prices,
            session_open_prices,
            tp_pct,
            tsl_pct,
        )

        # ----------------------------------------------------
        # Copy the qualified-bar history.
        # ----------------------------------------------------

        history = history_base.copy()

        history["tp_pct"] = (
            tp_pct * 100.0
        )

        history["tsl_pct"] = (
            tsl_pct * 100.0
        )

        history["trade_created"] = False

        history["entry_price"] = np.nan
        history["take_profit_price"] = np.nan
        history["trailing_stop_price"] = np.nan
        history["highest_price_after_entry"] = np.nan

        history["exit_timestamp"] = pd.NaT
        history["exit_price"] = np.nan

        history["trade_profit_pct"] = np.nan
        history["trade_hold_minutes"] = np.nan

        history["exit_reason"] = pd.NA

        history["undelayed_pct_change"] = (
            (
                history["close"]
                - history["session_open"]
            )
            / history["session_open"]
            * 100.0
        )

        # ----------------------------------------------------
        # Attach each independent trade to its entry bar.
        # ----------------------------------------------------

        entry_index_values = (
            history.index
        )

        entry_index_to_position = {
            idx: position
            for position, idx
            in enumerate(
                entry_indices
            )
        }

        for idx in entry_index_values:

            if idx not in entry_index_to_position:
                continue

            trade_number = (
                entry_index_to_position[idx]
            )

            history.loc[
                idx,
                "trade_created"
            ] = True

            history.loc[
                idx,
                "entry_price"
            ] = entry_price_array[
                trade_number
            ]

            history.loc[
                idx,
                "take_profit_price"
            ] = take_profit_array[
                trade_number
            ]

            history.loc[
                idx,
                "trailing_stop_price"
            ] = trailing_stop_array[
                trade_number
            ]

            history.loc[
                idx,
                "highest_price_after_entry"
            ] = highest_price_array[
                trade_number
            ]

            exit_ns = (
                exit_timestamp_array[
                    trade_number
                ]
            )

            if exit_ns >= 0:

                history.loc[
                    idx,
                    "exit_timestamp"
                ] = pd.Timestamp(
                    exit_ns,
                    unit="ns",
                    tz="UTC",
                ).tz_convert(
                    TIMEZONE
                )

            history.loc[
                idx,
                "exit_price"
            ] = exit_price_array[
                trade_number
            ]

            history.loc[
                idx,
                "trade_profit_pct"
            ] = profit_pct_array[
                trade_number
            ]

            history.loc[
                idx,
                "trade_hold_minutes"
            ] = hold_minutes_array[
                trade_number
            ]

            reason_code = (
                exit_reason_array[
                    trade_number
                ]
            )

            if reason_code == 0:

                history.loc[
                    idx,
                    "exit_reason"
                ] = "TAKE_PROFIT"

            elif reason_code == 1:

                history.loc[
                    idx,
                    "exit_reason"
                ] = "TRAILING_STOP"

            elif reason_code == 2:

                history.loc[
                    idx,
                    "exit_reason"
                ] = "SESSION_END_LIQUIDATION"

        # ----------------------------------------------------
        # Final column ordering.
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

            "exit_timestamp",
            "exit_price",

            "trade_profit_pct",
            "trade_hold_minutes",
            "exit_reason",

            "undelayed_pct_change",
        ]

        history = (
            history[
                HISTORY_COLUMNS
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # ====================================================
        # 21. WRITE PARQUET DIRECTLY TO R2
        # ====================================================

        tp_label = int(
            tp_pct * 100
        )

        tsl_label = int(
            tsl_pct * 100
        )

        output_filename = (
            f"{symbol}_"
            f"tp{tp_label}_"
            f"tsl{tsl_label}.parquet"
        )

        r2_key = (
            f"{HISTORY_PREFIX}/"
            f"{output_filename}"
        )

        # ----------------------------------------------------
        # Convert dataframe to Arrow table.
        # ----------------------------------------------------

        table = pa.Table.from_pandas(
            history,
            preserve_index=False,
        )

        # ----------------------------------------------------
        # Write Parquet into memory.
        #
        # Nothing is written to local disk.
        # ----------------------------------------------------

        buffer = io.BytesIO()

        pq.write_table(
            table,
            buffer,
            compression="snappy",
        )

        buffer.seek(0)

        # ----------------------------------------------------
        # Upload directly to R2.
        # ----------------------------------------------------

        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=r2_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        print(
            f"Uploaded: "
            f"s3://{R2_BUCKET_NAME}/{r2_key} "
            f"({len(history):,} qualified bars)"
        )

        # ----------------------------------------------------
        # Release this TP/TSL history.
        # ----------------------------------------------------

        del history
        del table
        del buffer

        del entry_price_array
        del take_profit_array
        del trailing_stop_array
        del highest_price_array
        del exit_timestamp_array
        del exit_price_array
        del profit_pct_array
        del hold_minutes_array
        del exit_reason_array
        del undelayed_trade_array

        gc.collect()

    # ========================================================
    # 22. FINISH STOCK
    # ========================================================

    del history_base
    del df

    del timestamps_ns
    del session_ids
    del session_final_flags
    del high_prices
    del low_prices
    del close_prices
    del session_open_prices
    del entry_indices

    gc.collect()

    print(
        f"Finished {symbol}"
    )


# ============================================================
# 23. FINISH
# ============================================================

print("All stocks processed.")