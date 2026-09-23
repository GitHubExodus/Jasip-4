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

    # ============================================================
    # 11. NUMBA TRADE SIMULATION
    # ============================================================

    from numba import njit


    @njit
    def simulate_trades(
        timestamps_ns,
        session_ids,
        highs,
        lows,
        closes,
        entry_signal,
        session_final,
        session_opens,
        tp_pct,
        tsl_pct,
    ):
        """
        Every entry signal is simulated independently.

        There is:
            - no portfolio
            - no buying power
            - no position interaction
            - no equity curve

        The entry candle does NOT manage the trade.

        Subsequent bars in the same session are checked.

        SESSION_END_LIQUIDATION takes priority on the actual
        final bar of the session and exits at that bar's close.
        """

        n = len(highs)

        trade_created = np.zeros(n, dtype=np.bool_)

        entry_prices = np.full(n, np.nan)
        take_profit_prices = np.full(n, np.nan)
        trailing_stop_prices = np.full(n, np.nan)
        highest_prices = np.full(n, np.nan)

        exit_prices = np.full(n, np.nan)
        trade_profit_pcts = np.full(n, np.nan)
        trade_hold_minutes = np.full(n, np.nan)

        # Exit reason:
        #
        # 0 = none
        # 1 = TAKE_PROFIT
        # 2 = TRAILING_STOP
        # 3 = SESSION_END_LIQUIDATION
        #
        exit_reasons = np.zeros(n, dtype=np.int8)

        undelayed_pcts = np.full(n, np.nan)

        for i in range(n):

            if not entry_signal[i]:
                continue

            trade_created[i] = True

            entry_price = highs[i] * 1.0025
            tp_price = entry_price * (1.0 + tp_pct)

            entry_prices[i] = entry_price
            take_profit_prices[i] = tp_price

            highest_price = entry_price

            # --------------------------------------------------------
            # Search forward through subsequent bars in SAME session.
            # --------------------------------------------------------

            exited = False

            for j in range(i + 1, n):

                # Never cross into another trading session.
                if session_ids[j] != session_ids[i]:
                    break

                bar_high = highs[j]
                bar_low = lows[j]

                # ----------------------------------------------------
                # SESSION END
                #
                # The actual final bar closes the trade at its CLOSE.
                #
                # This is deliberately checked BEFORE TP/TSL because
                # the final-bar liquidation is required to use that
                # bar's close price.
                # ----------------------------------------------------

                if session_final[j]:

                    exit_price = closes[j]

                    exit_prices[i] = exit_price
                    exit_reasons[i] = 3

                    highest_prices[i] = highest_price

                    trailing_stop_prices[i] = (
                        highest_price * (1.0 - tsl_pct)
                    )

                    trade_profit_pcts[i] = (
                        (exit_price - entry_price)
                        / entry_price
                        * 100.0
                    )

                    trade_hold_minutes[i] = (
                        (timestamps_ns[j] - timestamps_ns[i])
                        / 60_000_000_000.0
                    )

                    undelayed_pcts[i] = (
                        (closes[j] - session_opens[i])
                        / session_opens[i]
                        * 100.0
                    )

                    exited = True
                    break

                # ----------------------------------------------------
                # Update highest price reached AFTER entry.
                # ----------------------------------------------------

                if bar_high > highest_price:
                    highest_price = bar_high

                trailing_stop = (
                    highest_price * (1.0 - tsl_pct)
                )

                # ----------------------------------------------------
                # TAKE PROFIT
                # ----------------------------------------------------

                if bar_high >= tp_price:

                    exit_price = tp_price

                    exit_prices[i] = exit_price
                    exit_reasons[i] = 1

                    highest_prices[i] = highest_price
                    trailing_stop_prices[i] = trailing_stop

                    trade_profit_pcts[i] = (
                        (exit_price - entry_price)
                        / entry_price
                        * 100.0
                    )

                    trade_hold_minutes[i] = (
                        (timestamps_ns[j] - timestamps_ns[i])
                        / 60_000_000_000.0
                    )

                    undelayed_pcts[i] = (
                        (closes[j] - session_opens[i])
                        / session_opens[i]
                        * 100.0
                    )

                    exited = True
                    break

                # ----------------------------------------------------
                # TRAILING STOP
                # ----------------------------------------------------

                if bar_low <= trailing_stop:

                    exit_price = trailing_stop

                    exit_prices[i] = exit_price
                    exit_reasons[i] = 2

                    highest_prices[i] = highest_price
                    trailing_stop_prices[i] = trailing_stop

                    trade_profit_pcts[i] = (
                        (exit_price - entry_price)
                        / entry_price
                        * 100.0
                    )

                    trade_hold_minutes[i] = (
                        (timestamps_ns[j] - timestamps_ns[i])
                        / 60_000_000_000.0
                    )

                    undelayed_pcts[i] = (
                        (closes[j] - session_opens[i])
                        / session_opens[i]
                        * 100.0
                    )

                    exited = True
                    break

            # --------------------------------------------------------
            # Safety fallback.
            #
            # This occurs when the entry itself is the final bar.
            # --------------------------------------------------------

            if not exited:

                if session_final[i]:

                    exit_price = closes[i]

                    exit_prices[i] = exit_price
                    exit_reasons[i] = 3

                    highest_prices[i] = highest_price

                    trailing_stop_prices[i] = (
                        highest_price * (1.0 - tsl_pct)
                    )

                    trade_profit_pcts[i] = (
                        (exit_price - entry_price)
                        / entry_price
                        * 100.0
                    )

                    trade_hold_minutes[i] = 0.0

                    undelayed_pcts[i] = (
                        (closes[i] - session_opens[i])
                        / session_opens[i]
                        * 100.0
                    )


        return (
            trade_created,
            entry_prices,
            take_profit_prices,
            trailing_stop_prices,
            highest_prices,
            exit_prices,
            trade_profit_pcts,
            trade_hold_minutes,
            exit_reasons,
            undelayed_pcts,
        )


    # ============================================================
    # 12. PREPARE ARRAYS FOR NUMBA
    # ============================================================

    # Convert timestamps to integer nanoseconds so Numba can
    # calculate holding time efficiently.

    timestamps_ns = (
        df["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    session_ids = (
        pd.factorize(
            df["session_date"],
            sort=False,
        )[0]
        .astype(np.int64)
    )

    highs = (
        df["high"]
        .to_numpy(dtype=np.float64)
    )

    lows = (
        df["low"]
        .to_numpy(dtype=np.float64)
    )

    closes = (
        df["close"]
        .to_numpy(dtype=np.float64)
    )

    entry_signal_array = (
        df["entry_signal"]
        .to_numpy(dtype=np.bool_)
    )

    session_final_array = (
        df["is_session_final_bar"]
        .to_numpy(dtype=np.bool_)
    )

    session_opens = (
        df["session_open"]
        .to_numpy(dtype=np.float64)
    )


    # ============================================================
    # 13. TP / TSL CONFIGURATIONS
    # ============================================================

    TP_TSL_CONFIGS = [
        (0.05, 0.10),
        (0.05, 0.05),
        (0.10, 0.05),
        (0.10, 0.10),
        (0.20, 0.10),
        (0.20, 0.15),
        (0.30, 0.05),
        (0.30, 0.10),
        (0.30, 0.20),
    ]


    # ============================================================
    # 14. SAVE HISTORY DIRECTLY TO R2
    # ============================================================

    HISTORY_PREFIX = "history/"


    for tp_pct, tsl_pct in TP_TSL_CONFIGS:

        print(
            f"{symbol}: "
            f"running TP {tp_pct:.0%} / TSL {tsl_pct:.0%}"
        )

        # --------------------------------------------------------
        # Run Numba simulation.
        # --------------------------------------------------------

        (
            trade_created,
            entry_prices,
            take_profit_prices,
            trailing_stop_prices,
            highest_prices,
            exit_prices,
            trade_profit_pcts,
            trade_hold_minutes,
            exit_reasons,
            undelayed_pcts,
        ) = simulate_trades(
            timestamps_ns,
            session_ids,
            highs,
            lows,
            closes,
            entry_signal_array,
            session_final_array,
            session_opens,
            tp_pct,
            tsl_pct,
        )

        # --------------------------------------------------------
        # Convert numeric exit reasons to readable strings.
        # --------------------------------------------------------

        exit_reason_strings = np.full(
            len(df),
            None,
            dtype=object,
        )

        exit_reason_strings[
            exit_reasons == 1
        ] = "TAKE_PROFIT"

        exit_reason_strings[
            exit_reasons == 2
        ] = "TRAILING_STOP"

        exit_reason_strings[
            exit_reasons == 3
        ] = "SESSION_END_LIQUIDATION"

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Build history using the ORIGINAL df indices.
        #
        # This guarantees that entry statistics are attached to
        # the exact qualifying bar that generated the entry.
        # --------------------------------------------------------

        qualified_mask = (
            df["qualified"].to_numpy(dtype=np.bool_)
        )

        history = df.loc[qualified_mask].copy()

        # --------------------------------------------------------
        # Attach trade information using positional boolean arrays.
        # --------------------------------------------------------

        qualified_positions = np.flatnonzero(
            qualified_mask
        )

        history["trade_created"] = (
            trade_created[qualified_positions]
        )

        history["entry_price"] = (
            entry_prices[qualified_positions]
        )

        history["take_profit_price"] = (
            take_profit_prices[qualified_positions]
        )

        history["trailing_stop_price"] = (
            trailing_stop_prices[qualified_positions]
        )

        history["highest_price_after_entry"] = (
            highest_prices[qualified_positions]
        )

        history["exit_price"] = (
            exit_prices[qualified_positions]
        )

        history["trade_profit_pct"] = (
            trade_profit_pcts[qualified_positions]
        )

        history["trade_hold_minutes"] = (
            trade_hold_minutes[qualified_positions]
        )

        history["exit_reason"] = (
            exit_reason_strings[qualified_positions]
        )

        history["symbol"] = symbol

        history["tp_pct"] = tp_pct * 100.0

        history["tsl_pct"] = tsl_pct * 100.0

        # --------------------------------------------------------
        # Undelayed percentage change for EVERY qualified bar.
        #
        # Session open -> CURRENT BAR CLOSE.
        # --------------------------------------------------------

        history["undelayed_pct_change"] = (
            (
                history["close"]
                - history["session_open"]
            )
            / history["session_open"]
            * 100.0
        )

        history["entry_signal"] = (
            history["entry_signal"].astype(bool)
        )

        # --------------------------------------------------------
        # History columns.
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

        history = (
            history
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Filename.
        # --------------------------------------------------------

        tp_label = int(tp_pct * 100)
        tsl_label = int(tsl_pct * 100)

        output_filename = (
            f"{symbol}_tp{tp_label}_tsl{tsl_label}.parquet"
        )

        r2_key = (
            f"{HISTORY_PREFIX}{output_filename}"
        )

        # --------------------------------------------------------
        # Write Parquet into memory.
        #
        # Nothing is saved to the laptop's disk.
        # --------------------------------------------------------

        parquet_buffer = io.BytesIO()

        history.to_parquet(
            parquet_buffer,
            engine="pyarrow",
            index=False,
        )

        parquet_buffer.seek(0)

        # --------------------------------------------------------
        # Upload directly to Cloudflare R2.
        # --------------------------------------------------------

        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=r2_key,
            Body=parquet_buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        print(
            f"{symbol}: "
            f"TP {tp_pct:.0%} / TSL {tsl_pct:.0%} -> "
            f"{int(trade_created.sum()):,} trades"
        )

        print(
            f"Uploaded: "
            f"s3://{R2_BUCKET_NAME}/{r2_key}"
        )

        del history
        del parquet_buffer

        gc.collect()


# ============================================================
# 15. FINISH STOCK PROCESSING
# ============================================================

del df

gc.collect()

print(f"Finished {symbol}")
print("-" * 60)


# ============================================================
# 16. FINISH
# ============================================================

print("All stocks processed.")