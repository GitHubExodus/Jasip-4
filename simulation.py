import numpy as np
import pandas as pd
from numba import njit


@njit
def simulate(open_price, low, close_to_low, ema_200_low):
    n = len(open_price)
    profit = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(open_price[i]):
            continue

        entry = open_price[i]

        for j in range(i, n):
            if np.isnan(close_to_low[j]) or np.isnan(ema_200_low[j]):
                continue

            # Stop when the current close-to-low move
            # becomes more negative than its 200 EMA.
            if close_to_low[j] < ema_200_low[j]:
                profit[i] = (low[j] / entry - 1.0) * 100.0
                break

    return profit


class Simulation:
    def __init__(self, data):
        self.data = data.copy()

    def run(self):
        d = self.data

        prev_close = d["close"].shift(1)

        close_to_high = (
            (d["high"] / prev_close - 1.0) * 100.0
        )

        close_to_low = (
            (d["low"] / prev_close - 1.0) * 100.0
        )

        ema_200_high = close_to_high.ewm(
            span=200,
            adjust=False
        ).mean()

        ema_200_low = close_to_low.ewm(
            span=200,
            adjust=False
        ).mean()

        profit = simulate(
            d["open"].to_numpy(dtype=np.float64),
            d["low"].to_numpy(dtype=np.float64),
            close_to_low.to_numpy(dtype=np.float64),
            ema_200_low.to_numpy(dtype=np.float64)
        )

        return pd.DataFrame({
            "profit": profit,
            "close_to_high": close_to_high,
            "close_to_low": close_to_low,
            "ema_200_high": ema_200_high,
            "ema_200_low": ema_200_low
        }, index=d.index)