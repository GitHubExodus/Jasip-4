from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")


@dataclass
class TradeIntervalState:
    symbol: str
    entry_timestamp: datetime
    last_interval_index: int = 0


class TradeIntervalTracking:
    def __init__(
        self,
        interval_minutes: int = 5,
    ):
        self.interval_minutes = interval_minutes
        self.trades: dict[str, TradeIntervalState] = {}

    def add_trade(
        self,
        symbol: str,
        entry_timestamp: datetime,
    ) -> None:
        entry_timestamp = self._normalize_timestamp(
            entry_timestamp
        )

        self.trades[symbol] = TradeIntervalState(
            symbol=symbol,
            entry_timestamp=entry_timestamp,
            last_interval_index=0,
        )

    def remove_trade(
        self,
        symbol: str,
    ) -> None:
        self.trades.pop(symbol, None)

    def get_trade(
        self,
        symbol: str,
    ) -> TradeIntervalState | None:
        return self.trades.get(symbol)

    def get_interval_index(
        self,
        symbol: str,
        now: datetime | None = None,
    ) -> int:
        state = self.trades.get(symbol)

        if state is None:
            return 0

        if now is None:
            now = datetime.now(NEW_YORK)
        else:
            now = self._normalize_timestamp(now)

        elapsed_seconds = (
            now - state.entry_timestamp
        ).total_seconds()

        if elapsed_seconds < 0:
            return 0

        interval_seconds = (
            self.interval_minutes * 60
        )

        return int(
            elapsed_seconds // interval_seconds
        )

    def is_ready_to_sell(
        self,
        symbol: str,
        now: datetime | None = None,
    ) -> bool:
        interval_index = self.get_interval_index(
            symbol,
            now,
        )

        return interval_index >= 1

    def is_new_interval(
        self,
        symbol: str,
        now: datetime | None = None,
    ) -> bool:
        state = self.trades.get(symbol)

        if state is None:
            return False

        current_interval = self.get_interval_index(
            symbol,
            now,
        )

        return current_interval > state.last_interval_index

    def mark_interval_processed(
        self,
        symbol: str,
        now: datetime | None = None,
    ) -> None:
        state = self.trades.get(symbol)

        if state is None:
            return

        current_interval = self.get_interval_index(
            symbol,
            now,
        )

        if current_interval > state.last_interval_index:
            state.last_interval_index = current_interval

    def get_elapsed_seconds(
        self,
        symbol: str,
        now: datetime | None = None,
    ) -> float:
        state = self.trades.get(symbol)

        if state is None:
            return 0.0

        if now is None:
            now = datetime.now(NEW_YORK)
        else:
            now = self._normalize_timestamp(now)

        elapsed_seconds = (
            now - state.entry_timestamp
        ).total_seconds()

        return max(0.0, elapsed_seconds)

    def _normalize_timestamp(
        self,
        timestamp: datetime,
    ) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(
                tzinfo=NEW_YORK
            )

        return timestamp.astimezone(
            NEW_YORK
        )

    def get_all_states(
        self,
    ) -> dict[str, TradeIntervalState]:
        return self.trades
