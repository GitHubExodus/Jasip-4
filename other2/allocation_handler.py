from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AllocationState:
    symbol: str

    price: float = 0.0
    percent_change: float = 0.0

    is_active: bool = False

    allocation_ratio: float = 0.0
    allocation_percent: float = 0.0
    allocated_dollars: float = 0.0
    target_shares: int = 0


class AllocationHandler:
    def __init__(self, initial_buying_power: float = 10_000.0):
        self.buying_power = float(initial_buying_power)
        self.stocks: dict[str, AllocationState] = {}

    # ---------------------------------------------------------
    # STOCK MANAGEMENT
    # ---------------------------------------------------------

    def add_stock(self, symbol: str) -> None:
        if symbol not in self.stocks:
            self.stocks[symbol] = AllocationState(
                symbol=symbol
            )

    def get_stock(self, symbol: str) -> AllocationState | None:
        return self.stocks.get(symbol)

    # ---------------------------------------------------------
    # JS SCREENER UPDATE
    # ---------------------------------------------------------

    def update_stock(
        self,
        symbol: str,
        price: float,
        percent_change: float,
    ) -> None:
        self.add_stock(symbol)

        stock = self.stocks[symbol]

        stock.price = float(price)
        stock.percent_change = float(percent_change)
        stock.is_active = True

    def mark_inactive_stocks(
        self,
        active_symbols: set[str],
    ) -> None:
        for symbol, stock in self.stocks.items():
            stock.is_active = symbol in active_symbols

    # ---------------------------------------------------------
    # ALLOCATION CALCULATION
    # ---------------------------------------------------------

    def recalculate(self) -> None:
        active_positive_stocks = [
            stock
            for stock in self.stocks.values()
            if stock.is_active
            and stock.percent_change > 0
            and stock.price > 0
        ]

        total_positive_change = sum(
            stock.percent_change
            for stock in active_positive_stocks
        )

        # Reset allocation values for every stock first.
        for stock in self.stocks.values():
            stock.allocation_ratio = 0.0
            stock.allocation_percent = 0.0
            stock.allocated_dollars = 0.0
            stock.target_shares = 0

        if total_positive_change <= 0:
            return

        for stock in active_positive_stocks:
            stock.allocation_ratio = (
                stock.percent_change
                / total_positive_change
            )

            stock.allocation_percent = (
                stock.allocation_ratio * 100.0
            )

            stock.allocated_dollars = (
                stock.allocation_ratio
                * self.buying_power
            )

            stock.target_shares = int(
                stock.allocated_dollars
                / stock.price
            )

    # ---------------------------------------------------------
    # BUYING POWER
    # ---------------------------------------------------------

    def get_buying_power(self) -> float:
        return self.buying_power

    def set_buying_power(self, amount: float) -> None:
        self.buying_power = max(0.0, float(amount))

        self.recalculate()

    def deduct_buying_power(self, amount: float) -> None:
        self.buying_power = max(
            0.0,
            self.buying_power - float(amount),
        )

        self.recalculate()

    def add_buying_power(self, amount: float) -> None:
        self.buying_power += float(amount)

        self.recalculate()

    # ---------------------------------------------------------
    # TRADE EVENTS
    # ---------------------------------------------------------

    def record_buy_fill(
        self,
        symbol: str,
        quantity: int,
        fill_price: float,
    ) -> float:
        spent = (
            int(quantity)
            * float(fill_price)
        )

        self.deduct_buying_power(spent)

        return spent

    def record_sell_fill(
        self,
        symbol: str,
        quantity: int,
        fill_price: float,
    ) -> float:
        returned_cash = (
            int(quantity)
            * float(fill_price)
        )

        self.add_buying_power(returned_cash)

        return returned_cash

    # ---------------------------------------------------------
    # ALLOCATION QUERIES
    # ---------------------------------------------------------

    def get_target_shares(self, symbol: str) -> int:
        stock = self.stocks.get(symbol)

        if stock is None:
            return 0

        if not stock.is_active:
            return 0

        return stock.target_shares

    def get_allocation(self, symbol: str) -> float:
        stock = self.stocks.get(symbol)

        if stock is None:
            return 0.0

        return stock.allocated_dollars

    def get_allocation_percent(self, symbol: str) -> float:
        stock = self.stocks.get(symbol)

        if stock is None:
            return 0.0

        return stock.allocation_percent

    # ---------------------------------------------------------
    # STATE ACCESS
    # ---------------------------------------------------------

    def get_all_states(self) -> dict[str, AllocationState]:
        return self.stocks
  