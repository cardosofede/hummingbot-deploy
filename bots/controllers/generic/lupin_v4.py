import time
from decimal import Decimal
from typing import List, Tuple, Dict, Set

import pandas_ta as ta  # noqa: F401
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import TradeType, PriceType, OrderType, PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.controller_base import ControllerConfigBase, ControllerBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, \
    TripleBarrierConfig, TrailingStop
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction, CreateExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class LupinV4Config(ControllerConfigBase):
    """
    Configuration required to run the D-Man Maker V2 strategy.
    """
    controller_name: str = "lupin_v4"
    candles_config: List[CandlesConfig] = []
    connector_name: str = "binance_perpetual"
    trading_pair: str = "DOGE-USDT"
    executor_amount_quote: Decimal = Decimal("10")
    spreads: List[Decimal] = [Decimal("0.01"), Decimal("0.02"), Decimal("0.04"), Decimal("0.08")]
    amounts_quote_pct: List[Decimal] = [Decimal("0.1"), Decimal("0.2"), Decimal("0.4"), Decimal("0.8")]
    total_amount_quote: Decimal = Decimal("1000")
    min_spread_between_executors: Decimal = Decimal("1")
    executor_refresh_time: int = 20
    executor_spread: Decimal = Decimal("0.002")
    leverage: int = 20
    interval: str = "1m"
    reversion_window: int = 4
    natr_window: int = 20
    position_mode: PositionMode = PositionMode.HEDGE
    global_stop_loss: Decimal = Decimal("0.05")
    take_profit: Decimal = Decimal("1")
    time_limit: int = 86400
    trailing_stop: TrailingStop = TrailingStop(
        activation_price=Decimal("0.8"),
        trailing_delta=Decimal("0.1")
    )

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.trading_pair)
        return markets


class LupinV4(ControllerBase):
    def __init__(self, config: LupinV4Config, *args, **kwargs):
        self.config = config
        normalized_amounts_quote_pct = [amount / sum(config.amounts_quote_pct) for amount in config.amounts_quote_pct]
        self.amounts_quote = [config.total_amount_quote * amount for amount in normalized_amounts_quote_pct]
        self.cumulative_amounts_quote = [sum(self.amounts_quote[:i + 1]) for i in range(len(self.amounts_quote))]
        self.max_records = 1000
        self.processed_data = {}
        if self.config.candles_config == []:
            self.config.candles_config = [CandlesConfig(
                connector=config.connector_name,
                trading_pair=config.trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        base_triple_barrier = TripleBarrierConfig(
            take_profit=self.config.take_profit,
            time_limit=self.config.time_limit,
            trailing_stop=self.config.trailing_stop,
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.LIMIT,
        )
        return base_triple_barrier.new_instance_with_adjusted_volatility(volatility_factor=self.current_natr())

    @property
    def min_spread_between_executors(self) -> Decimal:
        return self.config.min_spread_between_executors * Decimal(self.current_natr())

    def current_natr(self) -> float:
        return self.processed_data["natr"].iloc[-1]

    def active_executors_by_side(self, side) -> List[ExecutorInfo]:
        return self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == side and x.is_active)

    def active_long_executors_order_placed(self) -> List[ExecutorInfo]:
        return self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == TradeType.BUY and not x.is_trading)

    def active_short_executors_order_placed(self) -> List[ExecutorInfo]:
        return self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == TradeType.SELL and not x.is_trading)

    def unrealized_pnl(self) -> Decimal:
        return sum([executor.net_pnl_quote for executor in self.executors_info if executor.is_active])

    def get_executor_stats_by_side(self, side: TradeType) -> Tuple[Decimal, Decimal, Decimal]:
        min_price = Decimal("inf")
        max_price = Decimal("-inf")
        executors = self.active_executors_by_side(side)
        min_price = min([executor.config.entry_price for executor in executors], default=min_price)
        max_price = max([executor.config.entry_price for executor in executors], default=max_price)
        size = sum([executor.filled_amount_quote for executor in executors])
        return min_price, max_price, size

    def get_mid_price(self) -> Decimal:
        return self.market_data_provider.get_price_by_type(connector_name=self.config.connector_name,
                                                           trading_pair=self.config.trading_pair,
                                                           price_type=PriceType.MidPrice)

    def create_executor_action(self, side):
        spread_multiplier = -1 if side == TradeType.BUY else 1
        entry_price = self.processed_data["mid_price"] * (1 + spread_multiplier * self.config.executor_spread)
        amount = self.config.executor_amount_quote / entry_price
        return CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=PositionExecutorConfig(
                timestamp=self.market_data_provider.time(),
                trading_pair=self.config.trading_pair,
                connector_name=self.config.connector_name,
                side=side,
                entry_price=entry_price,
                amount=amount,
                triple_barrier_config=self.triple_barrier_config,
        ))

    def create_actions_proposal(self) -> List[ExecutorAction]:
        """
        Create actions proposal based on the current state of the strategy. There are two main ways to create actions:
        1. Reaching a new level: We compute the distance of the price the each BEP and if this distance is greater than
        the spread of any spread of the list, we take loop for each spread lower than the distance and sum the amount
        corresponding to the spread, if the amount is greater than the current imbalance, the new level is reached and
        we follow it with a trailing stop.
        2. Hedge: If the current imbalance is greater than the min_imbalance_to_hedge_quote, and the current price is
        greater/lower depending on the trade type, we run a trailing stop to hedge the imbalance.
        """
        actions = []

        for side in [TradeType.BUY, TradeType.SELL]:
            processed_data = self.processed_data[side]
            if len(processed_data["active_order_placed"]) == 0:
                if processed_data["size"] == Decimal("0"):
                    action = self.create_executor_action(side)
                    actions.append(action)
                elif processed_data["size"] + self.config.executor_amount_quote < processed_data["max_size"] and \
                        processed_data["reverting"] and \
                        processed_data["distance_to_closest"] > self.min_spread_between_executors:
                    action = self.create_executor_action(side)
                    actions.append(action)
        return actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Stop actions proposal based on the current state of the strategy. This is due stop loss and to execute it
        all the levels must be reached and the PNL should be lower than the stop loss.
        """
        actions = []
        if self.processed_data["unrealized_pnl"] < -self.config.global_stop_loss * self.config.total_amount_quote:
            actions.extend(self.stop_all_executors())
            self.logger().info("Stop loss reached")
        for side in [TradeType.BUY, TradeType.SELL]:
            processed_data = self.processed_data[side]
            for executor in processed_data["active_order_placed"]:
                if executor.config.timestamp + self.config.executor_refresh_time < self.market_data_provider.time():
                    actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=executor.id))
        return actions

    def stop_all_executors(self) -> List[ExecutorAction]:
        return [StopExecutorAction(
            controller_id=self.config.id,
            executor_id=executor.id) for executor in self.executors_info]

    async def update_processed_data(self):
        candles_df = self.market_data_provider.get_candles_df(self.config.connector_name, self.config.trading_pair,
                                                              self.config.interval, self.max_records)
        natr = ta.natr(candles_df["high"], candles_df["low"], candles_df["close"], self.config.natr_window) / 100
        returns_reversion_window = candles_df["close"].pct_change().tail(self.config.reversion_window).sum()
        high_reversion_window = candles_df["high"].tail(self.config.reversion_window).max()
        low_reversion_window = candles_df["low"].tail(self.config.reversion_window).min()
        mid_price = self.get_mid_price()
        min_long_price, max_long_price, size_long = self.get_executor_stats_by_side(side=TradeType.BUY)
        min_short_price, max_short_price, size_short = self.get_executor_stats_by_side(side=TradeType.SELL)
        distance_to_min_buy = (min_long_price - mid_price) / mid_price
        distance_to_max_buy = (max_long_price - mid_price) / mid_price
        distance_to_min_sell = (mid_price - min_short_price) / mid_price
        distance_to_max_sell = (mid_price - max_short_price) / mid_price

        current_buy_level = next((i for i, spread in enumerate(self.config.spreads) if distance_to_max_buy < spread), None)
        current_sell_level = next((i for i, spread in enumerate(self.config.spreads) if distance_to_min_sell < spread), None)

        max_size_long = self.cumulative_amounts_quote[current_buy_level] if current_buy_level is not None else Decimal("0")
        max_size_short = self.cumulative_amounts_quote[current_sell_level] if current_sell_level is not None else Decimal("0")

        active_long_order_placed = self.active_long_executors_order_placed()
        active_short_order_placed = self.active_short_executors_order_placed()
        self.processed_data = {
            TradeType.BUY: {
                "max_size": max_size_long,
                "size": size_long,
                "distance_to_closest": distance_to_min_buy,
                "distance_to_origin": distance_to_max_buy,
                "reverting": returns_reversion_window > 0 and low_reversion_window < mid_price,
                "active_order_placed": active_long_order_placed
            },
            TradeType.SELL: {
                "max_size": max_size_short,
                "size": size_short,
                "distance_to_closest": distance_to_max_sell,
                "distance_to_origin": distance_to_min_sell,
                "reverting": returns_reversion_window < 0 and high_reversion_window > mid_price,
                "active_order_placed": active_short_order_placed
            },
            "unrealized_pnl": self.unrealized_pnl(),
            "mid_price": mid_price,
            "natr": natr,
        }

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def to_format_status(self) -> List[str]:
        lines = []
        lines.append(f"""

Lupin V3 Config:
    Total amount: {self.config.total_amount_quote} | Order amount: {self.config.executor_amount_quote}
    Spreads: {[round(float(spread) * 100, 2) for spread in self.config.spreads]} | 
    Cumulative Amounts: {[round(float(amount), 2) for amount in self.cumulative_amounts_quote]}
    Min spread between executors: {self.config.min_spread_between_executors}
    Executor refresh time: {self.config.executor_refresh_time} | Executor spread: {self.config.executor_spread}
    Leverage: {self.config.leverage} | Interval: {self.config.interval}
    Reversion window: {self.config.reversion_window}
    Global stop loss: {self.config.global_stop_loss} | Take profit: {self.config.take_profit} | Time limit: {self.config.time_limit}
    Trail stop: {self.config.trailing_stop}
    Triple Barrier: {self.triple_barrier_config}
    NATR window: {self.config.natr_window} | NATR: {self.current_natr()*100:.2f}%
""")
        for side in [TradeType.BUY, TradeType.SELL]:
            processed_data = self.processed_data.get(side)
            if processed_data is None:
                continue
            lines.append(f"{side.name} side:")
            lines.append(f"  - Size: {processed_data['size']:.2f} | Max size: {processed_data['max_size']:.2f}")
            lines.append(f"  - Distance to closest: {processed_data['distance_to_closest']*100:.2f}%| Distance to origin: {processed_data['distance_to_origin']*100:.2f}%")
            lines.append(f"  - Reverting: {processed_data['reverting']}")
            lines.append(f"  - Active order placed: {len(processed_data['active_order_placed'])}")
        return lines
