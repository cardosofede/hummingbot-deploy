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
    TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction, CreateExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class LupinConfig(ControllerConfigBase):
    """
    Configuration required to run the D-Man Maker V2 strategy.
    """
    controller_name: str = "lupin"
    candles_config: List[CandlesConfig] = []
    connector_name: str = Field(
        default="binance_perpetual",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the connector name: "
        ))
    trading_pair: str = Field(
        default="WLD-USDT",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the trading pair: "
        ))
    initial_amount_quote: Decimal = Field(
        default=Decimal("10"),
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the initial amount in quote to open long/short: "
        ))
    spreads: List[Decimal] = [Decimal("0.01"), Decimal("0.02"), Decimal("0.04"), Decimal("0.08")]
    amounts_quote_pct: List[Decimal] = [Decimal("0.1"), Decimal("0.2"), Decimal("0.4"), Decimal("0.8")]
    total_amount_quote: Decimal = Decimal("1000")
    hedge_profit_pct: Decimal = Field(
        default=Decimal("0.002"),
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=False)
    )
    min_imbalance_to_hedge_quote: Decimal = Field(
        default=Decimal("2"),
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=False))
    leverage: int = 20
    interval: str = "1m"
    reversion_window: int = 4
    hedge_reversion_window: int = 2
    position_mode: PositionMode = PositionMode.HEDGE
    stop_loss: Decimal = Decimal("0.05")

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.trading_pair)
        return markets


class Lupin(ControllerBase):
    def __init__(self, config: LupinConfig, *args, **kwargs):
        self.config = config
        normalized_amounts_quote_pct = [amount / sum(config.amounts_quote_pct) for amount in config.amounts_quote_pct]
        self.amounts_quote = [config.total_amount_quote * amount for amount in normalized_amounts_quote_pct]
        self.cumulative_amounts_quote = [sum(self.amounts_quote[:i + 1]) for i in range(len(self.amounts_quote))]
        self.current_level = 0
        self.processed_data = {}
        if self.config.candles_config == []:
            self.config.candles_config = [CandlesConfig(
                connector=config.connector_name,
                trading_pair=config.trading_pair,
                interval=config.interval,
                max_records=1000
            )]
        super().__init__(config, *args, **kwargs)

    def active_long_executors_trading(self) -> List[ExecutorInfo]:
        return self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == TradeType.BUY and x.is_trading)

    def active_short_executors_trading(self) -> List[ExecutorInfo]:
        return self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == TradeType.SELL and x.is_trading)

    def active_long_executors_order_placed(self) -> List[ExecutorInfo]:
        return self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == TradeType.BUY and not x.is_trading)

    def active_short_executors_order_placed(self) -> List[ExecutorInfo]:
        return self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == TradeType.SELL and not x.is_trading)

    def current_bep_and_size_long(self) -> Tuple[Decimal, Decimal]:
        long_executors = self.active_long_executors_trading()
        total_amount = sum([executor.filled_amount_quote for executor in long_executors])
        if total_amount == 0:
            return Decimal("0"), Decimal("0")
        average_price = sum(
            [executor.filled_amount_quote * executor.custom_info["current_position_average_price"] for executor in
             long_executors]) / total_amount
        return average_price, total_amount

    def current_bep_and_size_short(self) -> Tuple[Decimal, Decimal]:
        short_executors = self.active_short_executors_trading()
        total_amount = sum([executor.filled_amount_quote for executor in short_executors])
        if total_amount == 0:
            return Decimal("0"), Decimal("0")
        average_price = sum(
            [executor.filled_amount_quote * executor.custom_info["current_position_average_price"] for executor in
             short_executors]) / total_amount
        return average_price, total_amount

    def current_pnl(self) -> Decimal:
        return sum([executor.net_pnl_quote for executor in self.executors_info])

    def resulting_bep_after_trade(self, bep: Decimal, bep_size: Decimal, trade_price: Decimal,
                                  trade_size: Decimal) -> Decimal:
        return (bep * bep_size + trade_price * trade_size) / (bep_size + trade_size)

    def get_price_to_move_bep(self, bep: Decimal, bep_size: Decimal, target_bep: Decimal,
                              order_size: Decimal) -> Decimal:
        order_price = (target_bep * (bep_size + order_size) - bep * bep_size) / order_size
        return order_price

    def current_distance_to_bep(self, bep: Decimal, current_price: Decimal) -> Decimal:
        return (current_price - bep) / bep if bep != 0 else Decimal("0")

    def get_mid_price(self) -> Decimal:
        return self.market_data_provider.get_price_by_type(connector_name=self.config.connector_name,
                                                           trading_pair=self.config.trading_pair,
                                                           price_type=PriceType.MidPrice)

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
        actions.extend(self.create_startup_actions())
        actions.extend(self.create_new_level_actions())
        actions.extend(self.create_hedge_actions())
        return actions

    def create_startup_actions(self) -> List[ExecutorAction]:
        sides = []

        if self.processed_data["long_size"] == Decimal("0") and len(self.processed_data["active_long_order_placed"]) == 0:
            sides.append(TradeType.BUY)
        if self.processed_data["short_size"] == Decimal("0") and len(self.processed_data["active_short_order_placed"]) == 0:
            sides.append(TradeType.SELL)
        actions = [CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=PositionExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                side=side,
                triple_barrier_config=TripleBarrierConfig(open_order_type=OrderType.MARKET),
                entry_price=self.processed_data["mid_price"],
                amount=self.config.initial_amount_quote / self.processed_data["mid_price"],
                leverage=self.config.leverage,
            )) for side in sides]
        return actions

    def create_hedge_actions(self) -> List[ExecutorAction]:
        actions = []
        if abs(self.processed_data["current_imbalance"]) > self.config.min_imbalance_to_hedge_quote:
            if self.processed_data["current_imbalance"] > 0 and self.processed_data["hedge_price_passed"] and \
                    self.processed_data["hedge_reverting_down"]:
                actions.extend(self.stop_all_executors())
            elif self.processed_data["current_imbalance"] < 0 and self.processed_data["hedge_price_passed"] and \
                    self.processed_data["hedge_reverting_up"]:
                actions.extend(self.stop_all_executors())
        return actions

    def create_new_level_actions(self) -> List[ExecutorAction]:
        actions = []
        next_level = self.processed_data["next_level"]
        if next_level is not None:
            if self.processed_data["short_bep_to_mid"] > self.processed_data["spread_next_level"] and self.processed_data["reverting_down"]\
                    and len(self.processed_data["active_short_order_placed"]) == 0:
                actions.append(CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=PositionExecutorConfig(
                        timestamp=time.time(),
                        connector_name=self.config.connector_name,
                        trading_pair=self.config.trading_pair,
                        side=TradeType.SELL,
                        triple_barrier_config=TripleBarrierConfig(open_order_type=OrderType.MARKET),
                        entry_price=self.processed_data["mid_price"],
                        amount=self.processed_data["amount_next_level"] / self.processed_data["mid_price"],
                        leverage=self.config.leverage,
                    ))
                )
            elif -self.processed_data["long_bep_to_mid"] > self.processed_data["spread_next_level"] and self.processed_data["reverting_up"]\
                    and len(self.processed_data["active_long_order_placed"]) == 0:
                actions.append(CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=PositionExecutorConfig(
                        timestamp=time.time(),
                        connector_name=self.config.connector_name,
                        trading_pair=self.config.trading_pair,
                        side=TradeType.BUY,
                        triple_barrier_config=TripleBarrierConfig(open_order_type=OrderType.MARKET),
                        entry_price=self.processed_data["mid_price"],
                        amount=self.processed_data["amount_next_level"] / self.processed_data["mid_price"],
                        leverage=self.config.leverage,
                    ))
                )
        return actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Stop actions proposal based on the current state of the strategy. This is due stop loss and to execute it
        all the levels must be reached and the PNL should be lower than the stop loss.
        """
        actions = []
        if self.processed_data["next_level"] is None and self.current_pnl() < -self.config.stop_loss * self.config.total_amount_quote:
            actions.extend(self.stop_all_executors())
            self.logger().info("Stop loss reached")
        return actions

    def stop_all_executors(self) -> List[ExecutorAction]:
        return [StopExecutorAction(
            controller_id=self.config.id,
            executor_id=executor.id) for executor in self.executors_info]

    async def update_processed_data(self):
        candles_df = self.market_data_provider.get_candles_df(self.config.connector_name, self.config.trading_pair,
                                                              "1m", 1000)

        returns_reversion_window = candles_df["close"].pct_change().tail(self.config.reversion_window).sum()
        high_reversion_window = candles_df["high"].tail(self.config.reversion_window).max()
        low_reversion_window = candles_df["low"].tail(self.config.reversion_window).min()
        returns_hedge_reversion_window = candles_df["close"].pct_change().tail(self.config.hedge_reversion_window).sum()
        long_bep, long_size = self.current_bep_and_size_long()
        short_bep, short_size = self.current_bep_and_size_short()
        mid_price = self.get_mid_price()
        current_imbalance = long_size - short_size
        cum_amount_minus_imbalance = [amount - abs(current_imbalance) - self.config.min_imbalance_to_hedge_quote for
                                      amount in self.cumulative_amounts_quote]
        next_level = next((i for i, amount in enumerate(cum_amount_minus_imbalance) if amount > 0), None)
        amount_next_level = self.amounts_quote[next_level] if next_level is not None else None
        spread_next_level5 = self.config.spreads[next_level] if next_level is not None else None
        long_bep_to_mid = self.current_distance_to_bep(long_bep, mid_price)
        short_bep_to_mid = self.current_distance_to_bep(short_bep, mid_price)
        if current_imbalance > self.config.min_imbalance_to_hedge_quote:
            balance_price = self.get_price_to_move_bep(short_bep, short_size, long_bep, current_imbalance)
            hedge_price = balance_price * (1 + self.config.hedge_profit_pct)
            hedge_price_passed = hedge_price < mid_price
        elif current_imbalance < -self.config.min_imbalance_to_hedge_quote:
            balance_price = self.get_price_to_move_bep(long_bep, long_size, short_bep, -current_imbalance)
            hedge_price = balance_price * (1 - self.config.hedge_profit_pct)
            hedge_price_passed = hedge_price > mid_price
        else:
            balance_price = None
            hedge_price_passed = False
            hedge_price = None
        active_long_order_placed = self.active_long_executors_order_placed()
        active_short_order_placed = self.active_short_executors_order_placed()
        self.processed_data = {
            "active_long_order_placed": active_long_order_placed,
            "active_short_order_placed": active_short_order_placed,
            "long_bep": long_bep,
            "long_size": long_size,
            "short_bep": short_bep,
            "short_size": short_size,
            "total_volume": long_size + short_size,
            "current_pnl": self.current_pnl(),
            "current_imbalance": current_imbalance,
            "balance_price": balance_price,
            "mid_price": mid_price,
            "next_level": next_level,
            "amount_next_level": amount_next_level,
            "spread_next_level": spread_next_level,
            "long_bep_to_mid": long_bep_to_mid,
            "short_bep_to_mid": short_bep_to_mid,
            "hedge_price": hedge_price,
            "hedge_price_passed": hedge_price_passed,
            "reverting_up": returns_reversion_window > 0 and low_reversion_window < mid_price,
            "reverting_down": returns_reversion_window < 0 and high_reversion_window > mid_price,
            "hedge_reverting_up": returns_hedge_reversion_window > 0,
            "hedge_reverting_down": returns_hedge_reversion_window < 0
        }

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def to_format_status(self) -> List[str]:
        return [
            f"Long BEP: {self.processed_data['long_bep']} | Size: {self.processed_data['long_size']}",
            f"Short BEP: {self.processed_data['short_bep']} | Size: {self.processed_data['short_size']}",
            f"long_bep_to_mid: {self.processed_data['long_bep_to_mid']} | short_bep_to_mid {self.processed_data['short_bep_to_mid']}"
            f"Total Volume: {self.processed_data['total_volume']}",
            f"Current Imbalance: {self.processed_data['current_imbalance']} | Current PnL: {self.processed_data['current_pnl']}",
            f"Balance Price: {self.processed_data['balance_price']} | Mid Price: {self.processed_data['mid_price']}",
            f"Next Level: {self.processed_data['next_level']}",
            f"Amount Next Level: {self.processed_data['amount_next_level']} | Spread Next Level: {self.processed_data['spread_next_level']}",
            f"Hedge Price Passed: {self.processed_data['hedge_price_passed']} | Hedge Price: {self.processed_data['hedge_price']}",
            f"Reverting Up: {self.processed_data['reverting_up']} | Reverting Down: {self.processed_data['reverting_down']}"
        ]
