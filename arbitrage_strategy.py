# strategies/arbitrage_strategy.py

import logging
import pandas as pd
from .base_strategy import BaseStrategy
from trading_bot.utils.config import STRATEGY_PARAMS

class ArbitrageStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.params = STRATEGY_PARAMS['ARBITRAGE']
        self.name = "Arbitrage"
        logging.info(f"Initialized {self.name} Strategy")

    def get_signal(self, data, instrument):
        try:
            if data is None or data.empty:
                return {'signal': None, 'analysis': None, 'potential_directions': None}

            # Calculate price differences between exchanges
            price_diff = self.calculate_price_difference(data)
            
            # Generate signal based on arbitrage opportunity
            signal = None
            stop_loss = None
            take_profit = None
            
            if price_diff > self.params['min_spread']:
                signal = "BUY"
                stop_loss = data['close'].iloc[-1] * (1 - self.params['stop_loss_pct'])
                take_profit = data['close'].iloc[-1] * (1 + self.params['take_profit_pct'])
            elif price_diff < -self.params['min_spread']:
                signal = "SELL"
                stop_loss = data['close'].iloc[-1] * (1 + self.params['stop_loss_pct'])
                take_profit = data['close'].iloc[-1] * (1 - self.params['take_profit_pct'])

            analysis = f"Price difference: {price_diff:.5f}, Threshold: {self.params['min_spread']:.5f}"
            potential_directions = ["BUY" if price_diff > 0 else "SELL"]

            return {
                'signal': signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'analysis': analysis,
                'potential_directions': potential_directions
            }

        except Exception as e:
            logging.error(f"Error in arbitrage signal generation: {str(e)}")
            return {'signal': None, 'analysis': None, 'potential_directions': None}

    def calculate_price_difference(self, data):
        """Calculate price differences between exchanges"""
        try:
            # Get current prices from OANDA
            current_price = self.api.get_current_price(data.name)
            if current_price:
                return current_price.get('ask', 0) - current_price.get('bid', 0)  # Use get() with default value to handle missing keys
            return 0
        except Exception as e:
              logging.error(f"Error calculating price difference: {str(e)}")
              return 0
