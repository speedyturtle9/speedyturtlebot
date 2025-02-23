# strategies/breakout_strategy.py

import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
import logging
from .base_strategy import BaseStrategy
from trading_bot.utils.config import STOP_LOSS_PERCENT

class BreakoutStrategy(BaseStrategy):
    def __init__(self, period=20, multiplier=2):
        super().__init__()
        self.period = period
        self.multiplier = multiplier

    def get_signal(self, df, instrument):
        try:
            precision = self._get_price_precision(instrument)
            
            # Calculate Bollinger Bands
            bb = BollingerBands(close=df['close'], window=self.period, window_dev=self.multiplier)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            
            # Calculate ATR for dynamic stop loss
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.period)
            df['atr'] = atr.average_true_range()
            
            current_price = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            # Check for breakouts
            signal = None
            if current_price > df['bb_upper'].iloc[-2]:  # Breakout above
                signal = "BUY"
                stop_loss = self.round_price(current_price - current_atr * 2, instrument)
                take_profit = self.round_price(current_price + current_atr * 3, instrument)
            
            elif current_price < df['bb_lower'].iloc[-2]:  # Breakout below
                signal = "SELL"
                stop_loss = self.round_price(current_price + current_atr * 2, instrument)
                take_profit = self.round_price(current_price - current_atr * 3, instrument)
            
            logging.info(f"Breakout Analysis - Price: {current_price:.{precision}f}, Signal: {signal}")
            return signal, stop_loss if signal else None, take_profit if signal else None
            
        except Exception as e:
            logging.error(f"Error in Breakout strategy: {str(e)}")
            return None, None, None