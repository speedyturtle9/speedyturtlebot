# strategies/trend_following.py

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import logging
from .base_strategy import BaseStrategy
from trading_bot.utils.config import STOP_LOSS_PERCENT

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, short_period=20, long_period=50, rsi_period=14):
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period
        self.rsi_period = rsi_period

    def get_signal(self, df, instrument):
        try:
            precision = self._get_price_precision(instrument)
            
            # Calculate indicators
            df['sma_short'] = SMAIndicator(close=df['close'], window=self.short_period).sma_indicator()
            df['sma_long'] = SMAIndicator(close=df['close'], window=self.long_period).sma_indicator()
            df['rsi'] = RSIIndicator(close=df['close'], window=self.rsi_period).rsi()
            
            # MACD
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            current_price = df['close'].iloc[-1]
            
            # Generate signals
            signal = None
            if (df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1] and 
                df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and 
                df['rsi'].iloc[-1] > 50):
                signal = "BUY"
                stop_loss = self.round_price(current_price * (1 - STOP_LOSS_PERCENT), instrument)
                take_profit = self.round_price(current_price * (1 + STOP_LOSS_PERCENT * 2), instrument)
            
            elif (df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1] and 
                  df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and 
                  df['rsi'].iloc[-1] < 50):
                signal = "SELL"
                stop_loss = self.round_price(current_price * (1 + STOP_LOSS_PERCENT), instrument)
                take_profit = self.round_price(current_price * (1 - STOP_LOSS_PERCENT * 2), instrument)
            
            logging.info(f"Trend Following Analysis - Price: {current_price:.{precision}f}, Signal: {signal}")
            return signal, stop_loss if signal else None, take_profit if signal else None
            
        except Exception as e:
            logging.error(f"Error in Trend Following strategy: {str(e)}")
            return None, None, None