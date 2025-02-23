# strategies/support_resistance.py

import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
import logging
from .base_strategy import BaseStrategy
from trading_bot.utils.config import STOP_LOSS_PERCENT

class SupportResistanceStrategy(BaseStrategy):
    def __init__(self, period=20, num_points=5):
        super().__init__()
        self.period = period
        self.num_points = num_points

    def find_support_resistance(self, df):
        highs = []
        lows = []
        
        for i in range(self.num_points, len(df) - self.num_points):
            # Check for support
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, self.num_points+1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, self.num_points+1)):
                lows.append(df['low'].iloc[i])
            
            # Check for resistance
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, self.num_points+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, self.num_points+1)):
                highs.append(df['high'].iloc[i])
        
        return np.array(lows), np.array(highs)

    def get_signal(self, df, instrument):
        try:
            precision = self._get_price_precision(instrument)
            
            # Find support and resistance levels
            supports, resistances = self.find_support_resistance(df)
            
            if len(supports) == 0 or len(resistances) == 0:
                return None, None, None
            
            current_price = df['close'].iloc[-1]
            
            # Calculate ATR for dynamic stop loss
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.period)
            current_atr = atr.average_true_range().iloc[-1]
            
            # Find nearest support and resistance
            nearest_support = supports[np.abs(supports - current_price).argmin()]
            nearest_resistance = resistances[np.abs(resistances - current_price).argmin()]
            
            signal = None
            if current_price <= nearest_support + current_atr:  # Near support
                signal = "BUY"
                stop_loss = self.round_price(nearest_support - current_atr, instrument)
                take_profit = self.round_price(nearest_resistance, instrument)
            
            elif current_price >= nearest_resistance - current_atr:  # Near resistance
                signal = "SELL"
                stop_loss = self.round_price(nearest_resistance + current_atr, instrument)
                take_profit = self.round_price(nearest_support, instrument)
            
            logging.info(f"Support/Resistance Analysis - Price: {current_price:.{precision}f}, Signal: {signal}")
            return signal, stop_loss if signal else None, take_profit if signal else None
            
        except Exception as e:
            logging.error(f"Error in Support/Resistance strategy: {str(e)}")
            return None, None, None