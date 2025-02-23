# strategies/momentum_strategy.py

import logging
import pandas as pd
import numpy as np
from.base_strategy import BaseStrategy
from trading_bot.utils.config import STRATEGY_PARAMS

class MomentumStrategy(BaseStrategy):
    def __init__(self):
        """Initialize Momentum Strategy"""
        super().__init__()
        self.params = STRATEGY_PARAMS['MOMENTUM']
        self.name = "Momentum"
        logging.info(f"Initialized {self.name} Strategy")

    def get_signal(self, data, instrument):
        """Generate trading signals based on momentum indicators"""
        try:
            if data is None or data.empty:
                return {
                    'signal': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'analysis': None,
                    'potential_directions': []
                }

            # Calculate momentum indicators
            data = self.calculate_indicators(data)
            
            # Get current values
            current_price = data['close'].iloc[-1]
            current_rsi = data['rsi'].iloc[-1]
            current_macd = data['macd'].iloc[-1]
            current_macd_signal = data['macd_signal'].iloc[-1]
            
            # Initialize variables
            signal = None
            stop_loss = None
            take_profit = None
            
            # Generate trading signals
            if (current_rsi < self.params['rsi_oversold'] and 
                current_macd > current_macd_signal):
                signal = "BUY"
                stop_loss = current_price * (1 - self.params['stop_loss_pct'])
                take_profit = current_price * (1 + self.params['take_profit_pct'])
                
            elif (current_rsi > self.params['rsi_overbought'] and 
                  current_macd < current_macd_signal):
                signal = "SELL"
                stop_loss = current_price * (1 + self.params['stop_loss_pct'])
                take_profit = current_price * (1 - self.params['take_profit_pct'])

            # Prepare analysis message
            analysis = (
                f"Momentum Analysis:\n"
                f"RSI: {current_rsi:.2f}\n"
                f"MACD: {current_macd:.5f}\n"
                f"MACD Signal: {current_macd_signal:.5f}\n"
                f"Current Price: {current_price:.5f}"
            )

            potential_directions = self.get_potential_directions(data)

            return {
                'signal': signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'analysis': analysis,
                'potential_directions': potential_directions
            }

        except Exception as e:
            logging.error(f"Error in momentum signal generation: {str(e)}")
            return {
                'signal': None,
                'stop_loss': None,
                'take_profit': None,
                'analysis': f"Error: {str(e)}",
                'potential_directions': []
            }

    def calculate_indicators(self, data):
        """Calculate momentum indicators"""
        try:
            df = data.copy()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['close'].ewm(span=self.params['macd_fast'], adjust=False).mean()
            exp2 = df['close'].ewm(span=self.params['macd_slow'], adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=self.params['macd_signal'], adjust=False).mean()
            
            return df

        except Exception as e:
            logging.error(f"Error calculating momentum indicators: {str(e)}")
            return data

    def get_potential_directions(self, data):
        """Get potential movement directions"""
        try:
            directions = []
            if data is None or data.empty:
                return directions

            # Get latest indicator values
            current_rsi = data['rsi'].iloc[-1]
            current_macd = data['macd'].iloc[-1]
            current_macd_signal = data['macd_signal'].iloc[-1]

            # Check RSI conditions
            if current_rsi < self.params['rsi_oversold']:
                directions.append("BUY")
            elif current_rsi > self.params['rsi_overbought']:
                directions.append("SELL")

            # Check MACD conditions
            if current_macd > current_macd_signal:
                directions.append("BUY")
            elif current_macd < current_macd_signal:
                directions.append("SELL")

            return list(set(directions))  # Remove duplicates

        except Exception as e:
            logging.error(f"Error getting potential directions: {str(e)}")
            return []

    def validate_signal(self, data, signal):
        """Validate trading signal with additional conditions"""
        try:
            if signal is None:
                return False

            # Get latest values
            current_price = data['close'].iloc[-1]
            current_volume = data['volume'].iloc[-1] if 'volume' in data else None

            # Volume validation if available
            if current_volume is not None:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                if current_volume < avg_volume * 0.7:  # Volume should be at least 70% of average
                    return False

            # Trend validation using simple moving averages
            short_ma = data['close'].rolling(window=10).mean().iloc[-1]
            long_ma = data['close'].rolling(window=30).mean().iloc[-1]

            if signal == "BUY" and short_ma < long_ma:
                return False
            elif signal == "SELL" and short_ma > long_ma:
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating signal: {str(e)}")
            return False
