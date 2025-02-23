# strategies/mean_reversion_strategy.py

import logging
import numpy as np
import pandas as pd
from trading_bot.utils.config import STRATEGY_PARAMS

class MeanReversionStrategy:
    def __init__(self):
        """Initialize Mean Reversion Strategy"""
        self.params = STRATEGY_PARAMS['MEAN_REVERSION']
        self.name = "Mean Reversion"
        logging.info(f"Initialized {self.name} Strategy")

    def get_signal(self, data, instrument):
        """Generate trading signals based on mean reversion"""
        try:
            if data is None or data.empty:
                return {'signal': None, 'stop_loss': None, 'take_profit': None}

            # Calculate indicators
            data = self.calculate_indicators(data)
            
            # Get current values
            current_price = data['close'].iloc[-1]
            current_bb_upper = data['bb_upper'].iloc[-1]
            current_bb_lower = data['bb_lower'].iloc[-1]
            current_rsi = data['rsi'].iloc[-1]
            
            # Generate signal
            signal = None
            stop_loss = None
            take_profit = None
            
            # Check oversold conditions
            if (current_price <= current_bb_lower and 
                current_rsi <= self.params['rsi_oversold']):
                signal = "BUY"
                stop_loss = current_price - (current_price * 0.01)  # 1% stop loss
                take_profit = current_price + (current_price * 0.02)  # 2% take profit
                
            # Check overbought conditions
            elif (current_price >= current_bb_upper and 
                  current_rsi >= self.params['rsi_overbought']):
                signal = "SELL"
                stop_loss = current_price + (current_price * 0.01)  # 1% stop loss
                take_profit = current_price - (current_price * 0.02)  # 2% take profit

            # Add additional signal validation
            if signal:
                if not self.validate_signal(data, signal):
                    return {'signal': None, 'stop_loss': None, 'take_profit': None}

            return {
                'signal': signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'indicators': {
                    'rsi': current_rsi,
                    'bb_upper': current_bb_upper,
                    'bb_lower': current_bb_lower
                }
            }

        except Exception as e:
            logging.error(f"Error generating mean reversion signal: {str(e)}")
            return {'signal': None, 'stop_loss': None, 'take_profit': None}

    def calculate_indicators(self, data):
        """Calculate technical indicators for mean reversion"""
        try:
            df = data.copy()
            
            # Calculate Bollinger Bands
            df['sma'] = df['close'].rolling(window=self.params['period']).mean()
            df['std'] = df['close'].rolling(window=self.params['period']).std()
            df['bb_upper'] = df['sma'] + (df['std'] * self.params['std_dev'])
            df['bb_lower'] = df['sma'] - (df['std'] * self.params['std_dev'])
            
            # Calculate RSI
            df['rsi'] = self.calculate_rsi(df['close'], self.params['rsi_period'])
            
            # Calculate volatility
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # Calculate price momentum
            df['momentum'] = df['close'].pct_change(periods=10)
            
            # Calculate Average True Range (ATR)
            df['atr'] = self.calculate_atr(df)
            
            return df

        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            return data

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi

        except Exception as e:
            logging.error(f"Error calculating RSI: {str(e)}")
            return None

    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close'].shift()
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr

        except Exception as e:
            logging.error(f"Error calculating ATR: {str(e)}")
            return None

    def validate_signal(self, data, signal):
        """Validate trading signal with additional conditions"""
        try:
            # Get recent data
            recent_data = data.tail(20)
            
            # Check volume trend if available
            if 'volume' in recent_data.columns:
                avg_volume = recent_data['volume'].mean()
                current_volume = recent_data['volume'].iloc[-1]
                if current_volume < avg_volume * 0.7:  # Volume should be at least 70% of average
                    return False
            
            # Check volatility
            current_volatility = recent_data['volatility'].iloc[-1]
            if current_volatility > 0.02:  # More than 2% volatility might be too risky
                return False
            
            # Check momentum alignment
            current_momentum = recent_data['momentum'].iloc[-1]
            if (signal == "BUY" and current_momentum < 0) or \
               (signal == "SELL" and current_momentum > 0):
                return False
            
            # Check if price is not too far from moving average
            current_price = recent_data['close'].iloc[-1]
            sma = recent_data['sma'].iloc[-1]
            price_deviation = abs(current_price - sma) / sma
            if price_deviation > 0.03:  # More than 3% deviation might be too risky
                return False
            
            return True

        except Exception as e:
            logging.error(f"Error validating signal: {str(e)}")
            return False

    def calculate_position_size(self, account_balance, current_price, atr):
        """Calculate position size based on ATR"""
        try:
            risk_per_trade = 0.02  # 2% risk per trade
            risk_amount = account_balance * risk_per_trade
            
            # Use ATR to determine stop loss distance
            stop_loss_distance = atr * 2  # 2 ATR units for stop loss
            
            if stop_loss_distance == 0:
                return 0
                
            position_size = risk_amount / stop_loss_distance
            return position_size

        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0

    def adjust_for_trend(self, data, signal):
        """Adjust signal based on overall trend"""
        try:
            # Calculate longer-term trend
            long_sma = data['close'].rolling(window=50).mean()
            medium_sma = data['close'].rolling(window=20).mean()
            
            current_price = data['close'].iloc[-1]
            current_long_sma = long_sma.iloc[-1]
            current_medium_sma = medium_sma.iloc[-1]
            
            # Check trend alignment
            if signal == "BUY":
                if current_price < current_long_sma and current_price < current_medium_sma:
                    return None  # Don't buy against strong downtrend
            elif signal == "SELL":
                if current_price > current_long_sma and current_price > current_medium_sma:
                    return None  # Don't sell against strong uptrend
                    
            return signal

        except Exception as e:
            logging.error(f"Error adjusting for trend: {str(e)}")
            return signal

    def get_strategy_metrics(self):
        """Get strategy performance metrics"""
        try:
            return {
                'name': self.name,
                'parameters': self.params,
                'description': "Mean reversion strategy using Bollinger Bands and RSI"
            }
        except Exception as e:
            logging.error(f"Error getting strategy metrics: {str(e)}")
            return None

    def get_graph_analysis(self, data, instrument):
        """Get graph analysis for the instrument"""
        try:
            # Implement graph analysis logic here
            return "Graph analysis details"

        except Exception as e:
            logging.error(f"Error getting graph analysis: {str(e)}")
            return ""

    def get_potential_directions(self, data, instrument):
        """Get potential movement directions for the instrument"""
        try:
            # Implement potential movement directions logic here
            return "Potential movement directions"

        except Exception as e:
            logging.error(f"Error getting potential movement directions: {str(e)}")
            return ""
