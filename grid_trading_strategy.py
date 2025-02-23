# strategies/grid_trading_strategy.py

import logging
import pandas as pd
import numpy as np
from.base_strategy import BaseStrategy
from trading_bot.utils.config import STRATEGY_PARAMS

class GridTradingStrategy(BaseStrategy):
    def __init__(self):
        """Initialize Grid Trading Strategy"""
        super().__init__()
        self.params = STRATEGY_PARAMS['GRID']
        self.name = "Grid Trading"
        self.grid_levels = []
        logging.info(f"Initialized {self.name} Strategy")

    def get_signal(self, data, instrument):
        """Generate trading signals based on grid levels"""
        try:
            if data is None or data.empty:
                return {
                    'signal': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'analysis': "No data available",
                    'potential_directions': []
                }

            # Calculate grid levels if not already calculated
            if not self.grid_levels:
                self.calculate_grid_levels(data)
            
            if not self.grid_levels:
                return {
                    'signal': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'analysis': "No grid levels available",
                    'potential_directions': []
                }

            current_price = data['close'].iloc[-1]
            
            # Find nearest grid levels
            buy_levels = [level for level in self.grid_levels if level < current_price]
            sell_levels = [level for level in self.grid_levels if level > current_price]
            
            nearest_buy_level = max(buy_levels) if buy_levels else None
            nearest_sell_level = min(sell_levels) if sell_levels else None
            
            # Initialize variables
            signal = None
            stop_loss = None
            take_profit = None
            
            # Generate signals based on price position relative to grid levels
            if nearest_buy_level is not None and (current_price - nearest_buy_level) / nearest_buy_level < self.params['grid_threshold']:
                signal = "BUY"
                stop_loss = nearest_buy_level * (1 - self.params['stop_loss_pct'])
                take_profit = nearest_sell_level if nearest_sell_level else current_price * (1 + self.params['take_profit_pct'])
            
            elif nearest_sell_level is not None and (nearest_sell_level - current_price) / current_price < self.params['grid_threshold']:
                signal = "SELL"
                stop_loss = nearest_sell_level * (1 + self.params['stop_loss_pct'])
                take_profit = nearest_buy_level if nearest_buy_level else current_price * (1 - self.params['take_profit_pct'])

            # Prepare analysis message with safe formatting
            analysis = "Grid Trading Analysis:\n"
            analysis += f"Current Price: {current_price:.5f}\n"
            analysis += f"Nearest Buy Level: {nearest_buy_level:.5f if nearest_buy_level is not None else 'None'}\n"
            analysis += f"Nearest Sell Level: {nearest_sell_level:.5f if nearest_sell_level is not None else 'None'}\n"
            analysis += f"Total Grid Levels: {len(self.grid_levels)}"

            return {
                'signal': signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'analysis': analysis,
                'potential_directions': self.get_potential_directions(data)
            }

        except Exception as e:
            logging.error(f"Error in grid trading signal generation: {str(e)}")
            return {
                'signal': None,
                'stop_loss': None,
                'take_profit': None,
                'analysis': f"Error: {str(e)}",
                'potential_directions': []
            }

    def calculate_grid_levels(self, data):
        """Calculate grid levels based on price range"""
        try:
            # Calculate price range
            high = data['high'].max()
            low = data['low'].min()
            price_range = high - low
            
            # Calculate grid size
            grid_size = price_range / (self.params['grid_levels'] - 1)
            
            # Generate grid levels
            self.grid_levels = [
                low + (i * grid_size)
                for i in range(self.params['grid_levels'])
            ]
            
            logging.info(f"Calculated {len(self.grid_levels)} grid levels")
            
        except Exception as e:
            logging.error(f"Error calculating grid levels: {str(e)}")
            self.grid_levels = []

    def get_potential_directions(self, data):
        """Get potential movement directions"""
        try:
            directions = []
            if data is None or data.empty or not self.grid_levels:
                return directions

            current_price = data['close'].iloc[-1]
            
            # Find nearest grid levels
            buy_levels = [level for level in self.grid_levels if level < current_price]
            sell_levels = [level for level in self.grid_levels if level > current_price]
            
            nearest_buy_level = max(buy_levels) if buy_levels else None
            nearest_sell_level = min(sell_levels) if sell_levels else None

            # Check buy conditions
            if nearest_buy_level is not None:
                if (current_price - nearest_buy_level) / nearest_buy_level < self.params['grid_threshold']:
                    directions.append("BUY")

            # Check sell conditions
            if nearest_sell_level is not None:
                if (nearest_sell_level - current_price) / current_price < self.params['grid_threshold']:
                    directions.append("SELL")

            return directions

        except Exception as e:
            logging.error(f"Error getting potential directions: {str(e)}")
            return []

    def validate_grid_levels(self):
        """Validate grid levels"""
        try:
            if not self.grid_levels:
                return False

            # Check if grid levels are properly spaced
            grid_distances = np.diff(self.grid_levels)
            avg_distance = np.mean(grid_distances)
            
            # Check if grid distances are relatively uniform
            distance_variation = np.std(grid_distances) / avg_distance
            if distance_variation > 0.1:  # Allow 10% variation
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating grid levels: {str(e)}")
            return False

    def adjust_grid_levels(self, data):
        """Dynamically adjust grid levels based on market conditions"""
        try:
            if data is None or data.empty:
                return

            # Calculate volatility
            volatility = data['close'].pct_change().std()
            
            # Adjust grid threshold based on volatility
            if volatility > self.params['volatility_threshold']:
                self.params['grid_threshold'] *= 1.5  # Increase threshold in high volatility
            else:
                self.params['grid_threshold'] = self.params['default_grid_threshold']

            # Recalculate grid levels if needed
            if not self.validate_grid_levels():
                self.calculate_grid_levels(data)

        except Exception as e:
            logging.error(f"Error adjusting grid levels: {str(e)}")
