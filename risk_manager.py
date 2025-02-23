# utils/risk_manager.py

import logging
from datetime import datetime, timedelta
from trading_bot.utils.config import (
    RISK_LEVELS,
    CURRENT_RISK_PROFILE,
    MAX_DAILY_TRADES,
    MAX_DAILY_LOSS_PERCENT,
    MAX_TRADES_PER_INSTRUMENT,
    INSTRUMENT_SETTINGS
)

class RiskManager:
    def __init__(self):
        """Initialize Risk Manager"""
        try:
            self.risk_settings = RISK_LEVELS[CURRENT_RISK_PROFILE]
            self.daily_trades = {}
            self.daily_losses = {}
            self.instrument_trades = {}
            self.last_reset = datetime.now().date()
            self.position_sizes = {}
            logging.info(f"Risk Manager initialized with {CURRENT_RISK_PROFILE} profile")
        except Exception as e:
            logging.error(f"Failed to initialize Risk Manager: {str(e)}")
            raise

    def reset_daily_metrics(self):
        """Reset daily trading metrics"""
        try:
            current_date = datetime.now().date()
            if current_date != self.last_reset:
                self.daily_trades = {}
                self.daily_losses = {}
                self.last_reset = current_date
                logging.info("Daily metrics reset")
        except Exception as e:
            logging.error(f"Error resetting daily metrics: {str(e)}")

    def can_trade(self, instrument, account_balance):
        """Check if trading is allowed based on risk parameters"""
        try:
            self.reset_daily_metrics()
            
            # Check daily trade limit
            total_daily_trades = sum(self.daily_trades.values())
            if total_daily_trades >= MAX_DAILY_TRADES:
                logging.warning("Daily trade limit reached")
                return False

            # Check instrument-specific trade limit
            if self.instrument_trades.get(instrument, 0) >= MAX_TRADES_PER_INSTRUMENT:
                logging.warning(f"Trade limit reached for {instrument}")
                return False

            # Check daily loss limit
            total_daily_loss = sum(self.daily_losses.values())
            max_daily_loss = account_balance * (MAX_DAILY_LOSS_PERCENT / 100)
            if total_daily_loss >= max_daily_loss:
                logging.warning("Daily loss limit reached")
                return False

            return True

        except Exception as e:
            logging.error(f"Error checking trade permissions: {str(e)}")
            return False

    def calculate_position_size(self, instrument, account_balance, current_price):
        """Calculate appropriate position size"""
        try:
            # Get instrument settings
            settings = INSTRUMENT_SETTINGS.get(instrument, {})
            if not settings:
                logging.error(f"No settings found for {instrument}")
                return 0

            # Calculate base position size
            risk_amount = account_balance * (self.risk_settings['risk_per_trade'] / 100)
            
            # Adjust for instrument specifics
            pip_value = settings['pip_value']
            stop_loss_pips = self.risk_settings['stop_loss_pips']
            
            # Calculate position size
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Apply maximum position size limit
            position_size = min(position_size, self.risk_settings['max_position_size'])
            
            # Round to nearest standard lot size
            position_size = round(position_size / 1000) * 1000
            
            self.position_sizes[instrument] = position_size
            return position_size

        except Exception as e:
            logging.error(f"Error calculating position size for {instrument}: {str(e)}")
            return 0

    def calculate_stop_loss(self, instrument, entry_price, direction):
        """Calculate stop loss level"""
        try:
            settings = INSTRUMENT_SETTINGS.get(instrument, {})
            if not settings:
                return None

            pip_value = settings['pip_value']
            stop_loss_pips = self.risk_settings['stop_loss_pips']
            
            if direction == "BUY":
                stop_loss = entry_price - (stop_loss_pips * pip_value)
            else:
                stop_loss = entry_price + (stop_loss_pips * pip_value)
                
            return stop_loss

        except Exception as e:
            logging.error(f"Error calculating stop loss for {instrument}: {str(e)}")
            return None

    def calculate_take_profit(self, instrument, entry_price, direction):
        """Calculate take profit level"""
        try:
            settings = INSTRUMENT_SETTINGS.get(instrument, {})
            if not settings:
                return None

            pip_value = settings['pip_value']
            stop_loss_pips = self.risk_settings['stop_loss_pips']
            reward_ratio = 1.5  # Risk:Reward ratio
            
            if direction == "BUY":
                take_profit = entry_price + (stop_loss_pips * pip_value * reward_ratio)
            else:
                take_profit = entry_price - (stop_loss_pips * pip_value * reward_ratio)
                
            return take_profit

        except Exception as e:
            logging.error(f"Error calculating take profit for {instrument}: {str(e)}")
            return None

    def update_trade_metrics(self, instrument, profit_loss):
        """Update trading metrics after trade completion"""
        try:
            # Update daily trades
            self.daily_trades[instrument] = self.daily_trades.get(instrument, 0) + 1
            
            # Update instrument trades
            self.instrument_trades[instrument] = self.instrument_trades.get(instrument, 0) + 1
            
            # Update losses if applicable
            if profit_loss < 0:
                self.daily_losses[instrument] = self.daily_losses.get(instrument, 0) + abs(profit_loss)
                
            logging.info(f"Updated metrics for {instrument} - P/L: {profit_loss}")

        except Exception as e:
            logging.error(f"Error updating trade metrics: {str(e)}")

    def adjust_for_volatility(self, instrument, atr):
        """Adjust risk parameters based on volatility"""
        try:
            if not atr:
                return
                
            # Adjust position size based on volatility
            base_position = self.position_sizes.get(instrument, 0)
            if base_position > 0:
                volatility_factor = 1 - min(atr * 10, 0.5)  # Reduce size for high volatility
                adjusted_position = base_position * volatility_factor
                self.position_sizes[instrument] = adjusted_position
                
            logging.info(f"Adjusted position size for {instrument} based on volatility")

        except Exception as e:
            logging.error(f"Error adjusting for volatility: {str(e)}")

    def get_risk_metrics(self):
        """Get current risk metrics"""
        try:
            return {
                'daily_trades': sum(self.daily_trades.values()),
                'daily_losses': sum(self.daily_losses.values()),
                'instruments_traded': len(self.instrument_trades),
                'risk_profile': CURRENT_RISK_PROFILE,
                'position_sizes': self.position_sizes.copy()
            }
        except Exception as e:
            logging.error(f"Error getting risk metrics: {str(e)}")
            return {}

    def validate_trade(self, instrument, direction, size, entry_price, stop_loss, take_profit):
        """Validate trade parameters"""
        try:
            # Check position size
            if size <= 0 or size > self.risk_settings['max_position_size']:
                logging.warning(f"Invalid position size for {instrument}: {size}")
                return False

            # Check stop loss
            if not stop_loss or (direction == "BUY" and stop_loss >= entry_price) or \
               (direction == "SELL" and stop_loss <= entry_price):
                logging.warning(f"Invalid stop loss for {instrument}")
                return False

            # Check take profit
            if not take_profit or (direction == "BUY" and take_profit <= entry_price) or \
               (direction == "SELL" and take_profit >= entry_price):
                logging.warning(f"Invalid take profit for {instrument}")
                return False

            # Check risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(entry_price - take_profit)
            if risk == 0 or reward/risk < 1.5:
                logging.warning(f"Invalid risk-reward ratio for {instrument}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating trade parameters: {str(e)}")
            return False

    def cleanup(self):
        """Cleanup risk manager resources"""
        try:
            self.daily_trades.clear()
            self.daily_losses.clear()
            self.instrument_trades.clear()
            self.position_sizes.clear()
            logging.info("Risk Manager cleanup completed")
        except Exception as e:
            logging.error(f"Error in cleanup: {str(e)}")
