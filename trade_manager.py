# utils/trade_manager.py

import logging
from datetime import datetime, timedelta
from trading_bot.utils.config import (
    RISK_LEVELS,
    CURRENT_RISK_PROFILE,
    MAX_TRADES_PER_INSTRUMENT,
    MAX_DAILY_TRADES,
    MAX_DAILY_LOSS_PERCENT
)

class TradeManager:
    def __init__(self, oanda_api, risk_manager, position_manager):
        """Initialize Trade Manager"""
        try:
            self.api = oanda_api
            self.risk_manager = risk_manager
            self.position_manager = position_manager
            self.risk_settings = RISK_LEVELS[CURRENT_RISK_PROFILE]
            self.trades = {}
            self.trade_history = []
            self.daily_stats = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit': 0,
                'loss': 0
            }
            self.last_reset = datetime.now().date()
            logging.info("Trade Manager initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Trade Manager: {str(e)}")
            raise

    def execute_trade(self, instrument, signal, strategy_name):
        """Execute a trade based on signal"""
        try:
            # Validate trading conditions
            if not self.validate_trading_conditions(instrument):
                return None

            # Get current price
            current_price = self.api.get_current_price(instrument)
            if not current_price:
                logging.error(f"Could not get current price for {instrument}")
                return None

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                instrument=instrument,
                account_balance=self.get_account_balance(),
                current_price=current_price['ask'] if signal['signal'] == "BUY" else current_price['bid']
            )

            if not position_size:
                logging.warning(f"Invalid position size for {instrument}")
                return None

            # Create order
            units = position_size if signal['signal'] == "BUY" else -position_size
            order_response = self.api.create_order(
                instrument=instrument,
                units=units,
                stop_loss_price=signal['stop_loss'],
                take_profit_price=signal['take_profit']
            )

            if order_response and 'orderFillTransaction' in order_response:
                trade_id = order_response['orderFillTransaction']['id']
                
                # Record trade
                trade_details = {
                    'id': trade_id,
                    'instrument': instrument,
                    'direction': signal['signal'],
                    'size': abs(units),
                    'entry_price': float(order_response['orderFillTransaction']['price']),
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'strategy': strategy_name,
                    'open_time': datetime.now(),
                    'status': 'OPEN'
                }
                
                self.trades[trade_id] = trade_details
                self.update_daily_stats('trades')
                
                logging.info(f"Trade executed for {instrument}: {trade_id}")
                return trade_id
            
            return None

        except Exception as e:
            logging.error(f"Error executing trade: {str(e)}")
            return None

    def close_trade(self, trade_id, reason="manual"):
        """Close an existing trade"""
        try:
            if trade_id not in self.trades:
                logging.warning(f"Trade not found: {trade_id}")
                return False

            trade = self.trades[trade_id]
            
            # Close position
            close_response = self.api.close_position(trade['instrument'])
            
            if close_response and 'orderFillTransaction' in close_response:
                # Calculate profit/loss
                close_price = float(close_response['orderFillTransaction']['price'])
                pnl = self.calculate_pnl(trade, close_price)
                
                # Update trade details
                trade['close_price'] = close_price
                trade['close_time'] = datetime.now()
                trade['pnl'] = pnl
                trade['status'] = 'CLOSED'
                trade['close_reason'] = reason
                
                # Update statistics
                self.update_trade_statistics(trade)
                
                # Move to history
                self.trade_history.append(trade)
                del self.trades[trade_id]
                
                logging.info(f"Trade closed: {trade_id}, P/L: {pnl}")
                return True
            
            return False

        except Exception as e:
            logging.error(f"Error closing trade: {str(e)}")
            return False

    def update_trade_statistics(self, trade):
        """Update trading statistics"""
        try:
            if trade['pnl'] > 0:
                self.update_daily_stats('wins')
                self.update_daily_stats('profit', trade['pnl'])
            else:
                self.update_daily_stats('losses')
                self.update_daily_stats('loss', abs(trade['pnl']))

        except Exception as e:
            logging.error(f"Error updating trade statistics: {str(e)}")

    def validate_trading_conditions(self, instrument):
        """Validate if trading conditions are met"""
        try:
            # Check daily limits
            if not self.check_daily_limits():
                return False

            # Check instrument-specific limits
            instrument_trades = len([t for t in self.trades.values() 
                                  if t['instrument'] == instrument])
            if instrument_trades >= MAX_TRADES_PER_INSTRUMENT:
                logging.warning(f"Maximum trades reached for {instrument}")
                return False

            # Check account status
            if not self.check_account_status():
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating trading conditions: {str(e)}")
            return False

    def check_daily_limits(self):
        """Check if daily trading limits have been reached"""
        try:
            current_date = datetime.now().date()
            
            # Reset daily stats if new day
            if current_date != self.last_reset:
                self.reset_daily_stats()
                self.last_reset = current_date

            # Check number of trades
            if self.daily_stats['trades'] >= MAX_DAILY_TRADES:
                logging.warning("Daily trade limit reached")
                return False

            # Check daily loss limit
            account_balance = self.get_account_balance()
            max_daily_loss = account_balance * (MAX_DAILY_LOSS_PERCENT / 100)
            
            if self.daily_stats['loss'] >= max_daily_loss:
                logging.warning("Daily loss limit reached")
                return False

            return True

        except Exception as e:
            logging.error(f"Error checking daily limits: {str(e)}")
            return False

    def check_account_status(self):
        """Check if account is suitable for trading"""
        try:
            account = self.api.get_account_summary()
            
            # Check margin level
            margin_available = float(account['account']['marginAvailable'])
            margin_used = float(account['account']['marginUsed'])
            
            if margin_used > 0:
                margin_ratio = margin_used / (margin_available + margin_used)
                if margin_ratio > 0.7:  # 70% margin usage
                    logging.warning("High margin usage")
                    return False

            return True

        except Exception as e:
            logging.error(f"Error checking account status: {str(e)}")
            return False

    def get_account_balance(self):
        """Get current account balance"""
        try:
            account = self.api.get_account_summary()
            return float(account['account']['balance'])
        except Exception as e:
            logging.error(f"Error getting account balance: {str(e)}")
            return 0

    def calculate_pnl(self, trade, close_price):
        """Calculate profit/loss for a trade"""
        try:
            direction_multiplier = 1 if trade['direction'] == "BUY" else -1
            price_difference = close_price - trade['entry_price']
            return direction_multiplier * price_difference * trade['size']
        except Exception as e:
            logging.error(f"Error calculating P/L: {str(e)}")
            return 0

    def update_daily_stats(self, key, value=1):
        """Update daily statistics"""
        try:
            if key in self.daily_stats:
                if isinstance(value, (int, float)):
                    self.daily_stats[key] += value
                else:
                    logging.warning(f"Invalid value type for daily stats update: {type(value)}")
        except Exception as e:
            logging.error(f"Error updating daily stats: {str(e)}")

    def reset_daily_stats(self):
        """Reset daily statistics"""
        try:
            self.daily_stats = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit': 0,
                'loss': 0
            }
            logging.info("Daily statistics reset")
        except Exception as e:
            logging.error(f"Error resetting daily stats: {str(e)}")

    def get_trade_summary(self):
        """Get summary of all trades"""
        try:
            return {
                'active_trades': len(self.trades),
                'total_trades': len(self.trade_history) + len(self.trades),
                'daily_stats': self.daily_stats,
                'open_positions': [
                    {
                        'instrument': t['instrument'],
                        'direction': t['direction'],
                        'pnl': self.calculate_pnl(
                            t, 
                            self.api.get_current_price(t['instrument'])['bid']
                        )
                    }
                    for t in self.trades.values()
                ]
            }
        except Exception as e:
            logging.error(f"Error getting trade summary: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup trade manager resources"""
        try:
            # Close all open trades
            for trade_id in list(self.trades.keys()):
                self.close_trade(trade_id, reason="cleanup")
            
            self.trades.clear()
            logging.info("Trade Manager cleanup completed")
        except Exception as e:
            logging.error(f"Error in cleanup: {str(e)}")
