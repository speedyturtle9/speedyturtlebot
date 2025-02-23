# utils/position_manager.py

import logging
from datetime import datetime
from trading_bot.utils.config import (
    RISK_LEVELS,
    CURRENT_RISK_PROFILE,
    INSTRUMENT_SETTINGS,
    MAX_TRADES_PER_INSTRUMENT
)

class PositionManager:
    def __init__(self, oanda_api, risk_manager):
        """Initialize Position Manager"""
        try:
            self.api = oanda_api
            self.risk_manager = risk_manager
            self.positions = {}
            self.position_history = []
            self.risk_settings = RISK_LEVELS[CURRENT_RISK_PROFILE]
            logging.info("Position Manager initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Position Manager: {str(e)}")
            raise

    def open_position(self, instrument, signal_type, size, stop_loss=None, take_profit=None):
        """Open a new position"""
        try:
            # Validate position parameters
            if not self.validate_position_parameters(instrument, signal_type, size):
                return None

            # Create order
            order_response = self.api.create_order(
                instrument=instrument,
                units=size if signal_type == "BUY" else -size,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit
            )

            if order_response and 'orderFillTransaction' in order_response:
                position_id = order_response['orderFillTransaction']['id']
                
                # Record position details
                position_details = {
                    'id': position_id,
                    'instrument': instrument,
                    'type': signal_type,
                    'size': size,
                    'entry_price': float(order_response['orderFillTransaction']['price']),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'open_time': datetime.now(),
                    'status': 'OPEN'
                }
                
                self.positions[position_id] = position_details
                logging.info(f"Position opened: {position_id} for {instrument}")
                return position_id
            
            return None

        except Exception as e:
            logging.error(f"Error opening position: {str(e)}")
            return None

    def close_position(self, position_id, reason="manual"):
        """Close an existing position"""
        try:
            if position_id not in self.positions:
                logging.warning(f"Position not found: {position_id}")
                return False

            position = self.positions[position_id]
            
            # Close the position
            close_response = self.api.close_position(position['instrument'])
            
            if close_response and 'orderFillTransaction' in close_response:
                # Update position details
                position['close_price'] = float(close_response['orderFillTransaction']['price'])
                position['close_time'] = datetime.now()
                position['status'] = 'CLOSED'
                position['close_reason'] = reason
                
                # Calculate P/L
                position['pnl'] = self.calculate_pnl(position)
                
                # Move to history
                self.position_history.append(position)
                del self.positions[position_id]
                
                logging.info(f"Position closed: {position_id}, P/L: {position['pnl']}")
                return True
            
            return False

        except Exception as e:
            logging.error(f"Error closing position: {str(e)}")
            return False

    def modify_position(self, position_id, stop_loss=None, take_profit=None):
        """Modify an existing position"""
        try:
            if position_id not in self.positions:
                return False

            position = self.positions[position_id]
            modifications = {}

            if stop_loss is not None:
                modifications['stopLoss'] = {
                    'price': str(stop_loss),
                    'timeInForce': 'GTC'
                }

            if take_profit is not None:
                modifications['takeProfit'] = {
                    'price': str(take_profit),
                    'timeInForce': 'GTC'
                }

            if modifications:
                response = self.api.modify_order(position_id, modifications)
                if response:
                    position['stop_loss'] = stop_loss if stop_loss is not None else position['stop_loss']
                    position['take_profit'] = take_profit if take_profit is not None else position['take_profit']
                    logging.info(f"Position {position_id} modified")
                    return True

            return False

        except Exception as e:
            logging.error(f"Error modifying position: {str(e)}")
            return False

    def validate_position_parameters(self, instrument, signal_type, size):
        """Validate position parameters"""
        try:
            # Check instrument limits
            instrument_positions = len([p for p in self.positions.values() 
                                     if p['instrument'] == instrument])
            if instrument_positions >= MAX_TRADES_PER_INSTRUMENT:
                logging.warning(f"Maximum positions reached for {instrument}")
                return False

            # Validate size
            if not self.risk_manager.validate_position_size(instrument, size):
                return False

            # Check if instrument is tradeable
            current_price = self.api.get_current_price(instrument)
            if not current_price:
                logging.warning(f"Could not get current price for {instrument}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating position parameters: {str(e)}")
            return False

    def calculate_pnl(self, position):
        """Calculate position P/L"""
        try:
            direction_multiplier = 1 if position['type'] == "BUY" else -1
            price_difference = position['close_price'] - position['entry_price']
            return direction_multiplier * price_difference * position['size']
        except Exception as e:
            logging.error(f"Error calculating P/L: {str(e)}")
            return 0

    def get_position_exposure(self, instrument):
        """Get current exposure for an instrument"""
        try:
            exposure = sum(
                p['size'] if p['type'] == "BUY" else -p['size']
                for p in self.positions.values()
                if p['instrument'] == instrument
            )
            return exposure
        except Exception as e:
            logging.error(f"Error calculating position exposure: {str(e)}")
            return 0

    def get_open_positions(self):
        """Get all open positions"""
        try:
            return {
                'count': len(self.positions),
                'positions': [
                    {
                        'id': p_id,
                        'instrument': p['instrument'],
                        'type': p['type'],
                        'size': p['size'],
                        'entry_price': p['entry_price'],
                        'current_price': self.api.get_current_price(p['instrument']),
                        'pnl': self.calculate_unrealized_pnl(p)
                    }
                    for p_id, p in self.positions.items()
                ]
            }
        except Exception as e:
            logging.error(f"Error getting open positions: {str(e)}")
            return {'count': 0, 'positions': []}

    def calculate_unrealized_pnl(self, position):
        """Calculate unrealized P/L for a position"""
        try:
            current_price = self.api.get_current_price(position['instrument'])
            if current_price:
                direction_multiplier = 1 if position['type'] == "BUY" else -1
                price_difference = current_price['bid'] - position['entry_price']
                return direction_multiplier * price_difference * position['size']
            return 0
        except Exception as e:
            logging.error(f"Error calculating unrealized P/L: {str(e)}")
            return 0

    def check_stop_levels(self):
        """Check and update stop levels for all positions"""
        try:
            for position_id, position in list(self.positions.items()):
                current_price = self.api.get_current_price(position['instrument'])
                if not current_price:
                    continue

                price = current_price['bid'] if position['type'] == "BUY" else current_price['ask']
                
                # Check stop loss
                if position['stop_loss'] is not None:
                    if (position['type'] == "BUY" and price <= position['stop_loss']) or \
                       (position['type'] == "SELL" and price >= position['stop_loss']):
                        self.close_position(position_id, reason="stop_loss")
                        continue

                # Check take profit
                if position['take_profit'] is not None:
                    if (position['type'] == "BUY" and price >= position['take_profit']) or \
                       (position['type'] == "SELL" and price <= position['take_profit']):
                        self.close_position(position_id, reason="take_profit")

        except Exception as e:
            logging.error(f"Error checking stop levels: {str(e)}")

    def get_position_summary(self):
        """Get summary of all positions"""
        try:
            return {
                'open_positions': len(self.positions),
                'total_exposure': sum(abs(p['size']) for p in self.positions.values()),
                'unrealized_pnl': sum(self.calculate_unrealized_pnl(p) for p in self.positions.values()),
                'position_history': len(self.position_history)
            }
        except Exception as e:
            logging.error(f"Error getting position summary: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup position manager resources"""
        try:
            # Close all open positions
            for position_id in list(self.positions.keys()):
                self.close_position(position_id, reason="cleanup")
            
            self.positions.clear()
            logging.info("Position Manager cleanup completed")
        except Exception as e:
            logging.error(f"Error in cleanup: {str(e)}")
