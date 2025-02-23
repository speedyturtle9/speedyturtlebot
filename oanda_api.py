import logging
from oandapyV20 import API
from oandapyV20.endpoints.pricing import PricingInfo
from oandapyV20.endpoints.orders import OrderCreate, OrderDetails
from oandapyV20.endpoints.positions import OpenPositions, PositionClose
from oandapyV20.endpoints.accounts import AccountSummary
from oandapyV20.endpoints.trades import TradeDetails
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import time
from trading_bot.utils.config import OANDA_API_KEY, OANDA_ACCOUNT_ID

class OandaAPI:
    def __init__(self) -> None:
        """Initialize OANDA API connection"""
        try:
            logging.info("Initializing OANDA API connection...")
            self.api = API(access_token=OANDA_API_KEY, environment="practice")
            self.account_id = OANDA_ACCOUNT_ID
            self.validate_connection()
            logging.info("OANDA API connection established successfully")
        except Exception as e:
            logging.error(f"Failed to initialize OANDA API: {str(e)}")
            raise

    def validate_connection(self) -> bool:
        """Validate API connection and credentials"""
        try:
            self.get_account_summary()
            return True
        except Exception as e:
            logging.error(f"API connection validation failed: {str(e)}")
            return False

    def get_account_summary(self) -> dict:
        """Get account summary information"""
        try:
            r = AccountSummary(self.account_id)
            response = self.api.request(r)
            logging.info("Retrieved account summary successfully")
            return response
        except Exception as e:
            logging.error(f"Error getting account summary: {str(e)}")
            raise

    def get_historical_data(self, instrument: str, count: int = 1000, granularity: str = 'H1') -> pd.DataFrame:
        """Get historical price data"""
        try:
            params = {
                "count": count,
                "granularity": granularity,
                "price": "M"  # Midpoint pricing
            }
            
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            response = self.api.request(r)
            
            if not response or 'candles' not in response:
                logging.error(f"Failed to retrieve historical data for {instrument}")
                return None
                
            data = []
            for candle in response['candles']:
                if candle['complete']:
                    data.append({
                        'time': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            if not data:
                logging.error(f"No historical data available for {instrument}")
                return None
                
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            logging.info(f"Retrieved {len(df)} candles for {instrument}")
            return df
            
        except Exception as e:
            logging.error(f"Error getting historical data for {instrument}: {str(e)}")
            return None

    def get_current_price(self, instrument: str) -> dict:
        """Get current price for an instrument"""
        try:
            params = {"instruments": instrument}
            r = PricingInfo(accountID=self.account_id, params=params)
            response = self.api.request(r)
            
            if response and 'prices' in response and len(response['prices']) > 0:
                return {
                    'bid': float(response['prices'][0]['bids'][0]['price']),
                    'ask': float(response['prices'][0]['asks'][0]['price'])
                }
            logging.error(f"Failed to retrieve current price for {instrument}")
            return None
        except Exception as e:
            logging.error(f"Error getting current price for {instrument}: {str(e)}")
            return None

    def create_order(self, instrument: str, units: int, stop_loss_price: float = None, take_profit_price: float = None) -> dict:
        """Create a new trading order"""
        try:
            precision = 3 if 'JPY' in instrument else 5
            
            data = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(int(units)),
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT"
                }
            }
            
            if stop_loss_price is not None:
                data["order"]["stopLossOnFill"] = {
                    "price": f"{float(stop_loss_price):.{precision}f}",
                    "timeInForce": "GTC"
                }
                
            if take_profit_price is not None:
                data["order"]["takeProfitOnFill"] = {
                    "price": f"{float(take_profit_price):.{precision}f}",
                    "timeInForce": "GTC"
                }
            
            r = OrderCreate(self.account_id, data=data)
            response = self.api.request(r)
            logging.info(f"Order created for {instrument}: {units} units")
            return response
            
        except Exception as e:
            logging.error(f"Error creating order for {instrument}: {str(e)}")
            raise

    def modify_order(self, order_id: str, modifications: dict) -> dict:
        """Modify an existing order"""
        try:
            r = OrderDetails(self.account_id, order_id)
            response = self.api.request(r)
            if response:
                # Apply modifications
                data = {"order": modifications}
                r = OrderCreate(self.account_id, data=data)
                return self.api.request(r)
            return None
        except Exception as e:
            logging.error(f"Error modifying order {order_id}: {str(e)}")
            return None

    def get_open_positions(self) -> dict:
        """Get all open positions"""
        try:
            r = OpenPositions(self.account_id)
            response = self.api.request(r)
            logging.info(f"Retrieved {len(response['positions'])} open positions")
            return response
        except Exception as e:
            logging.error(f"Error getting open positions: {str(e)}")
            raise

    def close_position(self, instrument: str) -> dict:
        """Close position for an instrument"""
        try:
            data = {"longUnits": "ALL"}
            r = PositionClose(self.account_id, instrument, data)
            response = self.api.request(r)
            logging.info(f"Closed position for {instrument}")
            return response
        except Exception as e:
            logging.error(f"Error closing position for {instrument}: {str(e)}")
            raise

    def get_trade_status(self, trade_id: str) -> dict:
        """Get status of a specific trade"""
        try:
            r = TradeDetails(self.account_id, trade_id)
            response = self.api.request(r)
            return response
        except Exception as e:
            logging.error(f"Error getting trade status for {trade_id}: {str(e)}")
            return None

    def get_instrument_properties(self, instrument: str) -> dict:
        """Get properties for an instrument"""
        try:
            params = {"instruments": instrument}
            r = PricingInfo(accountID=self.account_id, params=params)
            response = self.api.request(r)
            
            if response and 'prices' in response and len(response['prices']) > 0:
                price_data = response['prices'][0]
                return {
                    'pipLocation': price_data.get('pipLocation', -4),
                    'marginRate': price_data.get('marginRate', '0.05'),
                    'tradeUnitsPrecision': price_data.get('tradeUnitsPrecision', 0),
                    'minimumTradeSize': price_data.get('minimumTradeSize', '1'),
                    'pipValue': self.calculate_pip_value(instrument)
                }
            return None
            
        except Exception as e:
            logging.error(f"Error getting instrument properties for {instrument}: {str(e)}")
            return None

    def calculate_pip_value(self, instrument: str) -> float:
        """Calculate pip value for an instrument"""
        try:
            price = self.get_current_price(instrument)
            if not price:
                return None
                
            # Default pip value calculation
            pip_value = 0.0001
            if 'JPY' in instrument:
                pip_value = 0.01
                
            return pip_value
            
        except Exception as e:
            logging.error(f"Error calculating pip value for {instrument}: {str(e)}")
            return None

    def validate_instrument(self, instrument: str) -> bool:
        """Validate if an instrument is tradeable"""
        try:
            # Try to get current price
            price = self.get_current_price(instrument)
            if price is not None:
                return True
                
            # If price check fails, try to get historical data
            data = self.get_historical_data(instrument, count=1)
            return data is not None and not data.empty
            
        except Exception as e:
            logging.error(f"Error validating instrument {instrument}: {str(e)}")
            return False