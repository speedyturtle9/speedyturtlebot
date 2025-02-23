# utils/instrument_validator.py

import logging
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingInfo
from datetime import datetime
from trading_bot.utils.config import (
    INSTRUMENTS,
    OANDA_API_KEY,
    OANDA_ACCOUNT_ID,
    FOREX_PAIRS,
    COMMODITIES,
    INDICES,
    TIMEFRAME,
    EXECUTION,
    INSTRUMENT_SETTINGS,
    TRADING_HOURS,
)
from trading_bot.utils.config import (
    OANDA_API_KEY,
    OANDA_ACCOUNT_ID,
    FOREX_PAIRS,  # Import FOREX_PAIRS
    COMMODITIES,  # Import COMMODITIES
    INDICES,  # Import INDICES
)

class InstrumentValidator:
    def __init__(self):
        try:
            self.api = API(access_token=OANDA_API_KEY, environment="practice")
            self.account_id = OANDA_ACCOUNT_ID
            self.available_instruments = []
            
            # Use INSTRUMENTS directly
            self.all_instruments = INSTRUMENTS
            
            # Get execution settings
            self.max_spread = EXECUTION['max_spread_percent']
            
            logging.info(f"Instrument validator initialized with {len(self.all_instruments)} instruments")
            
        except Exception as e:
            logging.error(f"Error initializing InstrumentValidator: {str(e)}")
            raise

    def validate_instruments(self):
        """Validate all trading instruments"""
        try:
            logging.info("Starting instrument validation...")
            validated_instruments = []
            failed_instruments = []
            
            for instrument in self.all_instruments:
                try:
                    if self._validate_single_instrument(instrument):
                        validated_instruments.append(instrument)
                        logging.info(f"Validated {instrument}")
                    else:
                        failed_instruments.append(instrument)
                        logging.warning(f"Failed to validate {instrument}")
                except Exception as e:
                    failed_instruments.append(instrument)
                    logging.error(f"Error validating {instrument}: {str(e)}")
            
            # Update available instruments
            self.available_instruments = validated_instruments
            
            # Generate validation summary
            summary = self._generate_validation_summary(validated_instruments, failed_instruments)
            
            logging.info(summary)
            return validated_instruments
            
        except Exception as e:
            logging.error(f"Error in validate_instruments: {str(e)}")
            return
        
    def get_instrument_type(self, instrument):
        """Get the type of an instrument"""
        if instrument in FOREX_PAIRS:  # Use the imported FOREX_PAIRS
            return "FOREX"
        elif instrument in COMMODITIES:  # Use the imported COMMODITIES
            return "COMMODITIES"
        elif instrument in INDICES:  # Use the imported INDICES
            return "INDICES"
        return "UNKNOWN"

    def is_tradeable(self, instrument):
        """Check if instrument is currently tradeable"""
        try:
            # Check if instrument is validated
            if instrument not in self.available_instruments:
                return False

            # Check trading hours
            if not self.is_trading_hour(instrument):
                return False

            # Get current pricing
            pricing_params = {"instruments": instrument}
            pricing_request = PricingInfo(accountID=self.account_id, params=pricing_params)
            pricing_response = self.api.request(pricing_request)
            
            if pricing_response.get('prices'):
                price = pricing_response['prices']
                
                # Check if tradeable
                if not price.get('tradeable', True):
                    return False
                    
                # Check spread
                ask = float(price['asks']['price'])
                bid = float(price['bids']['price'])
                spread = (ask - bid) / bid
                
                instrument_config = INSTRUMENT_SETTINGS.get(instrument, {
                    'min_spread': self.max_spread
                })
                
                return spread <= instrument_config['min_spread']
            
            return False

        except Exception as e:
            logging.error(f"Error checking tradeability for {instrument}: {str(e)}")
            return False

    def is_trading_hour(self, instrument):
        """Check if current time is within trading hours"""
        try:
            instrument_type = self.get_instrument_type(instrument)
            trading_hours = TRADING_HOURS.get(instrument_type)
            
            if not trading_hours:
                logging.warning(f"No trading hours defined for {instrument_type}")
                return False
                
            current_time = datetime.now().time()
            
            # Check if within main trading hours
            if trading_hours['start'] <= current_time <= trading_hours['end']:
                # For forex, check active sessions
                if instrument_type == 'FOREX' and 'active_sessions' in trading_hours:
                    for session in trading_hours['active_sessions'].values():
                        if session['start'] <= current_time <= session['end']:
                            return True
                    return False
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking trading hours for {instrument}: {str(e)}")
            return False

    def get_instrument_type(self, instrument):
        """Get the type of an instrument"""
        if instrument in FOREX_PAIRS:  # Use FOREX_PAIRS from config
            return "FOREX"
        elif instrument in COMMODITIES:  # Use COMMODITIES from config
            return "COMMODITIES"
        elif instrument in INDICES:  # Use INDICES from config
            return "INDICES"
        return "UNKNOWN"

    def _validate_single_instrument(self, instrument):
        try:
            # Get instrument-specific settings
            instrument_config = INSTRUMENT_SETTINGS.get(instrument, {
                'pip_value': 0.0001,
                'min_spread': self.max_spread
            })

            # Check if instrument exists and can fetch candles
            params = {
                "count": 1,
                "granularity": TIMEFRAME,
                "price": "M"
            }
        
            try:
                candles_request = InstrumentsCandles(instrument=instrument, params=params)
                candles_response = self.api.request(candles_request)
            
                if not candles_response.get('candles'):
                   logging.warning(f"No candles data available for {instrument}")
                   return False
                    
            except Exception as e:
                logging.warning(f"Failed to get candles for {instrument}: {str(e)}")
                return False

            # Check current pricing and spread
            try:
                pricing_params = {"instruments": instrument}
                pricing_request = PricingInfo(accountID=self.account_id, params=pricing_params)
                pricing_response = self.api.request(pricing_request)
    
                if pricing_response:
                   for price in pricing_response:
                       # Check if tradeable
                       if not price.get('tradeable', True):
                           logging.warning(f"{instrument} is not tradeable")
                           return False
                        
                       # Calculate and check spread
                       ask = float(price['asks'][0]['price'])
                       bid = float(price['bids'][0]['price'])
                       spread = (ask - bid) / bid
                    
                       if spread > instrument_config['min_spread']:
                           logging.warning(
                               f"Spread too high for {instrument}: "
                               f"{spread*100:.3f}% > {instrument_config['min_spread']*100:.3f}%"
                           )
                           return False
                         
                       logging.info(
                           f"Validated {instrument} - "
                           f"Spread: {spread*100:.3f}%, "
                           f"Pip Value: {instrument_config['pip_value']}"
                       )
                       return True
                else:
                    logging.warning(f"No pricing data available for {instrument}")
                    return False    
                logging.warning(f"No pricing data available for {instrument}")
                return False
            
            except Exception as e:
                logging.warning(f"Failed to get pricing for {instrument}: {str(e)}")
                return False

        except Exception as e:
            logging.error(f"Validation failed for {instrument}: {str(e)}")
            return False
            


    def _generate_validation_summary(self, validated_instruments, failed_instruments):
        """Generate detailed validation summary"""
        total_passed = len(validated_instruments)
        total_failed = len(failed_instruments)
        total = total_passed + total_failed

        summary = "\n=== Instrument Validation Summary ===\n"
        summary += f"\nOverall Results:\n"
        summary += f"Total Validated: {total_passed}/{total} ({total_passed / total * 100:.1f}%)\n"
        summary += f"Total Failed: {total_failed}/{total}\n"
        summary += f"Validated instruments: {validated_instruments}\n"
        summary += f"Failed instruments: {failed_instruments}\n"
        
        return summary