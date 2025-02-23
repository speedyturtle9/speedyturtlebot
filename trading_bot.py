import time
from typing import Self
import pandas as pd
import schedule
import logging
from datetime import datetime

from trading_bot.utils.monitoring import SystemMonitor
from .utils import news_analyzer
from trading_bot.utils.oanda_api import OandaAPI
from.utils.telegram_bot import TelegramBot
from trading_bot.utils.instrument_validator import InstrumentValidator
from trading_bot.utils.news_analyzer import NewsFeedAnalyzer
from.strategies.mean_reversion import MeanReversionStrategy
from.strategies.ml_strategy import MLStrategy
from.strategies.arbitrage_strategy import ArbitrageStrategy
from.strategies.momentum_strategy import MomentumStrategy
from.strategies.grid_trading_strategy import GridTradingStrategy
from trading_bot.utils.config import (
    TIMEFRAME,
    RISK_LEVELS,
    CURRENT_RISK_PROFILE,
    NEWS_API_KEY,
    NEWS_UPDATE_INTERVAL,
    NEWS_IMPACT_THRESHOLD,
    ML_MIN_TRAINING_SAMPLES,
    ML_TRAINING_INTERVAL,
    INSTRUMENTS,
    STRATEGY_AGREEMENT_THRESHOLDS,
    MAX_TRADES_PER_INSTRUMENT,
    MAX_DAILY_TRADES,
    MAX_DAILY_LOSS_PERCENT,
    RETRY_ATTEMPTS,
    RETRY_WAIT_TIME,
    INSTRUMENT_SETTINGS,
    RISK_SETTINGS,
    TRADING_HOURS
)

# Apply risk profile settings
RISK_SETTINGS = RISK_LEVELS[CURRENT_RISK_PROFILE]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class TradingBot:
    """Trading Bot main class"""
    
    def __init__(self) -> None:
        """Initialize Trading Bot with all required components"""
        # Setup logging first
        self._setup_logging()
        logging.info("Initializing Trading Bot...")
        
        # Initialize all components with proper error handling
        self._init_api_connections()
        self._init_strategies()
        self._init_tracking_variables()
        self._init_news_analyzer()
        self._init_monitoring()
        
        # Start initialization sequence
        self.startup_sequence()
        logging.info("Trading Bot initialization completed")

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )

    def _init_api_connections(self) -> None:
        """Initialize API connections"""
        try:
            self.oanda = OandaAPI()
            logging.info("OANDA API initialized")
        except Exception as e:
            logging.critical(f"Failed to initialize OANDA API: {str(e)}")
            raise

        try:
            self.telegram = TelegramBot()
            logging.info("Telegram Bot initialized")
        except Exception as e:
            logging.critical(f"Failed to initialize Telegram Bot: {str(e)}")
            raise

        try:
            self.instrument_validator = InstrumentValidator()
            logging.info("Instrument validator initialized")
        except Exception as e:
            logging.critical(f"Failed to initialize Instrument Validator: {str(e)}")
            raise

    def _init_strategies(self) -> None:
        """Initialize trading strategies"""
        try:
            self.strategies = {
                'Mean Reversion': MeanReversionStrategy(),
                'Machine Learning': MLStrategy(),
                'Arbitrage': ArbitrageStrategy(),
                'Momentum': MomentumStrategy(),
                'Grid Trading': GridTradingStrategy()
            }
            logging.info(f"Initialized {len(self.strategies)} trading strategies")
        except Exception as e:
            logging.critical(f"Failed to initialize strategies: {str(e)}")
            raise

    def _init_tracking_variables(self) -> None:
        """Initialize tracking and performance variables"""
        try:
            # Trading tracking
            self.active_trades = {}
            self.last_balance_check = 0
            self.balance_check_interval = 30
            self.last_instrument_index = 0
            self.daily_trades = 0
            self.daily_loss = 0
            self.last_reset_day = datetime.now().date()
            
            # Performance tracking
            self.strategy_performance = {
                name: {
                    'wins': 0,
                    'losses': 0,
                    'total_profit': 0,
                    'total_loss': 0
                }
                for name in self.strategies.keys()
            }
            
            # Data storage
            self.historical_data = {}
            self.available_instruments = []
            
            logging.info("Tracking variables initialized")
        except Exception as e:
            logging.critical(f"Failed to initialize tracking variables: {str(e)}")
            raise

    def _init_news_analyzer(self) -> None:
        """Initialize news analyzer"""
        try:
            self.news_analyzer = NewsFeedAnalyzer()
            logging.info("News analyzer initialized")
            
            if NEWS_API_KEY:
                self.news_analyzer = news_analyzer(NEWS_API_KEY)
                self.news_cache = {}
                self.last_news_update = 0
                logging.info("News analysis initialized with API key")
            else:
                logging.warning("No NEWS_API_KEY found - news analysis disabled")
        except Exception as e:
            logging.error(f"Failed to initialize news analyzer: {str(e)}")
            self.news_analyzer = None

    def _init_monitoring(self) -> None:
        """Initialize system monitoring"""
        try:
            self.system_monitor = SystemMonitor()
            
            # Schedule monitoring tasks
            schedule.every(1).minutes.do(self.check_system_health)
            schedule.every(1).hours.do(self.generate_health_report)
            
            logging.info("System monitoring initialized")
        except Exception as e:
            logging.error(f"Failed to initialize system monitoring: {str(e)}")
            self.system_monitor = None

    def check_system_health(self) -> None:
        """Check system health and send alerts if needed"""
        if not self.system_monitor:
            return

        try:
            alerts = self.system_monitor.check_alerts()
            if alerts:
                alert_message = "⚠️ System Alerts:\n" + "\n".join(alerts)
                self.telegram.send_message(alert_message)
        except Exception as e:
            logging.error(f"Error checking system health: {str(e)}")

    def generate_health_report(self) -> None:
        """Generate and send system health report"""
        if not self.system_monitor:
            return

        try:
            report = self.system_monitor.get_performance_report()
            
            message = "📊 System Health Report\n\n"
            for metric, data in report['metrics'].items():
                message += f"{metric}:\n"
                message += f"  Current: {data['current']:.1f}%\n"
                message += f"  1h Avg: {data['avg_1h']:.1f}%\n"
                message += f"  24h Avg: {data['avg_24h']:.1f}%\n\n"
            
            self.telegram.send_message(message)
        except Exception as e:
            logging.error(f"Error generating health report: {str(e)}")

    def startup_sequence(self) -> None:
        """Execute startup sequence"""
        logging.info("Starting bot initialization sequence...")
        
    def startup_sequence(self):
        """Execute complete startup sequence"""
        try:
            logging.info("Starting initialization sequence...")
            self.telegram.send_message("🚀 Starting Trading Bot initialization...")
            
            # 1. Validate instruments
            self.validate_instruments()
            if not self.available_instruments:
                raise ValueError("No valid instruments available for trading")
            
            # 2. Get historical data
            self.get_all_historical_data()
            
            # 3. Validate strategies
            self.validate_strategies()
            
            # 4. Check account status
            self.check_account_status()
            
            # 5. Train ML model
            self.check_and_train_ml()
            
            # 6. Send startup complete message
            startup_msg = (
                "✅ Trading Bot Initialization Complete\n"
                f"Active Instruments: {len(self.available_instruments)}\n"
                f"Active Strategies: {len(self.strategies)}\n"
                f"News Analysis: {'Enabled' if hasattr(self, 'news_analyzer') else 'Disabled'}\n"
                "Bot is ready to trade"
            )
            self.telegram.send_message(startup_msg)
            logging.info("Initialization sequence completed successfully")
            
        except Exception as e:
            error_msg = f"Startup sequence failed: {str(e)}"
            logging.error(error_msg)
            self.telegram.send_error(error_msg)
            raise

    def get_all_historical_data(self):
        """Retrieve historical data for all instruments"""
        try:
            self.telegram.send_message("Starting historical data retrieval...")
            logging.info("Starting historical data retrieval...")

            for instrument in self.available_instruments:
                try:
                   data = self.oanda.get_historical_data(instrument)
                   if data is not None and not data.empty:
                       self.historical_data[instrument] = data
                       logging.info(f"Historical data retrieved for {instrument}")
                   else:
                    logging.warning(f"No historical data available for {instrument}")
                except Exception as e:
                    logging.error(f"Error retrieving data for {instrument}: {str(e)}")

            logging.info("Historical data retrieval complete")

        except Exception as e:
            error_msg = f"Error in get_all_historical_data: {str(e)}"
            logging.error(error_msg)
            self.telegram.send_error(error_msg)

    def get_strategy_consensus(self, signals, instrument):
        """
        Determine trading consensus based on multiple strategy signals
        """
        try:
            if not signals:
                logging.info(f"{instrument}: No signals available")
                return None

            # Initialize counters
            buy_signals = 0
            sell_signals = 0
            total_valid_signals = 0

            # Count valid signals
            for strategy_name, signal in signals.items():
                if signal and isinstance(signal, dict) and 'signal' in signal:
                    if signal['signal'] == 'BUY':
                       buy_signals += 1
                       total_valid_signals += 1
                elif signal['signal'] == 'SELL':
                    sell_signals += 1
                    total_valid_signals += 1

            # If no valid signals, return None
            if total_valid_signals == 0:
                logging.info(f"{instrument}: No valid signals detected")
                return None

            # Calculate agreement percentages
            buy_agreement = (buy_signals / total_valid_signals) * 100 if total_valid_signals > 0 else 0
            sell_agreement = (sell_signals / total_valid_signals) * 100 if total_valid_signals > 0 else 0

            # Get required agreement threshold
            if total_valid_signals == 2:
               required_agreement = STRATEGY_AGREEMENT_THRESHOLDS['2_strategies']
            elif total_valid_signals == 3:
                 required_agreement = STRATEGY_AGREEMENT_THRESHOLDS['3_strategies']
            else:
                required_agreement = STRATEGY_AGREEMENT_THRESHOLDS['default']

            # Log analysis details
            analysis = (
                f"\nStrategy Agreement Analysis for {instrument}:\n"
                f"Total Valid Signals: {total_valid_signals}\n"
                f"Buy Signals: {buy_signals} ({buy_agreement:.1f}%)\n"
                f"Sell Signals: {sell_signals} ({sell_agreement:.1f}%)\n"
                f"Required Agreement: {required_agreement}%"
            ) 
            logging.info(analysis)

            # Determine consensus
            if buy_agreement >= required_agreement:
               consensus_msg = f"{instrument}: BUY consensus reached with {buy_agreement:.1f}% agreement"
               logging.info(consensus_msg)
               return 'BUY'
            elif sell_agreement >= required_agreement:
               consensus_msg = f"{instrument}: SELL consensus reached with {sell_agreement:.1f}% agreement"
               logging.info(consensus_msg)
               return 'SELL'
            else:
                no_consensus_msg = (
                    f"{instrument}: No consensus reached. "
                    f"Buy: {buy_agreement:.1f}%, Sell: {sell_agreement:.1f}%"
               )
                logging.info(no_consensus_msg)
                return None

        except Exception as e:
            logging.error(f"Error in get_strategy_consensus for {instrument}: {str(e)}")
            return None

    def analyze_news(self, instrument):
        """
        Analyze news sentiment for an instrument
        """
        try:
            if not hasattr(self, 'news_analyzer'):
                return None

            # Check if we need to update news cache
            current_time = time.time()
            if current_time - self.last_news_update > NEWS_UPDATE_INTERVAL:
                self.news_cache = self.news_analyzer.get_latest_news()
                self.last_news_update = current_time

            # Get relevant news for the instrument
            instrument_news = self.news_analyzer.filter_news_for_instrument(
                self.news_cache, 
                instrument
            )

            if not instrument_news:
                return None

            # Calculate sentiment
            sentiment = self.news_analyzer.calculate_sentiment(instrument_news)
            logging.info(f"News sentiment for {instrument}: {sentiment}")
            
            return sentiment

        except Exception as e:
            logging.error(f"Error analyzing news for {instrument}: {str(e)}")
            return None

    def should_trade_based_on_news(self, instrument, sentiment):
        """
        Determine if trading should proceed based on news sentiment
        """
        try:
            if sentiment is None:
                return True

            # Check if sentiment is too extreme
            if abs(sentiment) > NEWS_IMPACT_THRESHOLD:
                msg = (
                    f"⚠️ High news impact detected for {instrument}\n"
                    f"Sentiment: {sentiment:.2f}\n"
                    "Trading suspended for this instrument"
                )
                self.telegram.send_message(msg)
                logging.warning(f"Trading suspended for {instrument} due to high news impact")
                return False

            return True

        except Exception as e:
            logging.error(f"Error in news-based trading decision: {str(e)}")
            return True  # Default to allowing trades if error occurs
 
    def calculate_position_size(self, instrument, signal_type):
        """Calculate appropriate position size based on risk parameters"""
        try:
            if not instrument or not signal_type:
               logging.warning("Missing instrument or signal type for position size calculation")
               return None

             # Get account information
            account = self.oanda.get_account_summary()
            if not account:
                logging.error("Failed to get account summary")
                return None
            
            balance = float(account['account']['balance'])

            # Get instrument settings
            settings = INSTRUMENT_SETTINGS.get(instrument, {})
            if not settings:
                logging.error(f"No settings found for {instrument}")
                return None

            # Calculate base position size based on risk percentage
            risk_amount = balance * (RISK_SETTINGS['risk_per_trade'] / 100)
        
            # Get current price
            current_price = self.oanda.get_current_price(instrument)
            if not current_price:
                logging.error(f"Failed to get current price for {instrument}")
                return None

            # Calculate position size based on stop loss pips
            pip_value = settings['pip_value']
            stop_loss_pips = RISK_SETTINGS['stop_loss_pips']
        
            if pip_value <= 0 or stop_loss_pips <= 0:
                logging.error(f"Invalid pip value or stop loss pips for {instrument}")
                return None

            position_size = (risk_amount / (stop_loss_pips * pip_value))
        
            # Apply maximum position size limit
            position_size = min(position_size, RISK_SETTINGS['max_position_size'])
        
            # Round to nearest standard lot size
            position_size = round(position_size / 1000) * 1000
        
           # Adjust direction based on signal
            if signal_type == 'SELL':
             position_size = -position_size
            
            logging.info(f"Calculated position size for {instrument}: {position_size}")
            return position_size
  
        except Exception as e:
            logging.error(f"Error calculating position size for {instrument}: {str(e)}")
            return None

    def update_trade_tracking(self, instrument, signal_type):
        """Update trade tracking metrics"""
        try:
            # Increment daily trades counter
            self.daily_trades += 1
            
            msg = (
                f"📈 Trade Tracking Update\n"
                f"Instrument: {instrument}\n"
                f"Signal: {signal_type}\n"
                f"Daily Trades: {self.daily_trades}/{MAX_DAILY_TRADES}"
            )
            self.telegram.send_message(msg)
            
            # Check if we're approaching limits
            if self.daily_trades >= MAX_DAILY_TRADES - 1:
                warning_msg = (
                    "⚠️ Approaching daily trade limit\n"
                    f"Trades today: {self.daily_trades}\n"
                    f"Maximum: {MAX_DAILY_TRADES}"
                )
                self.telegram.send_message(warning_msg)
                
            logging.info(
                f"Updated trade tracking - "
                f"Daily trades: {self.daily_trades}, "
                f"Daily loss: ${self.daily_loss:.2f}"
            )

        except Exception as e:
            logging.error(f"Error updating trade tracking: {str(e)}")

    def update_historical_data(self):
        """Update historical data for all instruments"""
        for instrument in self.available_instruments:
            try:
                # Fetch latest data from OandaAPI
                latest_data = self.oanda.get_historical_data(instrument, count=1) # Fetch only the latest candle
            
                # Check if new data is available and append it
                if latest_data is not None and not latest_data.empty and latest_data.index[-1] not in self.historical_data[instrument].index:
                   self.historical_data[instrument] = pd.concat([self.historical_data[instrument], latest_data])
                   logging.info(f"Historical data updated for {instrument}")
                else:
                    logging.info(f"No new data available for {instrument}")

            except Exception as e:
                logging.error(f"Error updating historical data for {instrument}: {e}")

    def monitor_active_trades(self):
        """Monitor and manage active trades"""
        try:
            if not self.active_trades:
                return

            logging.info(f"Monitoring {len(self.active_trades)} active trades")
            
            for trade_id, trade_info in list(self.active_trades.items()):
                try:
                    # Get current trade status
                    trade_status = self.oanda.get_trade_status(trade_id)
                    
                    if trade_status['status'] == 'CLOSED':
                        # Calculate profit/loss
                        profit = float(trade_status['realizedPL'])
                        
                        # Update tracking
                        if profit < 0:
                            self.daily_loss += abs(profit)
                            
                        # Update strategy performance
                        strategy = trade_info.get('strategy', 'Unknown')
                        if strategy in self.strategy_performance:
                            if profit > 0:
                                self.strategy_performance[strategy]['wins'] += 1
                                self.strategy_performance[strategy]['total_profit'] += profit
                            else:
                                self.strategy_performance[strategy]['losses'] += 1
                                self.strategy_performance[strategy]['total_loss'] += abs(profit)
                        
                        # Remove from active trades
                        self.active_trades.pop(trade_id)
                        
                        # Send notification
                        msg = (
                            f"Trade Closed: {trade_info['instrument']}\n"
                            f"Type: {trade_info['type']}\n"
                            f"Profit/Loss: ${profit:.2f}"
                        )
                        self.telegram.send_message(msg)
                        
                except Exception as e:
                    logging.error(f"Error monitoring trade {trade_id}: {str(e)}")

        except Exception as e:
            error_msg = f"Error in trade monitoring: {str(e)}"
            logging.error(error_msg)
            self.telegram.send_error(error_msg)

    def analyze_instrument(self, instrument):
        """Analyze an instrument and execute trades if conditions are met"""
        try:
            logging.info(f"Analyzing {instrument}")
        
            # Get news analysis first
            news_analysis = self.news_analyzer.analyze_news_for_instrument(instrument)
        
            if news_analysis:
                # Check if news sentiment/impact prevents trading
                if not news_analysis['should_trade']:
                    news_alert = (
                     f"⚠️ High Impact News Alert - {instrument}\n"
                     f"Sentiment: {news_analysis['sentiment']:.2f}\n"
                     f"Impact: {news_analysis['impact']:.2f}\n\n"
                     "Recent News:\n"
                    )
                
                    for news in news_analysis['news_items'][:3]:
                        news_alert += f"- {news['title']}\n"
                        news_alert += f"  Link: {news['link']}\n"
                
                    self.telegram.send_message(news_alert)
                    logging.warning(f"Trading suspended for {instrument} due to news impact")
                    return

            # Get signals from all strategies
            signals = {}
            for name, strategy in self.strategies.items():
                try:
                    if self.historical_data.get(instrument) is None:
                       logging.warning(f"No historical data available for {instrument}")
                       continue
                    
                    signal = strategy.get_signal(self.historical_data[instrument].copy(), instrument)
                    if signal:
                        signals[name] = signal
                        logging.info(f"{name} signal for {instrument}: {signal.get('signal', 'None')}")

                except Exception as e:
                    logging.error(f"Error getting signal from {name}: {str(e)}")

            # Get consensus only if we have signals
            if signals:
                consensus = self.get_strategy_consensus(signals, instrument)
            
                if consensus:
                    # Calculate position size
                    position_size = self.calculate_position_size(instrument, consensus)

                    if position_size:
                        # Execute trade
                        trade_result = self.oanda.create_order(
                            instrument=instrument,
                            units=position_size,
                            signal_type=consensus
                        )

                        if trade_result:
                            self.update_trade_tracking(instrument, consensus)

            # Send analysis summary
            analysis_summary = f"📊 Analysis Summary for {instrument}:\n"
            for strategy_name, signal in signals.items():
                signal_value = signal.get('signal', 'No Signal') if signal else 'No Signal'
                analysis_summary += f"{strategy_name}: {signal_value}\n"
        
            self.telegram.send_message(analysis_summary)

        except Exception as e:
            logging.error(f"Error analyzing {instrument}: {str(e)}")
                                  
    def is_trading_hour(self, instrument):
        """Check if current time is within trading hours for the instrument"""
        try:
            current_time = datetime.now().time()
            
            # Get instrument type (FOREX, COMMODITIES, INDICES)
            instrument_type = 'FOREX'  # Default to FOREX
            if instrument in ['XAU_USD', 'XAG_USD']:
                instrument_type = 'COMMODITIES'
            elif instrument in ['US30_USD', 'SPX500_USD', 'NAS100_USD']:
                instrument_type = 'INDICES'
            
            # Get trading hours for instrument type
            trading_hours = TRADING_HOURS.get(instrument_type, TRADING_HOURS['FOREX'])
            
            # Check if current time is within trading hours
            if trading_hours['start'] <= current_time <= trading_hours['end']:
                # For forex, check active sessions if defined
                if instrument_type == 'FOREX' and 'active_sessions' in trading_hours:
                    for session in trading_hours['active_sessions'].values():
                        if session['start'] <= current_time <= session['end']:
                            return True
                    return False
                return True
                
            logging.info(f"{instrument} - Outside trading hours")
            return False
            
        except Exception as e:
            logging.error(f"Error checking trading hours for {instrument}: {str(e)}")
            return False

    def check_daily_limits(self):
        """Check if daily trading limits have been reached"""
        try:
            current_date = datetime.now().date()
            
            # Reset daily tracking if new day
            if current_date != self.last_reset_day:
                self.daily_trades = 0
                self.daily_loss = 0
                self.last_reset_day = current_date
                logging.info("Reset daily trading limits")
                return True
            
            # Check number of trades
            if self.daily_trades >= MAX_DAILY_TRADES:
                msg = (
                    "📊 Daily trade limit reached\n"
                    f"Trades today: {self.daily_trades}\n"
                    f"Maximum: {MAX_DAILY_TRADES}"
                )
                self.telegram.send_message(msg)
                return False
            
            # Check loss limit
            account = self.oanda.get_account_summary()
            balance = float(account['account']['balance'])
            
            if self.daily_loss >= (balance * MAX_DAILY_LOSS_PERCENT / 100):
                msg = (
                    "⚠️ Daily loss limit reached\n"
                    f"Loss today: ${self.daily_loss:.2f}\n"
                    f"Maximum allowed: ${balance * MAX_DAILY_LOSS_PERCENT / 100:.2f}"
                )
                self.telegram.send_message(msg)
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking daily limits: {str(e)}")
            return False
        
    def validate_instruments(self):
        """Validate trading instruments"""
        try:
            logging.info(f"Starting instrument validation for {len(INSTRUMENTS)} instruments...")
        
            # Use the instrument validator to validate instruments
            validated_instruments = self.instrument_validator.validate_instruments()

            if validated_instruments:
               self.available_instruments = validated_instruments
               validation_msg = (
                "🔍 Instrument Validation Results:\n"
                f"Successfully validated: {len(validated_instruments)}\n"
                "Valid Instruments:\n" + "\n".join(validated_instruments)
            )
            else:
            # Use default instruments if validation fails
             default_instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY']
             self.available_instruments = default_instruments
             validation_msg = (
                "⚠️ Instrument Validation Failed\n"
                "Using default instruments:\n" + "\n".join(default_instruments)
            )
            logging.warning("No instruments validated. Using default instruments.")
        
            self.telegram.send_message(validation_msg)
            logging.info(f"Instrument validation complete. {len(self.available_instruments)} instruments available.")
        
            return True
        
        except Exception as e:
              error_msg = f"Error in instrument validation: {str(e)}"
              logging.error(error_msg)
              self.telegram.send_error(error_msg)
              return False

    def send_daily_summary(self):
        """Send daily trading summary"""
        try:
            # Get account information
            account = self.oanda.get_account_summary()
            balance = float(account['account']['balance'])
            
            # Calculate daily statistics
            total_trades = self.daily_trades
            win_rate = 0
            if total_trades > 0:
                wins = sum(1 for trade in self.active_trades.values() if float(trade.get('realizedPL', 0)) > 0)
                win_rate = (wins / total_trades) * 100
                
            # Create summary message
            summary = (
                "📊 Daily Trading Summary\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
                f"Total Trades: {total_trades}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Daily Loss: ${self.daily_loss:.2f}\n"
                f"Account Balance: ${balance:.2f}\n\n"
                "Strategy Performance:\n"
            )
            
            for strategy, perf in self.strategy_performance.items():
                total_trades = perf['wins'] + perf['losses']
                if total_trades > 0:
                    strategy_win_rate = (perf['wins'] / total_trades * 100)
                    net_profit = perf['total_profit'] - perf['total_loss']
                    
                    summary += (
                        f"\n{strategy}:\n"
                        f"Trades: {total_trades}\n"
                        f"Win Rate: {strategy_win_rate:.1f}%\n"
                        f"Net Profit: ${net_profit:.2f}"
                    )
                
            self.telegram.send_message(summary)
            logging.info("Daily summary sent")

        except Exception as e:
            error_msg = f"Error sending daily summary: {str(e)}"
            logging.error(error_msg)
            self.telegram.send_error(error_msg)

    def run(self):
        """Main bot execution loop"""
        try:
            self.telegram.send_message("🤖 Trading Bot Started")
            logging.info("Trading Bot Started")
            
            # Schedule tasks
            schedule.every().hour.do(self.update_historical_data)
            schedule.every(15).minutes.do(self.monitor_active_trades)
            schedule.every().day.at("23:45").do(self.send_daily_summary)
            
            while True:
                try:
                    schedule.run_pending()
                    
                    if self.check_daily_limits():
                        self.check_account_status()
                        
                        if self.available_instruments:
                            instrument = self.available_instruments[self.last_instrument_index]
                            
                            if self.is_trading_hour(instrument):
                                self.analyze_instrument(instrument)
                            
                            self.last_instrument_index = (self.last_instrument_index + 1) % len(
                                self.available_instruments)
                    
                    time.sleep(30)  # Wait 30 seconds between iterations
                    
                except Exception as e:
                    error_msg = f"Error in main loop: {str(e)}"
                    logging.error(error_msg)
                    self.telegram.send_error(error_msg)
                    time.sleep(30)
                    
        except Exception as e:
            error_msg = f"Critical error in run method: {str(e)}"
            logging.error(error_msg)
            self.telegram.send_error(error_msg)

    def cleanup(self):
        """Clean up resources and close positions"""
        try:
            cleanup_msg = "🔄 Starting bot shutdown sequence..."
            logging.info(cleanup_msg)
            self.telegram.send_message(cleanup_msg)
            
            # Close all active trades
            for trade_id, trade_info in list(self.active_trades.items()):
                try:
                    self.oanda.close_position(trade_info['instrument'])
                    self.telegram.send_message(
                        f"✅ Closed position for {trade_info['instrument']}\n"
                        f"Trade ID: {trade_id}"
                    )
                    self.active_trades.pop(trade_id, None)
                except Exception as e:
                    logging.error(f"Error closing position {trade_id}: {str(e)}")
            
            # Send final summary
            final_msg = "🔄 Bot shutdown complete"
            if len(self.active_trades) == 0:
                final_msg += " - All positions closed"
            else:
                final_msg += f" - {len(self.active_trades)} positions could not be closed"
            
            self.telegram.send_message(final_msg)
            
        except Exception as e:
            error_msg = f"Error in cleanup: {str(e)}"
            logging.error(error_msg)
            self.telegram.send_error(error_msg)

    def validate_strategies(self):
        """Validate trading strategies"""
        try:
            logging.info("Starting strategy validation...")
            self.telegram.send_message("🔄 Starting strategy validation...")

             # Validate each strategy with historical data
            for strategy_name, strategy in self.strategies.items():
                 for instrument in self.available_instruments:
                     try:
                         # Get historical data for the instrument
                         data = self.historical_data.get(instrument)
                         if data is None:
                             logging.warning(f"No historical data for {instrument} to validate {strategy_name}")
                             continue

                         # Validate the strategy
                         strategy.get_signal(data.copy(), instrument)
                         logging.info(f"Strategy '{strategy_name}' validated successfully")
                         break  # Move to the next strategy if validation passes for at least one instrument

                     except Exception as e:
                         logging.error(f"Error validating strategy '{strategy_name}' for {instrument}: {str(e)}")

            self.telegram.send_message("✅ Strategy validation complete")
            logging.info("Strategy validation complete")

        except Exception as e:
             error_msg = f"Error in validate_strategies: {str(e)}"
             logging.error(error_msg)
             self.telegram.send_error(error_msg)

    def check_account_status(self):
        """Check account status and balance"""
        try:
            logging.info("Checking account status...")
            self.telegram.send_message("🔄 Checking account status...")

            # Get account balance and profit/loss
            account = self.oanda.get_account_summary()
            balance = float(account['account']['balance'])
            pl = float(account['account']['pl'])

            # Send account status update
            account_status_msg = (
                 "✅ Account status check successful\n"
                 f"Balance: ${balance:.2f}\n"
                 f"Profit/Loss: ${pl:.2f}"
            )
            self.telegram.send_message(account_status_msg)

            logging.info("Account status check completed successfully")

        except Exception as e:
            error_msg = f"Error checking account status: {str(e)}"
            logging.error(error_msg)
            self.telegram.send_error(error_msg)

    def check_and_train_ml(self):
        """Check if ML model needs training and train if necessary"""
        try:
            ml_strategy = self.strategies.get('Machine Learning')
            if not ml_strategy:
                return

            logging.info("Checking ML model status...")
            self.telegram.send_message("🔄 Checking ML model status...")

            for instrument in self.available_instruments:
                try:
                    data = self.historical_data.get(instrument)
                    if data is None or len(data) < ML_MIN_TRAINING_SAMPLES:
                        logging.warning(f"Insufficient data for {instrument}. Need {ML_MIN_TRAINING_SAMPLES} samples, got {len(data) if data is not None else 0}")
                        continue

                    if instrument not in ml_strategy.models or \
                       (datetime.now() - ml_strategy.last_training.get(instrument, datetime.min)).total_seconds() \
                       >= ML_TRAINING_INTERVAL:
                        ml_strategy.train_model(data.copy(), instrument)
                        logging.info(f"ML model trained for {instrument}")

                except Exception as e:
                    logging.error(f"Error training ML model for {instrument}: {str(e)}")

            logging.info("ML model check complete")
            self.telegram.send_message("✅ ML model check complete")

        except Exception as e:
            logging.error(f"Error in check_and_train_ml: {str(e)}")

    def validate_instruments(self):
        """Validate trading instruments"""
        try:
            logging.info(f"Starting instrument validation for {len(INSTRUMENTS)} instruments...")
        
            # Use the instrument validator to validate instruments
            validated_instruments = self.instrument_validator.validate_instruments()

            if validated_instruments:
               self.available_instruments = validated_instruments
               validation_msg = (
                   "🔍 Instrument Validation Results:\n"
                   f"Successfully validated: {len(validated_instruments)}\n"
                   "Valid Instruments:\n" + "\n".join(validated_instruments)
                )
            else:
                # Use default instruments if validation fails
                default_instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY']
                self.available_instruments = default_instruments
                validation_msg = (
                    "⚠️ Instrument Validation Failed\n"
                    "Using default instruments:\n" + "\n".join(default_instruments)
                )
                logging.warning("No instruments validated. Using default instruments.")
          
            self.telegram.send_message(validation_msg)
            logging.info(f"Instrument validation complete. {len(self.available_instruments)} instruments available.")
        
            return True
        
        except Exception as e:
            error_msg = f"Error in instrument validation: {str(e)}"
            logging.error(error_msg)
            self.telegram.send_error(error_msg)
            return False
        
    def update_historical_data(self):
        """Update historical data for all instruments"""
        try:
            if not self.available_instruments:
                logging.warning("No instruments available to update historical data")
                return

            logging.info("Updating historical data...")
            self.telegram.send_message("🔄 Updating historical data...")

            for instrument in self.available_instruments:
                try:
                    # Get latest historical data
                    new_data = self.oanda.get_historical_data(instrument, count=1)  # Get only the latest candle
                    if new_data is None or new_data.empty:
                       logging.warning(f"Failed to update historical data for {instrument}")
                       continue
  
                    # Update the data in self.historical_data
                    if instrument in self.historical_data:
                        # Ensure the new data is not already in historical_data
                        if new_data.index not in self.historical_data[instrument].index:
                           self.historical_data[instrument] = pd.concat(
                               [self.historical_data[instrument], new_data]
                           )
                           # Optionally remove duplicates and keep only the latest data
                           self.historical_data[instrument] = self.historical_data[instrument][
                               ~self.historical_data[instrument].index.duplicated(keep='last')
                           ]
                        else:
                            logging.info(f"No new data to add for {instrument}")
                    else:
                        self.historical_data[instrument] = new_data

                    logging.info(f"Historical data updated for {instrument}")

                except Exception as e:
                   logging.error(f"Error updating data for {instrument}: {str(e)}")

            logging.info("Historical data update complete")
            self.telegram.send_message("✅ Historical data updated")

        except Exception as e:
            error_msg = f"Error in update_historical_data: {str(e)}"
            logging.error(error_msg)
        self.telegram.send_error(error_msg)
    
if __name__ == "__main__":
    bot = None
    try:
        bot = TradingBot()
        bot.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received - shutting down")
        if bot is not None:
            bot.cleanup()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        if bot is not None:
            bot.cleanup()
