# utils/telegram_bot.py

import telegram
from telegram.error import TelegramError
import asyncio
import logging
from trading_bot.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class TelegramBot:
    def __init__(self):
        """Initialize Telegram bot with provided credentials"""
        try:
            self.bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
            self.chat_id = str(TELEGRAM_CHAT_ID)
            self.validate_connection()
            logging.info("Telegram bot initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Telegram bot: {str(e)}")
            raise

    def validate_connection(self):
        """Validate bot connection and permissions"""
        try:
            asyncio.get_event_loop().run_until_complete(
                self._send_message_async("🔄 Validating Telegram bot connection...")
            )
            return True
        except Exception as e:
            logging.error(f"Telegram bot validation failed: {str(e)}")
            return False

    async def _send_message_async(self, message):
        """Send message asynchronously"""
        try:
            # Replace problematic characters and escape special characters
            message = message.replace('\\n', '\n')  # Convert string newlines to actual newlines
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=None  # Disable markdown parsing for reliability
            )
        except TelegramError as e:
            logging.error(f"Telegram Error: {e}")
            logging.error(f"Failed to send message to chat_id: {self.chat_id}")
            logging.error(f"Message content: {message}")
        except Exception as e:
            logging.error(f"Error sending telegram message: {e}")

    def send_message(self, message):
        """Send message with error handling"""
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            loop.run_until_complete(self._send_message_async(message))
        except Exception as e:
            logging.error(f"Error in send_message: {e}")

    def send_trade_signal(self, instrument, direction, price, stop_loss, take_profit, news_articles):
       """Send trade signal notification"""
       try:
           precision = 3 if 'JPY' in instrument else 5
           message = (
               "🔔 Trade Signal\n"
               f"Instrument: {instrument}\n"
               f"Direction: {direction}\n"
               f"Entry Price: {price:.{precision}f}\n"
               f"Stop Loss: {stop_loss:.{precision}f}\n"
               f"Take Profit: {take_profit:.{precision}f}\n\n"
               "Analysis Details:\n"
               # Add analysis details here
           )
           self.send_message(message)

           # Send news articles
           news_msg = "📰 News Articles:\n"
           for article in news_articles:
               news_msg += f"{article['title']}\n{article['url']}\n\n"
           self.send_message(news_msg)

       except Exception as e:
           logging.error(f"Error sending trade signal: {e}")


    def send_trade_execution(self, instrument, direction, units, price):
        """Send trade execution notification"""
        try:
            precision = 3 if 'JPY' in instrument else 5
            message = (
                "✅ Trade Executed\n"
                f"Instrument: {instrument}\n"
                f"Direction: {direction}\n"
                f"Units: {units}\n"
                f"Price: {price:.{precision}f}"
            )
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending trade execution: {e}")

    def send_error(self, error_message):
        """Send error notification"""
        try:
            # Simplify error messages to avoid formatting issues
            if isinstance(error_message, dict):
                error_message = f"Error: {error_message.get('errorMessage', 'Unknown error')}"
            elif isinstance(error_message, str):
                error_message = f"Error: {error_message}"
            
            message = f"❌ {error_message}"
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending error message: {e}")

    def send_analysis_update(self, instrument, mr_signal, ml_signal, current_price):
        """Send analysis update notification"""
        try:
            precision = 3 if 'JPY' in instrument else 5
            message = (
                "📊 Analysis Update\n"
                f"Instrument: {instrument}\n"
                f"Mean Reversion: {mr_signal if mr_signal else 'No Signal'}\n"
                f"ML Signal: {ml_signal if ml_signal else 'No Signal'}\n"
                f"Current Price: {current_price:.{precision}f}"
            )
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending analysis update: {e}")

    def send_daily_summary(self, balance, pnl, active_trades, analyzed_instruments):
        """Send daily trading summary"""
        try:
            message = (
                "📈 Daily Summary\n"
                f"Balance: ${balance:.2f}\n"
                f"P/L: ${pnl:.2f}\n"
                f"Active Trades: {active_trades}\n"
                f"Instruments Analyzed: {analyzed_instruments}"
            )
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending daily summary: {e}")

    def send_performance_update(self, performance_metrics):
        """Send performance metrics update"""
        try:
            message = (
                "📊 Performance Update\n"
                f"Win Rate: {performance_metrics['win_rate']:.2f}%\n"
                f"Profit Factor: {performance_metrics['profit_factor']:.2f}\n"
                f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {performance_metrics['max_drawdown']:.2f}%"
            )
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending performance update: {e}")

    def send_risk_alert(self, message, risk_level='MEDIUM'):
        """Send risk management alert"""
        try:
            risk_emojis = {
                'LOW': '🟢',
                'MEDIUM': '🟡',
                'HIGH': '🔴'
            }
            
            alert = (
                f"{risk_emojis.get(risk_level, '⚠️')} Risk Alert\n"
                f"{message}"
            )
            self.send_message(alert)
        except Exception as e:
            logging.error(f"Error sending risk alert: {e}")

    def send_market_update(self, instrument, price_change, volume, sentiment=None):
        """Send market update notification"""
        try:
            precision = 3 if 'JPY' in instrument else 5
            message = (
                "🌍 Market Update\n"
                f"Instrument: {instrument}\n"
                f"Price Change: {price_change:.{precision}f}%\n"
                f"Volume: {volume:,}\n"
            )
            
            if sentiment is not None:
                message += f"Sentiment: {sentiment:.2f}"
                
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending market update: {e}")

    def send_strategy_alert(self, strategy_name, signal_type, confidence, details=None):
        """Send strategy-specific alert"""
        try:
            message = (
                "🎯 Strategy Alert\n"
                f"Strategy: {strategy_name}\n"
                f"Signal: {signal_type}\n"
                f"Confidence: {confidence:.2f}%\n"
            )
            
            if details:
                message += f"Details: {details}"
                
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending strategy alert: {e}")

    def send_system_status(self, status_data):
        """Send system status update"""
        try:
            message = (
                "🖥️ System Status\n"
                f"CPU Usage: {status_data['cpu_usage']:.1f}%\n"
                f"Memory Usage: {status_data['memory_usage']:.1f}%\n"
                f"Active Processes: {status_data['active_processes']}\n"
                f"Last Update: {status_data['last_update']}"
            )
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending system status: {e}")

    def send_startup_notification(self, config_summary):
        """Send bot startup notification"""
        try:
            message = (
                "🚀 Trading Bot Started\n"
                f"Risk Profile: {config_summary['risk_profile']}\n"
                f"Active Instruments: {len(config_summary['instruments'])}\n"
                f"Active Strategies: {len(config_summary['strategies'])}\n"
                "System Ready"
            )
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending startup notification: {e}")

    def send_shutdown_notification(self, summary):
        """Send bot shutdown notification"""
        try:
            message = (
                "🔄 Trading Bot Shutdown\n"
                f"Active Trades Closed: {summary['trades_closed']}\n"
                f"Daily P/L: ${summary['daily_pnl']:.2f}\n"
                f"Uptime: {summary['uptime']}\n"
                "System Stopped"
            )
            self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending shutdown notification: {e}")
