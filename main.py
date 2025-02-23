import os
import sys
import time
import logging
from dotenv import load_dotenv

# Ensure the project root directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from.trading_bot import TradingBot
from trading_bot.utils.telegram_bot import TelegramBot
from trading_bot.utils.config import (
    LOGGING,
    DEBUG
)

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """
    Configure logging based on project settings
    """
    log_level = getattr(logging, LOGGING['level'].upper())
    log_format = LOGGING['format']
    log_file = LOGGING['file']

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    full_log_path = os.path.join(log_dir, log_file)

    # Configure logging handlers
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(full_log_path),
            logging.StreamHandler()
        ]
    )

def main():
    """
    Main entry point for the trading bot application
    """
    # Setup logging
    setup_logging()

    # Initialize Telegram notification
    telegram = TelegramBot()
    bot = None
    
    try:
        # Send startup notification
        startup_msg = "🚀 Trading Bot Startup Sequence Initiated"
        logging.info(startup_msg)
        telegram.send_message(startup_msg)
        
        # Check if in paper trading mode
        if DEBUG.get('paper_trading', False):
            logging.warning("🧪 Running in Paper Trading Mode")
            telegram.send_message("🧪 Paper Trading Mode Activated")
        
        # Initialize and run the trading bot
        bot = TradingBot()
        bot.run()
        
    except KeyboardInterrupt:
        shutdown_msg = "👋 Shutdown Signal Received. Initiating Graceful Shutdown..."
        logging.info(shutdown_msg)
        telegram.send_message(shutdown_msg)
        
        if bot is not None:
            bot.cleanup()
            
    except Exception as critical_error:
        error_msg = (
            f"❌ Critical Error Detected\n"
            f"Error Details: {str(critical_error)}\n"
            f"Error Type: {type(critical_error).__name__}"
        )
        logging.critical(error_msg, exc_info=True)
        telegram.send_error(error_msg)
        
        if bot is not None:
            bot.cleanup()
        
        sys.exit(1)
        
    finally:
        final_msg = "✅ Trading Bot Shutdown Complete"
        logging.info(final_msg)
        telegram.send_message(final_msg)

def run_with_retry(max_retries=3, retry_delay=300):
    """
    Run the main function with a robust retry mechanism
    """
    telegram = TelegramBot()
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            main()
            break
            
        except Exception as retry_error:
            retry_count += 1
            error_details = (
                f"❌ Retry Attempt {retry_count}/{max_retries}\n"
                f"Error: {str(retry_error)}\n"
                f"Next Retry in {retry_delay} seconds..."
            )
            
            logging.error(error_details)
            telegram.send_error(error_details)
            
            if retry_count < max_retries:
                time.sleep(retry_delay)
            else:
                fatal_msg = (
                    "❌ Maximum Retry Attempts Exhausted\n"
                    "Trading Bot Cannot Recover. Shutting Down."
                )
                logging.critical(fatal_msg)
                telegram.send_error(fatal_msg)
                sys.exit(1)

def validate_environment():
    """
    Validate critical environment variables and dependencies
    """
    required_vars = [
        'OANDA_API_KEY',
        'OANDA_ACCOUNT_ID',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logging.critical(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    # Validate environment before starting
    validate_environment()
    
    # Run the trading bot with retry mechanism
    run_with_retry()
