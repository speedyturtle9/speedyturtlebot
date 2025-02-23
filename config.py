# config.py

import os
from dotenv import load_dotenv
from datetime import datetime, time

# Load environment variables
load_dotenv()

# API Configuration
OANDA_API_KEY= "4ceaadc8932f37bf29297b7d75f6bf5d-7e168cc583a46328268c8f2433f3d935"
OANDA_ACCOUNT_ID= "101-002-31103382-001"
TELEGRAM_BOT_TOKEN= "7841586345:AAEE-OX6q8Ei3dh-BPdywXbVLyHV1ZAzp6Q"
TELEGRAM_CHAT_ID= "5047191779"
NEWS_API_KEY= "cUPOBleS3T8glwJ3UJP3MyHwkYSV3cMw" # Optional, get from newsapi.org

# Validate required credentials
if not all([OANDA_API_KEY, OANDA_ACCOUNT_ID, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    raise ValueError("Missing required API credentials in .env file")



# Trading Instruments - Major Pairs Only
FOREX_PAIRS = [
    'EUR_USD',  # Euro
    'GBP_USD',  # British Pound
    'USD_JPY',  # Japanese Yen
    'AUD_USD',  # Australian Dollar
    'USD_CAD',  # Canadian Dollar
    'USD_CHF',  # Swiss Franc
    'EUR_JPY',  # Euro/Yen
    'GBP_JPY'   # Pound/Yen
]

# Major Commodities
COMMODITIES = [
    'XAU_USD',  # Gold
    'XAG_USD'   # Silver
]

# Major Indices
INDICES = [
    'US30_USD',    # US Wall St 30
    'SPX500_USD',  # US SPX 500
    'NAS100_USD'   # US Nas 100
]

# Combine all instruments
INSTRUMENTS = FOREX_PAIRS + COMMODITIES + INDICES

# Trading Instruments
INSTRUMENT_SETTINGS = {
    # Forex Pairs
    'EUR_USD': {'pip_value': 0.0001, 'min_spread': 0.00010},
    'GBP_USD': {'pip_value': 0.0001, 'min_spread': 0.00015},
    'USD_JPY': {'pip_value': 0.01, 'min_spread': 0.015},
    'AUD_USD': {'pip_value': 0.0001, 'min_spread': 0.00015},
    'USD_CAD': {'pip_value': 0.0001, 'min_spread': 0.00015},
    'USD_CHF': {'pip_value': 0.0001, 'min_spread': 0.00015},
    'EUR_JPY': {'pip_value': 0.01, 'min_spread': 0.015},
    'GBP_JPY': {'pip_value': 0.01, 'min_spread': 0.015},
    
    # Commodities
    'XAU_USD': {'pip_value': 0.01, 'min_spread': 0.35},
    'XAG_USD': {'pip_value': 0.01, 'min_spread': 0.03},
    
    # Indices
    'US30_USD': {'pip_value': 1.0, 'min_spread': 2.0},
    'SPX500_USD': {'pip_value': 0.1, 'min_spread': 0.4},
    'NAS100_USD': {'pip_value': 0.1, 'min_spread': 0.6}
}

# Timeframe Settings
TIMEFRAME = 'M30','H1','H2' # 30 -minutes candles # 1-hour candles # 2 -hours candles 


# Trading Hours (UTC)
TRADING_HOURS = {
    'FOREX': {
        'start': time(hour=0, minute=0),
        'end': time(hour=23, minute=59),
        'active_sessions': {
            'ASIAN': {'start': time(0, 0), 'end': time(8, 0)},
            'LONDON': {'start': time(8, 0), 'end': time(16, 0)},
            'NEW_YORK': {'start': time(13, 0), 'end': time(21, 0)}
        }
    },
    'COMMODITIES': {
        'start': time(hour=1, minute=0),
        'end': time(hour=23, minute=0)
    },
    'INDICES': {
        'start': time(hour=13, minute=30),  # 8:30 AM EST
        'end': time(hour=20, minute=0)      # 3:00 PM EST
    }
}

# Risk Management
RISK_LEVELS = {
    'LOW': {
        'position_size': 0.01,      # 1% of balance
        'stop_loss': 0.01,          # 1% risk per trade
        'max_trades': 3,            # Maximum concurrent trades
        'max_daily_loss': 0.02,     # 2% max daily loss
        'required_agreement': 1.0,   # 100% strategy agreement required
        'risk_per_trade': 1.0,      # 1% risk per trade
        'max_position_size': 100000, # Maximum position size
        'stop_loss_pips': 50        # Stop loss in pips
    },
    'MEDIUM': {
        'position_size': 0.02,      # 2% of balance
        'stop_loss': 0.02,          # 2% risk per trade
        'max_trades': 5,            # Maximum concurrent trades
        'max_daily_loss': 0.04,     # 4% max daily loss
        'required_agreement': 0.66,  # 66% strategy agreement required
        'risk_per_trade': 2.0,      # 2% risk per trade
        'max_position_size': 200000, # Maximum position size
        'stop_loss_pips': 40        # Stop loss in pips
    },
    'HIGH': {
        'position_size': 0.03,      # 3% of balance
        'stop_loss': 0.03,          # 3% risk per trade
        'max_trades': 7,            # Maximum concurrent trades
        'max_daily_loss': 0.06,     # 6% max daily loss
        'required_agreement': 0.60,  # 60% strategy agreement required
        'risk_per_trade': 3.0,      # 3% risk per trade
        'max_position_size': 300000, # Maximum position size
        'stop_loss_pips': 30        # Stop loss in pips
    }
}

# Current Risk Profile
CURRENT_RISK_PROFILE = 'LOW'

# Apply risk profile settings
RISK_SETTINGS = RISK_LEVELS[CURRENT_RISK_PROFILE]

# Strategy Parameters
STRATEGY_PARAMS = {
    'MEAN_REVERSION': {
        'period': 20,
        'std_dev': 2.0,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30
    },
    'TREND_FOLLOWING': {
        'fast_ma': 20,
        'slow_ma': 50,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    },
    'ML_STRATEGY': {
        'training_samples': 500,
        'prediction_threshold': 0.65,
        'retraining_interval': 24 * 3600  # 24 hours
    }
}

STRATEGY_PARAMS.update({
    'MOMENTUM': {
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04
    },
    'GRID': {
        'grid_levels': 10,
        'grid_threshold': 0.001,
        'default_grid_threshold': 0.001,
        'volatility_threshold': 0.02,
        'stop_loss_pct': 0.01,
        'take_profit_pct': 0.02
    },
    'ARBITRAGE': {
        'min_spread': 0.0001,        # Minimum spread required for arbitrage
        'max_spread': 0.01,          # Maximum spread to consider
        'min_profit_threshold': 0.0002,  # Minimum profit threshold
        'max_position_size': 100000,  # Maximum position size
        'stop_loss_pct': 0.001,      # Stop loss percentage
        'take_profit_pct': 0.002     # Take profit percentage
    }
})


# Strategy Agreement Thresholds
STRATEGY_AGREEMENT_THRESHOLDS = {
    '2_strategies': 100,  # 100% agreement needed for 2 strategies
    '3_strategies': 66,   # 66% agreement needed for 3 strategies
    'default': 60         # 60% agreement needed for 4+ strategies
}

# Execution Settings
EXECUTION = {
    'retry_attempts': 3,
    'retry_wait_time': 5,
    'max_spread_percent': 0.001,
    'order_type': 'MARKET',
    'time_in_force': 'FOK',
    'position_fill': 'DEFAULT'
}

# NEWS
NEWS_FEEDS = {
    'EUR_USD': [
        'https://feeds.bbci.co.uk/news/business/rss.xml',
        'https://rss.reuters.com/feed/businessNews',
        'https://www.ecb.europa.eu/rss/press.html'
    ],
    'GBP_USD': [
        'https://feeds.bbci.co.uk/news/business/rss.xml',
        'https://www.bankofengland.co.uk/rss/news'
    ],
    'USD_JPY': [
        'https://www.boj.or.jp/en/rss/release_2023.xml',
        'https://rss.reuters.com/feed/businessNews'
    ],
    'default': [
        'https://feeds.bbci.co.uk/news/business/rss.xml',
        'https://rss.reuters.com/feed/businessNews'
    ]
}

NEWS_KEYWORDS = {
    'EUR_USD': [
        'EUR', 'euro', 'USD', 'dollar', 'ECB', 'Fed',
        'inflation', 'interest rate', 'European Union',
        'eurozone', 'Federal Reserve'
    ],
    'GBP_USD': [
        'GBP', 'pound', 'sterling', 'USD', 'dollar',
        'Bank of England', 'Fed', 'Brexit', 'UK economy',
        'British economy'
    ],
    'USD_JPY': [
        'JPY', 'yen', 'USD', 'dollar', 'Bank of Japan',
        'Fed', 'Japanese economy', 'BOJ'
    ],
    'default': [
        'forex', 'currency', 'central bank', 'interest rate',
        'inflation', 'economy'
    ]
}

# News analysis settings
NEWS_CACHE_DURATION = 15  # Minutes
NEWS_SENTIMENT_THRESHOLD = 0.3  # Sentiment threshold for trade signals
NEWS_IMPACT_THRESHOLD = 0.7  # Impact threshold for trade signals
NEWS_UPDATE_INTERVAL = 900  # 15 minutes in seconds


# Trading Limits
MAX_TRADES_PER_INSTRUMENT = 2
MAX_DAILY_TRADES = 10
MAX_DAILY_LOSS_PERCENT = 2.0  # 2% maximum daily loss

# Retry Settings
RETRY_ATTEMPTS = 3
RETRY_WAIT_TIME = 5  # seconds

# News Settings
NEWS_UPDATE_INTERVAL = 1800  # 30 minutes
NEWS_IMPACT_THRESHOLD = 0.7  # High impact threshold

# Machine Learning Settings
ML_MIN_TRAINING_SAMPLES = 1000
ML_TRAINING_INTERVAL = 86400  # 24 hours in seconds

# Logging Settings
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    'file': 'trading_bot.log',
    'max_files': 5,
    'max_size': 10 * 1024 * 1024  # 10 MB
}

# Debug Settings
DEBUG = {
    'enabled': True,
    'paper_trading': True,
    'simulate_latency': False,
    'max_messages': 1000
}

# Notification Settings
NOTIFICATIONS = {
    'startup': True,
    'shutdown': True,
    'error': True,
    'trade': True,
    'fill': True,
    'close': True,
    'balance_interval': 3600  # 1 hour
}

# Performance Tracking
PERFORMANCE = {
    'enabled': True,
    'metrics': [
        'win_rate',
        'profit_factor',
        'sharpe_ratio',
        'max_drawdown',
        'average_win',
        'average_loss'
    ],
    'update_interval': 3600  # 1 hour
}

# Timezone
TIMEZONE = 'UTC','EST'

TIMEFRAME = 'M30','H1','H2'