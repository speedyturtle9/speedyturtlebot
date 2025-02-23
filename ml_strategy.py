# strategies/ml_strategy.py

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from trading_bot.utils.config import STRATEGY_PARAMS, ML_MIN_TRAINING_SAMPLES, ML_TRAINING_INTERVAL

class MLStrategy:
    def __init__(self):
        """Initialize Machine Learning Strategy"""
        self.params = STRATEGY_PARAMS['ML_STRATEGY']
        self.name = "Machine Learning"
        self.models = {}
        self.scalers = {}
        self.last_training = {}
        self.feature_columns = None
        logging.info(f"Initialized {self.name} Strategy")

    def get_signal(self, data, instrument):
        """Generate trading signals using ML predictions"""
        try:
            if instrument not in self.models:
                logging.warning(f"No trained model available for {instrument}")
                return {'signal': None, 'stop_loss': None, 'take_profit': None}

            # Prepare features
            features = self.prepare_features(data)
            if features is None or features.empty:
                return {'signal': None, 'stop_loss': None, 'take_profit': None}

            # Scale features
            scaled_features = self.scalers[instrument].transform(features.iloc[-1:])

            # Get prediction and probability
            prediction = self.models[instrument].predict(scaled_features)[0]
            probabilities = self.models[instrument].predict_proba(scaled_features)[0]
            confidence = max(probabilities)

            # Check confidence threshold
            if confidence < self.params['prediction_threshold']:
                return {'signal': None, 'stop_loss': None, 'take_profit': None}

            # Convert prediction to signal
            signal = "BUY" if prediction == 1 else "SELL"

            # Calculate stop loss and take profit
            current_price = data['close'].iloc[-1]
            atr = self.calculate_atr(data)
            
            stop_loss, take_profit = self.calculate_exit_points(
                signal=signal,
                current_price=current_price,
                atr=atr,
                confidence=confidence
            )

            return {
                'signal': signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'features': features.iloc[-1].to_dict()
            }

        except Exception as e:
            logging.error(f"Error generating ML signal for {instrument}: {str(e)}")
            return {'signal': None, 'stop_loss': None, 'take_profit': None}

    def train_model(self, data, instrument):
        """Train ML model for an instrument"""
        try:
            # Check if we have enough data
            if len(data) < ML_MIN_TRAINING_SAMPLES:
                logging.warning(f"Insufficient data for training {instrument} model")
                return False

            # Check if retraining is needed
            if instrument in self.last_training:
                time_since_training = datetime.now() - self.last_training[instrument]
                if time_since_training.total_seconds() < ML_TRAINING_INTERVAL:
                    return True

            # Prepare features and labels
            features = self.prepare_features(data)
            labels = self.prepare_labels(data)

            if features is None or labels is None:
                return False

            # Initialize scaler
            self.scalers[instrument] = StandardScaler()
            scaled_features = self.scalers[instrument].fit_transform(features)

            # Initialize and train model
            self.models[instrument] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            self.models[instrument].fit(scaled_features, labels)
            self.last_training[instrument] = datetime.now()

            logging.info(f"Successfully trained ML model for {instrument}")
            return True

        except Exception as e:
            logging.error(f"Error training ML model for {instrument}: {str(e)}")
            return False

    def prepare_features(self, data):
        """Prepare feature set for ML model"""
        try:
            df['returns'] = df['close'].pct_change(fill_method=None)

            # Technical indicators
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Price relative to moving averages
            df['price_to_sma_5'] = df['close'] / df['sma_5']
            df['price_to_sma_20'] = df['close'] / df['sma_20']
            df['price_to_sma_50'] = df['close'] / df['sma_50']
            
            # Momentum indicators
            df['rsi'] = self.calculate_rsi(df['close'])
            df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']

            # Additional features
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            df['close_open_ratio'] = (df['close'] - df['open']) / df['open']

            # Drop NaN values
            df = df.dropna()

            # Select feature columns
            feature_columns = [
                'returns', 'volatility',
                'price_to_sma_5', 'price_to_sma_20', 'price_to_sma_50',
                'rsi', 'macd', 'macd_signal',
                'high_low_ratio', 'close_open_ratio'
            ]
            
            if 'volume' in df.columns:
                feature_columns.append('volume_ratio')

            self.feature_columns = feature_columns
            return df[feature_columns]

        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            return None

    def prepare_labels(self, data):
        """Prepare labels for ML model"""
        try:
            # Calculate future returns
            future_returns = data['close'].shift(-5).pct_change(5)
            
            # Create binary labels
            labels = (future_returns > 0).astype(int)
            
            # Drop NaN values
            labels = labels.dropna()
            
            return labels

        except Exception as e:
            logging.error(f"Error preparing labels: {str(e)}")
            return None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logging.error(f"Error calculating RSI: {str(e)}")
            return None

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line
        except Exception as e:
            logging.error(f"Error calculating MACD: {str(e)}")
            return None, None

    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close'].shift()
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.iloc[-1]
        except Exception as e:
            logging.error(f"Error calculating ATR: {str(e)}")
            return None

    def calculate_exit_points(self, signal, current_price, atr, confidence):
        """Calculate stop loss and take profit levels"""
        try:
            if signal is None or atr is None:
                return None, None

            # Adjust multiplier based on confidence
            multiplier = 2 + (1 - confidence)  # Higher confidence = tighter stops

            if signal == "BUY":
                stop_loss = current_price - (atr * multiplier)
                take_profit = current_price + (atr * multiplier * 1.5)  # 1.5:1 reward-risk ratio
            else:
                stop_loss = current_price + (atr * multiplier)
                take_profit = current_price - (atr * multiplier * 1.5)

            return stop_loss, take_profit

        except Exception as e:
            logging.error(f"Error calculating exit points: {str(e)}")
            return None, None

    def validate_prediction(self, prediction, data):
        """Additional validation of ML predictions"""
        try:
            if prediction['signal'] is None:
                return False

            # Check market conditions
            current_price = data['close'].iloc[-1]
            sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['close'].rolling(window=50).mean().iloc[-1]
            
            # Trend alignment check
            trend_aligned = False
            if prediction['signal'] == "BUY":
                trend_aligned = current_price > sma_20 > sma_50
            else:
                trend_aligned = current_price < sma_20 < sma_50

            # Volume check
            if 'volume' in data.columns:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                volume_sufficient = current_volume >= avg_volume
            else:
                volume_sufficient = True

            # Volatility check
            volatility = data['close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
            volatility_acceptable = 0.10 <= volatility <= 0.40  # Acceptable volatility range

            # Combined validation
            return (trend_aligned and 
                   volume_sufficient and 
                   volatility_acceptable and 
                   prediction['confidence'] >= self.params['prediction_threshold'])

        except Exception as e:
            logging.error(f"Error validating prediction: {str(e)}")
            return False

    def update_model(self, instrument, new_data, actual_outcome):
        """Update model with new data"""
        try:
            if instrument not in self.models:
                return False

            # Prepare new features and label
            features = self.prepare_features(new_data)
            if features is None or features.empty:
                return False

            # Scale new features
            scaled_features = self.scalers[instrument].transform(features)

            # Update model with new data
            self.models[instrument].partial_fit(
                scaled_features,
                [1 if actual_outcome > 0 else 0],
                classes=[0, 1]
            )

            logging.info(f"Updated ML model for {instrument}")
            return True

        except Exception as e:
            logging.error(f"Error updating model for {instrument}: {str(e)}")
            return False

    def get_model_metrics(self, instrument):
        """Get model performance metrics"""
        try:
            if instrument not in self.models:
                return None

            return {
                'feature_importance': dict(zip(
                    self.feature_columns,
                    self.models[instrument].feature_importances_
                )),
                'last_training': self.last_training.get(instrument),
                'n_estimators': self.models[instrument].n_estimators,
                'max_depth': self.models[instrument].max_depth
            }

        except Exception as e:
            logging.error(f"Error getting model metrics for {instrument}: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup ML strategy resources"""
        try:
            self.models.clear()
            self.scalers.clear()
            self.last_training.clear()
            logging.info("ML Strategy cleanup completed")
        except Exception as e:
            logging.error(f"Error in ML strategy cleanup: {str(e)}")
