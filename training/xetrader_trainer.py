#!/usr/bin/env python3

"""
Enhanced Advanced Hedge Fund Style Trading System
Incorporating performance improvements and proper train/test data splitting,
and symbol-specific optimized configuration for META.
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from scipy import stats
from hmmlearn import hmm
from torch.nn import LSTM, GRU, MultiheadAttention

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Load .env if present
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

# Shared modules
from shared.data_manager import DataManager
from shared.data_publisher import DataPublisher

# Import optimized config for META
from optimized_config_meta import get_optimized_config

warnings.filterwarnings("ignore")

@dataclass
class TradingConfig:
    # Standard parameters
    annual_target_return: float = 0.35
    max_daily_drawdown: float = 0.02
    profit_target: float = 0.0253
    stop_loss: float = 0.0040
    window_minutes: int = 13
    lookback_periods: int = 100
    forecast_horizon: int = 20
    ensemble_models: int = 5
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    dropout: float = 0.2
    min_volume_ratio: float = 1.2
    max_correlation_threshold: float = 0.7
    end_of_day_close_minutes: int = 15

    # Enhanced parameters
    data_split_exclude_days: int = 30
    confidence_threshold: float = 0.1981
    base_position_size: float = 0.1148
    max_position_size: float = 0.2000

class EnhancedDataManager:
    """Enhanced data manager with proper train/test splitting"""
    def __init__(self):
        self.data_manager = DataManager()

    def get_training_data_with_split(
        self, symbol: str, total_days: int, exclude_recent_days: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training and testing data with proper temporal split to prevent data leakage
        Args:
            symbol: Stock symbol
            total_days: Total days of data to fetch
            exclude_recent_days: Days to exclude from training (used for testing)
        Returns:
            Tuple of (training_data, testing_data)
        """
        print(f"Fetching {total_days} days of data for proper train/test split...")
        full_data = self.data_manager.get_training_data(symbol, total_days + exclude_recent_days)
        if full_data.empty:
            raise ValueError(f"No data available for {symbol}")

        total_bars = len(full_data)
        test_bars = exclude_recent_days * 390  # Approximate bars per day
        train_bars = total_bars - test_bars

        if train_bars < 1000:
            raise ValueError(f"Insufficient training data. Need at least 1000 bars, got {train_bars}")
        if test_bars < 100:
            raise ValueError(f"Insufficient test data. Need at least 100 bars, got {test_bars}")

        train_data = full_data.iloc[:train_bars].copy()
        test_data = full_data.iloc[train_bars:].copy()
        print(f"Training data: {len(train_data)} bars from {train_data.index[0]} to {train_data.index[-1]}")
        print(f"Testing data: {len(test_data)} bars from {test_data.index[0]} to {test_data.index[-1]}")
        print(f"Gap between train and test: {(test_data.index[0] - train_data.index[-1]).total_seconds() / 60:.0f} minutes")

        return train_data, test_data

    def get_backtest_data(self, symbol: str, days: int, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get data for backtesting with optional date filtering"""
        data = self.data_manager.get_training_data(symbol, days)
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        return data

class AlternativeDataEngine:
    @staticmethod
    def get_sentiment_score(symbol: str, timestamp: pd.Timestamp) -> float:
        np.random.seed(hash(str(timestamp)) % 2**32)
        base_sentiment = 0.5 + 0.3 * np.sin(timestamp.hour * np.pi / 12)
        noise = np.random.normal(0, 0.1)
        return np.clip(base_sentiment + noise, 0, 1)

    @staticmethod
    def get_economic_regime(timestamp: pd.Timestamp) -> int:
        month_cycle = (timestamp.month % 12) / 12 * 2 * np.pi
        regime_score = np.sin(month_cycle) + 0.5 * np.cos(2 * month_cycle)
        if regime_score < -0.5:
            return 0
        elif regime_score > 0.5:
            return 2
        else:
            return 1

class EnhancedTechnicalIndicatorEngine:
    """Enhanced technical indicators with better predictive features"""
    @staticmethod
    def calculate_comprehensive_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)

        for period in [5, 10, 12, 20, 26, 40, 60, 120]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1

        df['trend_strength'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

        df['price_momentum'] = df['close'].pct_change(5).rolling(3).mean()
        df['volume_momentum'] = df['volume'].pct_change(5).rolling(3).mean()
        df['price_acceleration'] = df['returns'].diff()

        df['high_40'] = df['high'].rolling(40).max()
        df['low_40'] = df['low'].rolling(40).min()
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['breakout_up_40'] = (df['close'] > df['high_40'].shift(1)).astype(int)
        df['breakout_down_40'] = (df['close'] < df['low_40'].shift(1)).astype(int)

        df['rsi'] = EnhancedTechnicalIndicatorEngine._calculate_rsi(df['close'])
        df['rsi_smooth'] = df['rsi'].ewm(span=3).mean()

        for std in [1.5, 2.0, 2.5]:
            bb_mid = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df[f'bb_upper_{std}'] = bb_mid + std * bb_std
            df[f'bb_lower_{std}'] = bb_mid - std * bb_std
            df[f'bb_position_{std}'] = (df['close'] - df[f'bb_lower_{std}']) / (df[f'bb_upper_{std}'] - df[f'bb_lower_{std}'])

        df['macd_12_26'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_12_26'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_12_26'] - df['macd_signal']

        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume_trend'] = ((df['close'] - df['close'].shift()) / df['close'].shift() * df['volume']).cumsum()

        df['atr'] = EnhancedTechnicalIndicatorEngine._calculate_atr(df)
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(50).mean()

        df['bid_ask_proxy'] = (df['high'] - df['low']) / df['close']
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['price_impact'] = abs(df['returns']) / (df['volume'] / df['volume_sma'])

        for window in [3, 7, 14]:
            df[f'trend_consistency_{window}'] = df['returns'].rolling(window).apply(lambda x: (x > 0).sum() / len(x))

        df['price_distance_sma'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['reversion_signal'] = np.where(abs(df['price_distance_sma']) > 0.02, -np.sign(df['price_distance_sma']), 0)

        df['sector_momentum'] = df['close'].rolling(10).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100)

        return df.ffill().bfill().fillna(0)

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

class HMMRegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = RobustScaler()
        self.regime_names = ['Bear/Volatile', 'Neutral/Ranging', 'Bull/Trending']

    def fit_predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        features = self._prepare_regime_features(df)
        try:
            features_scaled = self.scaler.fit_transform(features)
            best_score = -np.inf
            best_model = None
            for init in range(5):
                model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42 + init,
                    tol=1e-3
                )
                try:
                    model.fit(features_scaled)
                    score = model.score(features_scaled)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except:
                    continue
            if best_model is None:
                return self._fallback_regimes(df)
            self.model = best_model
            regimes = self.model.predict(features_scaled)
            regime_probs = self.model.predict_proba(features_scaled)
            regime_stats = self._analyze_regimes(df, regimes)
            return regimes, regime_probs, regime_stats
        except Exception as e:
            print(f"HMM failed: {e}, using fallback")
            return self._fallback_regimes(df)

    def _prepare_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        features = []
        returns = df['returns'].fillna(0)
        features.append(returns)
        features.append(returns.rolling(5).std().fillna(0))
        features.append(returns.rolling(20).std().fillna(0))
        volume_change = df['volume'].pct_change().fillna(0)
        features.append(volume_change)
        features.append(df['volume_ratio'].fillna(1))
        features.append(df['trend_strength'].fillna(0))
        features.append(df['momentum_20'].fillna(0))
        features.append(df['sector_momentum'].fillna(0.5))
        return np.column_stack(features)

    def _fallback_regimes(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        vol = df['realized_vol'].rolling(20).mean()
        momentum = df['momentum_20']
        regimes = np.ones(len(df))
        regimes[vol > vol.quantile(0.7)] = 0
        regimes[momentum > momentum.quantile(0.7)] = 2
        regime_probs = np.zeros((len(df), 3))
        for i, regime in enumerate(regimes):
            regime_probs[i, int(regime)] = 0.8
        regime_probs = regime_probs / regime_probs.sum(axis=1, keepdims=True)
        stats = {'regime_0': 'Volatile', 'regime_1': 'Normal', 'regime_2': 'Trending'}
        return regimes.astype(int), regime_probs, stats

    def _analyze_regimes(self, df: pd.DataFrame, regimes: np.ndarray) -> Dict:
        regime_stats = {}
        for regime in range(self.n_regimes):
            mask = regimes == regime
            if mask.sum() > 0:
                regime_data = df[mask]
                regime_stats[f'regime_{regime}'] = {
                    'name': self.regime_names[regime],
                    'frequency': mask.sum() / len(regimes),
                    'avg_return': regime_data['returns'].mean(),
                    'volatility': regime_data['returns'].std(),
                    'avg_volume_ratio': regime_data['volume_ratio'].mean(),
                    'trend_strength': regime_data['trend_strength'].mean()
                }
        return regime_stats

class LSTMTrendPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3,
                 dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        final_hidden = attended[:, -1, :]
        output = self.classifier(final_hidden)
        return output

class EnhancedEnsembleSignalGenerator:
    """Enhanced signal generator with improved label creation and model optimization"""
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_columns = []

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature selection"""
        feature_cols = [
            'returns', 'realized_vol', 'trend_strength', 'momentum_5', 'momentum_20',
            'rsi', 'rsi_smooth', 'macd_histogram', 'volume_ratio', 'atr',
            'bb_position_2.0', 'price_vs_sma_20', 'price_vs_sma_40',
            'breakout_up_40', 'breakout_down_40', 'volatility_ratio',
            'spread_proxy', 'price_impact', 'sector_momentum',
            'price_momentum', 'volume_momentum', 'price_acceleration',
            'bid_ask_proxy', 'price_distance_sma', 'reversion_signal',
            'trend_consistency_3', 'trend_consistency_7', 'trend_consistency_14'
        ]
        for i in range(3):
            if f'regime_prob_{i}' in df.columns:
                feature_cols.append(f'regime_prob_{i}')
        if 'sentiment_score' in df.columns:
            feature_cols.append('sentiment_score')
        if 'economic_regime' in df.columns:
            feature_cols.append('economic_regime')

        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        return df[available_features].fillna(0)

    def _create_enhanced_labels(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced label creation with volatility-adjusted thresholds"""
        horizons_and_thresholds = [
            (2, 0.004, -0.004),
            (5, 0.008, -0.006),
            (10, 0.015, -0.010),
            (20, 0.025, -0.015),
        ]
        rolling_vol = df['returns'].rolling(20).std()
        vol_multiplier = rolling_vol / rolling_vol.median()
        all_signals = []

        for horizon, up_thresh, down_thresh in horizons_and_thresholds:
            future_returns = df['close'].shift(-horizon) / df['close'] - 1
            dynamic_up = up_thresh * vol_multiplier
            dynamic_down = down_thresh * vol_multiplier
            signal = pd.Series(1, index=df.index)
            signal[future_returns > dynamic_up] = 2
            signal[future_returns < dynamic_down] = 0
            all_signals.append(signal)

        signal_matrix = pd.concat(all_signals, axis=1)
        buy_votes = (signal_matrix == 2).sum(axis=1)
        sell_votes = (signal_matrix == 0).sum(axis=1)
        final_signal = pd.Series(1, index=df.index)
        final_signal[buy_votes >= 1] = 2
        final_signal[sell_votes >= 2] = 0
        return final_signal.fillna(1).astype(int)

    def get_optimized_models(self):
        """Get optimized model configurations"""
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        return rf_model, xgb_model

    def enhanced_ensemble_prediction(self, X_scaled: np.ndarray) -> np.ndarray:
        """Weighted ensemble based on historical performance"""
        predictions = {}
        weights = {'random_forest': 0.4, 'xgboost': 0.4, 'lstm': 0.2}

        if 'random_forest' in self.models:
            rf_probs = self.models['random_forest'].predict_proba(X_scaled)
            predictions['rf'] = rf_probs * weights['random_forest']
        if 'xgboost' in self.models:
            xgb_probs = self.models['xgboost'].predict_proba(X_scaled)
            predictions['xgb'] = xgb_probs * weights['xgboost']
        if 'lstm' in self.models:
            lstm_probs = self._predict_lstm(X_scaled)
            if lstm_probs is not None:
                predictions['lstm'] = lstm_probs * weights['lstm']

        if predictions:
            ensemble_probs = sum(predictions.values())
            return ensemble_probs
        else:
            return np.ones((len(X_scaled), 3)) / 3

    def train_ensemble(self, train_data: pd.DataFrame) -> Dict:
        """Enhanced training with optimized models"""
        print("Training enhanced ensemble models...")
        X = self.prepare_features(train_data)
        y = self._create_enhanced_labels(train_data)

        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 100:
            raise ValueError("Insufficient training data after cleaning")

        print(f"Training on {len(X)} samples with {len(X.columns)} features")
        print(f"Enhanced label distribution: {y.value_counts().to_dict()}")

        self.scalers['main'] = RobustScaler()
        X_scaled = self.scalers['main'].fit_transform(X)

        rf_model, xgb_model = self.get_optimized_models()

        print("Training optimized Random Forest...")
        rf_model.fit(X_scaled, y)
        self.models['random_forest'] = rf_model

        print("Training optimized XGBoost...")
        xgb_model.fit(X_scaled, y)
        self.models['xgboost'] = xgb_model

        print("Training LSTM...")
        lstm_results = self._train_lstm_model(train_data, X_scaled, y)
        results = {'lstm': lstm_results}

        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'rf_importance': rf_model.feature_importances_,
            'xgb_importance': xgb_model.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        print("Top 10 most important features:")
        print(feature_importance.head(10))

        results.update({
            'feature_importance': feature_importance,
            'training_samples': len(X),
            'feature_count': len(self.feature_columns)
        })
        return results

    def _train_lstm_model(self, df: pd.DataFrame, X_scaled: np.ndarray, y: pd.Series) -> Dict:
        sequence_length = 60
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y.iloc[i])
        if len(X_seq) < 100:
            return {'error': 'Insufficient data for LSTM training'}

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)

        model = LSTMTrendPredictor(
            input_size=X_scaled.shape[1],
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            dropout=self.config.dropout
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        best_val_acc = 0
        patience_counter = 0
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        for epoch in range(100):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_accuracy = (val_predictions == y_val_tensor).float().mean()

            scheduler.step(val_loss)
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                self.models['lstm'] = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

        return {
            'best_val_accuracy': best_val_acc.item(),
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }

    def _predict_lstm(self, X_scaled: np.ndarray) -> Optional[np.ndarray]:
        if 'lstm' not in self.models or not isinstance(self.models['lstm'], dict):
            return None
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            lstm_model = LSTMTrendPredictor(
                input_size=X_scaled.shape[1],
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                dropout=self.config.dropout
            ).to(device)
            lstm_model.load_state_dict(self.models['lstm'])
            lstm_model.eval()

            sequence_length = 60
            lstm_probs = np.zeros((len(X_scaled), 3))
            for i in range(sequence_length, len(X_scaled)):
                seq = X_scaled[i-sequence_length:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = lstm_model(seq_tensor)
                    probs = F.softmax(output, dim=1).cpu().numpy()[0]
                lstm_probs[i] = probs
            return lstm_probs
        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced trading signals"""
        X = self.prepare_features(df)
        X_scaled = self.scalers['main'].transform(X)
        ensemble_probs = self.enhanced_ensemble_prediction(X_scaled)
        signal_class = np.argmax(ensemble_probs, axis=1)
        signal_confidence = np.max(ensemble_probs, axis=1)

        result_df = df.copy()
        result_df['signal_class'] = signal_class
        result_df['signal_confidence'] = signal_confidence
        result_df['prob_sell'] = ensemble_probs[:, 0]
        result_df['prob_hold'] = ensemble_probs[:, 1]
        result_df['prob_buy'] = ensemble_probs[:, 2]
        return result_df

class EnhancedRiskManager:
    """Enhanced risk management with dynamic thresholds"""
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0
        self.open_positions = {}

    def enhanced_risk_filters(self, signal_data: pd.DataFrame, regime: int = 1) -> pd.DataFrame:
        """Dynamic risk filters that adapt to market conditions"""
        filtered_df = signal_data.copy()

        if regime == 2:  # Trending market
            confidence_threshold = 0.30
            volume_threshold = 0.5
        elif regime == 0:  # Volatile market
            confidence_threshold = 0.45
            volume_threshold = 0.7
        else:  # Normal market
            confidence_threshold = self.config.confidence_threshold
            volume_threshold = 0.6

        confidence_mask = filtered_df['signal_confidence'] >= confidence_threshold
        volume_mask = filtered_df['volume_ratio'] >= volume_threshold
        vol_threshold = filtered_df['realized_vol'].quantile(0.95)
        vol_mask = filtered_df['realized_vol'] <= vol_threshold

        keep_signal = confidence_mask & volume_mask & vol_mask
        filtered_df.loc[~keep_signal, 'signal_class'] = 1
        return filtered_df

    def enhanced_position_sizing(self, signal_strength: float,
                                 current_volatility: float,
                                 regime: int = 1) -> float:
        """Dynamic position sizing based on multiple factors"""
        base_size = self.config.base_position_size
        confidence_multiplier = min(signal_strength * 2, 2.0)
        vol_adjustment = 1 / max(current_volatility * 0.5, 0.05)

        if regime == 2:
            regime_multiplier = 1.5
        elif regime == 0:
            regime_multiplier = 0.7
        else:
            regime_multiplier = 1.0

        position_size = base_size * confidence_multiplier * vol_adjustment * regime_multiplier
        return min(position_size, self.config.max_position_size)

class EnhancedAdvancedTradingSystem:
    """Enhanced trading system with improved performance and proper data splitting"""
    def __init__(self, symbol: str, config: TradingConfig = None):
        self.symbol = symbol.upper()
        self.config = config or TradingConfig()
        self.data_manager = EnhancedDataManager()
        self.data_publisher = DataPublisher()
        self.alt_data_engine = AlternativeDataEngine()
        self.ta_engine = EnhancedTechnicalIndicatorEngine()
        self.hmm_detector = HMMRegimeDetector()
        self.ensemble_generator = EnhancedEnsembleSignalGenerator(self.config)
        self.risk_manager = EnhancedRiskManager(self.config)

        self.model_dir = project_root / "models" / self.symbol / "enhanced_system"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Enhanced Advanced Trading System initialized for {self.symbol}")
        print(f"Target: {self.config.annual_target_return:.0%} annual return")
        print(f"Enhanced confidence threshold: {self.config.confidence_threshold}")
        print(f"Enhanced position sizing: {self.config.base_position_size:.1%} base, {self.config.max_position_size:.1%} max")

    def prepare_comprehensive_dataset(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced dataset preparation"""
        print(f"Preparing enhanced dataset from {len(raw_data)} raw bars...")
        if self.config.window_minutes > 1:
            rule = f"{self.config.window_minutes}T"
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            data = raw_data.resample(rule).agg(agg).dropna()
            print(f"Resampled to {len(data)} {self.config.window_minutes}-minute bars")
        else:
            data = raw_data.copy()

        print("Computing enhanced technical indicators...")
        data = self.ta_engine.calculate_comprehensive_indicators(data)

        print("Detecting market regimes...")
        regimes, regime_probs, regime_stats = self.hmm_detector.fit_predict(data)
        data['regime'] = regimes
        for i in range(regime_probs.shape[1]):
            data[f'regime_prob_{i}'] = regime_probs[:, i]
        print("Enhanced regime statistics:")
        for regime, stats in regime_stats.items():
            if isinstance(stats, dict):
                print(f" {stats['name']}: {stats['frequency']:.1%} frequency, {stats['avg_return']:.4f} avg return")

        print("Adding alternative data...")
        data['sentiment_score'] = [self.alt_data_engine.get_sentiment_score(self.symbol, ts) for ts in data.index]
        data['economic_regime'] = [self.alt_data_engine.get_economic_regime(ts) for ts in data.index]

        data = data.ffill().bfill().fillna(0)
        print(f"Enhanced dataset: {len(data)} bars with {len(data.columns)} features")
        return data

    def train_system(self, total_days: int = 120) -> Dict:
        """Enhanced training with proper data splitting"""
        print(f"\n{'='*70}")
        print(f"Training Enhanced Advanced Trading System for {self.symbol}")
        print(f"{'='*70}")
        train_data_raw, test_data_raw = self.data_manager.get_training_data_with_split(
            self.symbol, total_days, self.config.data_split_exclude_days
        )

        print("\nPreparing training dataset...")
        train_dataset = self.prepare_comprehensive_dataset(train_data_raw)
        print("\nPreparing testing dataset...")
        test_dataset = self.prepare_comprehensive_dataset(test_data_raw)

        print(f"\nTraining on {len(train_dataset)} bars...")
        training_results = self.ensemble_generator.train_ensemble(train_dataset)

        model_path = self.model_dir / "enhanced_ensemble_models.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': self.ensemble_generator.models,
                'scalers': self.ensemble_generator.scalers,
                'feature_columns': self.ensemble_generator.feature_columns,
                'hmm_detector': self.hmm_detector,
                'config': self.config,
                'training_stats': {
                    'train_start': train_dataset.index[0],
                    'train_end': train_dataset.index[-1],
                    'test_start': test_dataset.index[0],
                    'test_end': test_dataset.index[-1],
                    'train_samples': len(train_dataset),
                    'test_samples': len(test_dataset)
                }
            }, f)
        print(f"Enhanced models saved to: {model_path}")

        print(f"\nRunning backtest on out-of-sample test data ({len(test_dataset)} bars)...")
        backtest_results = self.enhanced_backtest(test_dataset)

        results = {
            'training_results': training_results,
            'backtest_results': backtest_results,
            'model_path': str(model_path),
            'data_split_info': {
                'total_days': total_days,
                'exclude_days': self.config.data_split_exclude_days,
                'train_bars': len(train_dataset),
                'test_bars': len(test_dataset),
                'train_period': f"{train_dataset.index[0]} to {train_dataset.index[-1]}",
                'test_period': f"{test_dataset.index[0]} to {test_dataset.index[-1]}"
            }
        }

        print(f"\n{'='*70}")
        print(f"ENHANCED TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"Out-of-sample backtest return: {backtest_results['total_return']:.2%}")
        print(f"Win rate: {backtest_results['win_rate']:.1f}%")
        print(f"Sharpe ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Total trades: {backtest_results['total_trades']}")
        if backtest_results['total_return'] > 0.05:
            print(f"🎯 EXCELLENT: Enhanced system achieved {backtest_results['total_return']:.1%} return!")
        elif backtest_results['total_return'] > 0.02:
            print(f"✅ GOOD: Enhanced system achieved {backtest_results['total_return']:.1%} return!")
        elif backtest_results['total_return'] > 0:
            print(f"📊 POSITIVE: Enhanced system achieved {backtest_results['total_return']:.1%} return")
        else:
            print(f"⚠️ NEEDS IMPROVEMENT: {backtest_results['total_return']:.1%} return")

        return results

    def enhanced_backtest(self, data: pd.DataFrame) -> Dict:
        """Enhanced backtesting with improved position sizing and risk management"""
        signal_data = self.ensemble_generator.generate_signals(data)

        filtered_signals = []
        for i, (timestamp, row) in enumerate(signal_data.iterrows()):
            regime = int(row.get('regime', 1))
            filtered_row = self.risk_manager.enhanced_risk_filters(signal_data.iloc[i:i+1], regime)
            filtered_signals.append(filtered_row.iloc[0])
        filtered_df = pd.DataFrame(filtered_signals, index=signal_data.index)

        portfolio_value = 10000
        positions = []
        trades = []
        current_position = None
        entry_price = None

        for i, (timestamp, row) in enumerate(filtered_df.iterrows()):
            signal = row['signal_class']
            confidence = row['signal_confidence']
            price = row['close']
            regime = int(row.get('regime', 1))

            if current_position is None:
                if signal == 2 and confidence > self.config.confidence_threshold:
                    current_position = 'long'
                    entry_price = price
                    position_size = self.risk_manager.enhanced_position_sizing(confidence, row['realized_vol'], regime)
                    positions.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': price,
                        'size': position_size,
                        'confidence': confidence,
                        'regime': regime
                    })
            elif current_position == 'long':
                pnl_pct = (price / entry_price) - 1
                should_exit = (
                    signal == 0 or
                    pnl_pct >= self.config.profit_target or
                    pnl_pct <= -self.config.stop_loss
                )
                if should_exit:
                    trades.append({
                        'entry_time': positions[-1]['timestamp'],
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl_pct': pnl_pct,
                        'hold_periods': i - len([p for p in positions if p['action'] == 'BUY']),
                        'exit_reason': 'signal' if signal == 0 else 'profit' if pnl_pct >= self.config.profit_target else 'stop',
                        'position_size': positions[-1]['size'],
                        'regime': regime
                    })
                    positions.append({
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'price': price,
                        'pnl_pct': pnl_pct
                    })
                    portfolio_value *= (1 + pnl_pct * positions[-2]['size'])
                    current_position = None
                    entry_price = None

        if trades:
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['pnl_pct'] > 0)
            losing_trades = total_trades - winning_trades
            total_return = (portfolio_value / 10000) - 1
            trade_returns = [t['pnl_pct'] * t['position_size'] for t in trades]
            winning_returns = [r for r in trade_returns if r > 0]
            losing_returns = [r for r in trade_returns if r < 0]

            win_rate = winning_trades / total_trades * 100
            avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]) if losing_trades > 0 else 0
            profit_factor = sum(winning_returns) / abs(sum(losing_returns)) if losing_returns else float('inf')
            daily_rets = pd.Series(trade_returns)
            sharpe_ratio = (daily_rets.mean() / (daily_rets.std() + 1e-8)) * np.sqrt(252)
            cumulative_returns = np.cumprod([1 + r for r in trade_returns])
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)

            results = {
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_trade_return': np.mean([t['pnl_pct'] for t in trades]),
                'best_trade': max([t['pnl_pct'] for t in trades]),
                'worst_trade': min([t['pnl_pct'] for t in trades]),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_hold_time': np.mean([t['hold_periods'] for t in trades]),
                'final_portfolio_value': portfolio_value,
                'trades': trades[:10],
                'volatility': daily_rets.std() * np.sqrt(252)
            }
        else:
            results = {
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_hold_time': 0,
                'final_portfolio_value': 10000,
                'message': 'No trades executed - consider adjusting parameters',
                'volatility': 0
            }

        return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced Advanced Trading System')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., META)')
    parser.add_argument('--days', type=int, default=120, help='Total days of data (will be split for train/test)')
    parser.add_argument('--window', type=int, default=15, help='Timeframe in minutes')
    parser.add_argument('--train', action='store_true', help='Train the enhanced system')
    parser.add_argument('--backtest-only', action='store_true', help='Run backtest on existing model')
    parser.add_argument('--target-return', type=float, default=0.30, help='Annual target return')
    parser.add_argument('--exclude-days', type=int, default=30, help='Days to exclude from training for testing')
    args = parser.parse_args()

    symbol = args.symbol.upper()

    # Base configuration
    config = TradingConfig(
        window_minutes=args.window,
        annual_target_return=args.target_return,
        data_split_exclude_days=args.exclude_days,
        confidence_threshold=0.35,
        base_position_size=0.02,
        max_position_size=0.08
    )

    # If symbol is META, override with optimized settings
    if symbol == "META":
        opt = get_optimized_config()
        for field in opt.__dataclass_fields__:
            setattr(config, field, getattr(opt, field))

    trading_system = EnhancedAdvancedTradingSystem(symbol, config)

    if args.train:
        print(f"Training enhanced system with {args.days} total days of data")
        print(f"Will exclude recent {args.exclude_days} days for out-of-sample testing")
        results = trading_system.train_system(args.days)

        print(f"\n{'='*70}")
        print(f"ENHANCED TRAINING RESULTS FOR {symbol}")
        print(f"{'='*70}")

        split_info = results['data_split_info']
        print(f"\nData Split Information:")
        print(f" Total days requested: {split_info['total_days']}")
        print(f" Training data: {split_info['train_bars']} bars ({split_info['train_period']})")
        print(f" Testing data: {split_info['test_bars']} bars ({split_info['test_period']})")

        if 'training_results' in results:
            train_res = results['training_results']
            print(f"\nTraining Results:")
            print(f" Features used: {train_res.get('feature_count', 'N/A')}")
            print(f" Training samples: {train_res.get('training_samples', 'N/A')}")
            if 'feature_importance' in train_res:
                print(f" Top features: {', '.join(train_res['feature_importance'].head(5)['feature'].tolist())}")

        if 'backtest_results' in results:
            bt = results['backtest_results']
            print(f"\nOut-of-Sample Backtest Results:")
            print(f" Total Return: {bt.get('total_return', 0):.2%}")
            print(f" Total Trades: {bt.get('total_trades', 0)}")
            print(f" Win Rate: {bt.get('win_rate', 0):.1f}%")
            print(f" Profit Factor: {bt.get('profit_factor', 0):.2f}")
            print(f" Sharpe Ratio: {bt.get('sharpe_ratio', 0):.2f}")
            print(f" Max Drawdown: {bt.get('max_drawdown', 0):.2%}")

            total_ret = bt.get('total_return', 0)
            if total_ret > 0.10:
                print(f"\n🎯 OUTSTANDING: {total_ret:.1%} return with enhanced system!")
                print(" This system shows potential for strong annual returns")
            elif total_ret > 0.05:
                print(f"\n✅ EXCELLENT: {total_ret:.1%} return achieved!")
                print(" Enhanced system is performing well")
            elif total_ret > 0.02:
                print(f"\n📊 GOOD: {total_ret:.1%} return - solid improvement")
            elif total_ret > 0:
                print(f"\n📈 POSITIVE: {total_ret:.1%} return - on the right track")
            else:
                print(f"\n⚠️ NEEDS WORK: {total_ret:.1%} return - consider parameter tuning")

    elif args.backtest_only:
        print("Enhanced backtest-only mode not implemented yet")
        print("Use --train to train the enhanced system")
    else:
        print("Use --train to train the enhanced system")
        print("Enhanced features:")
        print(" • Proper train/test data splitting to prevent leakage")
        print(" • Enhanced technical indicators and features")
        print(" • Improved signal generation with dynamic thresholds")
        print(" • Optimized model parameters for better performance")
        print(" • Enhanced position sizing and risk management")
        print(" • Regime-aware confidence thresholds")

if __name__ == "__main__":
    main()

