#!/usr/bin/env python3
"""
Advanced Hedge Fund Style Trading System
Incorporating Renaissance Technologies methodologies and latest predictive analytics
- Multi-strategy ensemble approach
- Advanced ML with LSTM + ensemble methods  
- HMM regime detection + 40-20 trend following
- Intraday focus with mandatory end-of-day closure
- Systematic risk management eliminating bias
Target: 20-42% annual returns through systematic trading
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from scipy import stats
from hmmlearn import hmm

# Deep learning imports
from torch.nn import LSTM, GRU, MultiheadAttention

# Environment setup
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

from shared.data_manager import DataManager
from shared.data_publisher import DataPublisher

warnings.filterwarnings("ignore")

@dataclass
class TradingConfig:
    """Configuration for advanced trading system"""
    # Performance targets
    annual_target_return: float = 0.30  # 30% target (conservative vs Renaissance's 70%)
    max_daily_drawdown: float = 0.02   # 2% max daily loss
    profit_target: float = 0.025       # 2.5% per trade target (realistic intraday)
    stop_loss: float = 0.015           # 1.5% stop loss
    
    # Time windows
    window_minutes: int = 15
    lookback_periods: int = 100
    forecast_horizon: int = 20
    
    # Model parameters  
    ensemble_models: int = 5
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    dropout: float = 0.2
    
    # Trading rules
    min_volume_ratio: float = 1.2      # Minimum volume for trade entry
    max_correlation_threshold: float = 0.7  # Position correlation limit
    end_of_day_close_minutes: int = 15 # Close all positions 15min before close

class AlternativeDataEngine:
    """Alternative data processing (simulated for demonstration)"""
    
    @staticmethod
    def get_sentiment_score(symbol: str, timestamp: pd.Timestamp) -> float:
        """Simulated sentiment analysis (in production: integrate real news APIs)"""
        # This would integrate with news APIs, social media, earnings calls
        # For now, generate realistic sentiment patterns
        np.random.seed(hash(str(timestamp)) % 2**32)
        base_sentiment = 0.5 + 0.3 * np.sin(timestamp.hour * np.pi / 12)
        noise = np.random.normal(0, 0.1)
        return np.clip(base_sentiment + noise, 0, 1)
    
    @staticmethod
    def get_economic_regime(timestamp: pd.Timestamp) -> int:
        """Simulated economic regime detection (0=recession, 1=growth, 2=inflation)"""
        # In production: integrate with economic indicators, yield curves, etc.
        month_cycle = (timestamp.month % 12) / 12 * 2 * np.pi
        regime_score = np.sin(month_cycle) + 0.5 * np.cos(2 * month_cycle)
        if regime_score < -0.5:
            return 0  # Recession regime
        elif regime_score > 0.5:
            return 2  # Inflation regime
        else:
            return 1  # Growth regime

class TechnicalIndicatorEngine:
    """Advanced technical analysis with Renaissance-style indicators"""
    
    @staticmethod
    def calculate_comprehensive_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 50+ technical indicators used by top hedge funds"""
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Multi-timeframe moving averages (Renaissance approach)
        for period in [5, 10, 12, 20, 26, 40, 60, 120]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        # Trend strength indicators
        df['trend_strength'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # The famous "40-20" breakout system used by hedge funds
        df['high_40'] = df['high'].rolling(40).max()
        df['low_40'] = df['low'].rolling(40).min()
        df['high_20'] = df['high'].rolling(20).max()  
        df['low_20'] = df['low'].rolling(20).min()
        
        # Breakout signals
        df['breakout_up_40'] = (df['close'] > df['high_40'].shift(1)).astype(int)
        df['breakout_down_40'] = (df['close'] < df['low_40'].shift(1)).astype(int)
        df['breakout_exit_20'] = ((df['close'] < df['low_20'].shift(1)) | 
                                  (df['close'] > df['high_20'].shift(1))).astype(int)
        
        # Advanced oscillators
        df['rsi'] = TechnicalIndicatorEngine._calculate_rsi(df['close'])
        df['rsi_smooth'] = df['rsi'].ewm(span=3).mean()
        
        # Bollinger Bands with multiple standard deviations
        for std in [1.5, 2.0, 2.5]:
            bb_mid = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df[f'bb_upper_{std}'] = bb_mid + std * bb_std
            df[f'bb_lower_{std}'] = bb_mid - std * bb_std
            df[f'bb_position_{std}'] = (df['close'] - df[f'bb_lower_{std}']) / \
                                       (df[f'bb_upper_{std}'] - df[f'bb_lower_{std}'])
        
        # MACD variations
        df['macd_12_26'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_12_26'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_12_26'] - df['macd_signal']
        
        # Volume analysis (crucial for institutional strategies)
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume_trend'] = ((df['close'] - df['close'].shift()) / 
                                    df['close'].shift() * df['volume']).cumsum()
        
        # Volatility indicators
        df['atr'] = TechnicalIndicatorEngine._calculate_atr(df)
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(50).mean()
        
        # Market microstructure proxies
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['price_impact'] = abs(df['returns']) / (df['volume'] / df['volume_sma'])
        
        # Cross-asset momentum (simulated)
        df['sector_momentum'] = df['close'].rolling(10).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )
        
        return df.fillna(method='ffill').fillna(0)
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with proper handling"""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection (Renaissance approach)"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = RobustScaler()
        self.regime_names = ['Bear/Volatile', 'Neutral/Ranging', 'Bull/Trending']
        
    def fit_predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Fit HMM and return regimes with detailed analysis"""
        
        # Prepare features for regime detection
        features = self._prepare_regime_features(df)
        
        try:
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit HMM with multiple initializations for stability
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
            
            # Get regime predictions
            regimes = self.model.predict(features_scaled)
            regime_probs = self.model.predict_proba(features_scaled)
            
            # Analyze regime characteristics
            regime_stats = self._analyze_regimes(df, regimes)
            
            return regimes, regime_probs, regime_stats
            
        except Exception as e:
            print(f"HMM failed: {e}, using fallback")
            return self._fallback_regimes(df)
    
    def _prepare_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime detection"""
        features = []
        
        # Returns and volatility
        returns = df['returns'].fillna(0)
        features.append(returns)
        features.append(returns.rolling(5).std().fillna(0))
        features.append(returns.rolling(20).std().fillna(0))
        
        # Volume characteristics  
        volume_change = df['volume'].pct_change().fillna(0)
        features.append(volume_change)
        features.append(df['volume_ratio'].fillna(1))
        
        # Trend characteristics
        features.append(df['trend_strength'].fillna(0))
        features.append(df['momentum_20'].fillna(0))
        
        # Cross-correlations (simulated)
        features.append(df['sector_momentum'].fillna(0.5))
        
        return np.column_stack(features)
    
    def _fallback_regimes(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Simple fallback regime detection"""
        vol = df['realized_vol'].rolling(20).mean()
        momentum = df['momentum_20']
        
        regimes = np.ones(len(df))  # Default to regime 1
        regimes[vol > vol.quantile(0.7)] = 0  # High vol = regime 0
        regimes[momentum > momentum.quantile(0.7)] = 2  # Strong momentum = regime 2
        
        # Create dummy probabilities
        regime_probs = np.zeros((len(df), 3))
        for i, regime in enumerate(regimes):
            regime_probs[i, int(regime)] = 0.8
            regime_probs[i, :] = regime_probs[i, :] / regime_probs[i, :].sum()
        
        stats = {'regime_0': 'Volatile', 'regime_1': 'Normal', 'regime_2': 'Trending'}
        return regimes.astype(int), regime_probs, stats
    
    def _analyze_regimes(self, df: pd.DataFrame, regimes: np.ndarray) -> Dict:
        """Analyze characteristics of each regime"""
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
    """Advanced LSTM for price direction prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer LSTM with dropout
        self.lstm = LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        
        # Classification head
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
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism (self-attention)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last timestep
        final_hidden = attended[:, -1, :]
        
        # Classification
        output = self.classifier(final_hidden)
        return output

class EnsembleSignalGenerator:
    """Multi-model ensemble for signal generation"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ensemble models"""
        
        # Core technical features  
        feature_cols = [
            'returns', 'realized_vol', 'trend_strength', 'momentum_5', 'momentum_20',
            'rsi', 'rsi_smooth', 'macd_histogram', 'volume_ratio', 'atr',
            'bb_position_2.0', 'price_vs_sma_20', 'price_vs_sma_40',
            'breakout_up_40', 'breakout_down_40', 'volatility_ratio',
            'spread_proxy', 'price_impact', 'sector_momentum'
        ]
        
        # Add regime features
        for i in range(3):
            if f'regime_prob_{i}' in df.columns:
                feature_cols.append(f'regime_prob_{i}')
        
        # Add alternative data
        if 'sentiment_score' in df.columns:
            feature_cols.append('sentiment_score')
        if 'economic_regime' in df.columns:
            feature_cols.append('economic_regime')
        
        # Select available features
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        
        return df[available_features].fillna(0)
    
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create sophisticated multi-class labels for direction prediction"""
        
        # Look ahead multiple periods for robust labeling
        horizons = [5, 10, 15, 20]  # 15-min bars: 1.25hr, 2.5hr, 3.75hr, 5hr ahead
        
        signals = []
        for h in horizons:
            future_returns = df['close'].shift(-h) / df['close'] - 1
            
            # Multi-threshold classification
            signal = pd.Series(1, index=df.index)  # Default: hold
            signal[future_returns > self.config.profit_target] = 2  # Buy signal
            signal[future_returns < -self.config.stop_loss] = 0   # Sell/short signal
            
            signals.append(signal)
        
        # Ensemble voting across horizons
        signal_matrix = pd.concat(signals, axis=1)
        final_signal = signal_matrix.mode(axis=1)[0]  # Majority vote
        
        return final_signal.fillna(1).astype(int)  # Default to hold if no majority
    
    def train_ensemble(self, train_data: pd.DataFrame) -> Dict:
        """Train ensemble of models"""
        
        print("Training ensemble models...")
        
        # Prepare features and labels
        X = self.prepare_features(train_data)
        y = self.create_labels(train_data)
        
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            raise ValueError("Insufficient training data after cleaning")
        
        print(f"Training on {len(X)} samples with {len(X.columns)} features")
        print(f"Label distribution: {y.value_counts().to_dict()}")
        
        # Scale features
        self.scalers['main'] = RobustScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        
        # Train multiple models
        results = {}
        
        # 1. Random Forest (Renaissance favorite)
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_scaled, y)
        self.models['random_forest'] = rf_model
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(feature_importance.head(10))
        
        # 2. XGBoost (modern ensemble method)
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_scaled, y)
        self.models['xgboost'] = xgb_model
        
        # 3. LSTM Model (for sequence learning)
        print("Training LSTM...")
        lstm_results = self._train_lstm_model(train_data, X_scaled, y)
        results['lstm'] = lstm_results
        
        # Model validation using time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in [('rf', rf_model), ('xgb', xgb_model)]:
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                score = model.score(X_val_fold, y_val_fold)
                scores.append(score)
            
            results[name] = {
                'cv_score_mean': np.mean(scores),
                'cv_score_std': np.std(scores),
                'feature_importance': feature_importance if name == 'rf' else None
            }
            
            print(f"{name.upper()} CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        return results
    
    def _train_lstm_model(self, df: pd.DataFrame, X_scaled: np.ndarray, y: pd.Series) -> Dict:
        """Train LSTM model for sequence prediction"""
        
        # Create sequences
        sequence_length = 60  # 15 hours of 15-min data
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y.iloc[i])
        
        if len(X_seq) < 100:
            return {'error': 'Insufficient data for LSTM training'}
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Train/validation split
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Convert to PyTorch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        
        # Initialize model
        model = LSTMTrendPredictor(
            input_size=X_scaled.shape[1],
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            dropout=self.config.dropout
        ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
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
            
            # Validation
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
                # Save best model
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
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble trading signals"""
        
        # Prepare features
        X = self.prepare_features(df)
        X_scaled = self.scalers['main'].transform(X)
        
        # Get predictions from each model
        predictions = {}
        
        # Random Forest
        if 'random_forest' in self.models:
            rf_probs = self.models['random_forest'].predict_proba(X_scaled)
            predictions['rf'] = rf_probs
        
        # XGBoost
        if 'xgboost' in self.models:
            xgb_probs = self.models['xgboost'].predict_proba(X_scaled)
            predictions['xgb'] = xgb_probs
        
        # LSTM (if trained successfully)
        if 'lstm' in self.models and isinstance(self.models['lstm'], dict):
            # Reconstruct and use LSTM model
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                lstm_model = LSTMTrendPredictor(
                    input_size=X_scaled.shape[1],
                    hidden_size=self.config.lstm_hidden_size,
                    num_layers=self.config.lstm_num_layers,
                    dropout=self.config.dropout
                ).to(device)
                lstm_model.load_state_dict(self.models['lstm'])
                
                # Create sequences for LSTM
                sequence_length = 60
                lstm_probs = np.zeros((len(X_scaled), 3))
                
                for i in range(sequence_length, len(X_scaled)):
                    seq = X_scaled[i-sequence_length:i]
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        lstm_model.eval()
                        output = lstm_model(seq_tensor)
                        probs = F.softmax(output, dim=1).cpu().numpy()[0]
                        lstm_probs[i] = probs
                
                predictions['lstm'] = lstm_probs
                
            except Exception as e:
                print(f"LSTM prediction failed: {e}")
        
        # Ensemble the predictions
        if predictions:
            # Weighted ensemble (equal weights for simplicity)
            ensemble_probs = np.zeros((len(X_scaled), 3))
            
            for model_name, probs in predictions.items():
                ensemble_probs += probs
            
            ensemble_probs /= len(predictions)
            
            # Generate final signals
            signal_class = np.argmax(ensemble_probs, axis=1)
            signal_confidence = np.max(ensemble_probs, axis=1)
            
            # Add signals to dataframe
            result_df = df.copy()
            result_df['signal_class'] = signal_class
            result_df['signal_confidence'] = signal_confidence
            result_df['prob_sell'] = ensemble_probs[:, 0]
            result_df['prob_hold'] = ensemble_probs[:, 1]
            result_df['prob_buy'] = ensemble_probs[:, 2]
            
            return result_df
        
        else:
            # Fallback: simple signals
            result_df = df.copy()
            result_df['signal_class'] = 1  # Hold
            result_df['signal_confidence'] = 0.5
            return result_df

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0
        self.open_positions = {}
        self.correlation_matrix = None
        
    def check_position_limits(self, signal_df: pd.DataFrame, current_time: pd.Timestamp) -> pd.DataFrame:
        """Apply risk management filters to trading signals"""
        
        filtered_df = signal_df.copy()
        
        # 1. Daily drawdown limit
        if self.daily_pnl <= -self.config.max_daily_drawdown:
            print(f"Daily drawdown limit reached: {self.daily_pnl:.2%}")
            filtered_df['signal_class'] = 1  # Force hold
            return filtered_df
        
        # 2. End-of-day closure rule
        market_close = current_time.replace(hour=16, minute=0, second=0)
        close_time = market_close - timedelta(minutes=self.config.end_of_day_close_minutes)
        
        if current_time >= close_time:
            print("End-of-day closure window - no new positions")
            filtered_df['signal_class'] = 1  # Force hold/close
            return filtered_df
        
        # 3. Volume filter
        volume_filter = filtered_df['volume_ratio'] >= self.config.min_volume_ratio
        filtered_df.loc[~volume_filter, 'signal_class'] = 1
        
        # 4. Confidence threshold
        confidence_filter = filtered_df['signal_confidence'] >= 0.6
        filtered_df.loc[~confidence_filter, 'signal_class'] = 1
        
        # 5. Volatility regime filter (avoid trading in extreme volatility)
        vol_filter = filtered_df['realized_vol'] <= filtered_df['realized_vol'].quantile(0.95)
        filtered_df.loc[~vol_filter, 'signal_class'] = 1
        
        return filtered_df
    
    def calculate_position_size(self, signal_strength: float, current_volatility: float) -> float:
        """Calculate position size based on Kelly Criterion and volatility"""
        
        # Base position size (1% of portfolio)
        base_size = 0.01
        
        # Adjust for signal confidence
        confidence_multiplier = signal_strength
        
        # Adjust for volatility (inverse relationship)
        vol_adjustment = 1 / max(current_volatility, 0.1)
        
        # Kelly-inspired sizing
        position_size = base_size * confidence_multiplier * vol_adjustment
        
        # Cap at maximum 5% per position
        return min(position_size, 0.05)

class AdvancedTradingSystem:
    """Main trading system integrating all components"""
    
    def __init__(self, symbol: str, config: TradingConfig = None):
        self.symbol = symbol.upper()
        self.config = config or TradingConfig()
        
        # Initialize components
        self.data_manager = DataManager()
        self.data_publisher = DataPublisher()
        self.alt_data_engine = AlternativeDataEngine()
        self.ta_engine = TechnicalIndicatorEngine()
        self.hmm_detector = HMMRegimeDetector()
        self.ensemble_generator = EnsembleSignalGenerator(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Create model directory
        self.model_dir = project_root / "models" / symbol / "advanced_system"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Advanced Trading System initialized for {symbol}")
        print(f"Target: {self.config.annual_target_return:.0%} annual return")
        print(f"Window: {self.config.window_minutes} minutes")
    
    def prepare_comprehensive_dataset(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive dataset with all features"""
        
        print(f"Preparing comprehensive dataset from {len(raw_data)} raw bars...")
        
        # Resample to target timeframe
        if self.config.window_minutes > 1:
            rule = f"{self.config.window_minutes}T"
            agg = {
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            data = raw_data.resample(rule).agg(agg).dropna()
            print(f"Resampled to {len(data)} {self.config.window_minutes}-minute bars")
        else:
            data = raw_data.copy()
        
        # Technical indicators
        print("Computing technical indicators...")
        data = self.ta_engine.calculate_comprehensive_indicators(data)
        
        # HMM regime detection
        print("Detecting market regimes...")
        regimes, regime_probs, regime_stats = self.hmm_detector.fit_predict(data)
        
        data['regime'] = regimes
        for i in range(regime_probs.shape[1]):
            data[f'regime_prob_{i}'] = regime_probs[:, i]
        
        print("Regime statistics:")
        for regime, stats in regime_stats.items():
            if isinstance(stats, dict):
                print(f"  {stats['name']}: {stats['frequency']:.1%} frequency, "
                      f"{stats['avg_return']:.3f} avg return")
        
        # Alternative data (simulated)
        print("Adding alternative data...")
        data['sentiment_score'] = [
            self.alt_data_engine.get_sentiment_score(self.symbol, ts) 
            for ts in data.index
        ]
        data['economic_regime'] = [
            self.alt_data_engine.get_economic_regime(ts)
            for ts in data.index
        ]
        
        # Final cleanup
        data = data.fillna(method='ffill').fillna(0)
        
        print(f"Final dataset: {len(data)} bars with {len(data.columns)} features")
        return data
    
    def train_system(self, days: int = 90) -> Dict:
        """Train the complete trading system"""
        
        print(f"\n{'='*60}")
        print(f"Training Advanced Trading System for {self.symbol}")
        print(f"{'='*60}")
        
        # Load data
        raw_data = self.data_manager.get_training_data(self.symbol, days)
        if raw_data.empty:
            raise ValueError("No training data available")
        
        print(f"Loaded {len(raw_data)} 1-minute bars")
        
        # Prepare comprehensive dataset
        full_dataset = self.prepare_comprehensive_dataset(raw_data)
        
        # Train/test split (80/20)
        split_idx = int(len(full_dataset) * 0.8)
        train_data = full_dataset.iloc[:split_idx]
        test_data = full_dataset.iloc[split_idx:]
        
        print(f"Training data: {len(train_data)} bars")
        print(f"Test data: {len(test_data)} bars")
        
        # Train ensemble models
        training_results = self.ensemble_generator.train_ensemble(train_data)
        
        # Save trained models
        model_path = self.model_dir / "ensemble_models.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': self.ensemble_generator.models,
                'scalers': self.ensemble_generator.scalers,
                'feature_columns': self.ensemble_generator.feature_columns,
                'hmm_detector': self.hmm_detector,
                'config': self.config
            }, f)
        
        print(f"Models saved to: {model_path}")
        
        # Backtest on test data
        print("\nRunning backtest on test data...")
        backtest_results = self.backtest(test_data)
        
        # Combine results
        results = {
            'training_results': training_results,
            'backtest_results': backtest_results,
            'model_path': str(model_path)
        }
        
        print(f"\nTraining completed!")
        print(f"Backtest returns: {backtest_results['total_return']:.1%}")
        print(f"Win rate: {backtest_results['win_rate']:.1%}")
        print(f"Sharpe ratio: {backtest_results['sharpe_ratio']:.2f}")
        
        return results
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """Comprehensive backtesting"""
        
        # Generate signals
        signal_data = self.ensemble_generator.generate_signals(data)
        
        # Apply risk management
        filtered_signals = []
        for i, (timestamp, row) in enumerate(signal_data.iterrows()):
            filtered_row = self.risk_manager.check_position_limits(
                signal_data.iloc[i:i+1], timestamp
            )
            filtered_signals.append(filtered_row.iloc[0])
        
        filtered_df = pd.DataFrame(filtered_signals, index=signal_data.index)
        
        # Simulate trading
        portfolio_value = 10000
        positions = []
        trades = []
        daily_returns = []
        
        current_position = None
        entry_price = None
        
        for i, (timestamp, row) in enumerate(filtered_df.iterrows()):
            signal = row['signal_class']
            confidence = row['signal_confidence']
            price = row['close']
            
            # Position management
            if current_position is None:  # No position
                if signal == 2 and confidence > 0.6:  # Buy signal
                    current_position = 'long'
                    entry_price = price
                    position_size = self.risk_manager.calculate_position_size(
                        confidence, row['realized_vol']
                    )
                    positions.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': price,
                        'size': position_size,
                        'confidence': confidence
                    })
                    
            elif current_position == 'long':  # Long position
                # Check exit conditions
                pnl_pct = (price / entry_price) - 1
                
                should_exit = (
                    signal == 0 or  # Sell signal
                    pnl_pct >= self.config.profit_target or  # Take profit
                    pnl_pct <= -self.config.stop_loss or  # Stop loss
                    timestamp.hour >= 15  # End of day
                )
                
                if should_exit:
                    trades.append({
                        'entry_time': positions[-1]['timestamp'],
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl_pct': pnl_pct,
                        'hold_periods': i - len([p for p in positions if p['action'] == 'BUY']),
                        'exit_reason': 'signal' if signal == 0 else 
                                      'profit' if pnl_pct >= self.config.profit_target else
                                      'stop' if pnl_pct <= -self.config.stop_loss else 'eod'
                    })
                    
                    positions.append({
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'price': price,
                        'pnl_pct': pnl_pct
                    })
                    
                    # Update portfolio
                    portfolio_value *= (1 + pnl_pct * positions[-2]['size'])
                    
                    # Reset position
                    current_position = None
                    entry_price = None
        
        # Calculate performance metrics
        if trades:
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['pnl_pct'] > 0)
            total_return = (portfolio_value / 10000) - 1
            
            daily_rets = pd.Series([t['pnl_pct'] for t in trades])
            sharpe_ratio = daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(252)
            
            max_drawdown = 0
            peak = portfolio_value
            for trade in trades:
                current_value = peak * (1 + trade['pnl_pct'])
                if current_value < peak:
                    drawdown = (peak - current_value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                else:
                    peak = current_value
            
            avg_hold_time = np.mean([t['hold_periods'] for t in trades])
            
            results = {
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades * 100,
                'avg_trade_return': np.mean([t['pnl_pct'] for t in trades]),
                'best_trade': max([t['pnl_pct'] for t in trades]),
                'worst_trade': min([t['pnl_pct'] for t in trades]),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_hold_time': avg_hold_time,
                'final_portfolio_value': portfolio_value,
                'trades': trades[:10]  # First 10 trades for inspection
            }
        else:
            results = {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'message': 'No trades executed'
            }
        
        return results
    
    def generate_live_signals(self, current_data: pd.DataFrame) -> Dict:
        """Generate real-time trading signals"""
        
        # Prepare data
        enriched_data = self.prepare_comprehensive_dataset(current_data)
        
        # Generate signals
        signal_data = self.ensemble_generator.generate_signals(enriched_data)
        
        # Apply risk filters
        current_time = signal_data.index[-1]
        filtered_signals = self.risk_manager.check_position_limits(
            signal_data.iloc[-1:], current_time
        )
        
        latest_signal = filtered_signals.iloc[-1]
        
        return {
            'timestamp': current_time,
            'signal_class': int(latest_signal['signal_class']),
            'signal_confidence': float(latest_signal['signal_confidence']),
            'prob_buy': float(latest_signal.get('prob_buy', 0)),
            'prob_hold': float(latest_signal.get('prob_hold', 0)),
            'prob_sell': float(latest_signal.get('prob_sell', 0)),
            'current_price': float(latest_signal['close']),
            'volume_ratio': float(latest_signal['volume_ratio']),
            'regime': int(latest_signal['regime']),
            'regime_confidence': float(max([
                latest_signal.get('regime_prob_0', 0),
                latest_signal.get('regime_prob_1', 0),
                latest_signal.get('regime_prob_2', 0)
            ]))
        }

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Advanced Hedge Fund Style Trading System')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., META)')
    parser.add_argument('--days', type=int, default=90, help='Days of training data')
    parser.add_argument('--window', type=int, default=15, help='Timeframe in minutes')
    parser.add_argument('--train', action='store_true', help='Train the system')
    parser.add_argument('--backtest', action='store_true', help='Run backtest only')
    parser.add_argument('--target-return', type=float, default=0.30, help='Annual target return')
    
    args = parser.parse_args()
    
    # Configuration
    config = TradingConfig(
        window_minutes=args.window,
        annual_target_return=args.target_return
    )
    
    # Initialize system
    trading_system = AdvancedTradingSystem(args.symbol, config)
    
    if args.train:
        # Train the system
        results = trading_system.train_system(args.days)
        
        print(f"\n{'='*60}")
        print(f"TRAINING RESULTS FOR {args.symbol}")
        print(f"{'='*60}")
        
        if 'backtest_results' in results:
            bt = results['backtest_results']
            print(f"Backtest Performance:")
            print(f"  Total Return: {bt.get('total_return', 0):.1%}")
            print(f"  Total Trades: {bt.get('total_trades', 0)}")
            print(f"  Win Rate: {bt.get('win_rate', 0):.1%}")
            print(f"  Sharpe Ratio: {bt.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {bt.get('max_drawdown', 0):.1%}")
            
            if bt.get('total_return', 0) > 0.15:  # 15%+ return
                print(f"\n🎯 SUCCESS: System achieved {bt['total_return']:.1%} return!")
                print("   This system shows potential for 20-42% annual returns")
            else:
                print(f"\n📊 System trained successfully with {bt.get('total_return', 0):.1%} return")
                print("   Consider adjusting parameters or adding more data")
    
    elif args.backtest:
        # Load existing model and backtest
        print("Loading existing model for backtesting...")
        # Implementation for loading and backtesting existing models
        pass
    
    else:
        print("Use --train to train the system or --backtest to test existing models")

if __name__ == "__main__":
    main()

