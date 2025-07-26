#!/usr/bin/env python3
"""
Advanced Trading System Backtester
Loads trained models and runs comprehensive backtesting analysis
"""

import os
import sys
import argparse
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Project setup
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

from shared.data_manager import DataManager
from shared.data_publisher import DataPublisher

warnings.filterwarnings("ignore")

from dataclasses import dataclass

@dataclass
class TradingConfig:
    annual_target_return: float = 0.30
    max_daily_drawdown: float = 0.02
    profit_target: float = 0.025
    stop_loss: float = 0.015
    window_minutes: int = 15
    lookback_periods: int = 100
    forecast_horizon: int = 20
    ensemble_models: int = 5
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    dropout: float = 0.2
    min_volume_ratio: float = 1.2
    max_correlation_threshold: float = 0.7
    end_of_day_close_minutes: int = 15


# Add missing classes that were saved in the pickle file
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

class TechnicalIndicatorEngine:
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
        
        df['high_40'] = df['high'].rolling(40).max()
        df['low_40'] = df['low'].rolling(40).min()
        df['high_20'] = df['high'].rolling(20).max()  
        df['low_20'] = df['low'].rolling(20).min()
        
        df['breakout_up_40'] = (df['close'] > df['high_40'].shift(1)).astype(int)
        df['breakout_down_40'] = (df['close'] < df['low_40'].shift(1)).astype(int)
        df['breakout_exit_20'] = ((df['close'] < df['low_20'].shift(1)) | 
                                  (df['close'] > df['high_20'].shift(1))).astype(int)
        
        df['rsi'] = TechnicalIndicatorEngine._calculate_rsi(df['close'])
        df['rsi_smooth'] = df['rsi'].ewm(span=3).mean()
        
        for std in [1.5, 2.0, 2.5]:
            bb_mid = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df[f'bb_upper_{std}'] = bb_mid + std * bb_std
            df[f'bb_lower_{std}'] = bb_mid - std * bb_std
            df[f'bb_position_{std}'] = (df['close'] - df[f'bb_lower_{std}']) / \
                                       (df[f'bb_upper_{std}'] - df[f'bb_lower_{std}'])
        
        df['macd_12_26'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_12_26'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_12_26'] - df['macd_signal']
        
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume_trend'] = ((df['close'] - df['close'].shift()) / 
                                    df['close'].shift() * df['volume']).cumsum()
        
        df['atr'] = TechnicalIndicatorEngine._calculate_atr(df)
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(50).mean()
        
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['price_impact'] = abs(df['returns']) / (df['volume'] / df['volume_sma'])
        
        df['sector_momentum'] = df['close'].rolling(10).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )
        
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
        from hmmlearn import hmm
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
            regime_probs[i, :] = regime_probs[i, :] / regime_probs[i, :].sum()
        
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

class BacktestConfig:
    """Configuration for backtesting"""
    def __init__(self):
        self.initial_capital = 10000
        self.max_position_size = 0.05  # 5% max per trade
        self.transaction_cost = 0.001   # 0.1% transaction cost
        self.slippage = 0.0005         # 0.05% slippage
        self.confidence_threshold = 0.4
        self.volume_threshold = 0.6
        self.volatility_percentile = 0.98

class LSTMTrendPredictor(nn.Module):
    """LSTM model for loading saved weights"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        
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

class TradingModelLoader:
    """Loads and manages trained trading models"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.config = None
        self.hmm_detector = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models from pickle file"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading models from: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.models = saved_data['models']
        self.scalers = saved_data['scalers']
        self.feature_columns = saved_data['feature_columns']
        self.config = saved_data['config']
        self.hmm_detector = saved_data.get('hmm_detector')
        
        print(f"Loaded models: {list(self.models.keys())}")
        print(f"Feature columns: {len(self.feature_columns)}")
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using ensemble of loaded models"""
        
        # Prepare features (same as training)
        feature_data = data[self.feature_columns].fillna(0)
        X_scaled = self.scalers['main'].transform(feature_data)
        
        predictions = {}
        
        # Random Forest predictions
        if 'random_forest' in self.models:
            rf_probs = self.models['random_forest'].predict_proba(X_scaled)
            predictions['rf'] = rf_probs
        
        # XGBoost predictions
        if 'xgboost' in self.models:
            xgb_probs = self.models['xgboost'].predict_proba(X_scaled)
            predictions['xgb'] = xgb_probs
        
        # LSTM predictions (if available)
        if 'lstm' in self.models and isinstance(self.models['lstm'], dict):
            try:
                lstm_probs = self._predict_lstm(X_scaled)
                if lstm_probs is not None:
                    predictions['lstm'] = lstm_probs
            except Exception as e:
                print(f"LSTM prediction failed: {e}")
        
        # Ensemble predictions
        if predictions:
            ensemble_probs = np.zeros((len(X_scaled), 3))
            
            for model_name, probs in predictions.items():
                ensemble_probs += probs
            
            ensemble_probs /= len(predictions)
            
            signal_class = np.argmax(ensemble_probs, axis=1)
            signal_confidence = np.max(ensemble_probs, axis=1)
            
            result_df = data.copy()
            result_df['signal_class'] = signal_class
            result_df['signal_confidence'] = signal_confidence
            result_df['prob_sell'] = ensemble_probs[:, 0]
            result_df['prob_hold'] = ensemble_probs[:, 1]
            result_df['prob_buy'] = ensemble_probs[:, 2]
            
            return result_df
        else:
            result_df = data.copy()
            result_df['signal_class'] = 1
            result_df['signal_confidence'] = 0.5
            return result_df
    
    def _predict_lstm(self, X_scaled: np.ndarray) -> Optional[np.ndarray]:
        """Generate LSTM predictions"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Reconstruct LSTM model
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

class AdvancedBacktester:
    """Comprehensive backtesting engine"""
    
    def __init__(self, model_loader: TradingModelLoader, config: BacktestConfig = None):
        self.model_loader = model_loader
        self.config = config or BacktestConfig()
        self.trading_config = model_loader.config
        
    def run_backtest(self, data: pd.DataFrame, start_date: str = None, end_date: str = None) -> Dict:
        """Run comprehensive backtest"""
        
        # Filter data by date range if specified
        if start_date or end_date:
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
        
        print(f"Running backtest on {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        
        # Generate signals
        print("Generating trading signals...")
        signal_data = self.model_loader.predict(data)
        
        # Debug signal distribution
        signal_dist = signal_data['signal_class'].value_counts().to_dict()
        confidence_mean = signal_data['signal_confidence'].mean()
        print(f"Signal distribution: {signal_dist}")
        print(f"Average signal confidence: {confidence_mean:.3f}")
        
        # Apply filters
        filtered_signals = self._apply_risk_filters(signal_data)
        
        # Debug filtered signals
        filtered_dist = filtered_signals['signal_class'].value_counts().to_dict()
        print(f"Filtered signal distribution: {filtered_dist}")
        
        # Execute trades
        trades, portfolio_history = self._execute_trades(filtered_signals)
        
        print(f"Executed {len(trades)} trades")
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, portfolio_history, data)
        
        # Generate detailed results
        results = {
            'trades': trades,
            'portfolio_history': portfolio_history,
            'metrics': metrics,
            'signal_data': signal_data,
            'filtered_signals': filtered_signals,
            'backtest_period': {
                'start': data.index[0],
                'end': data.index[-1],
                'days': (data.index[-1] - data.index[0]).days
            }
        }
        
        return results
    
    def _apply_risk_filters(self, signal_data: pd.DataFrame) -> pd.DataFrame:
        """Apply risk management filters"""
        filtered_df = signal_data.copy()
        
        # Confidence filter
        confidence_mask = filtered_df['signal_confidence'] >= self.config.confidence_threshold
        filtered_df.loc[~confidence_mask, 'signal_class'] = 1
        
        # Volume filter
        volume_mask = filtered_df['volume_ratio'] >= self.config.volume_threshold
        filtered_df.loc[~volume_mask, 'signal_class'] = 1
        
        # Volatility filter
        vol_threshold = filtered_df['realized_vol'].quantile(self.config.volatility_percentile)
        vol_mask = filtered_df['realized_vol'] <= vol_threshold
        filtered_df.loc[~vol_mask, 'signal_class'] = 1
        
        return filtered_df
    
    def _execute_trades(self, signal_data: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Execute trades based on signals"""
        trades = []
        portfolio_history = []
        
        portfolio_value = self.config.initial_capital
        cash = self.config.initial_capital
        position = None
        entry_info = None
        
        for i, (timestamp, row) in enumerate(signal_data.iterrows()):
            signal = row['signal_class']
            confidence = row['signal_confidence']
            price = row['close']
            
            # Portfolio tracking
            current_value = cash
            if position:
                current_value += position['shares'] * price
            
            portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': current_value,
                'cash': cash,
                'position_value': position['shares'] * price if position else 0,
                'price': price
            })
            
            # Entry logic
            if position is None and signal == 2:  # Buy signal
                position_size = min(self.config.max_position_size, confidence * 0.1)
                trade_value = current_value * position_size
                
                # Account for transaction costs and slippage
                effective_price = price * (1 + self.config.slippage + self.config.transaction_cost)
                shares = trade_value / effective_price
                
                if trade_value <= cash:
                    position = {
                        'shares': shares,
                        'entry_price': effective_price,
                        'entry_time': timestamp,
                        'entry_confidence': confidence
                    }
                    cash -= trade_value
                    entry_info = position.copy()
            
            # Exit logic
            elif position is not None:
                pnl_pct = (price / position['entry_price']) - 1
                
                should_exit = (
                    signal == 0 or  # Sell signal
                    pnl_pct >= self.trading_config.profit_target or  # Take profit
                    pnl_pct <= -self.trading_config.stop_loss  # Stop loss
                )
                
                if should_exit:
                    # Account for transaction costs and slippage
                    effective_price = price * (1 - self.config.slippage - self.config.transaction_cost)
                    trade_value = position['shares'] * effective_price
                    
                    # Record trade
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': position['entry_price'],
                        'exit_price': effective_price,
                        'shares': position['shares'],
                        'entry_confidence': position['entry_confidence'],
                        'exit_signal': signal,
                        'gross_pnl': (effective_price - position['entry_price']) * position['shares'],
                        'gross_pnl_pct': (effective_price / position['entry_price']) - 1,
                        'hold_periods': i - signal_data.index.get_loc(position['entry_time']),
                        'exit_reason': 'signal' if signal == 0 else 
                                      'profit' if pnl_pct >= self.trading_config.profit_target else 'stop'
                    }
                    trades.append(trade)
                    
                    cash += trade_value
                    position = None
        
        # Close any remaining position
        if position:
            final_price = signal_data.iloc[-1]['close']
            effective_price = final_price * (1 - self.config.slippage - self.config.transaction_cost)
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': signal_data.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': effective_price,
                'shares': position['shares'],
                'entry_confidence': position['entry_confidence'],
                'exit_signal': 1,
                'gross_pnl': (effective_price - position['entry_price']) * position['shares'],
                'gross_pnl_pct': (effective_price / position['entry_price']) - 1,
                'hold_periods': len(signal_data) - signal_data.index.get_loc(position['entry_time']),
                'exit_reason': 'forced_close'
            }
            trades.append(trade)
        
        return trades, portfolio_history
    
    def _calculate_metrics(self, trades: List[Dict], portfolio_history: List[Dict], 
                          data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return {
                'total_return': 0,
                'annual_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'avg_trade_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_hold_time': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        # Portfolio performance
        portfolio_df = pd.DataFrame(portfolio_history)
        initial_value = portfolio_df.iloc[0]['portfolio_value']
        final_value = portfolio_df.iloc[-1]['portfolio_value']
        
        total_return = (final_value / initial_value) - 1
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change().fillna(0)
        
        # Time-based metrics
        total_days = (data.index[-1] - data.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(total_days, 1)) - 1
        
        # Trade analysis
        trade_returns = [t['gross_pnl_pct'] for t in trades]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        # Performance metrics
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else float('inf')
        
        # Risk metrics
        daily_returns = portfolio_df['daily_return'].dropna()
        sharpe_ratio = (daily_returns.mean() / (daily_returns.std() + 1e-8)) * np.sqrt(252)
        
        # Drawdown calculation
        portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding(min_periods=1).max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min()
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade timing
        avg_hold_time = np.mean([t['hold_periods'] for t in trades])
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'avg_trade_return': np.mean(trade_returns),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_time': avg_hold_time,
            'best_trade': max(trade_returns) if trade_returns else 0,
            'worst_trade': min(trade_returns) if trade_returns else 0,
            'volatility': daily_returns.std() * np.sqrt(252),
            'final_portfolio_value': final_value
        }

class BacktestAnalyzer:
    """Generate detailed analysis and visualizations"""
    
    def __init__(self, results: Dict):
        self.results = results
        self.trades = results['trades']
        self.portfolio_history = pd.DataFrame(results['portfolio_history'])
        self.metrics = results['metrics']
    
    def print_summary(self):
        """Print comprehensive backtest summary"""
        print(f"\n{'='*80}")
        print(f"BACKTEST RESULTS SUMMARY")
        print(f"{'='*80}")
        
        period = self.results['backtest_period']
        print(f"Period: {period['start'].strftime('%Y-%m-%d')} to {period['end'].strftime('%Y-%m-%d')} ({period['days']} days)")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Total Return:        {self.metrics['total_return']:8.1%}")
        print(f"  Annualized Return:   {self.metrics['annual_return']:8.1%}")
        print(f"  Sharpe Ratio:        {self.metrics['sharpe_ratio']:8.2f}")
        print(f"  Calmar Ratio:        {self.metrics['calmar_ratio']:8.2f}")
        print(f"  Max Drawdown:        {self.metrics['max_drawdown']:8.1%}")
        print(f"  Volatility:          {self.metrics['volatility']:8.1%}")
        
        print(f"\nTRADE STATISTICS:")
        print(f"  Total Trades:        {self.metrics['total_trades']:8d}")
        print(f"  Winning Trades:      {self.metrics['winning_trades']:8d}")
        print(f"  Losing Trades:       {self.metrics['losing_trades']:8d}")
        print(f"  Win Rate:            {self.metrics['win_rate']:8.1f}%")
        print(f"  Profit Factor:       {self.metrics['profit_factor']:8.2f}")
        print(f"  Average Trade:       {self.metrics['avg_trade_return']:8.1%}")
        print(f"  Average Win:         {self.metrics['avg_win']:8.1%}")
        print(f"  Average Loss:        {self.metrics['avg_loss']:8.1%}")
        print(f"  Best Trade:          {self.metrics['best_trade']:8.1%}")
        print(f"  Worst Trade:         {self.metrics['worst_trade']:8.1%}")
        print(f"  Avg Hold Time:       {self.metrics['avg_hold_time']:8.1f} periods")
        
        if self.trades:
            print(f"\nRECENT TRADES (Last 10):")
            print(f"{'Entry Time':<19} {'Exit Time':<19} {'P&L%':<8} {'Hold':<6} {'Reason':<8}")
            print(f"{'-'*19} {'-'*19} {'-'*8} {'-'*6} {'-'*8}")
            
            for trade in self.trades[-10:]:
                entry_str = trade['entry_time'].strftime('%Y-%m-%d %H:%M')
                exit_str = trade['exit_time'].strftime('%Y-%m-%d %H:%M')
                pnl_str = f"{trade['gross_pnl_pct']:.1%}"
                hold_str = f"{trade['hold_periods']:d}"
                reason_str = trade['exit_reason'][:8]
                
                print(f"{entry_str:<19} {exit_str:<19} {pnl_str:<8} {hold_str:<6} {reason_str:<8}")
    
    def export_trades(self, filepath: str):
        """Export trades to CSV"""
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(filepath, index=False)
            print(f"Trades exported to: {filepath}")
    
    def export_summary(self, filepath: str):
        """Export summary to JSON"""
        summary = {
            'backtest_period': {
                'start': self.results['backtest_period']['start'].isoformat(),
                'end': self.results['backtest_period']['end'].isoformat(),
                'days': self.results['backtest_period']['days']
            },
            'metrics': self.metrics,
            'trade_count': len(self.trades)
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary exported to: {filepath}")

def main():
    """Main backtesting function"""
    parser = argparse.ArgumentParser(description='Advanced Trading System Backtester')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., META)')
    parser.add_argument('--model-path', help='Path to trained model file')
    parser.add_argument('--days', type=int, default=60, help='Days of data for backtesting')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--export-trades', help='Path to export trades CSV')
    parser.add_argument('--export-summary', help='Path to export summary JSON')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = project_root / "models" / args.symbol / "advanced_system" / "ensemble_models.pkl"
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first or specify the correct path with --model-path")
        return
    
    # Load data
    data_manager = DataManager()
    print(f"Loading data for {args.symbol}...")
    
    data = data_manager.get_training_data(args.symbol, args.days)
    if data.empty:
        print(f"No data available for {args.symbol}")
        return
    
    print(f"Loaded {len(data)} 1-minute bars")
    
    # Load trained models
    try:
        model_loader = TradingModelLoader(model_path)
        
        # Resample to training window
        window_min = model_loader.config.window_minutes
        if window_min > 1:
            rule = f"{window_min}T"
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            data = data.resample(rule).agg(agg).dropna()
        
        print(f"Resampled to {len(data)} {window_min}-minute bars")
        
        # Apply technical analysis
        ta_engine = TechnicalIndicatorEngine()
        data = ta_engine.calculate_comprehensive_indicators(data)
        
        # Add regime detection
        if model_loader.hmm_detector:
            try:
                regimes, regime_probs, _ = model_loader.hmm_detector.fit_predict(data)
                data['regime'] = regimes
                for i in range(regime_probs.shape[1]):
                    data[f'regime_prob_{i}'] = regime_probs[:, i]
            except Exception as e:
                print(f"HMM regime detection failed: {e}, using fallback")
                # Simple fallback
                data['regime'] = 1
                for i in range(3):
                    data[f'regime_prob_{i}'] = 0.33
        else:
            # No HMM detector in saved model
            data['regime'] = 1
            for i in range(3):
                data[f'regime_prob_{i}'] = 0.33
        
        # Add alternative data
        alt_engine = AlternativeDataEngine()
        data['sentiment_score'] = [alt_engine.get_sentiment_score(args.symbol, ts) for ts in data.index]
        data['economic_regime'] = [alt_engine.get_economic_regime(ts) for ts in data.index]
        
        data = data.ffill().bfill().fillna(0)
        print(f"Preprocessed data: {len(data)} bars with {len(data.columns)} features")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run backtest
    backtester = AdvancedBacktester(model_loader)
    
    print(f"\nRunning backtest...")
    results = backtester.run_backtest(data, args.start_date, args.end_date)
    
    # Analyze results
    analyzer = BacktestAnalyzer(results)
    analyzer.print_summary()
    
    # Generate outputs
    if args.export_trades:
        analyzer.export_trades(args.export_trades)
    
    if args.export_summary:
        analyzer.export_summary(args.export_summary)
    
    # Performance assessment
    metrics = results['metrics']
    if metrics['total_return'] > 0.2:  # 20%+ return
        print(f"\n🎯 EXCELLENT: {metrics['total_return']:.1%} return with {metrics['sharpe_ratio']:.2f} Sharpe!")
    elif metrics['total_return'] > 0.1:  # 10%+ return
        print(f"\n✅ GOOD: {metrics['total_return']:.1%} return achieved")
    elif metrics['total_return'] > 0:
        print(f"\n📊 POSITIVE: {metrics['total_return']:.1%} return")
    else:
        print(f"\n⚠️ LOSS: {metrics['total_return']:.1%} return - strategy needs improvement")

if __name__ == "__main__":
    main()

