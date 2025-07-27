#!/usr/bin/env python3

"""
Enhanced Trader Agent - FIXED VERSION with proper signal generation
"""

import os
import sys
import logging
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import importlib.util

# Load environment variables from .env file one level up
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load .env file from root level (one level up)
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

from shared.data_manager import DataManager
from shared.data_publisher import DataPublisher

# Import enhanced components from training directory
training_dir = project_root / "training"
sys.path.append(str(training_dir))

try:
    from xetrader_trainer import (
        EnhancedAdvancedTradingSystem,
        TradingConfig,
        load_optimized_config_for_symbol,
        EnhancedDataManager,
        EnhancedTechnicalIndicatorEngine,
        HMMRegimeDetector,
        AlternativeDataEngine
    )
except ImportError as e:
    print(f"Warning: Could not import enhanced components: {e}")
    print("Falling back to basic trading model")

class EnhancedTraderAgent:
    """
    Enhanced trading agent - FIXED VERSION with proper signal generation
    """
    
    def __init__(self, symbol: str, data_manager: DataManager = None, data_publisher: DataPublisher = None, 
                 simulator_mode: bool = False, debug_mode: bool = False):
        self.symbol = symbol.upper()
        self.simulator_mode = simulator_mode
        self.debug_mode = debug_mode
        
        # Initialize data components
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.data_manager = data_manager or DataManager(host=redis_host, port=redis_port)
        self.enhanced_data_manager = EnhancedDataManager()
        self.data_publisher = data_publisher or DataPublisher(host=redis_host, port=redis_port)
        
        self.logger = self._setup_logging()
        
        # Load optimized configuration
        self.config = self._load_optimized_config()
        
        # Initialize enhanced components
        self.ta_engine = EnhancedTechnicalIndicatorEngine()
        self.hmm_detector = HMMRegimeDetector()
        self.alt_data_engine = AlternativeDataEngine()
        
        # Load enhanced models
        self.enhanced_system = None
        self.ensemble_generator = None
        self._load_enhanced_models()
        
        # Trading state
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None
        self.current_regime = 1  # Default to neutral regime
        
        # Simulator state
        self.simulator_portfolio_value = 10000.0
        self.simulator_trades = []
        self.simulator_positions = []
        
        # Debug counters  
        self.debug_signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        self.debug_confidence_history = []
        self.debug_error_counts = {'signal_generation': 0, 'insufficient_data': 0, 'model_error': 0}
        
        self.logger.info(f"Enhanced Trader Agent initialized for {self.symbol}")
        self.logger.info(f"Simulator mode: {self.simulator_mode}")
        self.logger.info(f"Debug mode: {self.debug_mode}")
        self.logger.info(f"Using optimized config: {self.config is not None}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the trader agent."""
        logger = logging.getLogger(f"EnhancedTraderAgent_{self.symbol}")
        level = logging.DEBUG if self.debug_mode else getattr(logging, os.getenv('LOG_LEVEL', 'INFO'))
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _load_optimized_config(self) -> Optional[TradingConfig]:
        """Load optimized configuration for the symbol."""
        try:
            config = load_optimized_config_for_symbol(self.symbol)
            if config:
                self.logger.info(f"✅ Loaded optimized config for {self.symbol}")
                self.logger.info(f"   Confidence threshold: {config.confidence_threshold:.4f}")
                self.logger.info(f"   Position sizing: {config.base_position_size:.1%} - {config.max_position_size:.1%}")
                self.logger.info(f"   Label thresholds: UP={config.label_threshold_up:.4f}, DOWN={config.label_threshold_down:.4f}")
                return config
            else:
                self.logger.warning(f"No optimized config found for {self.symbol}, using default")
                return TradingConfig()
        except Exception as e:
            self.logger.error(f"Error loading optimized config: {e}")
            return TradingConfig()

    def _load_enhanced_models(self):
        """Load enhanced ensemble models from the training system."""
        try:
            # Path to enhanced models
            model_path = project_root / "models" / self.symbol / "enhanced_system" / "enhanced_ensemble_models.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Create enhanced system instance
                self.enhanced_system = EnhancedAdvancedTradingSystem(self.symbol, self.config)
                
                # Load the trained components
                self.enhanced_system.ensemble_generator.models = model_data['models']
                self.enhanced_system.ensemble_generator.scalers = model_data['scalers']
                self.enhanced_system.ensemble_generator.feature_columns = model_data['feature_columns']
                self.enhanced_system.hmm_detector = model_data['hmm_detector']
                
                self.ensemble_generator = self.enhanced_system.ensemble_generator
                self.hmm_detector = model_data['hmm_detector']
                
                self.logger.info(f"✅ Enhanced models loaded successfully")
                self.logger.info(f"   Models available: {list(model_data['models'].keys())}")
                self.logger.info(f"   Features: {len(model_data['feature_columns'])}")
                
                return True
            else:
                self.logger.warning(f"Enhanced models not found at {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading enhanced models: {e}")
            return False

    def prepare_enhanced_dataset_for_realtime(self, raw_data: pd.DataFrame, min_required_bars: int = 100) -> pd.DataFrame:
        """
        Prepare enhanced dataset for real-time trading with proper data handling.
        
        Key fix: Don't resample for real-time trading if it reduces data too much.
        Use 1-minute data directly and ensure we have enough historical context.
        """
        if len(raw_data) < min_required_bars:
            self.logger.warning(f"Insufficient raw data: {len(raw_data)} < {min_required_bars}")
            self.debug_error_counts['insufficient_data'] += 1
            return pd.DataFrame()
        
        try:
            # For real-time trading, use 1-minute data directly instead of resampling
            # The models can handle 1-minute data even if trained on resampled data
            data = raw_data.copy()
            
            if self.debug_mode:
                self.logger.debug(f"Using {len(data)} 1-minute bars for signal generation")
            
            # Calculate enhanced technical indicators
            data = self.ta_engine.calculate_comprehensive_indicators(data)
            
            # Detect market regimes if we have enough data
            if len(data) >= 50:
                try:
                    regimes, regime_probs, regime_stats = self.hmm_detector.fit_predict(data)
                    data['regime'] = regimes
                    for i in range(regime_probs.shape[1]):
                        data[f'regime_prob_{i}'] = regime_probs[:, i]
                    
                    # Update current regime
                    self.current_regime = regimes[-1] if len(regimes) > 0 else 1
                    if self.debug_mode:
                        self.logger.debug(f"Current regime: {self.current_regime}")
                except Exception as e:
                    self.logger.warning(f"Regime detection failed: {e}")
                    data['regime'] = 1
                    for i in range(3):
                        data[f'regime_prob_{i}'] = 0.33
                    self.current_regime = 1
            
            # Add alternative data
            data['sentiment_score'] = [self.alt_data_engine.get_sentiment_score(self.symbol, ts) for ts in data.index]
            data['economic_regime'] = [self.alt_data_engine.get_economic_regime(ts) for ts in data.index]
            
            # Clean the data
            data = data.ffill().bfill().fillna(0)
            
            if self.debug_mode:
                self.logger.debug(f"Enhanced dataset prepared: {len(data)} bars, {len(data.columns)} features")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error preparing enhanced dataset: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.debug_error_counts['model_error'] += 1
            return pd.DataFrame()

    def generate_signals_with_fallback(self, enhanced_data: pd.DataFrame) -> Dict:
        """
        Generate signals with proper error handling and fallbacks.
        """
        if enhanced_data.empty or len(enhanced_data) < 50:
            return {
                'signal_class': 1,  # Hold
                'signal_confidence': 0.0,
                'prob_buy': 0.33,
                'prob_sell': 0.33,
                'prob_hold': 0.34,
                'regime': 1,
                'error': 'Insufficient data'
            }
        
        try:
            # Generate signals using ensemble
            signal_data = self.ensemble_generator.generate_signals(enhanced_data)
            
            if signal_data.empty:
                self.debug_error_counts['signal_generation'] += 1
                if self.debug_mode:
                    self.logger.debug("Ensemble returned empty signal data")
                return self._fallback_signal()
            
            # Get the latest signal
            latest_signal = signal_data.iloc[-1]
            
            result = {
                'signal_class': int(latest_signal.get('signal_class', 1)),
                'signal_confidence': float(latest_signal.get('signal_confidence', 0.0)),
                'prob_buy': float(latest_signal.get('prob_buy', 0.33)),
                'prob_sell': float(latest_signal.get('prob_sell', 0.33)),
                'prob_hold': float(latest_signal.get('prob_hold', 0.34)),
                'regime': int(latest_signal.get('regime', 1)),
                'volume_ratio': float(latest_signal.get('volume_ratio', 1.0)),
                'realized_vol': float(latest_signal.get('realized_vol', 0.2))
            }
            
            if self.debug_mode:
                self.logger.debug(f"Generated signal: class={result['signal_class']}, conf={result['signal_confidence']:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in signal generation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.debug_error_counts['signal_generation'] += 1
            return self._fallback_signal()

    def _fallback_signal(self) -> Dict:
        """Fallback signal when ensemble fails."""
        return {
            'signal_class': 1,  # Hold
            'signal_confidence': 0.0,
            'prob_buy': 0.33,
            'prob_sell': 0.33,
            'prob_hold': 0.34,
            'regime': 1,
            'volume_ratio': 1.0,
            'realized_vol': 0.2,
            'fallback': True
        }

    def make_enhanced_decision(self, current_price: float, market_data: pd.DataFrame) -> Dict:
        """Make trading decision using enhanced ensemble models with proper error handling."""
        if self.ensemble_generator is None:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'Enhanced models not available'}
        
        try:
            # Prepare enhanced dataset for real-time use
            enhanced_data = self.prepare_enhanced_dataset_for_realtime(market_data, min_required_bars=100)
            
            if enhanced_data.empty:
                self.debug_signal_counts['hold'] += 1
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Insufficient enhanced data'}
            
            # Generate signals with fallback handling
            signal_result = self.generate_signals_with_fallback(enhanced_data)
            
            signal_class = signal_result['signal_class']
            confidence = signal_result['signal_confidence']
            regime = signal_result['regime']
            
            # Track confidence for debugging
            self.debug_confidence_history.append(confidence)
            if len(self.debug_confidence_history) > 100:
                self.debug_confidence_history = self.debug_confidence_history[-100:]
            
            # Debug signal information
            if self.debug_mode:
                self.logger.debug(f"Signal class: {signal_class}, Confidence: {confidence:.4f}")
                self.logger.debug(f"Buy prob: {signal_result['prob_buy']:.4f}, Sell prob: {signal_result['prob_sell']:.4f}")
                if 'fallback' in signal_result:
                    self.logger.debug("Using fallback signal due to error")
            
            # Apply enhanced risk filters with more permissive thresholds
            base_threshold = self.config.confidence_threshold
            
            # Make thresholds much more permissive for real-time trading
            if regime == 2:  # Trending market
                effective_threshold = base_threshold * 0.5  # Very permissive
            elif regime == 0:  # Volatile market
                effective_threshold = base_threshold * 0.8  # Moderately permissive
            else:  # Normal market
                effective_threshold = base_threshold * 0.6  # Quite permissive
            
            # Additional risk filters - more lenient
            volume_ratio = signal_result['volume_ratio']
            volatility = signal_result['realized_vol']
            
            volume_ok = volume_ratio >= 0.5  # Very permissive
            volatility_ok = volatility <= 1.0  # Very permissive
            
            if self.debug_mode:
                self.logger.debug(f"Effective threshold: {effective_threshold:.4f}, Volume ratio: {volume_ratio:.2f}, Volatility: {volatility:.4f}")
                self.logger.debug(f"Volume OK: {volume_ok}, Volatility OK: {volatility_ok}")
            
            # Determine action
            if self.position is None:  # Looking for entry
                if signal_class == 2 and confidence >= effective_threshold and volume_ok and volatility_ok:
                    # Calculate position size
                    position_size = self._calculate_enhanced_position_size(confidence, volatility, regime)
                    
                    self.debug_signal_counts['buy'] += 1
                    
                    return {
                        'action': 'buy',
                        'confidence': confidence,
                        'reason': f'Enhanced buy signal (regime: {regime}, conf: {confidence:.3f}, thresh: {effective_threshold:.3f})',
                        'price': current_price,
                        'position_size': position_size,
                        'regime': regime,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    self.debug_signal_counts['hold'] += 1
                    reason_parts = []
                    if signal_class != 2:
                        reason_parts.append(f"signal={signal_class}")
                    if confidence < effective_threshold:
                        reason_parts.append(f"conf={confidence:.3f}<{effective_threshold:.3f}")
                    if not volume_ok:
                        reason_parts.append(f"vol={volume_ratio:.2f}")
                    if not volatility_ok:
                        reason_parts.append(f"volatility_high={volatility:.3f}")
                    
                    return {
                        'action': 'hold',
                        'confidence': confidence,
                        'reason': f"No buy: {', '.join(reason_parts)}",
                        'timestamp': datetime.now().isoformat()
                    }
            
            else:  # In position, looking for exit
                # Check stop loss and take profit first
                if self.entry_price is not None:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                    
                    if pnl_pct <= -self.config.stop_loss:
                        return {
                            'action': 'sell',
                            'confidence': 1.0,
                            'reason': f'Stop loss: {pnl_pct:.3%}',
                            'price': current_price,
                            'pnl_pct': pnl_pct,
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    if pnl_pct >= self.config.profit_target:
                        return {
                            'action': 'sell',
                            'confidence': 1.0,
                            'reason': f'Take profit: {pnl_pct:.3%}',
                            'price': current_price,
                            'pnl_pct': pnl_pct,
                            'timestamp': datetime.now().isoformat()
                        }
                
                # Check for signal-based exit - very permissive
                if signal_class == 0 and confidence >= effective_threshold * 0.5:  # Very low threshold for exits
                    pnl_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price else 0
                    self.debug_signal_counts['sell'] += 1
                    return {
                        'action': 'sell',
                        'confidence': confidence,
                        'reason': f'Enhanced sell signal (regime: {regime}, conf: {confidence:.3f})',
                        'price': current_price,
                        'pnl_pct': pnl_pct,
                        'regime': regime,
                        'timestamp': datetime.now().isoformat()
                    }
                
                return {
                    'action': 'hold',
                    'confidence': confidence,
                    'reason': f'Holding position (signal: {signal_class}, conf: {confidence:.3f})',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error making enhanced decision: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.debug_error_counts['model_error'] += 1
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}

    def _calculate_enhanced_position_size(self, confidence: float, volatility: float, regime: int) -> float:
        """Calculate position size using enhanced risk management."""
        base_size = self.config.base_position_size
        
        # Confidence multiplier
        confidence_multiplier = min(confidence * 1.5, 1.8)
        
        # Volatility adjustment
        vol_adjustment = 1 / max(volatility * 0.3, 0.03)
        vol_adjustment = min(vol_adjustment, 2.0)
        
        # Regime adjustment
        if regime == 2:  # Trending
            regime_multiplier = 1.2
        elif regime == 0:  # Volatile
            regime_multiplier = 0.8
        else:  # Normal
            regime_multiplier = 1.0
        
        position_size = base_size * confidence_multiplier * vol_adjustment * regime_multiplier
        return min(position_size, self.config.max_position_size)

    def execute_trade(self, decision: Dict):
        """Execute trading decision and update state."""
        action = decision['action']
        price = decision.get('price')
        timestamp = decision.get('timestamp', datetime.now().isoformat())
        
        if action == 'buy' and self.position is None:
            # Execute buy
            self.position = 'long'
            self.entry_price = price
            self.entry_time = timestamp
            position_size = decision.get('position_size', self.config.base_position_size)
            
            trade_signal = {
                'symbol': self.symbol,
                'action': 'buy',
                'price': price,
                'position_size': position_size,
                'confidence': decision['confidence'],
                'reason': decision['reason'],
                'regime': decision.get('regime', 1),
                'timestamp': timestamp,
                'agent_id': f"enhanced_trader_agent_{self.symbol}",
                'simulator_mode': self.simulator_mode
            }
            
            if self.simulator_mode:
                self._execute_simulator_buy(trade_signal)
            else:
                self.data_publisher.publish_trade_signal(self.symbol, trade_signal)
            
            self.logger.info(f"🟢 BUY: {self.symbol} at ${price:.2f} | Size: {position_size:.1%} | Conf: {decision['confidence']:.3f}")
            
        elif action == 'sell' and self.position == 'long':
            # Execute sell
            pnl_pct = decision.get('pnl_pct', (price - self.entry_price) / self.entry_price if self.entry_price else 0)
            
            trade_signal = {
                'symbol': self.symbol,
                'action': 'sell',
                'price': price,
                'entry_price': self.entry_price,
                'pnl_pct': pnl_pct,
                'confidence': decision['confidence'],
                'reason': decision['reason'],
                'regime': decision.get('regime', 1),
                'timestamp': timestamp,
                'hold_duration': self._calculate_hold_duration(),
                'agent_id': f"enhanced_trader_agent_{self.symbol}",
                'simulator_mode': self.simulator_mode
            }
            
            if self.simulator_mode:
                self._execute_simulator_sell(trade_signal)
            else:
                self.data_publisher.publish_trade_signal(self.symbol, trade_signal)
            
            self.logger.info(f"🔴 SELL: {self.symbol} at ${price:.2f} | PnL: {pnl_pct:.3%} | Conf: {decision['confidence']:.3f}")
            
            # Reset position
            self.position = None
            self.entry_price = None
            self.entry_time = None

    def _execute_simulator_buy(self, trade_signal: Dict):
        """Execute buy in simulator mode."""
        self.simulator_positions.append(trade_signal)

    def _execute_simulator_sell(self, trade_signal: Dict):
        """Execute sell in simulator mode and update portfolio."""
        if self.simulator_positions:
            buy_signal = self.simulator_positions[-1]
            position_size = buy_signal.get('position_size', self.config.base_position_size)
            pnl_pct = trade_signal['pnl_pct']
            
            # Update portfolio value
            trade_pnl = self.simulator_portfolio_value * position_size * pnl_pct
            self.simulator_portfolio_value += trade_pnl
            
            # Record trade
            trade_record = {
                'entry_time': buy_signal['timestamp'],
                'exit_time': trade_signal['timestamp'],
                'entry_price': buy_signal['price'],
                'exit_price': trade_signal['price'],
                'position_size': position_size,
                'pnl_pct': pnl_pct,
                'pnl_dollar': trade_pnl,
                'reason': trade_signal['reason'],
                'hold_duration': trade_signal.get('hold_duration', 0)
            }
            
            self.simulator_trades.append(trade_record)
            self.simulator_positions.append(trade_signal)

    def _calculate_hold_duration(self) -> Optional[float]:
        """Calculate position hold duration in hours."""
        if self.entry_time:
            try:
                entry_dt = datetime.fromisoformat(self.entry_time.replace('Z', '+00:00'))
                current_dt = datetime.now()
                duration = (current_dt - entry_dt).total_seconds() / 3600
                return duration
            except Exception as e:
                self.logger.error(f"Error calculating hold duration: {e}")
        return None

    def get_simulator_performance(self) -> Dict:
        """Get simulator performance metrics."""
        if not self.simulator_trades:
            return {
                'total_trades': 0,
                'portfolio_value': self.simulator_portfolio_value,
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0
            }
        
        trades_df = pd.DataFrame(self.simulator_trades)
        winning_trades = sum(1 for t in self.simulator_trades if t['pnl_pct'] > 0)
        total_return = (self.simulator_portfolio_value / 10000) - 1
        
        return {
            'total_trades': len(self.simulator_trades),
            'portfolio_value': self.simulator_portfolio_value,
            'total_return': total_return,
            'win_rate': winning_trades / len(self.simulator_trades) * 100,
            'avg_return': trades_df['pnl_pct'].mean(),
            'best_trade': trades_df['pnl_pct'].max(),
            'worst_trade': trades_df['pnl_pct'].min(),
            'avg_hold_duration': trades_df['hold_duration'].mean()
        }

    def get_debug_stats(self) -> Dict:
        """Get debugging statistics."""
        avg_confidence = np.mean(self.debug_confidence_history) if self.debug_confidence_history else 0
        max_confidence = np.max(self.debug_confidence_history) if self.debug_confidence_history else 0
        
        return {
            'signal_counts': self.debug_signal_counts.copy(),
            'error_counts': self.debug_error_counts.copy(),
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'confidence_samples': len(self.debug_confidence_history),
            'effective_threshold': self.config.confidence_threshold * 0.6  # Current default
        }

    def run_never_seen_data_test(self, start_date: str, end_date: str = None):
        """Run test on never-seen data with improved error handling."""
        self.debug_mode = True
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info(f"🧪 Running never-seen data test from {start_date}")
        self.logger.info(f"Training cutoff was around 2025-07-23, testing on newer data")
        
        # Reset simulator state
        self.simulator_portfolio_value = 10000.0
        self.simulator_trades = []
        self.simulator_positions = []
        self.debug_signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        self.debug_confidence_history = []
        self.debug_error_counts = {'signal_generation': 0, 'insufficient_data': 0, 'model_error': 0}
        
        try:
            # Get never-seen data
            test_data = self.enhanced_data_manager.get_backtest_data(
                self.symbol, 
                days=30,
                start_date=start_date,
                end_date=end_date
            )
            
            if test_data.empty:
                self.logger.error("No test data available")
                return
            
            self.logger.info(f"Testing on {len(test_data)} bars from {test_data.index[0]} to {test_data.index[-1]}")
            
            # Use a larger window for better signal generation
            window_size = 300  # Increased from 200
            
            for i in range(window_size, len(test_data)):
                # Use more historical data for context
                current_data = test_data.iloc[max(0, i-window_size):i+1]
                current_price = current_data['close'].iloc[-1]
                
                # Make decision
                decision = self.make_enhanced_decision(current_price, current_data)
                
                # Execute if needed
                if decision['action'] in ['buy', 'sell']:
                    self.execute_trade(decision)
                
                # Log progress every 500 bars with detailed debug info
                if i % 500 == 0:
                    perf = self.get_simulator_performance()
                    debug_stats = self.get_debug_stats()
                    self.logger.info(f"Progress: {i}/{len(test_data)} | Portfolio: ${perf['portfolio_value']:.2f} | Return: {perf['total_return']:.2%}")
                    self.logger.info(f"Debug: Signals {debug_stats['signal_counts']}, Errors: {debug_stats['error_counts']}")
                    self.logger.info(f"Confidence: Avg={debug_stats['avg_confidence']:.4f}, Max={debug_stats['max_confidence']:.4f}, Samples={debug_stats['confidence_samples']}")
                elif i % 100 == 0:
                    perf = self.get_simulator_performance()
                    self.logger.info(f"Progress: {i}/{len(test_data)} | Portfolio: ${perf['portfolio_value']:.2f} | Return: {perf['total_return']:.2%}")
            
            # Final performance report
            final_perf = self.get_simulator_performance()
            debug_stats = self.get_debug_stats()
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"NEVER-SEEN DATA TEST RESULTS FOR {self.symbol}")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Test Period: {test_data.index[0]} to {test_data.index[-1]}")
            self.logger.info(f"Total Bars Processed: {len(test_data)}")
            self.logger.info(f"Total Trades: {final_perf['total_trades']}")
            self.logger.info(f"Final Portfolio Value: ${final_perf['portfolio_value']:.2f}")
            self.logger.info(f"Total Return: {final_perf['total_return']:.2%}")
            self.logger.info(f"Win Rate: {final_perf['win_rate']:.1f}%")
            
            self.logger.info(f"\nDEBUG STATISTICS:")
            self.logger.info(f"Signal Counts: {debug_stats['signal_counts']}")
            self.logger.info(f"Error Counts: {debug_stats['error_counts']}")
            self.logger.info(f"Average Confidence: {debug_stats['avg_confidence']:.4f}")
            self.logger.info(f"Max Confidence: {debug_stats['max_confidence']:.4f}")
            self.logger.info(f"Effective Threshold: {debug_stats['effective_threshold']:.4f}")
            self.logger.info(f"Confidence Samples: {debug_stats['confidence_samples']}")
            
            if final_perf['total_trades'] > 0:
                self.logger.info(f"\nTRADE STATISTICS:")
                self.logger.info(f"Average Return per Trade: {final_perf['avg_return']:.3%}")
                self.logger.info(f"Best Trade: {final_perf['best_trade']:.3%}")
                self.logger.info(f"Worst Trade: {final_perf['worst_trade']:.3%}")
                self.logger.info(f"Average Hold Duration: {final_perf['avg_hold_duration']:.1f} hours")
            
            # Performance assessment
            if final_perf['total_return'] > 0.05:
                self.logger.info("🎯 EXCELLENT: Strong performance on never-seen data!")
            elif final_perf['total_return'] > 0.02:
                self.logger.info("✅ GOOD: Positive performance on never-seen data")
            elif final_perf['total_return'] > 0:
                self.logger.info("📈 POSITIVE: Modest gains on never-seen data")
            else:
                if final_perf['total_trades'] == 0:
                    self.logger.info("⚠️ NO TRADES: Signal generation issues detected")
                    if debug_stats['error_counts']['signal_generation'] > 0:
                        self.logger.info(f"   Signal generation errors: {debug_stats['error_counts']['signal_generation']}")
                    if debug_stats['error_counts']['insufficient_data'] > 0:
                        self.logger.info(f"   Insufficient data errors: {debug_stats['error_counts']['insufficient_data']}")
                else:
                    self.logger.info("⚠️ NEEDS WORK: Negative performance on never-seen data")
            
            return final_perf
            
        except Exception as e:
            self.logger.error(f"Error running never-seen data test: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def process_market_data(self, current_price: float, market_data: pd.DataFrame):
        """Process market data and make trading decisions."""
        try:
            decision = self.make_enhanced_decision(current_price, market_data)
            
            if decision['action'] in ['buy', 'sell']:
                self.execute_trade(decision)
            else:
                if self.debug_mode:
                    self.logger.debug(f"Decision: {decision['reason']}")
                
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")

    def get_position_status(self) -> Dict:
        """Get current position and system status."""
        return {
            'symbol': self.symbol,
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'current_regime': self.current_regime,
            'enhanced_models_loaded': self.ensemble_generator is not None,
            'optimized_config_loaded': self.config is not None,
            'simulator_mode': self.simulator_mode,
            'simulator_performance': self.get_simulator_performance() if self.simulator_mode else None,
            'debug_stats': self.get_debug_stats() if self.debug_mode else None
        }

def main():
    """Main function for running the enhanced trader agent."""
    parser = argparse.ArgumentParser(description='Enhanced Trader Agent with optimized configurations - FIXED VERSION')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., META)')
    parser.add_argument('--mode', choices=['live', 'simulator', 'test-never-seen'], 
                       default='live', help='Operating mode')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    parser.add_argument('--start-date', help='Start date for testing (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for testing (YYYY-MM-DD)')
    parser.add_argument('--host', default=None, help='Redis host (overrides .env)')
    parser.add_argument('--port', type=int, default=None, help='Redis port (overrides .env)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Redis connection
    redis_host = args.host or os.getenv('REDIS_HOST', 'localhost')
    redis_port = args.port or int(os.getenv('REDIS_PORT', '6379'))
    
    # Initialize components
    data_manager = DataManager(host=redis_host, port=redis_port)
    data_publisher = DataPublisher(host=redis_host, port=redis_port)
    
    # Create enhanced agent
    simulator_mode = args.mode in ['simulator', 'test-never-seen']
    agent = EnhancedTraderAgent(args.symbol, data_manager, data_publisher, simulator_mode, args.debug)
    
    print(f"🚀 Enhanced Trader Agent for {args.symbol} - FIXED VERSION")
    print(f"Mode: {args.mode}")
    print(f"Redis: {redis_host}:{redis_port}")
    print(f"Debug: {args.debug}")
    print(f"Status: {agent.get_position_status()}")
    
    if args.mode == 'test-never-seen':
        # Test on never-seen data
        start_date = args.start_date or "2025-07-24"
        results = agent.run_never_seen_data_test(start_date, args.end_date)
        return results
    
    # Start heartbeat for live/simulator modes
    if not simulator_mode:
        data_publisher.start_heartbeat(f"enhanced_trader_agent_{args.symbol}")
    
    try:
        while True:
            # Get recent market data
            recent_data = data_manager.get_market_data(args.symbol, '1min')
            
            if not recent_data.empty:
                current_price = recent_data['close'].iloc[-1]
                # Use sufficient data for enhanced processing
                model_input_data = recent_data.tail(500)
                
                # Process the data
                agent.process_market_data(current_price, model_input_data)
                
                # Log status periodically
                if simulator_mode:
                    perf = agent.get_simulator_performance()
                    if perf['total_trades'] > 0:
                        print(f"Simulator: Portfolio=${perf['portfolio_value']:.2f} | Return={perf['total_return']:.2%} | Trades={perf['total_trades']}")
            else:
                print(f"No recent data available for {args.symbol}")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down enhanced trader agent...")
        if not simulator_mode:
            data_publisher.stop()
        
        # Show final results if in simulator mode
        if simulator_mode:
            final_perf = agent.get_simulator_performance()
            print(f"\nFinal Simulator Results:")
            print(f"Portfolio Value: ${final_perf['portfolio_value']:.2f}")
            print(f"Total Return: {final_perf['total_return']:.2%}")
            print(f"Total Trades: {final_perf['total_trades']}")
            if final_perf['total_trades'] > 0:
                print(f"Win Rate: {final_perf['win_rate']:.1f}%")

if __name__ == "__main__":
    main()

