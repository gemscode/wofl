import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

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

class TradingModel(nn.Module):
    """Neural network model for trading decisions."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(TradingModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return self.sigmoid(out)

class TraderAgent:
    """
    Live trading agent that uses trained models to make entry and exit decisions.
    Loads models from the ../models directory and publishes trading signals.
    """

    def __init__(self, symbol: str, data_manager: DataManager = None, data_publisher: DataPublisher = None):
        self.symbol = symbol.upper()
        
        # Initialize data manager and publisher with environment variables
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        
        self.data_manager = data_manager or DataManager(host=redis_host, port=redis_port)
        self.data_publisher = data_publisher or DataPublisher(host=redis_host, port=redis_port)
        
        self.logger = self._setup_logging()

        # Model paths
        self.models_dir = project_root / "models" / self.symbol
        self.entry_model_path = self.models_dir / "entry.pt"
        self.exit_model_path = self.models_dir / "exit.pt"

        # Device setup
        self.device = self._get_device()

        # Load models
        self.entry_model = None
        self.exit_model = None
        self.entry_params = None
        self.exit_params = None
        self._load_models()

        # Trading state
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None

        # Trading parameters
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '0.02'))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '0.04'))
        
        self.logger.info(f"TraderAgent initialized for {self.symbol}")
        self.logger.info(f"Models directory: {self.models_dir}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the trader agent."""
        logger = logging.getLogger(f"TraderAgent_{self.symbol}")
        logger.setLevel(getattr(logging, os.getenv('LOG_LEVEL', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _get_device(self) -> torch.device:
        """Get the appropriate device for model inference."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info("Using CUDA for model inference")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            self.logger.info("Using MPS for model inference")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU for model inference")
        return device

    def _load_models(self):
        """Load trained models from disk."""
        try:
            # Load entry model
            if self.entry_model_path.exists():
                checkpoint = torch.load(self.entry_model_path, map_location=self.device)
                self.entry_params = checkpoint.get('model_params', {})
                
                # Create model with saved parameters
                input_size = self.entry_params.get('input_size', 5)
                hidden_size = self.entry_params.get('hidden_size', 64)
                num_layers = self.entry_params.get('num_layers', 2)
                dropout = self.entry_params.get('dropout', 0.2)
                
                self.entry_model = TradingModel(input_size, hidden_size, num_layers, dropout)
                self.entry_model.load_state_dict(checkpoint['model_state_dict'])
                self.entry_model.to(self.device)
                self.entry_model.eval()
                
                self.logger.info(f"Entry model loaded successfully")
                self.logger.info(f"Entry model params: {self.entry_params}")
            else:
                self.logger.warning(f"Entry model not found at {self.entry_model_path}")

            # Load exit model
            if self.exit_model_path.exists():
                checkpoint = torch.load(self.exit_model_path, map_location=self.device)
                self.exit_params = checkpoint.get('model_params', {})
                
                # Create model with saved parameters
                input_size = self.exit_params.get('input_size', 5)
                hidden_size = self.exit_params.get('hidden_size', 64)
                num_layers = self.exit_params.get('num_layers', 2)
                dropout = self.exit_params.get('dropout', 0.2)
                
                self.exit_model = TradingModel(input_size, hidden_size, num_layers, dropout)
                self.exit_model.load_state_dict(checkpoint['model_state_dict'])
                self.exit_model.to(self.device)
                self.exit_model.eval()
                
                self.logger.info(f"Exit model loaded successfully")
                self.logger.info(f"Exit model params: {self.exit_params}")
            else:
                self.logger.warning(f"Exit model not found at {self.exit_model_path}")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.entry_model = None
            self.exit_model = None

    def _prepare_model_input(self, data: pd.DataFrame, sequence_length: int = 60) -> torch.Tensor:
        """
        Prepare input data for model inference.
        
        Args:
            data: DataFrame with OHLCV data
            sequence_length: Length of input sequence
            
        Returns:
            Tensor ready for model input
        """
        if len(data) < sequence_length:
            self.logger.warning(f"Insufficient data for model input: {len(data)} < {sequence_length}")
            return None
            
        # Use the last sequence_length rows
        recent_data = data.tail(sequence_length).copy()
        
        # Calculate technical indicators
        recent_data['returns'] = recent_data['close'].pct_change()
        recent_data['volatility'] = recent_data['returns'].rolling(window=10).std()
        recent_data['rsi'] = self._calculate_rsi(recent_data['close'])
        recent_data['ma_ratio'] = recent_data['close'] / recent_data['close'].rolling(window=20).mean()
        recent_data['volume_ratio'] = recent_data['volume'] / recent_data['volume'].rolling(window=20).mean()
        
        # Select features (OHLC + indicators)
        features = ['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility', 'rsi', 'ma_ratio', 'volume_ratio']
        
        # Handle missing values
        feature_data = recent_data[features].fillna(method='ffill').fillna(0)
        
        # Normalize features
        feature_data = self._normalize_features(feature_data)
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(feature_data.values).unsqueeze(0).to(self.device)
        
        return tensor_data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for model input."""
        normalized_data = data.copy()
        
        # Price-based features: normalize by close price
        price_features = ['open', 'high', 'low', 'close']
        for feature in price_features:
            if feature in normalized_data.columns:
                normalized_data[feature] = normalized_data[feature] / normalized_data['close'].iloc[-1]
        
        # Volume: normalize by mean
        if 'volume' in normalized_data.columns:
            mean_volume = normalized_data['volume'].mean()
            if mean_volume != 0:
                normalized_data['volume'] = normalized_data['volume'] / mean_volume
        
        # Other features are already normalized or ratios
        return normalized_data

    def make_entry_decision(self, current_price: float, market_data: pd.DataFrame) -> Dict:
        """
        Make entry decision using the trained entry model.
        
        Args:
            current_price: Current market price
            market_data: Recent market data for analysis
            
        Returns:
            Dictionary containing decision details
        """
        if self.entry_model is None:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No entry model available'}
        
        if self.position is not None:
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Already in {self.position} position'}
        
        try:
            # Prepare model input
            model_input = self._prepare_model_input(market_data)
            if model_input is None:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            # Make prediction
            with torch.no_grad():
                prediction = self.entry_model(model_input)
                confidence = prediction.item()
            
            # Make decision based on confidence
            if confidence > self.confidence_threshold:
                action = 'buy'
                reason = f'Entry signal with confidence {confidence:.3f}'
            else:
                action = 'hold'
                reason = f'Confidence too low: {confidence:.3f} < {self.confidence_threshold}'
            
            return {
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error making entry decision: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}

    def make_exit_decision(self, current_price: float, market_data: pd.DataFrame) -> Dict:
        """
        Make exit decision using the trained exit model and risk management rules.
        
        Args:
            current_price: Current market price
            market_data: Recent market data for analysis
            
        Returns:
            Dictionary containing decision details
        """
        if self.position is None:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No position to exit'}
        
        # Check stop loss and take profit first
        if self.entry_price is not None:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            if pnl_pct <= -self.stop_loss_pct:
                return {
                    'action': 'sell',
                    'confidence': 1.0,
                    'reason': f'Stop loss triggered: {pnl_pct:.3%}',
                    'price': current_price,
                    'pnl_pct': pnl_pct,
                    'timestamp': datetime.now().isoformat()
                }
            
            if pnl_pct >= self.take_profit_pct:
                return {
                    'action': 'sell',
                    'confidence': 1.0,
                    'reason': f'Take profit triggered: {pnl_pct:.3%}',
                    'price': current_price,
                    'pnl_pct': pnl_pct,
                    'timestamp': datetime.now().isoformat()
                }
        
        # Use exit model if available
        if self.exit_model is not None:
            try:
                # Prepare model input
                model_input = self._prepare_model_input(market_data)
                if model_input is None:
                    return {'action': 'hold', 'confidence': 0.0, 'reason': 'Insufficient data for exit model'}
                
                # Make prediction
                with torch.no_grad():
                    prediction = self.exit_model(model_input)
                    confidence = prediction.item()
                
                # Make decision based on confidence
                if confidence > self.confidence_threshold:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price else 0
                    return {
                        'action': 'sell',
                        'confidence': confidence,
                        'reason': f'Exit signal with confidence {confidence:.3f}',
                        'price': current_price,
                        'pnl_pct': pnl_pct,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'action': 'hold',
                        'confidence': confidence,
                        'reason': f'Exit confidence too low: {confidence:.3f}',
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except Exception as e:
                self.logger.error(f"Error making exit decision: {e}")
                return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
        
        return {'action': 'hold', 'confidence': 0.0, 'reason': 'No exit model available'}

    def execute_trade(self, decision: Dict):
        """
        Execute a trading decision and update position state.
        
        Args:
            decision: Decision dictionary from entry/exit methods
        """
        action = decision['action']
        price = decision.get('price')
        timestamp = decision.get('timestamp', datetime.now().isoformat())
        
        if action == 'buy' and self.position is None:
            # Execute buy order
            self.position = 'long'
            self.entry_price = price
            self.entry_time = timestamp
            
            # Publish trade signal
            trade_signal = {
                'symbol': self.symbol,
                'action': 'buy',
                'price': price,
                'confidence': decision['confidence'],
                'reason': decision['reason'],
                'timestamp': timestamp,
                'agent_id': f"trader_agent_{self.symbol}"
            }
            
            self.data_publisher.publish_trade_signal(self.symbol, trade_signal)
            self.logger.info(f"EXECUTED BUY: {self.symbol} at ${price:.2f} | Confidence: {decision['confidence']:.3f}")
            
        elif action == 'sell' and self.position == 'long':
            # Execute sell order
            pnl_pct = decision.get('pnl_pct', (price - self.entry_price) / self.entry_price if self.entry_price else 0)
            
            # Publish trade signal
            trade_signal = {
                'symbol': self.symbol,
                'action': 'sell',
                'price': price,
                'entry_price': self.entry_price,
                'pnl_pct': pnl_pct,
                'confidence': decision['confidence'],
                'reason': decision['reason'],
                'timestamp': timestamp,
                'hold_duration': self._calculate_hold_duration(),
                'agent_id': f"trader_agent_{self.symbol}"
            }
            
            self.data_publisher.publish_trade_signal(self.symbol, trade_signal)
            self.logger.info(f"EXECUTED SELL: {self.symbol} at ${price:.2f} | PnL: {pnl_pct:.3%} | Confidence: {decision['confidence']:.3f}")
            
            # Reset position
            self.position = None
            self.entry_price = None
            self.entry_time = None

    def _calculate_hold_duration(self) -> Optional[float]:
        """Calculate how long the position was held in hours."""
        if self.entry_time:
            try:
                entry_dt = datetime.fromisoformat(self.entry_time.replace('Z', '+00:00'))
                current_dt = datetime.now()
                duration = (current_dt - entry_dt).total_seconds() / 3600  # hours
                return duration
            except Exception as e:
                self.logger.error(f"Error calculating hold duration: {e}")
        return None

    def process_market_data(self, current_price: float, market_data: pd.DataFrame):
        """
        Process new market data and make trading decisions.
        
        Args:
            current_price: Current market price
            market_data: Recent market data DataFrame
        """
        try:
            if self.position is None:
                # Look for entry opportunities
                decision = self.make_entry_decision(current_price, market_data)
                if decision['action'] == 'buy':
                    self.execute_trade(decision)
                else:
                    self.logger.debug(f"Entry decision: {decision['reason']}")
            else:
                # Look for exit opportunities
                decision = self.make_exit_decision(current_price, market_data)
                if decision['action'] == 'sell':
                    self.execute_trade(decision)
                else:
                    self.logger.debug(f"Exit decision: {decision['reason']}")
                    
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")

    def get_position_status(self) -> Dict:
        """Get current position status."""
        return {
            'symbol': self.symbol,
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'models_loaded': {
                'entry': self.entry_model is not None,
                'exit': self.exit_model is not None
            }
        }

    def reload_models(self):
        """Reload models from disk (useful for live model updates)."""
        self.logger.info("Reloading models...")
        self._load_models()

def main():
    """Main function for running the trader agent."""
    import argparse

    parser = argparse.ArgumentParser(description='Run trader agent for a stock symbol')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    parser.add_argument('--host', default=None, help='Redis host (overrides .env)')
    parser.add_argument('--port', type=int, default=None, help='Redis port (overrides .env)')
    args = parser.parse_args()

    # Use command line args or fall back to environment variables
    redis_host = args.host or os.getenv('REDIS_HOST', 'localhost')
    redis_port = args.port or int(os.getenv('REDIS_PORT', '6379'))

    # Initialize components
    data_manager = DataManager(host=redis_host, port=redis_port)
    data_publisher = DataPublisher(host=redis_host, port=redis_port)
    agent = TraderAgent(args.symbol, data_manager, data_publisher)

    # Start heartbeat
    data_publisher.start_heartbeat(f"trader_agent_{args.symbol}")

    print(f"🚀 Starting trader agent for {args.symbol}")
    print(f"Redis connection: {redis_host}:{redis_port}")
    print(f"Position status: {agent.get_position_status()}")

    try:
        while True:
            # Get recent market data
            recent_data = data_manager.get_market_data(args.symbol, '1min')
            if not recent_data.empty:
                current_price = recent_data['close'].iloc[-1]
                # Use last 100 data points for model input
                model_input_data = recent_data.tail(100)
                # Process the data
                agent.process_market_data(current_price, model_input_data)
            else:
                print(f"No recent data available for {args.symbol}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Shutting down trader agent...")
        data_publisher.stop()

if __name__ == "__main__":
    main()

