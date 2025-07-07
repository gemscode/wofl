#!/usr/bin/env python3

import redis
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import time
import sys
import os
import argparse
from datetime import datetime, timedelta
import json
from collections import deque
import threading
import signal

# Add the optimization path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization'))

from pattern_explorer import PatternExplorer

# Import model class from your training script
class EnhancedEntryModel(nn.Module):
    def __init__(self, input_features=8, sequence_length=15, cnn_filters=32, cnn_kernel=2, lstm_hidden=64, lstm_layers=1, dropout=0.2, num_classes=3):
        super(EnhancedEntryModel, self).__init__()
        self.conv1 = nn.Conv1d(input_features, cnn_filters, kernel_size=cnn_kernel, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters*2, kernel_size=cnn_kernel, padding=1)
        self.conv3 = nn.Conv1d(cnn_filters*2, cnn_filters, kernel_size=cnn_kernel, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.bn2 = nn.BatchNorm1d(cnn_filters*2)
        self.bn3 = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.fc2 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.fc3 = nn.Linear(lstm_hidden // 2, num_classes)
        self.confidence_head = nn.Linear(lstm_hidden * 2, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.bilstm(x)
        features = lstm_out[:, -1, :]
        predictions = self.fc1(features)
        predictions = self.relu(predictions)
        predictions = self.dropout(predictions)
        predictions = self.fc2(predictions)
        predictions = self.relu(predictions)
        predictions = self.dropout(predictions)
        predictions = self.fc3(predictions)
        confidence = torch.sigmoid(self.confidence_head(features))
        return predictions, confidence

class EntryExitDataPreprocessor:
    def __init__(self, sequence_length=15):
        self.sequence_length = sequence_length
        self.scalers = {}
    
    def prepare_market_features(self, data):
        features_df = pd.DataFrame()
        features_df['returns_1'] = data['close'].pct_change(1)
        features_df['returns_3'] = data['close'].pct_change(3)
        features_df['returns_5'] = data['close'].pct_change(5)
        features_df['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
        features_df['volume_momentum'] = data['volume'].pct_change(1)
        features_df['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
        features_df['volatility'] = data['close'].rolling(5).std() / data['close'].rolling(5).mean()
        features_df['rsi_fast'] = self._calculate_rsi(data['close'], 5)
        features_df = features_df.bfill().ffill()
        return features_df
    
    def _calculate_rsi(self, series, period=5):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs)) / 100.0
    
    def prepare_sequence(self, data):
        """Prepare the most recent sequence for prediction"""
        if len(data) < self.sequence_length:
            return None
        
        features = self.prepare_market_features(data)
        
        # Scale features (in production, you'd load pre-fitted scalers)
        for column in features.columns:
            if column not in self.scalers:
                self.scalers[column] = MinMaxScaler(feature_range=(-1, 1))
                # Fit on available data
                self.scalers[column].fit(features[column].values.reshape(-1, 1))
            
            features[column] = self.scalers[column].transform(features[column].values.reshape(-1, 1)).flatten()
        
        # Get the most recent sequence
        sequence = features.iloc[-self.sequence_length:].values
        return sequence

class RealTimeTradingAgent:
    def __init__(self, symbol, model_path, stream_name=None, consumer_name=None, redis_host='localhost', output_stream=None):
        self.symbol = symbol.upper()
        self.stream_name = stream_name or f"trading_stream_{self.symbol}"
        self.consumer_name = consumer_name or f"agent_{self.symbol}_{int(time.time())}"
        self.group_name = "trading_agents"
        self.redis_host = redis_host
        
        # NEW: Output stream for publishing analysis results
        self.output_stream = output_stream or f"trading_analysis_{self.symbol}"
        
        # OPTIMIZED: Reduced buffer size from 100 to 30 for faster response
        self.data_buffer = deque(maxlen=30)  # Much more responsive
        self.tick_buffer = deque(maxlen=5)  # Buffer for 5 ticks before analysis
        
        # Initialize components
        self.device = self._get_device()
        self.preprocessor = EntryExitDataPreprocessor()
        self.model = self._load_model(model_path)
        
        # Redis connection
        try:
            self.redis_client = redis.Redis(host=self.redis_host, port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("✅ Connected to Redis successfully")
        except redis.ConnectionError:
            print("❌ Failed to connect to Redis. Make sure Redis is running.")
            sys.exit(1)
        
        # Trading state
        self.running = False
        self.recommendations = []
        self.last_recommendation_time = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _get_device(self):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✅ Using Mac M-series GPU (MPS)")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("✅ Using NVIDIA GPU (CUDA)")
            return torch.device("cuda")
        else:
            print("⚠️ Using CPU only")
            return torch.device("cpu")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            print(f"📂 Loading model from: {model_path}")
            model = EnhancedEntryModel()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            print("✅ Model loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _parse_market_data(self, message):
        """Parse market data from Redis message with robust error handling"""
        try:
            message_id, data = message  # Unpack message
            
            # Handle different possible field names and structures
            parsed_data = {}
            
            # Skip initialization messages
            if data.get('type') == 'stream_init':
                return None
            
            # Parse timestamp
            if 'timestamp' in data:
                try:
                    parsed_data['timestamp'] = pd.to_datetime(data['timestamp'])
                except:
                    # Fallback to current time if timestamp parsing fails
                    parsed_data['timestamp'] = pd.to_datetime('now')
            else:
                parsed_data['timestamp'] = pd.to_datetime('now')
            
            # Parse OHLCV data with fallbacks
            try:
                parsed_data['open'] = float(data.get('open', data.get('close', 0)))
                parsed_data['high'] = float(data.get('high', data.get('close', 0)))
                parsed_data['low'] = float(data.get('low', data.get('close', 0)))
                parsed_data['close'] = float(data.get('close', 0))
                parsed_data['volume'] = int(data.get('volume', 0))
            except (ValueError, TypeError) as e:
                return None
            
            # Validate that we have essential data
            if parsed_data['close'] == 0:
                return None
            
            return parsed_data
            
        except Exception as e:
            return None
    
    def _should_analyze(self):
        """FIXED: Check if we should run analysis (every 30 seconds or 5 ticks)"""
        if len(self.tick_buffer) >= 5:
            if self.last_recommendation_time is None:
                return True
            
            # FIXED: Changed from 300 seconds (5 minutes) to 30 seconds
            time_diff = datetime.now() - self.last_recommendation_time
            if time_diff.total_seconds() >= 30:  # Changed from 300 to 30
                return True
        
        return False
    
    def _calculate_entry_prices(self, recommendation, current_data):
        """Calculate specific entry and exit price levels"""
        current_price = recommendation['current_price']
        
        # Calculate recent volatility for price levels
        recent_prices = [point['close'] for point in list(self.data_buffer)[-20:]]
        volatility = np.std(recent_prices) if len(recent_prices) > 1 else 0.01
        
        # Calculate support and resistance levels
        high_20 = max(recent_prices) if recent_prices else current_price
        low_20 = min(recent_prices) if recent_prices else current_price
        
        price_levels = {}
        
        if recommendation['action'] == 'BUY' or recommendation['prediction_probs'][1] > 0.35:
            # BUY entry levels
            price_levels['buy_aggressive'] = current_price  # Current market price
            price_levels['buy_conservative'] = current_price * 0.995  # 0.5% below current
            price_levels['buy_dip'] = low_20 * 1.001  # Just above recent low
            
            # Target prices for BUY
            price_levels['target_1'] = current_price * 1.015  # 1.5% profit target
            price_levels['target_2'] = current_price * 1.025  # 2.5% profit target
            price_levels['stop_loss'] = current_price * 0.985  # 1.5% stop loss
            
        elif recommendation['action'] == 'SELL' or recommendation['prediction_probs'][2] > 0.35:
            # SELL entry levels
            price_levels['sell_aggressive'] = current_price  # Current market price
            price_levels['sell_conservative'] = current_price * 1.005  # 0.5% above current
            price_levels['sell_bounce'] = high_20 * 0.999  # Just below recent high
            
            # Target prices for SELL
            price_levels['target_1'] = current_price * 0.985  # 1.5% profit target
            price_levels['target_2'] = current_price * 0.975  # 2.5% profit target
            price_levels['stop_loss'] = current_price * 1.015  # 1.5% stop loss
        
        return price_levels
    
    def _generate_recommendation(self):
        """Generate trading recommendation with detailed confidence reporting and price tracking"""
        try:
            # OPTIMIZED: Reduced minimum requirement from 15 to 15 (same, but now with 30-point buffer)
            if len(self.data_buffer) < 15:
                return None
            
            # Convert buffer to DataFrame
            df_data = pd.DataFrame(list(self.data_buffer))
            df_data.set_index('timestamp', inplace=True)
            
            # Prepare sequence for model
            sequence = self.preprocessor.prepare_sequence(df_data)
            if sequence is None:
                return None
            
            # Make prediction
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                predictions, confidence = self.model(sequence_tensor)
                
                # Get prediction and confidence
                probs = torch.softmax(predictions, dim=1)
                _, predicted = torch.max(predictions, 1)
                
                prediction = predicted.item()
                confidence_score = confidence.item()
            
            # Map prediction to action
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action = action_map[prediction]
            
            current_price = df_data['close'].iloc[-1]
            
            # NEW: Track highest and lowest prices in buffer
            high_price = df_data['high'].max()
            low_price = df_data['low'].min()
            
            # ENHANCED: Return recommendation with price context
            recommendation = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'action': action,
                'confidence': confidence_score,
                'current_price': current_price,
                'high_price': high_price,      # NEW: Add high price tracking
                'low_price': low_price,        # NEW: Add low price tracking
                'prediction_probs': probs.cpu().numpy().tolist()[0],
                'data_points_used': len(self.data_buffer),
                'high_confidence': confidence_score > 0.5
            }
            
            return recommendation
            
        except Exception as e:
            print(f"❌ Error generating recommendation: {e}")
            return None
    
    def _display_enhanced_recommendation(self, recommendation):
        """FIXED: Enhanced display with corrected logic - no more contradictions"""
        if recommendation is None:
            return
        
        # Calculate entry prices
        price_levels = self._calculate_entry_prices(recommendation, self.data_buffer)
        
        # Determine confidence level description
        conf_score = recommendation['confidence']
        if conf_score >= 0.7:
            conf_level = "🟢 HIGH"
        elif conf_score >= 0.5:
            conf_level = "🟡 MEDIUM"
        else:
            conf_level = "🔴 LOW"
        
        # Calculate price movement context
        current_price = recommendation['current_price']
        high_price = recommendation['high_price']
        low_price = recommendation['low_price']
        
        high_distance = ((current_price - high_price) / high_price) * 100
        low_distance = ((current_price - low_price) / low_price) * 100
        
        print("\n" + "="*70)
        print("🤖 ENHANCED TRADING ANALYSIS")
        print("="*70)
        print(f"📊 Symbol: {recommendation['symbol']}")
        print(f"💰 Current Price: ${current_price:.4f}")
        print(f"📈 High (30-min): ${high_price:.4f} ({high_distance:+.1f}%)")
        print(f"📉 Low (30-min): ${low_price:.4f} ({low_distance:+.1f}%)")
        print(f"🎯 Recommended Action: **{recommendation['action']}**")
        print(f"📈 Confidence Level: {conf_level} ({conf_score:.1%})")
        
        # FIXED: Model-aligned logic instead of contradictory position-based warnings
        buy_prob = recommendation['prediction_probs'][1]
        sell_prob = recommendation['prediction_probs'][2]
        hold_prob = recommendation['prediction_probs'][0]
        
        # Check if price is near highs or lows
        near_high = high_distance > -1.0  # Within 1% of recent high
        near_low = low_distance < 1.0     # Within 1% of recent low
        
        # FIXED: Logic that aligns with model predictions
        if sell_prob > 0.35 and conf_score > 0.5:
            print("💡 **SELL SIGNAL DETECTED** - Model favors selling")
            if near_high:
                print("✅ **CONFIRMED**: Price near high supports sell signal")
            elif near_low:
                print("⚠️ **UNUSUAL**: Selling near low - exercise caution")
            
            print(f"\n💰 SELL ENTRY LEVELS:")
            print(f"   🔴 Aggressive: ${price_levels.get('sell_aggressive', 0):.4f} (Market Order)")
            print(f"   🟡 Conservative: ${price_levels.get('sell_conservative', 0):.4f} (Limit Order)")
            print(f"   🔵 Bounce Sell: ${price_levels.get('sell_bounce', 0):.4f} (Resistance Level)")
            
            print(f"\n🎯 PROFIT TARGETS:")
            print(f"   🥇 Target 1: ${price_levels.get('target_1', 0):.4f} (-1.5%)")
            print(f"   🥈 Target 2: ${price_levels.get('target_2', 0):.4f} (-2.5%)")
            print(f"   🛑 Stop Loss: ${price_levels.get('stop_loss', 0):.4f} (+1.5%)")
            
        elif buy_prob > 0.35 and conf_score > 0.5:
            print("💡 **BUY SIGNAL DETECTED** - Model favors buying")
            if near_low:
                print("✅ **CONFIRMED**: Price near low supports buy signal")
            elif near_high:
                print("⚠️ **CAUTION**: Buying near recent high - higher risk entry")
                print("💡 **SUGGESTION**: Consider smaller position size")
            
            print(f"\n💰 BUY ENTRY LEVELS:")
            print(f"   🟢 Aggressive: ${price_levels.get('buy_aggressive', 0):.4f} (Market Order)")
            print(f"   🟡 Conservative: ${price_levels.get('buy_conservative', 0):.4f} (Limit Order)")
            print(f"   🔵 Dip Buy: ${price_levels.get('buy_dip', 0):.4f} (Support Level)")
            
            print(f"\n🎯 PROFIT TARGETS:")
            print(f"   🥇 Target 1: ${price_levels.get('target_1', 0):.4f} (+1.5%)")
            print(f"   🥈 Target 2: ${price_levels.get('target_2', 0):.4f} (+2.5%)")
            print(f"   🛑 Stop Loss: ${price_levels.get('stop_loss', 0):.4f} (-1.5%)")
            
        elif near_high and buy_prob > sell_prob and buy_prob > 0.25:
            print("⚠️ **MIXED SIGNAL**: Near high but model still favors buying")
            print("💡 **SUGGESTION**: Wait for pullback or use smaller position")
            print(f"🔍 **ANALYSIS**: BUY {buy_prob:.1%} vs SELL {sell_prob:.1%}")
            
        elif near_low and sell_prob > buy_prob and sell_prob > 0.25:
            print("⚠️ **MIXED SIGNAL**: Near low but model still favors selling")
            print("💡 **SUGGESTION**: Wait for bounce confirmation")
            print(f"🔍 **ANALYSIS**: SELL {sell_prob:.1%} vs BUY {buy_prob:.1%}")
            
        elif hold_prob > 0.6:
            print("⚠️ **STRONG HOLD SIGNAL** - Model suggests no action")
            print("💡 **GUIDANCE**: Market conditions unclear, wait for better setup")
            
        else:
            print("⚠️ **MONITOR ONLY** - Wait for clearer directional signal")
            print(f"🔍 **PROBABILITIES**: HOLD {hold_prob:.1%} | BUY {buy_prob:.1%} | SELL {sell_prob:.1%}")
        
        print(f"\n📊 Prediction Breakdown:")
        print(f"   • HOLD: {recommendation['prediction_probs'][0]:.1%}")
        print(f"   • BUY:  {recommendation['prediction_probs'][1]:.1%}")
        print(f"   • SELL: {recommendation['prediction_probs'][2]:.1%}")
        
        print(f"\n📈 Analysis Details:")
        print(f"   • Data Points Used: {recommendation['data_points_used']} (30-min window)")
        print(f"   • Analysis Time: {datetime.fromisoformat(recommendation['timestamp']).strftime('%H:%M:%S')}")
        print("="*70)
    
    def _publish_analysis_to_stream(self, recommendation):
        """FIXED: Publish trading analysis to Redis stream with proper data type conversion"""
        try:
            if recommendation is None:
                return
            
            # Calculate price context
            current_price = recommendation['current_price']
            high_price = recommendation['high_price']
            low_price = recommendation['low_price']
            
            high_distance = ((current_price - high_price) / high_price) * 100
            low_distance = ((current_price - low_price) / low_price) * 100
            
            # Determine confidence level
            conf_score = recommendation['confidence']
            if conf_score >= 0.7:
                conf_level = "HIGH"
                conf_emoji = "🟢"
            elif conf_score >= 0.5:
                conf_level = "MEDIUM"
                conf_emoji = "🟡"
            else:
                conf_level = "LOW"
                conf_emoji = "🔴"
            
            # Calculate entry prices
            price_levels = self._calculate_entry_prices(recommendation, self.data_buffer)
            
            # FIXED: Enhanced logic for actionable signals
            buy_prob = recommendation['prediction_probs'][1]
            sell_prob = recommendation['prediction_probs'][2]
            near_high = high_distance > -1.0
            near_low = low_distance < 1.0
            
            # Determine signal type and guidance based on model predictions
            if sell_prob > 0.35 and conf_score > 0.5:
                signal_type = "SELL_SIGNAL"
                guidance = f"Model favors selling ({sell_prob:.1%} probability)"
                actionable = True
            elif buy_prob > 0.35 and conf_score > 0.5:
                signal_type = "BUY_SIGNAL"
                guidance = f"Model favors buying ({buy_prob:.1%} probability)"
                actionable = True
            elif near_high and buy_prob > sell_prob and buy_prob > 0.25:
                signal_type = "MIXED_SIGNAL"
                guidance = "Near high but model favors buying - use caution"
                actionable = False
            elif near_low and sell_prob > buy_prob and sell_prob > 0.25:
                signal_type = "MIXED_SIGNAL"
                guidance = "Near low but model favors selling - wait for confirmation"
                actionable = False
            else:
                signal_type = "MONITOR"
                guidance = "Wait for clearer directional signal"
                actionable = False
            
            # FIXED: Convert all boolean values to strings for Redis compatibility
            analysis_message = {
                # Basic Information
                'timestamp': recommendation['timestamp'],
                'symbol': recommendation['symbol'],
                'analysis_type': 'TRADING_RECOMMENDATION',
                
                # Price Information
                'current_price': current_price,
                'high_price': high_price,
                'low_price': low_price,
                'high_distance_pct': round(high_distance, 2),
                'low_distance_pct': round(low_distance, 2),
                
                # Model Predictions
                'recommended_action': recommendation['action'],
                'confidence_score': round(conf_score, 3),
                'confidence_level': conf_level,
                'confidence_emoji': conf_emoji,
                'hold_prob': round(recommendation['prediction_probs'][0], 3),
                'buy_prob': round(recommendation['prediction_probs'][1], 3),
                'sell_prob': round(recommendation['prediction_probs'][2], 3),
                
                # Signal Analysis
                'signal_type': signal_type,
                'guidance': guidance,
                'actionable': 'true' if actionable else 'false',  # FIXED: Convert bool to string
                'high_confidence': 'true' if recommendation['high_confidence'] else 'false',  # FIXED: Convert bool to string
                
                # Entry Levels (if applicable) - Convert to JSON string
                'entry_levels': json.dumps(price_levels) if actionable else '{}',
                
                # Analysis Context
                'data_points_used': recommendation['data_points_used'],
                'analysis_window': '30_minutes',
                'near_high': 'true' if near_high else 'false',  # FIXED: Convert bool to string
                'near_low': 'true' if near_low else 'false',    # FIXED: Convert bool to string
                
                # Metadata
                'agent_id': self.consumer_name,
                'model_version': 'enhanced_entry_v1',
                'published_at': datetime.now().isoformat()
            }
            
            # Publish to Redis stream
            message_id = self.redis_client.xadd(self.output_stream, analysis_message)
            
            # Console confirmation (minimal)
            print(f"📡 Analysis published to stream: {self.output_stream} | ID: {message_id}")
            print(f"🎯 {signal_type}: {recommendation['action']} @ ${current_price:.4f} ({conf_emoji} {conf_score:.1%})")
            
            return message_id
            
        except Exception as e:
            print(f"❌ Error publishing to stream: {e}")
            return None
    
    def start_trading(self):
        """Start the real-time trading agent with optimized output"""
        print(f"🚀 Starting Real-Time Trading Agent for {self.symbol}")
        print(f"📊 Input Stream: {self.stream_name}")
        print(f"📡 Output Stream: {self.output_stream}")
        print(f"👤 Consumer: {self.consumer_name}")
        print(f"🎯 Confidence Threshold: 50% (for actionable signals)")
        print(f"⚡ Analysis Window: 30 data points (optimized for responsiveness)")
        print(f"⏱️ Analysis Frequency: Every 30 seconds (FIXED)")
        
        self.running = True
        
        try:
            # Create consumer group if it doesn't exist
            try:
                self.redis_client.xgroup_create(self.stream_name, self.group_name, id='$', mkstream=True)
                print(f"👥 Created consumer group: {self.group_name}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    print(f"👥 Consumer group already exists: {self.group_name}")
                else:
                    raise e
            
            print("🎧 Monitoring market data and publishing analysis...\n")
            
            while self.running:
                try:
                    # Read from stream
                    messages = self.redis_client.xreadgroup(
                        self.group_name,
                        self.consumer_name,
                        {self.stream_name: '>'},
                        count=1,
                        block=1000  # Block for 1 second
                    )
                    
                    if messages:
                        for stream, stream_messages in messages:
                            for message in stream_messages:
                                # Parse market data (no debug output)
                                market_data = self._parse_market_data(message)
                                if market_data:
                                    # Add to buffers
                                    self.data_buffer.append(market_data)
                                    self.tick_buffer.append(market_data)
                                    
                                    # Simple progress indicator with buffer size update
                                    print(f"📈 {market_data['timestamp'].strftime('%H:%M:%S')} | "
                                          f"${market_data['close']:.4f} | "
                                          f"Buffer: {len(self.data_buffer)}/30", end='\r')
                                    
                                    # Check if we should analyze
                                    if self._should_analyze():
                                        print("\n🔍 Analyzing market data...")
                                        recommendation = self._generate_recommendation()
                                        
                                        if recommendation:
                                            # Display enhanced recommendation (console)
                                            self._display_enhanced_recommendation(recommendation)
                                            
                                            # Publish to Redis stream (for other clients)
                                            self._publish_analysis_to_stream(recommendation)
                                            
                                            if recommendation['high_confidence']:
                                                self.recommendations.append(recommendation)
                                        
                                        # Clear tick buffer and update time
                                        self.tick_buffer.clear()
                                        self.last_recommendation_time = datetime.now()
                                
                                # Acknowledge message
                                self.redis_client.xack(self.stream_name, self.group_name, message[0])
                
                except redis.ConnectionError:
                    print("❌ Lost connection to Redis, attempting to reconnect...")
                    time.sleep(5)
                except Exception as e:
                    print(f"⚠️ Error in main loop: {e}")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n🛑 Trading agent stopped by user")
        finally:
            self.running = False
            print(f"\n📊 Session Summary:")
            print(f"   • Total Analyses: {len(self.recommendations) + 1}")
            print(f"   • Output Stream: {self.output_stream}")
            print(f"   • High-Confidence Signals: {len([r for r in self.recommendations if r['high_confidence']])}")
            print(f"   • Analysis Window: 30 data points (optimized)")
            print(f"   • Analysis Frequency: 30 seconds (fixed)")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Real-time trading agent with fixed Redis publishing')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., GERN)')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--stream', type=str, help='Custom input stream name')
    parser.add_argument('--output-stream', type=str, help='Custom output stream name')
    parser.add_argument('--consumer', type=str, help='Custom consumer name')
    parser.add_argument('--redis-host', type=str, default='localhost', help='Redis server host')
    args = parser.parse_args()
    
    # Default model path
    model_path = args.model or f"checkpoints/entry_model_{args.symbol}.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    # Initialize and start trading agent
    agent = RealTimeTradingAgent(
        args.symbol, 
        model_path, 
        args.stream, 
        args.consumer, 
        args.redis_host,
        args.output_stream
    )
    agent.start_trading()

if __name__ == "__main__":
    main()

