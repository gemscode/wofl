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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization'))

from pattern_explorer import PatternExplorer

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
        if len(data) < self.sequence_length:
            return None
        
        features = self.prepare_market_features(data)
        
        for column in features.columns:
            if column not in self.scalers:
                self.scalers[column] = MinMaxScaler(feature_range=(-1, 1))
                self.scalers[column].fit(features[column].values.reshape(-1, 1))
            
            features[column] = self.scalers[column].transform(features[column].values.reshape(-1, 1)).flatten()
        
        sequence = features.iloc[-self.sequence_length:].values
        return sequence

class RealTimeTradingAgent:
    def __init__(self, symbol, model_path, stream_name=None, consumer_name=None, redis_host='trader.wolfx0.com', output_stream=None):
        self.symbol = symbol.upper()
        self.stream_name = stream_name or f"trading_stream_{self.symbol}"
        self.consumer_name = consumer_name or f"agent_{self.symbol}_{int(time.time())}"
        self.group_name = "trading_agents"
        self.redis_host = redis_host
        
        self.output_stream = output_stream or f"trading_analysis_{self.symbol}"
        
        self.data_buffer = deque(maxlen=30)
        self.tick_buffer = deque(maxlen=5)
        
        self.device = self._get_device()
        self.preprocessor = EntryExitDataPreprocessor()
        self.model = self._load_model(model_path)
        
        try:
            redis_config = {
                'host': self.redis_host,
                'port': 6379,
                'db': 0,
                'decode_responses': True
            }
            
            if self.redis_host.lower() not in ['localhost', '127.0.0.1', '::1']:
                try:
                    with open('.redis_passwd', 'r') as f:
                        password = f.read().strip()
                    if password:
                        redis_config['password'] = password
                        print(f"Using password authentication for {self.redis_host}")
                    else:
                        print(f"Password file is empty for {self.redis_host}")
                except FileNotFoundError:
                    print(f"Password file '.redis_passwd' not found for {self.redis_host}")
                    print("Create the file with: echo 'your_password' > .redis_passwd")
                    sys.exit(1)
                except Exception as e:
                    print(f"Error reading password file: {e}")
                    sys.exit(1)
            else:
                print(f"No authentication required for localhost connection")
            
            self.redis_client = redis.Redis(**redis_config)
            self.redis_client.ping()
            print(f"Connected to Redis successfully ({self.redis_host})")
            
        except redis.AuthenticationError:
            print(f"Redis authentication failed for {self.redis_host}")
            print("Check your password in .redis_passwd file")
            sys.exit(1)
        except redis.ConnectionError:
            print(f"Failed to connect to Redis at {self.redis_host}")
            print("Make sure Redis is running and accessible")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected Redis connection error: {e}")
            sys.exit(1)
        
        self.running = False
        self.recommendations = [] 
        self.last_recommendation_time = None
        
        signal.signal(signal.SIGINT, self._signal_handler) 
        signal.signal(signal.SIGTERM, self._signal_handler)
 
    def _get_device(self):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Using Mac M-series GPU (MPS)")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("Using NVIDIA GPU (CUDA)")
            return torch.device("cuda")
        else:
            print("Using CPU only")
            return torch.device("cpu")
    
    def _load_model(self, model_path):
        try:
            print(f"Loading model from: {model_path}")
            model = EnhancedEntryModel()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _parse_market_data(self, message):
        try:
            message_id, data = message
            
            parsed_data = {}
            
            if data.get('type') == 'stream_init':
                return None
            
            if 'timestamp' in data:
                try:
                    parsed_data['timestamp'] = pd.to_datetime(data['timestamp'])
                except:
                    parsed_data['timestamp'] = pd.to_datetime('now')
            else:
                parsed_data['timestamp'] = pd.to_datetime('now')
            
            try:
                parsed_data['open'] = float(data.get('open', data.get('close', 0)))
                parsed_data['high'] = float(data.get('high', data.get('close', 0)))
                parsed_data['low'] = float(data.get('low', data.get('close', 0)))
                parsed_data['close'] = float(data.get('close', 0))
                parsed_data['volume'] = int(data.get('volume', 0))
            except (ValueError, TypeError) as e:
                return None
            
            if parsed_data['close'] == 0:
                return None
            
            return parsed_data
            
        except Exception as e:
            return None
    
    def _should_analyze(self):
        if len(self.tick_buffer) >= 5:
            if self.last_recommendation_time is None:
                return True
            
            time_diff = datetime.now() - self.last_recommendation_time
            if time_diff.total_seconds() >= 30:
                return True
        
        return False
    
    def _calculate_entry_prices(self, recommendation, current_data):
        current_price = float(recommendation['current_price'])
        
        recent_prices = [float(point['close']) for point in list(self.data_buffer)[-20:]]
        volatility = np.std(recent_prices) if len(recent_prices) > 1 else 0.01
        
        high_20 = max(recent_prices) if recent_prices else current_price
        low_20 = min(recent_prices) if recent_prices else current_price
        
        price_levels = {}
        
        if recommendation['action'] == 'BUY' or recommendation['prediction_probs'][1] > 0.35:
            price_levels['buy_aggressive'] = float(current_price)
            price_levels['buy_conservative'] = float(current_price * 0.995)
            price_levels['buy_dip'] = float(low_20 * 1.001)
            
            price_levels['target_1'] = float(current_price * 1.015)
            price_levels['target_2'] = float(current_price * 1.025)
            price_levels['stop_loss'] = float(current_price * 0.985)
            
        elif recommendation['action'] == 'SELL' or recommendation['prediction_probs'][2] > 0.35:
            price_levels['sell_aggressive'] = float(current_price)
            price_levels['sell_conservative'] = float(current_price * 1.005)
            price_levels['sell_bounce'] = float(high_20 * 0.999)
            
            price_levels['target_1'] = float(current_price * 0.985)
            price_levels['target_2'] = float(current_price * 0.975)
            price_levels['stop_loss'] = float(current_price * 1.015)
        
        return price_levels
    
    def _generate_recommendation(self):
        try:
            if len(self.data_buffer) < 15:
                return None
            
            df_data = pd.DataFrame(list(self.data_buffer))
            df_data.set_index('timestamp', inplace=True)
            
            sequence = self.preprocessor.prepare_sequence(df_data)
            if sequence is None:
                return None
            
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                predictions, confidence = self.model(sequence_tensor)
                
                probs = torch.softmax(predictions, dim=1)
                _, predicted = torch.max(predictions, 1)
                
                prediction = predicted.item()
                confidence_score = confidence.item()
            
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action = action_map[prediction]
            
            current_price = float(df_data['close'].iloc[-1])
            high_price = float(df_data['high'].max())
            low_price = float(df_data['low'].min())
            
            recommendation = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'action': action,
                'confidence': float(confidence_score),
                'current_price': current_price,
                'high_price': high_price,
                'low_price': low_price,
                'prediction_probs': [float(p) for p in probs.cpu().numpy().tolist()[0]],
                'data_points_used': len(self.data_buffer),
                'high_confidence': confidence_score > 0.5
            }
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return None
    
    def _display_enhanced_recommendation(self, recommendation):
        if recommendation is None:
            return
        
        price_levels = self._calculate_entry_prices(recommendation, self.data_buffer)
        
        conf_score = recommendation['confidence']
        if conf_score >= 0.7:
            conf_level = "HIGH"
        elif conf_score >= 0.5:
            conf_level = "MEDIUM"
        else:
            conf_level = "LOW"
        
        current_price = recommendation['current_price']
        high_price = recommendation['high_price']
        low_price = recommendation['low_price']
        
        high_distance = ((current_price - high_price) / high_price) * 100
        low_distance = ((current_price - low_price) / low_price) * 100
        
        print("\n" + "="*70)
        print("TRADING ANALYSIS")
        print("="*70)
        print(f"Symbol: {recommendation['symbol']}")
        print(f"Current Price: ${current_price:.4f}")
        print(f"High (30-min): ${high_price:.4f} ({high_distance:+.1f}%)")
        print(f"Low (30-min): ${low_price:.4f} ({low_distance:+.1f}%)")
        print(f"Recommended Action: {recommendation['action']}")
        print(f"Confidence Level: {conf_level} ({conf_score:.1%})")
        
        buy_prob = recommendation['prediction_probs'][1]
        sell_prob = recommendation['prediction_probs'][2]
        hold_prob = recommendation['prediction_probs'][0]
        
        near_high = high_distance > -1.0
        near_low = low_distance < 1.0
        
        if sell_prob > 0.35 and conf_score > 0.5:
            print("SELL SIGNAL DETECTED - Model favors selling")
            if near_high:
                print("CONFIRMED: Price near high supports sell signal")
            elif near_low:
                print("UNUSUAL: Selling near low - exercise caution")
            
            print(f"\nSELL ENTRY LEVELS:")
            print(f"   Aggressive: ${price_levels.get('sell_aggressive', 0):.4f} (Market Order)")
            print(f"   Conservative: ${price_levels.get('sell_conservative', 0):.4f} (Limit Order)")
            print(f"   Bounce Sell: ${price_levels.get('sell_bounce', 0):.4f} (Resistance Level)")
            
            print(f"\nPROFIT TARGETS:")
            print(f"   Target 1: ${price_levels.get('target_1', 0):.4f} (-1.5%)")
            print(f"   Target 2: ${price_levels.get('target_2', 0):.4f} (-2.5%)")
            print(f"   Stop Loss: ${price_levels.get('stop_loss', 0):.4f} (+1.5%)")
            
        elif buy_prob > 0.35 and conf_score > 0.5:
            print("BUY SIGNAL DETECTED - Model favors buying")
            if near_low:
                print("CONFIRMED: Price near low supports buy signal")
            elif near_high:
                print("CAUTION: Buying near recent high - higher risk entry")
                print("SUGGESTION: Consider smaller position size")
            
            print(f"\nBUY ENTRY LEVELS:")
            print(f"   Aggressive: ${price_levels.get('buy_aggressive', 0):.4f} (Market Order)")
            print(f"   Conservative: ${price_levels.get('buy_conservative', 0):.4f} (Limit Order)")
            print(f"   Dip Buy: ${price_levels.get('buy_dip', 0):.4f} (Support Level)")
            
            print(f"\nPROFIT TARGETS:")
            print(f"   Target 1: ${price_levels.get('target_1', 0):.4f} (+1.5%)")
            print(f"   Target 2: ${price_levels.get('target_2', 0):.4f} (+2.5%)")
            print(f"   Stop Loss: ${price_levels.get('stop_loss', 0):.4f} (-1.5%)")
            
        elif near_high and buy_prob > sell_prob and buy_prob > 0.25:
            print("MIXED SIGNAL: Near high but model still favors buying")
            print("SUGGESTION: Wait for pullback or use smaller position")
            print(f"ANALYSIS: BUY {buy_prob:.1%} vs SELL {sell_prob:.1%}")
            
        elif near_low and sell_prob > buy_prob and sell_prob > 0.25:
            print("MIXED SIGNAL: Near low but model still favors selling")
            print("SUGGESTION: Wait for bounce confirmation")
            print(f"ANALYSIS: SELL {sell_prob:.1%} vs BUY {buy_prob:.1%}")
            
        elif hold_prob > 0.6:
            print("STRONG HOLD SIGNAL - Model suggests no action")
            print("GUIDANCE: Market conditions unclear, wait for better setup")
            
        else:
            print("MONITOR ONLY - Wait for clearer directional signal")
            print(f"PROBABILITIES: HOLD {hold_prob:.1%} | BUY {buy_prob:.1%} | SELL {sell_prob:.1%}")
        
        print(f"\nPrediction Breakdown:")
        print(f"   HOLD: {recommendation['prediction_probs'][0]:.1%}")
        print(f"   BUY:  {recommendation['prediction_probs'][1]:.1%}")
        print(f"   SELL: {recommendation['prediction_probs'][2]:.1%}")
        
        print(f"\nAnalysis Details:")
        print(f"   Data Points Used: {recommendation['data_points_used']} (30-min window)")
        print(f"   Analysis Time: {datetime.fromisoformat(recommendation['timestamp']).strftime('%H:%M:%S')}")
        print("="*70)
    
    def _publish_analysis_to_stream(self, recommendation):
        try:
            if recommendation is None:
                return
            
            current_price = float(recommendation['current_price'])
            high_price = float(recommendation['high_price'])
            low_price = float(recommendation['low_price'])
            
            high_distance = float(((current_price - high_price) / high_price) * 100)
            low_distance = float(((current_price - low_price) / low_price) * 100)
            
            conf_score = float(recommendation['confidence'])
            if conf_score >= 0.7:
                conf_level = "HIGH"
                conf_emoji = "HIGH"
            elif conf_score >= 0.5:
                conf_level = "MEDIUM"
                conf_emoji = "MEDIUM"
            else:
                conf_level = "LOW"
                conf_emoji = "LOW"
            
            price_levels = self._calculate_entry_prices(recommendation, self.data_buffer)
            
            buy_prob = float(recommendation['prediction_probs'][1])
            sell_prob = float(recommendation['prediction_probs'][2])
            near_high = high_distance > -1.0
            near_low = low_distance < 1.0
            
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
            
            analysis_message = {
                'timestamp': recommendation['timestamp'],
                'symbol': recommendation['symbol'],
                'analysis_type': 'TRADING_RECOMMENDATION',
                
                'current_price': current_price,
                'high_price': high_price,
                'low_price': low_price,
                'high_distance_pct': round(high_distance, 2),
                'low_distance_pct': round(low_distance, 2),
                
                'recommended_action': recommendation['action'],
                'confidence_score': round(conf_score, 3),
                'confidence_level': conf_level,
                'confidence_emoji': conf_emoji,
                'hold_prob': round(float(recommendation['prediction_probs'][0]), 3),
                'buy_prob': round(float(recommendation['prediction_probs'][1]), 3),
                'sell_prob': round(float(recommendation['prediction_probs'][2]), 3),
                
                'signal_type': signal_type,
                'guidance': guidance,
                'actionable': 'true' if actionable else 'false',
                'high_confidence': 'true' if recommendation['high_confidence'] else 'false',
                
                'entry_levels': json.dumps(price_levels) if actionable else '{}',
                
                'data_points_used': int(recommendation['data_points_used']),
                'analysis_window': '30_minutes',
                'near_high': 'true' if near_high else 'false',
                'near_low': 'true' if near_low else 'false',
                
                'agent_id': self.consumer_name,
                'model_version': 'enhanced_entry_v1',
                'published_at': datetime.now().isoformat()
            }
            
            message_id = self.redis_client.xadd(self.output_stream, analysis_message)
            
            print(f"Analysis published to stream: {self.output_stream} | ID: {message_id}")
            print(f"{signal_type}: {recommendation['action']} @ ${current_price:.4f} ({conf_emoji} {conf_score:.1%})")
            
            return message_id
            
        except Exception as e:
            print(f"Error publishing to stream: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def start_trading(self):
        print(f"Starting Real-Time Trading Agent for {self.symbol}")
        print(f"Input Stream: {self.stream_name}")
        print(f"Output Stream: {self.output_stream}")
        print(f"Consumer: {self.consumer_name}")
        print(f"Confidence Threshold: 50% (for actionable signals)")
        print(f"Analysis Window: 30 data points (optimized for responsiveness)")
        print(f"Analysis Frequency: Every 30 seconds")
        
        self.running = True
        
        try:
            try:
                self.redis_client.xgroup_create(self.stream_name, self.group_name, id='$', mkstream=True)
                print(f"Created consumer group: {self.group_name}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    print(f"Consumer group already exists: {self.group_name}")
                else:
                    raise e
            
            print("Monitoring market data and publishing analysis...\n")
            
            while self.running:
                try:
                    messages = self.redis_client.xreadgroup(
                        self.group_name,
                        self.consumer_name,
                        {self.stream_name: '>'},
                        count=1,
                        block=1000
                    )
                    
                    if messages:
                        for stream, stream_messages in messages:
                            for message in stream_messages:
                                market_data = self._parse_market_data(message)
                                if market_data:
                                    self.data_buffer.append(market_data)
                                    self.tick_buffer.append(market_data)
                                    
                                    print(f"{market_data['timestamp'].strftime('%H:%M:%S')} | "
                                          f"${market_data['close']:.4f} | "
                                          f"Buffer: {len(self.data_buffer)}/30", end='\r')
                                    
                                    if self._should_analyze():
                                        print("\nAnalyzing market data...")
                                        recommendation = self._generate_recommendation()
                                        
                                        if recommendation:
                                            self._display_enhanced_recommendation(recommendation)
                                            
                                            self._publish_analysis_to_stream(recommendation)
                                            
                                            if recommendation['high_confidence']:
                                                self.recommendations.append(recommendation)
                                        
                                        self.tick_buffer.clear()
                                        self.last_recommendation_time = datetime.now()
                                
                                self.redis_client.xack(self.stream_name, self.group_name, message[0])
                
                except redis.ConnectionError:
                    print("Lost connection to Redis, attempting to reconnect...")
                    time.sleep(5)
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nTrading agent stopped by user")
        finally:
            self.running = False
            print(f"\nSession Summary:")
            print(f"   Total Analyses: {len(self.recommendations) + 1}")
            print(f"   Output Stream: {self.output_stream}")
            print(f"   High-Confidence Signals: {len([r for r in self.recommendations if r['high_confidence']])}")
            print(f"   Analysis Window: 30 data points (optimized)")
            print(f"   Analysis Frequency: 30 seconds (fixed)")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Real-time trading agent with NumPy data type fix')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., GERN)')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--stream', type=str, help='Custom input stream name')
    parser.add_argument('--output-stream', type=str, help='Custom output stream name')
    parser.add_argument('--consumer', type=str, help='Custom consumer name')
    parser.add_argument('--redis-host', type=str, default='localhost', help='Redis server host')
    args = parser.parse_args()
    
    model_path = args.model or f"models/entry_model_{args.symbol}.pt"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    
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

