#!/usr/bin/env python3

import redis
import json
import argparse
from datetime import datetime
import signal
import sys
import time
import os

class TradingAnalysisSubscriber:
    def __init__(self, symbol, redis_host='trader.wolfx0.com', redis_port=6379, stream_name=None, consumer_name=None):
        self.symbol = symbol.upper()
        self.stream_name = stream_name or f"trading_analysis_{self.symbol}"
        self.consumer_name = consumer_name or f"subscriber_{int(datetime.now().timestamp())}"
        self.group_name = "analysis_subscribers"
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.running = False
        
        # Read Redis password from file
        redis_password = self._read_redis_password()
        
        # Redis connection with authentication
        try:
            redis_params = {
                'host': self.redis_host,
                'port': self.redis_port,
                'db': 0,
                'decode_responses': True
            }
            
            # Only add password for non-localhost connections
            if self.redis_host.lower() not in ['localhost', '127.0.0.1', '::1']:
                if redis_password:
                    redis_params['password'] = redis_password
                    print(f"Using password authentication for {self.redis_host}")
                else:
                    print(f"No password found for {self.redis_host}")
            else:
                print(f"No authentication required for localhost connection")
            
            self.redis_client = redis.Redis(**redis_params)
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
        
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _read_redis_password(self):
        """Read Redis password from .redis_passwd file"""
        try:
            with open('.redis_passwd', 'r') as f:
                password = f.read().strip()
            if password:
                return password
            else:
                print("Warning: .redis_passwd file is empty")
                return None
        except FileNotFoundError:
            print("Warning: .redis_passwd file not found")
            return None
        except Exception as e:
            print(f"Error reading .redis_passwd file: {e}")
            return None
    
    def _signal_handler(self, signum, frame):
        print(f"\nShutting down subscriber...")
        self.running = False
    
    def _safe_float_convert(self, value):
        """Safely convert string values to float, handling NumPy format"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            if isinstance(value, str):
                if value.startswith('np.float64(') and value.endswith(')'):
                    number_str = value[11:-1]
                    return float(number_str)
                elif value.startswith('np.float32(') and value.endswith(')'):
                    number_str = value[11:-1]
                    return float(number_str)
                else:
                    return float(value)
            
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _display_analysis(self, analysis):
        """Display formatted trading analysis"""
        try:
            print("\n" + "="*70)
            print("LIVE TRADING ANALYSIS")
            print("="*70)
            print(f"Symbol: {analysis['symbol']}")
            print(f"Time: {datetime.fromisoformat(analysis['timestamp']).strftime('%H:%M:%S')}")
            
            current_price = self._safe_float_convert(analysis['current_price'])
            high_price = self._safe_float_convert(analysis['high_price'])
            low_price = self._safe_float_convert(analysis['low_price'])
            high_distance = self._safe_float_convert(analysis.get('high_distance_pct', 0))
            low_distance = self._safe_float_convert(analysis.get('low_distance_pct', 0))
            
            print(f"Current Price: ${current_price:.4f}")
            print(f"High (30-min): ${high_price:.4f} ({high_distance:+.1f}%)")
            print(f"Low (30-min): ${low_price:.4f} ({low_distance:+.1f}%)")
            
            print(f"\nRecommended Action: {analysis['recommended_action']}")
            
            confidence_score = self._safe_float_convert(analysis['confidence_score'])
            print(f"Confidence: {analysis['confidence_level']} ({confidence_score:.1%})")
            print(f"Signal Type: {analysis['signal_type']}")
            print(f"Guidance: {analysis['guidance']}")
            
            if analysis['actionable'] == 'true':
                print("ACTIONABLE SIGNAL")
            else:
                print("MONITOR ONLY")
            
            hold_prob = self._safe_float_convert(analysis['hold_prob'])
            buy_prob = self._safe_float_convert(analysis['buy_prob'])
            sell_prob = self._safe_float_convert(analysis['sell_prob'])
            
            print(f"\nPrediction Breakdown:")
            print(f"   HOLD: {hold_prob:.1%}")
            print(f"   BUY:  {buy_prob:.1%}")
            print(f"   SELL: {sell_prob:.1%}")
            
            if analysis.get('entry_levels') and analysis['actionable'] == 'true':
                try:
                    entry_levels = json.loads(analysis['entry_levels'])
                    if entry_levels:
                        if 'buy_aggressive' in entry_levels:
                            print(f"\nBUY ENTRY LEVELS:")
                            print(f"   Aggressive: ${self._safe_float_convert(entry_levels.get('buy_aggressive', 0)):.4f}")
                            print(f"   Conservative: ${self._safe_float_convert(entry_levels.get('buy_conservative', 0)):.4f}")
                            print(f"   Target 1: ${self._safe_float_convert(entry_levels.get('target_1', 0)):.4f}")
                            print(f"   Stop Loss: ${self._safe_float_convert(entry_levels.get('stop_loss', 0)):.4f}")
                        elif 'sell_aggressive' in entry_levels:
                            print(f"\nSELL ENTRY LEVELS:")
                            print(f"   Aggressive: ${self._safe_float_convert(entry_levels.get('sell_aggressive', 0)):.4f}")
                            print(f"   Conservative: ${self._safe_float_convert(entry_levels.get('sell_conservative', 0)):.4f}")
                            print(f"   Target 1: ${self._safe_float_convert(entry_levels.get('target_1', 0)):.4f}")
                            print(f"   Stop Loss: ${self._safe_float_convert(entry_levels.get('stop_loss', 0)):.4f}")
                except json.JSONDecodeError:
                    pass
            
            print(f"\nAnalysis Details:")
            print(f"   Data Points: {analysis.get('data_points_used', 'N/A')} ({analysis.get('analysis_window', '30_minutes')})")
            print(f"   Agent ID: {analysis.get('agent_id', 'N/A')}")
            print("="*70)
            
        except Exception as e:
            print(f"Error displaying analysis: {e}")
    
    def start_listening(self):
        """Start listening for trading analysis"""
        print(f"Starting Analysis Subscriber for {self.symbol}")
        print(f"Stream: {self.stream_name}")
        print(f"Consumer: {self.consumer_name}")
        print(f"Redis: {self.redis_host}:{self.redis_port}")
        
        self.running = True
        
        try:
            try:
                self.redis_client.xgroup_create(self.stream_name, self.group_name, id='$', mkstream=True)
                print(f"Created consumer group: {self.group_name}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    print(f"Consumer group exists: {self.group_name}")
                elif "NOGROUP" in str(e):
                    print(f"Stream doesn't exist yet - waiting for trading agent...")
                else:
                    print(f"Error creating consumer group: {e}")
            
            print("Waiting for live trading analysis...\n")
            
            message_count = 0
            
            while self.running:
                try:
                    print("Waiting for analysis...", end='\r')
                    
                    messages = self.redis_client.xreadgroup(
                        self.group_name,
                        self.consumer_name,
                        {self.stream_name: '>'},
                        count=1,
                        block=2000
                    )
                    
                    if messages:
                        for stream, stream_messages in messages:
                            for message in stream_messages:
                                message_id, data = message
                                message_count += 1
                                
                                if data.get('analysis_type') == 'TRADING_RECOMMENDATION':
                                    print(f"\nAnalysis #{message_count} received")
                                    self._display_analysis(data)
                                
                                self.redis_client.xack(self.stream_name, self.group_name, message_id)
                
                except redis.ResponseError as e:
                    if "NOGROUP" in str(e):
                        print("\nStream not ready yet, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        print(f"\nRedis error: {e}")
                        break
                except redis.ConnectionError:
                    print("\nLost Redis connection, reconnecting...")
                    time.sleep(5)
                except Exception as e:
                    print(f"\nError: {e}")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nSubscriber stopped")
        finally:
            self.running = False
            print(f"\nSession Summary:")
            print(f"   Total Messages: {message_count}")
            print(f"   Stream: {self.stream_name}")

def main():
    parser = argparse.ArgumentParser(description='Trading analysis stream subscriber')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., GERN)')
    parser.add_argument('--redis-host', type=str, default='trader.wolfx0.com', help='Redis server host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis server port')
    parser.add_argument('--stream', type=str, help='Custom stream name')
    parser.add_argument('--consumer', type=str, help='Custom consumer name')
    args = parser.parse_args()
    
    subscriber = TradingAnalysisSubscriber(
        symbol=args.symbol,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        stream_name=args.stream,
        consumer_name=args.consumer
    )
    subscriber.start_listening()

if __name__ == "__main__":
    main()

