#!/usr/bin/env python3

import redis
import json
import argparse
from datetime import datetime
import signal
import sys
import time

class TradingAnalysisSubscriber:
    def __init__(self, symbol, redis_host='localhost', stream_name=None, consumer_name=None):
        self.symbol = symbol.upper()
        self.stream_name = stream_name or f"trading_analysis_{self.symbol}"
        self.consumer_name = consumer_name or f"test_subscriber_{int(datetime.now().timestamp())}"
        self.group_name = "test_subscribers"
        self.redis_host = redis_host
        self.running = False
        
        # Redis connection
        try:
            self.redis_client = redis.Redis(host=self.redis_host, port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("✅ Connected to Redis successfully")
        except redis.ConnectionError:
            print("❌ Failed to connect to Redis")
            sys.exit(1)
        
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n🛑 Shutting down subscriber...")
        self.running = False
    
    def _safe_float_convert(self, value):
        """Safely convert string values to float, handling NumPy format"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            # Handle NumPy format like 'np.float64(1.54)'
            if isinstance(value, str):
                if value.startswith('np.float64(') and value.endswith(')'):
                    # Extract the number from np.float64(1.54)
                    number_str = value[11:-1]  # Remove 'np.float64(' and ')'
                    return float(number_str)
                elif value.startswith('np.float32(') and value.endswith(')'):
                    # Extract the number from np.float32(1.54)
                    number_str = value[11:-1]  # Remove 'np.float32(' and ')'
                    return float(number_str)
                elif value.startswith('np.int64(') and value.endswith(')'):
                    # Extract the number from np.int64(123)
                    number_str = value[9:-1]  # Remove 'np.int64(' and ')'
                    return float(number_str)
                elif value.startswith('np.int32(') and value.endswith(')'):
                    # Extract the number from np.int32(123)
                    number_str = value[9:-1]  # Remove 'np.int32(' and ')'
                    return float(number_str)
                else:
                    return float(value)
            
            return float(value)
        except (ValueError, TypeError) as e:
            print(f"⚠️ Error converting {value} to float: {e}")
            return 0.0
    
    def _safe_int_convert(self, value):
        """Safely convert string values to int"""
        try:
            if isinstance(value, int):
                return value
            
            # Handle NumPy format
            if isinstance(value, str):
                if value.startswith('np.int64(') and value.endswith(')'):
                    number_str = value[9:-1]
                    return int(number_str)
                elif value.startswith('np.int32(') and value.endswith(')'):
                    number_str = value[9:-1]
                    return int(number_str)
            
            return int(float(str(value)))
        except (ValueError, TypeError):
            return 0
    
    def _display_analysis(self, analysis):
        """Display formatted trading analysis with robust error handling"""
        try:
            print("\n" + "="*70)
            print("🤖 LIVE TRADING ANALYSIS")
            print("="*70)
            print(f"📊 Symbol: {analysis['symbol']}")
            print(f"⏰ Time: {datetime.fromisoformat(analysis['timestamp']).strftime('%H:%M:%S')}")
            
            # FIXED: Use safe conversion for all numeric values
            current_price = self._safe_float_convert(analysis['current_price'])
            high_price = self._safe_float_convert(analysis['high_price'])
            low_price = self._safe_float_convert(analysis['low_price'])
            high_distance = self._safe_float_convert(analysis['high_distance_pct'])
            low_distance = self._safe_float_convert(analysis['low_distance_pct'])
            
            print(f"💰 Current Price: ${current_price:.4f}")
            print(f"📈 High (30-min): ${high_price:.4f} ({high_distance:+.1f}%)")
            print(f"📉 Low (30-min): ${low_price:.4f} ({low_distance:+.1f}%)")
            
            print(f"\n🎯 Recommended Action: **{analysis['recommended_action']}**")
            
            confidence_score = self._safe_float_convert(analysis['confidence_score'])
            print(f"📈 Confidence: {analysis['confidence_emoji']} {analysis['confidence_level']} ({confidence_score:.1%})")
            print(f"🔍 Signal Type: {analysis['signal_type']}")
            print(f"💡 Guidance: {analysis['guidance']}")
            
            if analysis['actionable'] == 'true':
                print("✅ **ACTIONABLE SIGNAL**")
            else:
                print("⚠️ **MONITOR ONLY**")
            
            # FIXED: Safe conversion for probabilities
            hold_prob = self._safe_float_convert(analysis['hold_prob'])
            buy_prob = self._safe_float_convert(analysis['buy_prob'])
            sell_prob = self._safe_float_convert(analysis['sell_prob'])
            
            print(f"\n📊 Prediction Breakdown:")
            print(f"   • HOLD: {hold_prob:.1%}")
            print(f"   • BUY:  {buy_prob:.1%}")
            print(f"   • SELL: {sell_prob:.1%}")
            
            # Display entry levels if available
            if analysis.get('entry_levels') and analysis['actionable'] == 'true':
                try:
                    entry_levels = json.loads(analysis['entry_levels'])
                    if entry_levels:
                        if 'buy_aggressive' in entry_levels:
                            print(f"\n💰 BUY ENTRY LEVELS:")
                            print(f"   🟢 Aggressive: ${self._safe_float_convert(entry_levels.get('buy_aggressive', 0)):.4f}")
                            print(f"   🟡 Conservative: ${self._safe_float_convert(entry_levels.get('buy_conservative', 0)):.4f}")
                            print(f"   🔵 Dip Buy: ${self._safe_float_convert(entry_levels.get('buy_dip', 0)):.4f}")
                            print(f"\n🎯 PROFIT TARGETS:")
                            print(f"   🥇 Target 1: ${self._safe_float_convert(entry_levels.get('target_1', 0)):.4f}")
                            print(f"   🥈 Target 2: ${self._safe_float_convert(entry_levels.get('target_2', 0)):.4f}")
                            print(f"   🛑 Stop Loss: ${self._safe_float_convert(entry_levels.get('stop_loss', 0)):.4f}")
                        elif 'sell_aggressive' in entry_levels:
                            print(f"\n💰 SELL ENTRY LEVELS:")
                            print(f"   🔴 Aggressive: ${self._safe_float_convert(entry_levels.get('sell_aggressive', 0)):.4f}")
                            print(f"   🟡 Conservative: ${self._safe_float_convert(entry_levels.get('sell_conservative', 0)):.4f}")
                            print(f"   🔵 Bounce Sell: ${self._safe_float_convert(entry_levels.get('sell_bounce', 0)):.4f}")
                            print(f"\n🎯 PROFIT TARGETS:")
                            print(f"   🥇 Target 1: ${self._safe_float_convert(entry_levels.get('target_1', 0)):.4f}")
                            print(f"   🥈 Target 2: ${self._safe_float_convert(entry_levels.get('target_2', 0)):.4f}")
                            print(f"   🛑 Stop Loss: ${self._safe_float_convert(entry_levels.get('stop_loss', 0)):.4f}")
                except json.JSONDecodeError:
                    pass
            
            # Additional analysis context
            data_points = self._safe_int_convert(analysis['data_points_used'])
            print(f"\n📈 Analysis Details:")
            print(f"   • Data Points: {data_points} ({analysis['analysis_window']})")
            print(f"   • Near High: {'Yes' if analysis.get('near_high') == 'true' else 'No'}")
            print(f"   • Near Low: {'Yes' if analysis.get('near_low') == 'true' else 'No'}")
            print(f"   • High Confidence: {'Yes' if analysis.get('high_confidence') == 'true' else 'No'}")
            print(f"   • Agent ID: {analysis['agent_id']}")
            print(f"   • Model Version: {analysis.get('model_version', 'N/A')}")
            print("="*70)
            
        except Exception as e:
            print(f"⚠️ Error displaying analysis: {e}")
            print(f"📄 Raw data keys: {list(analysis.keys())}")
            print(f"📄 Problematic fields:")
            for key, value in analysis.items():
                if isinstance(value, str) and ('np.' in value or 'float' in value):
                    print(f"   • {key}: {value}")
    
    def _display_stream_stats(self):
        """Display stream statistics"""
        try:
            info = self.redis_client.xinfo_stream(self.stream_name)
            print(f"\n📊 Stream Statistics:")
            print(f"   • Stream: {self.stream_name}")
            print(f"   • Total Messages: {info.get('length', 0)}")
            print(f"   • Consumer Groups: {info.get('groups', 0)}")
            if info.get('last-entry'):
                last_entry = info['last-entry']
                print(f"   • Last Message ID: {last_entry[0]}")
        except redis.ResponseError:
            print(f"⚠️ Stream {self.stream_name} not found")
    
    def start_listening(self):
        """Start listening for trading analysis"""
        print(f"🎧 Starting Analysis Subscriber for {self.symbol}")
        print(f"📡 Stream: {self.stream_name}")
        print(f"👤 Consumer: {self.consumer_name}")
        print(f"👥 Group: {self.group_name}")
        
        self.running = True
        
        try:
            # Create consumer group
            try:
                self.redis_client.xgroup_create(self.stream_name, self.group_name, id='$', mkstream=True)
                print(f"👥 Created consumer group: {self.group_name}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    print(f"👥 Consumer group exists: {self.group_name}")
                elif "NOGROUP" in str(e):
                    print(f"⚠️ Stream doesn't exist yet - waiting for trading agent...")
                else:
                    print(f"⚠️ Error creating consumer group: {e}")
            
            # Display initial stream stats
            self._display_stream_stats()
            
            print("🔔 Waiting for live trading analysis...\n")
            
            message_count = 0
            last_message_time = None
            
            while self.running:
                try:
                    print("⏳ Waiting for analysis...", end='\r')
                    
                    messages = self.redis_client.xreadgroup(
                        self.group_name,
                        self.consumer_name,
                        {self.stream_name: '>'},
                        count=1,
                        block=2000  # Block for 2 seconds
                    )
                    
                    if messages:
                        for stream, stream_messages in messages:
                            for message in stream_messages:
                                message_id, data = message
                                message_count += 1
                                last_message_time = datetime.now()
                                
                                if data.get('analysis_type') == 'TRADING_RECOMMENDATION':
                                    print(f"\n📨 Message #{message_count} received at {last_message_time.strftime('%H:%M:%S')}")
                                    self._display_analysis(data)
                                else:
                                    print(f"\n📨 Non-analysis message received: {data.get('analysis_type', 'Unknown')}")
                                
                                # Acknowledge message
                                self.redis_client.xack(self.stream_name, self.group_name, message_id)
                    else:
                        # Show periodic status updates
                        current_time = datetime.now()
                        if last_message_time:
                            time_since_last = (current_time - last_message_time).total_seconds()
                            if time_since_last > 60:  # Show status every minute
                                print(f"\n⏰ No messages for {time_since_last:.0f} seconds. Total received: {message_count}")
                                last_message_time = current_time
                
                except redis.ResponseError as e:
                    if "NOGROUP" in str(e):
                        print("\n⚠️ Stream not ready yet, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        print(f"\n❌ Redis error: {e}")
                        break
                except redis.ConnectionError:
                    print("\n❌ Lost Redis connection, reconnecting...")
                    time.sleep(5)
                except Exception as e:
                    print(f"\n⚠️ Unexpected error: {e}")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n🛑 Subscriber stopped by user")
        finally:
            self.running = False
            print(f"\n📊 Session Summary:")
            print(f"   • Total Messages Processed: {message_count}")
            print(f"   • Stream: {self.stream_name}")
            print(f"   • Consumer: {self.consumer_name}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced trading analysis stream subscriber')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., GERN)')
    parser.add_argument('--stream', type=str, help='Custom stream name')
    parser.add_argument('--consumer', type=str, help='Custom consumer name')
    parser.add_argument('--redis-host', type=str, default='localhost', help='Redis server host')
    args = parser.parse_args()
    
    subscriber = TradingAnalysisSubscriber(args.symbol, args.redis_host, args.stream, args.consumer)
    subscriber.start_listening()

if __name__ == "__main__":
    main()

