#!/usr/bin/env python3

import redis
import pandas as pd
import numpy as np
import time
import sys
import os
import argparse
from datetime import datetime, timedelta
import json

# Add the optimization path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization'))

from data_manager import FixedDataManager

class TradingDataPublisher:
    def __init__(self, symbol, stream_name=None, source='alpha'):
        self.symbol = symbol.upper()
        self.source = source.upper()
        self.stream_name = stream_name or f"trading_stream_{self.symbol}"
        
        # Connect to Redis
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("✅ Connected to Redis successfully")
        except redis.ConnectionError:
            print("❌ Failed to connect to Redis. Make sure Redis is running.")
            sys.exit(1)
        
        # Initialize data manager
        self.data_manager = FixedDataManager(f"S_{self.symbol}_{self.source}")
        
    def load_trading_day(self, target_date=None):
        """Load one day of trading data"""
        try:
            available_days = self.data_manager.available_days
            
            if target_date:
                if target_date in available_days:
                    selected_day = target_date
                else:
                    print(f"❌ Date {target_date} not available")
                    return None
            else:
                # Use the most recent day
                selected_day = available_days[-1]
            
            print(f"📅 Loading data for {selected_day}")
            # Use load_multiple_days with a single day list
            data = self.data_manager.load_multiple_days([selected_day])
            
            if data.empty:
                print(f"❌ No data available for {selected_day}")
                return None
                
            print(f"✅ Loaded {len(data)} data points for {selected_day}")
            return data, selected_day
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _convert_timestamp(self, timestamp):
        """Convert timestamp to proper datetime object"""
        try:
            # If it's already a datetime object
            if isinstance(timestamp, pd.Timestamp):
                return timestamp.to_pydatetime()
            elif isinstance(timestamp, datetime):
                return timestamp
            # If it's a string, try to parse it
            elif isinstance(timestamp, str):
                return pd.to_datetime(timestamp).to_pydatetime()
            # If it's an integer (Unix timestamp)
            elif isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp)
            else:
                # Fallback: try to convert using pandas
                return pd.to_datetime(timestamp).to_pydatetime()
        except Exception as e:
            print(f"⚠️ Error converting timestamp {timestamp}: {e}")
            return datetime.now()
    
    def publish_market_data(self, data, trading_date, publish_interval=5):
        """Publish market data to Redis stream with specified interval"""
        if data is None or data.empty:
            print("❌ No data to publish")
            return
        
        print(f"🚀 Starting to publish {len(data)} data points")
        print(f"⏱️ Publishing interval: {publish_interval} seconds per minute of data")
        print(f"📊 Estimated total time: {len(data) * publish_interval / 60:.1f} minutes")
        
        published_count = 0
        start_time = time.time()
        
        try:
            for idx, (timestamp, row) in enumerate(data.iterrows()):
                # Convert timestamp to proper datetime object
                dt_timestamp = self._convert_timestamp(timestamp)
                
                # Prepare market data message
                market_data = {
                    'symbol': self.symbol,
                    'timestamp': dt_timestamp.isoformat(),  # Now safe to call isoformat()
                    'trading_date': trading_date,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                    'sequence_number': idx + 1,
                    'total_points': len(data),
                    'published_at': datetime.now().isoformat()
                }
                
                # Add to Redis stream
                message_id = self.redis_client.xadd(self.stream_name, market_data)
                published_count += 1
                
                # Progress reporting
                if published_count % 50 == 0:
                    elapsed = time.time() - start_time
                    progress = (published_count / len(data)) * 100
                    eta = (elapsed / published_count) * (len(data) - published_count)
                    print(f"📈 Published {published_count}/{len(data)} ({progress:.1f}%) - ETA: {eta/60:.1f} min")
                
                # Sleep to simulate real-time data feed
                time.sleep(publish_interval)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Publishing interrupted by user")
        except Exception as e:
            print(f"❌ Error during publishing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            elapsed = time.time() - start_time
            print(f"\n✅ Publishing complete!")
            print(f"📊 Published {published_count} data points in {elapsed/60:.1f} minutes")
            print(f"🎯 Stream: {self.stream_name}")
    
    def get_stream_info(self):
        """Get information about the current stream"""
        try:
            info = self.redis_client.xinfo_stream(self.stream_name)
            print(f"📊 Stream Info for {self.stream_name}:")
            print(f"   Length: {info['length']} messages")
            print(f"   Consumer Groups: {info['groups']}")
            print(f"   First Entry: {info['first-entry']}")
            print(f"   Last Entry: {info['last-entry']}")
        except redis.ResponseError:
            print(f"❌ Stream {self.stream_name} does not exist")

def main():
    parser = argparse.ArgumentParser(description='Publish trading data to Redis stream')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., GERN)')
    parser.add_argument('--stream', type=str, help='Custom stream name')
    parser.add_argument('--source', type=str, default='alpha', choices=['alpha', 'yahoo'])
    parser.add_argument('--date', type=str, help='Specific date to publish (YYYY-MM-DD)')
    parser.add_argument('--interval', type=int, default=5, help='Publish interval in seconds (default: 5)')
    parser.add_argument('--info', action='store_true', help='Show stream info only')
    args = parser.parse_args()
    
    # Initialize publisher
    publisher = TradingDataPublisher(args.symbol, args.stream, args.source)
    
    if args.info:
        publisher.get_stream_info()
        return
    
    # Load and publish data
    result = publisher.load_trading_day(args.date)
    if result:
        data, trading_date = result
        publisher.publish_market_data(data, trading_date, args.interval)
    else:
        print("❌ Failed to load trading data")

if __name__ == "__main__":
    main()
