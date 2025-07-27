#!/usr/bin/env python3

import redis
import pandas as pd
import time
import sys
import os
import argparse
from datetime import datetime
import json

# Add the optimization path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization'))

from data_manager import FixedDataManager

class TradingDataPublisher:
    def __init__(self, symbol, stream_name=None):
        self.symbol = symbol.upper()
        self.stream_name = stream_name or f"trading_stream_{self.symbol}"
        
        # Redis connection with authentication
        base_dir = os.path.dirname(os.path.abspath(__file__))
        passwd_file = os.path.join(base_dir, ".redis_passwd")
        
        try:
            with open(passwd_file, "r") as f:
                redis_password = f.read().strip()
            
            self.redis_client = redis.Redis(
                host="trader.wolfx0.com",
                port=6379,
                db=0,
                password=redis_password,
                decode_responses=True,
            )
            self.redis_client.ping()
            print("✅ Connected to Redis successfully")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            sys.exit(1)
        
        # Initialize data manager for Alpha Vantage
        self.data_manager = FixedDataManager(self.symbol)

    def load_trading_day(self, offset_days=0, target_date=None):
        """Load trading day data from Alpha Vantage"""
        try:
            # For Alpha Vantage, we get the latest available data
            data, trading_date = self.data_manager.get_latest_trading_day()
            
            if data is None or data.empty:
                print("❌ No data available from Alpha Vantage")
                return None
            
            print(f"📅 Loaded trading data for {trading_date}")
            print(f"📊 Data points: {len(data)}")
            print(f"⏰ Time range: {data.index.min()} to {data.index.max()}")
            
            return data, trading_date
            
        except Exception as e:
            print(f"❌ Error loading trading data: {e}")
            return None

    def _convert_timestamp(self, ts):
        """Convert timestamp to datetime object"""
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
        return pd.to_datetime(ts).to_pydatetime()

    def publish_market_data(self, data, trading_date, publish_interval=5):
        """Publish market data to Redis stream"""
        total = len(data)
        start_time = time.time()
        
        print(f"🚀 Starting to publish {total} data points")
        print(f"⏱️ Publish interval: {publish_interval} seconds")
        print(f"📡 Target stream: {self.stream_name}")
        
        for idx, (ts, row) in enumerate(data.iterrows(), 1):
            # Create message for Redis stream
            message = {
                "symbol": self.symbol,
                "timestamp": self._convert_timestamp(ts).isoformat(),
                "trading_date": trading_date,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "sequence_number": str(idx),
                "total_points": str(total),
                "published_at": datetime.now().isoformat(),
            }
            
            # Publish to Redis stream
            try:
                message_id = self.redis_client.xadd(self.stream_name, message)
                
                # Progress reporting
                if idx % 50 == 0 or idx == total:
                    elapsed = time.time() - start_time
                    progress = (idx / total) * 100
                    
                    if idx < total:
                        eta_seconds = (elapsed / idx) * (total - idx)
                        eta_minutes = eta_seconds / 60
                        print(f"📈 Published {idx}/{total} ({progress:.1f}%) - ETA: {eta_minutes:.1f} min")
                    else:
                        print(f"✅ Published {idx}/{total} ({progress:.1f}%) - Complete!")
                
            except Exception as e:
                print(f"❌ Error publishing message {idx}: {e}")
                continue
            
            # Wait before next publish
            if idx < total:  # Don't sleep after the last message
                time.sleep(publish_interval)
        
        elapsed_total = time.time() - start_time
        print(f"🎉 Publishing complete! Total time: {elapsed_total/60:.1f} minutes")

    def get_stream_info(self):
        """Get information about the Redis stream"""
        try:
            info = self.redis_client.xinfo_stream(self.stream_name)
            print("📊 Stream Information:")
            print(json.dumps(info, indent=2, default=str))
        except redis.ResponseError:
            print(f"❌ Stream '{self.stream_name}' not found")

    def test_alpha_vantage_connection(self):
        """Test Alpha Vantage API connection"""
        try:
            print("🧪 Testing Alpha Vantage connection...")
            data, trading_date = self.data_manager.get_latest_trading_day()
            
            if data is not None and not data.empty:
                print(f"✅ Alpha Vantage connection successful")
                print(f"📅 Latest trading date: {trading_date}")
                print(f"📊 Sample data points: {len(data)}")
                print(f"💰 Latest price: ${data['close'].iloc[-1]:.4f}")
                return True
            else:
                print("❌ No data received from Alpha Vantage")
                return False
                
        except Exception as e:
            print(f"❌ Alpha Vantage connection failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Trading Data Publisher with Alpha Vantage')
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., GERN)")
    parser.add_argument("--stream", help="Custom stream name")
    parser.add_argument("-d", "--days", type=int, default=0, help="Days offset (not used with Alpha Vantage)")
    parser.add_argument("--date", help="Specific date (not used with Alpha Vantage)")
    parser.add_argument("--interval", type=int, default=5, help="Publish interval in seconds")
    parser.add_argument("--info", action="store_true", help="Show stream information")
    parser.add_argument("--test", action="store_true", help="Test Alpha Vantage connection")
    args = parser.parse_args()
    
    # Initialize publisher
    publisher = TradingDataPublisher(args.symbol, args.stream)
    
    # Handle different modes
    if args.info:
        publisher.get_stream_info()
        return
    
    if args.test:
        if publisher.test_alpha_vantage_connection():
            print("✅ System ready for data publishing")
        else:
            print("❌ System not ready - fix Alpha Vantage connection")
        return
    
    # Load and publish data
    print(f"📡 Loading latest trading data for {args.symbol}...")
    result = publisher.load_trading_day(offset_days=args.days, target_date=args.date)
    
    if result:
        data, trading_date = result
        print(f"🚀 Starting publication for {trading_date}")
        publisher.publish_market_data(data, trading_date, args.interval)
    else:
        print("❌ No data available to publish")

if __name__ == "__main__":
    main()

