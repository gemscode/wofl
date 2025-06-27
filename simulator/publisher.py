#!/usr/bin/env python3
"""
WolfXE Enhanced Simulation Publisher
Fetches Yahoo Finance data and publishes to Redis with daily stream separation
"""

import redis
import json
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import argparse
import sys
import os

class WolfXEEnhancedSimulationPublisher:
    """Enhanced simulation publisher with daily stream separation"""
    
    def __init__(self, redis_host='trader.wolfx0.com', redis_port=6379, 
                 redis_password=None, base_symbol='GERN', speed_multiplier=1.0,
                 pipeline_batch_size=50):
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=10,
            retry_on_timeout=True
        )
        
        # Configuration
        self.base_symbol = base_symbol.upper()
        self.sim_symbol = f"S_{self.base_symbol}"
        self.speed_multiplier = speed_multiplier
        self.pipeline_batch_size = pipeline_batch_size
        
        # Use the new simulator script SHA
        self.script_sha = "05c697d715f73983c6c81190e7a6691799c9dd85"
        
        # Data storage
        self.historical_data = None
        self.daily_data_groups = {}
        
        # Performance tracking
        self.published_count = 0
        self.failed_count = 0
        self.start_time = None
        
        print("=" * 80)
        print("WOLFXE ENHANCED SIMULATION PUBLISHER")
        print("=" * 80)
        print(f"BASE SYMBOL:       {self.base_symbol}")
        print(f"SIMULATION SYMBOL: {self.sim_symbol}")
        print(f"SPEED MULTIPLIER:  {self.speed_multiplier}x")
        print(f"REDIS ENDPOINT:    {redis_host}:{redis_port}")
        print(f"PIPELINE BATCH:    {self.pipeline_batch_size}")
        print(f"SCRIPT SHA:        {self.script_sha}")
        print("=" * 80)

    def test_redis_connection(self):
        """Test Redis connection"""
        try:
            pong = self.redis_client.ping()
            print(f"[CONNECTION] Redis ping successful: {pong}")
            return True
        except Exception as e:
            print(f"[ERROR] Redis connection failed: {e}")
            return False

    def fetch_yahoo_data(self, period="60d", interval="5m"):
        """Fetch intraday data from Yahoo Finance"""
        print(f"[DATA] Fetching {period} of {interval} data for {self.base_symbol}...")
        
        try:
            ticker = yf.Ticker(self.base_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"[ERROR] No data returned for {self.base_symbol}")
                return None
            
            data = data.reset_index()
            data = data.sort_values('Datetime')
            
            print(f"[DATA] Retrieved {len(data)} data points")
            print(f"[DATA] Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
            
            return data
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch Yahoo Finance data: {e}")
            return None

    def group_data_by_trading_days(self):
        """Group historical data by trading days"""
        if self.historical_data is None:
            return
        
        print("[DATA] Grouping data by trading days...")
        
        # Group by date
        self.historical_data['TradingDate'] = self.historical_data['Datetime'].dt.date
        grouped = self.historical_data.groupby('TradingDate')
        
        for trading_date, day_data in grouped:
            # Skip weekends
            if trading_date.weekday() >= 5:
                continue
            
            date_str = trading_date.strftime('%Y-%m-%d')
            self.daily_data_groups[date_str] = day_data.reset_index(drop=True)
            
        print(f"[DATA] Created {len(self.daily_data_groups)} trading day groups")
        
        # Show sample of trading days
        sample_days = list(self.daily_data_groups.keys())[:5]
        print(f"[DATA] Sample trading days: {sample_days}")

    def convert_to_timesale_format(self, row, trading_date):
        """Convert Yahoo Finance data row to timesale format with trading date"""
        close_price = float(row['Close'])
        open_price = float(row['Open'])
        high_price = float(row['High'])
        low_price = float(row['Low'])
        volume = int(row['Volume']) if row['Volume'] > 0 else 1000
        
        # Use original timestamp from Yahoo Finance
        base_timestamp = int(row['Datetime'].timestamp() * 1000)
        
        # Generate multiple ticks per bar
        ticks = []
        for i in range(5):
            progress = i / 4.0
            
            if i == 0:
                tick_price = open_price
            elif i == 4:
                tick_price = close_price
            else:
                price_range = close_price - open_price
                tick_price = open_price + (price_range * progress)
                tick_price = max(low_price, min(high_price, tick_price))
            
            spread_percent = 0.001
            spread = tick_price * spread_percent
            bid_price = tick_price - spread/2
            ask_price = tick_price + spread/2
            
            tick_volume = volume // 5 if i < 4 else volume - (volume // 5) * 4
            
            tick_data = {
                "type": "timesale",
                "symbol": self.sim_symbol,
                "exch": "SIM",
                "bid": f"{bid_price:.2f}",
                "ask": f"{ask_price:.2f}",
                "last": f"{tick_price:.2f}",
                "size": str(tick_volume),
                "date": str(base_timestamp + (i * 60000)),
                "seq": int((base_timestamp + i * 60000) % 1000000),
                "flag": "",
                "cancel": False,
                "correction": False,
                "session": "simulation",
                "trading_date": trading_date  # Add trading date
            }
            
            ticks.append(tick_data)
        
        return ticks

    def publish_daily_data(self, trading_date):
        """Publish all data for a specific trading day"""
        if trading_date not in self.daily_data_groups:
            print(f"[ERROR] No data found for trading date: {trading_date}")
            return False
        
        day_data = self.daily_data_groups[trading_date]
        print(f"[PUBLISH] Publishing {len(day_data)} bars for {trading_date}")
        
        tick_batch = []
        published_ticks = 0
        
        try:
            for index, row in day_data.iterrows():
                ticks = self.convert_to_timesale_format(row, trading_date)
                
                for tick_data in ticks:
                    tick_batch.append(tick_data)
                    
                    if len(tick_batch) >= self.pipeline_batch_size:
                        successful, failed = self.publish_data_batch_pipeline(tick_batch)
                        published_ticks += successful
                        tick_batch = []
            
            # Publish remaining ticks
            if tick_batch:
                successful, failed = self.publish_data_batch_pipeline(tick_batch)
                published_ticks += successful
            
            print(f"[PUBLISH] Completed {trading_date}: {published_ticks} ticks published")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to publish data for {trading_date}: {e}")
            return False

    def publish_data_batch_pipeline(self, tick_batch):
        """Publish batch of data points using Redis pipeline"""
        try:
            pipeline = self.redis_client.pipeline()
            
            for tick_data in tick_batch:
                pipeline.evalsha(
                    self.script_sha,
                    0,
                    json.dumps(tick_data)
                )
            
            results = pipeline.execute()
            
            successful = sum(1 for result in results if result is not None)
            failed = len(results) - successful
            
            self.published_count += successful
            self.failed_count += failed
            
            return successful, failed
            
        except Exception as e:
            print(f"[ERROR] Failed to publish batch: {e}")
            return 0, len(tick_batch)

    def publish_all_historical_data(self):
        """Publish all historical data organized by trading days"""
        if not self.daily_data_groups:
            print("[ERROR] No daily data groups available")
            return
        
        print(f"[PUBLISH] Publishing data for {len(self.daily_data_groups)} trading days")
        self.start_time = time.time()
        
        successful_days = 0
        for trading_date in sorted(self.daily_data_groups.keys()):
            if self.publish_daily_data(trading_date):
                successful_days += 1
        
        elapsed_time = time.time() - self.start_time
        rate = self.published_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n[COMPLETE] Historical data publishing finished!")
        print(f"[STATS] Trading days: {successful_days}/{len(self.daily_data_groups)}")
        print(f"[STATS] Total ticks: {self.published_count}")
        print(f"[STATS] Failed ticks: {self.failed_count}")
        print(f"[STATS] Duration: {elapsed_time:.1f} seconds")
        print(f"[STATS] Rate: {rate:.1f} ticks/second")

    def list_available_days(self):
        """List available trading days in Redis"""
        try:
            pattern = f"sim:{self.sim_symbol}:*"
            keys = self.redis_client.keys(pattern)
            
            trading_days = []
            for key in keys:
                # Extract date from key like "sim:S_GERN:2024-01-15"
                parts = key.split(':')
                if len(parts) >= 3:
                    trading_days.append(parts[2])
            
            trading_days.sort()
            
            print(f"[INFO] Available trading days for {self.sim_symbol}:")
            for day in trading_days:
                stream_name = f"sim:{self.sim_symbol}:{day}"
                length = self.redis_client.xlen(stream_name)
                print(f"  {day}: {length} messages")
            
            return trading_days
            
        except Exception as e:
            print(f"[ERROR] Failed to list available days: {e}")
            return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='WolfXE Enhanced Simulation Publisher',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--symbol', default='GERN', 
                       help='Base stock symbol to simulate')
    parser.add_argument('--period', default='60d', 
                       help='Yahoo Finance period')
    parser.add_argument('--interval', default='5m', 
                       help='Yahoo Finance interval')
    parser.add_argument('--redis-password', 
                       help='Redis authentication password')
    parser.add_argument('--redis-port', type=int, default=6379, 
                       help='Redis server port')
    parser.add_argument('--redis-host', default='trader.wolfx0.com',
                       help='Redis server hostname')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Pipeline batch size')
    parser.add_argument('--list-days', action='store_true',
                       help='List available trading days in Redis')
    
    args = parser.parse_args()
    
    # Load Redis password
    redis_password = args.redis_password
    if not redis_password:
        try:
            with open('.redis_passwd', 'r') as f:
                redis_password = f.read().strip()
        except FileNotFoundError:
            print("[ERROR] Redis password not provided and .redis_passwd file not found")
            sys.exit(1)
    
    # Create publisher
    publisher = WolfXEEnhancedSimulationPublisher(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=redis_password,
        base_symbol=args.symbol,
        pipeline_batch_size=args.batch_size
    )
    
    # Test connection
    if not publisher.test_redis_connection():
        sys.exit(1)
    
    # List days if requested
    if args.list_days:
        publisher.list_available_days()
        return
    
    # Fetch and publish data
    publisher.historical_data = publisher.fetch_yahoo_data(
        period=args.period, 
        interval=args.interval
    )
    
    if publisher.historical_data is None:
        sys.exit(1)
    
    # Group by trading days and publish
    publisher.group_data_by_trading_days()
    publisher.publish_all_historical_data()

if __name__ == "__main__":
    main()

