#!/usr/bin/env python3
"""
WolfXE Simulation Preparation Tool - 40 Days Training, 20 Days Testing
Automates data preparation, training, and testing for any ticker
"""

import sys
import argparse
import yfinance as yf
import pandas as pd
import redis
import json
from datetime import datetime, timedelta
from pathlib import Path
import time

# FIXED: Remove the incorrect import path
# The simulator is self-contained and doesn't need the production trader
# from rw_wolfxe.production.trader_wolfxe import WolfXEnhancedTradingSystem

class SimulationPreparator:
    """Automates complete simulation preparation process with 40/20 split"""
    
    def __init__(self, redis_host='trader.wolfx0.com', redis_port=6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.batch_size = 100
        
    def connect_redis(self):
        """Connect to Redis with password from file"""
        try:
            with open('.redis_passwd', 'r') as f:
                password = f.read().strip()
            
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=password,
                decode_responses=True,
                socket_timeout=10
            )
            
            pong = self.redis_client.ping()
            print(f"Redis connection successful: {pong}")
            return True
            
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return False
    
    def check_existing_simulation(self, symbol):
        """Check if simulation data already exists"""
        sim_symbol = f"S_{symbol.upper()}"
        pattern = f"sim:{sim_symbol}:*"
        
        try:
            keys = self.redis_client.keys(pattern)
            
            if keys:
                print(f"SIMULATION DATA EXISTS FOR {sim_symbol}")
                print(f"   Found {len(keys)} days of data")
                
                dates = []
                for key in keys:
                    parts = key.split(':')
                    if len(parts) >= 3:
                        dates.append(parts[2])
                
                if dates:
                    dates.sort()
                    print(f"   Date range: {dates[0]} to {dates[-1]}")
                
                return True, len(keys)
            else:
                print(f"NO SIMULATION DATA FOUND FOR {sim_symbol}")
                return False, 0
                
        except Exception as e:
            print(f"Error checking existing data: {e}")
            return False, 0
    
    def validate_ticker(self, symbol):
        """Validate if ticker exists and has recent data"""
        print(f"VALIDATING TICKER {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get recent daily data first
            daily_data = ticker.history(period="5d", interval="1d")
            
            if daily_data.empty:
                print(f"No recent daily data found for {symbol}")
                return False
            
            # Check if 5-minute data is available
            try:
                intraday_data = ticker.history(period="7d", interval="5m")
                if intraday_data.empty:
                    print(f"No 5-minute intraday data available for {symbol}")
                    return "daily_only"
                else:
                    print(f"Ticker {symbol} validated - intraday data available")
                    return True
            except:
                print(f"5-minute data not available for {symbol}")
                return "daily_only"
                
        except Exception as e:
            print(f"Ticker validation failed: {e}")
            return False
    
    def download_ticker_data(self, symbol, days=60):
        """Download ticker data with Yahoo Finance limitations handled"""
        print(f"DOWNLOADING {days} DAYS OF DATA FOR {symbol}")
        print("=" * 60)
        
        validation = self.validate_ticker(symbol)
        if validation == False:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            
            if validation == "daily_only":
                print(f"   Using daily data (5-minute not available)")
                df = ticker.history(period=f"{days}d", interval="1d")
                
                if df.empty:
                    print("No daily data available")
                    return None
                
                # Convert daily to simulated 5-minute data
                df = self.simulate_intraday_from_daily(df)
                
            else:
                # Use 5-minute data (limited to last 60 days by Yahoo)
                max_days = min(days, 59)
                print(f"   Using 5-minute data (limited to {max_days} days)")
                
                df = ticker.history(period=f"{max_days}d", interval="5m")
                
                if df.empty:
                    print("No 5-minute data available")
                    return None
            
            df = df.dropna()
            
            if len(df) < 100:
                print(f"Insufficient data: only {len(df)} points")
                return None
            
            print(f"Downloaded {len(df)} data points")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"Download failed: {e}")
            return None
    
    def simulate_intraday_from_daily(self, daily_df):
        """Convert daily data to simulated 5-minute intervals"""
        print("Simulating 5-minute data from daily data...")
        
        intraday_data = []
        
        for date, row in daily_df.iterrows():
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            volume = row['Volume']
            
            # Generate 78 5-minute intervals for each trading day
            for i in range(78):
                time_progress = i / 77.0
                
                import random
                random.seed(int(date.timestamp()) + i)
                
                base_price = open_price + (close_price - open_price) * time_progress
                price_range = high_price - low_price
                noise = (random.random() - 0.5) * price_range * 0.3
                
                current_price = base_price + noise
                current_price = max(low_price, min(high_price, current_price))
                
                market_open = date.replace(hour=9, minute=30)
                interval_time = market_open + timedelta(minutes=i * 5)
                
                volume_factor = 1.5 if i < 10 or i > 67 else 1.0
                interval_volume = int(volume / 78 * volume_factor)
                
                intraday_data.append({
                    'timestamp': interval_time,
                    'Open': current_price,
                    'High': current_price * 1.001,
                    'Low': current_price * 0.999,
                    'Close': current_price,
                    'Volume': interval_volume
                })
        
        df = pd.DataFrame(intraday_data)
        df.set_index('timestamp', inplace=True)
        
        print(f"Generated {len(df)} simulated 5-minute intervals")
        return df
    
    def convert_to_timesale_format(self, df, symbol):
        """Convert data to timesale format"""
        print("Converting to timesale format...")
        
        timesale_data = []
        
        for timestamp, row in df.iterrows():
            entry = {
                'timestamp': timestamp,
                'price': row['Close'],
                'volume': int(row['Volume']),
                'bid': row['Low'],
                'ask': row['High'],
                'symbol': f"S_{symbol.upper()}",
                'trading_date': timestamp.strftime('%Y-%m-%d')
            }
            timesale_data.append(entry)
        
        print(f"Converted {len(timesale_data)} entries to timesale format")
        return timesale_data
    
    def upload_to_redis(self, timesale_data, symbol):
        """Upload timesale data to Redis streams"""
        print(f"UPLOADING DATA TO REDIS FOR S_{symbol}")
        print("=" * 60)
        
        sim_symbol = f"S_{symbol.upper()}"
        
        # Group data by trading day
        daily_data = {}
        for entry in timesale_data:
            trading_date = entry['trading_date']
            
            if trading_date not in daily_data:
                daily_data[trading_date] = []
            daily_data[trading_date].append(entry)
        
        print(f"   Grouped into {len(daily_data)} trading days")
        
        # Upload data day by day
        uploaded_days = 0
        total_messages = 0
        
        for trading_date, day_entries in daily_data.items():
            stream_name = f"sim:{sim_symbol}:{trading_date}"
            
            try:
                # Upload entire day in one pipeline
                pipe = self.redis_client.pipeline()
                
                for entry in day_entries:
                    # Convert to Redis stream format
                    stream_entry = {
                        'timestamp': entry['timestamp'].isoformat(),
                        'price': str(entry['price']),
                        'volume': str(entry['volume']),
                        'bid': str(entry['bid']),
                        'ask': str(entry['ask']),
                        'symbol': entry['symbol'],
                        'trading_date': entry['trading_date']
                    }
                    pipe.xadd(stream_name, stream_entry)
                
                # Execute all at once
                pipe.execute()
                
                total_messages += len(day_entries)
                uploaded_days += 1
                
                print(f"   Day {uploaded_days}: {len(day_entries)} messages uploaded")
                
            except Exception as e:
                print(f"   Error uploading {trading_date}: {e}")
        
        print(f"Upload complete: {uploaded_days} days, {total_messages} messages")
        return uploaded_days
    
    def prepare_ticker(self, symbol, force_download=False):
        """Complete preparation process for a ticker"""
        symbol = symbol.upper()
        
        print(f"PREPARING SIMULATION FOR {symbol}")
        print("=" * 80)
        
        # Step 1: Check existing data
        exists, day_count = self.check_existing_simulation(symbol)
        
        if exists and not force_download:
            print(f"   Simulation data already exists ({day_count} days)")
            response = input("   Proceed anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("   Skipping preparation")
                return False
        
        # Step 2: Download data
        df = self.download_ticker_data(symbol, days=60)
        if df is None:
            return False
        
        # Step 3: Convert to timesale format
        timesale_data = self.convert_to_timesale_format(df, symbol)
        
        # Step 4: Upload to Redis
        uploaded_days = self.upload_to_redis(timesale_data, symbol)
        if uploaded_days == 0:
            return False
        
        print(f"\nPREPARATION COMPLETE FOR {symbol}")
        print("=" * 80)
        print("Data downloaded and uploaded")
        print(f"Ready to run: python main.py --symbol {symbol} --budget 25000")
        
        return True

def main():
    """Main preparation function"""
    parser = argparse.ArgumentParser(description='Prepare simulation data for any ticker')
    parser.add_argument('symbol', help='Stock symbol to prepare (e.g., AAPL, TSLA, MSFT)')
    parser.add_argument('--force', action='store_true', help='Force re-download even if data exists')
    parser.add_argument('--redis-host', default='trader.wolfx0.com', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    # Initialize preparator
    prep = SimulationPreparator(args.redis_host, args.redis_port)
    
    # Connect to Redis
    if not prep.connect_redis():
        print("Failed to connect to Redis")
        return False
    
    # Prepare the ticker
    success = prep.prepare_ticker(args.symbol, args.force)
    
    if success:
        print(f"\n{args.symbol} is ready for simulation!")
    else:
        print(f"\nFailed to prepare {args.symbol}")
    
    return success

if __name__ == "__main__":
    main()

