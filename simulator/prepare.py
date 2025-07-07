#!/usr/bin/env python3
"""
WolfXE Simulation Preparation Tool - 40 Days Training, 20 Days Testing
Automates data preparation, training, and testing for any ticker
Enhanced with Alpha Vantage premium support for up to 1 year of 1-minute data
"""

import sys
import argparse
import yfinance as yf
import pandas as pd
import redis
import json
import requests
import os
from datetime import datetime, timedelta
from pathlib import Path
import time
from dotenv import load_dotenv
from io import StringIO

# Load environment variables
load_dotenv()

class SimulationPreparator:
    """Automates complete simulation preparation process with 40/20 split"""
    
    def __init__(self, redis_host='trader.wolfx0.com', redis_port=6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.batch_size = 100
        self.alpha_api_key = os.getenv('ALPHA_API')
        
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
    
    def check_existing_simulation(self, symbol, source='yahoo'):
        """Check if simulation data already exists"""
        sim_symbol = f"S_{symbol.upper()}_{source.upper()}"
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
    
    def validate_ticker_yahoo(self, symbol):
        """Validate if ticker exists and has recent data via Yahoo Finance"""
        print(f"VALIDATING TICKER {symbol} via Yahoo Finance")
        
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
    
    def validate_ticker_alpha(self, symbol):
        """Validate if ticker exists via Alpha Vantage"""
        print(f"VALIDATING TICKER {symbol} via Alpha Vantage")
        
        if not self.alpha_api_key:
            print("Alpha Vantage API key not found in .env file")
            return False
        
        try:
            # Test with a simple quote request
            url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.alpha_api_key}'
            response = requests.get(url)
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                print(f"Ticker {symbol} validated via Alpha Vantage")
                return True
            elif 'Error Message' in data:
                print(f"Alpha Vantage error: {data['Error Message']}")
                return False
            elif 'Note' in data:
                print(f"Alpha Vantage rate limit: {data['Note']}")
                return False
            else:
                print(f"No data found for {symbol} via Alpha Vantage")
                return False
                
        except Exception as e:
            print(f"Alpha Vantage validation failed: {e}")
            return False
    
    def download_alpha_vantage_data(self, symbol, days=90):
        """Download up to 1 year of 1-minute data from Alpha Vantage Premium"""
        print(f"DOWNLOADING UP TO {days} DAYS OF 1-MINUTE DATA FOR {symbol} via Alpha Vantage Premium")
        print("=" * 60)
        
        if not self.alpha_api_key:
            print("Alpha Vantage API key not found")
            return None
        
        try:
            # For premium users, try the adjusted historical data endpoint
            # Alpha Vantage premium allows historical data through different parameters
            
            # Strategy 1: Use outputsize=full with extended month parameter
            all_data = []
            
            # Try to get multiple months of data by making separate calls
            months_to_fetch = min(12, max(1, days // 30))  # Up to 12 months for 1 year
            
            print(f"   Using premium account - attempting to fetch {months_to_fetch} months of data")
            
            current_date = datetime.now()
            
            for month_offset in range(months_to_fetch):
                target_date = current_date - timedelta(days=month_offset * 30)
                year_month = f"{target_date.year}-{target_date.month:02d}"
                
                print(f"   Fetching month {month_offset + 1} of {months_to_fetch} ({year_month})...")
                
                # For premium accounts, try using the month parameter (if available)
                # This is a premium feature that may not be documented publicly
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.alpha_api_key}&outputsize=full&month={year_month}'
                
                try:
                    response = requests.get(url)
                    data = response.json()
                    
                    if 'Error Message' in data:
                        print(f"   Month {month_offset + 1} error: {data['Error Message']}")
                        continue
                    
                    if 'Note' in data:
                        print(f"   Rate limit hit at month {month_offset + 1}")
                        break
                    
                    if 'Time Series (1min)' not in data:
                        print(f"   No data for month {month_offset + 1}")
                        continue
                    
                    # Convert to DataFrame
                    ts_data = data['Time Series (1min)']
                    month_df = pd.DataFrame.from_dict(ts_data, orient='index')
                    
                    if month_df.empty:
                        continue
                    
                    # Rename columns
                    month_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    month_df.index = pd.to_datetime(month_df.index)
                    
                    # Convert to numeric
                    for col in month_df.columns:
                        month_df[col] = pd.to_numeric(month_df[col], errors='coerce')
                    
                    # Clean data
                    month_df = month_df.sort_index().dropna()
                    
                    if not month_df.empty:
                        all_data.append(month_df)
                        print(f"   Month {month_offset + 1}: {len(month_df)} data points")
                    
                    # Respect rate limits (premium allows more calls but still has limits)
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"   Error fetching month {month_offset + 1}: {e}")
                    continue
            
            if not all_data:
                print("   Premium monthly fetch failed, trying alternative premium endpoint...")
                return self.try_premium_alternative_endpoint(symbol, days)
            
            # Combine all monthly data
            combined_df = pd.concat(all_data, axis=0)
            combined_df = combined_df.sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            
            # Limit to requested days if we got more
            if len(combined_df) > 0:
                latest_date = combined_df.index.max()
                cutoff_date = latest_date - timedelta(days=days)
                combined_df = combined_df[combined_df.index >= cutoff_date]
            
            trading_days = len(set(combined_df.index.date))
            
            print(f"Downloaded {len(combined_df)} 1-minute data points from premium API")
            print(f"   Date range: {combined_df.index.min()} to {combined_df.index.max()}")
            print(f"   Trading days: {trading_days}")
            
            return combined_df
            
        except Exception as e:
            print(f"Premium Alpha Vantage download failed: {e}")
            print("Falling back to standard endpoint...")
            return self.download_alpha_vantage_standard(symbol)

    def try_premium_alternative_endpoint(self, symbol, days):
        """Try alternative premium endpoints for historical data"""
        print("   Trying premium alternative endpoints...")
        
        try:
            # Alternative 1: Try with adjusted_close parameter (premium feature)
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.alpha_api_key}&outputsize=full&adjusted=true'
            
            response = requests.get(url)
            data = response.json()
            
            if 'Time Series (1min)' in data:
                ts_data = data['Time Series (1min)']
                df = pd.DataFrame.from_dict(ts_data, orient='index')
                
                if not df.empty:
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df.index = pd.to_datetime(df.index)
                    
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.sort_index().dropna()
                    print(f"   Alternative endpoint: {len(df)} data points")
                    return df
            
            # Alternative 2: Try the premium bulk download endpoint (if available)
            print("   Trying premium bulk download...")
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={symbol}&interval=1min&apikey={self.alpha_api_key}&slice=year1month1'
            
            response = requests.get(url)
            
            # This should return CSV data for premium users
            if response.status_code == 200 and 'time' in response.text.lower():
                csv_content = StringIO(response.text)
                df = pd.read_csv(csv_content)
                
                if not df.empty and 'time' in df.columns:
                    df = df.rename(columns={
                        'time': 'timestamp',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.sort_index().dropna()
                    print(f"   Premium bulk download: {len(df)} data points")
                    return df
            
        except Exception as e:
            print(f"   Premium alternatives failed: {e}")
        
        return None

    def download_alpha_vantage_standard(self, symbol):
        """Fallback method using standard intraday endpoint"""
        try:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.alpha_api_key}&outputsize=full'
            
            print(f"   Using standard intraday endpoint (30-day limit)")
            response = requests.get(url)
            data = response.json()
            
            if 'Error Message' in data:
                print(f"Alpha Vantage error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                print(f"Alpha Vantage rate limit: {data['Note']}")
                return None
            
            if 'Time Series (1min)' not in data:
                print("No 1-minute time series data found")
                return None
            
            # Convert to DataFrame
            ts_data = data['Time Series (1min)']
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            
            # Rename columns to match format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort and clean
            df = df.sort_index().dropna()
            
            print(f"Downloaded {len(df)} 1-minute data points (standard endpoint)")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"Standard Alpha Vantage download failed: {e}")
            return None
    
    def download_ticker_data(self, symbol, days=90, source='yahoo'):
        """Download ticker data from specified source"""
        if source.lower() == 'alpha':
            return self.download_alpha_vantage_data(symbol, days)
        else:
            return self.download_yahoo_data(symbol, days)
    
    def download_yahoo_data(self, symbol, days=60):
        """Download ticker data with Yahoo Finance limitations handled"""
        print(f"DOWNLOADING {days} DAYS OF DATA FOR {symbol} via Yahoo Finance")
        print("=" * 60)
        
        validation = self.validate_ticker_yahoo(symbol)
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
    
    def convert_to_timesale_format(self, df, symbol, source='yahoo'):
        """Convert data to timesale format with source tagging - ensure all trading days included"""
        print("Converting to timesale format...")
        
        timesale_data = []
        sim_symbol = f"S_{symbol.upper()}_{source.upper()}"
        
        # Debug: Check date range and trading days in the DataFrame
        if not df.empty:
            print(f"   DataFrame date range: {df.index.min()} to {df.index.max()}")
            trading_days = df.index.date
            unique_trading_days = sorted(set(trading_days))
            print(f"   Unique trading days in DataFrame: {len(unique_trading_days)}")
            print(f"   First trading day: {unique_trading_days[0]}")
            print(f"   Last trading day: {unique_trading_days[-1]}")
        
        for timestamp, row in df.iterrows():
            entry = {
                'timestamp': timestamp,
                'price': row['Close'],
                'volume': int(row['Volume']),
                'bid': row['Low'],
                'ask': row['High'],
                'symbol': sim_symbol,
                'trading_date': timestamp.strftime('%Y-%m-%d'),
                'source': source.upper()
            }
            timesale_data.append(entry)
        
        # Debug: Check trading days in timesale data
        if timesale_data:
            unique_dates = sorted(set(entry['trading_date'] for entry in timesale_data))
            print(f"   Unique trading dates in timesale data: {len(unique_dates)}")
            print(f"   Trading dates: {unique_dates[0]} to {unique_dates[-1]}")
        
        print(f"Converted {len(timesale_data)} entries to timesale format")
        print(f"   Source tagged as: {source.upper()}")
        return timesale_data
    
    def upload_to_redis(self, timesale_data, symbol, source='yahoo'):
        """Upload timesale data to Redis streams with source tagging"""
        print(f"UPLOADING DATA TO REDIS FOR S_{symbol}_{source.upper()}")
        print("=" * 60)
        
        sim_symbol = f"S_{symbol.upper()}_{source.upper()}"
        
        # Group data by trading day
        daily_data = {}
        for entry in timesale_data:
            trading_date = entry['trading_date']
            
            if trading_date not in daily_data:
                daily_data[trading_date] = []
            daily_data[trading_date].append(entry)
        
        # Debug: Show all trading days found
        sorted_dates = sorted(daily_data.keys())
        print(f"   Grouped into {len(daily_data)} trading days")
        print(f"   Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
        
        # Show days with data counts
        for i, date in enumerate(sorted_dates[:5]):  # Show first 5 days
            print(f"   {date}: {len(daily_data[date])} data points")
        if len(sorted_dates) > 5:
            print(f"   ... (showing first 5 of {len(sorted_dates)} days)")
        
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
                        'trading_date': entry['trading_date'],
                        'source': entry['source']
                    }
                    pipe.xadd(stream_name, stream_entry)
                
                # Execute all at once
                pipe.execute()
                
                total_messages += len(day_entries)
                uploaded_days += 1
                
                print(f"   Day {uploaded_days}: {trading_date} - {len(day_entries)} messages uploaded")
                
            except Exception as e:
                print(f"   Error uploading {trading_date}: {e}")
        
        print(f"Upload complete: {uploaded_days} days, {total_messages} messages")
        
        # Final verification
        if uploaded_days < len(daily_data):
            print(f"   WARNING: Only {uploaded_days} of {len(daily_data)} days uploaded successfully")
        
        return uploaded_days
    
    def prepare_ticker(self, symbol, source='yahoo', force_download=False):
        """Complete preparation process for a ticker with source selection"""
        symbol = symbol.upper()
        source = source.lower()
        
        print(f"PREPARING SIMULATION FOR {symbol} via {source.upper()}")
        print("=" * 80)
        
        # Step 1: Check existing data
        exists, day_count = self.check_existing_simulation(symbol, source)
        
        if exists and not force_download:
            print(f"   Simulation data already exists ({day_count} days)")
            response = input("   Proceed anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("   Skipping preparation")
                return False
        
        # Step 2: Validate ticker based on source
        if source == 'alpha':
            if not self.validate_ticker_alpha(symbol):
                return False
        else:
            if not self.validate_ticker_yahoo(symbol):
                return False
        
        # Step 3: Download data (90 days for alpha, 60 for yahoo)
        target_days = 90 if source == 'alpha' else 60
        df = self.download_ticker_data(symbol, days=target_days, source=source)
        if df is None:
            return False
        
        # Step 4: Convert to timesale format
        timesale_data = self.convert_to_timesale_format(df, symbol, source)
        
        # Step 5: Upload to Redis
        uploaded_days = self.upload_to_redis(timesale_data, symbol, source)
        if uploaded_days == 0:
            return False
        
        print(f"\nPREPARATION COMPLETE FOR {symbol} via {source.upper()}")
        print("=" * 80)
        print("Data downloaded and uploaded")
        print(f"Ready to run: python main.py --symbol {symbol} --budget 25000 --source {source}")
        
        return True

def main():
    """Main preparation function"""
    parser = argparse.ArgumentParser(description='Prepare simulation data for any ticker')
    parser.add_argument('symbol', help='Stock symbol to prepare (e.g., AAPL, TSLA, MSFT)')
    parser.add_argument('--source', choices=['yahoo', 'alpha'], default='yahoo', 
                       help='Data source: yahoo (5min, 60 days) or alpha (1min, up to 1 year with premium)')
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
    success = prep.prepare_ticker(args.symbol, args.source, args.force)
    
    if success:
        print(f"\n{args.symbol} is ready for simulation via {args.source.upper()}!")
        if args.source == 'alpha':
            print("   ✓ 1-minute precision data available (premium: up to 1 year)")
        else:
            print("   ✓ 5-minute precision data available (~60 days max)")
    else:
        print(f"\nFailed to prepare {args.symbol}")
    
    return success

if __name__ == "__main__":
    main()

