#!/usr/bin/env python3
"""
RW WolfXE Simulation Preparation Tool - 40 Days Training, 20 Days Testing
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

# Add path for imports
sys.path.append(str(Path(__file__).parent.parent))
from rw_wolfxe.production.trader_wolfxe import WolfXEnhancedTradingSystem

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
            print(f"✅ Redis connection successful: {pong}")
            return True
            
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False
    
    def check_existing_simulation(self, symbol):
        """Check if simulation data already exists"""
        sim_symbol = f"S_{symbol.upper()}"
        pattern = f"sim:{sim_symbol}:*"
        
        try:
            keys = self.redis_client.keys(pattern)
            
            if keys:
                print(f"📊 SIMULATION DATA EXISTS FOR {sim_symbol}")
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
                print(f"📊 NO SIMULATION DATA FOUND FOR {sim_symbol}")
                return False, 0
                
        except Exception as e:
            print(f"❌ Error checking existing data: {e}")
            return False, 0
    
    def validate_ticker(self, symbol):
        """Validate if ticker exists and has recent data"""
        print(f"🔍 VALIDATING TICKER {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get recent daily data first
            daily_data = ticker.history(period="5d", interval="1d")
            
            if daily_data.empty:
                print(f"❌ No recent daily data found for {symbol}")
                return False
            
            # Check if 5-minute data is available
            try:
                intraday_data = ticker.history(period="7d", interval="5m")
                if intraday_data.empty:
                    print(f"⚠️  No 5-minute intraday data available for {symbol}")
                    return "daily_only"
                else:
                    print(f"✅ Ticker {symbol} validated - intraday data available")
                    return True
            except:
                print(f"⚠️  5-minute data not available for {symbol}")
                return "daily_only"
                
        except Exception as e:
            print(f"❌ Ticker validation failed: {e}")
            return False
    
    def download_ticker_data(self, symbol, days=60):
        """Download ticker data with Yahoo Finance limitations handled"""
        print(f"📥 DOWNLOADING {days} DAYS OF DATA FOR {symbol}")
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
                    print("❌ No daily data available")
                    return None
                
                # Convert daily to simulated 5-minute data
                df = self.simulate_intraday_from_daily(df)
                
            else:
                # Use 5-minute data (limited to last 60 days by Yahoo)
                max_days = min(days, 59)
                print(f"   Using 5-minute data (limited to {max_days} days)")
                
                df = ticker.history(period=f"{max_days}d", interval="5m")
                
                if df.empty:
                    print("❌ No 5-minute data available")
                    return None
            
            df = df.dropna()
            
            if len(df) < 100:
                print(f"❌ Insufficient data: only {len(df)} points")
                return None
            
            print(f"✅ Downloaded {len(df)} data points")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def simulate_intraday_from_daily(self, daily_df):
        """Convert daily data to simulated 5-minute intervals"""
        print("🔄 Simulating 5-minute data from daily data...")
        
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
        
        print(f"✅ Generated {len(df)} simulated 5-minute intervals")
        return df
    
    def convert_to_timesale_format(self, df, symbol):
        """Convert data to timesale format"""
        print("🔄 Converting to timesale format...")
        
        timesale_data = []
        
        for timestamp, row in df.iterrows():
            entry = {
                'type': 'timesale',
                'symbol': symbol.upper(),
                'last': str(row['Close']),
                'size': str(int(row['Volume'])),
                'bid': str(row['Low']),
                'ask': str(row['High']),
                'date': str(int(timestamp.timestamp() * 1000))
            }
            timesale_data.append(entry)
        
        print(f"✅ Converted {len(timesale_data)} entries to timesale format")
        return timesale_data
    
    def upload_to_redis(self, timesale_data, symbol):
        """Upload timesale data to Redis with MAXIMUM speed"""
        print(f"📤 FAST UPLOADING DATA TO REDIS FOR S_{symbol}")
        print("=" * 60)
        
        sim_symbol = f"S_{symbol.upper()}"
        
        # Group data by trading day
        daily_data = {}
        for entry in timesale_data:
            timestamp = datetime.fromtimestamp(int(entry['date']) / 1000)
            trading_date = timestamp.strftime('%Y-%m-%d')
            
            if trading_date not in daily_data:
                daily_data[trading_date] = []
            daily_data[trading_date].append(entry)
        
        print(f"   Grouped into {len(daily_data)} trading days")
        
        # FASTEST: Upload all entries for each day in one pipeline
        uploaded_days = 0
        total_messages = 0
        
        for trading_date, day_entries in daily_data.items():
            stream_name = f"sim:{sim_symbol}:{trading_date}"
            
            try:
                # Upload entire day in one pipeline
                pipe = self.redis_client.pipeline()
                
                for entry in day_entries:
                    pipe.xadd(stream_name, entry)
                
                # Execute all at once
                pipe.execute()
                
                total_messages += len(day_entries)
                uploaded_days += 1
                
                print(f"   ⚡ Day {uploaded_days}: {len(day_entries)} messages uploaded")
                
            except Exception as e:
                print(f"   Error uploading {trading_date}: {e}")
        
        print(f"✅ FAST upload complete: {uploaded_days} days, {total_messages} messages")
        return uploaded_days
    
    def train_agents(self, symbol, training_days=40):
        """Train both agents on first 40 days - NO SEPARATION"""
        print(f"🤖 TRAINING ALL AGENTS FOR {symbol} ({training_days} DAYS)")
        print("=" * 60)
        
        sim_symbol = f"S_{symbol.upper()}"
        
        # Get available days
        pattern = f"sim:{sim_symbol}:*"
        keys = self.redis_client.keys(pattern)
        
        if len(keys) < training_days:
            print(f"❌ Insufficient data: {len(keys)} days available, need {training_days}")
            return False
        
        # Sort days and take first N for training
        days = []
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 3:
                days.append(parts[2])
        
        days.sort()
        training_days_list = days[:training_days]
        
        print(f"   Training on {len(training_days_list)} days: {training_days_list[0]} to {training_days_list[-1]}")
        print(f"   Reserving {len(days) - len(training_days_list)} days for testing")
        
        # Load training data
        all_data = []
        
        for i, day in enumerate(training_days_list):
            if i % 5 == 0:
                print(f"   Loading day {i+1}/{len(training_days_list)}: {day}")
            
            stream_name = f"sim:{sim_symbol}:{day}"
            messages = self.redis_client.xrange(stream_name)
            
            for message_id, fields in messages:
                if fields.get('type') == 'timesale':
                    timestamp_ms = int(fields.get('date', message_id.split('-')[0]))
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                    
                    all_data.append({
                        'timestamp': timestamp,
                        'Open': float(fields.get('last', 0)),
                        'High': float(fields.get('ask', 0)),
                        'Low': float(fields.get('bid', 0)),
                        'Close': float(fields.get('last', 0)),
                        'Volume': int(fields.get('size', 0))
                    })
        
        if len(all_data) < 500:
            print(f"❌ Insufficient training data: {len(all_data)} points")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df.set_index('timestamp', inplace=True)
        df = df.dropna()
        
        print(f"   Training dataset: {len(df)} data points")
        
        # Initialize and train WolfXE system - ALL AGENTS TOGETHER
        try:
            print("   Initializing WolfXE Enhanced Trading System...")
            wolfxe_system = WolfXEnhancedTradingSystem(symbol.upper(), 25000)
            
            print("   Training ALL agents together on complete dataset...")
            success = wolfxe_system.load_or_train_all_agents(df)
            
            if success:
                # Save training metadata
                training_info = {
                    'symbol': symbol.upper(),
                    'training_completed': datetime.now().isoformat(),
                    'data_points': len(df),
                    'training_days': len(training_days_list),
                    'total_available_days': len(days),
                    'trained_on': 'automated_preparation_unified',
                    'model_version': '2.1_automated_unified',
                    'long_term_trained': True,
                    'intraday_trained': True,
                    'risk_management': 'corrected',
                    'training_method': 'unified_all_agents'
                }
                
                training_file = f"training_metadata_{symbol.upper()}.json"
                with open(training_file, 'w') as f:
                    json.dump(training_info, f, indent=2)
                
                print(f"✅ ALL agents training completed successfully")
                print(f"   Metadata saved to {training_file}")
                return True
            else:
                print("❌ Training failed")
                return False
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def test_simulation(self, symbol):
        """Test the trained system on remaining days - FIXED LOGIC"""
        print(f"🧪 TESTING SIMULATION FOR {symbol} (REMAINING DAYS)")
        print("=" * 60)
        
        sim_symbol = f"S_{symbol.upper()}"
        pattern = f"sim:{sim_symbol}:*"
        keys = self.redis_client.keys(pattern)
        
        days = []
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 3:
                days.append(parts[2])
        
        days.sort()
        total_days = len(days)
        
        print(f"   Total available days: {total_days}")
        print(f"   Used for training: 40 days")
        print(f"   Available for testing: {total_days - 40} days")
        
        # FIXED: Use whatever days are available for testing
        if total_days <= 40:
            print(f"❌ No days available for testing (need more than 40 days total)")
            return False
        
        test_days_available = total_days - 40
        test_days_list = days[40:]  # Use all remaining days
        
        print(f"   Test period: {test_days_list[0]} to {test_days_list[-1]}")
        print(f"   Test days count: {len(test_days_list)}")
        print("   ✅ Ready for simulation testing with main.py")
        
        return True
    
    def prepare_ticker(self, symbol, force_download=False):
        """Complete preparation process for a ticker"""
        symbol = symbol.upper()
        
        print(f"🚀 PREPARING SIMULATION FOR {symbol} (UNIFIED TRAINING)")
        print("=" * 80)
        
        # Step 1: Check existing data
        exists, day_count = self.check_existing_simulation(symbol)
        
        if exists and not force_download:
            print(f"   Simulation data already exists ({day_count} days)")
            response = input("   Proceed with training anyway? (y/N): ").strip().lower()
            if response == 'y':
                if self.train_agents(symbol):
                    return self.test_simulation(symbol)
                return False
            else:
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
        
        # Step 5: Train ALL agents together on first 40 days
        if not self.train_agents(symbol):
            return False
        
        # Step 6: Prepare for testing on remaining days
        if not self.test_simulation(symbol):
            return False
        
        print(f"\n🎉 PREPARATION COMPLETE FOR {symbol}")
        print("=" * 80)
        print("✅ Data downloaded and uploaded")
        print("✅ ALL agents trained together on first 40 days")
        print("✅ Remaining days reserved for testing")
        print(f"\nReady to run: python main.py --symbol {symbol} --budget 25000")
        
        return True

def main():
    """Main preparation function"""
    parser = argparse.ArgumentParser(description='Prepare simulation data for any ticker (unified training)')
    parser.add_argument('symbol', help='Stock symbol to prepare (e.g., AAPL, TSLA, MSFT)')
    parser.add_argument('--force', action='store_true', help='Force re-download even if data exists')
    parser.add_argument('--redis-host', default='trader.wolfx0.com', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    # Initialize preparator
    prep = SimulationPreparator(args.redis_host, args.redis_port)
    
    # Connect to Redis
    if not prep.connect_redis():
        print("❌ Failed to connect to Redis")
        return False
    
    # Prepare the ticker
    success = prep.prepare_ticker(args.symbol, args.force)
    
    if success:
        print(f"\n✅ {args.symbol} is ready for simulation!")
    else:
        print(f"\n❌ Failed to prepare {args.symbol}")
    
    return success

if __name__ == "__main__":
    main()

