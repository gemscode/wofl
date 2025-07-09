import redis
import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta
import json

class FixedDataManager:
    def __init__(self, symbol, redis_client=None):
        self.symbol = symbol.replace("S_", "").replace("_ALPHA", "")  # Clean symbol
        self.redis_client = redis_client or self.connect_redis()
        
        # Alpha Vantage configuration
        self.alpha_key = self._get_alpha_key()
        self.base_url = "https://www.alphavantage.co/query"
        
        print(f"✅ Data Manager initialized for {self.symbol}")

    def connect_redis(self):
        """Connect to Redis with authentication"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            passwd_file = os.path.join(base_dir, ".redis_passwd")
            
            with open(passwd_file, 'r') as f:
                password = f.read().strip()
            
            r = redis.Redis(
                host='trader.wolfx0.com',
                port=6379,
                password=password,
                decode_responses=True
            )
            r.ping()
            print("✅ Redis connection successful")
            return r
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            raise

    def _get_alpha_key(self):
        """Get Alpha Vantage API key from environment or file"""
        # Try environment variable first
        alpha_key = os.getenv('ALPHA_VANTAGE_KEY')
        
        if not alpha_key:
            # Try reading from file
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                key_file = os.path.join(base_dir, ".alpha_key")
                with open(key_file, 'r') as f:
                    alpha_key = f.read().strip()
            except FileNotFoundError:
                print("❌ Alpha Vantage API key not found. Please set ALPHA_VANTAGE_KEY environment variable or create .alpha_key file")
                raise
        
        return alpha_key

    def fetch_intraday_data(self, interval='1min', outputsize='full'):
        """Fetch intraday data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': self.symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self.alpha_key
        }
        
        try:
            print(f"📡 Fetching {interval} data for {self.symbol} from Alpha Vantage...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage Error: {data['Error Message']}")
            
            if 'Note' in data:
                print(f"⚠️ API Note: {data['Note']}")
                time.sleep(60)  # Wait if rate limited
                return self.fetch_intraday_data(interval, outputsize)
            
            # Extract time series data
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                raise ValueError(f"No time series data found for {self.symbol}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ Fetched {len(df)} data points for {self.symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching Alpha Vantage data: {e}")
            raise

    def get_latest_trading_day(self):
        """Get the most recent trading day data"""
        df = self.fetch_intraday_data()
        
        if df.empty:
            return None, None
        
        # Get the latest trading date
        latest_date = df['timestamp'].dt.date.max()
        
        # Filter data for the latest trading day
        day_data = df[df['timestamp'].dt.date == latest_date].copy()
        day_data = day_data.set_index('timestamp')
        
        return day_data, latest_date.strftime('%Y-%m-%d')

    def load_multiple_days(self, days):
        """Load data for multiple days (for compatibility with existing code)"""
        # For Alpha Vantage, we'll just return the latest day's data
        # since we're fetching live data
        day_data, trading_date = self.get_latest_trading_day()
        
        if day_data is not None:
            print(f"📊 Loaded {len(day_data)} data points for {trading_date}")
            return day_data
        else:
            print("❗ No data available")
            return pd.DataFrame()

    @property
    def available_days(self):
        """Return available trading days (for compatibility)"""
        # For live Alpha Vantage data, return recent trading days
        today = datetime.now().date()
        days = []
        
        # Generate last 5 trading days (approximate)
        for i in range(10):  # Check last 10 days to account for weekends
            day = today - timedelta(days=i)
            # Skip weekends (rough approximation)
            if day.weekday() < 5:  # Monday = 0, Friday = 4
                days.append(day.strftime('%Y-%m-%d'))
            if len(days) >= 5:
                break
        
        return days

    def load_day_data(self, day):
        """Load data for a specific day (for compatibility)"""
        # For Alpha Vantage, return latest data regardless of requested day
        day_data, trading_date = self.get_latest_trading_day()
        
        if day_data is not None:
            print(f"✅ Loaded {len(day_data)} data points for {trading_date}")
            return day_data
        else:
            return pd.DataFrame()

