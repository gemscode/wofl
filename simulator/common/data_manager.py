import pandas as pd
import redis
from datetime import datetime
import os

class FixedDataManagerLocal:
    def __init__(self, symbol, redis_host='localhost', redis_port=6379):
        self.symbol = symbol
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.available_days = []
        self.connect_redis()
        self.discover_available_days()

    def connect_redis(self):
        try:
            if self.redis_host in ['localhost', '127.0.0.1']:
                self.redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    decode_responses=True
                )
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                passwd_file = os.path.join(base_dir, ".redis_passwd")
                with open(passwd_file, 'r') as f:
                    password = f.read().strip()
                self.redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    password=password,
                    decode_responses=True
                )
            self.redis_client.ping()
        except Exception as e:
            print(f"[ERROR] Redis connection failed: {e}")
            raise

    def discover_available_days(self):
        pattern = f"sim:{self.symbol}:*"
        try:
            keys = self.redis_client.keys(pattern)
            days = []
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 3:
                    days.append(parts[2])
            days = sorted(days)
            self.available_days = days
            print(f"✅ Found {len(days)} days of data ({days[0]} to {days[-1]})")
        except Exception as e:
            self.available_days = []

    def load_day_data(self, day):
        stream_key = f"sim:{self.symbol}:{day}"
        try:
            entries = self.redis_client.xrange(stream_key)
            data_points = []
            for entry_id, fields in entries:
                try:
                    price = float(fields.get('price', 0))
                    data_points.append({
                        'timestamp': fields.get('timestamp', datetime.now().isoformat()),
                        'open': float(fields.get('open', price)),
                        'high': float(fields.get('high', price)),
                        'low': float(fields.get('low', price)),
                        'close': float(fields.get('close', price)),
                        'price': price,
                        'volume': float(fields.get('volume', 1000))
                    })
                except:
                    continue
            if data_points:
                return pd.DataFrame(data_points)
            else:
                return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()

    def load_multiple_days(self, days):
        print(f"📊 Loading {len(days)} days of data...")
        all_data = []
        for day in days:
            df = self.load_day_data(day)
            if not df.empty:
                all_data.append(df)
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            if 'timestamp' in combined_df.columns:
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                combined_df.set_index('timestamp', inplace=True)
                combined_df = combined_df.sort_index()
            print(f"✅ Loaded {len(combined_df)} total data points")
            return combined_df
        else:
            return pd.DataFrame()

