import redis
import pandas as pd
from datetime import datetime

class FixedDataManager:
    def __init__(self, symbol, redis_client=None):
        self.symbol = symbol
        self.redis_client = redis_client or self.connect_redis()
        self.available_days = self.discover_available_days()

    def connect_redis(self):
        print("Connecting to Redis...")
        with open('.redis_passwd', 'r') as f:
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

    def discover_available_days(self):
        print(f"Discovering available days for {self.symbol}...")
        pattern = f"sim:{self.symbol}:*"
        keys = self.redis_client.keys(pattern)
        days = []
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 3:
                days.append(parts[2])
        days = sorted(days)
        print(f"Found {len(days)} days")
        return days

    def load_day_data(self, day):
        stream_key = f"sim:{self.symbol}:{day}"
        entries = self.redis_client.xrange(stream_key)
        data_points = []
        for entry_id, fields in entries:
            try:
                price = float(fields.get('price', 0))
                open_ = float(fields.get('open', price))
                high = float(fields.get('high', price))
                low = float(fields.get('low', price))
                close = float(fields.get('close', price))
                volume = float(fields.get('volume', 1000))
                timestamp = fields.get('timestamp', datetime.now().isoformat())
                data_points.append({'timestamp': timestamp, 'open': open_, 'high': high, 'low': low, 'close': close, 'price': price, 'volume': volume})
            except Exception:
                continue
        print(f"✅ Loaded {len(data_points)} data points for {day}")
        return pd.DataFrame(data_points)

    def load_multiple_days(self, days):
        all_data = []
        for day in days:
            df = self.load_day_data(day)
            if not df.empty:
                all_data.append(df)
        if all_data:
            print(f"📊 Loaded {sum(len(df) for df in all_data)} total data points from {len(all_data)} days")
            return pd.concat(all_data).reset_index(drop=True)
        print("❗ No data loaded!")
        return pd.DataFrame()


