# trading/data_manager.py
import os
from datetime import datetime
import pandas as pd
import redis

# ❶  Centralised config
from trading.redis_config import REDIS_CONFIG


class FixedDataManager:
    """
    Loads intra-day candle data that was published to Redis Streams by the
    simulator / live collector.  Each stream key is:

        sim:{SYMBOL}:{YYYY-MM-DD}
    """
    def __init__(self, symbol: str, redis_client: redis.Redis | None = None):
        self.symbol = symbol.upper()
        self.redis = redis_client or self._connect()
        self.available_days = self._discover_days()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _connect(self) -> redis.Redis:
        """Single place to establish a Redis connection."""
        cfg = REDIS_CONFIG                 # host, port, db, password, tls…
        client = redis.Redis(**cfg, decode_responses=True)
        client.ping()
        print(f"✅  Redis[{cfg['host']}:{cfg['port']}/{cfg['db']}] connected")
        return client

    def _discover_days(self) -> list[str]:
        """Return sorted list of all trading days stored for this symbol."""
        pattern = f"sim:{self.symbol}:*"
        days = sorted(k.split(':')[2] for k in self.redis.keys(pattern))
        print(f"🔍  Found {len(days)} day(s) for {self.symbol}")
        return days

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def load_day(self, day: str) -> pd.DataFrame:
        """Load one day as a DataFrame (timestamp-ordered)."""
        stream = f"sim:{self.symbol}:{day}"
        rows = (
            {
                "timestamp": f.get("timestamp", datetime.utcnow().isoformat()),
                "open":   float(f.get("open",   0)),
                "high":   float(f.get("high",   0)),
                "low":    float(f.get("low",    0)),
                "close":  float(f.get("close",  0)),
                "volume": float(f.get("volume", 0)),
            }
            for _id, f in self.redis.xrange(stream)
            if "close" in f                              # basic sanity check
        )
        df = pd.DataFrame(rows)
        print(f"✅  {day}: {len(df):,} rows loaded")
        return df

    def load_days(self, days: list[str]) -> pd.DataFrame:
        """Concatenate several days (keep chronological order)."""
        frames = [self.load_day(d) for d in days]
        frames = [f for f in frames if not f.empty]
        if not frames:
            print("⚠️  No data found for requested range")
            return pd.DataFrame()
        joined = pd.concat(frames).reset_index(drop=True)
        print(f"📊  {len(joined):,} rows loaded from {len(frames)} day(s)")
        return joined

