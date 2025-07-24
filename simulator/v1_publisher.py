#!/usr/bin/env python3
"""
Publisher script to fetch 1-minute intraday data for stock ABC and push to Redis stream S_LIVE_ABC
"""

import redis
import yfinance as yf
import json
import time
from datetime import datetime, timedelta
import argparse

class StockDataPublisher:
    def __init__(self, redis_host='trader.wolfx0.com', redis_port=6379):
        """Initialize publisher with Redis connection"""
        try:
            with open('.redis_passwd', 'r') as f:
                password = f.read().strip()
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=password,
                decode_responses=True
            )
            
            self.redis_client.ping()
            print(f"✓ Connected to Redis at {redis_host}:{redis_port}")
            
        except Exception as e:
            print(f"✗ Redis connection failed: {e}")
            raise
    
    def fetch_intraday_data(self, symbol, period='1d', interval='1m'):
        """Fetch 1-minute intraday data for the specified symbol"""
        try:
            print(f"📊 Fetching {interval} data for {symbol} (period: {period})...")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"✗ No data fetched for symbol {symbol}")
                return None
            
            print(f"✓ Fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            print(f"✗ Error fetching data for {symbol}: {e}")
            return None
    
    def publish_to_stream(self, symbol, data, stream_name=None, delay=0.1):
        """Publish data to Redis stream"""
        if stream_name is None:
            stream_name = f'S_LIVE_{symbol}'
        
        try:
            # Clear existing stream for fresh start
            self.redis_client.delete(stream_name)
            print(f"🗑️  Cleared existing stream: {stream_name}")
            
            published_count = 0
            
            print(f"📡 Publishing to stream: {stream_name}")
            print("=" * 60)
            
            for timestamp, row in data.iterrows():
                entry = {
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'open': str(round(row['Open'], 4)),
                    'high': str(round(row['High'], 4)),
                    'low': str(round(row['Low'], 4)),
                    'close': str(round(row['Close'], 4)),
                    'price': str(round(row['Close'], 4)),  # Use close as current price
                    'volume': str(int(row['Volume'])),
                    'source': 'YAHOO_LIVE'
                }
                
                # Add to Redis stream
                stream_id = self.redis_client.xadd(stream_name, entry)
                published_count += 1
                
                # Display progress
                if published_count % 10 == 0 or published_count <= 5:
                    print(f"📈 [{published_count:3d}] {timestamp.strftime('%H:%M:%S')} | "
                          f"Price: ${float(entry['price']):7.2f} | "
                          f"Volume: {int(entry['volume']):,}")
                
                # Add delay for real-time simulation
                if delay > 0:
                    time.sleep(delay)
            
            print("=" * 60)
            print(f"✅ Published {published_count} entries to Redis stream {stream_name}")
            return published_count
            
        except Exception as e:
            print(f"✗ Error publishing to stream: {e}")
            return 0
    
    def publish_symbol(self, symbol, period='1d', interval='1m', delay=0.1):
        """Complete workflow: fetch and publish data for a symbol"""
        print(f"\n🚀 Starting publisher for {symbol}")
        print(f"📅 Period: {period} | Interval: {interval}")
        
        # Fetch data
        data = self.fetch_intraday_data(symbol, period, interval)
        if data is None:
            return False
        
        # Publish to stream
        count = self.publish_to_stream(symbol, data, delay=delay)
        
        if count > 0:
            print(f"🎉 Successfully published {count} data points for {symbol}")
            return True
        else:
            print(f"❌ Failed to publish data for {symbol}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Stock Data Publisher')
    parser.add_argument('--symbol', default='ABC', help='Stock symbol to fetch (default: ABC)')
    parser.add_argument('--period', default='1d', help='Data period (default: 1d)')
    parser.add_argument('--interval', default='1m', help='Data interval (default: 1m)')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between publishes in seconds (default: 0.1)')
    parser.add_argument('--host', default='trader.wolfx0.com', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    try:
        # Create publisher
        publisher = StockDataPublisher(args.host, args.port)
        
        # Publish data
        success = publisher.publish_symbol(
            symbol=args.symbol,
            period=args.period,
            interval=args.interval,
            delay=args.delay
        )
        
        if success:
            print(f"\n✅ Publisher completed successfully for {args.symbol}")
            print(f"🔗 Stream available at: S_LIVE_{args.symbol}")
        else:
            print(f"\n❌ Publisher failed for {args.symbol}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Publisher stopped by user")
    except Exception as e:
        print(f"\n💥 Publisher error: {e}")

if __name__ == '__main__':
    main()

