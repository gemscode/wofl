#!/usr/bin/env python3
"""
Publisher script that works with existing Lua function without changing the function
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
            
            # Load your existing Lua function SHA
            try:
                with open('router_v2.sha', 'r') as f:
                    self.lua_sha = f.read().strip()
                print(f"✓ Loaded Lua function SHA: {self.lua_sha}")
            except FileNotFoundError:
                print("⚠️  function.sha not found - will use direct Redis commands")
                self.lua_sha = None
            
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
    
    def convert_to_lua_format(self, symbol, timestamp, row, data_type='summary'):
        """Convert Yahoo Finance data to format expected by existing Lua function"""
        
        if data_type == 'timesale':
            # Format for timesale type (your existing Lua function expects this)
            return {
                'symbol': symbol,
                'type': 'timesale',
                'last': str(round(row['Close'], 4)),
                'size': str(int(row['Volume'])),
                'date': timestamp.isoformat(),
                'bid': str(round(row['Low'], 4)),   # Use low as bid approximation
                'ask': str(round(row['High'], 4)),  # Use high as ask approximation
                'exch': 'YAHOO',
                'seq': str(int(timestamp.timestamp())),
                'flag': 'normal',
                'cancel': False,
                'correction': False,
                'session': 'regular'
            }
        else:
            # Format for summary type (your existing Lua function expects this)
            return {
                'symbol': symbol,
                'type': 'summary',
                'open': str(round(row['Open'], 4)),
                'high': str(round(row['High'], 4)),
                'low': str(round(row['Low'], 4)),
                'close': str(round(row['Close'], 4)),
                'prevClose': str(round(row['Open'], 4)),  # Approximation
                'date': timestamp.isoformat()
            }
    
    def publish_with_existing_lua(self, symbol, data, delay=0.1, data_type='summary'):
        """Publish data using your existing Lua function (unchanged)"""
        if not self.lua_sha:
            print("❌ Lua function SHA not available")
            return 0
        
        try:
            published_count = 0
            
            print(f"📡 Publishing {data_type} data using existing Lua function")
            print("=" * 60)
            
            for timestamp, row in data.iterrows():
                # Convert to format your existing Lua function expects
                lua_data = self.convert_to_lua_format(symbol, timestamp, row, data_type)
                
                # Convert to JSON string (as your Lua function expects)
                json_data = json.dumps(lua_data)
                
                # Call your existing Lua function (no changes needed)
                result = self.redis_client.evalsha(self.lua_sha, 0, json_data)
                published_count += 1
                
                # Display progress
                if published_count % 10 == 0 or published_count <= 5:
                    if data_type == 'timesale':
                        print(f"📈 [{published_count:3d}] {timestamp.strftime('%H:%M:%S')} | "
                              f"Last: ${float(lua_data['last']):7.2f} | "
                              f"Size: {int(lua_data['size']):,}")
                    else:
                        print(f"📈 [{published_count:3d}] {timestamp.strftime('%H:%M:%S')} | "
                              f"OHLC: ${float(lua_data['open']):.2f}/"
                              f"${float(lua_data['high']):.2f}/"
                              f"${float(lua_data['low']):.2f}/"
                              f"${float(lua_data['close']):.2f}")
                
                # Add delay for real-time simulation
                if delay > 0:
                    time.sleep(delay)
            
            print("=" * 60)
            print(f"✅ Published {published_count} entries using existing Lua function")
            return published_count
            
        except Exception as e:
            print(f"✗ Error publishing with Lua function: {e}")
            print(f"Error details: {str(e)}")
            return 0
    
    def publish_to_stream_fallback(self, symbol, data, stream_name=None, delay=0.1):
        """Fallback: publish directly to Redis stream if Lua function not available"""
        if stream_name is None:
            stream_name = f'S_LIVE_{symbol}'
        
        try:
            # Clear existing stream for fresh start
            self.redis_client.delete(stream_name)
            print(f"🗑️  Cleared existing stream: {stream_name}")
            
            published_count = 0
            
            print(f"📡 Publishing to stream: {stream_name} (fallback mode)")
            print("=" * 60)
            
            for timestamp, row in data.iterrows():
                entry = {
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'open': str(round(row['Open'], 4)),
                    'high': str(round(row['High'], 4)),
                    'low': str(round(row['Low'], 4)),
                    'close': str(round(row['Close'], 4)),
                    'price': str(round(row['Close'], 4)),
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
    
    def publish_symbol(self, symbol, period='1d', interval='1m', delay=0.1, data_type='summary', use_lua=True):
        """Complete workflow: fetch and publish data for a symbol"""
        print(f"\n🚀 Starting publisher for {symbol}")
        print(f"📅 Period: {period} | Interval: {interval} | Type: {data_type}")
        print(f"🔧 Mode: {'Lua Function' if use_lua and self.lua_sha else 'Direct Stream'}")
        
        # Fetch data
        data = self.fetch_intraday_data(symbol, period, interval)
        if data is None:
            return False
        
        # Publish data
        if use_lua and self.lua_sha:
            # Use your existing Lua function
            count = self.publish_with_existing_lua(symbol, data, delay=delay, data_type=data_type)
            
            if count > 0:
                print(f"🎉 Successfully published {count} data points using Lua function")
                print(f"\n📊 Your Lua function created these streams:")
                print(f"   • ticker:{symbol}")
                print(f"   • ticker:ALL") 
                print(f"   • type:{data_type}")
        else:
            # Fallback to direct stream publishing
            count = self.publish_to_stream_fallback(symbol, data, delay=delay)
            
            if count > 0:
                print(f"🎉 Successfully published {count} data points to direct stream")
                print(f"🔗 Stream available at: S_LIVE_{symbol}")
        
        return count > 0

def main():
    parser = argparse.ArgumentParser(description='Stock Data Publisher (Compatible with existing Lua function)')
    parser.add_argument('--symbol', default='ABC', help='Stock symbol to fetch (default: ABC)')
    parser.add_argument('--period', default='1d', help='Data period (default: 1d)')
    parser.add_argument('--interval', default='1m', help='Data interval (default: 1m)')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between publishes in seconds (default: 0.1)')
    parser.add_argument('--type', choices=['summary', 'timesale'], default='summary', help='Data type for Lua function (default: summary)')
    parser.add_argument('--no-lua', action='store_true', help='Skip Lua function and use direct stream publishing')
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
            delay=args.delay,
            data_type=args.type,
            use_lua=not args.no_lua
        )
        
        if success:
            print(f"\n✅ Publisher completed successfully for {args.symbol}")
        else:
            print(f"\n❌ Publisher failed for {args.symbol}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Publisher stopped by user")
    except Exception as e:
        print(f"\n💥 Publisher error: {e}")

if __name__ == '__main__':
    main()

