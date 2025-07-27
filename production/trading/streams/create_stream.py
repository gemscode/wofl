#!/usr/bin/env python3

import redis
import sys
import argparse
from datetime import datetime

def create_trading_stream(symbol, stream_name=None):
    """Create Redis stream for trading data if it doesn't exist"""
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test connection
        r.ping()
        print("✅ Connected to Redis successfully")
        
        # Generate stream name if not provided
        if stream_name is None:
            stream_name = f"trading_stream_{symbol.upper()}"
        
        # Check if stream exists
        try:
            stream_info = r.xinfo_stream(stream_name)
            print(f"📊 Stream '{stream_name}' already exists with {stream_info['length']} messages")
            return stream_name
        except redis.ResponseError:
            # Stream doesn't exist, create it by adding a dummy message
            print(f"🔧 Creating new stream: {stream_name}")
            
            # Add initial message to create the stream
            message_id = r.xadd(stream_name, {
                'type': 'stream_init',
                'symbol': symbol.upper(),
                'timestamp': datetime.now().isoformat(),
                'status': 'initialized'
            })
            
            print(f"✅ Stream '{stream_name}' created successfully")
            print(f"📝 Initial message ID: {message_id}")
            
            # Create consumer group for the trading agent
            try:
                r.xgroup_create(stream_name, 'trading_agents', id='0', mkstream=True)
                print(f"👥 Consumer group 'trading_agents' created")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    print(f"👥 Consumer group 'trading_agents' already exists")
                else:
                    print(f"⚠️ Error creating consumer group: {e}")
            
            return stream_name
            
    except redis.ConnectionError:
        print("❌ Failed to connect to Redis. Make sure Redis is running.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error creating stream: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Create Redis stream for trading data')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., GERN)')
    parser.add_argument('--stream', type=str, help='Custom stream name')
    args = parser.parse_args()
    
    print(f"🎯 Creating trading stream for {args.symbol}")
    stream_name = create_trading_stream(args.symbol, args.stream)
    print(f"🎉 Stream setup complete: {stream_name}")

if __name__ == "__main__":
    main()

