#!/usr/bin/env python3
"""
WolfXE Simulator Data Inspector
Inspect Redis simulator streams to see available data
"""

import redis
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import argparse
import sys
import os

class WolfXESimulatorInspector:
    """Inspector for WolfXE simulator data in Redis"""
    
    def __init__(self, redis_host='trader.wolfx0.com', redis_port=6379, redis_password=None):
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            socket_timeout=10
        )
        
        # Test connection
        self.test_redis_connection()
        
        print("=" * 80)
        print("WOLFXE SIMULATOR DATA INSPECTOR")
        print("=" * 80)
        print(f"REDIS ENDPOINT:   {redis_host}:{redis_port}")
        print("=" * 80)

    def test_redis_connection(self):
        """Test Redis connection"""
        try:
            pong = self.redis_client.ping()
            print(f"[CONNECTION] Redis server status: {pong}")
            return True
        except Exception as e:
            print(f"[ERROR] Redis connection failed: {e}")
            sys.exit(1)

    def discover_simulation_streams(self):
        """Discover all simulation streams"""
        try:
            # Look for daily simulation streams
            daily_pattern = "sim:S_*"
            daily_keys = self.redis_client.keys(daily_pattern)
            
            # Look for regular simulation streams
            ticker_pattern = "ticker:S_*"
            ticker_keys = self.redis_client.keys(ticker_pattern)
            
            return {
                'daily_streams': sorted(daily_keys),
                'ticker_streams': sorted(ticker_keys)
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to discover streams: {e}")
            return {'daily_streams': [], 'ticker_streams': []}

    def analyze_daily_streams(self, daily_streams):
        """Analyze daily simulation streams"""
        if not daily_streams:
            print("[INFO] No daily simulation streams found")
            return
        
        print("\nDAILY SIMULATION STREAMS:")
        print("=" * 80)
        
        # Group by symbol
        symbols = {}
        for stream in daily_streams:
            parts = stream.split(':')
            if len(parts) >= 3:
                symbol = parts[1]  # S_GERN
                date = parts[2]    # 2024-01-15
                
                if symbol not in symbols:
                    symbols[symbol] = []
                symbols[symbol].append(date)
        
        # Display each symbol's data
        for symbol, dates in symbols.items():
            dates.sort()
            print(f"\nSYMBOL: {symbol}")
            print(f"  Trading Days: {len(dates)}")
            print(f"  Date Range: {dates[0]} to {dates[-1]}")
            
            # Sample a few days to show data
            sample_days = dates[:3] if len(dates) >= 3 else dates
            for date in sample_days:
                stream_name = f"sim:{symbol}:{date}"
                try:
                    length = self.redis_client.xlen(stream_name)
                    print(f"    {date}: {length} messages")
                except Exception as e:
                    print(f"    {date}: Error - {e}")

    def analyze_ticker_streams(self, ticker_streams):
        """Analyze ticker simulation streams"""
        if not ticker_streams:
            print("[INFO] No ticker simulation streams found")
            return
        
        print("\nTICKER SIMULATION STREAMS:")
        print("=" * 80)
        
        for stream in ticker_streams:
            symbol = stream.replace('ticker:', '')
            try:
                info = self.redis_client.xinfo_stream(stream)
                length = info['length']
                
                # Get first and last messages
                first_msg = self.redis_client.xrange(stream, count=1)
                last_msg = self.redis_client.xrevrange(stream, count=1)
                
                if first_msg and last_msg:
                    first_timestamp = int(first_msg[0][1].get('date', first_msg[0][0].split('-')[0]))
                    last_timestamp = int(last_msg[0][1].get('date', last_msg[0][0].split('-')[0]))
                    
                    first_date = datetime.fromtimestamp(first_timestamp / 1000)
                    last_date = datetime.fromtimestamp(last_timestamp / 1000)
                    
                    print(f"\nSYMBOL: {symbol}")
                    print(f"  Messages: {length}")
                    print(f"  First: {first_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Last: {last_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Span: {(last_date - first_date).days} days")
                else:
                    print(f"\nSYMBOL: {symbol}")
                    print(f"  Messages: {length}")
                    print(f"  No message data available")
                    
            except Exception as e:
                print(f"\nSYMBOL: {symbol}")
                print(f"  Error: {e}")

    def inspect_stream_sample(self, stream_name, count=5):
        """Inspect sample messages from a stream"""
        try:
            print(f"\nSAMPLE MESSAGES FROM: {stream_name}")
            print("-" * 60)
            
            messages = self.redis_client.xrevrange(stream_name, count=count)
            
            if not messages:
                print("No messages found")
                return
            
            for i, (message_id, fields) in enumerate(messages, 1):
                print(f"Message {i}:")
                print(f"  ID: {message_id}")
                
                # Parse timestamp if available
                if 'date' in fields:
                    try:
                        timestamp = int(fields['date'])
                        dt = datetime.fromtimestamp(timestamp / 1000)
                        print(f"  Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        print(f"  Date: {fields['date']}")
                
                # Show key fields
                for key, value in fields.items():
                    if key not in ['date']:
                        print(f"  {key}: {value}")
                print()
                
        except Exception as e:
            print(f"Error inspecting stream {stream_name}: {e}")

    def analyze_data_quality(self, stream_name):
        """Analyze data quality for a stream"""
        try:
            print(f"\nDATA QUALITY ANALYSIS: {stream_name}")
            print("-" * 60)
            
            # Get stream info
            info = self.redis_client.xinfo_stream(stream_name)
            total_messages = info['length']
            
            # Sample messages for analysis
            sample_size = min(100, total_messages)
            messages = self.redis_client.xrange(stream_name, count=sample_size)
            
            if not messages:
                print("No messages to analyze")
                return
            
            # Analyze message structure
            field_counts = {}
            price_values = []
            volume_values = []
            timestamps = []
            
            for message_id, fields in messages:
                # Count fields
                for field in fields.keys():
                    field_counts[field] = field_counts.get(field, 0) + 1
                
                # Collect numeric data
                if 'last' in fields:
                    try:
                        price_values.append(float(fields['last']))
                    except:
                        pass
                
                if 'size' in fields:
                    try:
                        volume_values.append(int(fields['size']))
                    except:
                        pass
                
                if 'date' in fields:
                    try:
                        timestamps.append(int(fields['date']))
                    except:
                        pass
            
            # Display analysis
            print(f"Total Messages: {total_messages}")
            print(f"Sample Size: {sample_size}")
            print(f"\nField Frequency:")
            for field, count in sorted(field_counts.items()):
                percentage = (count / sample_size) * 100
                print(f"  {field}: {count}/{sample_size} ({percentage:.1f}%)")
            
            if price_values:
                print(f"\nPrice Analysis:")
                print(f"  Min Price: ${min(price_values):.2f}")
                print(f"  Max Price: ${max(price_values):.2f}")
                print(f"  Avg Price: ${sum(price_values)/len(price_values):.2f}")
            
            if volume_values:
                print(f"\nVolume Analysis:")
                print(f"  Min Volume: {min(volume_values):,}")
                print(f"  Max Volume: {max(volume_values):,}")
                print(f"  Avg Volume: {sum(volume_values)//len(volume_values):,}")
            
            if timestamps:
                timestamps.sort()
                first_time = datetime.fromtimestamp(timestamps[0] / 1000)
                last_time = datetime.fromtimestamp(timestamps[-1] / 1000)
                print(f"\nTime Analysis:")
                print(f"  First: {first_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Last: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Span: {(last_time - first_time).total_seconds() / 3600:.1f} hours")
                
        except Exception as e:
            print(f"Error analyzing data quality: {e}")

    def export_stream_data(self, stream_name, output_file=None):
        """Export stream data to CSV"""
        try:
            if not output_file:
                safe_name = stream_name.replace(':', '_').replace('*', 'all')
                output_file = f"simulator_data_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            print(f"\nEXPORTING: {stream_name} to {output_file}")
            
            messages = self.redis_client.xrange(stream_name)
            
            if not messages:
                print("No data to export")
                return
            
            # Convert to DataFrame
            data = []
            for message_id, fields in messages:
                row = {'message_id': message_id}
                row.update(fields)
                
                # Parse timestamp
                if 'date' in fields:
                    try:
                        timestamp = int(fields['date'])
                        dt = datetime.fromtimestamp(timestamp / 1000)
                        row['datetime'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        row['datetime'] = fields['date']
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            
            print(f"Exported {len(data)} records to {output_file}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")

    def run_full_inspection(self):
        """Run complete inspection of simulator data"""
        print("\nDISCOVERING SIMULATION STREAMS...")
        streams = self.discover_simulation_streams()
        
        # Analyze daily streams
        self.analyze_daily_streams(streams['daily_streams'])
        
        # Analyze ticker streams
        self.analyze_ticker_streams(streams['ticker_streams'])
        
        # Interactive inspection
        self.interactive_inspection(streams)

    def interactive_inspection(self, streams):
        """Interactive inspection mode"""
        all_streams = streams['daily_streams'] + streams['ticker_streams']
        
        if not all_streams:
            print("\n[INFO] No simulation streams found")
            return
        
        while True:
            print(f"\nINTERACTIVE INSPECTION")
            print("=" * 40)
            print("Available commands:")
            print("  1. List all streams")
            print("  2. Inspect stream sample")
            print("  3. Analyze data quality")
            print("  4. Export stream data")
            print("  5. Exit")
            
            try:
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == '1':
                    print(f"\nALL SIMULATION STREAMS ({len(all_streams)}):")
                    for i, stream in enumerate(all_streams, 1):
                        length = self.redis_client.xlen(stream)
                        print(f"  {i:2d}. {stream} ({length} messages)")
                
                elif choice == '2':
                    stream = self.select_stream(all_streams)
                    if stream:
                        count = input("Number of messages to show (default 5): ").strip()
                        count = int(count) if count.isdigit() else 5
                        self.inspect_stream_sample(stream, count)
                
                elif choice == '3':
                    stream = self.select_stream(all_streams)
                    if stream:
                        self.analyze_data_quality(stream)
                
                elif choice == '4':
                    stream = self.select_stream(all_streams)
                    if stream:
                        filename = input("Output filename (press enter for auto): ").strip()
                        self.export_stream_data(stream, filename if filename else None)
                
                elif choice == '5':
                    break
                
                else:
                    print("Invalid option")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    def select_stream(self, streams):
        """Helper to select a stream interactively"""
        if not streams:
            print("No streams available")
            return None
        
        print(f"\nSELECT STREAM:")
        for i, stream in enumerate(streams, 1):
            length = self.redis_client.xlen(stream)
            print(f"  {i:2d}. {stream} ({length} messages)")
        
        try:
            choice = input(f"\nSelect stream (1-{len(streams)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(streams):
                    return streams[idx]
            print("Invalid selection")
            return None
        except:
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='WolfXE Simulator Data Inspector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--redis-host', default='trader.wolfx0.com',
                       help='Redis server hostname')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis server port')
    parser.add_argument('--redis-password',
                       help='Redis authentication password')
    parser.add_argument('--stream',
                       help='Specific stream to inspect')
    parser.add_argument('--export',
                       help='Export stream data to CSV file')
    parser.add_argument('--sample', type=int, default=5,
                       help='Number of sample messages to show')
    
    args = parser.parse_args()
    
    # Load Redis password
    redis_password = args.redis_password
    if not redis_password:
        try:
            with open('.redis_passwd', 'r') as f:
                redis_password = f.read().strip()
        except FileNotFoundError:
            print("[ERROR] Redis password required")
            sys.exit(1)
    
    # Create inspector
    inspector = WolfXESimulatorInspector(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=redis_password
    )
    
    # Run specific inspection if requested
    if args.stream:
        if args.export:
            inspector.export_stream_data(args.stream, args.export)
        else:
            inspector.inspect_stream_sample(args.stream, args.sample)
            inspector.analyze_data_quality(args.stream)
    else:
        # Run full inspection
        inspector.run_full_inspection()

if __name__ == "__main__":
    main()

