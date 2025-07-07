#!/usr/bin/env python3
"""
Alpha Vantage Data Checker - Check which symbols have sufficient data for training
"""

import redis
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

class AlphaDataChecker:
    """Check Alpha Vantage data availability and quality"""
    
    def __init__(self, redis_host='trader.wolfx0.com', redis_port=6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        
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
    
    def discover_alpha_symbols(self):
        """Discover all Alpha Vantage symbols in Redis"""
        try:
            # Look for all simulation data with ALPHA source
            pattern = "sim:S_*_ALPHA:*"
            keys = self.redis_client.keys(pattern)
            
            symbols = set()
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 2:
                    symbol = parts[1]  # S_SYMBOL_ALPHA
                    symbols.add(symbol)
            
            return sorted(list(symbols))
            
        except Exception as e:
            print(f"Error discovering symbols: {e}")
            return []
    
    def analyze_symbol_data(self, symbol):
        """Analyze data quality and quantity for a symbol"""
        try:
            pattern = f"sim:{symbol}:*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return None
            
            # Extract dates and sort
            dates = []
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 3:
                    dates.append(parts[2])
            
            dates.sort()
            
            # Analyze data for each day
            daily_stats = {}
            total_points = 0
            
            for date in dates:
                stream_key = f"sim:{symbol}:{date}"
                try:
                    # Get stream length
                    stream_info = self.redis_client.xinfo_stream(stream_key)
                    length = stream_info.get('length', 0)
                    
                    daily_stats[date] = {
                        'data_points': length,
                        'stream_key': stream_key
                    }
                    total_points += length
                    
                except Exception as e:
                    daily_stats[date] = {
                        'data_points': 0,
                        'error': str(e)
                    }
            
            # Calculate date range
            if dates:
                start_date = datetime.strptime(dates[0], '%Y-%m-%d')
                end_date = datetime.strptime(dates[-1], '%Y-%m-%d')
                date_range_days = (end_date - start_date).days + 1
            else:
                date_range_days = 0
            
            return {
                'symbol': symbol,
                'trading_days': len(dates),
                'date_range_days': date_range_days,
                'first_date': dates[0] if dates else None,
                'last_date': dates[-1] if dates else None,
                'total_data_points': total_points,
                'avg_points_per_day': total_points / len(dates) if dates else 0,
                'daily_stats': daily_stats,
                'dates': dates
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def sample_data_quality(self, symbol, sample_date=None):
        """Sample actual data to check quality"""
        try:
            if not sample_date:
                # Get available dates
                pattern = f"sim:{symbol}:*"
                keys = self.redis_client.keys(pattern)
                if not keys:
                    return None
                
                # Use the most recent date
                dates = []
                for key in keys:
                    parts = key.split(':')
                    if len(parts) >= 3:
                        dates.append(parts[2])
                
                if not dates:
                    return None
                
                sample_date = max(dates)
            
            stream_key = f"sim:{symbol}:{sample_date}"
            
            # Get first 10 and last 10 entries
            first_entries = self.redis_client.xrange(stream_key, count=10)
            last_entries = self.redis_client.xrevrange(stream_key, count=10)
            
            sample_data = {
                'sample_date': sample_date,
                'stream_key': stream_key,
                'first_entries': [],
                'last_entries': [],
                'data_quality': {}
            }
            
            # Process first entries
            for entry_id, fields in first_entries:
                sample_data['first_entries'].append({
                    'id': entry_id,
                    'timestamp': fields.get('timestamp'),
                    'price': fields.get('price'),
                    'volume': fields.get('volume'),
                    'source': fields.get('source')
                })
            
            # Process last entries
            for entry_id, fields in last_entries:
                sample_data['last_entries'].append({
                    'id': entry_id,
                    'timestamp': fields.get('timestamp'),
                    'price': fields.get('price'),
                    'volume': fields.get('volume'),
                    'source': fields.get('source')
                })
            
            # Analyze data quality
            if first_entries:
                first_timestamp = first_entries[0][1].get('timestamp')
                last_timestamp = last_entries[0][1].get('timestamp') if last_entries else None
                
                sample_data['data_quality'] = {
                    'has_timestamps': bool(first_timestamp),
                    'has_prices': bool(first_entries[0][1].get('price')),
                    'has_volume': bool(first_entries[0][1].get('volume')),
                    'source_tagged': first_entries[0][1].get('source') == 'ALPHA',
                    'first_timestamp': first_timestamp,
                    'last_timestamp': last_timestamp
                }
            
            return sample_data
            
        except Exception as e:
            print(f"Error sampling data for {symbol}: {e}")
            return None
    
    def check_training_readiness(self, symbol_data, min_days=10, min_points_per_day=100):
        """Check if symbol data is ready for training"""
        if not symbol_data:
            return False, "No data available"
        
        # Check minimum trading days
        if symbol_data['trading_days'] < min_days:
            return False, f"Insufficient trading days: {symbol_data['trading_days']} < {min_days}"
        
        # Check average data points per day
        if symbol_data['avg_points_per_day'] < min_points_per_day:
            return False, f"Insufficient data density: {symbol_data['avg_points_per_day']:.0f} < {min_points_per_day} points/day"
        
        # Check for recent data (within last 30 days)
        if symbol_data['last_date']:
            last_date = datetime.strptime(symbol_data['last_date'], '%Y-%m-%d')
            days_old = (datetime.now() - last_date).days
            if days_old > 30:
                return False, f"Data too old: {days_old} days since last update"
        
        return True, "Ready for training"
    
    def generate_report(self, detailed=False, min_days=10, min_points_per_day=100):
        """Generate comprehensive data readiness report"""
        print("ALPHA VANTAGE DATA READINESS REPORT")
        print("=" * 60)
        
        symbols = self.discover_alpha_symbols()
        
        if not symbols:
            print("No Alpha Vantage symbols found in Redis")
            return
        
        print(f"Found {len(symbols)} Alpha Vantage symbols")
        print()
        
        ready_symbols = []
        not_ready_symbols = []
        
        for symbol in symbols:
            print(f"Analyzing {symbol}...")
            
            # Get basic stats
            data = self.analyze_symbol_data(symbol)
            if not data:
                print(f"  ❌ Failed to analyze data")
                not_ready_symbols.append((symbol, "Analysis failed"))
                continue
            
            # Check readiness
            is_ready, reason = self.check_training_readiness(data, min_days, min_points_per_day)
            
            if is_ready:
                ready_symbols.append(symbol)
                status = "✅ READY"
            else:
                not_ready_symbols.append((symbol, reason))
                status = "❌ NOT READY"
            
            print(f"  {status}")
            print(f"    Trading days: {data['trading_days']}")
            print(f"    Date range: {data['first_date']} to {data['last_date']}")
            print(f"    Total data points: {data['total_data_points']:,}")
            print(f"    Avg points/day: {data['avg_points_per_day']:.0f}")
            
            if not is_ready:
                print(f"    Reason: {reason}")
            
            # Detailed analysis if requested
            if detailed:
                sample = self.sample_data_quality(symbol)
                if sample:
                    quality = sample['data_quality']
                    print(f"    Data quality:")
                    print(f"      Timestamps: {'✅' if quality.get('has_timestamps') else '❌'}")
                    print(f"      Prices: {'✅' if quality.get('has_prices') else '❌'}")
                    print(f"      Volume: {'✅' if quality.get('has_volume') else '❌'}")
                    print(f"      Source tagged: {'✅' if quality.get('source_tagged') else '❌'}")
                    
                    if quality.get('first_timestamp') and quality.get('last_timestamp'):
                        print(f"      Time range: {quality['first_timestamp'][:19]} to {quality['last_timestamp'][:19]}")
            
            print()
        
        # Summary
        print("SUMMARY")
        print("=" * 60)
        print(f"Ready for training: {len(ready_symbols)} symbols")
        print(f"Not ready: {len(not_ready_symbols)} symbols")
        print()
        
        if ready_symbols:
            print("✅ READY SYMBOLS:")
            for symbol in ready_symbols:
                print(f"  {symbol}")
            print()
        
        if not_ready_symbols:
            print("❌ NOT READY SYMBOLS:")
            for symbol, reason in not_ready_symbols:
                print(f"  {symbol}: {reason}")
            print()
        
        # Training commands for ready symbols
        if ready_symbols:
            print("TRAINING COMMANDS:")
            print("=" * 60)
            for symbol in ready_symbols:
                base_symbol = symbol.replace('S_', '').replace('_ALPHA', '')
                print(f"python optimization/strategy_optimizer.py --symbol {base_symbol} --source alpha --episodes 300")
            print()
    
    def export_ready_symbols(self, filename="alpha_ready_symbols.json"):
        """Export ready symbols to JSON file"""
        symbols = self.discover_alpha_symbols()
        ready_data = {}
        
        for symbol in symbols:
            data = self.analyze_symbol_data(symbol)
            if data:
                is_ready, reason = self.check_training_readiness(data)
                if is_ready:
                    ready_data[symbol] = {
                        'trading_days': data['trading_days'],
                        'date_range': f"{data['first_date']} to {data['last_date']}",
                        'total_points': data['total_data_points'],
                        'avg_points_per_day': data['avg_points_per_day'],
                        'ready_for_training': True,
                        'last_checked': datetime.now().isoformat()
                    }
        
        with open(filename, 'w') as f:
            json.dump(ready_data, f, indent=2)
        
        print(f"Exported {len(ready_data)} ready symbols to {filename}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Check Alpha Vantage data readiness for training')
    parser.add_argument('--detailed', action='store_true', help='Show detailed data quality analysis')
    parser.add_argument('--min-days', type=int, default=10, help='Minimum trading days required')
    parser.add_argument('--min-points', type=int, default=100, help='Minimum data points per day')
    parser.add_argument('--export', help='Export ready symbols to JSON file')
    parser.add_argument('--symbol', help='Check specific symbol only')
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = AlphaDataChecker()
    
    # Connect to Redis
    if not checker.connect_redis():
        return
    
    if args.symbol:
        # Check specific symbol
        symbol = args.symbol.upper()
        if not symbol.startswith('S_'):
            symbol = f"S_{symbol}_ALPHA"
        
        print(f"Checking {symbol}...")
        data = checker.analyze_symbol_data(symbol)
        
        if data:
            is_ready, reason = checker.check_training_readiness(data, args.min_days, args.min_points)
            
            print(f"Symbol: {symbol}")
            print(f"Status: {'✅ READY' if is_ready else '❌ NOT READY'}")
            print(f"Trading days: {data['trading_days']}")
            print(f"Date range: {data['first_date']} to {data['last_date']}")
            print(f"Total data points: {data['total_data_points']:,}")
            print(f"Avg points/day: {data['avg_points_per_day']:.0f}")
            
            if not is_ready:
                print(f"Reason: {reason}")
            
            if args.detailed:
                sample = checker.sample_data_quality(symbol)
                if sample:
                    print("\nSample data:")
                    print(json.dumps(sample, indent=2))
        else:
            print(f"No data found for {symbol}")
    
    else:
        # Generate full report
        checker.generate_report(detailed=args.detailed, min_days=args.min_days, min_points_per_day=args.min_points)
        
        # Export if requested
        if args.export:
            checker.export_ready_symbols(args.export)

if __name__ == "__main__":
    main()

