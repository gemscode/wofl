#!/usr/bin/env python3
"""
WolfXE Simulation Publisher
Fetches Yahoo Finance intraday data and publishes to Redis for testing
"""

import redis
import json
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import argparse
import sys
import os

class WolfXESimulationPublisher:
    """Simulation publisher for WolfXE testing with Yahoo Finance data"""
    
    def __init__(self, redis_host='trader.wolfx0.com', redis_port=6379, 
                 redis_password=None, base_symbol='GERN', speed_multiplier=1.0,
                 pipeline_batch_size=50):
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=10,
            retry_on_timeout=True
        )
        
        # Configuration
        self.base_symbol = base_symbol.upper()
        self.sim_symbol = f"S_{self.base_symbol}"  # Simulation symbol
        self.speed_multiplier = speed_multiplier  # Speed up simulation
        self.pipeline_batch_size = pipeline_batch_size  # Pipeline batch size
        
        # Use the existing script SHA that's already loaded in Redis
        # This should be the same SHA from your stock publisher
        self.script_sha = "a93dea221ee3152e8bec657a34a8ff7441426223"  # Your existing script
        
        # Data storage
        self.historical_data = None
        self.current_index = 0
        
        # Performance tracking
        self.published_count = 0
        self.failed_count = 0
        self.start_time = None
        
        print("=" * 80)
        print("WOLFXE SIMULATION PUBLISHER")
        print("=" * 80)
        print(f"BASE SYMBOL:       {self.base_symbol}")
        print(f"SIMULATION SYMBOL: {self.sim_symbol}")
        print(f"SPEED MULTIPLIER:  {self.speed_multiplier}x")
        print(f"REDIS ENDPOINT:    {redis_host}:{redis_port}")
        print(f"PIPELINE BATCH:    {self.pipeline_batch_size}")
        print(f"USING SCRIPT SHA:  {self.script_sha}")
        print("=" * 80)

    def load_redis_password(self, provided_password=None):
        """Load Redis password from multiple sources"""
        if provided_password:
            return provided_password
        
        # Try to read from .redis_passwd file
        try:
            with open('.redis_passwd', 'r') as f:
                password = f.read().strip()
            if password:
                print("[CONFIG] Redis password loaded from .redis_passwd file")
                return password
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[WARNING] Error reading .redis_passwd file: {e}")
        
        # Prompt user
        import getpass
        try:
            password = getpass.getpass("Enter Redis password: ")
            return password
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(0)

    def fetch_yahoo_data(self, period="60d", interval="5m"):
        """Fetch intraday data from Yahoo Finance"""
        print(f"[DATA] Fetching {period} of {interval} data for {self.base_symbol}...")
        
        try:
            # Create ticker object
            ticker = yf.Ticker(self.base_symbol)
            
            # Fetch intraday data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"[ERROR] No data returned for {self.base_symbol}")
                return None
            
            # Reset index to get datetime as column
            data = data.reset_index()
            
            # Sort by datetime to ensure chronological order
            data = data.sort_values('Datetime')
            
            print(f"[DATA] Retrieved {len(data)} data points")
            print(f"[DATA] Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
            
            return data
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch Yahoo Finance data: {e}")
            return None

    def convert_to_timesale_format(self, row):
        """Convert Yahoo Finance data row to multiple timesale ticks per bar"""
        close_price = float(row['Close'])
        open_price = float(row['Open'])
        high_price = float(row['High'])
        low_price = float(row['Low'])
        volume = int(row['Volume']) if row['Volume'] > 0 else 1000
        
        # Generate multiple ticks per 5-minute bar for more realistic data
        ticks = []
        base_timestamp = int(row['Datetime'].timestamp() * 1000)
        
        # Create 5 ticks per 5-minute bar (one per minute)
        for i in range(5):
            # Interpolate price movement within the bar
            progress = i / 4.0  # 0 to 1
            
            # Simple price interpolation
            if i == 0:
                tick_price = open_price
            elif i == 4:
                tick_price = close_price
            else:
                # Random walk between open and close, respecting high/low
                price_range = close_price - open_price
                tick_price = open_price + (price_range * progress)
                tick_price = max(low_price, min(high_price, tick_price))
            
            # Calculate realistic bid/ask spread
            spread_percent = 0.001
            spread = tick_price * spread_percent
            bid_price = tick_price - spread/2
            ask_price = tick_price + spread/2
            
            # Distribute volume across ticks
            tick_volume = volume // 5 if i < 4 else volume - (volume // 5) * 4
            
            tick_data = {
                "type": "timesale",
                "symbol": self.sim_symbol,
                "exch": "SIM",
                "bid": f"{bid_price:.2f}",
                "ask": f"{ask_price:.2f}",
                "last": f"{tick_price:.2f}",
                "size": str(tick_volume),
                "date": str(base_timestamp + (i * 60000)),  # Add 1 minute per tick
                "seq": int((base_timestamp + i * 60000) % 1000000),
                "flag": "",
                "cancel": False,
                "correction": False,
                "session": "simulation"
            }
            
            ticks.append(tick_data)
        
        return ticks

    def publish_data_batch_pipeline(self, tick_batch):
        """Publish batch of data points using Redis pipeline for efficiency"""
        try:
            # Create pipeline for batch operations
            pipeline = self.redis_client.pipeline()
            
            # Add all ticks to pipeline
            for tick_data in tick_batch:
                pipeline.evalsha(
                    self.script_sha,
                    0,  # No keys needed
                    json.dumps(tick_data)
                )
            
            # Execute pipeline
            results = pipeline.execute()
            
            # Count successful operations
            successful = sum(1 for result in results if result is not None)
            failed = len(results) - successful
            
            self.published_count += successful
            self.failed_count += failed
            
            return successful, failed
            
        except redis.exceptions.ResponseError as e:
            if "NOSCRIPT" in str(e):
                print(f"[ERROR] Script SHA {self.script_sha} not found in Redis")
                print("Please make sure your stock publisher script is loaded first")
                return 0, len(tick_batch)
            else:
                print(f"[ERROR] Redis pipeline error: {e}")
                return 0, len(tick_batch)
        except Exception as e:
            print(f"[ERROR] Failed to publish batch: {e}")
            return 0, len(tick_batch)

    def run_simulation(self, replay_speed_seconds=5.0):
        """Run simulation with pipeline batching for efficiency"""
        if self.historical_data is None:
            print("[ERROR] No historical data loaded")
            return
        
        total_ticks = len(self.historical_data) * 5  # 5 ticks per bar
        print(f"[SIMULATION] Starting data replay with {total_ticks} ticks...")
        print(f"[SIMULATION] Using pipeline batches of {self.pipeline_batch_size}")
        print(f"[SIMULATION] Replay speed: {replay_speed_seconds/self.speed_multiplier:.1f} seconds per data point")
        print(f"[SIMULATION] Estimated duration: {(len(self.historical_data) * replay_speed_seconds/self.speed_multiplier)/60:.1f} minutes")
        print("[SIMULATION] Press Ctrl+C to stop")
        
        self.start_time = time.time()
        
        try:
            tick_batch = []
            tick_count = 0
            
            for index, row in self.historical_data.iterrows():
                # Convert to multiple timesale ticks
                ticks = self.convert_to_timesale_format(row)
                
                for tick_data in ticks:
                    tick_count += 1
                    tick_batch.append(tick_data)
                    
                    # Publish batch when it reaches batch size
                    if len(tick_batch) >= self.pipeline_batch_size:
                        successful, failed = self.publish_data_batch_pipeline(tick_batch)
                        
                        if successful > 0:
                            print(f"[BATCH] Published {successful}/{len(tick_batch)} ticks "
                                  f"(Total: {self.published_count}/{tick_count})")
                        
                        if failed > 0:
                            print(f"[WARNING] {failed} ticks failed in batch")
                        
                        tick_batch = []  # Reset batch
                        
                        # Sleep between batches (adjusted by speed multiplier)
                        time.sleep((replay_speed_seconds / 5) / self.speed_multiplier)
            
            # Publish remaining ticks in final batch
            if tick_batch:
                successful, failed = self.publish_data_batch_pipeline(tick_batch)
                print(f"[FINAL] Published {successful}/{len(tick_batch)} remaining ticks")
            
            # Performance summary
            elapsed_time = time.time() - self.start_time
            rate = self.published_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n[COMPLETE] Simulation finished!")
            print(f"[STATS] Published: {self.published_count} ticks")
            print(f"[STATS] Failed: {self.failed_count} ticks")
            print(f"[STATS] Success Rate: {self.published_count/(self.published_count+self.failed_count)*100:.1f}%")
            print(f"[STATS] Duration: {elapsed_time:.1f} seconds")
            print(f"[STATS] Rate: {rate:.1f} ticks/second")
                
        except KeyboardInterrupt:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            print(f"\n[SIMULATION] Stopped by user at tick {tick_count}")
            print(f"[STATS] Published: {self.published_count} ticks in {elapsed_time:.1f} seconds")
        except Exception as e:
            print(f"[ERROR] Simulation error: {e}")

    def run_continuous_simulation(self, replay_speed_seconds=5.0):
        """Run continuous simulation with pipeline batching"""
        if self.historical_data is None:
            print("[ERROR] No historical data loaded")
            return
        
        total_ticks = len(self.historical_data) * 5
        print(f"[SIMULATION] Starting continuous simulation...")
        print(f"[SIMULATION] Will loop through {total_ticks} ticks continuously")
        print(f"[SIMULATION] Using pipeline batches of {self.pipeline_batch_size}")
        print("[SIMULATION] Press Ctrl+C to stop")
        
        self.start_time = time.time()
        
        try:
            loop_count = 0
            tick_batch = []
            
            while True:
                loop_count += 1
                print(f"\n[SIMULATION] Starting loop #{loop_count}")
                
                for index, row in self.historical_data.iterrows():
                    # Convert to timesale format with current timestamp for real-time feel
                    ticks = self.convert_to_timesale_format(row)
                    
                    for tick_data in ticks:
                        # Update timestamp to current time for real-time simulation
                        current_timestamp = int(time.time() * 1000)
                        tick_data['date'] = str(current_timestamp)
                        
                        tick_batch.append(tick_data)
                        
                        # Publish batch when it reaches batch size
                        if len(tick_batch) >= self.pipeline_batch_size:
                            successful, failed = self.publish_data_batch_pipeline(tick_batch)
                            
                            if successful > 0:
                                print(f"[BATCH] Loop #{loop_count}: Published {successful}/{len(tick_batch)} ticks "
                                      f"(Total: {self.published_count})")
                            
                            tick_batch = []  # Reset batch
                            
                            # Sleep between batches
                            time.sleep(replay_speed_seconds / self.speed_multiplier)
                
                # Publish remaining ticks at end of loop
                if tick_batch:
                    successful, failed = self.publish_data_batch_pipeline(tick_batch)
                    tick_batch = []
                
        except KeyboardInterrupt:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            rate = self.published_count / elapsed_time if elapsed_time > 0 else 0
            print(f"\n[SIMULATION] Stopped by user after {loop_count} loops")
            print(f"[STATS] Published: {self.published_count} ticks in {elapsed_time:.1f} seconds")
            print(f"[STATS] Rate: {rate:.1f} ticks/second")
        except Exception as e:
            print(f"[ERROR] Continuous simulation error: {e}")

    def test_redis_connection(self):
        """Test Redis connection and script availability"""
        try:
            pong = self.redis_client.ping()
            print(f"[CONNECTION] Redis ping successful: {pong}")
            
            # Test if the routing script exists
            exists = self.redis_client.script_exists(self.script_sha)[0]
            if exists:
                print(f"[CONNECTION] Routing script available: {self.script_sha}")
            else:
                print(f"[WARNING] Routing script {self.script_sha} not found")
                print("Please make sure your stock publisher is running or has loaded the script")
                return False
            
            return True
        except Exception as e:
            print(f"[ERROR] Redis connection failed: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='WolfXE Simulation Publisher - Yahoo Finance Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--symbol', default='GERN', 
                       help='Base stock symbol to simulate')
    parser.add_argument('--period', default='60d', 
                       help='Yahoo Finance period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--interval', default='5m', 
                       help='Yahoo Finance interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)')
    parser.add_argument('--speed', type=float, default=1.0, 
                       help='Simulation speed multiplier (1.0 = normal, 2.0 = 2x faster)')
    parser.add_argument('--replay-speed', type=float, default=5.0, 
                       help='Seconds between data points (before speed multiplier)')
    parser.add_argument('--continuous', action='store_true', 
                       help='Run continuous simulation (loop through data)')
    parser.add_argument('--redis-password', 
                       help='Redis authentication password (will read from .redis_passwd if not provided)')
    parser.add_argument('--redis-port', type=int, default=6379, 
                       help='Redis server port')
    parser.add_argument('--redis-host', default='trader.wolfx0.com',
                       help='Redis server hostname')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Pipeline batch size for Redis operations')
    
    args = parser.parse_args()
    
    # Load Redis password if not provided
    redis_password = args.redis_password
    if not redis_password:
        try:
            with open('.redis_passwd', 'r') as f:
                redis_password = f.read().strip()
            print("[CONFIG] Redis password loaded from .redis_passwd file")
        except FileNotFoundError:
            print("[ERROR] Redis password not provided and .redis_passwd file not found")
            print("Please either:")
            print("  1. Use --redis-password argument")
            print("  2. Create .redis_passwd file with your password")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to read .redis_passwd file: {e}")
            sys.exit(1)
    
    # Create simulation publisher
    publisher = WolfXESimulationPublisher(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=redis_password,
        base_symbol=args.symbol,
        speed_multiplier=args.speed,
        pipeline_batch_size=args.batch_size
    )
    
    # Test Redis connection and script availability
    if not publisher.test_redis_connection():
        sys.exit(1)
    
    # Fetch historical data
    publisher.historical_data = publisher.fetch_yahoo_data(
        period=args.period, 
        interval=args.interval
    )
    
    if publisher.historical_data is None:
        sys.exit(1)
    
    # Run simulation
    if args.continuous:
        publisher.run_continuous_simulation(args.replay_speed)
    else:
        publisher.run_simulation(args.replay_speed)

if __name__ == "__main__":
    main()

