#!/usr/bin/env python3
"""
Ticker Selection and Optimizer Launcher - Performance Optimized with Directory Fix
"""

import redis
import subprocess
import sys
import os
from pathlib import Path

def connect_redis():
    """Connect to Redis"""
    try:
        with open('.redis_passwd', 'r') as f:
            password = f.read().strip()
        
        redis_client = redis.Redis(
            host='trader.wolfx0.com',
            port=6379,
            password=password,
            decode_responses=True
        )
        redis_client.ping()
        print("Redis connection successful")
        return redis_client
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return None

def get_data_point_counts_batch(redis_client, symbols):
    """Get data point counts for all symbols using Redis pipeline (FAST)"""
    print("Counting data points using batch operations...")
    
    symbol_counts = {}
    
    # Use Redis pipeline for batch operations
    pipe = redis_client.pipeline()
    
    # Collect all keys for all symbols
    all_operations = []
    
    for symbol in symbols:
        pattern = f"sim:{symbol}:*"
        keys = redis_client.keys(pattern)
        
        symbol_counts[symbol] = {'keys': keys, 'total_points': 0}
        
        # Add xlen operations to pipeline
        for key in keys:
            pipe.xlen(key)
            all_operations.append((symbol, key))
    
    if not all_operations:
        return symbol_counts
    
    print(f"Executing {len(all_operations)} operations in batch...")
    
    # Execute all operations at once
    try:
        results = pipe.execute()
        
        # Map results back to symbols
        for i, (symbol, key) in enumerate(all_operations):
            if i < len(results):
                symbol_counts[symbol]['total_points'] += results[i]
    
    except Exception as e:
        print(f"Batch operation failed: {e}")
        # Fallback to slower method for critical symbols only
        return get_data_point_counts_fallback(redis_client, symbols[:3])  # Only first 3 symbols
    
    return symbol_counts

def get_data_point_counts_fallback(redis_client, symbols):
    """Fallback method - sample only first few days per symbol"""
    print("Using fallback sampling method...")
    
    symbol_counts = {}
    
    for symbol in symbols:
        pattern = f"sim:{symbol}:*"
        keys = redis_client.keys(pattern)
        
        if not keys:
            symbol_counts[symbol] = {'keys': [], 'total_points': 0}
            continue
        
        # Sample only first 5 days to estimate
        sample_keys = sorted(keys)[:5]
        sample_total = 0
        
        for key in sample_keys:
            try:
                count = redis_client.xlen(key)
                sample_total += count
            except:
                continue
        
        # Estimate total based on sample
        if sample_keys:
            avg_per_day = sample_total / len(sample_keys)
            estimated_total = int(avg_per_day * len(keys))
        else:
            estimated_total = 0
        
        symbol_counts[symbol] = {
            'keys': keys,
            'total_points': estimated_total,
            'estimated': True
        }
        
        print(f"  {symbol}: {estimated_total} points (estimated from {len(sample_keys)} days)")
    
    return symbol_counts

def discover_available_tickers(redis_client):
    """Discover available simulation tickers with fast data validation"""
    print("Discovering available tickers...")
    
    pattern = "sim:S_*"
    keys = redis_client.keys(pattern)
    
    if not keys:
        print("No simulation data found")
        return []
    
    # Extract unique symbols quickly
    symbols = set()
    symbol_info = {}
    
    for key in keys:
        parts = key.split(':')
        if len(parts) >= 3:
            symbol = parts[1]
            date = parts[2]
            
            if symbol not in symbols:
                symbols.add(symbol)
                symbol_info[symbol] = {'days': 0, 'first_date': date, 'last_date': date}
            
            symbol_info[symbol]['days'] += 1
            if date < symbol_info[symbol]['first_date']:
                symbol_info[symbol]['first_date'] = date
            if date > symbol_info[symbol]['last_date']:
                symbol_info[symbol]['last_date'] = date
    
    # Get data point counts using fast batch method
    sorted_symbols = sorted(list(symbols))
    symbol_counts = get_data_point_counts_batch(redis_client, sorted_symbols)
    
    # Merge data point info
    for symbol in sorted_symbols:
        if symbol in symbol_counts:
            data_points = symbol_counts[symbol]['total_points']
            symbol_info[symbol]['data_points'] = data_points
            
            # Calculate training data points (70% of total)
            training_points = int(data_points * 0.7)
            symbol_info[symbol]['training_points'] = training_points
            
            # Add estimation flag if applicable
            if symbol_counts[symbol].get('estimated'):
                symbol_info[symbol]['estimated'] = True
            
            # Determine status based on training data availability
            if training_points >= 100:
                symbol_info[symbol]['status'] = 'READY'
            elif training_points >= 50:
                symbol_info[symbol]['status'] = 'LIMITED'
            else:
                symbol_info[symbol]['status'] = 'INSUFFICIENT'
        else:
            symbol_info[symbol]['data_points'] = 0
            symbol_info[symbol]['training_points'] = 0
            symbol_info[symbol]['status'] = 'NO_DATA'
    
    print(f"Found {len(sorted_symbols)} tickers with simulation data:")
    print("-" * 100)
    print(f"{'NUM':<4} {'SYMBOL':<8} {'DAYS':<6} {'TOTAL_PTS':<10} {'TRAIN_PTS':<10} {'DATE_RANGE':<25} {'STATUS':<12}")
    print("-" * 100)
    
    ready_count = 0
    for i, symbol in enumerate(sorted_symbols, 1):
        info = symbol_info[symbol]
        date_range = f"{info['first_date']} to {info['last_date']}"
        
        if info['status'] == 'READY':
            ready_count += 1
        
        # Add estimation indicator
        total_pts_str = str(info['data_points'])
        if info.get('estimated'):
            total_pts_str += "*"
        
        print(f"{i:<4} {symbol:<8} {info['days']:<6} {total_pts_str:<10} {info['training_points']:<10} {date_range:<25} {info['status']:<12}")
    
    print("-" * 100)
    print(f"Summary: {ready_count} tickers ready for optimization")
    if any(info.get('estimated') for info in symbol_info.values()):
        print("* Estimated data points based on sample")
    
    return sorted_symbols, symbol_info

def select_ticker(symbols, symbol_info):
    """Interactive ticker selection with data validation"""
    if not symbols:
        print("No tickers available for optimization")
        return None
    
    # Show only ready tickers by default
    ready_symbols = [s for s in symbols if symbol_info[s]['status'] == 'READY']
    
    if ready_symbols:
        print(f"\nRecommended tickers (sufficient training data):")
        for i, symbol in enumerate(ready_symbols, 1):
            info = symbol_info[symbol]
            est_flag = " (estimated)" if info.get('estimated') else ""
            print(f"  {i}. {symbol} ({info['training_points']} training points{est_flag})")
    
    print(f"\nSelect ticker for optimization:")
    
    while True:
        try:
            choice = input(f"Enter ticker number (1-{len(symbols)}), symbol name, or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Operation cancelled")
                return None
            
            selected_symbol = None
            
            # Check if input is a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(symbols):
                    selected_symbol = symbols[idx]
            else:
                # Check if input is a symbol name
                choice_upper = choice.upper()
                if choice_upper in symbols:
                    selected_symbol = choice_upper
                elif f"S_{choice_upper}" in symbols:
                    selected_symbol = f"S_{choice_upper}"
            
            if selected_symbol:
                info = symbol_info[selected_symbol]
                
                print(f"\nSelected ticker: {selected_symbol}")
                print(f"Trading days: {info['days']}")
                print(f"Total data points: {info['data_points']}")
                print(f"Training data points: {info['training_points']}")
                print(f"Date range: {info['first_date']} to {info['last_date']}")
                print(f"Status: {info['status']}")
                
                if info.get('estimated'):
                    print("NOTE: Data points are estimated based on sampling")
                
                if info['status'] == 'INSUFFICIENT':
                    print(f"WARNING: Insufficient training data ({info['training_points']} < 50 points)")
                    confirm = input("This ticker may not produce reliable results. Continue? (y/N): ").strip().lower()
                    if confirm != 'y':
                        continue
                elif info['status'] == 'LIMITED':
                    print(f"WARNING: Limited training data ({info['training_points']} < 100 points)")
                    confirm = input("Results may be less reliable with limited data. Continue? (y/N): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                return selected_symbol
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(symbols)} or a valid symbol name")
                
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None
        except Exception as e:
            print(f"Error: {e}")

def launch_optimizer(symbol, episodes=300):
    """Launch the optimizer for selected symbol with correct working directory"""
    print(f"\nLaunching optimizer for {symbol}")
    print(f"Training episodes: {episodes}")
    print("-" * 60)
    
    # Get current working directory
    current_dir = Path.cwd()
    
    # Check if optimizer script exists in current directory
    optimizer_script = current_dir / "optimization" / "strategy_optimizer.py"
    if not optimizer_script.exists():
        print(f"ERROR: Optimizer script not found: {optimizer_script}")
        print("Ensure you are running from the simulator directory")
        print(f"Current directory: {current_dir}")
        return False
    
    try:
        # Use current Python interpreter and set working directory explicitly
        cmd = [
            sys.executable,  # This should use the current Python interpreter
            "optimization/strategy_optimizer.py",  # Relative path from current directory
            "--symbol", symbol,
            "--episodes", str(episodes)
        ]
        
        print(f"Working directory: {current_dir}")
        print(f"Python interpreter: {sys.executable}")
        print(f"Executing: {' '.join(cmd)}")
        print("=" * 60)
        
        # Run the optimizer with explicit working directory
        process = subprocess.Popen(
            cmd,
            cwd=str(current_dir),  # Explicitly set working directory
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            print("\nOptimization completed successfully")
        else:
            print(f"\nOptimization failed with return code: {return_code}")
        
        return return_code == 0
        
    except Exception as e:
        print(f"Failed to launch optimizer: {e}")
        return False

def main():
    """Main function with environment debugging"""
    print("WolfXE Ticker Selection and Optimizer Launcher")
    print("=" * 60)
    
    # Debug current environment
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    print(f"Python interpreter: {sys.executable}")
    print(f"Script location: {Path(__file__).absolute()}")
    
    # Check if we're in the right place
    optimizer_path = current_dir / "optimization" / "strategy_optimizer.py"
    redis_passwd_path = current_dir / ".redis_passwd"
    
    print(f"Optimizer script exists: {optimizer_path.exists()}")
    print(f"Redis password file exists: {redis_passwd_path.exists()}")
    
    if not optimizer_path.exists():
        print("\nERROR: Not in the correct directory!")
        print("Please run this script from ~/alphax0/wofl/simulator")
        return
    
    if not redis_passwd_path.exists():
        print("\nERROR: .redis_passwd file not found!")
        print("Please ensure .redis_passwd exists in the simulator directory")
        return
    
    print("-" * 60)
    
    # Connect to Redis
    redis_client = connect_redis()
    if not redis_client:
        return
    
    # Discover available tickers
    symbols, symbol_info = discover_available_tickers(redis_client)
    if not symbols:
        return
    
    # Select ticker
    selected_symbol = select_ticker(symbols, symbol_info)
    if not selected_symbol:
        return
    
    # Configure episodes
    try:
        episodes_input = input(f"\nTraining episodes (default: 300): ").strip()
        episodes = int(episodes_input) if episodes_input else 300
        
        if episodes < 50:
            print("WARNING: Less than 50 episodes may produce poor results")
        elif episodes > 1000:
            print("WARNING: More than 1000 episodes will take significant time")
            
    except ValueError:
        episodes = 300
        print(f"Using default: {episodes} episodes")
    
    # Launch optimizer
    print(f"\nStarting optimization process...")
    success = launch_optimizer(selected_symbol, episodes)
    
    if success:
        print(f"\nOptimization completed for {selected_symbol}")
        print(f"Results saved to: simple_hindsight_rl_results.json")
        print("You can now run the main simulator with optimized parameters")
    else:
        print(f"\nOptimization failed for {selected_symbol}")
        print("Check error messages above for troubleshooting")

if __name__ == "__main__":
    main()

