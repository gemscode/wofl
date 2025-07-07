#!/usr/bin/env python3
"""
Hindsight-Based Strategy Optimizer
Finds optimal trading opportunities using hindsight analysis
REALISTIC CONSTRAINTS: 2-5 trades per day maximum
"""

import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
from data_manager import FixedDataManager

class HindsightAnalyzer:
    """Analyzes historical data to identify REALISTIC trading opportunities"""
    
    def __init__(self, data, strategy_type='LONG_ONLY'):
        self.data = data
        self.strategy_type = strategy_type
        self.optimal_trades = []
        
    def analyze_optimal_trades(self, max_trades_per_day=2):
        """Find optimal trading opportunities with STRICT realistic limits"""
        print(f"🔍 Analyzing optimal trades for {self.strategy_type} strategy...")
        print(f"   📊 REALISTIC CONSTRAINTS:")
        print(f"   - Maximum trades per day: {max_trades_per_day}")
        print(f"   - Minimum profit threshold: 2.0%")
        print(f"   - Minimum hold time: 60 minutes")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Group by day for realistic analysis
        daily_groups = df.groupby(df.index.date)
        
        optimal_trades = []
        total_profitable_days = 0
        
        for day_idx, (day, day_data) in enumerate(daily_groups):
            if len(day_data) < 200:  # Need substantial data
                continue
                
            # Find optimal trades for this day
            day_trades = self.find_realistic_daily_trades(day_data, max_trades_per_day)
            
            if day_trades:
                optimal_trades.extend(day_trades)
                total_profitable_days += 1
            
            if day_idx % 10 == 0:
                print(f"   Processed {day_idx}/{len(daily_groups)} days...")
        
        self.optimal_trades = optimal_trades
        
        print(f"\n✅ HINDSIGHT ANALYSIS COMPLETE:")
        print(f"   - Total optimal trades found: {len(optimal_trades)}")
        print(f"   - Profitable days: {total_profitable_days}/{len(daily_groups)}")
        print(f"   - Average trades per profitable day: {len(optimal_trades)/max(1, total_profitable_days):.1f}")
        
        if optimal_trades:
            profits = [t['profit_pct'] for t in optimal_trades]
            durations = [t['hold_duration'] for t in optimal_trades]
            print(f"   - Average profit per trade: {np.mean(profits):.2%}")
            print(f"   - Average hold duration: {np.mean(durations):.1f} minutes")
            print(f"   - Best trade profit: {max(profits):.2%}")
        
        return optimal_trades
    
    def find_realistic_daily_trades(self, day_data, max_trades=2):
        """Find REALISTIC optimal trades for a single day"""
        prices = day_data['price'].values
        timestamps = day_data.index
        
        if len(prices) < 100:
            return []
        
        # STRICT constraints for realistic trading
        MIN_PROFIT_THRESHOLD = 0.02    # 2% minimum profit
        MIN_HOLD_MINUTES = 60          # 1 hour minimum hold
        MIN_PRICE_MOVE = 0.015         # 1.5% minimum price movement
        
        trades = []
        
        # Find significant price movements (not noise)
        significant_points = self.find_significant_price_points(prices, timestamps, MIN_PRICE_MOVE)
        
        if self.strategy_type in ['LONG_ONLY', 'LONG_SHORT']:
            long_trades = self.find_long_opportunities(
                significant_points, MIN_PROFIT_THRESHOLD, MIN_HOLD_MINUTES
            )
            trades.extend(long_trades)
        
        if self.strategy_type in ['SHORT_ONLY', 'LONG_SHORT']:
            short_trades = self.find_short_opportunities(
                significant_points, MIN_PROFIT_THRESHOLD, MIN_HOLD_MINUTES
            )
            trades.extend(short_trades)
        
        # Sort by profit and take only the best trades
        trades.sort(key=lambda x: x['profit_pct'], reverse=True)
        return trades[:max_trades]
    
    def find_significant_price_points(self, prices, timestamps, min_move_pct):
        """Find significant price movements, not noise"""
        points = []
        window = 20  # Look at 20-point windows
        
        for i in range(window, len(prices) - window):
            current_price = prices[i]
            
            # Check if this is a local minimum
            left_window = prices[i-window:i]
            right_window = prices[i:i+window]
            
            if (current_price <= np.min(left_window) and 
                current_price <= np.min(right_window)):
                points.append({
                    'index': i,
                    'price': current_price,
                    'timestamp': timestamps[i],
                    'type': 'LOW'
                })
            
            # Check if this is a local maximum
            elif (current_price >= np.max(left_window) and 
                  current_price >= np.max(right_window)):
                points.append({
                    'index': i,
                    'price': current_price,
                    'timestamp': timestamps[i],
                    'type': 'HIGH'
                })
        
        return points
    
    def find_long_opportunities(self, points, min_profit, min_hold_minutes):
        """Find realistic long trading opportunities"""
        trades = []
        
        lows = [p for p in points if p['type'] == 'LOW']
        highs = [p for p in points if p['type'] == 'HIGH']
        
        for low in lows:
            for high in highs:
                if high['timestamp'] <= low['timestamp']:
                    continue
                
                # Check hold duration
                hold_duration = (high['timestamp'] - low['timestamp']).total_seconds() / 60
                if hold_duration < min_hold_minutes:
                    continue
                
                # Check profit
                profit_pct = (high['price'] - low['price']) / low['price']
                if profit_pct < min_profit:
                    continue
                
                trades.append({
                    'entry_time': low['timestamp'],
                    'exit_time': high['timestamp'],
                    'entry_price': low['price'],
                    'exit_price': high['price'],
                    'profit_pct': profit_pct,
                    'action': 'LONG',
                    'hold_duration': hold_duration
                })
                break  # Take first profitable exit
        
        return trades
    
    def find_short_opportunities(self, points, min_profit, min_hold_minutes):
        """Find realistic short trading opportunities"""
        trades = []
        
        lows = [p for p in points if p['type'] == 'LOW']
        highs = [p for p in points if p['type'] == 'HIGH']
        
        for high in highs:
            for low in lows:
                if low['timestamp'] <= high['timestamp']:
                    continue
                
                # Check hold duration
                hold_duration = (low['timestamp'] - high['timestamp']).total_seconds() / 60
                if hold_duration < min_hold_minutes:
                    continue
                
                # Check profit (for short selling)
                profit_pct = (high['price'] - low['price']) / high['price']
                if profit_pct < min_profit:
                    continue
                
                trades.append({
                    'entry_time': high['timestamp'],
                    'exit_time': low['timestamp'],
                    'entry_price': high['price'],
                    'exit_price': low['price'],
                    'profit_pct': profit_pct,
                    'action': 'SHORT',
                    'hold_duration': hold_duration
                })
                break  # Take first profitable exit
        
        return trades
    
    def extract_trading_patterns(self):
        """Extract patterns from optimal trades"""
        if not self.optimal_trades:
            return {}
        
        patterns = {
            'total_trades': len(self.optimal_trades),
            'profit_distribution': [t['profit_pct'] for t in self.optimal_trades],
            'duration_distribution': [t['hold_duration'] for t in self.optimal_trades],
            'action_distribution': {}
        }
        
        # Action distribution
        for trade in self.optimal_trades:
            action = trade['action']
            patterns['action_distribution'][action] = patterns['action_distribution'].get(action, 0) + 1
        
        # Calculate statistics
        profits = patterns['profit_distribution']
        durations = patterns['duration_distribution']
        
        patterns['statistics'] = {
            'avg_profit': np.mean(profits),
            'median_profit': np.median(profits),
            'min_profit': min(profits),
            'max_profit': max(profits),
            'avg_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'success_rate': len([p for p in profits if p > 0]) / len(profits)
        }
        
        return patterns

class HindsightOptimizer:
    """Main hindsight optimizer with realistic constraints"""
    
    def __init__(self, symbol, source='alpha'):
        self.symbol = symbol
        self.source = source
        self.data_manager = FixedDataManager(symbol)
        
    def optimize_strategy(self, strategy_type, max_trades_per_day=2):
        """Optimize strategy using hindsight analysis"""
        print(f"\n🧠 HINDSIGHT OPTIMIZATION FOR {strategy_type}")
        print("=" * 60)
        
        # Load data
        available_days = self.data_manager.available_days
        if len(available_days) < 20:
            raise ValueError(f"Insufficient days: only {len(available_days)} available")
        
        # Split data (70% train, 30% test)
        split_point = int(len(available_days) * 0.7)
        train_days = available_days[:split_point]
        test_days = available_days[split_point:]
        
        print(f"📊 Data split: {len(train_days)} training days, {len(test_days)} test days")
        
        # Load training data
        train_data = self.data_manager.load_multiple_days(train_days)
        
        # Validate data quality
        is_valid, message = self.data_manager.validate_data_quality(train_data)
        if not is_valid:
            raise ValueError(f"Data quality check failed: {message}")
        
        print(f"✅ Data quality validated: {message}")
        
        # Perform hindsight analysis
        analyzer = HindsightAnalyzer(train_data, strategy_type)
        optimal_trades = analyzer.analyze_optimal_trades(max_trades_per_day)
        
        if not optimal_trades:
            return {
                'strategy': strategy_type,
                'status': 'NO_OPPORTUNITIES',
                'message': 'No profitable opportunities found with realistic constraints',
                'train_trades': 0,
                'expected_daily_trades': 0
            }
        
        # Extract patterns
        patterns = analyzer.extract_trading_patterns()
        
        # Calculate realistic performance metrics
        total_profit = sum(t['profit_pct'] for t in optimal_trades)
        trading_days = len(train_days)
        expected_daily_trades = len(optimal_trades) / trading_days
        
        # Simulate realistic returns (with transaction costs)
        TRANSACTION_COST = 0.002  # 0.2% per trade
        net_profits = [t['profit_pct'] - (2 * TRANSACTION_COST) for t in optimal_trades]
        realistic_total_return = sum(net_profits)
        
        print(f"\n📈 REALISTIC PERFORMANCE PROJECTION:")
        print(f"   - Expected daily trades: {expected_daily_trades:.1f}")
        print(f"   - Gross profit potential: {total_profit:.2%}")
        print(f"   - Net profit (after costs): {realistic_total_return:.2%}")
        print(f"   - Average profit per trade: {np.mean(net_profits):.2%}")
        print(f"   - Success rate: {patterns['statistics']['success_rate']:.1%}")
        
        return {
            'strategy': strategy_type,
            'status': 'SUCCESS',
            'train_trades': len(optimal_trades),
            'trading_days': trading_days,
            'expected_daily_trades': expected_daily_trades,
            'gross_return': total_profit,
            'net_return': realistic_total_return,
            'avg_profit_per_trade': np.mean(net_profits),
            'success_rate': patterns['statistics']['success_rate'],
            'patterns': patterns,
            'optimal_trades': optimal_trades[:10]  # Save sample trades
        }
    
    def optimize_all_strategies(self, max_trades_per_day=2):
        """Optimize all strategies with realistic constraints"""
        strategies = ['LONG_ONLY', 'SHORT_ONLY', 'LONG_SHORT']
        results = {}
        
        print(f"\n🧠 REALISTIC HINDSIGHT OPTIMIZATION FOR {self.symbol}")
        print("=" * 80)
        
        for strategy in strategies:
            try:
                results[strategy] = self.optimize_strategy(strategy, max_trades_per_day)
            except Exception as e:
                print(f"❌ Error optimizing {strategy}: {e}")
                results[strategy] = {
                    'strategy': strategy,
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Find best strategy
        successful_strategies = {k: v for k, v in results.items() 
                               if v.get('status') == 'SUCCESS'}
        
        if successful_strategies:
            best_strategy = max(successful_strategies.keys(), 
                              key=lambda k: results[k].get('net_return', -1))
            
            print(f"\n🏆 BEST REALISTIC STRATEGY: {best_strategy}")
            best_result = results[best_strategy]
            print(f"   - Expected daily trades: {best_result.get('expected_daily_trades', 0):.1f}")
            print(f"   - Net return potential: {best_result.get('net_return', 0):.2%}")
            print(f"   - Success rate: {best_result.get('success_rate', 0):.1%}")
        else:
            best_strategy = 'NONE'
            print(f"\n❌ NO PROFITABLE STRATEGIES FOUND")
            print("   All strategies failed to meet realistic profitability thresholds")
        
        # Save results
        output_file = f'realistic_hindsight_results_{self.source}.json'
        final_results = {
            self.symbol: {
                'best_strategy': best_strategy,
                'strategies': results,
                'optimization_type': 'realistic_hindsight',
                'constraints': {
                    'max_trades_per_day': max_trades_per_day,
                    'min_profit_threshold': '2.0%',
                    'min_hold_time': '60 minutes',
                    'transaction_cost': '0.2%'
                },
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Saved realistic results to {output_file}")
        return final_results

def main():
    parser = argparse.ArgumentParser(description='Realistic Hindsight Strategy Optimizer')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to optimize')
    parser.add_argument('--source', type=str, choices=['alpha', 'yahoo'], default='alpha')
    parser.add_argument('--max-trades', type=int, default=2, help='Max trades per day')
    
    args = parser.parse_args()
    
    symbol_formatted = f"S_{args.symbol.upper()}_{args.source.upper()}"
    
    try:
        optimizer = HindsightOptimizer(symbol_formatted, args.source)
        results = optimizer.optimize_all_strategies(args.max_trades)
        
        print("\n" + "="*80)
        print("🧠 REALISTIC HINDSIGHT OPTIMIZATION COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

