#!/usr/bin/env python3
"""
Fast Trading Agent - Uses Pre-Optimized Parameters (No Training)
Enhanced for integration with Trading Dashboard Service on port 9900
"""

import json
import redis
import time
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).parent.parent))
from core.data_manager import DataManager

class FastTradingAgent:
    """Fast trading agent that uses pre-optimized parameters"""
    
    def __init__(self, symbol='S_AAPL', source='yahoo'):
        self.symbol = symbol
        self.source = source.lower()
        self.redis_client = self.connect_redis()
        self.data_manager = DataManager(self.redis_client, symbol, {'stock_symbol': symbol})
        
        # Load pre-optimized parameters (FAST) - now source-aware
        self.optimized_params = self.load_optimized_parameters()
        self.trading_mode = self.load_trading_mode()
        
        if not self.optimized_params:
            print(f"WARNING: No optimized parameters found for {symbol} with {source} source")
            print(f"Run the optimizer first: python optimization/strategy_optimizer.py --symbol {symbol} --source {source}")
            self.use_default_params()
        
        # Trading state
        self.portfolio_value = 25000
        self.cash = 25000
        self.holdings = 0
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.wins = 0
        self.price_history = []
        self.current_day_idx = 0
        self.current_step = 0
        
        # Dashboard integration
        self.dashboard_url = "http://localhost:9900"
        
    def connect_redis(self):
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
            print(f"Redis connected for {self.symbol}")
            return redis_client
            
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return None
    
    def load_optimized_parameters(self):
        """Load pre-optimized parameters (FAST - no training) - source-aware"""
        try:
            # Load from source-specific results file
            results_file = Path(f'simple_hindsight_rl_results_{self.source}.json')
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                if self.symbol in results:
                    best_strategy = results[self.symbol]['best_strategy']
                    strategy_data = results[self.symbol]['strategies'][best_strategy]
                    
                    print(f"Loaded optimized {best_strategy} parameters for {self.symbol} ({self.source})")
                    print(f"Test return: {strategy_data['test_return']:.4f}")
                    print(f"Data source: {strategy_data.get('data_source', 'Unknown')}")
                    print(f"Data interval: {strategy_data.get('data_interval', 'Unknown')}")
                    
                    # Extract parameters (if available) or use strategy type
                    return {
                        'strategy_type': best_strategy,
                        'test_return': strategy_data['test_return'],
                        'train_return': strategy_data['train_return'],
                        'data_source': strategy_data.get('data_source', self.source.upper()),
                        'data_interval': strategy_data.get('data_interval', '5min' if self.source == 'yahoo' else '1min')
                    }
            
            return None
            
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return None
    
    def load_trading_mode(self):
        """Load the best trading mode - source-aware"""
        try:
            results_file = Path(f'simple_hindsight_rl_results_{self.source}.json')
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                if self.symbol in results:
                    return results[self.symbol]['best_strategy']
            
            return 'LONG_ONLY'
            
        except Exception as e:
            return 'LONG_ONLY'
    
    def use_default_params(self):
        """Fallback to default parameters"""
        self.optimized_params = {
            'strategy_type': 'LONG_ONLY',
            'momentum_threshold': 0.01,
            'sma_periods': (5, 10, 20),
            'profit_target': 0.05,
            'stop_loss': 0.025,
            'position_size': 0.6,
            'min_hold_periods': 2,
            'volatility_filter': 0.01,
            'volume_filter': 1.0,
            'data_source': self.source.upper(),
            'data_interval': '5min' if self.source == 'yahoo' else '1min'
        }
        self.trading_mode = 'LONG_ONLY'
        print(f"Using default parameters for {self.source} source")
    
    def get_trading_decision(self, current_price):
        """Fast trading decision using optimized parameters"""
        if len(self.price_history) < 20:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': 'Insufficient price history',
                'sma_signal': 'NEUTRAL',
                'momentum': 0.0,
                'volatility': 0.0
            }
        
        # Extract recent prices
        prices = [p['price'] for p in self.price_history[-20:]]
        
        # Calculate technical indicators (FAST)
        sma_periods = self.optimized_params.get('sma_periods', (5, 10, 20))
        sma_short = np.mean(prices[-sma_periods[0]:])
        sma_med = np.mean(prices[-sma_periods[1]:]) if len(prices) >= sma_periods[1] else sma_short
        sma_long = np.mean(prices[-sma_periods[2]:]) if len(prices) >= sma_periods[2] else sma_med
        
        # Momentum
        momentum_period = 5
        momentum = (prices[-1] - prices[-momentum_period]) / prices[-momentum_period] if len(prices) >= momentum_period else 0
        
        # Volatility
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Apply optimized trading logic (FAST)
        action = 'HOLD'
        confidence = 0.5
        reasoning = 'Default hold'
        
        momentum_threshold = self.optimized_params.get('momentum_threshold', 0.01)
        
        # Enhanced strategy logic based on optimized parameters
        if self.trading_mode == 'LONG_ONLY':
            if sma_short > sma_med > sma_long and momentum > momentum_threshold:
                action = 'BUY' if self.position == 0 else 'HOLD'
                confidence = min(momentum * 20, 1.0)
                reasoning = f'Optimized bullish signal ({self.source} data)'
            elif momentum < -momentum_threshold and self.position > 0:
                action = 'SELL'
                confidence = min(abs(momentum) * 20, 1.0)
                reasoning = f'Optimized exit signal ({self.source} data)'
                
        elif self.trading_mode == 'SHORT_ONLY':
            if sma_short < sma_med < sma_long and momentum < -momentum_threshold:
                action = 'SHORT' if self.position == 0 else 'HOLD'
                confidence = min(abs(momentum) * 20, 1.0)
                reasoning = f'Optimized bearish signal ({self.source} data)'
            elif momentum > momentum_threshold and self.position < 0:
                action = 'COVER'
                confidence = min(momentum * 20, 1.0)
                reasoning = f'Optimized cover signal ({self.source} data)'
                
        elif self.trading_mode == 'LONG_SHORT':
            if sma_short > sma_med > sma_long and momentum > momentum_threshold:
                action = 'BUY' if self.position <= 0 else 'HOLD'
                confidence = min(momentum * 20, 1.0)
                reasoning = f'Optimized long signal ({self.source} data)'
            elif sma_short < sma_med < sma_long and momentum < -momentum_threshold:
                action = 'SHORT' if self.position >= 0 else 'HOLD'
                confidence = min(abs(momentum) * 20, 1.0)
                reasoning = f'Optimized short signal ({self.source} data)'
        
        # SMA signal
        if sma_short > sma_med > sma_long:
            sma_signal = 'BULLISH'
        elif sma_short < sma_med < sma_long:
            sma_signal = 'BEARISH'
        else:
            sma_signal = 'NEUTRAL'
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'sma_signal': sma_signal,
            'momentum': momentum,
            'volatility': volatility
        }
    
    def get_available_days(self):
        """Get available trading days for simulation"""
        self.data_manager.discover_available_days()
        return self.data_manager.available_days
    
    def simulate_trading_day(self, trading_day):
        """Simulate trading for one day"""
        print(f"Simulating {self.symbol} for {trading_day} ({self.source} source)")
        
        # Load day data
        day_data = self.data_manager.load_day_data(trading_day)
        if not day_data:
            print(f"No data for {trading_day}")
            return
        
        day_trades = 0
        for entry in day_data:
            try:
                # Extract price data
                price = float(entry['price'])
                timestamp = entry['timestamp']
                
                # Add to price history
                self.price_history.append({
                    'price': price,
                    'timestamp': timestamp,
                    'bid': float(entry.get('bid', price)),
                    'ask': float(entry.get('ask', price)),
                    'volume': int(entry.get('volume', 0))
                })
                
                # Keep only last 100 prices for agent
                if len(self.price_history) > 100:
                    self.price_history.pop(0)
                
                # Get agent decision (FAST)
                agent_decision = self.get_trading_decision(price)
                
                # Execute trade
                trade_executed = self.execute_trade(agent_decision, price, timestamp)
                if trade_executed:
                    day_trades += 1
                
                # Publish to dashboard stream (enhanced for new service)
                self.publish_to_dashboard(price, agent_decision, timestamp)
                
                self.current_step += 1
                
                # Small delay for real-time feel
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing entry: {e}")
                continue
        
        print(f"Day {trading_day} complete: {day_trades} trades, Portfolio: ${self.portfolio_value:.2f}")
    
    def execute_trade(self, decision, price, timestamp):
        """Execute simulated trade"""
        action = decision['action']
        
        if action == 'BUY' and self.position == 0:
            # Open long position
            shares = int(self.cash * 0.95 // price)
            if shares > 0:
                self.cash -= shares * price
                self.holdings = shares
                self.position = 1
                self.entry_price = price
                self.total_trades += 1
                print(f"BUY: {shares} shares at ${price:.2f} ({self.source} data)")
                return True
                
        elif action == 'SELL' and self.position == 1:
            # Close long position
            proceeds = self.holdings * price
            pnl = proceeds - (self.holdings * self.entry_price)
            self.cash += proceeds
            if pnl > 0:
                self.wins += 1
            print(f"SELL: {self.holdings} shares at ${price:.2f}, P&L: ${pnl:.2f} ({self.source} data)")
            self.holdings = 0
            self.position = 0
            self.total_trades += 1
            return True
            
        elif action == 'SHORT' and self.position == 0:
            # Open short position
            shares = int(self.cash * 0.95 // price)
            if shares > 0:
                self.cash += shares * price  # Receive cash from short
                self.holdings = -shares
                self.position = -1
                self.entry_price = price
                self.total_trades += 1
                print(f"SHORT: {shares} shares at ${price:.2f} ({self.source} data)")
                return True
                
        elif action == 'COVER' and self.position == -1:
            # Cover short position
            shares = abs(self.holdings)
            cost = shares * price
            pnl = shares * (self.entry_price - price)
            self.cash -= cost
            if pnl > 0:
                self.wins += 1
            print(f"COVER: {shares} shares at ${price:.2f}, P&L: ${pnl:.2f} ({self.source} data)")
            self.holdings = 0
            self.position = 0
            self.total_trades += 1
            return True
        
        # Update portfolio value
        if self.position == 1:  # Long
            self.portfolio_value = self.cash + (self.holdings * price)
        elif self.position == -1:  # Short
            self.portfolio_value = self.cash + (abs(self.holdings) * (self.entry_price - price))
        else:
            self.portfolio_value = self.cash
        
        return False
    
    def publish_to_dashboard(self, price, decision, timestamp):
        """Publish data to dashboard stream - enhanced for new service"""
        try:
            # Calculate price change
            price_change = 0
            change_percent = 0
            if len(self.price_history) > 1:
                prev_price = self.price_history[-2]['price']
                price_change = price - prev_price
                change_percent = (price_change / prev_price) * 100
            
            dashboard_data = {
                'symbol': self.symbol,
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'price': price,
                'price_change': price_change,
                'change_percent': change_percent,
                'action': decision['action'],
                'confidence': decision['confidence'],
                'holdings': self.holdings,
                'cash': self.cash,
                'portfolio_value': self.portfolio_value,
                'total_trades': self.total_trades,
                'win_rate': self.wins / max(1, self.total_trades),
                'sma_signal': decision['sma_signal'],
                'momentum': decision['momentum'],
                'volatility': decision['volatility'],
                'volume': self.price_history[-1].get('volume', 0) if self.price_history else 0,
                'position_type': 'LONG' if self.position == 1 else 'SHORT' if self.position == -1 else 'FLAT',
                'reasoning': decision['reasoning'],
                'data_source': self.source.upper(),
                'data_interval': self.optimized_params.get('data_interval', '5min' if self.source == 'yahoo' else '1min')
            }
            
            # Publish to Redis streams for new dashboard service
            stream_key = f"simulation:{self.symbol}:{self.source}:live"
            self.redis_client.xadd(stream_key, dashboard_data, maxlen=1000)
            
            # Update latest data for dashboard service
            latest_key = f"simulation:{self.symbol}:{self.source}:latest"
            self.redis_client.set(latest_key, json.dumps(dashboard_data), ex=300)  # 5 minute expiry
            
            # Also maintain backward compatibility
            old_stream_key = f"dashboard:{self.symbol}:live"
            self.redis_client.xadd(old_stream_key, dashboard_data, maxlen=1000)
            
            old_latest_key = f"dashboard:{self.symbol}:latest"
            self.redis_client.set(old_latest_key, json.dumps(dashboard_data))
            
        except Exception as e:
            print(f"Error publishing to dashboard: {e}")
    
    def notify_dashboard_service(self, action='start'):
        """Notify the dashboard service about simulation status"""
        try:
            url = f"{self.dashboard_url}/api/simulation-status"
            data = {
                'symbol': self.symbol,
                'source': self.source,
                'action': action,
                'strategy': self.trading_mode,
                'timestamp': datetime.now().isoformat()
            }
            
            requests.post(url, json=data, timeout=5)
            print(f"Notified dashboard service: {action} simulation for {self.symbol} ({self.source})")
            
        except Exception as e:
            print(f"Could not notify dashboard service: {e}")
    
    def run_simulation(self, days_to_simulate=None):
        """Run complete simulation"""
        # Notify dashboard service
        self.notify_dashboard_service('start')
        
        available_days = self.get_available_days()
        if not available_days:
            print(f"No data available for {self.symbol}")
            return
        
        if days_to_simulate:
            available_days = available_days[:days_to_simulate]
        
        print(f"Starting simulation for {self.symbol}")
        print(f"Data source: {self.source.upper()} ({self.optimized_params.get('data_interval', 'unknown')})")
        print(f"Trading mode: {self.trading_mode}")
        print(f"Days to simulate: {len(available_days)}")
        print(f"Initial portfolio: ${self.portfolio_value:.2f}")
        print("-" * 50)
        
        for day in available_days:
            self.simulate_trading_day(day)
        
        # Final results
        total_return = (self.portfolio_value - 25000) / 25000 * 100
        win_rate = self.wins / max(1, self.total_trades) * 100
        
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        print(f"Symbol: {self.symbol}")
        print(f"Data source: {self.source.upper()} ({self.optimized_params.get('data_interval', 'unknown')})")
        print(f"Trading mode: {self.trading_mode}")
        print(f"Days simulated: {len(available_days)}")
        print(f"Total trades: {self.total_trades}")
        print(f"Winning trades: {self.wins}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Final portfolio: ${self.portfolio_value:.2f}")
        print(f"Total return: {total_return:.2f}%")
        print("=" * 50)
        
        # Notify dashboard service
        self.notify_dashboard_service('complete')

def get_available_symbols():
    """Get list of available symbols for simulation"""
    try:
        with open('.redis_passwd', 'r') as f:
            password = f.read().strip()
        
        redis_client = redis.Redis(
            host='trader.wolfx0.com',
            port=6379,
            password=password,
            decode_responses=True
        )
        
        # Find all simulation symbols (both sources)
        symbols = set()
        for source in ['YAHOO', 'ALPHA']:
            pattern = f"sim:S_*_{source}:*"
            keys = redis_client.keys(pattern)
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 2:
                    symbols.add(parts[1])
        
        # Also check for symbols without source suffix
        pattern = "sim:S_*"
        keys = redis_client.keys(pattern)
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 2:
                symbol = parts[1]
                if '_YAHOO' not in symbol and '_ALPHA' not in symbol:
                    symbols.add(symbol)
        
        return sorted(list(symbols))
        
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []

def main():
    """Main simulation function with symbol and source selection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast Trading Agent - Uses Pre-Optimized Parameters')
    parser.add_argument('--symbol', help='Symbol to simulate (e.g., S_AAPL_ALPHA)')
    parser.add_argument('--source', choices=['yahoo', 'alpha'], default='yahoo', 
                       help='Data source: yahoo (5min) or alpha (1min)')
    parser.add_argument('--days', type=int, help='Number of days to simulate')
    parser.add_argument('--list', action='store_true', help='List available symbols')
    
    args = parser.parse_args()
    
    if args.list:
        symbols = get_available_symbols()
        print("Available symbols for simulation:")
        for i, symbol in enumerate(symbols, 1):
            print(f" {i}. {symbol}")
        return
    
    if args.symbol:
        symbol = args.symbol
        # Extract source from symbol if present
        if '_YAHOO' in symbol:
            source = 'yahoo'
            symbol = symbol.replace('_YAHOO', '')
        elif '_ALPHA' in symbol:
            source = 'alpha'
            symbol = symbol.replace('_ALPHA', '')
        else:
            source = args.source
    else:
        # Interactive symbol selection
        symbols = get_available_symbols()
        if not symbols:
            print("No symbols available for simulation")
            return
        
        print("Available symbols:")
        for i, sym in enumerate(symbols, 1):
            print(f" {i}. {sym}")
        
        try:
            choice = int(input(f"Select symbol (1-{len(symbols)}): ")) - 1
            symbol = symbols[choice]
            source = args.source
        except (ValueError, IndexError):
            print("Invalid selection")
            return
    
    # Run simulation with fast trading agent
    print(f"Starting simulation for {symbol} using {source.upper()} data source")
    agent = FastTradingAgent(symbol, source)
    agent.run_simulation(args.days)

if __name__ == "__main__":
    main()

