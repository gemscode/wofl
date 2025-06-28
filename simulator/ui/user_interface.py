#!/usr/bin/env python3
"""
User Interface Manager
"""

import sys
import json
from pathlib import Path

class UserInterface:
    
    def discover_available_tickers(self, redis_client):
        print("DISCOVERING AVAILABLE SIMULATION DATA...")
        print("=" * 80)
        
        try:
            pattern = "sim:S_*"
            keys = redis_client.keys(pattern)
            
            if not keys:
                print("WARNING: No simulation streams found")
                return {}
            
            symbols = {}
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 3:
                    symbol = parts[1]
                    date = parts[2]
                    
                    if symbol not in symbols:
                        symbols[symbol] = []
                    symbols[symbol].append(date)
            
            available_tickers = {}
            for symbol, dates in symbols.items():
                dates.sort()
                
                available_tickers[symbol] = {
                    'message_count': 'Available',
                    'trading_days': len(dates),
                    'first_date': dates[0] if dates else None,
                    'last_date': dates[-1] if dates else None,
                    'active': True
                }
            
            if available_tickers:
                print("AVAILABLE SIMULATION SYMBOLS:")
                print("-" * 80)
                print(f"{'SYMBOL':<10} {'DAYS':<8} {'DATE RANGE':<25} {'STATUS':<10}")
                print("-" * 80)
                
                for ticker, info in available_tickers.items():
                    date_range = f"{info['first_date']} to {info['last_date']}" if info['first_date'] else "N/A"
                    print(f"{ticker:<10} {info['trading_days']:<8} {date_range:<25} {'READY':<10}")
                
                print("-" * 80)
                print(f"Total simulation symbols: {len(available_tickers)}")
            
            return available_tickers
            
        except Exception as e:
            print(f"ERROR: Failed to discover simulation tickers: {e}")
            return {}
    
    def configure_trading_symbol(self, provided_symbol, available_tickers):
        if provided_symbol:
            symbol = provided_symbol.upper()
            if not symbol.startswith('S_'):
                symbol = f"S_{symbol}"
            
            if symbol in available_tickers:
                print(f"SELECTED SYMBOL: {symbol}")
                return symbol
            else:
                print(f"WARNING: Provided symbol '{symbol}' not found in simulation data")
        
        if not available_tickers:
            print("ERROR: No simulation tickers available.")
            sys.exit(1)
        
        available_symbols = list(available_tickers.keys())
        
        print("\nSIMULATION SYMBOL SELECTION:")
        print("=" * 40)
        print("AVAILABLE SIMULATION SYMBOLS:")
        for i, ticker in enumerate(available_symbols, 1):
            info = available_tickers[ticker]
            print(f"  {i}. {ticker} ({info['trading_days']} days)")
        
        while True:
            try:
                user_input = input(f"\nSelect symbol (1-{len(available_symbols)} or enter symbol): ").strip()
                
                if user_input.isdigit():
                    selection = int(user_input)
                    if 1 <= selection <= len(available_symbols):
                        selected_symbol = available_symbols[selection - 1]
                        print(f"SELECTED: {selected_symbol}")
                        return selected_symbol
                
                symbol = user_input.upper()
                if not symbol.startswith('S_'):
                    symbol = f"S_{symbol}"
                
                if symbol in available_tickers:
                    print(f"SELECTED: {symbol}")
                    return symbol
                else:
                    print(f"Symbol '{symbol}' not available.")
                    
            except KeyboardInterrupt:
                sys.exit(0)
    
    def configure_trading_budget(self, provided_budget):
        if provided_budget is not None and provided_budget > 0:
            print(f"TRADING BUDGET: ${provided_budget:,.2f}")
            return provided_budget
        
        while True:
            try:
                budget_input = input("Enter trading budget ($): ").strip().replace(',', '').replace('$', '')
                budget = float(budget_input)
                if budget > 0:
                    print(f"TRADING BUDGET SET: ${budget:,.2f}")
                    return budget
                print("Budget must be greater than zero.")
            except ValueError:
                print("Invalid amount. Please enter a numeric value.")
            except KeyboardInterrupt:
                sys.exit(0)
    
    def configure_trading_position(self, provided_position, stock_symbol):
        if provided_position is not None:
            print(f"CURRENT POSITION: {provided_position} shares")
            return provided_position
        
        while True:
            try:
                position_input = input(f"Current {stock_symbol} position (shares, default 0, or 'change' to select different symbol): ").strip()
                
                if position_input.lower() == 'change':
                    return 'CHANGE_SYMBOL'
                
                if not position_input:
                    position_input = "0"
                position = int(position_input)
                if position >= 0:
                    print(f"CURRENT POSITION SET: {position} shares")
                    return position
                print("Position cannot be negative.")
            except ValueError:
                print("Invalid input. Please enter a whole number or 'change'.")
            except KeyboardInterrupt:
                sys.exit(0)
    
    def configure_trading_strategy(self, provided_strategy):
        if provided_strategy:
            print(f"TRADING STRATEGY: {provided_strategy}")
            return provided_strategy
        
        print("\nTRADING STRATEGY SELECTION:")
        print("=" * 40)
        print("Available trading strategies:")
        print("  1. LONG_SHORT  - Buy and short sell (recommended)")
        print("  2. LONG_ONLY   - Traditional buy/sell only")
        print("  3. SHORT_ONLY  - Short selling only")
        print("")
        print("LONG_SHORT is recommended as it can profit in both")
        print("rising and falling markets.")
        
        while True:
            try:
                strategy_input = input("Select strategy (1-3 or strategy name, default: LONG_SHORT): ").strip()
                
                if not strategy_input or strategy_input == '1':
                    strategy = 'LONG_SHORT'
                elif strategy_input == '2':
                    strategy = 'LONG_ONLY'
                elif strategy_input == '3':
                    strategy = 'SHORT_ONLY'
                elif strategy_input.upper() in ['LONG_SHORT', 'LONG_ONLY', 'SHORT_ONLY']:
                    strategy = strategy_input.upper()
                else:
                    print("Invalid selection. Please choose 1, 2, 3, or strategy name.")
                    continue
                
                print(f"TRADING STRATEGY SET: {strategy}")
                return strategy
                
            except KeyboardInterrupt:
                sys.exit(0)
    
    def offer_post_simulation_options(self, simulator):
        if hasattr(simulator, 'aggressive_mode') and simulator.aggressive_mode:
            return
        
        print("\n" + "=" * 80)
        print("SIMULATION OPTIONS")
        print("=" * 80)
        print("Your simulation has completed. You can now:")
        print("  1. Exit and use current training for live trading")
        print("  2. Retrain with aggressive approach")
        print("  3. View detailed trade log")
        print("  4. Configure window range and rerun simulation")
        print("  5. Change trading strategy and rerun simulation")
        print("  6. Use optimized strategy (if available)")
        print("  7. Run optimization for this symbol")
        print("")
        
        while True:
            try:
                choice = input("Select option (1-7): ").strip()
                
                if choice == '1':
                    print("\nSIMULATION COMPLETE: System ready for live trading.")
                    return
                elif choice == '2':
                    self.run_aggressive_retrain(simulator)
                    return
                elif choice == '3':
                    self.show_detailed_trade_log(simulator)
                    continue
                elif choice == '4':
                    self.configure_window_range_and_rerun(simulator)
                    return
                elif choice == '5':
                    self.configure_strategy_and_rerun(simulator)
                    return
                elif choice == '6':
                    self.use_optimized_strategy(simulator)
                    return
                elif choice == '7':
                    self.run_optimization(simulator)
                    return
                else:
                    print("Invalid option. Please select 1-7.")
                    
            except KeyboardInterrupt:
                print("\nExiting simulation.")
                return
    
    def use_optimized_strategy(self, simulator):
        """Use pre-optimized strategy for the symbol"""
        try:
            with open('optimization_results.json', 'r') as f:
                results = json.load(f)
            
            symbol = simulator.config.get('stock_symbol', '')
            
            if symbol in results:
                optimal = results[symbol]
                best_strategy = optimal['best_strategy']
                best_params = optimal['strategies'][best_strategy]['parameters']
                
                print(f"\nUSING OPTIMIZED STRATEGY FOR {symbol}")
                print(f"Strategy: {best_strategy}")
                print(f"Parameters: {best_params}")
                
                # Apply optimized strategy
                simulator.config['trading_mode'] = best_strategy
                simulator.trading_executor.trading_mode = best_strategy
                simulator.trading_executor.apply_optimized_parameters(best_params)
                
                # Reset and rerun
                self.reset_simulator_state(simulator)
                simulator.run_complete_simulation()
                
                print(f"\nOPTIMIZED {best_strategy} SIMULATION COMPLETE")
            else:
                print(f"\nNo optimization results found for {symbol}")
                print("Run option 7 to optimize this symbol first")
        
        except FileNotFoundError:
            print("\nNo optimization results file found")
            print("Run option 7 to optimize symbols first")
        except Exception as e:
            print(f"\nError loading optimization results: {e}")
    
    def run_optimization(self, simulator):
        """Run optimization for current symbol"""
        symbol = simulator.config.get('stock_symbol', '')
        
        print(f"\nRUNNING OPTIMIZATION FOR {symbol}")
        print("This will test multiple parameter combinations to find the best strategy.")
        print("This process may take 5-10 minutes...")
        
        confirm = input("Proceed with optimization? (y/N): ").strip().lower()
        if confirm == 'y':
            try:
                # Create optimization directory if it doesn't exist
                Path('optimization').mkdir(exist_ok=True)
                
                # Run optimization inline (simplified version)
                print("OPTIMIZATION: Starting parameter search...")
                
                # Get current trading executor for optimization
                base_symbol = symbol[2:] if symbol.startswith('S_') else symbol
                
                # Load training data from data manager
                training_data = self.get_training_data(simulator.data_manager)
                
                if training_data is not None and len(training_data) > 100:
                    # Test all three strategies
                    best_results = {}
                    for mode in ['LONG_ONLY', 'SHORT_ONLY', 'LONG_SHORT']:
                        print(f"Optimizing {mode}...")
                        
                        from core.trading_executor import TradingExecutor
                        executor = TradingExecutor(base_symbol, 25000, 0, mode)
                        best_params = executor.optimize_strategy_parameters(training_data)
                        score = executor.backtest_parameters(training_data, best_params)
                        
                        best_results[mode] = {
                            'parameters': best_params,
                            'score': score
                        }
                        print(f"{mode} optimization complete - Score: {score:.3f}")
                    
                    # Find best strategy
                    best_strategy = max(best_results.keys(), key=lambda k: best_results[k]['score'])
                    
                    # Save results
                    optimization_data = {
                        symbol: {
                            'best_strategy': best_strategy,
                            'strategies': best_results,
                            'optimization_date': datetime.now().isoformat()
                        }
                    }
                    
                    # Load existing results and update
                    try:
                        with open('optimization_results.json', 'r') as f:
                            existing_results = json.load(f)
                    except FileNotFoundError:
                        existing_results = {}
                    
                    existing_results.update(optimization_data)
                    
                    with open('optimization_results.json', 'w') as f:
                        json.dump(existing_results, f, indent=2)
                    
                    print(f"\nOptimization completed successfully!")
                    print(f"Best strategy for {symbol}: {best_strategy}")
                    print(f"Score: {best_results[best_strategy]['score']:.3f}")
                    print("You can now use option 6 to run the optimized strategy")
                else:
                    print("Insufficient training data for optimization")
                    
            except Exception as e:
                print(f"Optimization failed: {e}")
                import traceback
                traceback.print_exc()
    
    def get_training_data(self, data_manager):
        """Extract training data from data manager"""
        try:
            # Get available days
            data_manager.discover_available_days()
            
            if len(data_manager.available_days) < 20:
                return None
            
            # Use first 70% for training
            train_days = data_manager.available_days[:int(len(data_manager.available_days) * 0.7)]
            
            # Load and combine data
            all_data = []
            for day in train_days:
                day_data = data_manager.load_day_data(day)
                if day_data:
                    for entry in day_data:
                        all_data.append({
                            'timestamp': datetime.fromtimestamp(int(entry['date']) / 1000),
                            'Open': float(entry['last']),
                            'High': float(entry['ask']),
                            'Low': float(entry['bid']),
                            'Close': float(entry['last']),
                            'Volume': int(entry['size'])
                        })
            
            if not all_data:
                return None
            
            import pandas as pd
            df = pd.DataFrame(all_data)
            df.set_index('timestamp', inplace=True)
            return df.dropna()
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            return None
    
    def reset_simulator_state(self, simulator):
        """Reset simulator state for rerun"""
        simulator.trading_executor.executed_trades = []
        simulator.daily_results = []
        simulator.trading_executor.current_budget = simulator.trading_executor.initial_budget
        simulator.trading_executor.current_position = 0
        simulator.trading_executor.position_entry_price = None
        simulator.trading_executor.intervals_since_last_trade = 0
        simulator.data_manager.price_history.clear()
    
    def configure_strategy_and_rerun(self, simulator):
        print("\nTRADING STRATEGY CONFIGURATION")
        print("=" * 40)
        
        current_strategy = simulator.config.get('trading_mode', 'LONG_SHORT')
        
        print(f"Current strategy: {current_strategy}")
        print("")
        print("Available strategies:")
        print("  1. LONG_SHORT  - Buy and short sell (profits in both directions)")
        print("  2. LONG_ONLY   - Traditional buy/sell only")
        print("  3. SHORT_ONLY  - Short selling only (profits from declines)")
        print("")
        
        # Performance hints based on current results
        if hasattr(simulator, 'daily_results') and simulator.daily_results:
            final_value = simulator.daily_results[-1]['end_value']
            initial_budget = simulator.config.get('initial_budget', 25000)
            current_return = ((final_value - initial_budget) / initial_budget) * 100
            
            if current_return < -2:
                print("HINT: Current strategy shows negative returns.")
                print("      Consider SHORT_ONLY if market is declining")
                print("      or LONG_SHORT for more flexibility.")
            elif current_return > 5:
                print("HINT: Current strategy is performing well.")
                print("      You may want to keep the same strategy.")
            print("")
        
        while True:
            try:
                strategy_input = input(f"Select new strategy (1-3, current: {current_strategy}): ").strip()
                
                if strategy_input == '1':
                    new_strategy = 'LONG_SHORT'
                elif strategy_input == '2':
                    new_strategy = 'LONG_ONLY'
                elif strategy_input == '3':
                    new_strategy = 'SHORT_ONLY'
                else:
                    print("Invalid selection. Please choose 1, 2, or 3.")
                    continue
                
                break
                
            except KeyboardInterrupt:
                print("\nCancelling strategy configuration")
                return
        
        if new_strategy == current_strategy:
            print(f"Strategy unchanged: {new_strategy}")
            confirm = input("Rerun simulation anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                return
        else:
            print(f"Strategy change: {current_strategy} → {new_strategy}")
            confirm = input("Proceed with new strategy? (y/N): ").strip().lower()
            if confirm != 'y':
                return
        
        print(f"\nRESTARTING SIMULATION WITH {new_strategy} STRATEGY")
        print("=" * 60)
        
        # Reset simulator state and update strategy
        simulator.config['trading_mode'] = new_strategy
        simulator.trading_executor.trading_mode = new_strategy
        self.reset_simulator_state(simulator)
        
        simulator.run_complete_simulation()
        
        print(f"\nSIMULATION WITH {new_strategy} STRATEGY COMPLETE")
        print("Compare results above with previous strategy")
    
    def configure_window_range_and_rerun(self, simulator):
        print("\nWINDOW RANGE CONFIGURATION")
        print("=" * 40)
        
        symbol = simulator.config.get('stock_symbol', 'UNKNOWN')
        symbol_clean = symbol.replace('S_', '') if symbol.startswith('S_') else symbol
        
        current_ranges = {
            'AAPL': (6, 8),
            'GERN': (2, 6),
            'TSLA': (3, 7),
            'NVDA': (4, 8),
            'MSFT': (5, 8),
            'AMZN': (4, 7),
            'GOOGL': (5, 9),
            'META': (3, 6),
            'SOUN': (2, 8),
        }
        
        current_min, current_max = current_ranges.get(symbol_clean, (2, 8))
        
        print(f"Symbol: {symbol_clean}")
        print(f"Current window range: {current_min}-{current_max} days")
        print("Window range must be between 2-8 days")
        print("")
        
        while True:
            try:
                min_input = input(f"Enter minimum window (2-7, current: {current_min}, or Enter to keep): ").strip()
                if not min_input:
                    min_window = current_min
                else:
                    min_window = int(min_input)
                    if not (2 <= min_window <= 7):
                        print("Minimum window must be between 2-7")
                        continue
                
                max_input = input(f"Enter maximum window ({min_window+1}-8, current: {current_max}, or Enter to keep): ").strip()
                if not max_input:
                    max_window = current_max
                else:
                    max_window = int(max_input)
                    if not (min_window + 1 <= max_window <= 8):
                        print(f"Maximum window must be between {min_window+1}-8")
                        continue
                
                break
                
            except ValueError:
                print("Please enter valid numbers")
            except KeyboardInterrupt:
                print("\nCancelling window configuration")
                return
        
        if min_window != current_min or max_window != current_max:
            print(f"\nWindow range change: {current_min}-{current_max} → {min_window}-{max_window} days")
        else:
            print(f"\nKeeping current window range: {min_window}-{max_window} days")
        
        confirm = input("Proceed with simulation? (y/N): ").strip().lower()
        
        if confirm == 'y':
            self.update_window_range(simulator, symbol_clean, min_window, max_window)
            
            print(f"\nRESTARTING SIMULATION WITH {min_window}-{max_window} DAY WINDOWS")
            print("=" * 60)
            
            self.reset_simulator_state(simulator)
            simulator.run_complete_simulation()
            
            print(f"\nSIMULATION WITH {min_window}-{max_window} DAY WINDOWS COMPLETE")
        else:
            print("Window configuration cancelled")
    
    def update_window_range(self, simulator, symbol, min_window, max_window):
        if hasattr(simulator, 'report_generator'):
            if not hasattr(simulator.report_generator, 'custom_ranges'):
                simulator.report_generator.custom_ranges = {}
            
            simulator.report_generator.custom_ranges[symbol] = (min_window, max_window)
            
            # Override the get_optimal_window_range method
            original_method = simulator.report_generator.get_optimal_window_range
            
            def custom_window_range(symbol_param):
                symbol_clean = symbol_param.replace('S_', '') if symbol_param.startswith('S_') else symbol_param
                if hasattr(simulator.report_generator, 'custom_ranges') and symbol_clean in simulator.report_generator.custom_ranges:
                    return simulator.report_generator.custom_ranges[symbol_clean]
                return original_method(symbol_param)
            
            simulator.report_generator.get_optimal_window_range = custom_window_range
            print(f"Updated window range for {symbol}: {min_window}-{max_window} days")
    
    def run_aggressive_retrain(self, simulator):
        print("\nSTARTING AGGRESSIVE RETRAINING...")
        print("This will use:")
        print("  - Enhanced risk management")
        print("  - Conservative position sizing")
        print("  - Higher trading frequency")
        print("  - Optimized trade amounts")
        print("  - Improved profit targeting")
        print("")
        
        confirm = input("Proceed with aggressive retraining? (y/N): ").strip().lower()
        if confirm == 'y':
            simulator.aggressive_mode = True
            self.reset_simulator_state(simulator)
            
            print("\nAGGRESSIVE MODE ACTIVATED")
            simulator.run_complete_simulation()
            
            print("\nAGGRESSIVE MODE RESULTS COMPLETE")
    
    def show_detailed_trade_log(self, simulator):
        if not hasattr(simulator, 'trading_executor') or not simulator.trading_executor.executed_trades:
            print("\nNo trades executed to display.")
            return
        
        print("\n" + "=" * 80)
        print("DETAILED TRADE LOG")
        print("=" * 80)
        
        for i, trade in enumerate(simulator.trading_executor.executed_trades, 1):
            print(f"Trade {i}:")
            print(f"  Time:     {trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Action:   {trade['action']}")
            print(f"  Shares:   {trade['shares']}")
            print(f"  Price:    ${trade['price']:.2f}")
            
            if trade['action'] in ['BUY', 'SELL_SHORT']:
                if 'cost' in trade:
                    print(f"  Cost:     ${trade['cost']:.2f}")
                if 'proceeds' in trade:
                    print(f"  Proceeds: ${trade['proceeds']:.2f}")
                if 'margin_required' in trade:
                    print(f"  Margin:   ${trade['margin_required']:.2f}")
            else:
                if 'proceeds' in trade:
                    print(f"  Proceeds: ${trade['proceeds']:.2f}")
                if 'cost' in trade:
                    print(f"  Cost:     ${trade['cost']:.2f}")
                if 'pnl' in trade:
                    print(f"  P&L:      ${trade['pnl']:+.2f}")
            
            print(f"  Reason:   {trade['reasoning']}")
            print("")
        
        input("Press Enter to continue...")

