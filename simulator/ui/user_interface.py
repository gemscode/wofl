#!/usr/bin/env python3
"""
User Interface Manager - Enterprise Clean Implementation with Window Configuration
"""

import sys

class UserInterface:
    """Professional user interface with window configuration options"""
    
    def discover_available_tickers(self, redis_client):
        """Professional ticker discovery"""
        print("DISCOVERING AVAILABLE SIMULATION DATA...")
        print("=" * 80)
        
        try:
            pattern = "sim:S_*"
            keys = redis_client.keys(pattern)
            
            if not keys:
                print("WARNING: No simulation streams found")
                return {}
            
            # Group by symbol
            symbols = {}
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 3:
                    symbol = parts[1]  # S_GERN
                    date = parts[2]    # 2024-01-15
                    
                    if symbol not in symbols:
                        symbols[symbol] = []
                    symbols[symbol].append(date)
            
            # Get stream information
            available_tickers = {}
            for symbol, dates in symbols.items():
                dates.sort()
                total_messages = 0
                
                for date in dates:
                    stream_name = f"sim:{symbol}:{date}"
                    try:
                        length = redis_client.xlen(stream_name)
                        total_messages += length
                    except:
                        pass
                
                available_tickers[symbol] = {
                    'message_count': total_messages,
                    'trading_days': len(dates),
                    'first_date': dates[0] if dates else None,
                    'last_date': dates[-1] if dates else None,
                    'active': True
                }
            
            # Display discovered tickers
            if available_tickers:
                print("AVAILABLE SIMULATION SYMBOLS:")
                print("-" * 80)
                print(f"{'SYMBOL':<10} {'MESSAGES':<12} {'DAYS':<8} {'DATE RANGE':<25} {'STATUS':<10}")
                print("-" * 80)
                
                for ticker, info in available_tickers.items():
                    date_range = f"{info['first_date']} to {info['last_date']}" if info['first_date'] else "N/A"
                    print(f"{ticker:<10} {info['message_count']:<12} {info['trading_days']:<8} {date_range:<25} {'READY':<10}")
                
                print("-" * 80)
                print(f"Total simulation symbols: {len(available_tickers)}")
            
            return available_tickers
            
        except Exception as e:
            print(f"ERROR: Failed to discover simulation tickers: {e}")
            return {}
    
    def configure_trading_symbol(self, provided_symbol, available_tickers):
        """Professional symbol configuration with change option"""
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
            print(f"  {i}. {ticker} ({info['trading_days']} days, {info['message_count']} messages)")
        
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
        """Professional budget configuration"""
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
        """Professional position configuration with symbol change option"""
        if provided_position is not None:
            print(f"CURRENT POSITION: {provided_position} shares")
            return provided_position
        
        while True:
            try:
                position_input = input(f"Current {stock_symbol} position (shares, default 0, or 'change' to select different symbol): ").strip()
                
                # Allow symbol change
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
    
    def offer_post_simulation_options(self, simulator):
        """Professional post-simulation options with window configuration"""
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
        print("")
        
        while True:
            try:
                choice = input("Select option (1-4): ").strip()
                
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
                
                else:
                    print("Invalid option. Please select 1, 2, 3, or 4.")
                    
            except KeyboardInterrupt:
                print("\nExiting simulation.")
                return
    
    def configure_window_range_and_rerun(self, simulator):
        """Configure window range and rerun simulation"""
        print("\nWINDOW RANGE CONFIGURATION")
        print("=" * 40)
        
        # Get current symbol
        symbol = simulator.config.get('stock_symbol', 'UNKNOWN')
        symbol_clean = symbol.replace('S_', '') if symbol.startswith('S_') else symbol
        
        # Show current default ranges
        current_ranges = {
            'AAPL': (6, 8),
            'GERN': (2, 6),
            'TSLA': (3, 7),
            'NVDA': (4, 8),
            'MSFT': (5, 8),
            'AMZN': (4, 7),
            'GOOGL': (5, 9),
            'META': (3, 6),
        }
        
        current_min, current_max = current_ranges.get(symbol_clean, (2, 8))
        
        print(f"Symbol: {symbol_clean}")
        print(f"Current window range: {current_min}-{current_max} days")
        print("Window range must be between 2-8 days")
        print("Tip: Based on analysis, consider adjusting if performance is poor")
        print("")
        
        # Get new window range
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
        
        # Show what will change
        if min_window != current_min or max_window != current_max:
            print(f"\nWindow range change: {current_min}-{current_max} → {min_window}-{max_window} days")
        else:
            print(f"\nKeeping current window range: {min_window}-{max_window} days")
        
        confirm = input("Proceed with simulation? (y/N): ").strip().lower()
        
        if confirm == 'y':
            # Update the report generator's window range
            self.update_window_range(simulator, symbol_clean, min_window, max_window)
            
            # Reset and rerun simulation
            print(f"\nRESTARTING SIMULATION WITH {min_window}-{max_window} DAY WINDOWS")
            print("=" * 60)
            
            # Reset simulator state
            simulator.trading_executor.executed_trades = []
            simulator.daily_results = []
            simulator.trading_executor.current_budget = simulator.initial_budget
            simulator.trading_executor.current_position = 0
            simulator.trading_executor.position_entry_price = None
            simulator.trading_executor.intervals_since_last_trade = 0
            simulator.data_manager.price_history.clear()
            
            # Run simulation with new window range
            simulator.run_complete_simulation()
            
            print(f"\nSIMULATION WITH {min_window}-{max_window} DAY WINDOWS COMPLETE")
            print("Compare results above with previous simulation")
        else:
            print("Window configuration cancelled")
    
    def update_window_range(self, simulator, symbol, min_window, max_window):
        """Update the window range in report generator"""
        # Update the optimal ranges in the report generator
        if hasattr(simulator, 'report_generator'):
            # Store the new range
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
        """Professional aggressive retraining"""
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
            # Reset simulator for aggressive mode
            simulator.aggressive_mode = True
            simulator.trading_executor.executed_trades = []
            simulator.daily_results = []
            simulator.trading_executor.current_budget = simulator.initial_budget
            simulator.trading_executor.current_position = 0
            simulator.trading_executor.position_entry_price = None
            simulator.trading_executor.intervals_since_last_trade = 0
            simulator.data_manager.price_history.clear()
            
            # Run aggressive simulation
            print("\nAGGRESSIVE MODE ACTIVATED")
            simulator.run_complete_simulation()
            
            print("\nAGGRESSIVE MODE RESULTS COMPLETE")
            print("Compare the results above with your previous standard mode results.")
    
    def show_detailed_trade_log(self, simulator):
        """Professional trade log display"""
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
            
            if trade['action'] == 'BUY':
                print(f"  Cost:     ${trade['cost']:.2f}")
            else:
                print(f"  Proceeds: ${trade['proceeds']:.2f}")
                print(f"  P&L:      ${trade['pnl']:+.2f}")
            
            print(f"  Reason:   {trade['reasoning']}")
            print("")
        
        input("Press Enter to continue...")

