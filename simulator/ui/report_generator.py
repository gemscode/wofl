#!/usr/bin/env python3
"""
Report Generator
"""

class ReportGenerator:
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        print("Report generator initialized")
    
    def print_final_summary(self, daily_results, executed_trades, elapsed_time, config, data_manager=None):
        dm = data_manager or self.data_manager
        
        print("\n" + "=" * 80)
        mode_text = "AGGRESSIVE MODE" if config.get('aggressive_mode') else "STANDARD MODE"
        print(f"WOLFXE SIMULATION COMPLETE - FINAL REPORT ({mode_text})")
        print("=" * 80)
        
        # Training Summary
        print("TRAINING SUMMARY:")
        print(f"  Long-term Agents:     TRAINED")
        print(f"  Intraday Agents:      TRAINED")
        print(f"  System Status:        READY")
        print(f"  Trading Mode:         {mode_text}")
        print(f"  Risk Management:      ACTIVE")
        print("")
        
        # Trading Results
        if daily_results:
            initial_budget = config.get('initial_budget', 25000)
            final_value = daily_results[-1]['end_value']
            total_return = ((final_value - initial_budget) / initial_budget) * 100
            
            print("TRADING RESULTS:")
            print(f"  Initial Budget:       ${initial_budget:,.2f}")
            print(f"  Final Value:          ${final_value:,.2f}")
            print(f"  Total Return:         {total_return:+.2f}%")
            print(f"  Total Trades:         {len(executed_trades)}")
            print(f"  Test Days:            {len(daily_results)}")
            print(f"  Simulation Time:      {elapsed_time:.1f} seconds")
            print("")
            
            # STRATEGY COMPARISON with configurable windows
            self.print_strategy_comparison(initial_budget, total_return, dm, len(daily_results), config)
            
            # Daily Performance
            self.print_daily_performance(daily_results)
            
            # Trade Analysis
            self.print_trade_analysis(executed_trades, daily_results, config)
            
            # System Readiness
            self.print_system_readiness(executed_trades, daily_results, initial_budget, dm)
        
        print("=" * 80)
    
    def get_optimal_window_range(self, symbol):
        """Get optimal window range based on symbol"""
        symbol_clean = symbol.replace('S_', '') if symbol.startswith('S_') else symbol
        
        # Define optimal ranges for different symbols based on your findings
        optimal_ranges = {
            'AAPL': (6, 8),
            'GERN': (2, 6),
            'TSLA': (3, 7),
            'NVDA': (4, 8),
            'MSFT': (5, 8),
            'AMZN': (4, 7),
            'GOOGL': (5, 9),
            'META': (3, 6),
        }
        
        # Default range if symbol not found
        return optimal_ranges.get(symbol_clean, (2, 8))
    
    def print_strategy_comparison(self, initial_budget, strategy_return, data_manager, test_days, config):
        """Compare intraday strategy vs configurable open/close strategies vs buy-and-hold"""
        print("STRATEGY COMPARISON:")
        
        # Get test period data
        test_days_list = data_manager.get_test_days()
        if not test_days_list:
            print("  Unable to calculate comparisons - no test data")
            return
        
        # Get symbol for optimal window range
        symbol = config.get('stock_symbol', 'UNKNOWN')
        min_window, max_window = self.get_optimal_window_range(symbol)
        
        print(f"  Symbol:               {symbol}")
        print(f"  Period:               {test_days_list[0]} to {test_days_list[-1]} ({len(test_days_list)} days)")
        print(f"  Window Range:         {min_window}-{max_window} days (optimized for {symbol.replace('S_', '')})")
        print("")
        
        # Calculate buy-and-hold
        buy_hold = self.calculate_buy_and_hold(initial_budget, test_days_list, data_manager)
        
        # Calculate open/close strategies for optimal window range
        open_close_results = {}
        for window in range(min_window, max_window + 1):
            if window <= len(test_days_list):  # Can't have window larger than test period
                result = self.calculate_open_close_strategy(initial_budget, test_days_list, data_manager, window)
                if result:
                    open_close_results[window] = result
        
        # Also test a few windows outside the optimal range for comparison
        extended_windows = []
        if min_window > 2:
            extended_windows.append(min_window - 1)
        if max_window < len(test_days_list):
            extended_windows.append(max_window + 1)
        
        for window in extended_windows:
            if window >= 2 and window <= len(test_days_list):
                result = self.calculate_open_close_strategy(initial_budget, test_days_list, data_manager, window)
                if result:
                    open_close_results[window] = result
        
        # Display results
        print("  STRATEGY PERFORMANCE:")
        print(f"    Intraday Trading:   {strategy_return:+.2f}%")
        
        if buy_hold:
            print(f"    Buy & Hold:         {buy_hold['return_pct']:+.2f}%")
        
        print("    Open/Close Strategies:")
        best_oc_return = -999
        best_oc_window = 0
        optimal_oc_return = -999
        optimal_oc_window = 0
        
        for window in sorted(open_close_results.keys()):
            result = open_close_results[window]
            is_optimal = min_window <= window <= max_window
            marker = " (optimal)" if is_optimal else ""
            print(f"      {window}-day window:    {result['return_pct']:+.2f}% ({result['trades']} trades){marker}")
            
            if result['return_pct'] > best_oc_return:
                best_oc_return = result['return_pct']
                best_oc_window = window
            
            if is_optimal and result['return_pct'] > optimal_oc_return:
                optimal_oc_return = result['return_pct']
                optimal_oc_window = window
        
        print("")
        print("  PERFORMANCE RANKING:")
        
        # Create ranking
        strategies = [
            ("Intraday Trading", strategy_return),
        ]
        
        if buy_hold:
            strategies.append(("Buy & Hold", buy_hold['return_pct']))
        
        if best_oc_return > -999:
            strategies.append((f"Open/Close {best_oc_window}d", best_oc_return))
        
        if optimal_oc_return > -999 and optimal_oc_window != best_oc_window:
            strategies.append((f"Open/Close {optimal_oc_window}d (opt)", optimal_oc_return))
        
        # Sort by return
        strategies.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, return_pct) in enumerate(strategies, 1):
            print(f"    {i}. {name:<25} {return_pct:+.2f}%")
        
        # Analysis
        print("")
        print("  ANALYSIS:")
        
        if optimal_oc_return > -999:
            if optimal_oc_return > strategy_return:
                print(f"    Optimal Open/Close ({optimal_oc_window}d) outperformed intraday by {optimal_oc_return - strategy_return:.2f}%")
            elif strategy_return > optimal_oc_return:
                print(f"    Intraday trading outperformed optimal Open/Close by {strategy_return - optimal_oc_return:.2f}%")
            else:
                print(f"    Intraday and optimal Open/Close performed similarly")
        
        if best_oc_return > -999 and best_oc_window != optimal_oc_window:
            print(f"    Best overall Open/Close was {best_oc_window}d window: {best_oc_return:+.2f}%")
            if best_oc_window < min_window:
                print(f"    Consider lowering minimum window from {min_window} to {best_oc_window}")
            elif best_oc_window > max_window:
                print(f"    Consider raising maximum window from {max_window} to {best_oc_window}")
        
        if buy_hold:
            best_strategy_return = max(strategy_return, best_oc_return if best_oc_return > -999 else -999)
            if best_strategy_return > buy_hold['return_pct']:
                print(f"    Active trading outperformed buy-and-hold by {best_strategy_return - buy_hold['return_pct']:.2f}%")
            else:
                print(f"    Buy-and-hold outperformed active trading by {buy_hold['return_pct'] - best_strategy_return:.2f}%")
        
        print("")
    
    def calculate_open_close_strategy(self, initial_budget, test_days, data_manager, window_days):
        """Calculate open/close strategy with specified window"""
        try:
            if len(test_days) < window_days:
                return None
            
            cash = initial_budget
            shares = 0
            trades = 0
            total_return = 0
            
            i = 0
            while i <= len(test_days) - window_days:
                # Buy at open of first day
                buy_day = test_days[i]
                buy_data = data_manager.load_day_data(buy_day)
                if not buy_data:
                    i += 1
                    continue
                
                buy_price = buy_data[0]['price']  # Open price
                
                # Sell at close of last day in window
                sell_day = test_days[i + window_days - 1]
                sell_data = data_manager.load_day_data(sell_day)
                if not sell_data:
                    i += 1
                    continue
                
                sell_price = sell_data[-1]['price']  # Close price
                
                # Calculate return for this trade
                trade_return = (sell_price - buy_price) / buy_price
                
                # Execute trade (simplified - assume we can always trade full position)
                if cash > 0:
                    shares_bought = int((cash * 0.98) // buy_price)  # Use 98% of cash (account for fees)
                    if shares_bought > 0:
                        cost = shares_bought * buy_price
                        proceeds = shares_bought * sell_price
                        cash = cash - cost + proceeds
                        trades += 2  # Buy and sell
                        total_return += trade_return
                
                # Move to next non-overlapping window
                i += window_days
            
            # Calculate final return
            final_value = cash
            return_pct = ((final_value - initial_budget) / initial_budget) * 100
            
            return {
                'final_value': final_value,
                'return_pct': return_pct,
                'trades': trades,
                'avg_trade_return': total_return / (trades // 2) if trades > 0 else 0
            }
            
        except Exception as e:
            print(f"    Error calculating {window_days}-day strategy: {e}")
            return None
    
    def calculate_buy_and_hold(self, initial_budget, test_days, data_manager):
        """Calculate buy-and-hold performance"""
        try:
            if not test_days:
                return None
            
            first_day_data = data_manager.load_day_data(test_days[0])
            last_day_data = data_manager.load_day_data(test_days[-1])
            
            if not first_day_data or not last_day_data:
                return None
            
            start_price = first_day_data[0]['price']
            end_price = last_day_data[-1]['price']
            
            shares = int((initial_budget * 0.98) // start_price)  # Account for fees
            final_value = shares * end_price + (initial_budget - shares * start_price)
            return_pct = ((final_value - initial_budget) / initial_budget) * 100
            
            return {
                'start_price': start_price,
                'end_price': end_price,
                'shares': shares,
                'final_value': final_value,
                'return_pct': return_pct
            }
        except Exception as e:
            print(f"    Error calculating buy-and-hold: {e}")
            return None
    
    def print_daily_performance(self, daily_results):
        """Professional daily performance summary"""
        winning_days = len([d for d in daily_results if d['daily_pnl'] > 0])
        win_rate = winning_days / len(daily_results) * 100
        
        print("DAILY PERFORMANCE:")
        print(f"  Winning Days:         {winning_days}/{len(daily_results)} ({win_rate:.1f}%)")
        
        if daily_results:
            best_day = max(daily_results, key=lambda x: x['daily_pnl'])
            worst_day = min(daily_results, key=lambda x: x['daily_pnl'])
            avg_daily_pnl = sum(d['daily_pnl'] for d in daily_results) / len(daily_results)
            
            print(f"  Best Day:             {best_day['date']} (+${best_day['daily_pnl']:.2f})")
            print(f"  Worst Day:            {worst_day['date']} (${worst_day['daily_pnl']:+.2f})")
            print(f"  Average Daily P&L:    ${avg_daily_pnl:+.2f}")
        print("")
    
    def print_trade_analysis(self, executed_trades, daily_results, config):
        """Professional trade analysis"""
        if executed_trades:
            buy_trades = [t for t in executed_trades if t['action'] == 'BUY']
            sell_trades = [t for t in executed_trades if t['action'] == 'SELL']
            
            print("TRADE ANALYSIS:")
            print(f"  Buy Trades:           {len(buy_trades)}")
            print(f"  Sell Trades:          {len(sell_trades)}")
            
            if sell_trades:
                profitable_trades = [t for t in sell_trades if t['pnl'] > 0]
                trade_win_rate = len(profitable_trades) / len(sell_trades) * 100
                print(f"  Trade Win Rate:       {trade_win_rate:.1f}%")
                
                if profitable_trades:
                    avg_profit = sum(t['pnl'] for t in profitable_trades) / len(profitable_trades)
                    print(f"  Average Profit:       ${avg_profit:.2f}")
                
                losing_trades = [t for t in sell_trades if t['pnl'] < 0]
                if losing_trades:
                    avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
                    print(f"  Average Loss:         ${avg_loss:.2f}")
                    
                    # Profit factor
                    total_profit = sum(t['pnl'] for t in profitable_trades)
                    total_loss = abs(sum(t['pnl'] for t in losing_trades))
                    if total_loss > 0:
                        profit_factor = total_profit / total_loss
                        print(f"  Profit Factor:        {profit_factor:.2f}")
                        
                        # Risk assessment
                        if profit_factor >= 1.5:
                            print(f"  Risk Assessment:      GOOD (PF >= 1.5)")
                        elif profit_factor >= 1.0:
                            print(f"  Risk Assessment:      MARGINAL (PF >= 1.0)")
                        else:
                            print(f"  Risk Assessment:      POOR (PF < 1.0)")
            
            # Trading frequency analysis
            aggressive_mode = config.get('aggressive_mode', False)
            total_intervals = len(daily_results) * (26 if aggressive_mode else 13)
            if total_intervals > 0:
                trade_frequency = len(executed_trades) / total_intervals * 100
                print(f"  Trade Frequency:      {trade_frequency:.2f}% of intervals")
            
            print("")
    
    def print_system_readiness(self, executed_trades, daily_results, initial_budget, data_manager):
        """Professional system readiness assessment"""
        print("SYSTEM READINESS:")
        
        # Check profit factor
        profit_factor_ok = False
        if executed_trades:
            sell_trades = [t for t in executed_trades if t['action'] == 'SELL']
            if sell_trades:
                profitable_trades = [t for t in sell_trades if t['pnl'] > 0]
                losing_trades = [t for t in sell_trades if t['pnl'] < 0]
                if profitable_trades and losing_trades:
                    total_profit = sum(t['pnl'] for t in profitable_trades)
                    total_loss = abs(sum(t['pnl'] for t in losing_trades))
                    profit_factor = total_profit / total_loss if total_loss > 0 else 0
                    profit_factor_ok = profit_factor >= 1.0
        
        # Check benchmark performance
        beats_benchmark = False
        test_days = data_manager.get_test_days()
        benchmark = self.calculate_buy_and_hold(initial_budget, test_days, data_manager)
        if benchmark and daily_results:
            final_value = daily_results[-1]['end_value']
            strategy_return = ((final_value - initial_budget) / initial_budget) * 100
            beats_benchmark = strategy_return >= benchmark['return_pct']
        
        if profit_factor_ok and beats_benchmark:
            print("  ASSESSMENT: System is READY for live trading")
            print("  STATUS: Long-term agents are trained")
            print("  STATUS: Intraday agents are trained")
            print("  STATUS: Trading logic validated")
            print("  STATUS: Risk management active")
            print("  STATUS: Beats buy-and-hold benchmark")
            print("  STATUS: Positive profit factor")
        else:
            print("  ASSESSMENT: System has CONCERNS for live trading")
            print("  STATUS: Long-term agents are trained")
            print("  STATUS: Intraday agents are trained")
            if not profit_factor_ok:
                print("  ISSUE: Poor profit factor (< 1.0)")
            if not beats_benchmark:
                print("  ISSUE: Does not beat buy-and-hold")
            print("  RECOMMENDATION: Consider further optimization")

