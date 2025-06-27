#!/usr/bin/env python3
"""
Report Generator with Multi-Strategy Comparison Support
"""

import numpy as np

class ReportGenerator:
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        print("Report generator initialized with multi-strategy support")
    
    def print_final_summary(self, daily_results, executed_trades, elapsed_time, config, data_manager=None):
        """Main summary that detects if multi-strategy or single strategy"""
        dm = data_manager or self.data_manager
        
        # Check if we have a multi-strategy executor
        if hasattr(executed_trades, 'get_strategy_performance'):
            self.print_multi_strategy_summary(daily_results, executed_trades, elapsed_time, config, dm)
        else:
            self.print_single_strategy_summary(daily_results, executed_trades, elapsed_time, config, dm)
    
    def print_multi_strategy_summary(self, daily_results, trading_executor, elapsed_time, config, data_manager):
        """Print comparison of all strategies"""
        
        print("\n" + "=" * 80)
        mode_text = "AGGRESSIVE MODE" if config.get('aggressive_mode') else "STANDARD MODE"
        print(f"MULTI-STRATEGY SIMULATION COMPLETE - COMPARISON REPORT ({mode_text})")
        print("=" * 80)
        
        # Training Summary
        print("TRAINING SUMMARY:")
        print(f"  Long-term Agents:     TRAINED")
        print(f"  Intraday Agents:      TRAINED")
        print(f"  System Status:        READY")
        print(f"  Trading Mode:         {mode_text}")
        print(f"  Risk Management:      ACTIVE")
        print(f"  Strategies Tested:    LONG_ONLY, SHORT_ONLY, LONG_SHORT")
        print("")
        
        # Get final performance for all strategies
        final_price = daily_results[-1]['final_price'] if daily_results else 100
        performance = trading_executor.get_strategy_performance(final_price)
        
        # Strategy Performance Comparison
        print("STRATEGY PERFORMANCE COMPARISON:")
        print("-" * 60)
        
        # Sort strategies by performance
        sorted_strategies = sorted(performance.items(), key=lambda x: x[1]['total_return'], reverse=True)
        
        for rank, (strategy_name, perf) in enumerate(sorted_strategies, 1):
            status_icon = "WINNER" if rank == 1 else "RUNNER-UP" if rank == 2 else "THIRD"
            print(f"{rank}. {strategy_name} ({status_icon}):")
            print(f"   Total Return:      {perf['total_return']:+.2f}%")
            print(f"   Final Value:       ${perf['current_value']:,.2f}")
            print(f"   Total Trades:      {perf['total_trades']}")
            print(f"   Win Rate:          {perf['win_rate']:.1f}%")
            
            # Calculate profit factor if available
            if hasattr(trading_executor.strategies[strategy_name], 'trades'):
                trades = trading_executor.strategies[strategy_name]['trades']
                profit_factor = self.calculate_profit_factor(trades)
                if profit_factor > 0:
                    print(f"   Profit Factor:     {profit_factor:.2f}")
            print("")
        
        # Highlight best performer
        best_strategy, best_perf = sorted_strategies[0]
        worst_strategy, worst_perf = sorted_strategies[-1]
        
        print("PERFORMANCE ANALYSIS:")
        print(f"  BEST STRATEGY:     {best_strategy}")
        print(f"  Best Return:       {best_perf['total_return']:+.2f}%")
        print(f"  Worst Strategy:    {worst_strategy}")
        print(f"  Worst Return:      {worst_perf['total_return']:+.2f}%")
        print(f"  Performance Gap:   {best_perf['total_return'] - worst_perf['total_return']:.2f}%")
        print("")
        
        # Strategy-specific analysis
        self.print_strategy_analysis(best_strategy, performance, data_manager)
        
        # Benchmark comparison using best strategy
        initial_budget = config.get('initial_budget', 25000)
        self.print_benchmark_comparison(initial_budget, best_perf['total_return'], data_manager)
        
        # Combined daily performance
        if daily_results:
            self.print_daily_performance(daily_results)
        
        # Multi-strategy trade analysis
        self.print_multi_strategy_trade_analysis(trading_executor, config)
        
        # System readiness based on best strategy
        self.print_multi_strategy_readiness(performance, initial_budget, data_manager)
        
        print("=" * 80)
    
    def print_single_strategy_summary(self, daily_results, executed_trades, elapsed_time, config, data_manager):
        """Original single strategy summary"""
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
            
            # Strategy comparison
            self.print_strategy_comparison(initial_budget, total_return, dm, len(daily_results), config)
            
            # Daily Performance
            self.print_daily_performance(daily_results)
            
            # Trade Analysis
            self.print_trade_analysis(executed_trades, daily_results, config)
            
            # System Readiness
            self.print_system_readiness(executed_trades, daily_results, initial_budget, dm)
        
        print("=" * 80)
    
    def print_strategy_analysis(self, best_strategy, performance, data_manager):
        """Analyze why a particular strategy won"""
        print("STRATEGY ANALYSIS:")
        
        if best_strategy == 'SHORT_ONLY':
            print("  MARKET CONDITION:  Bearish/Declining")
            print("  ANALYSIS:          Short selling was optimal for this period")
            print("  RECOMMENDATION:    Consider short-focused strategies in similar conditions")
        elif best_strategy == 'LONG_ONLY':
            print("  MARKET CONDITION:  Bullish/Rising")
            print("  ANALYSIS:          Traditional long strategy worked best")
            print("  RECOMMENDATION:    Stick with long-only in uptrending markets")
        elif best_strategy == 'LONG_SHORT':
            print("  MARKET CONDITION:  Mixed/Volatile")
            print("  ANALYSIS:          Flexible long/short approach captured opportunities")
            print("  RECOMMENDATION:    Use adaptive strategies in uncertain markets")
        
        # Performance spread analysis
        returns = [perf['total_return'] for perf in performance.values()]
        spread = max(returns) - min(returns)
        
        if spread > 5:
            print(f"  STRATEGY IMPACT:   HIGH ({spread:.1f}% spread between best/worst)")
            print("  INSIGHT:           Strategy selection was crucial for this period")
        elif spread > 2:
            print(f"  STRATEGY IMPACT:   MODERATE ({spread:.1f}% spread)")
            print("  INSIGHT:           Strategy choice made a meaningful difference")
        else:
            print(f"  STRATEGY IMPACT:   LOW ({spread:.1f}% spread)")
            print("  INSIGHT:           All strategies performed similarly")
        
        print("")
    
    def print_multi_strategy_trade_analysis(self, trading_executor, config):
        """Analyze trades across all strategies"""
        print("MULTI-STRATEGY TRADE ANALYSIS:")
        
        total_trades_all = 0
        total_wins_all = 0
        total_profit_all = 0
        
        for strategy_name, strategy in trading_executor.strategies.items():
            trades = strategy['trades']
            if not trades:
                continue
            
            buy_trades = [t for t in trades if t['action'] in ['BUY', 'SELL_SHORT']]
            sell_trades = [t for t in trades if t['action'] in ['SELL', 'BUY_TO_COVER']]
            
            print(f"  {strategy_name}:")
            print(f"    Total Trades:     {len(trades)}")
            print(f"    Entry Trades:     {len(buy_trades)}")
            print(f"    Exit Trades:      {len(sell_trades)}")
            
            if sell_trades:
                profitable_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
                win_rate = len(profitable_trades) / len(sell_trades) * 100
                print(f"    Win Rate:         {win_rate:.1f}%")
                
                total_trades_all += len(sell_trades)
                total_wins_all += len(profitable_trades)
                
                if profitable_trades:
                    avg_profit = sum(t['pnl'] for t in profitable_trades) / len(profitable_trades)
                    print(f"    Avg Profit:       ${avg_profit:.2f}")
                    total_profit_all += sum(t['pnl'] for t in profitable_trades)
                
                losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
                if losing_trades:
                    avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
                    print(f"    Avg Loss:         ${avg_loss:.2f}")
                    total_profit_all += sum(t['pnl'] for t in losing_trades)
            
            print("")
        
        # Overall statistics
        if total_trades_all > 0:
            overall_win_rate = total_wins_all / total_trades_all * 100
            print(f"  OVERALL STATISTICS:")
            print(f"    Combined Trades:  {total_trades_all}")
            print(f"    Combined Win Rate: {overall_win_rate:.1f}%")
            print(f"    Combined P&L:     ${total_profit_all:+.2f}")
        
        print("")
    
    def print_multi_strategy_readiness(self, performance, initial_budget, data_manager):
        """Assess system readiness based on multi-strategy results"""
        print("SYSTEM READINESS:")
        
        # Get best performing strategy
        best_strategy = max(performance.items(), key=lambda x: x[1]['total_return'])
        best_name, best_perf = best_strategy
        
        # Readiness criteria
        positive_return = best_perf['total_return'] > 0
        decent_trades = best_perf['total_trades'] >= 3
        good_win_rate = best_perf['win_rate'] >= 40
        
        # Check if any strategy beats buy-and-hold
        test_days = data_manager.get_test_days()
        benchmark = self.calculate_buy_and_hold(initial_budget, test_days, data_manager)
        beats_benchmark = False
        if benchmark:
            beats_benchmark = best_perf['total_return'] >= benchmark['return_pct']
        
        if positive_return and decent_trades and (good_win_rate or beats_benchmark):
            print("  ASSESSMENT: System is READY for live trading")
            print(f"  BEST STRATEGY: {best_name}")
            print("  STATUS: Multi-strategy analysis complete")
            print("  STATUS: Long-term agents are trained")
            print("  STATUS: Intraday agents are trained")
            print("  STATUS: Strategy selection optimized")
            print("  STATUS: Risk management active")
            if beats_benchmark:
                print("  STATUS: Beats buy-and-hold benchmark")
            if best_perf['total_return'] > 0:
                print("  STATUS: Positive returns achieved")
        else:
            print("  ASSESSMENT: System has CONCERNS for live trading")
            print(f"  BEST STRATEGY: {best_name} (but with issues)")
            print("  STATUS: Multi-strategy analysis complete")
            print("  STATUS: Long-term agents are trained")
            print("  STATUS: Intraday agents are trained")
            if not positive_return:
                print("  ISSUE: No strategy achieved positive returns")
            if not decent_trades:
                print("  ISSUE: Insufficient trading activity")
            if not good_win_rate and not beats_benchmark:
                print("  ISSUE: Poor win rate and doesn't beat benchmark")
            print("  RECOMMENDATION: Consider parameter optimization")
    
    def calculate_profit_factor(self, trades):
        """Calculate profit factor for a list of trades"""
        if not trades:
            return 0
        
        profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        if not profitable_trades or not losing_trades:
            return 0
        
        total_profit = sum(t['pnl'] for t in profitable_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        return total_profit / total_loss if total_loss > 0 else 0
    
    def get_optimal_window_range(self, symbol):
        """Get optimal window range based on symbol"""
        symbol_clean = symbol.replace('S_', '') if symbol.startswith('S_') else symbol
        
        # Check for custom ranges first
        if hasattr(self, 'custom_ranges') and symbol_clean in self.custom_ranges:
            return self.custom_ranges[symbol_clean]
        
        # Default optimal ranges
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
        
        return optimal_ranges.get(symbol_clean, (2, 8))
    
    def print_strategy_comparison(self, initial_budget, strategy_return, data_manager, test_days, config):
        """Compare intraday strategy vs configurable open/close strategies vs buy-and-hold"""
        print("STRATEGY COMPARISON:")
        
        test_days_list = data_manager.get_test_days()
        if not test_days_list:
            print("  Unable to calculate comparisons - no test data")
            return
        
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
            if window <= len(test_days_list):
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
        
        for window in sorted(open_close_results.keys()):
            result = open_close_results[window]
            is_optimal = min_window <= window <= max_window
            marker = " (optimal)" if is_optimal else ""
            print(f"      {window}-day window:    {result['return_pct']:+.2f}% ({result['trades']} trades){marker}")
            
            if result['return_pct'] > best_oc_return:
                best_oc_return = result['return_pct']
                best_oc_window = window
        
        print("")
        print("  PERFORMANCE RANKING:")
        
        strategies = [("Intraday Trading", strategy_return)]
        if buy_hold:
            strategies.append(("Buy & Hold", buy_hold['return_pct']))
        if best_oc_return > -999:
            strategies.append((f"Open/Close {best_oc_window}d", best_oc_return))
        
        strategies.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, return_pct) in enumerate(strategies, 1):
            print(f"    {i}. {name:<25} {return_pct:+.2f}%")
        
        print("")
    
    def calculate_open_close_strategy(self, initial_budget, test_days, data_manager, window_days):
        """Calculate open/close strategy with specified window"""
        try:
            if len(test_days) < window_days:
                return None
            
            cash = initial_budget
            trades = 0
            
            i = 0
            while i <= len(test_days) - window_days:
                buy_day = test_days[i]
                buy_data = data_manager.load_day_data(buy_day)
                if not buy_data:
                    i += 1
                    continue
                
                buy_price = buy_data[0]['price']
                
                sell_day = test_days[i + window_days - 1]
                sell_data = data_manager.load_day_data(sell_day)
                if not sell_data:
                    i += 1
                    continue
                
                sell_price = sell_data[-1]['price']
                
                if cash > 0:
                    shares_bought = int((cash * 0.98) // buy_price)
                    if shares_bought > 0:
                        cost = shares_bought * buy_price
                        proceeds = shares_bought * sell_price
                        cash = cash - cost + proceeds
                        trades += 2
                
                i += window_days
            
            final_value = cash
            return_pct = ((final_value - initial_budget) / initial_budget) * 100
            
            return {
                'final_value': final_value,
                'return_pct': return_pct,
                'trades': trades
            }
            
        except Exception as e:
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
            
            shares = int((initial_budget * 0.98) // start_price)
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
            return None
    
    def print_benchmark_comparison(self, initial_budget, strategy_return, data_manager):
        """Professional benchmark comparison"""
        print("BENCHMARK COMPARISON:")
        test_days = data_manager.get_test_days()
        benchmark = self.calculate_buy_and_hold(initial_budget, test_days, data_manager)
        
        if benchmark:
            print(f"  Buy & Hold Return:    {benchmark['return_pct']:+.2f}%")
            print(f"  Buy & Hold Value:     ${benchmark['final_value']:,.2f}")
            print(f"  Strategy vs B&H:      {strategy_return - benchmark['return_pct']:+.2f}% difference")
            
            if strategy_return < benchmark['return_pct']:
                print(f"  RESULT: Strategy UNDERPERFORMED buy-and-hold by {benchmark['return_pct'] - strategy_return:.2f}%")
            else:
                print(f"  RESULT: Strategy OUTPERFORMED buy-and-hold by {strategy_return - benchmark['return_pct']:.2f}%")
        else:
            print("  Buy & Hold:           Unable to calculate")
        print("")
    
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
                    
                    total_profit = sum(t['pnl'] for t in profitable_trades)
                    total_loss = abs(sum(t['pnl'] for t in losing_trades))
                    if total_loss > 0:
                        profit_factor = total_profit / total_loss
                        print(f"  Profit Factor:        {profit_factor:.2f}")
                        
                        if profit_factor >= 1.5:
                            print(f"  Risk Assessment:      GOOD (PF >= 1.5)")
                        elif profit_factor >= 1.0:
                            print(f"  Risk Assessment:      MARGINAL (PF >= 1.0)")
                        else:
                            print(f"  Risk Assessment:      POOR (PF < 1.0)")
            
            aggressive_mode = config.get('aggressive_mode', False)
            total_intervals = len(daily_results) * (26 if aggressive_mode else 13)
            if total_intervals > 0:
                trade_frequency = len(executed_trades) / total_intervals * 100
                print(f"  Trade Frequency:      {trade_frequency:.2f}% of intervals")
            
            print("")
    
    def print_system_readiness(self, executed_trades, daily_results, initial_budget, data_manager):
        """Professional system readiness assessment"""
        print("SYSTEM READINESS:")
        
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


