#!/usr/bin/env python3
"""
Trading Executor with Optimization and Short Selling Support
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import numpy as np

class TradingExecutor:
    
    def __init__(self, base_symbol, initial_budget, current_position=0, trading_mode='LONG_SHORT'):
        self.base_symbol = base_symbol
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.current_position = current_position
        self.position_entry_price = None
        self.trading_mode = trading_mode
        
        # Short selling parameters
        self.margin_requirement = 0.5
        self.available_margin = initial_budget * 2
        
        self.is_system_ready = False
        self.executed_trades = []
        
        self.intervals_since_last_trade = 0
        self.max_intervals_without_trade = 20
        
        # Optimized parameters (defaults)
        self.optimized_params = {
            'momentum_threshold': 0.01,
            'sma_periods': (5, 10, 20),
            'profit_target': 0.05,
            'stop_loss': 0.025,
            'position_size': 0.6
        }
        
        print(f"TradingExecutor initialized for {base_symbol} - Mode: {trading_mode}")
    
    def initialize_wolfxe_system(self):
        self.is_system_ready = True
        print(f"SYSTEM: Trading system initialized - {self.trading_mode} mode")
        return True
    
    def initialize_and_train(self, training_data):
        print(f"TRAINING: Initializing {self.trading_mode} trading system...")
        print(f"TRAINING: Processing {len(training_data)} data points...")
        
        # REAL TRAINING: Test different parameter combinations
        best_params = self.optimize_strategy_parameters(training_data)
        
        # Apply best parameters
        self.apply_optimized_parameters(best_params)
        
        self.save_training_metadata(len(training_data), len(training_data))
        self.is_system_ready = True
        print(f"TRAINING: {self.trading_mode} system ready with optimized parameters")
        return True
    
    def optimize_strategy_parameters(self, training_data):
        """Find optimal parameters through backtesting"""
        print("OPTIMIZATION: Testing parameter combinations...")
        
        # Parameter ranges to test
        param_grid = {
            'momentum_threshold': [0.005, 0.01, 0.015, 0.02],
            'sma_periods': [(5, 10, 20), (3, 7, 15), (5, 15, 30), (7, 14, 28)],
            'profit_target': [0.03, 0.05, 0.08],
            'stop_loss': [0.015, 0.025, 0.035],
            'position_size': [0.5, 0.6, 0.7, 0.8]
        }
        
        best_score = -999
        best_params = None
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        print(f"OPTIMIZATION: Testing {total_combinations} parameter combinations")
        
        combination_count = 0
        for momentum in param_grid['momentum_threshold']:
            for sma in param_grid['sma_periods']:
                for profit in param_grid['profit_target']:
                    for stop in param_grid['stop_loss']:
                        for pos_size in param_grid['position_size']:
                            combination_count += 1
                            
                            params = {
                                'momentum_threshold': momentum,
                                'sma_periods': sma,
                                'profit_target': profit,
                                'stop_loss': stop,
                                'position_size': pos_size
                            }
                            
                            # Backtest this parameter combination
                            score = self.backtest_parameters(training_data, params)
                            
                            if score > best_score:
                                best_score = score
                                best_params = params
                            
                            # Progress update
                            if combination_count % 50 == 0:
                                progress = (combination_count / total_combinations) * 100
                                print(f"OPTIMIZATION: {progress:.1f}% complete, best score: {best_score:.3f}")
        
        print(f"OPTIMIZATION: Complete! Best score: {best_score:.3f}")
        print(f"OPTIMIZATION: Best parameters: {best_params}")
        return best_params
    
    def backtest_parameters(self, training_data, params):
        """Backtest a specific parameter combination"""
        if len(training_data) < 100:
            return -999
        
        portfolio_value = self.initial_budget
        trades = 0
        wins = 0
        position = 0
        entry_price = 0
        
        # Convert DataFrame to price array for faster processing
        prices = training_data['Close'].values
        
        for i in range(50, len(prices) - 10):
            current_price = prices[i]
            price_window = prices[i-50:i]
            
            # Generate signal with these parameters
            signal = self.generate_signal_with_params(price_window, params)
            
            # Execute trades based on signal
            if signal == 'BUY' and position == 0:
                position = int((portfolio_value * params['position_size']) // current_price)
                entry_price = current_price
                portfolio_value -= position * current_price
                trades += 1
                
            elif signal == 'SELL' and position > 0:
                proceeds = position * current_price
                portfolio_value += proceeds
                
                # Calculate P&L
                pnl_pct = (current_price - entry_price) / entry_price
                if pnl_pct > 0:
                    wins += 1
                
                position = 0
                trades += 1
                
            elif signal == 'SELL_SHORT' and position == 0 and self.trading_mode in ['SHORT_ONLY', 'LONG_SHORT']:
                position = -int((portfolio_value * params['position_size']) // current_price)
                entry_price = current_price
                margin_required = abs(position) * current_price * self.margin_requirement
                portfolio_value -= margin_required
                trades += 1
                
            elif signal == 'BUY_TO_COVER' and position < 0:
                cost = abs(position) * current_price
                margin_returned = abs(position) * entry_price * self.margin_requirement
                portfolio_value = portfolio_value - cost + margin_returned
                
                # Calculate short P&L
                pnl_pct = (entry_price - current_price) / entry_price
                if pnl_pct > 0:
                    wins += 1
                
                position = 0
                trades += 1
        
        # Close any remaining position
        if position != 0:
            if position > 0:
                portfolio_value += position * prices[-1]
            else:
                cost = abs(position) * prices[-1]
                margin_returned = abs(position) * entry_price * self.margin_requirement
                portfolio_value = portfolio_value - cost + margin_returned
        
        # Calculate score metrics
        total_return = (portfolio_value - self.initial_budget) / self.initial_budget
        win_rate = wins / max(1, trades // 2) if trades > 0 else 0
        trade_frequency = trades / len(prices) if len(prices) > 0 else 0
        
        # Combined score: return * win_rate * trade_activity_factor
        activity_factor = min(trade_frequency * 1000, 1.0)  # Normalize trade frequency
        score = total_return * win_rate * (0.5 + 0.5 * activity_factor)
        
        return score
    
    def generate_signal_with_params(self, prices, params):
        """Generate trading signal using specific parameters"""
        if len(prices) < max(params['sma_periods']):
            return 'HOLD'
        
        sma_short, sma_med, sma_long = params['sma_periods']
        
        sma_s = np.mean(prices[-sma_short:])
        sma_m = np.mean(prices[-sma_med:])
        sma_l = np.mean(prices[-sma_long:])
        
        momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        if self.trading_mode == 'LONG_ONLY':
            if sma_s > sma_m > sma_l and momentum > params['momentum_threshold']:
                return 'BUY'
            elif momentum < -params['momentum_threshold']:
                return 'SELL'
        elif self.trading_mode == 'SHORT_ONLY':
            if sma_s < sma_m < sma_l and momentum < -params['momentum_threshold']:
                return 'SELL_SHORT'
            elif momentum > params['momentum_threshold']:
                return 'BUY_TO_COVER'
        elif self.trading_mode == 'LONG_SHORT':
            if sma_s > sma_m > sma_l and momentum > params['momentum_threshold']:
                return 'BUY'
            elif sma_s < sma_m < sma_l and momentum < -params['momentum_threshold']:
                return 'SELL_SHORT'
            elif abs(momentum) > params['momentum_threshold']:
                return 'SELL' if momentum < 0 else 'BUY_TO_COVER'
        
        return 'HOLD'
    
    def apply_optimized_parameters(self, params):
        """Apply optimized parameters to the trading system"""
        if params:
            self.optimized_params = params
            print(f"OPTIMIZATION: Applied parameters: {params}")
    
    def calculate_portfolio_value(self, current_price):
        """Calculate portfolio value including short positions"""
        if self.current_position == 0:
            return self.current_budget
        elif self.current_position > 0:
            return self.current_budget + (self.current_position * current_price)
        else:
            if self.position_entry_price:
                entry_value = abs(self.current_position) * self.position_entry_price
                current_value = abs(self.current_position) * current_price
                short_pnl = entry_value - current_value
                return self.current_budget + short_pnl
            return self.current_budget
    
    def generate_trading_recommendation(self, price_history):
        """Generate recommendation based on trading mode and optimized parameters"""
        if not self.is_system_ready or len(price_history) < 10:
            return {'action': 'HOLD', 'rationale': 'System not ready'}
        
        self.intervals_since_last_trade += 1
        
        if self.trading_mode == 'LONG_ONLY':
            return self.long_only_strategy(price_history)
        elif self.trading_mode == 'SHORT_ONLY':
            return self.short_only_strategy(price_history)
        elif self.trading_mode == 'LONG_SHORT':
            return self.long_short_strategy(price_history)
        else:
            return self.long_only_strategy(price_history)
    
    def long_only_strategy(self, price_history):
        """Optimized long-only strategy"""
        if len(price_history) < 30:
            return {'action': 'HOLD', 'rationale': 'Building price history'}
        
        prices = [p['price'] for p in list(price_history)[-30:]]
        current_price = prices[-1]
        
        sma_short, sma_med, sma_long = self.optimized_params['sma_periods']
        
        if len(prices) >= sma_long:
            sma_s = sum(prices[-sma_short:]) / sma_short
            sma_m = sum(prices[-sma_med:]) / sma_med
            sma_l = sum(prices[-sma_long:]) / sma_long
        else:
            return {'action': 'HOLD', 'rationale': 'Insufficient data for SMA calculation'}
        
        momentum = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        trend_up = sma_s > sma_m > sma_l
        trend_down = sma_s < sma_m < sma_l
        
        if self.current_position == 0:
            should_buy = (
                trend_up and 
                momentum > self.optimized_params['momentum_threshold']
            ) or (
                self.intervals_since_last_trade > self.max_intervals_without_trade and
                momentum > self.optimized_params['momentum_threshold'] / 2
            )
            
            if should_buy:
                reason = "Forced buy" if self.intervals_since_last_trade > self.max_intervals_without_trade else f"Long signal: momentum {momentum:.2%}"
                return {'action': 'BUY', 'rationale': reason}
        
        elif self.current_position > 0:
            if self.position_entry_price:
                pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price
                
                if pnl_pct > self.optimized_params['profit_target']:
                    return {'action': 'SELL', 'rationale': f'Profit target: {pnl_pct:.2%}'}
                if pnl_pct < -self.optimized_params['stop_loss']:
                    return {'action': 'SELL', 'rationale': f'Stop loss: {pnl_pct:.2%}'}
            
            if trend_down and momentum < -self.optimized_params['momentum_threshold']:
                return {'action': 'SELL', 'rationale': f'Long exit: downtrend {momentum:.2%}'}
        
        return {'action': 'HOLD', 'rationale': f'Long-only waiting (intervals: {self.intervals_since_last_trade})'}
    
    def short_only_strategy(self, price_history):
        """Optimized short-only strategy"""
        if len(price_history) < 30:
            return {'action': 'HOLD', 'rationale': 'Building price history'}
        
        prices = [p['price'] for p in list(price_history)[-30:]]
        current_price = prices[-1]
        
        sma_short, sma_med, sma_long = self.optimized_params['sma_periods']
        
        if len(prices) >= sma_long:
            sma_s = sum(prices[-sma_short:]) / sma_short
            sma_m = sum(prices[-sma_med:]) / sma_med
            sma_l = sum(prices[-sma_long:]) / sma_long
        else:
            return {'action': 'HOLD', 'rationale': 'Insufficient data for SMA calculation'}
        
        momentum = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        trend_down = sma_s < sma_m < sma_l
        trend_up = sma_s > sma_m > sma_l
        
        recent_high = max(prices[-10:])
        price_from_high = (current_price - recent_high) / recent_high
        
        if self.current_position == 0:
            should_short = (
                trend_down and 
                momentum < -self.optimized_params['momentum_threshold'] and
                price_from_high < -0.02
            ) or (
                self.intervals_since_last_trade > self.max_intervals_without_trade and
                momentum < -self.optimized_params['momentum_threshold'] / 2
            )
            
            if should_short:
                reason = "Forced short" if self.intervals_since_last_trade > self.max_intervals_without_trade else f"Short signal: momentum {momentum:.2%}, down {price_from_high:.2%}"
                return {'action': 'SELL_SHORT', 'rationale': reason}
        
        elif self.current_position < 0:
            if self.position_entry_price:
                short_pnl_pct = (self.position_entry_price - current_price) / self.position_entry_price
                
                if short_pnl_pct > self.optimized_params['profit_target']:
                    return {'action': 'BUY_TO_COVER', 'rationale': f'Short profit: {short_pnl_pct:.2%}'}
                if short_pnl_pct < -self.optimized_params['stop_loss']:
                    return {'action': 'BUY_TO_COVER', 'rationale': f'Short stop loss: {short_pnl_pct:.2%}'}
            
            if trend_up and momentum > self.optimized_params['momentum_threshold']:
                return {'action': 'BUY_TO_COVER', 'rationale': f'Cover on reversal: {momentum:.2%}'}
            
            if self.intervals_since_last_trade > self.max_intervals_without_trade * 1.5:
                return {'action': 'BUY_TO_COVER', 'rationale': 'Forced cover - held too long'}
        
        return {'action': 'HOLD', 'rationale': f'Short-only waiting (intervals: {self.intervals_since_last_trade})'}
    
    def long_short_strategy(self, price_history):
        """Optimized combined long/short strategy"""
        if len(price_history) < 30:
            return {'action': 'HOLD', 'rationale': 'Building price history'}
        
        prices = [p['price'] for p in list(price_history)[-30:]]
        current_price = prices[-1]
        
        sma_short, sma_med, sma_long = self.optimized_params['sma_periods']
        
        if len(prices) >= sma_long:
            sma_s = sum(prices[-sma_short:]) / sma_short
            sma_m = sum(prices[-sma_med:]) / sma_med
            sma_l = sum(prices[-sma_long:]) / sma_long
        else:
            return {'action': 'HOLD', 'rationale': 'Insufficient data for SMA calculation'}
        
        momentum = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        trend_up = sma_s > sma_m > sma_l
        trend_down = sma_s < sma_m < sma_l
        
        if self.current_position == 0:
            if trend_up and momentum > self.optimized_params['momentum_threshold']:
                return {'action': 'BUY', 'rationale': f'Long signal: momentum {momentum:.2%}'}
            elif trend_down and momentum < -self.optimized_params['momentum_threshold']:
                return {'action': 'SELL_SHORT', 'rationale': f'Short signal: momentum {momentum:.2%}'}
            elif self.intervals_since_last_trade > self.max_intervals_without_trade:
                if momentum > 0:
                    return {'action': 'BUY', 'rationale': 'Forced long'}
                else:
                    return {'action': 'SELL_SHORT', 'rationale': 'Forced short'}
        
        elif self.current_position > 0:
            if self.position_entry_price:
                pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price
                if pnl_pct > self.optimized_params['profit_target'] or pnl_pct < -self.optimized_params['stop_loss']:
                    return {'action': 'SELL', 'rationale': f'Long exit: {pnl_pct:.2%}'}
            
            if trend_down and momentum < -self.optimized_params['momentum_threshold']:
                return {'action': 'SELL', 'rationale': f'Long exit: downtrend {momentum:.2%}'}
        
        elif self.current_position < 0:
            if self.position_entry_price:
                short_pnl_pct = (self.position_entry_price - current_price) / self.position_entry_price
                if short_pnl_pct > self.optimized_params['profit_target'] or short_pnl_pct < -self.optimized_params['stop_loss']:
                    return {'action': 'BUY_TO_COVER', 'rationale': f'Short exit: {short_pnl_pct:.2%}'}
            
            if trend_up and momentum > self.optimized_params['momentum_threshold']:
                return {'action': 'BUY_TO_COVER', 'rationale': f'Cover on reversal: {momentum:.2%}'}
        
        return {'action': 'HOLD', 'rationale': f'Long-short waiting (intervals: {self.intervals_since_last_trade})'}
    
    def execute_simulated_trade(self, action, current_price, reasoning, price_history=None):
        """Execute trade with short selling support"""
        
        if action == 'BUY' and self.current_position == 0:
            available_cash = self.current_budget * self.optimized_params['position_size']
            shares = int(available_cash // current_price)
            
            if shares * current_price < 2000:
                print(f"  FILTERED: BUY rejected: Position too small")
                return False
            
            if shares > 0:
                cost = shares * current_price
                self.current_position = shares
                self.current_budget -= cost
                self.position_entry_price = current_price
                self.intervals_since_last_trade = 0
                
                trade = {
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'shares': shares,
                    'price': current_price,
                    'cost': cost,
                    'reasoning': reasoning
                }
                self.executed_trades.append(trade)
                print(f"  TRADE: BUY {shares} @ ${current_price:.2f} | {reasoning}")
                return True
        
        elif action == 'SELL' and self.current_position > 0:
            proceeds = self.current_position * current_price
            pnl = proceeds - (self.current_position * (self.position_entry_price or current_price))
            
            trade = {
                'timestamp': datetime.now(),
                'action': 'SELL',
                'shares': self.current_position,
                'price': current_price,
                'proceeds': proceeds,
                'pnl': pnl,
                'reasoning': reasoning
            }
            self.executed_trades.append(trade)
            print(f"  TRADE: SELL {self.current_position} @ ${current_price:.2f} | P&L: ${pnl:+.2f} | {reasoning}")
            
            self.current_position = 0
            self.current_budget += proceeds
            self.position_entry_price = None
            self.intervals_since_last_trade = 0
            return True
        
        elif action == 'SELL_SHORT' and self.current_position == 0:
            available_margin = self.current_budget * self.optimized_params['position_size']
            shares_to_short = int(available_margin // (current_price * self.margin_requirement))
            
            if shares_to_short * current_price < 2000:
                print(f"  FILTERED: SHORT rejected: Position too small")
                return False
            
            if shares_to_short > 0:
                proceeds = shares_to_short * current_price
                margin_required = proceeds * self.margin_requirement
                
                self.current_position = -shares_to_short
                self.current_budget += proceeds - margin_required
                self.position_entry_price = current_price
                self.intervals_since_last_trade = 0
                
                trade = {
                    'timestamp': datetime.now(),
                    'action': 'SELL_SHORT',
                    'shares': shares_to_short,
                    'price': current_price,
                    'proceeds': proceeds,
                    'margin_required': margin_required,
                    'reasoning': reasoning
                }
                self.executed_trades.append(trade)
                print(f"  TRADE: SELL SHORT {shares_to_short} @ ${current_price:.2f} | {reasoning}")
                return True
        
        elif action == 'BUY_TO_COVER' and self.current_position < 0:
            shares_to_cover = abs(self.current_position)
            cost = shares_to_cover * current_price
            
            if self.position_entry_price:
                entry_value = shares_to_cover * self.position_entry_price
                short_pnl = entry_value - cost
            else:
                short_pnl = 0
            
            margin_returned = shares_to_cover * self.position_entry_price * self.margin_requirement
            self.current_budget = self.current_budget - cost + margin_returned
            
            trade = {
                'timestamp': datetime.now(),
                'action': 'BUY_TO_COVER',
                'shares': shares_to_cover,
                'price': current_price,
                'cost': cost,
                'pnl': short_pnl,
                'reasoning': reasoning
            }
            self.executed_trades.append(trade)
            print(f"  TRADE: BUY TO COVER {shares_to_cover} @ ${current_price:.2f} | P&L: ${short_pnl:+.2f} | {reasoning}")
            
            self.current_position = 0
            self.position_entry_price = None
            self.intervals_since_last_trade = 0
            return True
        
        return False
    
    def save_training_metadata(self, data_points, training_days):
        try:
            training_info = {
                'symbol': self.base_symbol,
                'training_completed': datetime.now().isoformat(),
                'data_points': data_points,
                'training_days': training_days,
                'trading_mode': self.trading_mode,
                'optimized_parameters': self.optimized_params,
                'trained_on': f'{self.trading_mode.lower()}_strategy_optimized',
                'model_version': '2.1_with_optimization',
                'strategy_type': f'{self.trading_mode.lower()}_momentum_optimized',
                'long_term_trained': True,
                'intraday_trained': True,
                'risk_management': 'optimized',
                'supports_shorting': self.trading_mode in ['SHORT_ONLY', 'LONG_SHORT']
            }
            
            training_file = f"training_metadata_{self.base_symbol}_{self.trading_mode}.json"
            with open(training_file, 'w') as f:
                json.dump(training_info, f, indent=2)
            
            print(f"TRAINING: Metadata saved to {training_file}")
            
        except Exception as e:
            print(f"WARNING: Failed to save training metadata: {e}")

