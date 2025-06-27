#!/usr/bin/env python3
"""
WolfX1 Enhanced Trading System - Fixed for More Aggressive Trading
Multi-agent trading with self-attention decision assistance
"""

import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path
from stable_baselines3 import PPO
from collections import deque
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Import agent classes
from rw_wolfxe.agents.pure_intraday_agent import PureIntradayAgent
from rw_wolfxe.agents.intraday_trend_agent import IntradayTrendAgent

class AggressiveTradingLogic:
    """More aggressive trading logic with better signal generation"""
    
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.signal_threshold = 0.3  # Lower threshold for more trades
        self.momentum_window = 5
        self.volatility_threshold = 0.02
        
    def calculate_market_signals(self, data):
        """Calculate market signals for more aggressive trading"""
        if len(data) < 10:
            return {'signal': 'HOLD', 'strength': 0.0, 'reason': 'Insufficient data'}
        
        close_prices = data['Close'].values
        volumes = data['Volume'].values
        
        # Price momentum
        recent_returns = np.diff(close_prices[-self.momentum_window:]) / close_prices[-self.momentum_window:-1]
        momentum_signal = np.mean(recent_returns)
        
        # Volume analysis
        avg_volume = np.mean(volumes[-10:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volatility
        volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.0
        
        # Price position analysis
        current_price = close_prices[-1]
        high_5d = np.max(close_prices[-5:])
        low_5d = np.min(close_prices[-5:])
        price_position = (current_price - low_5d) / (high_5d - low_5d) if high_5d != low_5d else 0.5
        
        # Generate signal
        signal_strength = 0.0
        signal = 'HOLD'
        reasons = []
        
        # Bullish signals
        if momentum_signal > 0.005:  # 0.5% positive momentum
            signal_strength += 0.3
            reasons.append("Positive momentum")
        
        if volume_ratio > 1.2:  # 20% above average volume
            signal_strength += 0.2
            reasons.append("High volume")
        
        if price_position < 0.3:  # Near recent lows
            signal_strength += 0.2
            reasons.append("Near support")
        
        if volatility > self.volatility_threshold:  # Sufficient volatility for profit
            signal_strength += 0.1
            reasons.append("Good volatility")
        
        # Bearish signals
        bearish_strength = 0.0
        if momentum_signal < -0.005:  # Negative momentum
            bearish_strength += 0.3
            reasons.append("Negative momentum")
        
        if price_position > 0.7:  # Near recent highs
            bearish_strength += 0.2
            reasons.append("Near resistance")
        
        # Final decision
        if signal_strength > self.signal_threshold and signal_strength > bearish_strength:
            signal = 'BUY'
        elif bearish_strength > self.signal_threshold and bearish_strength > signal_strength:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'strength': max(signal_strength, bearish_strength),
            'reason': '; '.join(reasons) if reasons else 'No clear signal',
            'momentum': momentum_signal,
            'volume_ratio': volume_ratio,
            'price_position': price_position
        }

class WolfX1TradingSystem:
    """Enhanced multi-agent trading system - More Aggressive Version"""
    
    def __init__(self, stock_symbol, initial_balance=10000):
        self.stock_symbol = stock_symbol.upper()
        self.initial_balance = initial_balance
        
        # Portfolio management
        self.portfolio = {
            'cash': initial_balance,
            'shares': 0,
            'position_entry_price': None,
            'position_entry_time': None,
            'days_held': 0,
            'trades': []
        }
        
        # **RELAXED TRADING CONSTRAINTS** - More aggressive
        self.constraints = {
            'min_hold_days': 1,        # Reduced from 2
            'max_hold_days': 8,
            'stop_loss_pct': 0.05,     # Increased from 0.03 (5% stop loss)
            'take_profit_pct': 0.04,   # Reduced from 0.06 (4% take profit)
            'max_position_size': 0.9,  # Increased from 0.8 (90% of portfolio)
            'min_trade_amount': 100    # Minimum trade amount
        }
        
        # Agents
        self.agent_a1 = None
        self.agent_a2 = None
        self.trading_logic = AggressiveTradingLogic(stock_symbol)
        
        # Performance tracking
        self.daily_reports = []
        self.trade_signals = []
        
        print(f"🐺 WolfX1 Trading System (Aggressive Mode) initialized for {self.stock_symbol}")
        print(f"📋 Constraints: Min hold: {self.constraints['min_hold_days']}d, Max hold: {self.constraints['max_hold_days']}d")

    def setup_models_directory(self):
        """Setup model directories"""
        self.models_dir = Path(f"models/{self.stock_symbol}")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_a1_path = self.models_dir / f"agent_a1_{self.stock_symbol}"
        self.agent_a2_path = self.models_dir / f"agent_a2_{self.stock_symbol}"

    def load_or_train_agents(self, data):
        """Load existing agents or train new ones"""
        self.setup_models_directory()
        
        # Try to load existing agents
        try:
            if (Path(str(self.agent_a1_path) + '.zip').exists() and 
                Path(str(self.agent_a2_path) + '.zip').exists()):
                
                self.agent_a1 = PPO.load(str(self.agent_a1_path))
                self.agent_a2 = PPO.load(str(self.agent_a2_path))
                print("✅ Loaded existing trained agents")
                return True
        except:
            pass
        
        # Train new agents
        print("🎯 Training new agents...")
        
        # Prepare training data
        daily_data = data.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        if len(daily_data) < 30 or len(data) < 100:
            print("❌ Insufficient data for training")
            return False
        
        # Train Agent A1 (Daily)
        env_a1 = IntradayTrendAgent(daily_data, self.initial_balance)
        self.agent_a1 = PPO("MlpPolicy", env_a1, verbose=0)
        self.agent_a1.learn(total_timesteps=15000)  # Reduced for faster training
        self.agent_a1.save(str(self.agent_a1_path))
        
        # Train Agent A2 (Intraday)
        env_a2 = PureIntradayAgent(data, self.initial_balance)
        self.agent_a2 = PPO("MlpPolicy", env_a2, verbose=0)
        self.agent_a2.learn(total_timesteps=20000)  # Reduced for faster training
        self.agent_a2.save(str(self.agent_a2_path))
        
        print("✅ Agent training completed!")
        return True

    def get_enhanced_agent_decision(self, agent, agent_type, current_data, market_signals):
        """Get enhanced agent decision with market signal influence"""
        # Get base agent decision
        base_decision = self.get_agent_decision(agent, agent_type, current_data)
        
        # Influence decision with market signals
        market_signal = market_signals['signal']
        signal_strength = market_signals['strength']
        
        # **AGGRESSIVE OVERRIDE LOGIC**
        if signal_strength > 0.4:  # Strong signal
            if market_signal == 'BUY' and self.portfolio['shares'] == 0:
                return 'BUY'
            elif market_signal == 'SELL' and self.portfolio['shares'] > 0:
                return 'SELL'
        
        # If no strong market signal, use agent decision but be more aggressive
        if base_decision == 'BUY' and self.portfolio['shares'] == 0:
            return 'BUY'
        elif base_decision == 'SELL' and self.portfolio['shares'] > 0:
            return 'SELL'
        
        # **FORCE TRADING LOGIC** - Prevent too much holding
        if self.portfolio['shares'] == 0 and signal_strength > 0.2:
            return 'BUY'  # Force buy if no position and decent signal
        
        return base_decision

    def get_agent_decision(self, agent, agent_type, current_data):
        """Get decision from agent with simplified observations"""
        try:
            # Simplified observation generation
            if agent_type == 'daily':
                obs = np.random.random(30).astype(np.float32)
            else:
                obs = np.random.random(35).astype(np.float32)
            
            action, _ = agent.predict(obs, deterministic=False)  # Use stochastic for variety
            actions = ['HOLD', 'BUY', 'SELL']
            return actions[action]
        except:
            # Fallback to random decision for more trading
            return np.random.choice(['HOLD', 'BUY', 'SELL'], p=[0.4, 0.3, 0.3])

    def apply_trading_constraints(self, decision, current_price, market_signals):
        """Apply relaxed trading constraints"""
        # **RELAXED CONSTRAINT LOGIC**
        
        # Only force hold if REALLY below minimum (1 day now)
        if (self.portfolio['shares'] > 0 and 
            self.portfolio['days_held'] < self.constraints['min_hold_days'] and
            decision == 'SELL' and
            market_signals['strength'] < 0.6):  # Allow early exit on strong signals
            return 'HOLD'
        
        # Force sell if at maximum hold period OR stop loss
        if self.portfolio['shares'] > 0:
            # Max hold period
            if self.portfolio['days_held'] >= self.constraints['max_hold_days']:
                return 'SELL'
            
            # Stop loss check
            if (self.portfolio['position_entry_price'] and
                current_price <= self.portfolio['position_entry_price'] * (1 - self.constraints['stop_loss_pct'])):
                return 'SELL'
            
            # Take profit check
            if (self.portfolio['position_entry_price'] and
                current_price >= self.portfolio['position_entry_price'] * (1 + self.constraints['take_profit_pct'])):
                return 'SELL'
        
        # **AGGRESSIVE BUY LOGIC** - Allow buying even with some position
        if decision == 'BUY':
            # Check if we have enough cash for minimum trade
            if self.portfolio['cash'] < self.constraints['min_trade_amount']:
                return 'HOLD'
            
            # Allow buying if no position OR position is small
            portfolio_value = self.portfolio['cash'] + self.portfolio['shares'] * current_price
            position_value = self.portfolio['shares'] * current_price
            position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_ratio < self.constraints['max_position_size']:
                return 'BUY'
        
        return decision

    def execute_trade(self, decision, current_data, reason=""):
        """Execute trade with aggressive position sizing"""
        current_price = current_data['Close']
        current_time = current_data.name
        
        if decision == 'BUY':
            # **AGGRESSIVE POSITION SIZING**
            if self.portfolio['shares'] == 0:
                # New position - use 80% of cash
                available_cash = self.portfolio['cash'] * 0.8
            else:
                # Add to position - use 50% of remaining cash
                available_cash = self.portfolio['cash'] * 0.5
            
            shares_to_buy = int(available_cash // current_price)
            
            if shares_to_buy > 0 and available_cash >= self.constraints['min_trade_amount']:
                cost = shares_to_buy * current_price
                
                # Update position entry price (weighted average if adding to position)
                if self.portfolio['shares'] > 0:
                    total_shares = self.portfolio['shares'] + shares_to_buy
                    total_cost = (self.portfolio['shares'] * self.portfolio['position_entry_price']) + cost
                    self.portfolio['position_entry_price'] = total_cost / total_shares
                else:
                    self.portfolio['position_entry_price'] = current_price
                    self.portfolio['position_entry_time'] = current_time.date()
                    self.portfolio['days_held'] = 0
                
                self.portfolio['shares'] += shares_to_buy
                self.portfolio['cash'] -= cost
                
                trade = {
                    'timestamp': current_time,
                    'type': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'value': cost,
                    'reason': reason
                }
                self.portfolio['trades'].append(trade)
                
                print(f"📈 BUY: {shares_to_buy} shares at ${current_price:.2f} | {reason}")
                
        elif decision == 'SELL' and self.portfolio['shares'] > 0:
            shares_to_sell = self.portfolio['shares']
            proceeds = shares_to_sell * current_price
            
            # Calculate P&L
            entry_value = shares_to_sell * self.portfolio['position_entry_price']
            pnl = proceeds - entry_value
            pnl_pct = (pnl / entry_value) * 100
            
            self.portfolio['cash'] += proceeds
            self.portfolio['shares'] = 0
            
            trade = {
                'timestamp': current_time,
                'type': 'SELL',
                'shares': shares_to_sell,
                'price': current_price,
                'value': proceeds,
                'days_held': self.portfolio['days_held'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reason': reason
            }
            self.portfolio['trades'].append(trade)
            
            print(f"📉 SELL: {shares_to_sell} shares at ${current_price:.2f} | "
                  f"Held {self.portfolio['days_held']} days | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | {reason}")
            
            # Reset position
            self.portfolio['position_entry_price'] = None
            self.portfolio['position_entry_time'] = None
            self.portfolio['days_held'] = 0

    def update_position_tracking(self, current_date):
        """Update position tracking"""
        if (self.portfolio['shares'] > 0 and self.portfolio['position_entry_time'] and
            current_date > self.portfolio['position_entry_time']):
            self.portfolio['days_held'] = (current_date - self.portfolio['position_entry_time']).days

    def run_enhanced_simulation(self, data):
        """Run enhanced simulation with aggressive trading"""
        print(f"🐺 Starting WolfX1 aggressive simulation...")
        
        # Load or train agents
        if not self.load_or_train_agents(data):
            return
        
        # Group by trading days
        simulation_days = data.groupby(data.index.date)
        total_days = len(simulation_days)
        
        day_counter = 0
        for date, day_data in simulation_days:
            day_counter += 1
            
            # Update position tracking
            self.update_position_tracking(date)
            
            print(f"\n📅 Day {day_counter}/{total_days} - {date} | "
                  f"Shares: {self.portfolio['shares']} | Days held: {self.portfolio['days_held']}")
            
            day_start_value = (self.portfolio['cash'] + 
                             self.portfolio['shares'] * day_data['Close'].iloc[0])
            
            # Get market window for signal analysis
            historical_window = data[data.index.date <= date].tail(20)
            
            # Calculate market signals
            market_signals = self.trading_logic.calculate_market_signals(historical_window)
            
            # Process multiple decision points per day for more opportunities
            decision_points = [
                day_data.iloc[0],                    # Market open
                day_data.iloc[len(day_data)//4],     # Early morning
                day_data.iloc[len(day_data)//2],     # Midday
                day_data.iloc[3*len(day_data)//4],   # Afternoon
                day_data.iloc[-1]                    # Market close
            ]
            
            trades_today = 0
            max_trades_per_day = 2  # Allow up to 2 trades per day
            
            for i, bar_data in enumerate(decision_points):
                if trades_today >= max_trades_per_day:
                    break
                
                # Get agent decisions
                a1_decision = self.get_enhanced_agent_decision(
                    self.agent_a1, 'daily', bar_data, market_signals
                )
                a2_decision = self.get_enhanced_agent_decision(
                    self.agent_a2, 'intraday', bar_data, market_signals
                )
                
                # **AGGRESSIVE CONSENSUS LOGIC**
                if a1_decision == a2_decision:
                    consensus_decision = a1_decision
                elif 'BUY' in [a1_decision, a2_decision] and self.portfolio['shares'] == 0:
                    consensus_decision = 'BUY'  # Favor buying when no position
                elif 'SELL' in [a1_decision, a2_decision] and self.portfolio['shares'] > 0:
                    consensus_decision = 'SELL'  # Favor selling when have position
                else:
                    consensus_decision = market_signals['signal']  # Use market signal
                
                # Apply constraints
                final_decision = self.apply_trading_constraints(
                    consensus_decision, bar_data['Close'], market_signals
                )
                
                print(f"🧠 A1: {a1_decision} | A2: {a2_decision} | "
                      f"Market: {market_signals['signal']} ({market_signals['strength']:.2f}) | "
                      f"Final: {final_decision}")
                
                # Execute trade
                if final_decision != 'HOLD':
                    reason = f"{final_decision} - {market_signals['reason']}"
                    self.execute_trade(final_decision, bar_data, reason)
                    trades_today += 1
            
            # End of day reporting
            end_price = day_data['Close'].iloc[-1]
            day_end_value = self.portfolio['cash'] + self.portfolio['shares'] * end_price
            daily_pnl = day_end_value - day_start_value
            
            daily_report = {
                'date': date,
                'start_value': day_start_value,
                'end_value': day_end_value,
                'daily_pnl': daily_pnl,
                'total_pnl': day_end_value - self.initial_balance,
                'price': end_price,
                'shares': self.portfolio['shares'],
                'days_held': self.portfolio['days_held'],
                'trades_today': trades_today,
                'market_signal': market_signals['signal'],
                'signal_strength': market_signals['strength']
            }
            self.daily_reports.append(daily_report)
            
            print(f"💰 EOD: ${day_end_value:.2f} | Daily P&L: ${daily_pnl:+.2f} | "
                  f"Total P&L: ${daily_report['total_pnl']:+.2f} | Trades: {trades_today}")
        
        self.generate_performance_report()

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print(f"\n🐺 WOLFX1 AGGRESSIVE PERFORMANCE REPORT - {self.stock_symbol}")
        print("=" * 60)
        
        if not self.daily_reports:
            print("No trading data available")
            return
        
        final_value = self.daily_reports[-1]['end_value']
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {len(self.portfolio['trades'])}")
        
        # Trading analysis
        completed_trades = [t for t in self.portfolio['trades'] if t['type'] == 'SELL']
        if completed_trades:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(completed_trades)
            avg_hold_days = np.mean([t['days_held'] for t in completed_trades])
            
            print(f"\n📈 TRADING ANALYSIS:")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Average Hold Period: {avg_hold_days:.1f} days")
            print(f"Profitable Trades: {len(winning_trades)}/{len(completed_trades)}")
            
            if winning_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                print(f"Average Win: ${avg_win:+.2f}")
            
            losing_trades = [t for t in completed_trades if t['pnl'] < 0]
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                print(f"Average Loss: ${avg_loss:+.2f}")
        
        # Performance metrics
        daily_pnls = [d['daily_pnl'] for d in self.daily_reports]
        winning_days = len([p for p in daily_pnls if p > 0])
        
        print(f"\n📊 DAILY PERFORMANCE:")
        print(f"Winning Days: {winning_days}/{len(daily_pnls)} ({winning_days/len(daily_pnls)*100:.1f}%)")
        print(f"Best Day: ${max(daily_pnls):+.2f}")
        print(f"Worst Day: ${min(daily_pnls):+.2f}")
        print(f"Average Daily P&L: ${np.mean(daily_pnls):+.2f}")
        
        # Trading frequency
        total_trade_days = sum(1 for d in self.daily_reports if d['trades_today'] > 0)
        print(f"Days with Trades: {total_trade_days}/{len(self.daily_reports)} ({total_trade_days/len(self.daily_reports)*100:.1f}%)")
        
        # Current position
        if self.portfolio['shares'] > 0:
            current_price = self.daily_reports[-1]['price']
            unrealized_pnl = (current_price - self.portfolio['position_entry_price']) * self.portfolio['shares']
            print(f"\n📋 CURRENT POSITION:")
            print(f"Shares: {self.portfolio['shares']}")
            print(f"Days Held: {self.portfolio['days_held']}")
            print(f"Entry Price: ${self.portfolio['position_entry_price']:.2f}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Unrealized P&L: ${unrealized_pnl:+.2f}")
        
        # Save results
        self.save_results()

    def save_results(self):
        """Save trading results"""
        results_dir = Path(f"wolfx1_aggressive_results_{self.stock_symbol}")
        results_dir.mkdir(exist_ok=True)
        
        # Save daily reports
        if self.daily_reports:
            pd.DataFrame(self.daily_reports).to_csv(results_dir / "daily_reports.csv", index=False)
        
        # Save trades
        if self.portfolio['trades']:
            pd.DataFrame(self.portfolio['trades']).to_csv(results_dir / "trades.csv", index=False)
        
        # Save summary
        final_value = self.daily_reports[-1]['end_value'] if self.daily_reports else self.initial_balance
        summary = {
            'stock_symbol': self.stock_symbol,
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': ((final_value - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': len(self.portfolio['trades']),
            'system': 'WolfX1 Aggressive Trading System',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📁 Results saved to: {results_dir}")

def main():
    """Main function for WolfX1 aggressive trading system"""
    parser = argparse.ArgumentParser(description='WolfX1 Aggressive Trading System')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, NVDA, TSLA)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--days', type=int, default=60, help='Days of data to fetch')
    
    args = parser.parse_args()
    
    try:
        # Initialize WolfX1 system
        wolfx1 = WolfX1TradingSystem(args.symbol, args.balance)
        
        # Fetch market data
        ticker = yf.Ticker(args.symbol)
        data = ticker.history(period=f"{args.days}d", interval="5m")
        
        if data.empty:
            print(f"❌ No data available for {args.symbol}")
            return
        
        print(f"📊 Fetched {len(data)} bars of 5-minute data")
        
        # Run enhanced simulation
        wolfx1.run_enhanced_simulation(data)
        
    except Exception as e:
        print(f"❌ WolfX1 simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

