#!/usr/bin/env python3
"""
Enhanced Intraday Trading Simulator
Implements proper hold strategy with 8-day maximum holding period
"""

import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import time
import json
from pathlib import Path
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

# Import our agent classes
from rw_wolfxe.agents.pure_intraday_agent import PureIntradayAgent
from rw_wolfxe.agents.intraday_trend_agent import IntradayTrendAgent

class EnhancedIntradaySimulator:
    """Enhanced simulator with proper hold strategy and position management"""
    
    def __init__(self, stock_symbol, initial_balance=10000):
        self.stock_symbol = stock_symbol.upper()
        self.initial_balance = initial_balance
        self.portfolio = {
            'cash': initial_balance,
            'shares': 0,
            'total_value': initial_balance,
            'position_entry_time': None,
            'position_entry_price': None,
            'days_held': 0,
            'trades': []
        }
        
        # Hold strategy parameters
        self.min_hold_days = 2
        self.max_hold_days = 8
        self.optimal_hold_range = (3, 6)  # Sweet spot for holding
        
        # Simulation tracking
        self.daily_reports = []
        self.position_history = []
        
        # Models
        self.agent_a1 = None
        self.agent_a2 = None
        
        print(f"🎯 Enhanced Intraday Simulator for {self.stock_symbol}")
        print(f"📋 Hold Strategy: {self.min_hold_days}-{self.max_hold_days} days (optimal: {self.optimal_hold_range[0]}-{self.optimal_hold_range[1]} days)")

    def fetch_available_data(self):
        """Fetch maximum available data for any ticker"""
        print(f"📊 Fetching data for {self.stock_symbol}...")
        
        ticker = yf.Ticker(self.stock_symbol)
        all_data = {}
        
        try:
            # Get daily data for longer history
            print("📅 Fetching daily data (2 years)...")
            daily_data = ticker.history(period="2y", interval="1d")
            if not daily_data.empty:
                all_data['daily'] = daily_data
                print(f"✅ Daily data: {len(daily_data)} bars")
            else:
                print(f"❌ No daily data available for {self.stock_symbol}")
                return {}
            
            # Get intraday data
            print("📅 Fetching intraday data...")
            intraday_configs = [
                ("60d", "5m"),
                ("30d", "2m"),
                ("7d", "1m"),
            ]
            
            intraday_data = None
            for period, interval in intraday_configs:
                try:
                    data = ticker.history(period=period, interval=interval)
                    if not data.empty:
                        intraday_data = data
                        print(f"✅ Got {len(data)} bars of {interval} data")
                        break
                except:
                    continue
            
            if intraday_data is not None:
                all_data['intraday'] = intraday_data
            else:
                # Generate synthetic intraday from daily
                print("🔧 Generating synthetic intraday data...")
                synthetic = self.generate_synthetic_intraday(daily_data.tail(30))
                all_data['intraday'] = synthetic
                print(f"✅ Generated {len(synthetic)} synthetic bars")
            
            return all_data
            
        except Exception as e:
            print(f"❌ Error fetching data for {self.stock_symbol}: {e}")
            return {}

    def generate_synthetic_intraday(self, daily_data):
        """Generate realistic synthetic intraday data"""
        synthetic_data = []
        
        for date, row in daily_data.iterrows():
            if date.weekday() >= 5:  # Skip weekends
                continue
                
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            volume = row['Volume']
            
            # Generate 78 5-minute intervals per day (9:30 AM to 4:00 PM)
            intervals_per_day = 78
            daily_return = (close_price - open_price) / open_price
            
            for i in range(intervals_per_day):
                time_progress = i / (intervals_per_day - 1)
                
                # Create realistic intraday price movement
                if i == 0:
                    price = open_price
                elif i == intervals_per_day - 1:
                    price = close_price
                else:
                    # Add volatility and trend
                    trend_component = daily_return * time_progress
                    volatility = np.random.normal(0, 0.003)  # 0.3% volatility
                    
                    # U-shaped volatility (higher at open/close)
                    u_factor = 2 * abs(time_progress - 0.5)
                    volatility *= (1 + u_factor)
                    
                    price = open_price * (1 + trend_component + volatility)
                    price = max(low_price, min(high_price, price))
                
                # Generate OHLC for interval
                interval_volatility = (high_price - low_price) * 0.05
                interval_high = min(high_price, price + interval_volatility * np.random.uniform(0, 1))
                interval_low = max(low_price, price - interval_volatility * np.random.uniform(0, 1))
                
                # Realistic volume distribution
                volume_factor = np.random.lognormal(0, 0.5)  # Log-normal distribution
                interval_volume = (volume / intervals_per_day) * volume_factor
                
                # Create timestamp
                market_open = date.replace(hour=9, minute=30, second=0, microsecond=0)
                timestamp = market_open + timedelta(minutes=5*i)
                
                synthetic_data.append({
                    'timestamp': timestamp,
                    'Open': price,
                    'High': interval_high,
                    'Low': interval_low,
                    'Close': price,
                    'Volume': interval_volume
                })
        
        df = pd.DataFrame(synthetic_data)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()

    def prepare_data(self, all_data):
        """Prepare training and simulation data"""
        print("🔄 Preparing datasets...")
        
        # Daily data preparation
        daily_data = all_data['daily']
        daily_split = int(len(daily_data) * 0.85)  # 85% for training
        self.daily_training = daily_data.iloc[:daily_split].copy()
        self.daily_context = daily_data.iloc[daily_split:].copy()
        
        # Intraday data preparation
        intraday_data = all_data['intraday']
        intraday_split = int(len(intraday_data) * 0.85)  # 85% for training
        self.training_data = intraday_data.iloc[:intraday_split].copy()
        self.simulation_data = intraday_data.iloc[intraday_split:].copy()
        
        print(f"📊 Training: {len(self.daily_training)} daily, {len(self.training_data)} intraday")
        print(f"🎮 Simulation: {len(self.simulation_data)} bars")
        
        return True

    def train_agents_once(self):
        """Train agents once with enhanced strategy awareness"""
        print(f"🤖 Training agents with hold strategy awareness...")
        
        # Enhanced training for Agent A1 with hold strategy
        print("Training Agent A1 (Daily Trend + Hold Strategy)...")
        env_a1 = HoldAwareIntradayTrendAgent(self.daily_training, self.initial_balance, 
                                           self.min_hold_days, self.max_hold_days)
        self.agent_a1 = PPO("MlpPolicy", env_a1, verbose=0)
        self.agent_a1.learn(total_timesteps=30000)
        
        # Enhanced training for Agent A2 with position management
        print("Training Agent A2 (Intraday + Position Management)...")
        env_a2 = HoldAwarePureIntradayAgent(self.training_data, self.initial_balance,
                                          self.min_hold_days, self.max_hold_days)
        self.agent_a2 = PPO("MlpPolicy", env_a2, verbose=0)
        self.agent_a2.learn(total_timesteps=40000)
        
        print("✅ Enhanced agent training completed!")
        return True

    def calculate_hold_pressure(self):
        """Calculate pressure to hold or trade based on current position"""
        if self.portfolio['shares'] == 0:
            return 0.0  # No position, no hold pressure
        
        days_held = self.portfolio['days_held']
        
        if days_held < self.min_hold_days:
            return 1.0  # Strong pressure to hold (below minimum)
        elif days_held >= self.max_hold_days:
            return -1.0  # Strong pressure to sell (at maximum)
        elif self.optimal_hold_range[0] <= days_held <= self.optimal_hold_range[1]:
            return 0.0  # Neutral (in optimal range)
        else:
            # Gradual pressure as we approach limits
            if days_held < self.optimal_hold_range[0]:
                return 0.5  # Moderate pressure to hold
            else:
                return -0.5  # Moderate pressure to sell

    def get_enhanced_agent_decision(self, agent, agent_type, current_data):
        """Get decision with hold strategy integration"""
        obs = self.get_market_observation(current_data, agent_type)
        
        try:
            action, _ = agent.predict(obs, deterministic=True)
            actions = ['HOLD', 'BUY', 'SELL']
            base_decision = actions[action]
            
            # Apply hold strategy logic
            hold_pressure = self.calculate_hold_pressure()
            
            # Override decisions based on hold constraints
            if self.portfolio['shares'] > 0:
                if self.portfolio['days_held'] < self.min_hold_days:
                    # Force hold if below minimum
                    return 'HOLD'
                elif self.portfolio['days_held'] >= self.max_hold_days:
                    # Force sell if at maximum
                    return 'SELL'
            
            # If no position and agent wants to buy, allow it
            if self.portfolio['shares'] == 0 and base_decision == 'BUY':
                return 'BUY'
            
            # Apply hold pressure to modify decisions
            if hold_pressure > 0.5 and base_decision == 'SELL':
                return 'HOLD'  # Pressure to hold overrides sell
            elif hold_pressure < -0.5 and base_decision == 'HOLD':
                return 'SELL'  # Pressure to sell overrides hold
            
            return base_decision
            
        except Exception as e:
            return 'HOLD'

    def get_market_observation(self, current_data, agent_type):
        """Generate market observations with position awareness"""
        try:
            current_price = current_data['Close']
            
            if agent_type == 'daily':
                features = []
                
                # Price momentum features (10)
                daily_context = self.daily_context
                for period in [3, 5, 10, 15, 20, 30, 50, 100, 150, 200]:
                    if len(daily_context) >= period:
                        past_price = daily_context['Close'].iloc[-period]
                        momentum = (current_price - past_price) / past_price
                        features.append(momentum)
                    else:
                        features.append(0.0)
                
                # Volatility features (5)
                for window in [5, 10, 20, 50, 100]:
                    if len(daily_context) >= window:
                        prices = daily_context['Close'].iloc[-window:]
                        vol = prices.std() / prices.mean() if prices.mean() > 0 else 0.0
                        features.append(vol)
                    else:
                        features.append(0.0)
                
                # Enhanced portfolio features (8) - including hold strategy
                portfolio_value = self.portfolio['cash'] + self.portfolio['shares'] * current_price
                features.extend([
                    self.portfolio['cash'] / self.initial_balance,
                    self.portfolio['shares'] / 100.0,
                    portfolio_value / self.initial_balance,
                    (portfolio_value - self.initial_balance) / self.initial_balance,
                    1.0 if self.portfolio['shares'] > 0 else 0.0,
                    self.portfolio['days_held'] / self.max_hold_days,  # Hold progress
                    self.calculate_hold_pressure(),  # Hold pressure signal
                    1.0 if self.optimal_hold_range[0] <= self.portfolio['days_held'] <= self.optimal_hold_range[1] else 0.0
                ])
                
                # Market features (7)
                features.extend([
                    (current_data['High'] - current_data['Low']) / current_price,
                    (current_price - current_data['Open']) / current_data['Open'],
                    current_data['Volume'] / 1e6,
                    1.0 if self.portfolio['days_held'] >= self.max_hold_days else 0.0,  # Force sell signal
                    1.0 if self.portfolio['days_held'] < self.min_hold_days else 0.0,   # Force hold signal
                    1.0,  # Can trade flag
                    np.sin(2 * np.pi * len(daily_context) / 252)
                ])
                
                return np.array(features[:30], dtype=np.float32)
                
            else:  # intraday
                features = []
                hist_data = self.simulation_data[:current_data.name]
                
                # RSI (1)
                if len(hist_data) >= 14:
                    prices = hist_data['Close'].iloc[-14:]
                    delta = prices.diff()
                    gain = (delta.where(delta > 0, 0)).mean()
                    loss = (-delta.where(delta < 0, 0)).mean()
                    rs = gain / loss if loss != 0 else 100
                    rsi = 100 - (100 / (1 + rs))
                    features.append(rsi / 100.0)
                else:
                    features.append(0.5)
                
                # Price momentum (8)
                for period in [5, 10, 15, 20, 30, 60, 120, 240]:
                    if len(hist_data) >= period:
                        past_price = hist_data['Close'].iloc[-period]
                        momentum = (current_price - past_price) / past_price
                        features.append(momentum)
                    else:
                        features.append(0.0)
                
                # Moving averages (6)
                for period in [10, 20, 50, 100, 200, 300]:
                    if len(hist_data) >= period:
                        ma = hist_data['Close'].iloc[-period:].mean()
                        ma_ratio = (current_price - ma) / ma if ma > 0 else 0.0
                        features.append(ma_ratio)
                    else:
                        features.append(0.0)
                
                # Volume analysis (5)
                for period in [5, 15, 30, 60, 120]:
                    if len(hist_data) >= period:
                        avg_volume = hist_data['Volume'].iloc[-period:].mean()
                        volume_ratio = current_data['Volume'] / avg_volume if avg_volume > 0 else 1.0
                        features.append(min(volume_ratio, 5.0))
                    else:
                        features.append(1.0)
                
                # Enhanced market microstructure (10) - with position awareness
                hour = current_data.name.hour
                time_factor = (hour * 60 + current_data.name.minute) / (16 * 60)
                
                features.extend([
                    self.portfolio['cash'] / self.initial_balance,
                    self.portfolio['shares'] / 100.0,
                    1.0 if self.portfolio['shares'] > 0 else 0.0,
                    self.portfolio['days_held'] / self.max_hold_days,
                    time_factor,
                    1.0 if hour < 10 else 0.8 if hour > 15 else 0.5,
                    (current_data['High'] - current_data['Low']) / current_price,
                    len(self.portfolio['trades']) / 100.0,
                    self.calculate_hold_pressure(),  # Hold pressure
                    1.0 if self.portfolio['days_held'] >= self.max_hold_days else 0.0  # Force sell
                ])
                
                # Position management (5)
                if self.portfolio['shares'] > 0 and self.portfolio['position_entry_price']:
                    unrealized_pnl = (current_price - self.portfolio['position_entry_price']) / self.portfolio['position_entry_price']
                    features.extend([
                        unrealized_pnl,
                        self.portfolio['days_held'] / self.max_hold_days,
                        1.0 if unrealized_pnl > 0.02 else 0.0,
                        1.0 if self.portfolio['days_held'] > self.optimal_hold_range[1] else 0.0,
                        abs(unrealized_pnl)  # Risk measure
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                
                return np.array(features[:35], dtype=np.float32)
                
        except Exception as e:
            return np.zeros(30 if agent_type == 'daily' else 35, dtype=np.float32)

    def execute_trade(self, decision, current_data):
        """Execute trade with proper position tracking"""
        current_price = current_data['Close']
        current_date = current_data.name.date()
        
        if decision == 'BUY' and self.portfolio['shares'] == 0 and self.portfolio['cash'] > current_price * 10:
            # Buy position
            shares_to_buy = int(self.portfolio['cash'] * 0.95 // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.portfolio['shares'] = shares_to_buy
                self.portfolio['cash'] -= cost
                self.portfolio['position_entry_time'] = current_date
                self.portfolio['position_entry_price'] = current_price
                self.portfolio['days_held'] = 0
                
                trade = {
                    'timestamp': current_data.name,
                    'type': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'value': cost,
                    'reason': decision
                }
                self.portfolio['trades'].append(trade)
                
                print(f"📈 BUY: {shares_to_buy} shares at ${current_price:.2f} | Entry Day 0")
                
        elif decision == 'SELL' and self.portfolio['shares'] > 0:
            # Sell position
            shares_to_sell = self.portfolio['shares']
            proceeds = shares_to_sell * current_price
            
            # Calculate P&L
            entry_value = shares_to_sell * self.portfolio['position_entry_price']
            pnl = proceeds - entry_value
            
            self.portfolio['cash'] += proceeds
            self.portfolio['shares'] = 0
            
            trade = {
                'timestamp': current_data.name,
                'type': 'SELL',
                'shares': shares_to_sell,
                'price': current_price,
                'value': proceeds,
                'days_held': self.portfolio['days_held'],
                'pnl': pnl,
                'reason': f"{decision} (held {self.portfolio['days_held']} days)"
            }
            self.portfolio['trades'].append(trade)
            
            print(f"📉 SELL: {shares_to_sell} shares at ${current_price:.2f} | "
                  f"Held {self.portfolio['days_held']} days | P&L: ${pnl:+.2f}")
            
            # Reset position tracking
            self.portfolio['position_entry_time'] = None
            self.portfolio['position_entry_price'] = None
            self.portfolio['days_held'] = 0

    def update_position_tracking(self, current_date):
        """Update days held for current position"""
        if self.portfolio['shares'] > 0 and self.portfolio['position_entry_time']:
            days_held = (current_date - self.portfolio['position_entry_time']).days
            self.portfolio['days_held'] = days_held

    def run_enhanced_simulation(self):
        """Run simulation with proper hold strategy"""
        print(f"🎮 Starting enhanced simulation for {self.stock_symbol}...")
        
        # Group by trading days
        simulation_days = self.simulation_data.groupby(self.simulation_data.index.date)
        total_days = len(simulation_days)
        
        print(f"📅 Simulating {total_days} trading days with hold strategy")
        
        day_counter = 0
        for date, day_data in simulation_days:
            day_counter += 1
            
            # Update position tracking
            self.update_position_tracking(date)
            
            print(f"\n📅 Day {day_counter}/{total_days} - {date} | "
                  f"Position: {self.portfolio['shares']} shares | "
                  f"Days held: {self.portfolio['days_held']}")
            
            day_start_value = self.portfolio['cash'] + self.portfolio['shares'] * day_data['Close'].iloc[0]
            
            # Process key decision points in the day
            decision_points = [
                day_data.iloc[0],    # Market open
                day_data.iloc[len(day_data)//3],   # Mid-morning
                day_data.iloc[len(day_data)//2],   # Midday
                day_data.iloc[2*len(day_data)//3], # Afternoon
                day_data.iloc[-1]    # Market close
            ]
            
            for bar_data in decision_points:
                # Get enhanced agent decisions
                a1_decision = self.get_enhanced_agent_decision(self.agent_a1, 'daily', bar_data)
                a2_decision = self.get_enhanced_agent_decision(self.agent_a2, 'intraday', bar_data)
                
                # Enhanced consensus with hold strategy
                if a1_decision == a2_decision:
                    final_decision = a1_decision
                elif 'SELL' in [a1_decision, a2_decision] and self.portfolio['days_held'] >= self.max_hold_days:
                    final_decision = 'SELL'  # Force sell at max hold
                elif 'BUY' in [a1_decision, a2_decision] and self.portfolio['shares'] == 0:
                    final_decision = 'BUY'   # Allow buy if no position
                else:
                    final_decision = 'HOLD'  # Conservative default
                
                # Execute trade
                if final_decision != 'HOLD':
                    self.execute_trade(final_decision, bar_data)
                    break  # Only one major trade per day
            
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
                'trades_today': len([t for t in self.portfolio['trades'] if t['timestamp'].date() == date])
            }
            self.daily_reports.append(daily_report)
            
            print(f"💰 EOD: ${day_end_value:.2f} | Daily P&L: ${daily_pnl:+.2f} | "
                  f"Total P&L: ${daily_report['total_pnl']:+.2f}")
        
        self.generate_enhanced_report()

    def generate_enhanced_report(self):
        """Generate comprehensive report with hold strategy analysis"""
        print(f"\n📊 ENHANCED SIMULATION REPORT - {self.stock_symbol}")
        print("=" * 60)
        
        if not self.daily_reports:
            print("No trading data to report")
            return
        
        final_value = self.daily_reports[-1]['end_value']
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {len(self.portfolio['trades'])}")
        
        # Hold strategy analysis
        completed_trades = [t for t in self.portfolio['trades'] if t['type'] == 'SELL']
        if completed_trades:
            avg_hold_days = np.mean([t['days_held'] for t in completed_trades])
            hold_day_distribution = {}
            for t in completed_trades:
                days = t['days_held']
                hold_day_distribution[days] = hold_day_distribution.get(days, 0) + 1
            
            print(f"\n📈 HOLD STRATEGY ANALYSIS:")
            print(f"Average Hold Period: {avg_hold_days:.1f} days")
            print(f"Hold Distribution: {dict(sorted(hold_day_distribution.items()))}")
            
            # P&L by hold period
            profitable_trades = [t for t in completed_trades if t['pnl'] > 0]
            print(f"Profitable Trades: {len(profitable_trades)}/{len(completed_trades)} ({len(profitable_trades)/len(completed_trades)*100:.1f}%)")
        
        # Performance metrics
        daily_pnls = [d['daily_pnl'] for d in self.daily_reports]
        winning_days = len([p for p in daily_pnls if p > 0])
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"Winning Days: {winning_days}/{len(daily_pnls)} ({winning_days/len(daily_pnls)*100:.1f}%)")
        print(f"Best Day: ${max(daily_pnls):+.2f}")
        print(f"Worst Day: ${min(daily_pnls):+.2f}")
        print(f"Avg Daily P&L: ${np.mean(daily_pnls):+.2f}")
        
        # Current position
        if self.portfolio['shares'] > 0:
            print(f"\n📋 CURRENT POSITION:")
            print(f"Shares Held: {self.portfolio['shares']}")
            print(f"Days Held: {self.portfolio['days_held']}")
            print(f"Entry Price: ${self.portfolio['position_entry_price']:.2f}")
            current_price = self.daily_reports[-1]['price']
            unrealized_pnl = (current_price - self.portfolio['position_entry_price']) * self.portfolio['shares']
            print(f"Unrealized P&L: ${unrealized_pnl:+.2f}")
        
        self.save_enhanced_results()

    def save_enhanced_results(self):
        """Save enhanced results with hold strategy metrics"""
        results_dir = Path(f"enhanced_results_{self.stock_symbol}")
        results_dir.mkdir(exist_ok=True)
        
        # Save all data
        if self.daily_reports:
            pd.DataFrame(self.daily_reports).to_csv(results_dir / "daily_reports.csv", index=False)
        
        if self.portfolio['trades']:
            pd.DataFrame(self.portfolio['trades']).to_csv(results_dir / "trades.csv", index=False)
        
        # Enhanced summary
        summary = {
            'stock_symbol': self.stock_symbol,
            'initial_balance': self.initial_balance,
            'final_value': self.daily_reports[-1]['end_value'],
            'total_return': ((self.daily_reports[-1]['end_value'] - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': len(self.portfolio['trades']),
            'hold_strategy': {
                'min_days': self.min_hold_days,
                'max_days': self.max_hold_days,
                'optimal_range': self.optimal_hold_range
            },
            'simulation_date': datetime.now().isoformat()
        }
        
        with open(results_dir / "enhanced_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📁 Enhanced results saved to: {results_dir}")

# Enhanced agent classes with hold strategy awareness
class HoldAwareIntradayTrendAgent(IntradayTrendAgent):
    """Enhanced agent with hold strategy awareness"""
    
    def __init__(self, daily_data, initial_balance, min_hold_days, max_hold_days):
        super().__init__(daily_data, initial_balance)
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days

class HoldAwarePureIntradayAgent(PureIntradayAgent):
    """Enhanced agent with position management"""
    
    def __init__(self, minute_data, initial_balance, min_hold_days, max_hold_days):
        super().__init__(minute_data, initial_balance)
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days

def main():
    """Enhanced main function"""
    parser = argparse.ArgumentParser(description='Enhanced Intraday Trading Simulator')
    parser.add_argument('symbol', help='Stock symbol (any ticker: NVDA, AAPL, TSLA, MSFT, etc.)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--min-hold', type=int, default=2, help='Minimum hold days')
    parser.add_argument('--max-hold', type=int, default=8, help='Maximum hold days')
    
    args = parser.parse_args()
    
    try:
        # Initialize enhanced simulator
        simulator = EnhancedIntradaySimulator(args.symbol, args.balance)
        simulator.min_hold_days = args.min_hold
        simulator.max_hold_days = args.max_hold
        
        # Fetch data for any ticker
        all_data = simulator.fetch_available_data()
        if not all_data:
            print(f"❌ No data available for {args.symbol}")
            return
        
        # Prepare data
        simulator.prepare_data(all_data)
        
        # Train agents once (not daily)
        if not simulator.train_agents_once():
            print("❌ Training failed")
            return
        
        # Run enhanced simulation
        simulator.run_enhanced_simulation()
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

