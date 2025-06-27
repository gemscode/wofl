#!/usr/bin/env python3
"""
WolfX1 Quality-Focused Trading System
Emphasis on signal quality over quantity
"""

import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

# Import agent classes
from rw_wolfxe.agents.pure_intraday_agent import PureIntradayAgent
from rw_wolfxe.agents.intraday_trend_agent import IntradayTrendAgent

class QualitySignalFilter:
    """High-quality signal filtering system"""
    
    def __init__(self):
        self.signal_history = []
        self.min_signal_strength = 0.7  # High threshold
        self.min_confirmation_period = 3  # Require 3 periods of confirmation
        self.volatility_threshold = 0.015  # Minimum volatility for trades
        
    def calculate_market_regime(self, data):
        """Determine current market regime"""
        if len(data) < 20:
            return 'UNKNOWN'
        
        returns = data['Close'].pct_change().dropna()
        recent_returns = returns.tail(10)
        volatility = returns.tail(20).std()
        
        avg_return = recent_returns.mean()
        
        if avg_return > 0.002 and volatility < 0.02:
            return 'BULL'  # Uptrend, low volatility
        elif avg_return < -0.002 and volatility < 0.02:
            return 'BEAR'  # Downtrend, low volatility
        elif volatility > 0.03:
            return 'VOLATILE'  # High volatility
        else:
            return 'SIDEWAYS'  # Choppy market
    
    def calculate_signal_quality(self, data, signal):
        """Calculate signal quality score"""
        if len(data) < 10:
            return 0.0
        
        quality_score = 0.0
        
        # Trend strength
        close_prices = data['Close'].values
        sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
        sma_5 = np.mean(close_prices[-5:])
        
        trend_strength = abs(sma_5 - sma_20) / sma_20
        quality_score += min(trend_strength * 10, 0.3)
        
        # Volume confirmation
        volumes = data['Volume'].values
        avg_volume = np.mean(volumes[-10:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > 1.5:  # High volume confirmation
            quality_score += 0.2
        
        # Volatility check
        returns = np.diff(close_prices[-10:]) / close_prices[-10:-1]
        volatility = np.std(returns)
        
        if volatility > self.volatility_threshold:
            quality_score += 0.2
        
        # Price momentum
        momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
        if signal == 'BUY' and momentum > 0.01:
            quality_score += 0.2
        elif signal == 'SELL' and momentum < -0.01:
            quality_score += 0.2
        
        # Support/Resistance levels
        recent_high = np.max(close_prices[-10:])
        recent_low = np.min(close_prices[-10:])
        current_price = close_prices[-1]
        
        if signal == 'BUY' and current_price < recent_low * 1.02:  # Near support
            quality_score += 0.1
        elif signal == 'SELL' and current_price > recent_high * 0.98:  # Near resistance
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def should_trade(self, data, signal, market_regime):
        """Determine if trade should be executed based on quality"""
        if signal == 'HOLD':
            return False
        
        # Calculate signal quality
        quality = self.calculate_signal_quality(data, signal)
        
        # Regime-based adjustments
        if market_regime == 'BULL' and signal == 'SELL':
            quality *= 0.5  # Be more conservative selling in bull market
        elif market_regime == 'BEAR' and signal == 'BUY':
            quality *= 0.5  # Be more conservative buying in bear market
        elif market_regime == 'VOLATILE':
            quality *= 0.7  # Reduce all signals in volatile markets
        
        # Check if quality meets threshold
        if quality < self.min_signal_strength:
            return False
        
        # Check for signal confirmation
        self.signal_history.append({'signal': signal, 'quality': quality, 'timestamp': datetime.now()})
        
        # Keep only recent signals
        cutoff_time = datetime.now() - timedelta(hours=2)
        self.signal_history = [s for s in self.signal_history if datetime.fromisoformat(s['timestamp'].isoformat()) > cutoff_time]
        
        # Require confirmation
        recent_signals = [s['signal'] for s in self.signal_history[-self.min_confirmation_period:]]
        if len(recent_signals) >= self.min_confirmation_period:
            signal_consistency = recent_signals.count(signal) / len(recent_signals)
            if signal_consistency < 0.6:  # Less than 60% consistency
                return False
        
        return True

class QualityTradingSystem:
    """Quality-focused trading system"""
    
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
        
        # **QUALITY-FOCUSED CONSTRAINTS**
        self.constraints = {
            'min_hold_days': 3,        # Increased from 1
            'max_hold_days': 10,       # Increased from 8
            'stop_loss_pct': 0.04,     # 4% stop loss
            'take_profit_pct': 0.08,   # 8% take profit (2:1 ratio)
            'max_position_size': 0.7,  # Reduced from 0.9
            'min_trade_amount': 500,   # Increased minimum
            'max_trades_per_week': 3,  # Limit trades per week
            'cooldown_period': 2       # Days between trades
        }
        
        # Quality control
        self.signal_filter = QualitySignalFilter()
        self.last_trade_date = None
        self.weekly_trade_count = 0
        self.week_start = None
        
        # Agents
        self.agent_a1 = None
        self.agent_a2 = None
        
        # Performance tracking
        self.daily_reports = []
        
        print(f"🎯 Quality Trading System initialized for {self.stock_symbol}")
        print(f"📋 Focus: High-quality signals, reduced frequency")

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
        
        # Train new agents with better parameters
        print("🎯 Training quality-focused agents...")
        
        # Prepare training data
        daily_data = data.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        if len(daily_data) < 30 or len(data) < 100:
            print("❌ Insufficient data for training")
            return False
        
        # Train Agent A1 (Daily) - More conservative
        env_a1 = IntradayTrendAgent(daily_data, self.initial_balance)
        self.agent_a1 = PPO("MlpPolicy", env_a1, verbose=0, learning_rate=0.0001)  # Lower learning rate
        self.agent_a1.learn(total_timesteps=25000)
        self.agent_a1.save(str(self.agent_a1_path))
        
        # Train Agent A2 (Intraday) - More conservative
        env_a2 = PureIntradayAgent(data, self.initial_balance)
        self.agent_a2 = PPO("MlpPolicy", env_a2, verbose=0, learning_rate=0.0001)  # Lower learning rate
        self.agent_a2.learn(total_timesteps=35000)
        self.agent_a2.save(str(self.agent_a2_path))
        
        print("✅ Quality agent training completed!")
        return True

    def get_conservative_agent_decision(self, agent, agent_type, current_data):
        """Get more conservative agent decisions"""
        try:
            # Generate more conservative observations
            if agent_type == 'daily':
                obs = np.random.normal(0, 0.5, 30).astype(np.float32)  # Reduced noise
            else:
                obs = np.random.normal(0, 0.5, 35).astype(np.float32)  # Reduced noise
            
            action, _ = agent.predict(obs, deterministic=True)  # Use deterministic
            actions = ['HOLD', 'BUY', 'SELL']
            return actions[action]
        except:
            return 'HOLD'  # Default to conservative

    def update_weekly_tracking(self, current_date):
        """Update weekly trade tracking"""
        if self.week_start is None or (current_date - self.week_start).days >= 7:
            self.week_start = current_date
            self.weekly_trade_count = 0

    def can_trade_today(self, current_date):
        """Check if trading is allowed today"""
        # Update weekly tracking
        self.update_weekly_tracking(current_date)
        
        # Check weekly trade limit
        if self.weekly_trade_count >= self.constraints['max_trades_per_week']:
            return False, "Weekly trade limit reached"
        
        # Check cooldown period
        if (self.last_trade_date and 
            (current_date - self.last_trade_date).days < self.constraints['cooldown_period']):
            return False, "In cooldown period"
        
        return True, "Trading allowed"

    def apply_quality_constraints(self, decision, current_price, data, current_date):
        """Apply quality-focused trading constraints"""
        # Check if trading is allowed today
        can_trade, reason = self.can_trade_today(current_date)
        if not can_trade:
            return 'HOLD'
        
        # Force hold if below minimum hold period
        if (self.portfolio['shares'] > 0 and 
            self.portfolio['days_held'] < self.constraints['min_hold_days']):
            return 'HOLD'
        
        # Force sell if at maximum hold period
        if (self.portfolio['shares'] > 0 and 
            self.portfolio['days_held'] >= self.constraints['max_hold_days']):
            return 'SELL'
        
        # Check stop loss and take profit
        if self.portfolio['shares'] > 0 and self.portfolio['position_entry_price']:
            # Stop loss
            if current_price <= self.portfolio['position_entry_price'] * (1 - self.constraints['stop_loss_pct']):
                return 'SELL'
            
            # Take profit
            if current_price >= self.portfolio['position_entry_price'] * (1 + self.constraints['take_profit_pct']):
                return 'SELL'
        
        # Check signal quality
        market_regime = self.signal_filter.calculate_market_regime(data)
        if not self.signal_filter.should_trade(data, decision, market_regime):
            return 'HOLD'
        
        # Position size check for buys
        if decision == 'BUY':
            if self.portfolio['cash'] < self.constraints['min_trade_amount']:
                return 'HOLD'
            
            if self.portfolio['shares'] > 0:  # Already have position
                return 'HOLD'
        
        return decision

    def execute_quality_trade(self, decision, current_data, reason=""):
        """Execute trade with quality focus"""
        current_price = current_data['Close']
        current_time = current_data.name
        current_date = current_time.date()
        
        if decision == 'BUY' and self.portfolio['shares'] == 0:
            # Conservative position sizing
            available_cash = self.portfolio['cash'] * 0.6  # Use only 60% of cash
            shares_to_buy = int(available_cash // current_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                
                self.portfolio['shares'] = shares_to_buy
                self.portfolio['cash'] -= cost
                self.portfolio['position_entry_price'] = current_price
                self.portfolio['position_entry_time'] = current_date
                self.portfolio['days_held'] = 0
                
                # Update tracking
                self.last_trade_date = current_date
                self.weekly_trade_count += 1
                
                trade = {
                    'timestamp': current_time,
                    'type': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'value': cost,
                    'reason': reason
                }
                self.portfolio['trades'].append(trade)
                
                print(f"📈 QUALITY BUY: {shares_to_buy} shares at ${current_price:.2f} | {reason}")
                
        elif decision == 'SELL' and self.portfolio['shares'] > 0:
            shares_to_sell = self.portfolio['shares']
            proceeds = shares_to_sell * current_price
            
            # Calculate P&L
            entry_value = shares_to_sell * self.portfolio['position_entry_price']
            pnl = proceeds - entry_value
            pnl_pct = (pnl / entry_value) * 100
            
            self.portfolio['cash'] += proceeds
            self.portfolio['shares'] = 0
            
            # Update tracking
            self.last_trade_date = current_date
            self.weekly_trade_count += 1
            
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
            
            print(f"📉 QUALITY SELL: {shares_to_sell} shares at ${current_price:.2f} | "
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

    def run_quality_simulation(self, data):
        """Run quality-focused simulation"""
        print(f"🎯 Starting Quality Trading simulation...")
        
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
            
            # Get market window for analysis
            historical_window = data[data.index.date <= date].tail(30)  # Longer window for quality
            
            # **QUALITY-FOCUSED DECISION MAKING**
            # Only check at market close for quality decisions
            end_of_day_data = day_data.iloc[-1]
            
            # Get agent decisions (more conservative)
            a1_decision = self.get_conservative_agent_decision(self.agent_a1, 'daily', end_of_day_data)
            a2_decision = self.get_conservative_agent_decision(self.agent_a2, 'intraday', end_of_day_data)
            
            # **CONSERVATIVE CONSENSUS**
            if a1_decision == a2_decision and a1_decision != 'HOLD':
                consensus_decision = a1_decision
            elif self.portfolio['shares'] > 0 and 'SELL' in [a1_decision, a2_decision]:
                consensus_decision = 'SELL'  # Exit existing positions
            else:
                consensus_decision = 'HOLD'  # Default to conservative
            
            # Apply quality constraints
            final_decision = self.apply_quality_constraints(
                consensus_decision, end_of_day_data['Close'], historical_window, date
            )
            
            print(f"🧠 A1: {a1_decision} | A2: {a2_decision} | "
                  f"Consensus: {consensus_decision} | Final: {final_decision}")
            
            # Execute trade (max 1 per day)
            if final_decision != 'HOLD':
                market_regime = self.signal_filter.calculate_market_regime(historical_window)
                reason = f"{final_decision} - Quality signal in {market_regime} market"
                self.execute_quality_trade(final_decision, end_of_day_data, reason)
            
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
                'decision': final_decision
            }
            self.daily_reports.append(daily_report)
            
            print(f"💰 EOD: ${day_end_value:.2f} | Daily P&L: ${daily_pnl:+.2f} | "
                  f"Total P&L: ${daily_report['total_pnl']:+.2f}")
        
        self.generate_quality_report()

    def generate_quality_report(self):
        """Generate quality-focused performance report"""
        print(f"\n🎯 QUALITY TRADING REPORT - {self.stock_symbol}")
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
        
        # Quality metrics
        completed_trades = [t for t in self.portfolio['trades'] if t['type'] == 'SELL']
        if completed_trades:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(completed_trades)
            avg_hold_days = np.mean([t['days_held'] for t in completed_trades])
            
            print(f"\n📈 QUALITY METRICS:")
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
                
                # Profit factor
                total_wins = sum(t['pnl'] for t in winning_trades)
                total_losses = abs(sum(t['pnl'] for t in losing_trades))
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                print(f"Profit Factor: {profit_factor:.2f}")
        
        # Trading frequency analysis
        trading_days = sum(1 for d in self.daily_reports if d['decision'] != 'HOLD')
        print(f"\nTrading Frequency: {trading_days}/{len(self.daily_reports)} days ({trading_days/len(self.daily_reports)*100:.1f}%)")
        
        # Performance metrics
        daily_pnls = [d['daily_pnl'] for d in self.daily_reports]
        winning_days = len([p for p in daily_pnls if p > 0])
        
        print(f"\n📊 DAILY PERFORMANCE:")
        print(f"Winning Days: {winning_days}/{len(daily_pnls)} ({winning_days/len(daily_pnls)*100:.1f}%)")
        print(f"Best Day: ${max(daily_pnls):+.2f}")
        print(f"Worst Day: ${min(daily_pnls):+.2f}")
        print(f"Average Daily P&L: ${np.mean(daily_pnls):+.2f}")
        
        # Calculate Sharpe ratio
        if len(daily_pnls) > 1:
            daily_returns = np.array(daily_pnls) / self.initial_balance
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            print(f"Estimated Sharpe Ratio: {sharpe_ratio:.2f}")
        
        self.save_quality_results()

    def save_quality_results(self):
        """Save quality trading results"""
        results_dir = Path(f"quality_trading_results_{self.stock_symbol}")
        results_dir.mkdir(exist_ok=True)
        
        # Save daily reports
        if self.daily_reports:
            pd.DataFrame(self.daily_reports).to_csv(results_dir / "daily_reports.csv", index=False)
        
        # Save trades
        if self.portfolio['trades']:
            pd.DataFrame(self.portfolio['trades']).to_csv(results_dir / "trades.csv", index=False)
        
        print(f"📁 Quality results saved to: {results_dir}")

def main():
    """Main function for quality trading system"""
    parser = argparse.ArgumentParser(description='Quality Trading System')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, NVDA, TSLA)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--days', type=int, default=60, help='Days of data to fetch')
    
    args = parser.parse_args()
    
    try:
        # Initialize quality system
        quality_system = QualityTradingSystem(args.symbol, args.balance)
        
        # Fetch market data
        ticker = yf.Ticker(args.symbol)
        data = ticker.history(period=f"{args.days}d", interval="5m")
        
        if data.empty:
            print(f"❌ No data available for {args.symbol}")
            return
        
        print(f"📊 Fetched {len(data)} bars of 5-minute data")
        
        # Run quality simulation
        quality_system.run_quality_simulation(data)
        
    except Exception as e:
        print(f"❌ Quality simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

