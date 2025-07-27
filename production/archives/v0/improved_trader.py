#!/usr/bin/env python3
"""
Improved Trading Algorithm with Better Decision Logic
Addresses poor performance issues with enhanced strategy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from stable_baselines3 import PPO
import talib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ImprovedTradingAlgorithm:
    """Enhanced trading algorithm with better performance metrics"""
    
    def __init__(self, stock_symbol, initial_balance=10000):
        self.stock_symbol = stock_symbol.upper()
        self.initial_balance = initial_balance
        
        # Enhanced portfolio tracking
        self.portfolio = {
            'cash': initial_balance,
            'shares': 0,
            'position_entry_price': None,
            'position_entry_time': None,
            'days_held': 0,
            'stop_loss': None,
            'take_profit': None,
            'trades': []
        }
        
        # **IMPROVED STRATEGY PARAMETERS**
        self.strategy_params = {
            'min_hold_days': 2,
            'max_hold_days': 8,
            'stop_loss_pct': 0.03,      # 3% stop loss
            'take_profit_pct': 0.06,    # 6% take profit (2:1 ratio)
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.5,     # 50% above average volume
            'trend_confirmation': 3,     # Require 3-period trend confirmation
            'risk_per_trade': 0.02      # Risk 2% per trade
        }
        
        # Performance tracking
        self.performance_metrics = {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        # Technical indicators cache
        self.indicators = {}
        self.scaler = StandardScaler()
        
        print(f"🚀 Improved Trading Algorithm for {self.stock_symbol}")

    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        indicators = {}
        
        # Trend indicators
        indicators['sma_20'] = talib.SMA(close, timeperiod=20)
        indicators['sma_50'] = talib.SMA(close, timeperiod=50)
        indicators['ema_12'] = talib.EMA(close, timeperiod=12)
        indicators['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # Momentum indicators
        indicators['rsi'] = talib.RSI(close, timeperiod=14)
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
        indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close)
        
        # Volatility indicators
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
        indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Volume indicators
        indicators['volume_sma'] = talib.SMA(volume.astype(float), timeperiod=20)
        indicators['ad'] = talib.AD(high, low, close, volume.astype(float))
        
        return indicators

    def generate_enhanced_signals(self, data):
        """Generate trading signals using multiple confirmations"""
        indicators = self.calculate_technical_indicators(data)
        current_price = data['Close'].iloc[-1]
        
        signals = {
            'trend_signal': 0,      # -1 bearish, 0 neutral, 1 bullish
            'momentum_signal': 0,
            'volume_signal': 0,
            'volatility_signal': 0,
            'overall_signal': 'HOLD',
            'confidence': 0.0,
            'stop_loss_price': None,
            'take_profit_price': None
        }
        
        # **TREND ANALYSIS**
        sma_20 = indicators['sma_20'][-1] if not np.isnan(indicators['sma_20'][-1]) else current_price
        sma_50 = indicators['sma_50'][-1] if not np.isnan(indicators['sma_50'][-1]) else current_price
        ema_12 = indicators['ema_12'][-1] if not np.isnan(indicators['ema_12'][-1]) else current_price
        ema_26 = indicators['ema_26'][-1] if not np.isnan(indicators['ema_26'][-1]) else current_price
        
        # Multiple trend confirmations
        trend_signals = []
        if current_price > sma_20 > sma_50:
            trend_signals.append(1)
        elif current_price < sma_20 < sma_50:
            trend_signals.append(-1)
        else:
            trend_signals.append(0)
            
        if ema_12 > ema_26:
            trend_signals.append(1)
        else:
            trend_signals.append(-1)
            
        signals['trend_signal'] = np.mean(trend_signals)
        
        # **MOMENTUM ANALYSIS**
        rsi = indicators['rsi'][-1] if not np.isnan(indicators['rsi'][-1]) else 50
        macd = indicators['macd'][-1] if not np.isnan(indicators['macd'][-1]) else 0
        macd_signal = indicators['macd_signal'][-1] if not np.isnan(indicators['macd_signal'][-1]) else 0
        
        momentum_signals = []
        
        # RSI signals
        if rsi < self.strategy_params['rsi_oversold']:
            momentum_signals.append(1)  # Oversold - bullish
        elif rsi > self.strategy_params['rsi_overbought']:
            momentum_signals.append(-1)  # Overbought - bearish
        else:
            momentum_signals.append(0)
            
        # MACD signals
        if macd > macd_signal and macd > 0:
            momentum_signals.append(1)
        elif macd < macd_signal and macd < 0:
            momentum_signals.append(-1)
        else:
            momentum_signals.append(0)
            
        signals['momentum_signal'] = np.mean(momentum_signals)
        
        # **VOLUME ANALYSIS**
        current_volume = data['Volume'].iloc[-1]
        avg_volume = indicators['volume_sma'][-1] if not np.isnan(indicators['volume_sma'][-1]) else current_volume
        
        if current_volume > avg_volume * self.strategy_params['volume_threshold']:
            signals['volume_signal'] = 1  # High volume confirmation
        else:
            signals['volume_signal'] = 0
            
        # **VOLATILITY ANALYSIS**
        bb_upper = indicators['bb_upper'][-1] if not np.isnan(indicators['bb_upper'][-1]) else current_price * 1.02
        bb_lower = indicators['bb_lower'][-1] if not np.isnan(indicators['bb_lower'][-1]) else current_price * 0.98
        atr = indicators['atr'][-1] if not np.isnan(indicators['atr'][-1]) else current_price * 0.02
        
        if current_price <= bb_lower:
            signals['volatility_signal'] = 1  # Near lower band - potential buy
        elif current_price >= bb_upper:
            signals['volatility_signal'] = -1  # Near upper band - potential sell
        else:
            signals['volatility_signal'] = 0
            
        # **OVERALL SIGNAL GENERATION**
        signal_weights = {
            'trend': 0.4,
            'momentum': 0.3,
            'volume': 0.2,
            'volatility': 0.1
        }
        
        weighted_signal = (
            signals['trend_signal'] * signal_weights['trend'] +
            signals['momentum_signal'] * signal_weights['momentum'] +
            signals['volume_signal'] * signal_weights['volume'] +
            signals['volatility_signal'] * signal_weights['volatility']
        )
        
        # Calculate confidence based on signal alignment
        signal_alignment = abs(signals['trend_signal']) + abs(signals['momentum_signal'])
        signals['confidence'] = min(signal_alignment / 2.0, 1.0)
        
        # Generate final signal with confidence threshold
        confidence_threshold = 0.6
        
        if weighted_signal > 0.3 and signals['confidence'] > confidence_threshold:
            signals['overall_signal'] = 'BUY'
            # Set stop loss and take profit
            signals['stop_loss_price'] = current_price * (1 - self.strategy_params['stop_loss_pct'])
            signals['take_profit_price'] = current_price * (1 + self.strategy_params['take_profit_pct'])
        elif weighted_signal < -0.3 and signals['confidence'] > confidence_threshold:
            signals['overall_signal'] = 'SELL'
        else:
            signals['overall_signal'] = 'HOLD'
            
        return signals

    def calculate_position_size(self, current_price, stop_loss_price):
        """Calculate position size based on risk management"""
        if stop_loss_price is None:
            return int(self.portfolio['cash'] * 0.1 // current_price)  # 10% of cash
        
        # Risk-based position sizing
        risk_per_share = current_price - stop_loss_price
        max_risk_amount = self.portfolio['cash'] * self.strategy_params['risk_per_trade']
        
        if risk_per_share > 0:
            position_size = int(max_risk_amount // risk_per_share)
            # Don't exceed 50% of available cash
            max_shares = int(self.portfolio['cash'] * 0.5 // current_price)
            return min(position_size, max_shares)
        else:
            return int(self.portfolio['cash'] * 0.1 // current_price)

    def execute_improved_trade(self, signal_data, current_data):
        """Execute trades with improved logic"""
        signal = signal_data['overall_signal']
        confidence = signal_data['confidence']
        current_price = current_data['Close']
        current_time = current_data.name
        
        # **BUY LOGIC**
        if signal == 'BUY' and self.portfolio['shares'] == 0:
            stop_loss_price = signal_data['stop_loss_price']
            take_profit_price = signal_data['take_profit_price']
            
            shares_to_buy = self.calculate_position_size(current_price, stop_loss_price)
            
            if shares_to_buy > 0 and self.portfolio['cash'] >= shares_to_buy * current_price:
                cost = shares_to_buy * current_price
                
                self.portfolio['shares'] = shares_to_buy
                self.portfolio['cash'] -= cost
                self.portfolio['position_entry_price'] = current_price
                self.portfolio['position_entry_time'] = current_time.date()
                self.portfolio['stop_loss'] = stop_loss_price
                self.portfolio['take_profit'] = take_profit_price
                self.portfolio['days_held'] = 0
                
                trade = {
                    'timestamp': current_time,
                    'type': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'value': cost,
                    'confidence': confidence,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'reason': 'Signal-based entry'
                }
                self.portfolio['trades'].append(trade)
                
                print(f"📈 BUY: {shares_to_buy} shares at ${current_price:.2f} | "
                      f"Confidence: {confidence:.2f} | SL: ${stop_loss_price:.2f} | "
                      f"TP: ${take_profit_price:.2f}")
                
        # **SELL LOGIC** 
        elif self.portfolio['shares'] > 0:
            should_sell = False
            sell_reason = ""
            
            # Check stop loss
            if self.portfolio['stop_loss'] and current_price <= self.portfolio['stop_loss']:
                should_sell = True
                sell_reason = "Stop loss triggered"
                
            # Check take profit
            elif self.portfolio['take_profit'] and current_price >= self.portfolio['take_profit']:
                should_sell = True
                sell_reason = "Take profit triggered"
                
            # Check max hold period
            elif self.portfolio['days_held'] >= self.strategy_params['max_hold_days']:
                should_sell = True
                sell_reason = "Max hold period reached"
                
            # Check signal-based exit
            elif signal == 'SELL' and confidence > 0.7:
                should_sell = True
                sell_reason = "Signal-based exit"
                
            if should_sell:
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
                    'reason': sell_reason
                }
                self.portfolio['trades'].append(trade)
                
                print(f"📉 SELL: {shares_to_sell} shares at ${current_price:.2f} | "
                      f"Held {self.portfolio['days_held']} days | "
                      f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | {sell_reason}")
                
                # Reset position
                self.portfolio['position_entry_price'] = None
                self.portfolio['position_entry_time'] = None
                self.portfolio['stop_loss'] = None
                self.portfolio['take_profit'] = None
                self.portfolio['days_held'] = 0

    def run_improved_simulation(self, data):
        """Run simulation with improved algorithm"""
        print(f"🚀 Running improved simulation for {self.stock_symbol}...")
        
        # Ensure we have enough data for indicators
        if len(data) < 200:
            print("❌ Insufficient data for technical analysis")
            return
        
        # Group by trading days
        simulation_days = data.groupby(data.index.date)
        total_days = len(simulation_days)
        
        daily_reports = []
        day_counter = 0
        
        for date, day_data in simulation_days:
            day_counter += 1
            
            # Update days held
            if (self.portfolio['shares'] > 0 and 
                self.portfolio['position_entry_time'] and 
                date > self.portfolio['position_entry_time']):
                self.portfolio['days_held'] = (date - self.portfolio['position_entry_time']).days
            
            print(f"\n📅 Day {day_counter}/{total_days} - {date}")
            
            # Get historical data up to current day for indicators
            historical_data = data[data.index.date <= date]
            
            if len(historical_data) < 50:  # Need minimum data for indicators
                continue
                
            day_start_value = (self.portfolio['cash'] + 
                             self.portfolio['shares'] * day_data['Close'].iloc[0])
            
            # Generate signals using historical data
            signals = self.generate_enhanced_signals(historical_data)
            
            # Execute trades on key decision points
            decision_points = [
                day_data.iloc[len(day_data)//4],    # Early morning
                day_data.iloc[len(day_data)//2],    # Midday  
                day_data.iloc[3*len(day_data)//4],  # Afternoon
                day_data.iloc[-1]                   # Close
            ]
            
            for bar_data in decision_points:
                self.execute_improved_trade(signals, bar_data)
                
                # Only one major trade per day
                if len([t for t in self.portfolio['trades'] 
                       if t['timestamp'].date() == date]) > 0:
                    break
            
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
                'signal': signals['overall_signal'],
                'confidence': signals['confidence']
            }
            daily_reports.append(daily_report)
            
            print(f"💰 EOD: ${day_end_value:.2f} | Daily P&L: ${daily_pnl:+.2f} | "
                  f"Signal: {signals['overall_signal']} ({signals['confidence']:.2f})")
        
        self.calculate_performance_metrics()
        self.generate_improved_report(daily_reports)

    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        completed_trades = [t for t in self.portfolio['trades'] if t['type'] == 'SELL']
        
        if not completed_trades:
            return
        
        # Win rate
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        self.performance_metrics['win_rate'] = len(winning_trades) / len(completed_trades)
        
        # Average win/loss
        if winning_trades:
            self.performance_metrics['avg_win'] = np.mean([t['pnl'] for t in winning_trades])
        
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        if losing_trades:
            self.performance_metrics['avg_loss'] = abs(np.mean([t['pnl'] for t in losing_trades]))
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        
        if total_losses > 0:
            self.performance_metrics['profit_factor'] = total_wins / total_losses

    def generate_improved_report(self, daily_reports):
        """Generate comprehensive performance report"""
        print(f"\n📊 IMPROVED ALGORITHM REPORT - {self.stock_symbol}")
        print("=" * 60)
        
        final_value = daily_reports[-1]['end_value'] if daily_reports else self.initial_balance
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {len(self.portfolio['trades'])}")
        
        # Enhanced performance metrics
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"Win Rate: {self.performance_metrics['win_rate']:.1%}")
        print(f"Average Win: ${self.performance_metrics['avg_win']:+.2f}")
        print(f"Average Loss: ${self.performance_metrics['avg_loss']:+.2f}")
        print(f"Profit Factor: {self.performance_metrics['profit_factor']:.2f}")
        
        # Risk management analysis
        completed_trades = [t for t in self.portfolio['trades'] if t['type'] == 'SELL']
        if completed_trades:
            avg_hold = np.mean([t['days_held'] for t in completed_trades])
            print(f"Average Hold Period: {avg_hold:.1f} days")
            
            # Stop loss/take profit effectiveness
            sl_trades = len([t for t in completed_trades if 'Stop loss' in t['reason']])
            tp_trades = len([t for t in completed_trades if 'Take profit' in t['reason']])
            print(f"Stop Loss Exits: {sl_trades}")
            print(f"Take Profit Exits: {tp_trades}")

def main():
    """Main function for improved algorithm"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Trading Algorithm')
    parser.add_argument('symbol', help='Stock symbol')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    
    args = parser.parse_args()
    
    # Initialize improved algorithm
    algo = ImprovedTradingAlgorithm(args.symbol, args.balance)
    
    # Fetch data
    ticker = yf.Ticker(args.symbol)
    data = ticker.history(period="60d", interval="5m")
    
    if data.empty:
        print(f"❌ No data available for {args.symbol}")
        return
    
    # Run improved simulation
    algo.run_improved_simulation(data)

if __name__ == "__main__":
    main()

