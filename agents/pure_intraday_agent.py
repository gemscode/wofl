#!/usr/bin/env python3
"""
Pure Intraday Agent for minute-level trading
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PureIntradayAgent(gym.Env):
    def __init__(self, minute_data, initial_balance=10000):
        super(PureIntradayAgent, self).__init__()
        self.minute_data = minute_data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)
        self.reset()
    
    def reset(self, seed=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.minutes_since_last_trade = 0
        self.position_entry_price = None
        self.trade_history = []
        self.voluntary_trades = 0
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Create observation vector for intraday agent - FIXED to ensure exactly 35 features"""
        if self.current_step >= len(self.minute_data):
            self.current_step = len(self.minute_data) - 1
        
        current_row = self.minute_data.iloc[self.current_step]
        current_price = current_row['Close']
        
        features = []
        
        # Technical indicators (15 features)
        # RSI (1 feature)
        if self.current_step >= 14:
            prices = self.minute_data.iloc[self.current_step-13:self.current_step+1]['Close']
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).mean()
            loss = (-delta.where(delta < 0, 0)).mean()
            rs = gain / loss if loss != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi / 100.0)
        else:
            features.append(0.5)
        
        # Price momentum for different periods (8 features)
        for period in [5, 10, 15, 20, 30, 60, 120, 240]:
            if self.current_step >= period:
                past_price = self.minute_data.iloc[self.current_step - period]['Close']
                momentum = (current_price - past_price) / past_price
                features.append(momentum)
            else:
                features.append(0.0)
        
        # Moving averages (6 features) - FIXED: exactly 6 features
        for period in [10, 20, 50, 100, 200, 300]:
            if self.current_step >= period:
                ma = self.minute_data.iloc[self.current_step-period+1:self.current_step+1]['Close'].mean()
                ma_ratio = (current_price - ma) / ma if ma > 0 else 0.0
                features.append(ma_ratio)
            else:
                features.append(0.0)
        
        # Volume analysis (5 features)
        for period in [5, 15, 30, 60, 120]:
            if self.current_step >= period and 'Volume' in current_row:
                avg_volume = self.minute_data.iloc[self.current_step-period+1:self.current_step+1]['Volume'].mean()
                volume_ratio = current_row['Volume'] / avg_volume if avg_volume > 0 else 1.0
                features.append(min(volume_ratio, 5.0))  # Cap at 5x
            else:
                features.append(1.0)
        
        # Market microstructure (10 features)
        features.extend([
            self.balance / self.initial_balance,
            self.shares_held / 100.0,
            1.0 if self.shares_held > 0 else 0.0,
            self.minutes_since_last_trade / 11520.0,  # Normalize to max hold
            self.get_time_of_day_factor(),
            self.get_market_session_factor(),
            (current_row['High'] - current_row['Low']) / current_price if current_price > 0 else 0.0,
            len(self.trade_history) / 100.0,
            1.0 if self.is_end_of_day_approaching() else 0.0,
            self.voluntary_trades / max(1, len(self.trade_history))
        ])
        
        # Position management (5 features)
        if self.shares_held > 0 and self.position_entry_price:
            unrealized_pnl = (current_price - self.position_entry_price) / self.position_entry_price
            position_duration = self.minutes_since_last_trade
            features.extend([
                unrealized_pnl,
                position_duration / 480.0,  # Normalize to trading day
                1.0 if unrealized_pnl > 0.02 else 0.0,  # Profitable flag
                1.0 if position_duration > 240 else 0.0,  # Long hold flag
                self.calculate_position_risk()
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # FIXED: Ensure exactly 35 features
        # Current count: 1 (RSI) + 8 (momentum) + 6 (MA) + 5 (volume) + 10 (microstructure) + 5 (position) = 35
        
        # Verify we have exactly 35 features
        if len(features) != 35:
            # Pad or truncate to ensure exactly 35 features
            if len(features) < 35:
                features.extend([0.0] * (35 - len(features)))
            else:
                features = features[:35]
        
        # Ensure all features are finite and properly typed
        observation = np.array(features, dtype=np.float32)
        
        # Replace any NaN or inf values with 0
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation

    def get_time_of_day_factor(self):
        """Market session timing factor"""
        step_in_day = self.current_step % 390  # 390 minutes in trading day
        return step_in_day / 390.0

    def get_market_session_factor(self):
        """Market session factor (open, mid-day, close)"""
        step_in_day = self.current_step % 390
        if step_in_day < 30:  # First 30 minutes
            return 1.0
        elif step_in_day > 360:  # Last 30 minutes
            return 0.8
        else:  # Mid-day
            return 0.5

    def is_end_of_day_approaching(self):
        """Check if end of trading day is approaching"""
        step_in_day = self.current_step % 390
        return step_in_day > 360  # Last 30 minutes

    def calculate_position_risk(self):
        """Calculate position risk based on volatility"""
        if self.current_step >= 20:
            recent_prices = self.minute_data.iloc[self.current_step-19:self.current_step+1]['Close']
            volatility = recent_prices.std() / recent_prices.mean() if recent_prices.mean() > 0 else 0.0
            return min(volatility * 10, 1.0)  # Normalized risk
        return 0.0
    
    def step(self, action):
        self.minutes_since_last_trade += 1
        
        current_row = self.minute_data.iloc[self.current_step]
        current_price = current_row['Close']
        
        # Force trade if holding too long
        forced_trade = False
        if self.minutes_since_last_trade >= 11520:  # 8 days max
            forced_trade = True
            action = 2 if self.shares_held > 0 else 1
        
        reward = 0.0
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price and self.minutes_since_last_trade >= 15:  # Min 15 minutes
                shares_to_buy = int(self.balance // current_price)
                if shares_to_buy > 0:
                    self.shares_held += shares_to_buy
                    self.balance -= shares_to_buy * current_price
                    self.position_entry_price = current_price
                    self.minutes_since_last_trade = 0
                    self.trade_history.append(('BUY', current_price, shares_to_buy))
                    if not forced_trade:
                        self.voluntary_trades += 1
                    reward = 0.001  # Small reward for taking action
                
        elif action == 2:  # Sell
            if self.shares_held > 0 and self.minutes_since_last_trade >= 15:  # Min 15 minutes
                proceeds = self.shares_held * current_price
                self.balance += proceeds
                sold_shares = self.shares_held
                
                # Calculate profit
                if self.position_entry_price:
                    profit_pct = (current_price - self.position_entry_price) / self.position_entry_price
                    reward = profit_pct * 5  # Scale reward
                
                self.shares_held = 0
                self.position_entry_price = None
                self.minutes_since_last_trade = 0
                self.trade_history.append(('SELL', current_price, sold_shares))
                if not forced_trade:
                    self.voluntary_trades += 1
        
        # Portfolio value reward
        portfolio_value = self.balance + self.shares_held * current_price
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
        reward += portfolio_return * 0.01
        
        # Penalty for forced trades
        if forced_trade:
            reward -= 0.01
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.minute_data) - 1
        
        info = {
            'portfolio_value': portfolio_value,
            'forced_trade': forced_trade,
            'total_trades': len(self.trade_history)
        }
        
        return self._get_observation(), reward, done, False, info

