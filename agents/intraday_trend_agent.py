#!/usr/bin/env python3
"""
Intraday Trend Agent for daily trend analysis
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class IntradayTrendAgent(gym.Env):
    def __init__(self, daily_data, initial_balance=10000):
        super(IntradayTrendAgent, self).__init__()
        self.daily_data = daily_data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        self.reset()
    
    def reset(self, seed=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.hours_since_last_trade = 0
        self.trade_history = []
        self.voluntary_trades = 0
        self.forced_trades = 0
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Create observation vector for the agent"""
        if self.current_step >= len(self.daily_data):
            self.current_step = len(self.daily_data) - 1
        
        current_row = self.daily_data.iloc[self.current_step]
        current_price = current_row['Close']
        
        # Basic features (simplified for demonstration)
        features = []
        
        # Price momentum features (10 features)
        for period in [3, 5, 10, 15, 20, 30, 50, 100, 150, 200]:
            if self.current_step >= period:
                past_price = self.daily_data.iloc[self.current_step - period]['Close']
                momentum = (current_price - past_price) / past_price
                features.append(momentum)
            else:
                features.append(0.0)
        
        # Volatility features (5 features)
        for window in [5, 10, 20, 50, 100]:
            if self.current_step >= window:
                prices = self.daily_data.iloc[self.current_step-window+1:self.current_step+1]['Close']
                vol = prices.std() / prices.mean() if prices.mean() > 0 else 0.0
                features.append(vol)
            else:
                features.append(0.0)
        
        # Portfolio features (8 features)
        portfolio_value = self.balance + self.shares_held * current_price
        features.extend([
            self.balance / self.initial_balance,
            self.shares_held / 100.0,
            portfolio_value / self.initial_balance,
            (portfolio_value - self.initial_balance) / self.initial_balance,
            1.0 if self.shares_held > 0 else 0.0,
            self.hours_since_last_trade / 192.0,  # Normalize to max hold period
            len(self.trade_history) / 50.0,
            self.voluntary_trades / max(1, len(self.trade_history))
        ])
        
        # Market features (7 features)
        features.extend([
            (current_row['High'] - current_row['Low']) / current_price,
            (current_price - current_row['Open']) / current_row['Open'],
            current_row.get('Volume', 0) / 1e6 if 'Volume' in current_row else 0,
            1.0 if 24 <= self.hours_since_last_trade <= 96 else 0.0,  # Optimal hold range
            self.forced_trades / max(1, len(self.trade_history)),
            1.0 if self.hours_since_last_trade >= 4 else 0.0,  # Can trade flag
            np.sin(2 * np.pi * self.current_step / 252)  # Seasonal component
        ])
        
        return np.array(features[:30], dtype=np.float32)
    
    def step(self, action):
        self.hours_since_last_trade += 24  # Increment by day
        
        current_row = self.daily_data.iloc[self.current_step]
        current_price = current_row['Close']
        
        # Force trade if holding too long
        forced_trade = False
        if self.hours_since_last_trade >= 192:  # 8 days max
            forced_trade = True
            action = 2 if self.shares_held > 0 else 1
            self.forced_trades += 1
        
        reward = 0.0
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price:
                shares_to_buy = int(self.balance // current_price)
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.hours_since_last_trade = 0
                self.trade_history.append(('BUY', current_price, shares_to_buy))
                if not forced_trade:
                    self.voluntary_trades += 1
                reward = 0.01  # Small reward for taking action
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price
                self.balance += proceeds
                sold_shares = self.shares_held
                self.shares_held = 0
                self.hours_since_last_trade = 0
                self.trade_history.append(('SELL', current_price, sold_shares))
                if not forced_trade:
                    self.voluntary_trades += 1
                
                # Reward based on profit
                if len(self.trade_history) >= 2:
                    last_buy = None
                    for trade in reversed(self.trade_history[:-1]):
                        if trade[0] == 'BUY':
                            last_buy = trade
                            break
                    
                    if last_buy:
                        profit_pct = (current_price - last_buy[1]) / last_buy[1]
                        reward = profit_pct * 10  # Scale reward
        
        # Portfolio value reward
        portfolio_value = self.balance + self.shares_held * current_price
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
        reward += portfolio_return * 0.1
        
        # Penalty for forced trades
        if forced_trade:
            reward -= 0.05
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.daily_data) - 1
        
        info = {
            'portfolio_value': portfolio_value,
            'forced_trade': forced_trade,
            'total_trades': len(self.trade_history)
        }
        
        return self._get_observation(), reward, done, False, info

