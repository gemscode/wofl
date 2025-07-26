#!/usr/bin/env python3
"""
RL-Enhanced Trading System
Combines HMM + TA + Candlesticks with Deep RL to discover 12% profit patterns
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from finta import TA
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings

# Project setup
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

from shared.data_manager import DataManager
from shared.data_publisher import DataPublisher

warnings.filterwarnings("ignore")

# RL Components
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class RLConfig:
    state_dim: int = 50
    action_dim: int = 3  # 0=hold, 1=buy, 2=sell
    hidden_dim: int = 256
    lr: float = 1e-4
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    memory_size: int = 10000
    target_update: int = 100
    profit_target: float = 0.12
    stop_loss: float = 0.05

class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Dueling DQN architecture
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        features = self.feature_net(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class ReplayBuffer:
    """Experience replay buffer for RL training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class TradingEnvironment:
    """Trading environment for RL agent"""
    
    def __init__(self, data: pd.DataFrame, config: RLConfig):
        self.data = data.reset_index(drop=True)
        self.config = config
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.position = 0  # 0=no position, 1=long
        self.entry_price = 0
        self.portfolio_value = 10000  # Starting capital
        self.max_steps = len(self.data) - 1
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Extract comprehensive state from current market conditions"""
        if self.current_step >= len(self.data):
            return np.zeros(self.config.state_dim)
            
        row = self.data.iloc[self.current_step]
        
        # Price features (normalized)
        price_features = [
            row['close'] / row['open'] - 1,  # Intrabar return
            row['high'] / row['close'] - 1,   # Upper wick
            row['low'] / row['close'] - 1,    # Lower wick
            row['volume'] / row.get('volume_ma', row['volume']),  # Volume ratio
        ]
        
        # Technical indicators (already normalized in preprocessing)
        ta_features = [
            row.get('rsi', 50) / 100 - 0.5,  # RSI centered at 0
            row.get('macd', 0),
            row.get('bb_pos', 0.5) - 0.5,    # BB position centered
            row.get('atr_norm', 0),
            (row.get('close', 0) / row.get('sma20', row.get('close', 1))) - 1,
        ]
        
        # HMM regime features
        regime_features = [
            row.get('reg_state', 1) / 2,     # Normalize regime state
            row.get('reg_stab', 0.5),        # Regime stability
            row.get('reg_prob_0', 0.33),     # Regime probabilities
            row.get('reg_prob_1', 0.33),
            row.get('reg_prob_2', 0.33),
        ]
        
        # Candlestick patterns
        pattern_features = [
            row.get('hammer', 0),
            row.get('doji', 0),
            row.get('engulfing_bull', 0),
            row.get('shooting_star', 0),
            row.get('engulfing_bear', 0),
        ]
        
        # Position and market context
        context_features = [
            self.position,  # Current position
            (self.current_step / self.max_steps),  # Time progression
        ]
        
        # Lookback window features (recent price action)
        lookback = 10
        if self.current_step >= lookback:
            recent_data = self.data.iloc[self.current_step-lookback:self.current_step]
            momentum_features = [
                (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1,
                recent_data['volume'].mean() / recent_data['volume'].std() if recent_data['volume'].std() > 0 else 0,
                recent_data['close'].pct_change().std(),
            ]
        else:
            momentum_features = [0, 0, 0]
        
        # Combine all features
        all_features = (price_features + ta_features + regime_features + 
                       pattern_features + context_features + momentum_features)
        
        # Pad or truncate to exact state dimension
        if len(all_features) < self.config.state_dim:
            all_features.extend([0] * (self.config.state_dim - len(all_features)))
        else:
            all_features = all_features[:self.config.state_dim]
            
        return np.array(all_features, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return new state, reward, done, info"""
        if self.done:
            return self._get_state(), 0, True, {}
            
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        info = {'action': action, 'price': current_price}
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            info['action_taken'] = 'BUY'
            
        elif action == 2 and self.position == 1:  # Sell
            if self.entry_price > 0:
                pnl_pct = (current_price / self.entry_price) - 1
                
                # Reward structure for 12% target
                if pnl_pct >= self.config.profit_target:
                    reward = 100 * pnl_pct  # Big reward for hitting target
                elif pnl_pct >= 0.06:  # Halfway to target
                    reward = 50 * pnl_pct
                elif pnl_pct > 0:
                    reward = 10 * pnl_pct  # Small reward for any profit
                elif pnl_pct <= -self.config.stop_loss:
                    reward = -50  # Penalty for hitting stop loss
                else:
                    reward = 100 * pnl_pct  # Proportional loss penalty
                    
                self.portfolio_value *= (1 + pnl_pct)
                info['pnl_pct'] = pnl_pct
                info['action_taken'] = 'SELL'
                
            self.position = 0
            self.entry_price = 0
            
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= self.max_steps:
            self.done = True
            # Force close any open position
            if self.position == 1 and self.entry_price > 0:
                final_pnl = (current_price / self.entry_price) - 1
                reward += 50 * final_pnl  # Final position reward
                
        # Small penalty for holding too long without action
        if self.position == 1 and self.entry_price > 0:
            hold_periods = self.current_step - getattr(self, 'entry_step', self.current_step)
            if hold_periods > 50:  # Holding longer than 50 periods
                reward -= 0.1
                
        next_state = self._get_state()
        return next_state, reward, self.done, info

class PatternRecognizer:
    """Enhanced pattern recognition combining all analysis types"""
    
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect comprehensive candlestick patterns"""
        patterns = pd.DataFrame(index=df.index)
        
        # Basic candle properties
        body_size = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        total_range = df['high'] - df['low']
        total_range = total_range.replace(0, 1e-8)
        
        # Pattern detection
        patterns['hammer'] = ((lower_shadow > 2 * body_size) & 
                             (upper_shadow < 0.1 * total_range) & 
                             (body_size > 0.001 * df['close'])).astype(int)
        
        patterns['doji'] = ((body_size < 0.1 * total_range) &
                           (total_range > 0.002 * df['close'])).astype(int)
        
        patterns['shooting_star'] = ((upper_shadow > 2 * body_size) & 
                                   (lower_shadow < 0.1 * total_range) & 
                                   (body_size > 0.001 * df['close'])).astype(int)
        
        patterns['engulfing_bull'] = ((df['close'] > df['open']) & 
                                     (df['close'].shift(1) < df['open'].shift(1)) & 
                                     (df['open'] < df['close'].shift(1)) & 
                                     (df['close'] > df['open'].shift(1))).astype(int)
        
        patterns['engulfing_bear'] = ((df['close'] < df['open']) & 
                                     (df['close'].shift(1) > df['open'].shift(1)) & 
                                     (df['open'] > df['close'].shift(1)) & 
                                     (df['close'] < df['open'].shift(1))).astype(int)
        
        return patterns

class MarketRegimeAnalyzer:
    """HMM-based market regime detection"""
    
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.model = None
        self.scaler = StandardScaler()
        
    def fit_predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit HMM and return states and probabilities"""
        features = self._prepare_features(df)
        
        try:
            features_scaled = self.scaler.fit_transform(features)
            
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="spherical",
                n_iter=100,
                random_state=42
            )
            
            self.model.fit(features_scaled)
            states = self.model.predict(features_scaled)
            probs = self.model.predict_proba(features_scaled)
            
            return states, probs
            
        except Exception as e:
            print(f"HMM failed: {e}, using fallback")
            states = np.ones(len(df))
            probs = np.full((len(df), self.n_components), 1/self.n_components)
            return states, probs
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM"""
        returns = df['close'].pct_change().fillna(0).clip(-0.1, 0.1)
        volatility = returns.rolling(20, min_periods=5).std().fillna(0)
        volume_change = df['volume'].pct_change().fillna(0).clip(-5, 5)
        momentum = (df['close'] / df['close'].shift(10) - 1).fillna(0).clip(-0.5, 0.5)
        
        features = np.column_stack([returns, volatility, volume_change, momentum])
        return np.nan_to_num(features)

class RLTradingAgent:
    """Deep RL Trading Agent"""
    
    def __init__(self, config: RLConfig, device: torch.device):
        self.config = config
        self.device = device
        self.epsilon = config.epsilon_start
        
        # Networks
        self.q_network = DQNNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.target_network = DQNNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training components
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.memory_size)
        self.steps_done = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.config.batch_size:
            return
        
        experiences = self.memory.sample(self.config.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.config.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Update target network
        if self.steps_done % self.config.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps_done += 1
        
        return loss.item()

class RLTrainerSystem:
    """Main RL-enhanced trading system"""
    
    def __init__(self, symbol: str, window_min: int = 15):
        self.symbol = symbol.upper()
        self.window_min = window_min
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        
        # Initialize components
        self.pattern_recognizer = PatternRecognizer()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.config = RLConfig()
        
        # Data management
        self.dm = DataManager()
        self.pub = DataPublisher()
        
        print(f"RL Trading System initialized for {symbol} on {self.device}")
        print(f"Target: {self.config.profit_target*100}% profit using RL pattern discovery")
    
    def resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to desired timeframe"""
        rule = f"{self.window_min}T"
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        return df.resample(rule).agg(agg).dropna()
    
    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators and patterns"""
        print(f"Enriching {len(df)} bars with comprehensive analysis...")
        
        # Technical indicators
        df['sma20'] = df['close'].rolling(20, min_periods=10).mean()
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean().replace(0, 1e-8)
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # MACD
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_mid = df['close'].rolling(20, min_periods=10).mean()
        bb_std = df['close'].rolling(20, min_periods=10).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_pos'] = ((df['close'] - df['bb_lower']) / 
                       (df['bb_upper'] - df['bb_lower']).replace(0, 1e-8)).fillna(0.5)
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14, min_periods=7).mean()
        df['atr_norm'] = df['atr'] / df['close']
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20, min_periods=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1e-8)
        
        # Candlestick patterns
        patterns = self.pattern_recognizer.detect_candlestick_patterns(df)
        for col in patterns.columns:
            df[col] = patterns[col]
        
        # HMM regime analysis
        states, probs = self.regime_analyzer.fit_predict(df)
        df['reg_state'] = states
        df['reg_stab'] = pd.Series(states).rolling(10, min_periods=1).apply(
            lambda x: 1/max(len(np.unique(x)), 1), raw=True
        ).fillna(0.5)
        
        # Add regime probabilities
        for i in range(probs.shape[1]):
            df[f'reg_prob_{i}'] = probs[:, i]
        
        # Fill NaN values
        df = df.ffill().bfill().fillna(0)
        
        print(f"Enrichment complete: {len(df)} bars with {len(df.columns)} features")
        return df
    
    def train_rl_agent(self, data: pd.DataFrame, episodes: int = 1000) -> RLTradingAgent:
        """Train RL agent to discover 12% profit patterns"""
        print(f"\nTraining RL agent for {episodes} episodes...")
        print(f"Environment: {len(data)} bars, Target: {self.config.profit_target*100}% profit")
        
        # Initialize agent and environment
        agent = RLTradingAgent(self.config, self.device)
        env = TradingEnvironment(data, self.config)
        
        # Training metrics
        episode_rewards = []
        episode_profits = []
        successful_trades = 0
        total_trades = 0
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_pnl = 0
            trades_this_episode = 0
            
            while not env.done:
                # Select and execute action
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Track episode metrics
                episode_reward += reward
                if 'pnl_pct' in info:
                    episode_pnl += info['pnl_pct']
                    trades_this_episode += 1
                    total_trades += 1
                    if info['pnl_pct'] >= self.config.profit_target:
                        successful_trades += 1
                
                # Train agent
                if len(agent.memory) > agent.config.batch_size:
                    loss = agent.train_step()
                
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_profits.append(episode_pnl)
            
            # Progress reporting
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_profit = np.mean(episode_profits[-100:])
                success_rate = (successful_trades / max(total_trades, 1)) * 100
                
                print(f"Episode {episode+1:4d} | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Avg Profit: {avg_profit:6.1%} | "
                      f"Success Rate: {success_rate:5.1f}% | "
                      f"Epsilon: {agent.epsilon:.3f}")
        
        print(f"\nTraining completed!")
        print(f"Total trades: {total_trades}")
        print(f"Successful 12% trades: {successful_trades}")
        print(f"Success rate: {(successful_trades/max(total_trades,1))*100:.1f}%")
        
        return agent
    
    def evaluate_agent(self, agent: RLTradingAgent, data: pd.DataFrame) -> Dict:
        """Evaluate trained agent performance"""
        print(f"\nEvaluating agent on {len(data)} bars...")
        
        env = TradingEnvironment(data, self.config)
        state = env.reset()
        
        trades = []
        portfolio_values = [env.portfolio_value]
        
        while not env.done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            if 'pnl_pct' in info:
                trades.append({
                    'entry_price': env.entry_price if env.position == 0 else info['price'],
                    'exit_price': info['price'],
                    'pnl_pct': info['pnl_pct'],
                    'hit_target': info['pnl_pct'] >= self.config.profit_target
                })
            
            portfolio_values.append(env.portfolio_value)
            state = next_state
        
        # Calculate metrics
        total_trades = len(trades)
        profitable_trades = sum(1 for t in trades if t['pnl_pct'] > 0)
        target_hits = sum(1 for t in trades if t['hit_target'])
        
        avg_profit = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        max_profit = max([t['pnl_pct'] for t in trades]) if trades else 0
        
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        results = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'target_hits': target_hits,
            'win_rate': (profitable_trades / max(total_trades, 1)) * 100,
            'target_hit_rate': (target_hits / max(total_trades, 1)) * 100,
            'avg_profit_per_trade': avg_profit,
            'max_profit': max_profit,
            'total_return': total_return,
            'final_portfolio_value': portfolio_values[-1]
        }
        
        print(f"Evaluation Results:")
        print(f"  Total trades: {total_trades}")
        print(f"  Profitable trades: {profitable_trades} ({results['win_rate']:.1f}%)")
        print(f"  12% target hits: {target_hits} ({results['target_hit_rate']:.1f}%)")
        print(f"  Average profit per trade: {avg_profit:.1%}")
        print(f"  Maximum profit achieved: {max_profit:.1%}")
        print(f"  Total portfolio return: {total_return:.1%}")
        
        return results
    
    def run_full_training(self, symbol: str, days: int = 60):
        """Run complete RL training pipeline"""
        print(f"\n{'='*60}")
        print(f"RL-Enhanced Trading System for {symbol}")
        print(f"Target: {self.config.profit_target*100}% profit discovery")
        print(f"{'='*60}")
        
        # Load and prepare data
        raw_data = self.dm.get_training_data(symbol, days)
        if raw_data.empty:
            print("No data available. Please load market data first.")
            return None
        
        print(f"Loaded {len(raw_data)} 1-minute bars")
        
        # Resample and enrich
        resampled_data = self.resample_data(raw_data)
        print(f"Resampled to {len(resampled_data)} {self.window_min}-minute bars")
        
        enriched_data = self.enrich_data(resampled_data)
        
        # Split data for training and evaluation
        split_point = int(len(enriched_data) * 0.8)
        train_data = enriched_data.iloc[:split_point]
        eval_data = enriched_data.iloc[split_point:]
        
        print(f"Training set: {len(train_data)} bars")
        print(f"Evaluation set: {len(eval_data)} bars")
        
        # Train RL agent
        agent = self.train_rl_agent(train_data, episodes=2000)
        
        # Evaluate performance
        results = self.evaluate_agent(agent, eval_data)
        
        # Save the trained agent
        model_path = project_root / "models" / symbol / f"rl_agent_{self.window_min}min.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'q_network_state_dict': agent.q_network.state_dict(),
            'config': self.config,
            'results': results,
            'window_min': self.window_min
        }, model_path)
        
        print(f"\nAgent saved to: {model_path}")
        return agent, results

def main():
    parser = argparse.ArgumentParser(description='RL-Enhanced Trading System')
    parser.add_argument('--symbol', required=True, help='Stock symbol')
    parser.add_argument('--days', type=int, default=60, help='Days of training data')
    parser.add_argument('--window', type=int, default=15, help='Timeframe in minutes')
    parser.add_argument('--episodes', type=int, default=2000, help='Training episodes')
    
    args = parser.parse_args()
    
    # Initialize and run RL training system
    rl_system = RLTrainerSystem(args.symbol, args.window)
    agent, results = rl_system.run_full_training(args.symbol, args.days)
    
    if results and results['target_hit_rate'] > 0:
        print(f"\n🎯 SUCCESS: Agent discovered patterns achieving {results['target_hit_rate']:.1f}% success rate for 12% targets!")
    else:
        print(f"\n📊 Training complete. Agent learned trading patterns with {results['win_rate']:.1f}% win rate.")

if __name__ == "__main__":
    main()

