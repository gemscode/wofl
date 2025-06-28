#!/usr/bin/env python3
"""
Simple Hindsight RL Trading Strategy Optimizer - Minimal Monitoring
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import redis
import sys
import random
from collections import deque
import time

# GPU memory management for Mac M3
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.6'
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# GPU acceleration
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Mac M3 GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
except ImportError:
    print("ERROR: PyTorch not installed")
    sys.exit(1)

sys.path.append(str(Path(__file__).parent.parent))
from core.data_manager import DataManager

def clear_gpu_cache():
    """Clear GPU cache"""
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    import gc
    gc.collect()

class TradingEnvironment:
    """Base trading environment that works well (from original RL agent)"""
    
    def __init__(self, price_data, trading_mode='LONG_ONLY'):
        self.price_data = torch.tensor(price_data['Close'].values, dtype=torch.float32, device=device)
        self.trading_mode = trading_mode
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.cash = 25000.0
        self.shares = 0
        self.entry_price = 0
        self.current_step = 50  # Start after enough data for indicators
        self.max_steps = len(self.price_data) - 10
        self.total_return = 0
        self.trades = 0
        self.wins = 0
        
        return self.get_state()
    
    def get_state(self):
        """Get current state for RL agent"""
        if self.current_step >= len(self.price_data) - 20:
            return torch.zeros(15, device=device)
        
        # Price features
        current_price = self.price_data[self.current_step]
        prices_20 = self.price_data[max(0, self.current_step-20):self.current_step]
        
        # Technical indicators
        sma_5 = torch.mean(prices_20[-5:]) if len(prices_20) >= 5 else current_price
        sma_10 = torch.mean(prices_20[-10:]) if len(prices_20) >= 10 else current_price
        sma_20 = torch.mean(prices_20) if len(prices_20) >= 20 else current_price
        
        # Momentum indicators
        momentum_5 = (current_price - prices_20[-5]) / prices_20[-5] if len(prices_20) >= 5 else 0
        momentum_10 = (current_price - prices_20[-10]) / prices_20[-10] if len(prices_20) >= 10 else 0
        
        # Volatility
        if len(prices_20) > 1:
            returns = (prices_20[1:] - prices_20[:-1]) / prices_20[:-1]
            volatility = torch.std(returns)
        else:
            volatility = torch.tensor(0.0, device=device)
        
        # Position and P&L features
        position_pnl = 0
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:  # Long
                position_pnl = (current_price - self.entry_price) / self.entry_price
            else:  # Short
                position_pnl = (self.entry_price - current_price) / self.entry_price
        
        # Normalize features
        price_norm = current_price / torch.mean(prices_20) if len(prices_20) > 0 else 1.0
        
        state = torch.tensor([
            price_norm.item(),
            (sma_5 / current_price).item(),
            (sma_10 / current_price).item(), 
            (sma_20 / current_price).item(),
            momentum_5.item(),
            momentum_10.item(),
            volatility.item(),
            float(self.position),
            position_pnl,
            self.cash / 25000.0,  # Normalized cash
            float(self.shares) / 100.0,  # Normalized shares
            float(self.trades) / 100.0,  # Normalized trade count
            float(self.wins) / max(1, self.trades),  # Win rate
            self.total_return,
            float(self.current_step) / self.max_steps  # Progress
        ], device=device)
        
        return state
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        if self.current_step >= self.max_steps:
            return self.get_state(), 0, True
        
        current_price = self.price_data[self.current_step].item()
        reward = 0
        
        # Action: 0=hold, 1=buy/long, 2=sell/cover, 3=short (if supported)
        if action == 1:  # Buy/Long
            if self.position == 0:  # Open long position
                self.shares = int(self.cash * 0.95 // current_price)
                if self.shares > 0:
                    self.cash -= self.shares * current_price
                    self.position = 1
                    self.entry_price = current_price
                    self.trades += 1
            elif self.position == -1:  # Cover short
                self.cash -= self.shares * current_price  # Buy to cover
                pnl = self.shares * (self.entry_price - current_price)
                self.cash += pnl
                if pnl > 0:
                    self.wins += 1
                    reward = pnl / 1000.0  # Normalize reward
                else:
                    reward = pnl / 1000.0
                self.position = 0
                self.shares = 0
        
        elif action == 2:  # Sell/Cover
            if self.position == 1:  # Close long position
                proceeds = self.shares * current_price
                pnl = proceeds - (self.shares * self.entry_price)
                self.cash += proceeds
                if pnl > 0:
                    self.wins += 1
                    reward = pnl / 1000.0  # Normalize reward
                else:
                    reward = pnl / 1000.0
                self.position = 0
                self.shares = 0
        
        elif action == 3 and self.trading_mode in ['SHORT_ONLY', 'LONG_SHORT']:  # Short
            if self.position == 0:  # Open short position
                self.shares = int(self.cash * 0.95 // current_price)
                if self.shares > 0:
                    self.cash += self.shares * current_price  # Receive cash from short
                    self.position = -1
                    self.entry_price = current_price
                    self.trades += 1
        
        # Calculate total return
        portfolio_value = self.cash
        if self.position == 1:  # Long position
            portfolio_value += self.shares * current_price
        elif self.position == -1:  # Short position
            portfolio_value += self.shares * (self.entry_price - current_price)
        
        self.total_return = (portfolio_value - 25000) / 25000
        
        # Additional rewards for good behavior
        if self.trades > 0:
            win_rate = self.wins / self.trades
            if win_rate > 0.6:
                reward += 0.1  # Bonus for high win rate
        
        if self.total_return > 0.05:  # Bonus for achieving 5%+ return
            reward += 0.2
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self.get_state(), reward, done

class SimpleHindsightEnvironment(TradingEnvironment):
    """Simple hindsight learning built on working base environment"""
    
    def __init__(self, price_data, trading_mode='LONG_ONLY'):
        super().__init__(price_data, trading_mode)
        self.opportunity_tracker = []
        self.hindsight_penalties = 0
        self.hindsight_bonuses = 0
        self.missed_opportunities = deque(maxlen=50)  # Limit memory usage
    
    def step(self, action):
        """Enhanced step with simple hindsight learning"""
        # Use the original working step function
        state, reward, done = super().step(action)
        
        # Add simple hindsight analysis every 50 steps (reduced frequency for performance)
        if self.current_step % 50 == 0 and self.current_step > 50:
            hindsight_adjustment = self.simple_hindsight_analysis()
            reward += hindsight_adjustment
            
            if hindsight_adjustment < 0:
                self.hindsight_penalties += 1
            elif hindsight_adjustment > 0:
                self.hindsight_bonuses += 1
        
        return state, reward, done
    
    def simple_hindsight_analysis(self):
        """Simple hindsight: check if we missed obvious opportunities"""
        try:
            if self.current_step + 15 >= len(self.price_data):
                return 0.0
            
            current_price = self.price_data[self.current_step]
            
            # Look ahead 10-15 steps to see what happens
            future_prices = self.price_data[self.current_step+5:self.current_step+15]
            if len(future_prices) == 0:
                return 0.0
            
            max_future = torch.max(future_prices)
            min_future = torch.min(future_prices)
            
            # Calculate potential opportunities
            long_opportunity = (max_future - current_price) / current_price
            short_opportunity = (current_price - min_future) / current_price
            
            hindsight_adjustment = 0.0
            
            # Simple hindsight rules
            if long_opportunity > 0.025:  # 2.5%+ upside available
                if self.position <= 0:  # We're not long when we should be
                    hindsight_adjustment -= 0.03  # Small penalty
                    self.missed_opportunities.append({
                        'step': self.current_step,
                        'type': 'missed_long',
                        'opportunity': long_opportunity.item()
                    })
                elif self.position == 1:  # We are long - good!
                    hindsight_adjustment += 0.02  # Small bonus
            
            if short_opportunity > 0.025 and self.trading_mode in ['SHORT_ONLY', 'LONG_SHORT']:  # 2.5%+ downside available
                if self.position >= 0:  # We're not short when we should be
                    hindsight_adjustment -= 0.03  # Small penalty
                    self.missed_opportunities.append({
                        'step': self.current_step,
                        'type': 'missed_short',
                        'opportunity': short_opportunity.item()
                    })
                elif self.position == -1:  # We are short - good!
                    hindsight_adjustment += 0.02  # Small bonus
            
            return hindsight_adjustment
            
        except Exception as e:
            return 0.0
    
    def get_hindsight_report(self):
        """Generate simple hindsight report"""
        if not self.missed_opportunities:
            return "No significant missed opportunities identified."
        
        missed_long = len([op for op in self.missed_opportunities if op['type'] == 'missed_long'])
        missed_short = len([op for op in self.missed_opportunities if op['type'] == 'missed_short'])
        
        avg_missed_long = np.mean([op['opportunity'] for op in self.missed_opportunities if op['type'] == 'missed_long']) if missed_long > 0 else 0
        avg_missed_short = np.mean([op['opportunity'] for op in self.missed_opportunities if op['type'] == 'missed_short']) if missed_short > 0 else 0
        
        report = f"\nHINDSIGHT ANALYSIS:\n"
        report += f"Missed long opportunities: {missed_long} (avg: {avg_missed_long:.3f})\n"
        report += f"Missed short opportunities: {missed_short} (avg: {avg_missed_short:.3f})\n"
        report += f"Hindsight penalties: {self.hindsight_penalties}\n"
        report += f"Hindsight bonuses: {self.hindsight_bonuses}\n"
        
        return report

class TradingDQN(nn.Module):
    """Simple DQN that works well (from original)"""
    
    def __init__(self, state_size=15, action_size=4, hidden_size=128):
        super(TradingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SimpleHindsightRLOptimizer:
    """Simple hindsight RL optimizer built on working base"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.optimization_results = {}
        self.results_file = "simple_hindsight_rl_results.json"
        self.device = device
        
        # Use proven hyperparameters from working agent
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_size = 10000
        self.batch_size = 32
        self.target_update = 100
        
        self.load_optimization_results()
    
    def create_rl_agent(self, action_size):
        """Create RL agent using proven architecture"""
        self.q_network = TradingDQN(action_size=action_size).to(self.device)
        self.target_network = TradingDQN(action_size=action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=self.memory_size)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = state.unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        try:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.stack([e[0] for e in batch])
            actions = torch.tensor([e[1] for e in batch], device=self.device)
            rewards = torch.tensor([e[2] for e in batch], device=self.device)
            next_states = torch.stack([e[3] for e in batch])
            dones = torch.tensor([e[4] for e in batch], device=self.device)
            
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        except Exception as e:
            clear_gpu_cache()
    
    def train_rl_agent(self, env, episodes=300):
        """Train RL agent with simple hindsight learning - minimal output"""
        print(f"Training simple hindsight RL agent for {episodes} episodes...")
        
        best_return = -999
        best_episode = 0
        returns_history = []
        
        for episode in range(episodes):
            try:
                state = env.reset()
                total_reward = 0
                steps = 0
                
                while True:
                    action = self.act(state)
                    next_state, reward, done = env.step(action)
                    
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                # Train the network
                if len(self.memory) > self.batch_size:
                    self.replay()
                
                # Update target network
                if episode % self.target_update == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
                # Track performance
                final_return = env.total_return
                returns_history.append(final_return)
                
                if final_return > best_return:
                    best_return = final_return
                    best_episode = episode
                    print(f"NEW BEST: Episode {episode}, Return: {final_return:.4f}, Trades: {env.trades}, Win Rate: {env.wins/max(1,env.trades):.2%}")
                
                # Progress update every 10 episodes
                if episode % 10 == 0:
                    avg_return = np.mean(returns_history[-10:]) if len(returns_history) >= 10 else np.mean(returns_history)
                    print(f"Episode {episode}, Avg Return (last 10): {avg_return:.4f}, Best: {best_return:.4f}, Epsilon: {self.epsilon:.3f}")
                
                # Show hindsight report every 50 episodes
                if episode % 50 == 0 and hasattr(env, 'get_hindsight_report'):
                    print(env.get_hindsight_report())
                
                # Memory management every 10 episodes
                if episode % 10 == 0:
                    clear_gpu_cache()
                    
            except Exception as e:
                clear_gpu_cache()
                continue
        
        print(f"\nTraining complete! Best return: {best_return:.4f} at episode {best_episode}")
        return best_return
    
    def optimize_symbol_with_simple_hindsight(self, symbol, trading_modes=['LONG_ONLY', 'SHORT_ONLY', 'LONG_SHORT']):
        """Optimize symbol using simple hindsight RL"""
        print(f"\nSIMPLE HINDSIGHT RL OPTIMIZATION FOR {symbol}")
        print("=" * 60)
        
        # Load data
        config = {'stock_symbol': symbol}
        data_manager = DataManager(self.redis_client, symbol, config)
        data_manager.discover_available_days()
        
        if len(data_manager.available_days) < 30:
            print(f"Insufficient data: {len(data_manager.available_days)} days")
            return None
        
        # Split data
        train_days = data_manager.available_days[:int(len(data_manager.available_days) * 0.7)]
        test_days = data_manager.available_days[int(len(data_manager.available_days) * 0.7):]
        
        print(f"Training days: {len(train_days)}")
        print(f"Testing days: {len(test_days)}")
        
        # Load training data
        training_data = self.load_symbol_data(data_manager, train_days)
        if training_data is None or len(training_data) < 100:
            print(f"Insufficient training data")
            return None
        
        symbol_results = {}
        
        # Train simple hindsight RL agent for each trading mode
        for mode in trading_modes:
            print(f"\nSimple Hindsight RL Training for {mode} strategy...")
            
            # Set action size based on trading mode
            if mode == 'LONG_ONLY':
                self.action_size = 3  # hold, buy, sell
            elif mode == 'SHORT_ONLY':
                self.action_size = 3  # hold, short, cover
            else:  # LONG_SHORT
                self.action_size = 4  # hold, buy, sell, short
            
            try:
                # Create simple hindsight environment and agent
                env = SimpleHindsightEnvironment(training_data, mode)
                self.create_rl_agent(self.action_size)
                
                # Train agent
                best_return = self.train_rl_agent(env, episodes=300)
                
                # Test on validation data
                test_data = self.load_symbol_data(data_manager, test_days)
                if test_data is not None:
                    test_env = SimpleHindsightEnvironment(test_data, mode)
                    test_return = self.test_rl_agent(test_env)
                else:
                    test_return = -999
                
                symbol_results[mode] = {
                    'train_return': best_return,
                    'test_return': test_return,
                    'optimized_date': datetime.now().isoformat(),
                    'method': 'simple_hindsight_reinforcement_learning'
                }
                
                print(f"{mode} - Train Return: {best_return:.4f}, Test Return: {test_return:.4f}")
                
                # Clear memory between modes
                clear_gpu_cache()
                
            except Exception as e:
                print(f"Failed to optimize {mode}: {e}")
                symbol_results[mode] = {
                    'train_return': -999,
                    'test_return': -999,
                    'error': str(e)
                }
                clear_gpu_cache()
        
        # Find best strategy
        valid_results = {k: v for k, v in symbol_results.items() if v.get('test_return', -999) > -999}
        if valid_results:
            best_strategy = max(valid_results.keys(), key=lambda k: valid_results[k]['test_return'])
        else:
            best_strategy = list(symbol_results.keys())[0]
        
        self.optimization_results[symbol] = {
            'best_strategy': best_strategy,
            'strategies': symbol_results,
            'optimization_date': datetime.now().isoformat(),
            'method': 'simple_hindsight_reinforcement_learning'
        }
        
        self.save_optimization_results()
        
        print(f"\nBEST SIMPLE HINDSIGHT RL STRATEGY: {best_strategy}")
        print(f"Test Return: {symbol_results[best_strategy].get('test_return', 'N/A')}")
        
        return self.optimization_results[symbol]
    
    def test_rl_agent(self, env):
        """Test trained RL agent"""
        self.epsilon = 0  # No exploration during testing
        state = env.reset()
        
        with torch.no_grad():
            while True:
                action = self.act(state)
                state, _, done = env.step(action)
                if done:
                    break
        
        print(f"Test completed: Return {env.total_return:.4f}, Trades: {env.trades}, Win Rate: {env.wins/max(1,env.trades):.2%}")
        if hasattr(env, 'get_hindsight_report'):
            print(env.get_hindsight_report())
        
        return env.total_return
    
    def load_symbol_data(self, data_manager, days):
        """Load symbol data"""
        all_data = []
        
        for day in days:
            day_data = data_manager.load_day_data(day)
            if day_data:
                for entry in day_data:
                    if 'timestamp' not in entry or 'price' not in entry:
                        continue
                    
                    try:
                        timestamp = entry['timestamp']
                        if not isinstance(timestamp, datetime):
                            continue
                        
                        all_data.append({
                            'timestamp': timestamp,
                            'Open': float(entry['price']),
                            'High': float(entry['ask']),
                            'Low': float(entry['bid']),
                            'Close': float(entry['price']),
                            'Volume': int(entry['volume'])
                        })
                    except (ValueError, TypeError):
                        continue
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data)
        df.set_index('timestamp', inplace=True)
        df = df.dropna().drop_duplicates().sort_index()
        
        return df
    
    def load_optimization_results(self):
        """Load optimization results"""
        try:
            if Path(self.results_file).exists():
                with open(self.results_file, 'r') as f:
                    self.optimization_results = json.load(f)
                print(f"Loaded simple hindsight RL results for {len(self.optimization_results)} symbols")
        except Exception as e:
            self.optimization_results = {}
    
    def save_optimization_results(self):
        """Save optimization results"""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.optimization_results, f, indent=2)
            print(f"Saved simple hindsight RL optimization results")
        except Exception as e:
            print(f"Failed to save results: {e}")

def main():
    """Main simple hindsight RL optimization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Hindsight RL Strategy Optimizer')
    parser.add_argument('--symbol', help='Symbol to optimize')
    parser.add_argument('--episodes', type=int, default=300, help='Training episodes')
    
    args = parser.parse_args()
    
    # Redis connection
    try:
        with open('.redis_passwd', 'r') as f:
            password = f.read().strip()
        
        redis_client = redis.Redis(
            host='trader.wolfx0.com',
            port=6379,
            password=password,
            decode_responses=True
        )
        redis_client.ping()
        print("Redis connection successful")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return
    
    optimizer = SimpleHindsightRLOptimizer(redis_client)
    
    if args.symbol:
        optimizer.optimize_symbol_with_simple_hindsight(args.symbol)
    else:
        print("Usage: python optimization/strategy_optimizer.py --symbol S_AAPL --episodes 300")

if __name__ == "__main__":
    main()

