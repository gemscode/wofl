import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from data_manager import FixedDataManager
from pattern_explorer import PatternExplorer

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ Using Mac M-series GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ Using NVIDIA GPU (CUDA)")
        return torch.device("cuda")
    else:
        print("⚠️  Using CPU only")
        return torch.device("cpu")

class PatternAwareTradingEnv:
    def __init__(self, data, explorer, lookback=30, initial_cash=25000, reward_config=None):
        self.data = data.reset_index(drop=True)
        self.explorer = explorer
        self.lookback = lookback
        self.initial_cash = initial_cash
        self.reward_config = reward_config or {}
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.holdings = 0
        self.position = 0
        self.current_step = self.lookback
        self.total_trades = 0
        self.wins = 0
        self.portfolio_history = []
        self.returns_history = []
        self.drawdown_history = []
        self.peak_value = self.initial_cash
        return self.get_state()

    def get_state(self):
        window = self.data.iloc[self.current_step-self.lookback:self.current_step]
        price = window['close'].iloc[-1]
        sma_5 = window['sma_5'].iloc[-1]
        sma_10 = window['sma_10'].iloc[-1]
        sma_20 = window['sma_20'].iloc[-1]
        rsi_14 = window['rsi_14'].iloc[-1]
        base_state = np.array([
            price/100, sma_5/100, sma_10/100, sma_20/100, rsi_14/100,
            self.position, self.cash/10000, self.holdings/1000
        ], dtype=np.float32)
        guidance = self.explorer.generate_guidance(window)
        return np.concatenate([base_state, guidance]).astype(np.float32)

    def step(self, action):
        window = self.data.iloc[self.current_step-self.lookback:self.current_step]
        price = window['close'].iloc[-1]
        old_portfolio_value = self.cash + self.holdings * price
        
        # Execute trade
        trade_executed = False
        if action == 1 and self.position == 0:  # BUY
            shares = int(self.cash // price)
            if shares > 0:
                self.cash -= shares * price
                self.holdings += shares
                self.position = 1
                self.total_trades += 1
                trade_executed = True
        elif action == 2 and self.position == 1:  # SELL
            self.cash += self.holdings * price
            # Check if this was a winning trade
            if self.holdings * price > self.holdings * getattr(self, 'entry_price', price):
                self.wins += 1
            self.holdings = 0
            self.position = 0
            self.total_trades += 1
            trade_executed = True
        
        # Calculate portfolio value and metrics
        portfolio_value = self.cash + self.holdings * price
        self.portfolio_history.append(portfolio_value)
        
        # Calculate return
        if len(self.portfolio_history) > 1:
            period_return = (portfolio_value - old_portfolio_value) / old_portfolio_value
            self.returns_history.append(period_return)
        
        # Track peak and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.drawdown_history.append(drawdown)
        
        # Calculate reward based on configuration
        reward = self._calculate_reward(old_portfolio_value, portfolio_value, trade_executed, drawdown)
        
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        return self.get_state(), reward, done

    def _calculate_reward(self, old_value, new_value, trade_executed, drawdown):
        """Calculate reward based on reward configuration"""
        reward_type = self.reward_config.get('reward_type', 'default')
        
        if reward_type == 'sharpe_ratio':
            return self._sharpe_reward(old_value, new_value, trade_executed)
        elif reward_type == 'drawdown_penalty':
            return self._drawdown_penalty_reward(old_value, new_value, drawdown)
        elif reward_type == 'consistency':
            return self._consistency_reward(old_value, new_value)
        elif reward_type == 'risk_adjusted':
            return self._risk_adjusted_reward(old_value, new_value, drawdown)
        elif reward_type == 'momentum':
            return self._momentum_reward(old_value, new_value)
        elif reward_type == 'profit_factor':
            return self._profit_factor_reward(old_value, new_value, trade_executed)
        elif reward_type == 'calmar_ratio':
            return self._calmar_ratio_reward(old_value, new_value, drawdown)
        elif reward_type == 'sortino_ratio':
            return self._sortino_ratio_reward(old_value, new_value)
        elif reward_type == 'kelly_criterion':
            return self._kelly_criterion_reward(old_value, new_value, trade_executed)
        elif reward_type == 'tail_risk':
            return self._tail_risk_reward(old_value, new_value, drawdown)
        else:
            return self._default_reward(old_value, new_value, trade_executed)

    def _default_reward(self, old_value, new_value, trade_executed):
        """Default reward function"""
        reward = 0
        if trade_executed:
            reward = 0.01
        if len(self.portfolio_history) > 1:
            reward += (new_value - old_value) / 100
        return reward

    def _sharpe_reward(self, old_value, new_value, trade_executed):
        """Sharpe ratio optimized reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        risk_free_rate = self.reward_config.get('risk_free_rate', 0.02) / 252  # Daily risk-free rate
        
        if len(self.returns_history) > 10:
            excess_returns = np.array(self.returns_history) - risk_free_rate
            sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
            return sharpe * 0.1
        else:
            return (period_return - risk_free_rate) * 10

    def _drawdown_penalty_reward(self, old_value, new_value, drawdown):
        """Heavy drawdown penalty reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        max_dd_threshold = self.reward_config.get('max_drawdown_threshold', 0.1)
        penalty_multiplier = self.reward_config.get('drawdown_penalty_multiplier', 2.0)
        
        reward = period_return * 10
        if drawdown > max_dd_threshold:
            reward -= (drawdown - max_dd_threshold) * penalty_multiplier * 10
        
        return reward

    def _consistency_reward(self, old_value, new_value):
        """Consistency focused reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        volatility_penalty = self.reward_config.get('volatility_penalty', 0.2)
        
        if len(self.returns_history) > 5:
            volatility = np.std(self.returns_history[-5:])
            consistency_bonus = 1 / (1 + volatility * volatility_penalty)
            return period_return * 10 * consistency_bonus
        else:
            return period_return * 10

    def _risk_adjusted_reward(self, old_value, new_value, drawdown):
        """Comprehensive risk-adjusted reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        risk_free_rate = self.reward_config.get('risk_free_rate', 0.02) / 252
        var_penalty = self.reward_config.get('var_penalty', 0.15)
        
        excess_return = period_return - risk_free_rate
        risk_penalty = drawdown * var_penalty
        
        return (excess_return - risk_penalty) * 10

    def _momentum_reward(self, old_value, new_value):
        """Momentum enhanced reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        momentum_threshold = self.reward_config.get('momentum_threshold', 0.02)
        trend_bonus = self.reward_config.get('trend_following_bonus', 0.1)
        
        if len(self.returns_history) > 3:
            recent_trend = np.mean(self.returns_history[-3:])
            if abs(recent_trend) > momentum_threshold:
                if np.sign(period_return) == np.sign(recent_trend):
                    return period_return * 10 * (1 + trend_bonus)
        
        return period_return * 10

    def _profit_factor_reward(self, old_value, new_value, trade_executed):
        """Profit factor optimized reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        
        if trade_executed and len(self.returns_history) > 10:
            positive_returns = [r for r in self.returns_history if r > 0]
            negative_returns = [r for r in self.returns_history if r < 0]
            
            if positive_returns and negative_returns:
                profit_factor = sum(positive_returns) / abs(sum(negative_returns))
                min_pf = self.reward_config.get('min_profit_factor', 1.5)
                if profit_factor > min_pf:
                    return period_return * 10 * (1 + (profit_factor - min_pf) * 0.1)
        
        return period_return * 10

    def _calmar_ratio_reward(self, old_value, new_value, drawdown):
        """Calmar ratio optimized reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        
        if len(self.portfolio_history) > 50:
            total_return = (new_value - self.initial_cash) / self.initial_cash
            max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0.01
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            return calmar_ratio * 0.1
        else:
            return period_return * 10

    def _sortino_ratio_reward(self, old_value, new_value):
        """Sortino ratio optimized reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        target_return = self.reward_config.get('target_return', 0.0)
        
        if len(self.returns_history) > 10:
            downside_returns = [min(0, r - target_return) for r in self.returns_history]
            downside_deviation = np.sqrt(np.mean([r**2 for r in downside_returns]))
            
            if downside_deviation > 0:
                excess_return = np.mean(self.returns_history) - target_return
                sortino_ratio = excess_return / downside_deviation
                return sortino_ratio * 0.1
        
        return period_return * 10

    def _kelly_criterion_reward(self, old_value, new_value, trade_executed):
        """Kelly criterion based reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        
        if self.total_trades > 10:
            win_rate = self.wins / self.total_trades
            avg_win = np.mean([r for r in self.returns_history if r > 0]) if any(r > 0 for r in self.returns_history) else 0
            avg_loss = abs(np.mean([r for r in self.returns_history if r < 0])) if any(r < 0 for r in self.returns_history) else 0.01
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                optimal_position = max(0, min(1, kelly_fraction))
                return period_return * 10 * optimal_position
        
        return period_return * 10

    def _tail_risk_reward(self, old_value, new_value, drawdown):
        """Tail risk adjusted reward"""
        period_return = (new_value - old_value) / old_value if old_value > 0 else 0
        extreme_loss_threshold = self.reward_config.get('extreme_loss_threshold', 0.05)
        cvar_penalty = self.reward_config.get('cvar_penalty', 2.0)
        
        if drawdown > extreme_loss_threshold:
            return period_return * 10 - (drawdown - extreme_loss_threshold) * cvar_penalty * 10
        
        return period_return * 10

    def get_portfolio_return(self):
        if not self.portfolio_history:
            return 0
        return (self.portfolio_history[-1] - self.initial_cash) / self.initial_cash

    def get_sharpe_ratio(self):
        if len(self.returns_history) < 2:
            return 0
        return np.mean(self.returns_history) / (np.std(self.returns_history) + 1e-8)

    def get_max_drawdown(self):
        return max(self.drawdown_history) if self.drawdown_history else 0

    def get_win_rate(self):
        return self.wins / max(1, self.total_trades)

class DQNAgent:
    def __init__(self, state_size, action_size, device, lr=1e-3, epsilon_decay=0.995, 
                 batch_size=128, epsilon_min=0.05, memory_size=20000, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, gamma=None):
        if gamma is None:
            gamma = self.gamma
            
        if len(self.memory) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        # Efficient conversion to tensors
        states = torch.from_numpy(np.array([self.memory[i][0] for i in batch])).float().to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.from_numpy(np.array([self.memory[i][3] for i in batch])).float().to(self.device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.model(next_states).max(1)[0].detach()
        targets = rewards + gamma * next_q_values * (~dones)
        
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--source', type=str, default='alpha')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=1000)
    
    # Hyperparameter arguments for pipeline integration
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epsilon_min', type=float, default=0.05)
    parser.add_argument('--memory_size', type=int, default=20000)
    parser.add_argument('--gamma', type=float, default=0.95)
    
    # Reward function arguments
    parser.add_argument('--reward_type', type=str, default='default')
    parser.add_argument('--risk_free_rate', type=float, default=0.02)
    parser.add_argument('--volatility_penalty', type=float, default=0.2)
    parser.add_argument('--max_drawdown_threshold', type=float, default=0.1)
    parser.add_argument('--drawdown_penalty_multiplier', type=float, default=2.0)
    
    args = parser.parse_args()

    device = get_device()

    symbol = f"S_{args.symbol.upper()}_{args.source.upper()}"
    print(f"Loading data for symbol: {symbol}")
    dm = FixedDataManager(symbol)
    days = dm.available_days
    if not days:
        print("No data available for this symbol/source!")
        return
    split = int(len(days) * 0.7)
    train_days = days[:split]
    test_days = days[split:]
    print("Loading training data...")
    train_data = dm.load_multiple_days(train_days)
    print("Loading test data...")
    test_data = dm.load_multiple_days(test_days)
    if train_data.empty or test_data.empty:
        print("No data loaded for training or testing!")
        return

    # --- Exploration Phase ---
    print("Starting exploration phase...")
    explorer = PatternExplorer(train_data)
    train_data = explorer.preprocess_data()
    explorer.discover_patterns()
    explorer.analyze_patterns()

    # --- RL Training ---
    print("Starting RL training...")
    
    # Build reward configuration from arguments
    reward_config = {
        'reward_type': args.reward_type,
        'risk_free_rate': args.risk_free_rate,
        'volatility_penalty': args.volatility_penalty,
        'max_drawdown_threshold': args.max_drawdown_threshold,
        'drawdown_penalty_multiplier': args.drawdown_penalty_multiplier
    }
    
    env = PatternAwareTradingEnv(train_data, explorer, reward_config=reward_config)
    state_size = len(env.get_state())
    action_size = 3
    
    # Create agent with pipeline parameters
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size, 
        device=device,
        lr=args.learning_rate,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        epsilon_min=args.epsilon_min,
        memory_size=args.memory_size,
        gamma=args.gamma
    )
    
    episodes = args.episodes
    max_steps_per_episode = args.max_steps

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            if step_count % 200 == 0:
                print(f"Episode {ep+1}, Step {step_count}")
            if done or step_count >= max_steps_per_episode:
                break
        agent.replay()
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}, Portfolio Return: {env.get_portfolio_return():.2%}, Total Reward: {total_reward:.2f}")

    # --- Test ---
    print("Evaluating on test set...")
    test_data = explorer.preprocess_data()
    test_env = PatternAwareTradingEnv(test_data, explorer, reward_config=reward_config)
    
    # Set agent to no exploration for testing
    agent.epsilon = 0
    
    state = test_env.reset()
    step_count = 0
    while True:
        action = agent.act(state)
        state, reward, done = test_env.step(action)
        step_count += 1
        if done or step_count >= max_steps_per_episode:
            break
    
    # Print comprehensive results for pipeline parsing
    print(f"Test Portfolio Return: {test_env.get_portfolio_return():.2%}")
    print(f"Test Sharpe Ratio: {test_env.get_sharpe_ratio():.4f}")
    print(f"Test Max Drawdown: {test_env.get_max_drawdown():.4f}")
    print(f"Test Win Rate: {test_env.get_win_rate():.2%}")
    print(f"Total Trades: {test_env.total_trades}")

if __name__ == "__main__":
    main()

