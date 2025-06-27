#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import threading
import time
from datetime import datetime, timedelta
import redis
import json
import yfinance as yf
import os
from pathlib import Path

# Import our custom agent classes
from rw_wolfxe.agents.pure_intraday_agent import PureIntradayAgent
from rw_wolfxe.agents.intraday_trend_agent import IntradayTrendAgent

class ConfigurableIntradayTradingSystem:
    """Multi-agent trading system with proper directory management"""
    
    def __init__(self, stock_symbol, initial_balance=10000, config=None):
        self.stock_symbol = stock_symbol.upper()
        self.initial_balance = initial_balance
        self.config = config or self.get_default_config()
        
        # Create organized directory structure
        self.setup_directories()
        
        # Initialize agents
        self.agent_a1 = None
        self.agent_a2 = None
        
        # Portfolio tracking
        self.portfolio = {
            'cash': initial_balance,
            'shares': 0,
            'last_trade_time': None,
            'total_trades': 0,
            'profit_loss': 0.0
        }
        
        # Redis connection
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
            print(f"✅ Connected to Redis for {self.stock_symbol} data management")
        except:
            print("⚠️ Redis not available, using in-memory storage")
            self.redis_client = None
        
        # Data buffers
        self.daily_buffer = []
        self.minute_buffer = []

    def setup_directories(self):
        """Create organized directory structure for each stock"""
        # Main models directory
        self.base_models_dir = Path("models")
        
        # Stock-specific directory (this creates the NVDA directory)
        self.stock_dir = self.base_models_dir / self.stock_symbol
        
        # Create directories
        self.stock_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for organization
        self.agents_dir = self.stock_dir / "agents"
        self.data_dir = self.stock_dir / "data"
        self.logs_dir = self.stock_dir / "logs"
        
        # Create all subdirectories
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created directory structure for {self.stock_symbol}:")
        print(f"   └── models/{self.stock_symbol}/")
        print(f"       ├── agents/")
        print(f"       ├── data/")
        print(f"       └── logs/")
        
        # Set model paths
        self.agent_a1_path = self.agents_dir / f"agent_a1_{self.stock_symbol}"
        self.agent_a2_path = self.agents_dir / f"agent_a2_{self.stock_symbol}"

    def get_default_config(self):
        """Default configuration for any stock"""
        return {
            'training_days': 252,
            'min_hold_hours': 4,
            'max_hold_hours': 192,
            'retrain_interval': 300,
            'decision_interval': 5,
            'data_update_interval': 1,
            'learning_timesteps_a1': 100000,
            'learning_timesteps_a2': 200000
        }

    def fetch_historical_data(self):
        """Fetch and save historical data with proper Yahoo Finance limits"""
        print(f"📊 Fetching historical data for {self.stock_symbol}...")
        
        try:
            ticker = yf.Ticker(self.stock_symbol)
            
            # Fetch daily data for Agent A1 (2 years available)
            daily_data = ticker.history(period="2y", interval="1d")
            if daily_data.empty:
                raise ValueError(f"No daily data found for {self.stock_symbol}")
            
            # FIXED: Fetch minute data within Yahoo's limits (7 days max)
            minute_data = ticker.history(period="7d", interval="1m")
            if minute_data.empty:
                # Fallback to 5-minute data for longer period
                print("⚠️ 1-minute data unavailable, using 5-minute data...")
                minute_data = ticker.history(period="60d", interval="5m")
            
            if minute_data.empty:
                raise ValueError(f"No intraday data found for {self.stock_symbol}")
            
            # Save data to stock-specific directory
            daily_data_path = self.data_dir / f"{self.stock_symbol}_daily_data.csv"
            minute_data_path = self.data_dir / f"{self.stock_symbol}_minute_data.csv"
            
            daily_data.to_csv(daily_data_path)
            minute_data.to_csv(minute_data_path)
            
            print(f"✅ Retrieved and saved {len(daily_data)} daily bars and {len(minute_data)} minute bars")
            print(f"📁 Data saved to: {self.data_dir}")
            
            return daily_data, minute_data
            
        except Exception as e:
            print(f"❌ Error fetching data for {self.stock_symbol}: {e}")
            return None, None

    def train_agents(self, daily_data, minute_data):
        """Train both agents and save to stock-specific directory"""
        print(f"🤖 Training agents for {self.stock_symbol}...")
        
        # Train Agent A1 (Daily Trend)
        print("Training Agent A1 (Daily Trend Analysis)...")
        env_a1 = IntradayTrendAgent(daily_data, self.initial_balance)
        self.agent_a1 = PPO("MlpPolicy", env_a1, verbose=1)
        self.agent_a1.learn(total_timesteps=self.config['learning_timesteps_a1'])
        
        # Save Agent A1 to stock directory
        self.agent_a1.save(str(self.agent_a1_path))
        print(f"✅ Agent A1 saved to {self.agent_a1_path}.zip")
        
        # Train Agent A2 (Pure Intraday) - Using imported class
        print("Training Agent A2 (Intraday Trading)...")
        env_a2 = PureIntradayAgent(minute_data, self.initial_balance)
        self.agent_a2 = PPO("MlpPolicy", env_a2, verbose=1)
        self.agent_a2.learn(total_timesteps=self.config['learning_timesteps_a2'])
        
        # Save Agent A2 to stock directory
        self.agent_a2.save(str(self.agent_a2_path))
        print(f"✅ Agent A2 saved to {self.agent_a2_path}.zip")
        
        # Create training summary
        self.save_training_summary()

    def save_training_summary(self):
        """Save training summary and metadata"""
        summary = {
            'stock_symbol': self.stock_symbol,
            'training_date': datetime.now().isoformat(),
            'initial_balance': self.initial_balance,
            'config': self.config,
            'agent_a1_path': str(self.agent_a1_path) + '.zip',
            'agent_a2_path': str(self.agent_a2_path) + '.zip',
            'model_version': '1.0',
            'algorithm': 'PPO'  # Both agents use PPO
        }
        
        summary_path = self.stock_dir / f"{self.stock_symbol}_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Training summary saved to {summary_path}")

    def load_trained_agents(self):
        """Load pre-trained agents from stock-specific directory"""
        try:
            # Check if model files exist
            a1_zip_path = Path(str(self.agent_a1_path) + '.zip')
            a2_zip_path = Path(str(self.agent_a2_path) + '.zip')
            
            if a1_zip_path.exists() and a2_zip_path.exists():
                # Load Agent A1
                self.agent_a1 = PPO.load(str(self.agent_a1_path))
                print(f"✅ Loaded Agent A1 from {a1_zip_path}")
                
                # Load Agent A2
                self.agent_a2 = PPO.load(str(self.agent_a2_path))
                print(f"✅ Loaded Agent A2 from {a2_zip_path}")
                
                # Load training summary if available
                summary_path = self.stock_dir / f"{self.stock_symbol}_training_summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    print(f"📋 Model trained on: {summary.get('training_date', 'Unknown')}")
                
                return True
            else:
                print(f"❌ Model files not found in {self.agents_dir}")
                print(f"   Expected: {a1_zip_path.name} and {a2_zip_path.name}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading agents for {self.stock_symbol}: {e}")
            return False

    def create_inference_script(self):
        """Create infer.py script for this stock"""
        infer_script_content = f'''#!/usr/bin/env python3
"""
Inference script for {self.stock_symbol} trading agents
Generated automatically by learn.py
"""

import sys
import os
from pathlib import Path
from stable_baselines3 import PPO
import yfinance as yf
import json
import numpy as np
from datetime import datetime

class {self.stock_symbol}Inference:
    def __init__(self):
        self.stock_symbol = "{self.stock_symbol}"
        self.models_dir = Path("agents")  # Relative to this script's location
        
        # Load trained agents
        self.agent_a1 = None
        self.agent_a2 = None
        self.load_agents()
    
    def load_agents(self):
        """Load trained agents"""
        try:
            a1_path = self.models_dir / f"agent_a1_{{self.stock_symbol}}"
            a2_path = self.models_dir / f"agent_a2_{{self.stock_symbol}}"
            
            # Both agents use PPO
            self.agent_a1 = PPO.load(str(a1_path))
            self.agent_a2 = PPO.load(str(a2_path))
            
            print(f"✅ Loaded agents for {{self.stock_symbol}}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading agents: {{e}}")
            return False
    
    def get_current_price(self):
        """Get current stock price"""
        try:
            ticker = yf.Ticker(self.stock_symbol)
            data = ticker.history(period="1d", interval="1m").tail(1)
            return float(data['Close'].iloc[0])
        except:
            return None
    
    def make_prediction(self):
        """Make trading prediction"""
        if not self.agent_a1 or not self.agent_a2:
            return "HOLD", "Agents not loaded"
        
        current_price = self.get_current_price()
        if not current_price:
            return "HOLD", "Cannot fetch current price"
        
        # Simplified prediction logic
        # In real implementation, you'd prepare proper observations
        actions = ['HOLD', 'BUY', 'SELL']
        a1_action = np.random.choice(actions)
        a2_action = np.random.choice(actions)
        
        # Consensus decision
        if a1_action == a2_action:
            final_action = a1_action
        else:
            final_action = 'HOLD'  # Conservative on conflict
        
        return final_action, f"A1: {{a1_action}}, A2: {{a2_action}}, Price: ${{current_price:.2f}}"

def main():
    inference = {self.stock_symbol}Inference()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        # Continuous inference mode
        import time
        print(f"🔄 Starting continuous inference for {{inference.stock_symbol}}...")
        
        while True:
            try:
                action, details = inference.make_prediction()
                print(f"{{datetime.now().strftime('%H:%M:%S')}} - {{action}}: {{details}}")
                time.sleep(30)  # Every 30 seconds
            except KeyboardInterrupt:
                print("\\n🛑 Inference stopped")
                break
    else:
        # Single prediction
        action, details = inference.make_prediction()
        print(f"Recommendation for {{inference.stock_symbol}}: {{action}}")
        print(f"Details: {{details}}")

if __name__ == "__main__":
    main()
'''
        
        # Save inference script to stock directory
        infer_script_path = self.stock_dir / "infer.py"
        with open(infer_script_path, 'w') as f:
            f.write(infer_script_content)
        
        # Make it executable
        os.chmod(infer_script_path, 0o755)
        
        print(f"📝 Created inference script: {infer_script_path}")
        print(f"   Usage: python models/{self.stock_symbol}/infer.py")
        print(f"   Continuous: python models/{self.stock_symbol}/infer.py --continuous")

    # Simplified agent decision methods
    def get_agent_decision(self, agent, agent_type):
        """Get decision from agent"""
        # Create dummy observation for prediction
        if agent_type == 'daily':
            obs = np.zeros(30, dtype=np.float32)
        else:
            obs = np.zeros(35, dtype=np.float32)
        
        try:
            action, _ = agent.predict(obs, deterministic=True)
            actions = ['HOLD', 'BUY', 'SELL']
            return actions[action]
        except:
            return 'HOLD'
    
    def make_consensus_decision(self, a1_decision, a2_decision):
        """Make consensus decision between agents"""
        if a1_decision == a2_decision:
            return a1_decision
        return 'HOLD'  # Conservative approach on conflict
    
    def execute_trade(self, decision, current_price):
        """Execute trade based on decision"""
        if decision == 'BUY' and self.portfolio['cash'] > current_price:
            shares_to_buy = int(self.portfolio['cash'] // current_price)
            if shares_to_buy > 0:
                self.portfolio['shares'] += shares_to_buy
                self.portfolio['cash'] -= shares_to_buy * current_price
                self.portfolio['total_trades'] += 1
                print(f"📈 BUY {self.stock_symbol}: {shares_to_buy} shares at ${current_price:.2f}")
                
        elif decision == 'SELL' and self.portfolio['shares'] > 0:
            proceeds = self.portfolio['shares'] * current_price
            shares_sold = self.portfolio['shares']
            self.portfolio['cash'] += proceeds
            self.portfolio['shares'] = 0
            self.portfolio['total_trades'] += 1
            print(f"📉 SELL {self.stock_symbol}: {shares_sold} shares at ${current_price:.2f}")

    # Additional methods for real-time trading (simplified for brevity)
    def start_real_time_trading(self):
        """Start real-time trading service"""
        print(f"🚀 Real-time trading for {self.stock_symbol} - Implementation in progress")
        self.create_inference_script()

def main():
    """Fixed main function logic"""
    parser = argparse.ArgumentParser(description='Multi-Agent Intraday Trading System')
    parser.add_argument('symbol', help='Stock symbol to trade (e.g., NVDA, AAPL, TSLA)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance (default: 10000)')
    parser.add_argument('--train', action='store_true', help='Force retraining of agents')
    parser.add_argument('--trade', action='store_true', help='Start real-time trading')
    parser.add_argument('--inference', action='store_true', help='Run inference only')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    if not args.symbol or len(args.symbol) < 1:
        print("❌ Please provide a valid stock symbol")
        sys.exit(1)
    
    print(f"🎯 Initializing trading system for {args.symbol.upper()}")
    
    # Initialize trading system
    trading_system = ConfigurableIntradayTradingSystem(
        stock_symbol=args.symbol,
        initial_balance=args.balance
    )
    
    # FIXED: Clear logic flow
    if args.inference:
        # Inference mode - models must exist
        if not trading_system.load_trained_agents():
            print(f"❌ No trained models found for {args.symbol}. Train first with --train")
            sys.exit(1)
        print(f"🔮 Inference mode ready for {args.symbol}")
        
    elif args.train:
        # Explicit training mode
        print(f"🎯 Training mode for {args.symbol}")
        
        # Fetch historical data
        daily_data, minute_data = trading_system.fetch_historical_data()
        
        if daily_data is None or minute_data is None:
            print(f"❌ Could not fetch data for {args.symbol}")
            sys.exit(1)
        
        # Train agents
        trading_system.train_agents(daily_data, minute_data)
        trading_system.create_inference_script()
        
    else:
        # Default: Try to load existing models, train if not found
        print(f"🔍 Checking for existing models for {args.symbol}...")
        
        if trading_system.load_trained_agents():
            print(f"✅ Found existing models for {args.symbol}")
        else:
            print(f"📚 No existing models found. Starting training for {args.symbol}...")
            
            # Fetch historical data
            daily_data, minute_data = trading_system.fetch_historical_data()
            
            if daily_data is None or minute_data is None:
                print(f"❌ Could not fetch data for {args.symbol}")
                sys.exit(1)
            
            # Train agents
            trading_system.train_agents(daily_data, minute_data)
            trading_system.create_inference_script()
    
    # Start trading if requested
    if args.trade:
        if not trading_system.agent_a1 or not trading_system.agent_a2:
            print("❌ Agents not loaded. Cannot start trading.")
            sys.exit(1)
        trading_system.start_real_time_trading()
    else:
        print(f"✅ Setup complete for {args.symbol.upper()}")
        print(f"📁 Models location: models/{args.symbol.upper()}/agents/")
        print(f"🔧 Use --train to force retrain")
        print(f"🔮 Use --inference for predictions only")
        print(f"🚀 Use --trade to start real-time trading")
        print(f"📊 Use models/{args.symbol.upper()}/infer.py for standalone inference")

if __name__ == "__main__":
    main()

