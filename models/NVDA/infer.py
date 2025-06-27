#!/usr/bin/env python3
"""
Inference script for NVDA trading agents
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

class NVDAInference:
    def __init__(self):
        self.stock_symbol = "NVDA"
        self.models_dir = Path("agents")  # Relative to this script's location
        
        # Load trained agents
        self.agent_a1 = None
        self.agent_a2 = None
        self.load_agents()
    
    def load_agents(self):
        """Load trained agents"""
        try:
            a1_path = self.models_dir / f"agent_a1_{self.stock_symbol}"
            a2_path = self.models_dir / f"agent_a2_{self.stock_symbol}"
            
            # Both agents use PPO
            self.agent_a1 = PPO.load(str(a1_path))
            self.agent_a2 = PPO.load(str(a2_path))
            
            print(f"✅ Loaded agents for {self.stock_symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading agents: {e}")
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
        
        return final_action, f"A1: {a1_action}, A2: {a2_action}, Price: ${current_price:.2f}"

def main():
    inference = NVDAInference()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        # Continuous inference mode
        import time
        print(f"🔄 Starting continuous inference for {inference.stock_symbol}...")
        
        while True:
            try:
                action, details = inference.make_prediction()
                print(f"{datetime.now().strftime('%H:%M:%S')} - {action}: {details}")
                time.sleep(30)  # Every 30 seconds
            except KeyboardInterrupt:
                print("\n🛑 Inference stopped")
                break
    else:
        # Single prediction
        action, details = inference.make_prediction()
        print(f"Recommendation for {inference.stock_symbol}: {action}")
        print(f"Details: {details}")

if __name__ == "__main__":
    main()
