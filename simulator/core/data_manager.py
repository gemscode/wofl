#!/usr/bin/env python3
"""
Data Manager - EXACT Original Implementation
"""

import pandas as pd
from datetime import datetime
from collections import deque
from pathlib import Path
import json

class DataManager:
    """EXACT copy of data management from original"""
    
    def __init__(self, redis_client, stock_symbol, config):
        self.redis_client = redis_client
        self.stock_symbol = stock_symbol
        self.base_symbol = stock_symbol[2:] if stock_symbol.startswith('S_') else stock_symbol
        self.config = config
        
        # EXACT same data structures as original
        self.available_days = []
        self.price_history = deque(maxlen=500)
        
        print("✅ DataManager initialized with EXACT original functionality")
    
    def discover_available_days(self):
        """EXACT copy from original"""
        try:
            pattern = f"sim:{self.stock_symbol}:*"
            keys = self.redis_client.keys(pattern)
            
            days = []
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 3:
                    days.append(parts[2])
            
            self.available_days = sorted(days)
            return self.available_days
            
        except Exception as e:
            print(f"[ERROR] Failed to discover days: {e}")
            return []
    
    def check_existing_training(self):
        """EXACT copy from original"""
        training_file = f"training_metadata_{self.base_symbol}.json"
        models_dir = Path(f"models/{self.base_symbol}")
        
        try:
            if Path(training_file).exists():
                with open(training_file, 'r') as f:
                    training_info = json.load(f)
                
                training_date = datetime.fromisoformat(training_info['training_completed'])
                days_since = (datetime.now() - training_date).days
                
                print(f"[TRAINING] Found existing training from {training_date.strftime('%Y-%m-%d')}")
                print(f"[TRAINING] Days since training: {days_since}")
                print(f"[TRAINING] Training data points: {training_info['data_points']}")
                
                agent_a1_path = models_dir / f"agent_a1_{self.base_symbol}.zip"
                agent_a2_path = models_dir / f"agent_a2_{self.base_symbol}.zip"
                
                if agent_a1_path.exists() and agent_a2_path.exists():
                    print(f"[TRAINING] Long-term agent models found")
                    return True
                else:
                    print(f"[TRAINING] Metadata found but model files missing")
                    return False
            
            return False
            
        except Exception as e:
            print(f"[TRAINING] Error checking existing training: {e}")
            return False
    
    def prepare_long_term_training_data(self):
        """EXACT copy from original"""
        training_days_count = int(len(self.available_days) * 0.8)
        training_days_count = min(training_days_count, 1250)
        training_days = self.available_days[:training_days_count]
        
        print(f"[TRAINING] Using first {len(training_days)} days (80%) for long-term training")
        print(f"[TRAINING] Remaining {len(self.available_days) - len(training_days)} days for intraday training and testing")
        
        all_data = []
        for i, day in enumerate(training_days):
            if i % 10 == 0:
                print(f"[TRAINING] Loading day {i+1}/{len(training_days)}: {day}")
            
            day_data = self.load_day_data(day)
            all_data.extend(day_data)
        
        if len(all_data) < 1000:
            print(f"[ERROR] Insufficient training data: {len(all_data)} points")
            return None
        
        # EXACT same DataFrame conversion as original
        df_data = []
        for entry in all_data:
            df_data.append({
                'timestamp': entry['timestamp'],
                'Open': entry['price'],
                'High': entry['price'],
                'Low': entry['price'],
                'Close': entry['price'],
                'Volume': entry['volume']
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        print(f"[TRAINING] Training dataset: {len(df)} data points")
        
        return df
    
    def load_day_data(self, trading_date):
        """EXACT copy from original"""
        try:
            stream_name = f"sim:{self.stock_symbol}:{trading_date}"
            messages = self.redis_client.xrange(stream_name)
            
            day_data = []
            for message_id, fields in messages:
                if fields.get('type') == 'timesale':
                    timestamp_ms = int(fields.get('date', message_id.split('-')[0]))
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                    
                    entry = {
                        'timestamp': timestamp,
                        'price': float(fields.get('last', 0)),
                        'volume': int(fields.get('size', 0)),
                        'bid': float(fields.get('bid', 0)),
                        'ask': float(fields.get('ask', 0)),
                        'symbol': fields.get('symbol', ''),
                        'trading_date': trading_date
                    }
                    day_data.append(entry)
            
            return day_data
            
        except Exception as e:
            print(f"[ERROR] Failed to load data for {trading_date}: {e}")
            return []
    
    def get_test_days(self):
        """EXACT copy from original"""
        if len(self.available_days) < 15:
            print(f"[ERROR] Insufficient days for testing")
            return []
        
        test_start_idx = len(self.available_days) // 2 - 2
        return self.available_days[test_start_idx:test_start_idx+5]
    
    def train_intraday_agents(self):
        """EXACT copy from original"""
        intraday_days = self.available_days[-5:]
        
        print(f"[TRAINING] Intraday training on last 5 days: {intraday_days}")
        
        for day_num, day in enumerate(intraday_days, 1):
            print(f"[TRAINING] Intraday day {day_num}/5: {day}")
            
            day_data = self.load_day_data(day)
            if not day_data:
                continue
            
            if day_num == 1:
                interval_size = len(day_data) // 26
                print(f"[TRAINING] Day 1: Using 15-minute intervals ({interval_size} points each)")
            else:
                interval_size = len(day_data) // 78
                print(f"[TRAINING] Day {day_num}: Using 5-minute intervals ({interval_size} points each)")
            
            interval_size = max(1, interval_size)
            
            for i in range(0, len(day_data), interval_size):
                chunk = day_data[i:i+interval_size]
                for entry in chunk:
                    self.price_history.append(entry)
        
        return True

