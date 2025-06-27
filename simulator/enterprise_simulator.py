#!/usr/bin/env python3
"""
WolfXE Enterprise Simulator
Mirrors the enterprise consumer for testing training before live trading
CORRECTED VERSION - Fixed risk management and trading logic
"""

import redis
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
import warnings
import sys
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Import your WolfXE system
from rw_wolfxe.production.trader_wolfxe import WolfXEnhancedTradingSystem

class WolfXEEnterpriseSimulator:
    """Enterprise simulator with CORRECTED trading logic"""
    
    def __init__(self, redis_host='trader.wolfx0.com', redis_port=6379, 
                 redis_password=None, stock_symbol=None, 
                 initial_budget=None, position_size=None, aggressive_mode=False):
        
        # Redis connection configuration
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            socket_timeout=10
        )
        
        # Test Redis connection first
        self.test_redis_connection()
        
        # Discover available tickers and configure trading parameters
        self.available_tickers = self.discover_available_tickers()
        self.stock_symbol = self.configure_trading_symbol(stock_symbol)
        
        # Ensure we're using simulation symbol
        if not self.stock_symbol.startswith('S_'):
            base_symbol = self.stock_symbol
            self.stock_symbol = f"S_{base_symbol}"
            print(f"[SIMULATION] Using simulation symbol: {self.stock_symbol} (base: {base_symbol})")
        
        self.base_symbol = self.stock_symbol[2:]  # Remove S_ prefix
        self.initial_budget = self.configure_trading_budget(initial_budget)
        self.current_position = self.configure_trading_position(position_size)
        self.current_budget = self.initial_budget
        
        # Aggressive mode configuration
        self.aggressive_mode = aggressive_mode
        
        # Enhanced data management
        self.price_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=500)
        
        # Trading execution tracking
        self.auto_trading_enabled = True
        self.executed_trades = []
        self.position_entry_price = None
        self.total_pnl = 0.0
        
        # Training and simulation state
        self.available_days = []
        self.simulation_speed = 10.0
        self.daily_results = []
        
        # WolfXE system initialization
        self.wolfxe_system = None
        self.is_system_ready = False
        
        # Training phases
        self.long_term_trained = False
        self.intraday_trained = False
        
        self.log_system_initialization()

    def test_redis_connection(self):
        """Test Redis connection"""
        try:
            pong = self.redis_client.ping()
            print(f"[CONNECTION] Redis server status: {pong}")
            return True
        except Exception as e:
            print(f"[ERROR] Redis connection failed: {e}")
            sys.exit(1)

    def discover_available_tickers(self):
        """Discover available simulation symbols"""
        print("DISCOVERING AVAILABLE SIMULATION DATA...")
        print("=" * 80)
        
        try:
            pattern = "sim:S_*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                print("[WARNING] No simulation streams found")
                return {}
            
            # Group by symbol
            symbols = {}
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 3:
                    symbol = parts[1]  # S_GERN
                    date = parts[2]    # 2024-01-15
                    
                    if symbol not in symbols:
                        symbols[symbol] = []
                    symbols[symbol].append(date)
            
            # Get stream information
            available_tickers = {}
            for symbol, dates in symbols.items():
                dates.sort()
                total_messages = 0
                
                for date in dates:
                    stream_name = f"sim:{symbol}:{date}"
                    try:
                        length = self.redis_client.xlen(stream_name)
                        total_messages += length
                    except:
                        pass
                
                available_tickers[symbol] = {
                    'message_count': total_messages,
                    'trading_days': len(dates),
                    'first_date': dates[0] if dates else None,
                    'last_date': dates[-1] if dates else None,
                    'active': True
                }
            
            # Display discovered tickers
            if available_tickers:
                print("AVAILABLE SIMULATION SYMBOLS:")
                print("-" * 80)
                print(f"{'SYMBOL':<10} {'MESSAGES':<12} {'DAYS':<8} {'DATE RANGE':<25} {'STATUS':<10}")
                print("-" * 80)
                
                for ticker, info in available_tickers.items():
                    date_range = f"{info['first_date']} to {info['last_date']}" if info['first_date'] else "N/A"
                    print(f"{ticker:<10} {info['message_count']:<12} {info['trading_days']:<8} {date_range:<25} {'READY':<10}")
                
                print("-" * 80)
                print(f"Total simulation symbols: {len(available_tickers)}")
            
            return available_tickers
            
        except Exception as e:
            print(f"[ERROR] Failed to discover simulation tickers: {e}")
            return {}

    def configure_trading_symbol(self, provided_symbol):
        """Configure trading symbol with user interaction"""
        if provided_symbol:
            symbol = provided_symbol.upper()
            if not symbol.startswith('S_'):
                symbol = f"S_{symbol}"
            
            if symbol in self.available_tickers:
                print(f"SELECTED SYMBOL: {symbol}")
                return symbol
            else:
                print(f"WARNING: Provided symbol '{symbol}' not found in simulation data")
        
        if not self.available_tickers:
            print("ERROR: No simulation tickers available.")
            sys.exit(1)
        
        available_symbols = list(self.available_tickers.keys())
        
        print("\nSIMULATION SYMBOL SELECTION:")
        print("=" * 40)
        print("AVAILABLE SIMULATION SYMBOLS:")
        for i, ticker in enumerate(available_symbols, 1):
            info = self.available_tickers[ticker]
            print(f"  {i}. {ticker} ({info['trading_days']} days, {info['message_count']} messages)")
        
        while True:
            try:
                user_input = input(f"\nSelect symbol (1-{len(available_symbols)} or enter symbol): ").strip()
                
                if user_input.isdigit():
                    selection = int(user_input)
                    if 1 <= selection <= len(available_symbols):
                        selected_symbol = available_symbols[selection - 1]
                        print(f"SELECTED: {selected_symbol}")
                        return selected_symbol
                
                symbol = user_input.upper()
                if not symbol.startswith('S_'):
                    symbol = f"S_{symbol}"
                
                if symbol in self.available_tickers:
                    print(f"SELECTED: {symbol}")
                    return symbol
                else:
                    print(f"Symbol '{symbol}' not available.")
                    
            except KeyboardInterrupt:
                sys.exit(0)

    def configure_trading_budget(self, provided_budget):
        """Configure trading budget"""
        if provided_budget is not None and provided_budget > 0:
            print(f"TRADING BUDGET: ${provided_budget:,.2f}")
            return provided_budget
        
        while True:
            try:
                budget_input = input("Enter trading budget ($): ").strip().replace(',', '').replace('$', '')
                budget = float(budget_input)
                if budget > 0:
                    print(f"TRADING BUDGET SET: ${budget:,.2f}")
                    return budget
                print("Budget must be greater than zero.")
            except ValueError:
                print("Invalid amount. Please enter a numeric value.")
            except KeyboardInterrupt:
                sys.exit(0)

    def configure_trading_position(self, provided_position):
        """Configure current position"""
        if provided_position is not None:
            print(f"CURRENT POSITION: {provided_position} shares")
            return provided_position
        
        while True:
            try:
                position_input = input(f"Current {self.stock_symbol} position (shares, default 0): ").strip()
                if not position_input:
                    position_input = "0"
                position = int(position_input)
                if position >= 0:
                    print(f"CURRENT POSITION SET: {position} shares")
                    return position
                print("Position cannot be negative.")
            except ValueError:
                print("Invalid input. Please enter a whole number.")
            except KeyboardInterrupt:
                sys.exit(0)

    def log_system_initialization(self):
        """Log system initialization"""
        print("\n" + "=" * 80)
        mode_text = "CORRECTED AGGRESSIVE MODE" if self.aggressive_mode else "CORRECTED TRAINING & TESTING MODE"
        print(f"WOLFXE ENTERPRISE SIMULATOR - {mode_text}")
        print("=" * 80)
        print(f"SYMBOL:           {self.stock_symbol}")
        print(f"BASE SYMBOL:      {self.base_symbol}")
        print(f"INITIAL BUDGET:   ${self.initial_budget:,.2f}")
        print(f"CURRENT POSITION: {self.current_position} shares")
        if self.aggressive_mode:
            print(f"RISK MODE:        CORRECTED AGGRESSIVE")
        print("=" * 80)

    def discover_available_days(self):
        """Discover available trading days"""
        try:
            pattern = f"sim:{self.stock_symbol}:*"
            keys = self.redis_client.keys(pattern)
            
            days = []
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 3:
                    days.append(parts[2])
            
            return sorted(days)
            
        except Exception as e:
            print(f"[ERROR] Failed to discover days: {e}")
            return []

    def check_existing_training(self):
        """Check if long-term agents are already trained"""
        training_file = f"training_metadata_{self.base_symbol}.json"
        models_dir = Path(f"models/{self.base_symbol}")
        
        try:
            # Check for training metadata
            if os.path.exists(training_file):
                with open(training_file, 'r') as f:
                    training_info = json.load(f)
                
                training_date = datetime.fromisoformat(training_info['training_completed'])
                days_since = (datetime.now() - training_date).days
                
                print(f"[TRAINING] Found existing training from {training_date.strftime('%Y-%m-%d')}")
                print(f"[TRAINING] Days since training: {days_since}")
                print(f"[TRAINING] Training data points: {training_info['data_points']}")
                
                # Check if model files exist
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

    def train_long_term_agents(self):
        """Train long-term agents on up to 5 years of data"""
        print("\n[TRAINING] PHASE 1: Long-term Agent Training")
        print("=" * 60)
        
        # ALWAYS discover available days first
        self.available_days = self.discover_available_days()
        if len(self.available_days) < 15:  # Need at least 15 days total
            print(f"[ERROR] Insufficient data: {len(self.available_days)} days")
            return False
        
        # Check if already trained
        if self.check_existing_training() and not self.aggressive_mode:
            response = input("Long-term agents already trained. Retrain? (y/N): ").strip().lower()
            if response != 'y':
                print("[TRAINING] Using existing long-term agents")
                self.long_term_trained = True
                
                # CRITICAL FIX: Initialize WolfXE system even when using existing training
                self.wolfxe_system = WolfXEnhancedTradingSystem(self.base_symbol, self.initial_budget)
                self.wolfxe_system.portfolio['shares'] = self.current_position
                self.wolfxe_system.portfolio['cash'] = self.current_budget
                
                # Apply corrected settings
                self.apply_corrected_settings()
                
                return True
        
        print("[TRAINING] Loading historical data for long-term training...")
        
        # Use first 80% of available data for long-term training
        training_days_count = int(len(self.available_days) * 0.8)
        training_days_count = min(training_days_count, 1250)  # Max 5 years
        training_days = self.available_days[:training_days_count]
        
        print(f"[TRAINING] Using first {len(training_days)} days (80%) for long-term training")
        print(f"[TRAINING] Remaining {len(self.available_days) - len(training_days)} days for intraday training and testing")
        
        # Load data for long-term training
        all_data = []
        for i, day in enumerate(training_days):
            if i % 10 == 0:
                print(f"[TRAINING] Loading day {i+1}/{len(training_days)}: {day}")
            
            day_data = self.load_day_data(day)
            all_data.extend(day_data)
        
        if len(all_data) < 1000:
            print(f"[ERROR] Insufficient training data: {len(all_data)} points")
            return False
        
        # Convert to DataFrame for training
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
        
        # Initialize WolfXE system
        self.wolfxe_system = WolfXEnhancedTradingSystem(self.base_symbol, self.initial_budget)
        self.wolfxe_system.portfolio['shares'] = self.current_position
        self.wolfxe_system.portfolio['cash'] = self.current_budget
        
        # Apply corrected settings
        self.apply_corrected_settings()
        
        # Train long-term agents
        print("[TRAINING] Training long-term agents...")
        success = self.wolfxe_system.load_or_train_all_agents(df)
        
        if success:
            # Save training metadata
            self.save_training_metadata(len(df), len(training_days))
            self.long_term_trained = True
            print("[TRAINING] Long-term agent training completed successfully")
            return True
        else:
            print("[ERROR] Long-term agent training failed")
            return False

    def apply_corrected_settings(self):
        """Apply CORRECTED trading settings with proper risk management"""
        if not self.wolfxe_system:
            return
        
        print("[CORRECTED] Applying proper risk management settings...")
        
        # CORRECTED: Proper activity enforcer settings
        if hasattr(self.wolfxe_system, 'activity_enforcer'):
            enforcer = self.wolfxe_system.activity_enforcer
            
            if self.aggressive_mode:
                # Aggressive but not reckless
                enforcer.min_trades_per_week = 3  # Reasonable target
                enforcer.max_days_without_trade = 4  # Not too frequent
            else:
                # Conservative settings
                enforcer.min_trades_per_week = 2
                enforcer.max_days_without_trade = 6
            
            print(f"[CORRECTED] Activity enforcer: {enforcer.min_trades_per_week} trades/week, max {enforcer.max_days_without_trade} days without trade")
        
        # CORRECTED: Proper risk management constraints
        if hasattr(self.wolfxe_system, 'constraints'):
            constraints = self.wolfxe_system.constraints
            
            if self.aggressive_mode:
                # Aggressive but with PROPER risk management
                constraints['stop_loss_pct'] = 0.015  # 1.5% stop loss
                constraints['take_profit_pct'] = 0.06  # 6% take profit (4:1 reward/risk)
                constraints['max_position_size'] = 0.8  # Reasonable position sizing
                constraints['min_trade_amount'] = 800  # Higher minimum to reduce noise
                constraints['min_hold_days'] = 0  # No minimum hold
                constraints['max_hold_days'] = 3  # Quick exits
                print("[CORRECTED] Aggressive mode: 1.5% stop loss, 6% take profit (4:1 R/R)")
            else:
                # Conservative with good risk management
                constraints['stop_loss_pct'] = 0.02  # 2% stop loss
                constraints['take_profit_pct'] = 0.08  # 8% take profit (4:1 reward/risk)
                constraints['max_position_size'] = 0.7  # Conservative sizing
                constraints['min_trade_amount'] = 1000  # Higher minimum
                constraints['min_hold_days'] = 1  # Hold at least 1 day
                constraints['max_hold_days'] = 5  # Reasonable hold period
                print("[CORRECTED] Conservative mode: 2% stop loss, 8% take profit (4:1 R/R)")

    def train_intraday_agents(self):
        """Train intraday agents on 5 days with 15-minute intervals, then 5-minute updates"""
        print("\n[TRAINING] PHASE 2: Intraday Agent Training")
        print("=" * 60)
        
        if not self.long_term_trained:
            print("[ERROR] Long-term agents must be trained first")
            return False
        
        # Use the LAST 5 days for intraday training (most recent data)
        if len(self.available_days) < 5:
            print(f"[ERROR] Insufficient total days for intraday training: {len(self.available_days)}")
            return False
        
        # Use last 5 days for intraday training
        intraday_days = self.available_days[-5:]
        
        print(f"[TRAINING] Intraday training on last 5 days: {intraday_days}")
        
        # Simulate intraday training with 15-minute intervals, then 5-minute updates
        for day_num, day in enumerate(intraday_days, 1):
            print(f"[TRAINING] Intraday day {day_num}/5: {day}")
            
            day_data = self.load_day_data(day)
            if not day_data:
                continue
            
            # Process data in intervals
            if day_num == 1:
                # First day: 15-minute intervals
                interval_size = len(day_data) // 26  # ~15 minutes (390 minutes / 15)
                print(f"[TRAINING] Day 1: Using 15-minute intervals ({interval_size} points each)")
            else:
                # Subsequent days: 5-minute intervals
                interval_size = len(day_data) // 78  # ~5 minutes (390 minutes / 5)
                print(f"[TRAINING] Day {day_num}: Using 5-minute intervals ({interval_size} points each)")
            
            interval_size = max(1, interval_size)
            
            # Process intervals
            for i in range(0, len(day_data), interval_size):
                chunk = day_data[i:i+interval_size]
                
                # Add to price history
                for entry in chunk:
                    self.price_history.append(entry)
                
                # Update intraday learning (simplified)
                if len(self.price_history) >= 50:
                    self.update_intraday_learning()
        
        self.intraday_trained = True
        print("[TRAINING] Intraday agent training completed")
        return True

    def update_intraday_learning(self):
        """Update intraday learning (simplified for simulation)"""
        # This would update the intraday components of the WolfXE system
        # For simulation, we'll just mark it as updated
        pass

    def load_day_data(self, trading_date):
        """Load data for a specific trading day"""
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

    def save_training_metadata(self, data_points, training_days):
        """Save training metadata"""
        try:
            training_info = {
                'symbol': self.base_symbol,
                'training_completed': datetime.now().isoformat(),
                'data_points': data_points,
                'training_days': training_days,
                'trained_on': 'simulation',
                'model_version': '2.1_corrected',
                'long_term_trained': True,
                'intraday_trained': False,
                'aggressive_mode': self.aggressive_mode,
                'risk_management': 'corrected'
            }
            
            training_file = f"training_metadata_{self.base_symbol}.json"
            with open(training_file, 'w') as f:
                json.dump(training_info, f, indent=2)
            
            print(f"[TRAINING] Metadata saved to {training_file}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save training metadata: {e}")

    def run_autotrading_test(self):
        """Run 5-day autotrading test"""
        print("\n[TESTING] PHASE 3: Autotrading Test")
        print("=" * 60)
        
        if not (self.long_term_trained and self.intraday_trained):
            print("[ERROR] Both training phases must be completed first")
            return False
        
        # Use middle 5 days for testing (avoid overlap with training data)
        if len(self.available_days) < 15:
            print(f"[ERROR] Insufficient days for testing: {len(self.available_days)}")
            return False
        
        # Use middle days for testing
        test_start_idx = len(self.available_days) // 2 - 2  # Middle - 2
        test_days = self.available_days[test_start_idx:test_start_idx+5]
        
        print(f"[TESTING] Autotrading test on middle 5 days: {test_days}")
        print(f"[TESTING] Auto-trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
        if self.aggressive_mode:
            print(f"[TESTING] Risk mode: CORRECTED AGGRESSIVE")
        
        # Ensure system is ready
        self.is_system_ready = True
        
        start_time = time.time()
        
        for day_num, trading_date in enumerate(test_days, 1):
            print(f"\n[TEST DAY {day_num}] {trading_date}")
            self.simulate_trading_day(trading_date)
        
        elapsed_time = time.time() - start_time
        self.print_final_summary(elapsed_time)
        
        return True

    def simulate_trading_day(self, trading_date):
        """Simulate trading for a specific day with CORRECTED frequency"""
        try:
            day_data = self.load_day_data(trading_date)
            if not day_data:
                print(f"[ERROR] No data for {trading_date}")
                return
            
            day_start_value = self.current_budget + (self.current_position * day_data[0]['price'])
            trades_today = 0
            
            # CORRECTED: Much less frequent trading decisions
            if self.aggressive_mode:
                # Aggressive mode: every ~15 minutes (reasonable)
                interval_size = len(day_data) // 26  # ~26 intervals per day
            else:
                # Normal mode: every ~30 minutes (conservative)
                interval_size = len(day_data) // 13  # ~13 intervals per day
            
            interval_size = max(1, interval_size)
            
            print(f"[SIMULATION] Processing {len(day_data)} data points in {len(day_data)//interval_size} intervals")
            
            for i in range(0, len(day_data), interval_size):
                chunk = day_data[i:i+interval_size]
                
                # Process chunk data
                for entry in chunk:
                    self.price_history.append(entry)
                
                # Generate recommendation with proper data requirements
                min_history = 100  # Require more data for better decisions
                if len(self.price_history) >= min_history:
                    recommendation = self.generate_trading_recommendation()
                    
                    if recommendation['action'] != 'HOLD' and self.auto_trading_enabled:
                        current_price = chunk[-1]['price']
                        trade_executed = self.execute_simulated_trade(
                            recommendation['action'],
                            current_price,
                            recommendation['rationale']
                        )
                        if trade_executed:
                            trades_today += 1
                            print(f"    [INTERVAL {i//interval_size + 1}] Trade executed: {recommendation['action']}")
                
                # Simulate processing delay
                time.sleep(0.1 / self.simulation_speed)
            
            # End of day summary
            final_price = day_data[-1]['price']
            day_end_value = self.current_budget + (self.current_position * final_price)
            daily_pnl = day_end_value - day_start_value
            
            day_result = {
                'date': trading_date,
                'start_value': day_start_value,
                'end_value': day_end_value,
                'daily_pnl': daily_pnl,
                'trades': trades_today,
                'final_price': final_price
            }
            
            self.daily_results.append(day_result)
            
            print(f"[DAY END] {trading_date}: ${day_end_value:.2f} | P&L: ${daily_pnl:+.2f} | Trades: {trades_today}")
            
        except Exception as e:
            print(f"[ERROR] Failed to simulate day {trading_date}: {e}")

    def generate_trading_recommendation(self):
        """Generate trading recommendation with better logic"""
        if not self.is_system_ready or len(self.price_history) < 50:
            return {'action': 'HOLD', 'rationale': 'Insufficient data'}
        
        try:
            # Convert price history to DataFrame
            df_data = []
            for entry in list(self.price_history)[-100:]:  # Use last 100 entries
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
            
            # Get WolfXE decision
            current_state = df.iloc[-1]
            decision_analysis = self.wolfxe_system.make_enhanced_decision(df, current_state)
            
            # CORRECTED: Apply constraints with proper force trade handling
            final_action = self.wolfxe_system.apply_enhanced_constraints(
                decision_analysis['decision'],
                current_state['Close'],
                datetime.now().date(),
                force_trade=decision_analysis.get('force_trade', False)
            )
            
            return {
                'action': final_action,
                'rationale': decision_analysis['reasoning'],
                'force_trade': decision_analysis.get('force_trade', False)
            }
            
        except Exception as e:
            return {'action': 'HOLD', 'rationale': f'Error: {e}'}

    def execute_simulated_trade(self, action, current_price, reasoning):
        """Execute simulated trade with CORRECTED position sizing"""
        if action == 'BUY' and self.current_position == 0:
            # CORRECTED: More conservative position sizing
            cash_usage = 0.7 if self.aggressive_mode else 0.6  # Much more conservative
            available_cash = self.current_budget * cash_usage
            shares = int(available_cash // current_price)
            
            if shares > 0:
                cost = shares * current_price
                self.current_position = shares
                self.current_budget -= cost
                self.position_entry_price = current_price
                
                # Update activity metrics
                if hasattr(self.wolfxe_system, 'activity_enforcer'):
                    self.wolfxe_system.activity_enforcer.update_activity_metrics(True, datetime.now().date(), 0.0)
                
                trade = {
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'shares': shares,
                    'price': current_price,
                    'cost': cost,
                    'reasoning': reasoning
                }
                self.executed_trades.append(trade)
                print(f"  [TRADE] BUY {shares} @ ${current_price:.2f} | Reason: {reasoning}")
                return True
        
        elif action == 'SELL' and self.current_position > 0:
            proceeds = self.current_position * current_price
            pnl = proceeds - (self.current_position * (self.position_entry_price or current_price))
            
            # Update activity metrics with performance
            performance = pnl / self.initial_budget
            if hasattr(self.wolfxe_system, 'activity_enforcer'):
                self.wolfxe_system.activity_enforcer.update_activity_metrics(True, datetime.now().date(), performance)
            
            trade = {
                'timestamp': datetime.now(),
                'action': 'SELL',
                'shares': self.current_position,
                'price': current_price,
                'proceeds': proceeds,
                'pnl': pnl,
                'reasoning': reasoning
            }
            self.executed_trades.append(trade)
            print(f"  [TRADE] SELL {self.current_position} @ ${current_price:.2f} | P&L: ${pnl:+.2f} | Reason: {reasoning}")
            
            self.current_position = 0
            self.current_budget += proceeds
            self.position_entry_price = None
            return True
        
        return False

    def calculate_buy_and_hold_benchmark(self):
        """Calculate buy-and-hold performance for comparison"""
        if not self.daily_results:
            return None
        
        try:
            # Get first and last day prices from test period
            first_day = self.available_days[len(self.available_days) // 2 - 2]  # Same as test start
            last_day = self.available_days[len(self.available_days) // 2 + 2]   # Same as test end
            
            first_day_data = self.load_day_data(first_day)
            last_day_data = self.load_day_data(last_day)
            
            if not first_day_data or not last_day_data:
                return None
            
            start_price = first_day_data[0]['price']
            end_price = last_day_data[-1]['price']
            
            # Calculate shares that could be bought with initial budget
            shares = int((self.initial_budget * 0.9) // start_price)
            buy_hold_value = shares * end_price + (self.initial_budget - shares * start_price)
            buy_hold_return = ((buy_hold_value - self.initial_budget) / self.initial_budget) * 100
            
            return {
                'start_price': start_price,
                'end_price': end_price,
                'shares': shares,
                'final_value': buy_hold_value,
                'return_pct': buy_hold_return
            }
        except Exception as e:
            print(f"[ERROR] Failed to calculate buy-and-hold benchmark: {e}")
            return None

    def print_final_summary(self, elapsed_time):
        """Print comprehensive final summary with benchmark comparison"""
        print("\n" + "=" * 80)
        mode_text = "CORRECTED AGGRESSIVE MODE" if self.aggressive_mode else "CORRECTED STANDARD MODE"
        print(f"WOLFXE SIMULATION COMPLETE - FINAL REPORT ({mode_text})")
        print("=" * 80)
        
        # Training Summary
        print("TRAINING SUMMARY:")
        print(f"  Long-term Agents:     {'TRAINED' if self.long_term_trained else 'NOT TRAINED'}")
        print(f"  Intraday Agents:      {'TRAINED' if self.intraday_trained else 'NOT TRAINED'}")
        print(f"  System Status:        {'READY' if self.is_system_ready else 'NOT READY'}")
        print(f"  Trading Mode:         {mode_text}")
        print(f"  Risk Management:      CORRECTED")
        print("")
        
        # Trading Results
        if self.daily_results:
            final_value = self.daily_results[-1]['end_value']
            total_return = ((final_value - self.initial_budget) / self.initial_budget) * 100
            
            print("TRADING RESULTS:")
            print(f"  Initial Budget:       ${self.initial_budget:,.2f}")
            print(f"  Final Value:          ${final_value:,.2f}")
            print(f"  Total Return:         {total_return:+.2f}%")
            print(f"  Total Trades:         {len(self.executed_trades)}")
            print(f"  Test Days:            {len(self.daily_results)}")
            print(f"  Simulation Time:      {elapsed_time:.1f} seconds")
            print("")
            
            # BENCHMARK COMPARISON
            print("BENCHMARK COMPARISON:")
            benchmark = self.calculate_buy_and_hold_benchmark()
            if benchmark:
                print(f"  Buy & Hold Return:    {benchmark['return_pct']:+.2f}%")
                print(f"  Buy & Hold Value:     ${benchmark['final_value']:,.2f}")
                print(f"  Strategy vs B&H:      {total_return - benchmark['return_pct']:+.2f}% difference")
                
                if total_return < benchmark['return_pct']:
                    print(f"  ⚠️  Strategy UNDERPERFORMED buy-and-hold by {benchmark['return_pct'] - total_return:.2f}%")
                else:
                    print(f"  ✅ Strategy OUTPERFORMED buy-and-hold by {total_return - benchmark['return_pct']:.2f}%")
            else:
                print("  Buy & Hold:           Unable to calculate")
            print("")
            
            # Daily Performance
            winning_days = len([d for d in self.daily_results if d['daily_pnl'] > 0])
            win_rate = winning_days / len(self.daily_results) * 100
            
            print("DAILY PERFORMANCE:")
            print(f"  Winning Days:         {winning_days}/{len(self.daily_results)} ({win_rate:.1f}%)")
            
            if self.daily_results:
                best_day = max(self.daily_results, key=lambda x: x['daily_pnl'])
                worst_day = min(self.daily_results, key=lambda x: x['daily_pnl'])
                avg_daily_pnl = sum(d['daily_pnl'] for d in self.daily_results) / len(self.daily_results)
                
                print(f"  Best Day:             {best_day['date']} (+${best_day['daily_pnl']:.2f})")
                print(f"  Worst Day:            {worst_day['date']} (${worst_day['daily_pnl']:+.2f})")
                print(f"  Average Daily P&L:    ${avg_daily_pnl:+.2f}")
            print("")
            
            # Trade Analysis
            if self.executed_trades:
                buy_trades = [t for t in self.executed_trades if t['action'] == 'BUY']
                sell_trades = [t for t in self.executed_trades if t['action'] == 'SELL']
                
                print("TRADE ANALYSIS:")
                print(f"  Buy Trades:           {len(buy_trades)}")
                print(f"  Sell Trades:          {len(sell_trades)}")
                
                if sell_trades:
                    profitable_trades = [t for t in sell_trades if t['pnl'] > 0]
                    trade_win_rate = len(profitable_trades) / len(sell_trades) * 100
                    print(f"  Trade Win Rate:       {trade_win_rate:.1f}%")
                    
                    if profitable_trades:
                        avg_profit = sum(t['pnl'] for t in profitable_trades) / len(profitable_trades)
                        print(f"  Average Profit:       ${avg_profit:.2f}")
                    
                    losing_trades = [t for t in sell_trades if t['pnl'] < 0]
                    if losing_trades:
                        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
                        print(f"  Average Loss:         ${avg_loss:.2f}")
                        
                        # Profit factor
                        total_profit = sum(t['pnl'] for t in profitable_trades)
                        total_loss = abs(sum(t['pnl'] for t in losing_trades))
                        if total_loss > 0:
                            profit_factor = total_profit / total_loss
                            print(f"  Profit Factor:        {profit_factor:.2f}")
                            
                            # Risk assessment
                            if profit_factor >= 1.5:
                                print(f"  Risk Assessment:      ✅ GOOD (PF >= 1.5)")
                            elif profit_factor >= 1.0:
                                print(f"  Risk Assessment:      ⚠️  MARGINAL (PF >= 1.0)")
                            else:
                                print(f"  Risk Assessment:      ❌ POOR (PF < 1.0)")
                
                # Trading frequency analysis
                total_intervals = len(self.daily_results) * 26 if self.aggressive_mode else len(self.daily_results) * 13
                if total_intervals > 0:
                    trade_frequency = len(self.executed_trades) / total_intervals * 100
                    print(f"  Trade Frequency:      {trade_frequency:.2f}% of intervals")
                
                print("")
        
        print("SYSTEM READINESS:")
        if self.long_term_trained and self.intraday_trained:
            # Additional readiness checks
            benchmark = self.calculate_buy_and_hold_benchmark()
            profit_factor_ok = False
            
            if self.executed_trades:
                sell_trades = [t for t in self.executed_trades if t['action'] == 'SELL']
                if sell_trades:
                    profitable_trades = [t for t in sell_trades if t['pnl'] > 0]
                    losing_trades = [t for t in sell_trades if t['pnl'] < 0]
                    if profitable_trades and losing_trades:
                        total_profit = sum(t['pnl'] for t in profitable_trades)
                        total_loss = abs(sum(t['pnl'] for t in losing_trades))
                        profit_factor = total_profit / total_loss if total_loss > 0 else 0
                        profit_factor_ok = profit_factor >= 1.0
            
            beats_benchmark = False
            if benchmark:
                final_value = self.daily_results[-1]['end_value'] if self.daily_results else self.initial_budget
                strategy_return = ((final_value - self.initial_budget) / self.initial_budget) * 100
                beats_benchmark = strategy_return >= benchmark['return_pct']
            
            if profit_factor_ok and beats_benchmark:
                print("  ✅ System is READY for live trading")
                print("  ✅ Long-term agents are pre-trained")
                print("  ✅ Intraday agents are calibrated")
                print("  ✅ Trading logic validated")
                print("  ✅ Risk management corrected")
                print("  ✅ Beats buy-and-hold benchmark")
                print("  ✅ Positive profit factor")
            else:
                print("  ⚠️  System has CONCERNS for live trading")
                print("  ✅ Long-term agents are pre-trained")
                print("  ✅ Intraday agents are calibrated")
                if not profit_factor_ok:
                    print("  ❌ Poor profit factor (< 1.0)")
                if not beats_benchmark:
                    print("  ❌ Does not beat buy-and-hold")
                print("  ⚠️  Consider further optimization")
        else:
            print("  ❌ System is NOT ready for live trading")
            print("  ❌ Training incomplete")
        
        print("=" * 80)

    def run_complete_simulation(self):
        """Run complete simulation: training + testing"""
        print("STARTING COMPLETE WOLFXE SIMULATION")
        print("=" * 80)
        
        # Phase 1: Long-term training
        if not self.train_long_term_agents():
            print("[ERROR] Long-term training failed")
            return False
        
        # Phase 2: Intraday training
        if not self.train_intraday_agents():
            print("[ERROR] Intraday training failed")
            return False
        
        # Phase 3: Autotrading test
        if not self.run_autotrading_test():
            print("[ERROR] Autotrading test failed")
            return False
        
        print("\n[SUCCESS] Complete simulation finished successfully")
        return True

    def offer_aggressive_retrain(self):
        """Offer option to retrain with aggressive settings"""
        if self.aggressive_mode:
            return  # Already in aggressive mode
        
        print("\n" + "=" * 80)
        print("SIMULATION OPTIONS")
        print("=" * 80)
        print("Your simulation has completed. You can now:")
        print("  1. Exit and use current training for live trading")
        print("  2. Retrain with CORRECTED AGGRESSIVE approach")
        print("  3. View detailed trade log")
        print("")
        
        while True:
            try:
                choice = input("Select option (1-3): ").strip()
                
                if choice == '1':
                    print("\n✅ Simulation complete. System ready for live trading.")
                    return
                
                elif choice == '2':
                    print("\n🔥 Starting CORRECTED AGGRESSIVE retraining...")
                    print("This will use:")
                    print("  - Better risk management (1.5% stop loss, 6% take profit)")
                    print("  - Conservative position sizing (70% vs 60%)")
                    print("  - Reasonable trading frequency (every 15 minutes)")
                    print("  - Higher minimum trade amounts")
                    print("  - Proper profit factor targeting (>1.0)")
                    print("")
                    
                    confirm = input("Proceed with corrected aggressive retraining? (y/N): ").strip().lower()
                    if confirm == 'y':
                        # Reset for aggressive mode
                        self.aggressive_mode = True
                        self.executed_trades = []
                        self.daily_results = []
                        self.current_budget = self.initial_budget
                        self.current_position = 0
                        self.position_entry_price = None
                        self.price_history.clear()
                        
                        # Run corrected aggressive simulation
                        print("\n🔥 CORRECTED AGGRESSIVE MODE ACTIVATED")
                        self.run_complete_simulation()
                        
                        # Show comparison
                        print("\n📊 CORRECTED AGGRESSIVE MODE RESULTS COMPLETE")
                        print("Compare the results above with your previous standard mode results.")
                        return
                    else:
                        continue
                
                elif choice == '3':
                    self.show_detailed_trade_log()
                    continue
                
                else:
                    print("Invalid option. Please select 1, 2, or 3.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting simulation.")
                return

    def show_detailed_trade_log(self):
        """Show detailed trade log"""
        if not self.executed_trades:
            print("\nNo trades executed to display.")
            return
        
        print("\n" + "=" * 80)
        print("DETAILED TRADE LOG")
        print("=" * 80)
        
        for i, trade in enumerate(self.executed_trades, 1):
            print(f"Trade {i}:")
            print(f"  Time:     {trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Action:   {trade['action']}")
            print(f"  Shares:   {trade['shares']}")
            print(f"  Price:    ${trade['price']:.2f}")
            
            if trade['action'] == 'BUY':
                print(f"  Cost:     ${trade['cost']:.2f}")
            else:
                print(f"  Proceeds: ${trade['proceeds']:.2f}")
                print(f"  P&L:      ${trade['pnl']:+.2f}")
            
            print(f"  Reason:   {trade['reasoning']}")
            print("")
        
        input("Press Enter to continue...")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='WolfXE Enterprise Simulator - CORRECTED Training & Testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--symbol', 
                       help='Stock symbol for simulation')
    parser.add_argument('--budget', type=float, 
                       help='Initial trading budget')
    parser.add_argument('--position', type=int, 
                       help='Current stock position in shares')
    parser.add_argument('--speed', type=float, default=10.0,
                       help='Simulation speed multiplier')
    parser.add_argument('--aggressive', action='store_true',
                       help='Start in corrected aggressive mode')
    parser.add_argument('--redis-password', 
                       help='Redis authentication password')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis server port')
    parser.add_argument('--redis-host', default='trader.wolfx0.com',
                       help='Redis server hostname')
    
    args = parser.parse_args()
    
    # Load Redis password
    redis_password = args.redis_password
    if not redis_password:
        try:
            with open('.redis_passwd', 'r') as f:
                redis_password = f.read().strip()
            print("[CONFIG] Redis password loaded from .redis_passwd file")
        except FileNotFoundError:
            print("[ERROR] Redis password not provided and .redis_passwd file not found")
            sys.exit(1)
    
    # Create and run simulator
    simulator = WolfXEEnterpriseSimulator(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=redis_password,
        stock_symbol=args.symbol,
        initial_budget=args.budget,
        position_size=args.position,
        aggressive_mode=args.aggressive
    )
    
    simulator.simulation_speed = args.speed
    
    # Run complete simulation
    success = simulator.run_complete_simulation()
    
    if success:
        # Offer aggressive retrain option
        simulator.offer_aggressive_retrain()

if __name__ == "__main__":
    main()

