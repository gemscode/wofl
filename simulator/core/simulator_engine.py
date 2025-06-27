#!/usr/bin/env python3
"""
Core Simulation Engine - SIMPLIFIED Implementation
"""

import time
from datetime import datetime
from pathlib import Path
import json

class SimulatorEngine:
    """SIMPLIFIED simulation engine that actually works"""
    
    def __init__(self, config, data_manager, trading_executor, report_generator):
        self.config = config
        self.data_manager = data_manager
        self.trading_executor = trading_executor
        self.report_generator = report_generator
        
        # Simulation state
        self.simulation_speed = config.get('simulation_speed', 10.0)
        self.daily_results = []
        self.is_system_ready = False
        self.long_term_trained = False
        self.intraday_trained = False
        
        # Pass through config
        self.aggressive_mode = config.get('aggressive_mode', False)
        self.stock_symbol = config.get('stock_symbol')
        self.initial_budget = config.get('initial_budget')
        
        print("✅ SimulatorEngine initialized with SIMPLIFIED strategy")
    
    def run_complete_simulation(self):
        """Run complete simulation with simplified logic"""
        print("STARTING SIMPLIFIED WOLFXE SIMULATION")
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
        
        print("\n[SUCCESS] Simplified simulation finished successfully")
        return True
    
    def train_long_term_agents(self):
        """Train long-term agents"""
        print("\n[TRAINING] PHASE 1: Long-term Agent Training")
        print("=" * 60)
        
        self.data_manager.discover_available_days()
        if len(self.data_manager.available_days) < 15:
            print(f"[ERROR] Insufficient data: {len(self.data_manager.available_days)} days")
            return False
        
        # Check existing training
        if self.data_manager.check_existing_training() and not self.aggressive_mode:
            response = input("Long-term agents already trained. Retrain? (y/N): ").strip().lower()
            if response != 'y':
                print("[TRAINING] Using existing long-term agents")
                self.long_term_trained = True
                success = self.trading_executor.initialize_wolfxe_system()
                return success
        
        # Train with new data
        training_data = self.data_manager.prepare_long_term_training_data()
        if training_data is None:
            return False
        
        success = self.trading_executor.initialize_and_train(training_data)
        if success:
            self.long_term_trained = True
            print("[TRAINING] Long-term agent training completed successfully")
        
        return success
    
    def train_intraday_agents(self):
        """Train intraday agents"""
        print("\n[TRAINING] PHASE 2: Intraday Agent Training")
        print("=" * 60)
        
        if not self.long_term_trained:
            print("[ERROR] Long-term agents must be trained first")
            return False
        
        success = self.data_manager.train_intraday_agents()
        if success:
            self.intraday_trained = True
            print("[TRAINING] Intraday agent training completed")
        
        return success
    
    def run_autotrading_test(self):
        """Run autotrading test with simplified logic"""
        print("\n[TESTING] PHASE 3: Autotrading Test")
        print("=" * 60)
        
        if not (self.long_term_trained and self.intraday_trained):
            print("[ERROR] Both training phases must be completed first")
            return False
        
        test_days = self.data_manager.get_test_days()
        if not test_days:
            return False
        
        print(f"[TESTING] Autotrading test on {len(test_days)} days: {test_days}")
        print(f"[TESTING] Strategy: SIMPLIFIED MOMENTUM + LOW FREQUENCY")
        
        self.is_system_ready = True
        start_time = time.time()
        
        for day_num, trading_date in enumerate(test_days, 1):
            print(f"\n[TEST DAY {day_num}] {trading_date}")
            day_result = self.simulate_trading_day(trading_date)
            if day_result:
                self.daily_results.append(day_result)
        
        elapsed_time = time.time() - start_time
        
        # Generate final report
        self.report_generator.print_final_summary(
            self.daily_results,
            self.trading_executor.executed_trades,
            elapsed_time,
            self.config,
            self.data_manager
        )
        
        return True
    
    def simulate_trading_day(self, trading_date):
        """Simulate trading day with MUCH LESS FREQUENT decisions"""
        try:
            day_data = self.data_manager.load_day_data(trading_date)
            if not day_data:
                print(f"[ERROR] No data for {trading_date}")
                return None
            
            day_start_value = self.trading_executor.current_budget + (self.trading_executor.current_position * day_data[0]['price'])
            trades_today = 0
            
            # MUCH LESS FREQUENT trading decisions
            if self.aggressive_mode:
                interval_size = len(day_data) // 6  # Only 6 decisions per day
            else:
                interval_size = len(day_data) // 4  # Only 4 decisions per day
            
            interval_size = max(1, interval_size)
            
            print(f"[SIMULATION] Processing {len(day_data)} data points in {len(day_data)//interval_size} intervals (SIMPLIFIED)")
            
            # Trading loop with much less frequency
            for i in range(0, len(day_data), interval_size):
                chunk = day_data[i:i+interval_size]
                
                # Build price history
                for entry in chunk:
                    self.data_manager.price_history.append(entry)
                
                # Generate recommendation much less frequently
                min_history = 50  # Reasonable minimum
                if len(self.data_manager.price_history) >= min_history:
                    recommendation = self.trading_executor.generate_trading_recommendation(self.data_manager.price_history)
                    
                    if recommendation['action'] != 'HOLD':
                        current_price = chunk[-1]['price']
                        trade_executed = self.trading_executor.execute_simulated_trade(
                            recommendation['action'],
                            current_price,
                            recommendation['rationale'],
                            self.data_manager.price_history
                        )
                        if trade_executed:
                            trades_today += 1
                            print(f"    [INTERVAL {i//interval_size + 1}] Trade executed: {recommendation['action']}")
                
                # Longer simulation delay
                time.sleep(0.2 / self.simulation_speed)
            
            # End of day calculation
            final_price = day_data[-1]['price']
            day_end_value = self.trading_executor.current_budget + (self.trading_executor.current_position * final_price)
            daily_pnl = day_end_value - day_start_value
            
            day_result = {
                'date': trading_date,
                'start_value': day_start_value,
                'end_value': day_end_value,
                'daily_pnl': daily_pnl,
                'trades': trades_today,
                'final_price': final_price
            }
            
            print(f"[DAY END] {trading_date}: ${day_end_value:.2f} | P&L: ${daily_pnl:+.2f} | Trades: {trades_today}")
            
            return day_result
            
        except Exception as e:
            print(f"[ERROR] Failed to simulate day {trading_date}: {e}")
            return None
    
    @property
    def current_budget(self):
        """Get current budget from trading executor"""
        return self.trading_executor.current_budget
    
    @property
    def current_position(self):
        """Get current position from trading executor"""
        return self.trading_executor.current_position

