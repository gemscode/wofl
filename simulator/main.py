#!/usr/bin/env python3
"""
RW WolfXE Simulator - Main Entry Point
"""

import sys
from pathlib import Path

# Import from local modules
from core.simulator_engine import SimulatorEngine
from core.data_manager import DataManager
from core.trading_executor import TradingExecutor
from config.redis_config import RedisConfig
from ui.user_interface import UserInterface
from ui.report_generator import ReportGenerator

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RW WolfXE Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--symbol', help='Stock symbol for simulation')
    parser.add_argument('--budget', type=float, help='Initial trading budget')
    parser.add_argument('--position', type=int, help='Current stock position in shares')
    parser.add_argument('--speed', type=float, default=10.0, help='Simulation speed multiplier')
    parser.add_argument('--aggressive', action='store_true', help='Start in aggressive mode')
    parser.add_argument('--trading-mode', choices=['LONG_ONLY', 'SHORT_ONLY', 'LONG_SHORT'], 
                       default='LONG_SHORT', help='Trading mode (default: LONG_SHORT)')
    parser.add_argument('--redis-password', help='Redis authentication password')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis server port')
    parser.add_argument('--redis-host', default='trader.wolfx0.com', help='Redis server hostname')
    
    args = parser.parse_args()
    
    try:
        print("RW WOLFXE SIMULATOR")
        print("=" * 60)
        print("COMPONENTS:")
        print("  CONFIG: Redis configuration")
        print("  UI: User interface") 
        print("  REPORTING: Report generator")
        print("  CORE: Simulation engine")
        print("  DATA: Data manager")
        print("  TRADING: Trading executor")
        print("=" * 60)
        
        config = {
            'simulation_speed': args.speed,
            'aggressive_mode': args.aggressive,
            'trading_mode': args.trading_mode,
            'redis_host': args.redis_host,
            'redis_port': args.redis_port,
            'redis_password': args.redis_password
        }
        
        print("CONNECTING: Redis...")
        redis_config = RedisConfig(config)
        redis_client = redis_config.get_client()
        
        print("INITIALIZING: User interface...")
        ui_manager = UserInterface()
        
        # Discover available tickers
        available_tickers = ui_manager.discover_available_tickers(redis_client)
        
        # Configure trading symbol
        stock_symbol = ui_manager.configure_trading_symbol(args.symbol, available_tickers)
        
        # Configure trading budget
        initial_budget = ui_manager.configure_trading_budget(args.budget)
        
        # Configure trading position with symbol change option
        position_size = ui_manager.configure_trading_position(args.position, stock_symbol)
        
        # Handle symbol change request
        if position_size == 'CHANGE_SYMBOL':
            print("\nChanging symbol selection...")
            stock_symbol = ui_manager.configure_trading_symbol(None, available_tickers)
            position_size = ui_manager.configure_trading_position(args.position, stock_symbol)
        
        # Configure trading strategy
        trading_mode = ui_manager.configure_trading_strategy(args.trading_mode)
        
        # Update config with final parameters
        config.update({
            'stock_symbol': stock_symbol,
            'initial_budget': initial_budget,
            'position_size': position_size,
            'trading_mode': trading_mode
        })
        
        print("INITIALIZING: Data manager...")
        data_manager = DataManager(redis_client, stock_symbol, config)
        
        print("INITIALIZING: Trading executor...")
        base_symbol = stock_symbol[2:] if stock_symbol.startswith('S_') else stock_symbol
        trading_executor = TradingExecutor(base_symbol, initial_budget, position_size, trading_mode)
        
        print("INITIALIZING: Report generator...")
        report_generator = ReportGenerator(data_manager)
        
        print("CREATING: Simulation engine...")
        simulator = SimulatorEngine(config, data_manager, trading_executor, report_generator)
        
        print("\nSTARTING: Simulation...")
        success = simulator.run_complete_simulation()
        
        if success:
            ui_manager.offer_post_simulation_options(simulator)
        
    except KeyboardInterrupt:
        print("\nSHUTDOWN: Simulation terminated by user")
    except Exception as e:
        print(f"ERROR: Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


