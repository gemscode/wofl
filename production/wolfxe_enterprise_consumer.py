#!/usr/bin/env python3
"""
WolfXE Enterprise Trading System - Redis Stream Consumer
Professional-grade trading recommendation engine with Redis integration
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
warnings.filterwarnings('ignore')

# Import your WolfXE system
from rw_wolfxe.production.trader_wolfxe import WolfXEnhancedTradingSystem

class WolfXEEnterpriseConsumer:
    """Enterprise Redis stream consumer for WolfXE trading recommendations"""
    
    def __init__(self, redis_host='137.220.33.170', redis_port=6379, 
                 redis_password=None, stock_symbol=None, 
                 initial_budget=None, position_size=None, simulation_mode=False):
        
        # Store simulation mode
        self.simulation_mode = simulation_mode
        
        # Redis connection configuration
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=10,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test Redis connection first
        self.test_redis_connection()
        
        # Discover available tickers and configure trading parameters
        self.available_tickers = self.discover_available_tickers()
        self.stock_symbol = self.configure_trading_symbol(stock_symbol)
        
        # If in simulation mode, ensure we're using simulation symbol
        if self.simulation_mode and not self.stock_symbol.startswith('S_'):
            base_symbol = self.stock_symbol
            self.stock_symbol = f"S_{base_symbol}"
            print(f"[SIMULATION] Using simulation symbol: {self.stock_symbol} (base: {base_symbol})")
        
        self.initial_budget = self.configure_trading_budget(initial_budget)
        self.current_position = self.configure_trading_position(position_size)
        self.current_budget = self.initial_budget
        
        # Enhanced data management for OHLCV conversion
        self.price_history = deque(maxlen=200)
        self.volume_history = deque(maxlen=200)
        self.last_processed_id = {}
        self.current_minute_bar = {}  # For aggregating timesale to OHLCV
        
        # Trading execution tracking
        self.auto_trading_enabled = False  # Start with manual mode
        self.executed_trades = []
        self.portfolio_value_history = []
        self.last_trade_price = None
        self.position_entry_price = None
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        
        # ENHANCED: Trading days tracking
        self.trading_days = set()  # Track unique trading days
        self.system_start_time = datetime.now()
        self.first_data_timestamp = None
        self.min_trading_days = 5  # Minimum days before trading
        
        # WolfXE system initialization
        self.wolfxe_system = None
        self.is_system_ready = False
        
        # Stream configuration
        self.streams = [
            f"ticker:{self.stock_symbol}",
            "ticker:ALL",
            "type:timesale"
        ]
        
        # Consumer group configuration
        self.consumer_group = f"wolfxe_enterprise_{self.stock_symbol}"
        self.consumer_name = f"wolfxe_consumer_{int(time.time())}"
        
        self.log_system_initialization()

    def test_redis_connection(self):
        """Test Redis connection with detailed error handling"""
        try:
            pong = self.redis_client.ping()
            print(f"[CONNECTION] Redis server status: {pong}")
            return True
        except redis.exceptions.AuthenticationError:
            print("[ERROR] Redis authentication failed")
            print("Please check your Redis password in .redis_passwd file or --redis-password argument")
            sys.exit(1)
        except redis.exceptions.ConnectionError as e:
            print(f"[ERROR] Could not connect to Redis: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Redis connection error: {e}")
            sys.exit(1)

    def has_minimum_days_of_data(self, min_days=5):
        """Check if we have at least minimum days of trading data"""
        if not self.price_history:
            return False
        
        # Extract unique days from price_history timestamps
        unique_days = set()
        for entry in self.price_history:
            if 'timestamp' in entry:
                unique_days.add(entry['timestamp'].date())
            else:
                # If no timestamp, assume insufficient data
                return False
        
        return len(unique_days) >= min_days

    def load_historical_data_from_redis(self):
        """Load existing historical data from Redis streams"""
        try:
            # Read MORE messages to get enough days - increase from 2000 to 5000
            stream_name = f"ticker:{self.stock_symbol}"
            messages = self.redis_client.xrevrange(stream_name, count=5000)
            
            print(f"[DATA LOAD] Found {len(messages)} historical messages in Redis")
            
            if not messages:
                print(f"[ERROR] No messages found in stream {stream_name}")
                return
            
            # Process messages in chronological order (reverse the list)
            processed_count = 0
            for message_id, fields in reversed(messages):
                if fields.get('type') == 'timesale':
                    # Extract timestamp from Redis message ID
                    timestamp_ms = int(message_id.split('-')[0])
                    
                    # Create a modified message with the original timestamp
                    modified_fields = fields.copy()
                    modified_fields['date'] = str(timestamp_ms)
                    
                    # Process the historical data point
                    if self.process_market_data(modified_fields):
                        processed_count += 1
                    
                    # Stop when we have enough days
                    if len(self.trading_days) >= 10:  # Get extra days to be safe
                        break
            
            print(f"[DATA LOAD] Processed {processed_count} messages")
            print(f"[DATA LOAD] Result: {len(self.trading_days)} trading days, {len(self.price_history)} data points")
            
            # Debug: Show the trading days we found
            if self.trading_days:
                sorted_days = sorted(list(self.trading_days))
                print(f"[DATA LOAD] Trading days found: {sorted_days[:5]}...{sorted_days[-5:] if len(sorted_days) > 5 else ''}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load historical data from Redis: {e}")
            import traceback
            traceback.print_exc()

    def discover_available_tickers(self):
        """Discover available ticker symbols from Redis streams with enhanced error handling"""
        print("DISCOVERING AVAILABLE MARKET DATA...")
        print("=" * 80)
        
        try:
            # Scan for ticker streams with better error handling
            ticker_streams = []
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(cursor, match="ticker:*", count=100)
                for key in keys:
                    if key != "ticker:ALL" and self.redis_client.type(key) == 'stream':
                        ticker = key.replace("ticker:", "")
                        if ticker and len(ticker) <= 10:
                            ticker_streams.append(ticker)
                if cursor == 0:
                    break
            
            if not ticker_streams:
                print("[WARNING] No ticker streams found")
                return {}
            
            # Get stream information
            available_tickers = {}
            for ticker in sorted(ticker_streams):
                try:
                    stream_info = self.redis_client.xinfo_stream(f"ticker:{ticker}")
                    message_count = stream_info['length']
                    
                    # Get latest message to check recency
                    latest_messages = self.redis_client.xrevrange(f"ticker:{ticker}", count=1)
                    if latest_messages:
                        latest_timestamp = int(latest_messages[0][0].split('-')[0])
                        time_diff = int(time.time() * 1000) - latest_timestamp
                        minutes_ago = time_diff // (1000 * 60)
                        
                        available_tickers[ticker] = {
                            'message_count': message_count,
                            'last_update_minutes': minutes_ago,
                            'active': minutes_ago < 60
                        }
                except Exception as e:
                    print(f"[WARNING] Could not get info for ticker {ticker}: {e}")
                    continue
            
            # Display discovered tickers
            if available_tickers:
                print("AVAILABLE TICKER SYMBOLS:")
                print("-" * 80)
                print(f"{'SYMBOL':<10} {'MESSAGES':<12} {'LAST UPDATE':<15} {'STATUS':<10}")
                print("-" * 80)
                
                for ticker, info in available_tickers.items():
                    status = "ACTIVE" if info['active'] else "INACTIVE"
                    last_update = f"{info['last_update_minutes']}m ago" if info['last_update_minutes'] < 1440 else ">1d ago"
                    print(f"{ticker:<10} {info['message_count']:<12} {last_update:<15} {status:<10}")
                
                print("-" * 80)
                print(f"Total tickers discovered: {len(available_tickers)}")
            
            return available_tickers
            
        except Exception as e:
            print(f"[ERROR] Failed to discover tickers: {e}")
            return {}

    def configure_trading_symbol(self, provided_symbol):
        """Configure trading symbol with user interaction if needed"""
        if provided_symbol:
            symbol = provided_symbol.upper()
            # In simulation mode, convert to S_ format if needed
            if self.simulation_mode and not symbol.startswith('S_'):
                symbol = f"S_{symbol}"
            
            if symbol in self.available_tickers:
                print(f"SELECTED SYMBOL: {symbol}")
                return symbol
            else:
                print(f"WARNING: Provided symbol '{symbol}' not found in available data")
        
        if not self.available_tickers:
            print("ERROR: No tickers available. Please start the stock publisher first.")
            sys.exit(1)
        
        # Interactive symbol selection
        print("\nSYMBOL SELECTION:")
        print("=" * 40)
        
        # Filter tickers based on simulation mode
        if self.simulation_mode:
            available_symbols = [ticker for ticker in self.available_tickers.keys() if ticker.startswith('S_')]
            print("SIMULATION MODE: Showing simulation tickers only")
        else:
            available_symbols = [ticker for ticker in self.available_tickers.keys() if not ticker.startswith('S_')]
            print("LIVE MODE: Showing live tickers only")
        
        active_tickers = [ticker for ticker in available_symbols 
                         if self.available_tickers[ticker]['active']]
        
        if active_tickers:
            print("ACTIVE TICKERS (recommended):")
            for i, ticker in enumerate(active_tickers, 1):
                info = self.available_tickers[ticker]
                print(f"  {i}. {ticker} ({info['message_count']} messages, {info['last_update_minutes']}m ago)")
        
        print(f"\nALL AVAILABLE TICKERS: {', '.join(available_symbols)}")
        
        while True:
            try:
                if active_tickers:
                    user_input = input(f"\nSelect ticker (1-{len(active_tickers)} or enter symbol): ").strip()
                    
                    if user_input.isdigit():
                        selection = int(user_input)
                        if 1 <= selection <= len(active_tickers):
                            selected_symbol = active_tickers[selection - 1]
                            print(f"SELECTED: {selected_symbol}")
                            return selected_symbol
                        else:
                            print(f"Invalid selection. Please enter 1-{len(active_tickers)}")
                            continue
                else:
                    user_input = input("Enter ticker symbol: ").strip()
                
                # Direct symbol entry
                symbol = user_input.upper()
                if self.simulation_mode and not symbol.startswith('S_'):
                    symbol = f"S_{symbol}"
                
                if symbol in self.available_tickers:
                    print(f"SELECTED: {symbol}")
                    return symbol
                else:
                    print(f"Symbol '{symbol}' not available. Available: {', '.join(available_symbols)}")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                sys.exit(0)
            except Exception as e:
                print(f"Invalid input: {e}")

    def configure_trading_budget(self, provided_budget):
        """Configure trading budget with user interaction if needed"""
        if provided_budget is not None and provided_budget > 0:
            print(f"TRADING BUDGET: ${provided_budget:,.2f}")
            return provided_budget
        
        print("\nBUDGET CONFIGURATION:")
        print("=" * 40)
        print("Please specify your trading budget for this session.")
        print("This represents the maximum capital available for trading.")
        
        while True:
            try:
                budget_input = input("Enter trading budget ($): ").strip().replace(',', '').replace('$', '')
                
                if not budget_input:
                    print("Budget is required. Please enter a valid amount.")
                    continue
                
                budget = float(budget_input)
                
                if budget <= 0:
                    print("Budget must be greater than zero.")
                    continue
                
                if budget < 1000:
                    confirm = input(f"Budget ${budget:,.2f} is quite low. Continue? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                print(f"TRADING BUDGET SET: ${budget:,.2f}")
                return budget
                
            except ValueError:
                print("Invalid amount. Please enter a numeric value.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                sys.exit(0)

    def configure_trading_position(self, provided_position):
        """Configure current position with user interaction if needed"""
        if provided_position is not None:
            print(f"CURRENT POSITION: {provided_position} shares")
            return provided_position
        
        print("\nPOSITION CONFIGURATION:")
        print("=" * 40)
        print(f"Please specify your current position in {self.stock_symbol}.")
        print("Enter 0 if you don't currently hold any shares.")
        
        while True:
            try:
                position_input = input(f"Current {self.stock_symbol} position (shares): ").strip()
                
                if not position_input:
                    position_input = "0"
                
                position = int(position_input)
                
                if position < 0:
                    print("Position cannot be negative. Enter 0 for no position.")
                    continue
                
                print(f"CURRENT POSITION SET: {position} shares")
                return position
                
            except ValueError:
                print("Invalid input. Please enter a whole number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                sys.exit(0)

    def log_system_initialization(self):
        """Log system initialization with enterprise formatting"""
        print("\n" + "=" * 80)
        if self.simulation_mode:
            print("WOLFXE ENTERPRISE TRADING SYSTEM - SIMULATION MODE")
        else:
            print("WOLFXE ENTERPRISE TRADING SYSTEM - REDIS CONSUMER")
        print("=" * 80)
        print(f"SYMBOL:           {self.stock_symbol}")
        if self.simulation_mode:
            print(f"MODE:             SIMULATION")
        print(f"INITIAL BUDGET:   ${self.initial_budget:,.2f}")
        print(f"CURRENT POSITION: {self.current_position} shares")
        print(f"REDIS ENDPOINT:   {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}")
        print(f"CONSUMER GROUP:   {self.consumer_group}")
        print(f"MIN TRADING DAYS: {self.min_trading_days}")
        print("=" * 80)

    def setup_consumer_groups(self):
        """Configure Redis consumer groups for reliable stream processing"""
        for stream in self.streams:
            try:
                self.redis_client.xgroup_create(
                    stream, self.consumer_group, id='0', mkstream=True
                )
                print(f"[SETUP] Consumer group created for stream: {stream}")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    print(f"[SETUP] Consumer group exists for stream: {stream}")
                else:
                    print(f"[ERROR] Consumer group setup failed for {stream}: {e}")

    def validate_redis_connection(self):
        """Validate Redis connection and stream availability"""
        try:
            response = self.redis_client.ping()
            print(f"[CONNECTION] Redis server status: {response}")
            
            # Validate streams
            for stream in self.streams:
                try:
                    info = self.redis_client.xinfo_stream(stream)
                    print(f"[STREAM] {stream}: {info['length']} messages available")
                except redis.exceptions.ResponseError:
                    print(f"[WARNING] Stream {stream} not yet available")
            
            return True
            
        except Exception as e:
            print(f"[CRITICAL] Redis connection failed: {e}")
            return False

    def initialize_trading_system(self):
        """Initialize WolfXE trading system with configuration"""
        print("[SYSTEM] Initializing WolfXE Trading Engine...")
        
        try:
            # Use base symbol for WolfXE system (remove S_ prefix if simulation)
            base_symbol = self.stock_symbol
            if self.simulation_mode and base_symbol.startswith('S_'):
                base_symbol = base_symbol[2:]  # Remove S_ prefix
            
            self.wolfxe_system = WolfXEnhancedTradingSystem(
                base_symbol, 
                self.initial_budget
            )
            
            # Configure portfolio state
            self.wolfxe_system.portfolio['shares'] = self.current_position
            self.wolfxe_system.portfolio['cash'] = self.current_budget
            
            # ENHANCED: Load existing historical data from Redis immediately
            print("[SYSTEM] Loading existing historical data from Redis...")
            self.load_historical_data_from_redis()
            
            # Check if we have sufficient data after loading
            has_min_days = self.has_minimum_days_of_data(self.min_trading_days)
            has_min_data = len(self.price_history) >= 50
            
            print(f"[SYSTEM] Data check: {len(self.trading_days)} days (need {self.min_trading_days}), {len(self.price_history)} points (need 50)")
            
            if has_min_days and has_min_data:
                print("[SYSTEM] Sufficient data found - starting training...")
                historical_data = self.generate_historical_dataframe()
                print(f"[SYSTEM] Generated DataFrame with {len(historical_data)} rows")
                
                if len(historical_data) >= 100:  # Ensure we have enough data for training
                    system_ready = self.wolfxe_system.load_or_train_all_agents(historical_data)
                    self.is_system_ready = system_ready
                    print(f"[SYSTEM] Trading engine status: {'READY' if system_ready else 'PENDING'}")
                    print(f"[SYSTEM] Data loaded: {len(self.trading_days)} trading days, {len(self.price_history)} data points")
                else:
                    print(f"[SYSTEM] Insufficient DataFrame size: {len(historical_data)} rows (need 100+)")
                    self.is_system_ready = False
            else:
                print(f"[SYSTEM] Insufficient historical data after loading from Redis")
                print(f"[SYSTEM] Current: {len(self.trading_days)} trading days (need {self.min_trading_days}), {len(self.price_history)} data points (need 50)")
                self.is_system_ready = False
            
        except Exception as e:
            print(f"[CRITICAL] Trading system initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.is_system_ready = False

    def process_market_data(self, message_data):
        """Process incoming market data from Redis streams and convert to OHLCV format"""
        try:
            # Extract timesale data
            last_price = float(message_data.get('last', 0))
            volume = int(message_data.get('size', 0))
            timestamp_ms = int(message_data.get('date', int(time.time() * 1000)))
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            
            if last_price <= 0:
                return False
            
            # ENHANCED: Track trading days and first data timestamp
            trading_day = timestamp.date()
            old_day_count = len(self.trading_days)
            self.trading_days.add(trading_day)
            new_day_count = len(self.trading_days)
            
            # Debug: Print when we add a new trading day
            if new_day_count > old_day_count:
                print(f"[DATA DEBUG] New trading day added: {trading_day} (total: {new_day_count})")
            
            if self.first_data_timestamp is None:
                self.first_data_timestamp = timestamp
            
            # For initial data collection, add every tick as a data point
            market_tick = {
                'timestamp': timestamp,
                'Open': last_price,
                'High': last_price,
                'Low': last_price,
                'Close': last_price,
                'Volume': volume,
                'price': last_price,
                'volume': volume,
                'symbol': message_data.get('symbol', ''),
                'exchange': message_data.get('exch', '')
            }
            
            self.price_history.append(market_tick)
            self.volume_history.append(volume)
            
            return True
            
        except (ValueError, KeyError) as e:
            print(f"[ERROR] Market data processing failed: {e}")
            print(f"[ERROR] Message data: {message_data}")
            return False

    def generate_historical_dataframe(self):
        """Generate historical market data DataFrame with proper OHLCV format"""
        if len(self.price_history) < 10:
            return pd.DataFrame()
        
        # Convert price history to DataFrame with proper OHLCV structure
        market_data = []
        for price_point in self.price_history:
            market_data.append({
                'timestamp': price_point['timestamp'],
                'Open': price_point.get('Open', price_point['price']),
                'High': price_point.get('High', price_point['price']),
                'Low': price_point.get('Low', price_point['price']),
                'Close': price_point.get('Close', price_point['price']),
                'Volume': price_point.get('Volume', price_point.get('volume', 1000))
            })
        
        df = pd.DataFrame(market_data)
        df.set_index('timestamp', inplace=True)
        
        # Ensure proper data types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        return df

    def calculate_balanced_confidence(self, decision_analysis):
        """Calculate balanced confidence score based on multiple factors"""
        base_confidence = 0.3  # Start lower
        
        # Agent consensus factor
        base_decisions = decision_analysis.get('base_decisions', {})
        decisions = [base_decisions.get('a1', 'HOLD'), 
                    base_decisions.get('a2', 'HOLD'), 
                    base_decisions.get('candlestick', 'HOLD')]
        
        # Count consensus
        decision_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        for decision in decisions:
            decision_counts[decision] += 1
        
        max_consensus = max(decision_counts.values())
        consensus_factor = max_consensus / 3.0  # 0.33 to 1.0
        
        # Add consensus bonus
        base_confidence += consensus_factor * 0.3
        
        # Candlestick confidence factor
        candlestick_result = decision_analysis.get('candlestick_result', {})
        candlestick_confidence = candlestick_result.get('confidence', 0.0)
        base_confidence += candlestick_confidence * 0.2
        
        # Activity pressure penalty (reduce confidence for forced trades)
        activity_pressure = decision_analysis.get('activity_pressure', 0.0)
        if activity_pressure > 0.6:
            base_confidence -= 0.2  # Reduce confidence for forced trades
        
        # Override penalty
        if decision_analysis.get('should_override', False):
            base_confidence -= 0.1
        
        # Force trade penalty
        if decision_analysis.get('force_trade', False):
            base_confidence -= 0.15
        
        return max(0.1, min(base_confidence, 0.95))  # Keep between 10% and 95%

    def execute_simulated_trade(self, action, current_price, reasoning):
        """Execute simulated trade and update portfolio with enhanced debugging"""
        print(f"[TRADE DEBUG] Attempting {action} at ${current_price:.2f}")
        print(f"[TRADE DEBUG] Auto trading enabled: {self.auto_trading_enabled}")
        print(f"[TRADE DEBUG] Current position: {self.current_position}")
        print(f"[TRADE DEBUG] Available cash: ${self.current_budget:.2f}")
        print(f"[TRADE DEBUG] Trading days: {len(self.trading_days)}, Min required: {self.min_trading_days}")
        
        if not self.auto_trading_enabled:
            print(f"[TRADE DEBUG] Auto trading disabled - no execution")
            return False
        
        # ENHANCED: Check minimum trading days requirement
        if not self.has_minimum_days_of_data(self.min_trading_days):
            print(f"[TRADE DEBUG] Insufficient trading days: {len(self.trading_days)}/{self.min_trading_days} - no execution")
            return False
        
        timestamp = datetime.now()
        
        if action == 'BUY' and self.current_position == 0:
            # Calculate shares to buy (use 90% of available cash)
            available_cash = self.current_budget * 0.9
            shares_to_buy = int(available_cash // current_price)
            
            print(f"[TRADE DEBUG] BUY calculation: ${available_cash:.2f} / ${current_price:.2f} = {shares_to_buy} shares")
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.current_position = shares_to_buy
                self.current_budget -= cost
                self.position_entry_price = current_price
                self.last_trade_price = current_price
                
                trade = {
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost,
                    'reasoning': reasoning,
                    'portfolio_value': self.current_budget + (self.current_position * current_price)
                }
                
                self.executed_trades.append(trade)
                print(f"[TRADE EXECUTED] BUY {shares_to_buy} shares at ${current_price:.2f} | Cost: ${cost:.2f}")
                return True
            else:
                print(f"[TRADE DEBUG] Cannot buy - insufficient funds or shares_to_buy = 0")
        
        elif action == 'SELL' and self.current_position > 0:
            shares_to_sell = self.current_position
            proceeds = shares_to_sell * current_price
            
            print(f"[TRADE DEBUG] SELL calculation: {shares_to_sell} shares × ${current_price:.2f} = ${proceeds:.2f}")
            
            # Calculate P&L
            if self.position_entry_price:
                pnl = (current_price - self.position_entry_price) * shares_to_sell
                pnl_pct = ((current_price - self.position_entry_price) / self.position_entry_price) * 100
            else:
                pnl = 0
                pnl_pct = 0
            
            self.current_budget += proceeds
            self.current_position = 0
            self.total_pnl += pnl
            self.last_trade_price = current_price
            
            trade = {
                'timestamp': timestamp,
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': current_price,
                'proceeds': proceeds,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reasoning': reasoning,
                'portfolio_value': self.current_budget
            }
            
            self.executed_trades.append(trade)
            print(f"[TRADE EXECUTED] SELL {shares_to_sell} shares at ${current_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
            
            # Reset position tracking
            self.position_entry_price = None
            return True
        else:
            print(f"[TRADE DEBUG] No trade executed - conditions not met")
            print(f"[TRADE DEBUG] Action: {action}, Position: {self.current_position}")
        
        return False

    def generate_trading_recommendation(self):
        """Generate trading recommendation using WolfXE system with fixed confidence calculation"""
        if not self.is_system_ready or len(self.price_history) < 20:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'rationale': 'Insufficient market data or system not initialized',
                'current_price': self.price_history[-1]['price'] if self.price_history else 0,
                'position': self.current_position,
                'available_capital': self.current_budget,
                'system_status': 'PENDING'
            }
        
        try:
            # Generate market analysis
            historical_data = self.generate_historical_dataframe()
            current_market_state = historical_data.iloc[-1]
            
            # Execute WolfXE decision engine
            decision_analysis = self.wolfxe_system.make_enhanced_decision(
                historical_data, current_market_state
            )
            
            # Apply risk management constraints WITH force_trade flag
            final_action = self.wolfxe_system.apply_enhanced_constraints(
                decision_analysis['decision'], 
                current_market_state['Close'], 
                datetime.now().date(),
                force_trade=decision_analysis['force_trade']  # CRITICAL FIX: Pass force_trade flag
            )
            
            # Calculate BALANCED confidence (fix the 100% issue)
            confidence_score = self.calculate_balanced_confidence(decision_analysis)
            
            # Execute trade if auto-trading is enabled
            trade_executed = False
            if final_action != 'HOLD':
                trade_executed = self.execute_simulated_trade(
                    final_action, 
                    current_market_state['Close'], 
                    decision_analysis['reasoning']
                )
            
            return {
                'action': final_action,
                'confidence': confidence_score,
                'rationale': decision_analysis['reasoning'],
                'activity_pressure': decision_analysis['activity_pressure'],
                'urgency_level': decision_analysis['urgency_level'],
                'force_trade_flag': decision_analysis['force_trade'],
                'current_price': current_market_state['Close'],
                'position': self.current_position,
                'available_capital': self.current_budget,
                'agent_consensus': decision_analysis['base_decisions'],
                'system_status': 'ACTIVE',
                'trade_executed': trade_executed
            }
            
        except Exception as e:
            print(f"[ERROR] Recommendation generation failed: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'rationale': f'System error: {str(e)}',
                'current_price': self.price_history[-1]['price'] if self.price_history else 0,
                'position': self.current_position,
                'available_capital': self.current_budget,
                'system_status': 'ERROR'
            }

    def display_trading_recommendation(self, recommendation):
        """Display formatted trading recommendation with enhanced portfolio tracking"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print("\n" + "=" * 80)
        print(f"WOLFXE ENTERPRISE TRADING RECOMMENDATION | {timestamp}")
        print("=" * 80)
        
        # Market Information
        print("MARKET INFORMATION:")
        print(f"  Symbol:              {self.stock_symbol}")
        print(f"  Current Price:       ${recommendation['current_price']:.2f}")
        print(f"  System Status:       {recommendation['system_status']}")
        if self.simulation_mode:
            print(f"  Mode:                SIMULATION")
        print("")
        
        # ENHANCED: AUTOMATIC TRADING STATUS with days tracking
        print("AUTOMATIC TRADING STATUS:")
        print(f"  Auto Trading:        {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
        print(f"  Trading Days:        {len(self.trading_days)} days (min: {self.min_trading_days})")
        
        # Calculate data collection duration
        if self.first_data_timestamp:
            data_duration = datetime.now() - self.first_data_timestamp
            hours = int(data_duration.total_seconds() // 3600)
            minutes = int((data_duration.total_seconds() % 3600) // 60)
            print(f"  Data Collection:     {hours}h {minutes}m")
        
        print(f"  Original Budget:     ${self.initial_budget:,.2f}")
        print(f"  Current Position:    {recommendation['position']} shares")
        print(f"  Available Cash:      ${recommendation['available_capital']:,.2f}")
        
        # Calculate current portfolio value
        current_portfolio_value = recommendation['available_capital']
        if recommendation['position'] > 0:
            position_value = recommendation['position'] * recommendation['current_price']
            current_portfolio_value += position_value
            print(f"  Position Value:      ${position_value:,.2f}")
            if self.position_entry_price:
                unrealized_pnl = (recommendation['current_price'] - self.position_entry_price) * recommendation['position']
                unrealized_pnl_pct = ((recommendation['current_price'] - self.position_entry_price) / self.position_entry_price) * 100
                print(f"  Entry Price:         ${self.position_entry_price:.2f}")
                print(f"  Unrealized P&L:      ${unrealized_pnl:+.2f} ({unrealized_pnl_pct:+.1f}%)")
        
        print(f"  Total Portfolio:     ${current_portfolio_value:,.2f}")
        total_return = ((current_portfolio_value - self.initial_budget) / self.initial_budget) * 100
        print(f"  Total Return:        ${current_portfolio_value - self.initial_budget:+,.2f} ({total_return:+.1f}%)")
        print(f"  Executed Trades:     {len(self.executed_trades)}")
        
        # ENHANCED: Show trading readiness status
        has_min_days = self.has_minimum_days_of_data(self.min_trading_days)
        has_min_data = len(self.price_history) >= 50
        trading_ready = has_min_days and has_min_data and self.is_system_ready
        
        print(f"  Trading Ready:       {'YES' if trading_ready else 'NO'}")
        if not trading_ready:
            if not has_min_days:
                print(f"    - Need {self.min_trading_days - len(self.trading_days)} more trading days")
            if not has_min_data:
                print(f"    - Need {50 - len(self.price_history)} more data points")
            if not self.is_system_ready:
                print(f"    - System not initialized")
        
        print("")
        
        # Trading Recommendation
        print("TRADING RECOMMENDATION:")
        print(f"  Action:              {recommendation['action']}")
        print(f"  Confidence Level:    {recommendation['confidence']:.1%}")
        print(f"  Rationale:           {recommendation['rationale']}")
        if recommendation.get('trade_executed', False):
            print(f"  Trade Status:        EXECUTED")
        else:
            print(f"  Trade Status:        RECOMMENDATION ONLY")
        print("")
        
        # Advanced Analytics
        if 'activity_pressure' in recommendation:
            print("ADVANCED ANALYTICS:")
            print(f"  Activity Pressure:   {recommendation['activity_pressure']:.2f}")
            print(f"  Urgency Level:       {recommendation['urgency_level']}")
            
            if recommendation.get('force_trade_flag', False):
                print("  Force Trade:         TRIGGERED")
            
            if 'agent_consensus' in recommendation:
                consensus = recommendation['agent_consensus']
                print(f"  Agent Consensus:     A1({consensus.get('a1', 'N/A')}) | A2({consensus.get('a2', 'N/A')}) | Pattern({consensus.get('candlestick', 'N/A')})")
        
        print("=" * 80)
        
        # Show recent trades if any
        if self.executed_trades and len(self.executed_trades) > 0:
            print("RECENT TRADES:")
            for trade in self.executed_trades[-3:]:  # Show last 3 trades
                trade_time = trade['timestamp'].strftime('%H:%M:%S')
                if trade['action'] == 'BUY':
                    print(f"  {trade_time} | BUY  {trade['shares']} @ ${trade['price']:.2f} | Cost: ${trade['cost']:.2f}")
                else:
                    print(f"  {trade_time} | SELL {trade['shares']} @ ${trade['price']:.2f} | P&L: ${trade['pnl']:+.2f}")
            print("=" * 80)

    def read_market_streams(self):
        """Read market data from Redis streams"""
        try:
            # Read from configured streams
            stream_messages = self.redis_client.xreadgroup(
                self.consumer_group,
                self.consumer_name,
                {stream: '>' for stream in self.streams},
                count=10,
                block=1000
            )
            
            processed_messages = 0
            
            for stream, messages in stream_messages:
                for message_id, fields in messages:
                    # Process relevant market data
                    if fields.get('type') == 'timesale' and fields.get('symbol') == self.stock_symbol:
                        if self.process_market_data(fields):
                            processed_messages += 1
                    
                    # Acknowledge message processing
                    self.redis_client.xack(self.consumer_group, stream, message_id)
            
            return processed_messages > 0
            
        except redis.exceptions.ResponseError as e:
            if "NOGROUP" in str(e):
                print("[WARNING] Consumer group not found, initializing...")
                self.setup_consumer_groups()
            else:
                print(f"[ERROR] Stream read operation failed: {e}")
            return False
        except Exception as e:
            print(f"[CRITICAL] Stream processing error: {e}")
            return False

    def execute_trading_loop(self):
        """Execute main trading recommendation loop"""
        print("STARTING WOLFXE ENTERPRISE TRADING SYSTEM")
        print("Processing interval: 5 seconds")
        
        # Validate system prerequisites
        if not self.validate_redis_connection():
            return
        
        # Configure stream processing
        self.setup_consumer_groups()
        
        # Initialize trading engine
        self.initialize_trading_system()
        
        print("Main trading loop initiated...")
        
        last_recommendation_timestamp = 0
        
        try:
            while True:
                current_timestamp = time.time()
                
                # Process incoming market data
                new_data_available = self.read_market_streams()
                
                # Generate recommendations at 5-second intervals or when new data arrives
                if (current_timestamp - last_recommendation_timestamp >= 5.0 or new_data_available):
                    
                    # Re-initialize system if sufficient data is available
                    if not self.is_system_ready and len(self.price_history) >= 50 and self.has_minimum_days_of_data(self.min_trading_days):
                        self.initialize_trading_system()
                    
                    # Generate and display recommendation
                    recommendation = self.generate_trading_recommendation()
                    self.display_trading_recommendation(recommendation)
                    
                    last_recommendation_timestamp = current_timestamp
                
                # System processing interval
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] WolfXE Enterprise System terminated by user")
            self.print_final_summary()
        except Exception as e:
            print(f"[CRITICAL] System error: {e}")
            import traceback
            traceback.print_exc()

    def print_final_summary(self):
        """Print final trading summary"""
        if self.executed_trades:
            print("\n" + "=" * 80)
            print("FINAL TRADING SUMMARY")
            print("=" * 80)
            
            current_price = self.price_history[-1]['price'] if self.price_history else 0
            current_portfolio_value = self.current_budget
            if self.current_position > 0:
                current_portfolio_value += self.current_position * current_price
            
            total_return = ((current_portfolio_value - self.initial_budget) / self.initial_budget) * 100
            
            print(f"Initial Budget:       ${self.initial_budget:,.2f}")
            print(f"Final Portfolio:      ${current_portfolio_value:,.2f}")
            print(f"Total Return:         ${current_portfolio_value - self.initial_budget:+,.2f} ({total_return:+.1f}%)")
            print(f"Total Trades:         {len(self.executed_trades)}")
            print(f"Trading Days:         {len(self.trading_days)}")
            
            # Calculate win rate
            completed_trades = [t for t in self.executed_trades if t['action'] == 'SELL']
            if completed_trades:
                winning_trades = [t for t in completed_trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(completed_trades) * 100
                print(f"Win Rate:             {win_rate:.1f}%")
            
            print("=" * 80)

def main():
    """Main application entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='WolfXE Enterprise Trading System - Redis Consumer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--symbol', 
                       help='Stock symbol for trading analysis (will prompt if not provided)')
    parser.add_argument('--budget', type=float, 
                       help='Initial trading budget (will prompt if not provided)')
    parser.add_argument('--position', type=int, 
                       help='Current stock position in shares (will prompt if not provided)')
    parser.add_argument('--redis-password', 
                       help='Redis authentication password (will read from .redis_passwd if not provided)')
    parser.add_argument('--redis-port', type=int, default=6379, 
                       help='Redis server port')
    parser.add_argument('--simulation', action='store_true',
                       help='Run in simulation mode (uses S_ prefixed symbols)')
    parser.add_argument('--auto-trade', action='store_true',
                       help='Enable automatic trade execution')
    parser.add_argument('--min-days', type=int, default=5,
                       help='Minimum trading days before allowing trades')
    
    args = parser.parse_args()
    
    # Load Redis password if not provided
    redis_password = args.redis_password
    if not redis_password:
        try:
            with open('.redis_passwd', 'r') as f:
                redis_password = f.read().strip()
            print("[CONFIG] Redis password loaded from .redis_passwd file")
        except FileNotFoundError:
            print("[ERROR] Redis password not provided and .redis_passwd file not found")
            print("Please either:")
            print("  1. Use --redis-password argument")
            print("  2. Create .redis_passwd file with your password")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to read .redis_passwd file: {e}")
            sys.exit(1)
    
    # Initialize and execute trading system
    trading_system = WolfXEEnterpriseConsumer(
        redis_host='137.220.33.170',
        redis_port=args.redis_port,
        redis_password=redis_password,
        stock_symbol=args.symbol,
        initial_budget=args.budget,
        position_size=args.position,
        simulation_mode=args.simulation
    )
    
    # Configure minimum trading days
    trading_system.min_trading_days = args.min_days
    
    # Enable auto-trading if requested
    if args.auto_trade:
        trading_system.auto_trading_enabled = True
        print("[AUTO-TRADE] Automatic trading enabled")
    
    trading_system.execute_trading_loop()

if __name__ == "__main__":
    main()

