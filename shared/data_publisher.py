import redis
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import threading
import time
import os
import asyncio
import nest_asyncio
from pathlib import Path
from dotenv import load_dotenv
import pytz

# Enable nested async for IB integration
nest_asyncio.apply()

# Conditional IB import - only import if available
try:
    from ib_insync import *
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logging.warning("ib_insync not available. Install with: pip install ib_insync")

# Load environment variables from .env file one level up from shared directory
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

class DataPublisher:
    """
    Enhanced Data Publisher with Interactive Brokers support
    Handles publishing data to various Redis streams for different consumers.
    Supports multiple data sources: Alpha Vantage (--alpha) and Interactive Brokers (--ib)
    """

    def __init__(self, host=None, port=None, db=None, ib_host='127.0.0.1', ib_port=4002, ib_client_id=3):
        # Redis configuration
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', '6379'))
        self.db = db or int(os.getenv('REDIS_DB', '0'))
        
        # Initialize Redis client
        redis_password = self._read_redis_password()
        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=redis_password,
            decode_responses=True
        )
        
        # IB configuration
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.ib_client_id = ib_client_id
        self.ib = None
        self.ib_connected = False
        
        # Timezone for US markets
        self.us_eastern = pytz.timezone('US/Eastern')
        
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._heartbeat_thread = None
        
        # Data size limits to prevent timeouts
        self.MAX_DATA_POINTS = {
            '5 secs': 1500,   # ~2 hours of 5-second data
            '10 secs': 3000,  # ~8 hours of 10-second data
            '30 secs': 8000,  # ~67 hours of 30-second data
            '1 min': 10000,   # ~7 days of 1-minute data
            '5 mins': 20000,  # ~70 days of 5-minute data
            '15 mins': 30000, # ~312 days of 15-minute data
            '1 hour': 50000,  # ~5+ years of hourly data
            '1 day': 100000   # Many years of daily data
        }
        
        # Log connection details
        self.logger.info(f"DataPublisher initialized with timezone support")
        self.logger.info(f"Redis: {self.host}:{self.port}")
        self.logger.info(f"IB Gateway: {self.ib_host}:{self.ib_port} (Available: {IB_AVAILABLE})")

    def _read_redis_password(self):
        """Read Redis password from file"""
        try:
            with open('.redis_passwd', 'r') as f:
                return f.read().strip() or None
        except FileNotFoundError:
            return None

    def _get_last_business_days_with_timezone(self, num_days: int = 5) -> tuple:
        """
        Calculate the last N business days with proper US/Eastern timezone handling.
        
        Args:
            num_days: Number of business days to go back
            
        Returns:
            Tuple of (start_date, end_date) as timezone-aware datetime objects
        """
        # Get current time in US/Eastern
        now_et = datetime.now(self.us_eastern)
        
        # Find the most recent market close (4 PM ET)
        if now_et.weekday() < 5:  # Monday = 0, Friday = 4
            # It's a weekday
            if now_et.hour >= 16:  # After 4 PM ET
                end_date = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            else:
                # Before market close, use previous business day
                end_date = now_et - timedelta(days=1)
                while end_date.weekday() >= 5:  # Skip weekends
                    end_date -= timedelta(days=1)
                end_date = end_date.replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            # It's weekend, go back to Friday
            days_to_subtract = now_et.weekday() - 4  # Days back to Friday
            if now_et.weekday() == 6:  # Sunday
                days_to_subtract = 2
            else:  # Saturday
                days_to_subtract = 1
            
            end_date = now_et - timedelta(days=days_to_subtract)
            end_date = end_date.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Calculate the start date by going back the required number of business days
        start_date = end_date
        business_days_counted = 0
        
        while business_days_counted < num_days:
            start_date -= timedelta(days=1)
            if start_date.weekday() < 5:  # Is a weekday
                business_days_counted += 1
        
        # Set start time to market open (9:30 AM ET)
        start_date = start_date.replace(hour=9, minute=30, second=0, microsecond=0)
        
        return start_date, end_date

    def _calculate_optimal_duration_and_bar_size(self, days: int, requested_bar_size: str) -> tuple:
        """
        Calculate optimal duration and potentially adjust bar size to stay within data limits.
        
        Args:
            days: Number of business days requested
            requested_bar_size: Requested bar size
            
        Returns:
            Tuple of (adjusted_bar_size, duration_str, end_datetime_str, warning)
        """
        start_date, end_date = self._get_last_business_days_with_timezone(days)
        
        # Calculate trading hours per day (6.5 hours = 23,400 seconds)
        trading_seconds_per_day = 23400
        
        # Parse bar size to seconds
        bar_size_seconds = self._parse_bar_size_to_seconds(requested_bar_size)
        
        # Calculate estimated data points
        estimated_points = (days * trading_seconds_per_day) // bar_size_seconds
        
        # Check if we exceed limits
        max_points = self.MAX_DATA_POINTS.get(requested_bar_size, 5000)
        adjusted_bar_size = requested_bar_size
        warning = None
        
        if estimated_points > max_points:
            # Find a larger bar size that fits
            bar_size_options = [
                ('5 secs', 5), ('10 secs', 10), ('30 secs', 30), 
                ('1 min', 60), ('5 mins', 300), ('15 mins', 900), 
                ('1 hour', 3600), ('1 day', 86400)
            ]
            
            for bar_size_name, seconds in bar_size_options:
                estimated = (days * trading_seconds_per_day) // seconds
                if estimated <= self.MAX_DATA_POINTS.get(bar_size_name, 5000):
                    adjusted_bar_size = bar_size_name
                    warning = f"Bar size adjusted from {requested_bar_size} to {bar_size_name} to stay within data limits"
                    break
        
        # Calculate total calendar days between start and end
        total_days = (end_date - start_date).days + 1
        
        # Format for IB API
        if total_days <= 30:
            duration_str = f"{total_days} D"
        elif total_days <= 365:
            duration_str = f"{total_days} D"
        else:
            weeks = total_days // 7
            if weeks <= 52:
                duration_str = f"{weeks} W"
            else:
                months = total_days // 30
                duration_str = f"{months} M"
        
        # Format end time with proper timezone specification
        end_datetime_str = end_date.strftime('%Y%m%d %H:%M:%S US/Eastern')
        
        print(f"Business day calculation:")
        print(f"    Requested: {days} business days, {requested_bar_size}")
        print(f"    Start date: {start_date} ({start_date.tzname()})")
        print(f"    End date: {end_date} ({end_date.tzname()})")
        print(f"    Adjusted bar size: {adjusted_bar_size}")
        print(f"    Duration: {duration_str}")
        print(f"    End time (IB format): {end_datetime_str}")
        print(f"    Estimated data points: {estimated_points:,}")
        if warning:
            print(f"    WARNING: {warning}")
        
        return adjusted_bar_size, duration_str, end_datetime_str, warning

    def _parse_bar_size_to_seconds(self, bar_size: str) -> int:
        """Parse IB bar size string to seconds."""
        bar_size_map = {
            '1 secs': 1, '5 secs': 5, '10 secs': 10, '15 secs': 15, '30 secs': 30,
            '1 min': 60, '2 mins': 120, '3 mins': 180, '5 mins': 300, 
            '10 mins': 600, '15 mins': 900, '20 mins': 1200, '30 mins': 1800,
            '1 hour': 3600, '2 hours': 7200, '4 hours': 14400, '1 day': 86400
        }
        return bar_size_map.get(bar_size, 60)  # Default to 1 minute

    async def _connect_ib(self):
        """Connect to IB Gateway with detailed logging"""
        if not IB_AVAILABLE:
            raise Exception("ib_insync not available. Install with: pip install ib_insync")
        
        if not self.ib_connected:
            try:
                print(f"Connecting to IB Gateway at {self.ib_host}:{self.ib_port}...")
                self.ib = IB()
                
                # Set error handler
                def error_handler(reqId, errorCode, errorString, contract):
                    if errorCode in [2104, 2106, 2158]:  # Connection OK messages
                        print(f"    IB Info {errorCode}: {errorString}")
                    else:
                        print(f"    IB Error {errorCode}: {errorString}")
                    self.logger.info(f"IB Message {errorCode}: {errorString}")
                
                self.ib.errorEvent += error_handler
                
                await self.ib.connectAsync(self.ib_host, self.ib_port, clientId=self.ib_client_id)
                print(f"Connected to IB Gateway")
                
                # Request market data type (3 = delayed, 1 = live with subscription)
                self.ib.reqMarketDataType(3)
                print(f"Requested delayed market data")
                
                self.ib_connected = True
                
                # Test connection with account info
                try:
                    accounts = self.ib.managedAccounts()
                    print(f"Available accounts: {accounts}")
                except Exception as e:
                    print(f"Could not get account info: {e}")
                
                self.logger.info(f"Connected to IB Gateway at {self.ib_host}:{self.ib_port}")
                
            except Exception as e:
                print(f"Failed to connect to IB Gateway: {e}")
                self.logger.error(f"Failed to connect to IB Gateway: {e}")
                raise

    async def _disconnect_ib(self):
        """Disconnect from IB Gateway"""
        if self.ib_connected and self.ib:
            self.ib.disconnect()
            self.ib_connected = False
            self.logger.info("Disconnected from IB Gateway")

    async def fetch_ib_historical_data(self, symbol: str, days: int = 5, 
                                     bar_size: str = '1 min', what_to_show: str = 'TRADES',
                                     use_rth: bool = True) -> pd.DataFrame:
        """
        Fetch historical data from IB Gateway with proper timezone and data size handling.
        
        Args:
            symbol: Stock symbol (e.g., 'META', 'AAPL')
            days: Number of business days (automatically calculates proper date range)
            bar_size: Bar size ('1 min', '5 mins', '15 mins', '1 hour', '1 day', etc.)
            what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK')
            use_rth: Use regular trading hours only
            
        Returns:
            DataFrame with OHLCV data
        """
        
        print(f"Fetching historical data for {symbol}")
        print(f"    Business days requested: {days}")
        print(f"    Requested bar size: {bar_size}")
        print(f"    What to show: {what_to_show}")
        print(f"    Regular trading hours: {use_rth}")
        
        await self._connect_ib()
        
        try:
            # Create and qualify contract
            contract = Stock(symbol, 'SMART', 'USD')
            print(f"Created contract: {contract}")
            
            print(f"Validating contract...")
            qualified_contracts = await asyncio.wait_for(
                self.ib.qualifyContractsAsync(contract),
                timeout=10
            )
            
            if not qualified_contracts:
                print(f"Could not qualify contract for {symbol}")
                return pd.DataFrame()
            
            qualified_contract = qualified_contracts[0]
            print(f"Contract qualified: {qualified_contract}")
            
            # Calculate optimal parameters with timezone support
            adjusted_bar_size, duration_str, end_datetime_str, warning = self._calculate_optimal_duration_and_bar_size(
                days, bar_size
            )
            
            if warning:
                print(f"WARNING: {warning}")
            
            # Request historical data with proper timezone format
            print(f"Requesting historical data...")
            print(f"    Duration: {duration_str}")
            print(f"    Bar size: {adjusted_bar_size}")
            print(f"    End time: {end_datetime_str}")
            
            try:
                bars = await asyncio.wait_for(
                    self.ib.reqHistoricalDataAsync(
                        qualified_contract,
                        endDateTime=end_datetime_str,
                        durationStr=duration_str,
                        barSizeSetting=adjusted_bar_size,
                        whatToShow=what_to_show,
                        useRTH=use_rth,
                        formatDate=1
                    ),
                    timeout=120  # 2 minute timeout
                )
                
                if not bars:
                    print(f"No historical data received for {symbol}")
                    print("    Possible reasons:")
                    print("      - No market data subscription")
                    print("      - Invalid date range")
                    print("      - Symbol not found")
                    print("      - Data not available for requested time period")
                    return pd.DataFrame()
                
                print(f"Received {len(bars)} bars")
                
                # Show sample of first and last bars
                if len(bars) > 0:
                    print(f"    First bar: {bars[0].date} | O:{bars[0].open} H:{bars[0].high} L:{bars[0].low} C:{bars[0].close} V:{bars[0].volume}")
                    print(f"    Last bar:  {bars[-1].date} | O:{bars[-1].open} H:{bars[-1].high} L:{bars[-1].low} C:{bars[-1].close} V:{bars[-1].volume}")
                
                # Convert to DataFrame
                data = []
                for bar in bars:
                    data.append({
                        'timestamp': pd.Timestamp(bar.date),
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume) if bar.volume != -1 else 0
                    })
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                print(f"Converted to DataFrame: {len(df)} records")
                print(f"    Date range: {df.index.min()} to {df.index.max()}")
                
                # Filter to only business days if needed
                business_days_mask = df.index.dayofweek < 5
                df_filtered = df[business_days_mask]
                
                if len(df_filtered) != len(df):
                    print(f"Filtered to business days only: {len(df_filtered)} records")
                
                return df_filtered
                
            except asyncio.TimeoutError:
                print(f"Request timed out for {symbol}")
                print("    Suggestions:")
                print("      - Try fewer days or larger bar size")
                print("      - Check if markets were open during requested period")
                print("      - Verify market data subscriptions")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    async def get_training_data_ib(self, symbol: str, days: int = 5, 
                                 bar_size: str = '1 min') -> pd.DataFrame:
        """Get training data from IB Gateway using business days calculation"""
        try:
            df = await self.fetch_ib_historical_data(symbol, days, bar_size)
            
            if df.empty:
                self.logger.warning(f"No IB training data available for {symbol}")
                return pd.DataFrame()
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in IB data for {symbol}")
                return pd.DataFrame()
            
            self.logger.info(f"Retrieved {len(df)} bars of IB training data for {symbol} ({days} business days)")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting IB training data for {symbol}: {e}")
            return pd.DataFrame()

    def get_training_data_alpha(self, symbol: str, days: int = 5) -> pd.DataFrame:
        """Get training data from Alpha Vantage (placeholder for existing implementation)"""
        self.logger.info(f"Alpha Vantage data fetching for {symbol} ({days} business days - placeholder)")
        return pd.DataFrame()

    async def get_training_data(self, symbol: str, days: int = 5, 
                              data_source: str = 'alpha', **kwargs) -> pd.DataFrame:
        """
        Universal training data fetcher - supports multiple data sources
        Always uses business days calculation.
        
        Args:
            symbol: Stock symbol
            days: Number of business days of historical data
            data_source: Data source ('alpha', 'ib')
            **kwargs: Additional arguments for specific data sources
            
        Returns:
            DataFrame with training data
        """
        symbol = symbol.upper()
        
        print(f"Getting {days} business days of training data for {symbol} from {data_source}")
        
        if data_source.lower() == 'ib':
            bar_size = kwargs.get('bar_size', '1 min')
            return await self.get_training_data_ib(symbol, days, bar_size)
        elif data_source.lower() == 'alpha':
            return self.get_training_data_alpha(symbol, days)
        else:
            self.logger.error(f"Unknown data source: {data_source}")
            return pd.DataFrame()

    def publish_training_data(self, symbol: str, data: pd.DataFrame,
                            training_type: str = 'optimization', data_source: str = 'alpha'):
        """
        Publish training data to the training stream with data source info
        CRITICAL FIX: Properly preserves timestamp column during serialization
        
        Args:
            symbol: Stock symbol
            data: Training data DataFrame
            training_type: Type of training ('optimization', 'retraining', 'validation')
            data_source: Source of the data ('alpha', 'ib')
        """
        try:
            stream_key = f"training_stream:{symbol}"
            
            # Calculate business days in the data
            if not data.empty:
                business_days_count = len(data[data.index.dayofweek < 5])
                start_date = data.index.min()
                end_date = data.index.max()
            else:
                business_days_count = 0
                start_date = None
                end_date = None
            
            message = {
                'symbol': symbol,
                'training_type': training_type,
                'data_source': data_source,
                'data_points': len(data),
                'business_days': business_days_count,
                'start_time': start_date.isoformat() if start_date else None,
                'end_time': end_date.isoformat() if end_date else None,
                'timestamp': datetime.now().isoformat(),
                'status': 'data_ready'
            }

            # Add stream entry
            self.redis_client.xadd(stream_key, message)
            
            # Store the actual data temporarily for the trainer to pick up
            data_key = f"training_data:{symbol}:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store data in chunks if it's large
            chunk_size = 1000
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i+chunk_size].copy()
                # CRITICAL FIX: Reset index to make timestamp a regular column
                chunk.reset_index(inplace=True)
                chunk_key = f"{data_key}:chunk_{i//chunk_size}"
                # Convert DataFrame chunk to JSON (now includes timestamp column)
                chunk_json = chunk.to_json(orient='records', date_format='iso')
                self.redis_client.setex(chunk_key, 3600, chunk_json)  # Expire in 1 hour

            # Store metadata about chunks
            metadata = {
                'total_chunks': (len(data) - 1) // chunk_size + 1,
                'chunk_size': chunk_size,
                'total_records': len(data),
                'business_days': business_days_count,
                'base_key': data_key,
                'data_source': data_source
            }
            self.redis_client.setex(f"{data_key}:metadata", 3600, json.dumps(metadata))

            self.logger.info(f"Published {data_source} training data for {symbol}: {len(data)} records ({business_days_count} business days)")

        except Exception as e:
            self.logger.error(f"Error publishing training data for {symbol}: {e}")

    async def start_ib_live_stream(self, symbol: str, bar_size: int = 5):
        """
        Start real-time data streaming from IB Gateway
        
        Args:
            symbol: Stock symbol
            bar_size: Real-time bar size in seconds (minimum 5)
        """
        await self._connect_ib()
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request real-time bars
            bars = self.ib.reqRealTimeBars(
                contract, 
                barSize=bar_size, 
                whatToShow='TRADES', 
                useRTH=True
            )
            
            if not bars:
                self.logger.error(f"Failed to start real-time bars for {symbol}")
                return
            
            stream_key = f"live_data_stream_ib:{symbol.lower()}"
            
            def on_bar_update(bars, has_new_bar):
                if has_new_bar:
                    bar = bars[-1]  # Get the latest bar
                    
                    bar_data = {
                        'symbol': symbol,
                        'data_source': 'interactive_brokers',
                        'timestamp': pd.Timestamp(bar.time).isoformat(),
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume) if bar.volume != -1 else 0,
                        'bar_size_seconds': bar_size,
                        'received_at': datetime.now().isoformat()
                    }
                    
                    # Publish to Redis stream
                    self.redis_client.xadd(stream_key, bar_data)
                    
                    # Keep stream size manageable
                    self.redis_client.xtrim(stream_key, maxlen=5000, approximate=True)
                    
                    self.logger.debug(f"Published live IB bar for {symbol}: ${bar.close}")
            
            # Set up the callback
            bars.updateEvent += on_bar_update
            
            self.logger.info(f"Started IB live data stream for {symbol} ({bar_size}s bars)")
            
        except Exception as e:
            self.logger.error(f"Error starting IB live data stream for {symbol}: {e}")

    def publish_live_data(self, symbol: str, market_data: Dict, data_source: str = 'alpha'):
        """
        Publish live market data to the live data stream with data source info
        
        Args:
            symbol: Stock symbol
            market_data: Market data dictionary with OHLCV data
            data_source: Source of the data ('alpha', 'ib')
        """
        try:
            stream_key = f"live_data_stream:{symbol}" if data_source == 'alpha' else f"live_data_stream_ib:{symbol}"
            message = {
                'symbol': symbol,
                'data_source': data_source,
                'timestamp': datetime.now().isoformat(),
                'data': json.dumps(market_data)
            }

            self.redis_client.xadd(stream_key, message)
            # Keep stream size manageable (last 10000 entries)
            self.redis_client.xtrim(stream_key, maxlen=10000, approximate=True)

            self.logger.debug(f"Published live {data_source} data for {symbol}")

        except Exception as e:
            self.logger.error(f"Error publishing live data for {symbol}: {e}")

    def publish_trade_signal(self, symbol: str, signal: Dict):
        """
        Publish trading signals to the signals stream
        Enhanced to include data source information
        """
        try:
            data_source = signal.get('data_source', 'unknown')
            stream_key = f"trade_signals:{symbol}"
            
            message = {
                'symbol': symbol,
                'signal_type': signal.get('action', 'unknown'),
                'data_source': data_source,
                'timestamp': datetime.now().isoformat(),
                'signal_data': json.dumps(signal)
            }

            self.redis_client.xadd(stream_key, message)
            
            # Also publish to a general signals channel for monitoring
            self.redis_client.publish('trading_signals', json.dumps({
                'symbol': symbol,
                'signal': signal,
                'data_source': data_source,
                'timestamp': datetime.now().isoformat()
            }))

            # Keep stream size manageable
            self.redis_client.xtrim(stream_key, maxlen=1000, approximate=True)

            self.logger.info(f"Published trade signal for {symbol}: {signal.get('action')} (source: {data_source})")

        except Exception as e:
            self.logger.error(f"Error publishing trade signal for {symbol}: {e}")

    def publish_model_update(self, symbol: str, model_type: str, model_path: str, metrics: Dict):
        """Publish model update notifications."""
        try:
            stream_key = "model_updates"
            message = {
                'symbol': symbol,
                'model_type': model_type,
                'model_path': model_path,
                'timestamp': datetime.now().isoformat(),
                'metrics': json.dumps(metrics),
                'status': 'updated'
            }

            self.redis_client.xadd(stream_key, message)
            
            # Also publish to a channel for immediate notifications
            self.redis_client.publish('model_updates', json.dumps({
                'symbol': symbol,
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'action': 'model_updated'
            }))

            self.logger.info(f"Published model update for {symbol}:{model_type}")

        except Exception as e:
            self.logger.error(f"Error publishing model update: {e}")

    def publish_training_status(self, symbol: str, status: str, details: Dict = None):
        """Publish training status updates."""
        try:
            stream_key = f"training_status:{symbol}"
            message = {
                'symbol': symbol,
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'details': json.dumps(details or {})
            }

            self.redis_client.xadd(stream_key, message)
            # Keep only recent status updates
            self.redis_client.xtrim(stream_key, maxlen=100, approximate=True)

            self.logger.info(f"Published training status for {symbol}: {status}")

        except Exception as e:
            self.logger.error(f"Error publishing training status: {e}")

    def publish_alert(self, alert_type: str, message: str, symbol: str = None, severity: str = 'info'):
        """Publish system alerts and notifications."""
        try:
            stream_key = "system_alerts"
            alert_data = {
                'alert_type': alert_type,
                'message': message,
                'symbol': symbol or 'system',
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            }

            self.redis_client.xadd(stream_key, alert_data)
            
            # Also publish to alerts channel for immediate notification
            self.redis_client.publish('alerts', json.dumps(alert_data))
            
            # Keep only recent alerts
            self.redis_client.xtrim(stream_key, maxlen=1000, approximate=True)

            self.logger.info(f"Published alert: {alert_type} - {message}")

        except Exception as e:
            self.logger.error(f"Error publishing alert: {e}")

    def start_heartbeat(self, component_name: str, interval: int = 30):
        """Start publishing heartbeat signals with IB connection status."""
        def heartbeat_worker():
            while not self._stop_event.is_set():
                try:
                    heartbeat_data = {
                        'component': component_name,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'alive',
                        'ib_available': IB_AVAILABLE,
                        'ib_connected': self.ib_connected
                    }

                    self.redis_client.setex(
                        f"heartbeat:{component_name}",
                        interval * 2,  # TTL is 2x interval
                        json.dumps(heartbeat_data)
                    )

                    self.logger.debug(f"Heartbeat sent for {component_name}")

                except Exception as e:
                    self.logger.error(f"Error sending heartbeat for {component_name}: {e}")

                self._stop_event.wait(interval)

        if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
            self._heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
            self._heartbeat_thread.start()
            self.logger.info(f"Started heartbeat for {component_name}")

    def stop_heartbeat(self):
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1)
        self.logger.info("Heartbeat stopped")

    def get_stream_info(self, stream_key: str) -> Dict:
        """Get information about a Redis stream."""
        try:
            info = self.redis_client.xinfo_stream(stream_key)
            return {
                'length': info.get('length', 0),
                'first_entry': info.get('first-entry'),
                'last_entry': info.get('last-entry'),
                'groups': info.get('groups', 0)
            }
        except Exception as e:
            self.logger.error(f"Error getting stream info for {stream_key}: {e}")
            return {}

    def cleanup_streams(self, max_age_hours: int = 24):
        """Clean up old stream entries to manage memory usage."""
        try:
            cutoff_timestamp = int((datetime.now().timestamp() - (max_age_hours * 3600)) * 1000)
            
            # Get all stream keys
            stream_patterns = [
                "training_stream:*",
                "live_data_stream:*",
                "live_data_stream_ib:*",
                "trade_signals:*",
                "training_status:*"
            ]

            for pattern in stream_patterns:
                keys = self.redis_client.keys(pattern)
                for key in keys:
                    # Remove entries older than cutoff
                    try:
                        self.redis_client.xtrim(key, minid=cutoff_timestamp, approximate=True)
                    except Exception as e:
                        self.logger.warning(f"Error trimming stream {key}: {e}")

            self.logger.info(f"Cleaned up streams older than {max_age_hours} hours")

        except Exception as e:
            self.logger.error(f"Error cleaning up streams: {e}")

    async def stop(self):
        """Stop all publisher operations."""
        self.stop_heartbeat()
        await self._disconnect_ib()
        self.logger.info("DataPublisher stopped")

# Async convenience function for command-line usage
async def fetch_and_publish_data(symbol: str, days: int = 5, data_source: str = 'alpha', 
                               bar_size: str = '1 min', **kwargs):
    """
    Convenience function to fetch and publish training data
    Can be called from xetrader_trainer.py with --ib or --alpha flags
    
    Args:
        symbol: Stock symbol
        days: Number of business days of data
        data_source: 'alpha' or 'ib'
        bar_size: Bar size for IB data
        **kwargs: Additional arguments
    """
    publisher = DataPublisher(**kwargs)
    
    try:
        print(f"Starting enhanced data fetch for {symbol}")
        print(f"    Source: {data_source}")
        print(f"    Business days: {days}")
        print(f"    Bar size: {bar_size}")
        
        if data_source == 'ib':
            print(f"    IB Gateway: {publisher.ib_host}:{publisher.ib_port}")
            
        data = await publisher.get_training_data(symbol, days, data_source, bar_size=bar_size)
        
        if not data.empty:
            business_days_count = len(data[data.index.dayofweek < 5])
            print(f"Data retrieved successfully!")
            print(f"    Total records: {len(data):,}")
            print(f"    Business day records: {business_days_count:,}")
            print(f"    Date range: {data.index.min()} to {data.index.max()}")
            print(f"    Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            
            # Publish to Redis
            print(f"Publishing to Redis...")
            publisher.publish_training_data(symbol, data, 'optimization', data_source)
            print(f"Successfully published {len(data):,} records for {symbol}!")
        else:
            print(f"No data retrieved for {symbol} from {data_source}")
            
    except Exception as e:
        print(f"Error in fetch_and_publish_data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await publisher.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Data Publisher with Interactive Brokers support')
    parser.add_argument('--symbol', required=True, help='Stock symbol')
    parser.add_argument('--days', type=int, default=5, help='Number of business days')
    parser.add_argument('--alpha', action='store_true', help='Use Alpha Vantage')
    parser.add_argument('--ib', action='store_true', help='Use Interactive Brokers')
    parser.add_argument('--bar-size', default='1 min', help='Bar size (will be auto-adjusted if needed)')
    parser.add_argument('--ib-host', default='127.0.0.1', help='IB Gateway host')
    parser.add_argument('--ib-port', type=int, default=4002, help='IB Gateway port')
    
    args = parser.parse_args()
    
    data_source = 'ib' if args.ib else 'alpha'
    
    print(f"Enhanced Data Publisher with Interactive Brokers Support")
    print(f"    Symbol: {args.symbol}")
    print(f"    Business days: {args.days}")
    print(f"    Data source: {data_source}")
    if data_source == 'ib':
        print(f"    Requested bar size: {args.bar_size}")
        print(f"    (Will be auto-adjusted if needed to prevent timeouts)")
    
    asyncio.run(fetch_and_publish_data(
        symbol=args.symbol,
        days=args.days,
        data_source=data_source,
        bar_size=args.bar_size,
        ib_host=args.ib_host,
        ib_port=args.ib_port
    ))

