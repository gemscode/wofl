import redis
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file one level up from shared directory
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

class DataManager:
    """
    Centralized data manager for handling all Redis connections and data operations.
    Manages market data, training data, and live trading data streams.
    Enhanced with support for multiple data sources (Alpha Vantage, Interactive Brokers).
    """

    def __init__(self, host=None, port=None, db=None):
        # Use provided parameters or fall back to environment variables
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', '6379'))
        self.db = db or int(os.getenv('REDIS_DB', '0'))
        
        # Read Redis password from file if available
        redis_password = self._read_redis_password()
        
        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=redis_password,
            decode_responses=True
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Log connection details
        self.logger.info(f"DataManager initialized with Redis connection: {self.host}:{self.port}")

    def _read_redis_password(self):
        """Read Redis password from file"""
        try:
            with open('.redis_passwd', 'r') as f:
                return f.read().strip() or None
        except FileNotFoundError:
            return None

    def get_market_data(self, symbol: str, timeframe: str = '1min',
                       start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve market data for a given symbol and timeframe.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GERN')
            timeframe: Time interval ('1min', '5min', '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            key = f"market_data:{symbol}:{timeframe}"
            if start_date and end_date:
                # Get data within date range
                start_ts = int(pd.Timestamp(start_date).timestamp())
                end_ts = int(pd.Timestamp(end_date).timestamp())
                data = self.redis_client.zrangebyscore(key, start_ts, end_ts)
            else:
                # Get all available data
                data = self.redis_client.zrange(key, 0, -1)
            
            if not data:
                self.logger.warning(f"No data found for {symbol}:{timeframe}")
                return pd.DataFrame()
            
            # Convert Redis data to DataFrame
            records = [json.loads(record) for record in data]
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data for {symbol}: {e}")
            return pd.DataFrame()

    def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """
        Store market data in Redis with timestamp-based keys for efficient retrieval.
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval
            data: DataFrame with OHLCV data and timestamp index
        """
        try:
            key = f"market_data:{symbol}:{timeframe}"
            for timestamp, row in data.iterrows():
                record = {
                    'timestamp': timestamp.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']) if 'volume' in row else 0
                }
                
                # Use timestamp as score for sorted set
                score = int(timestamp.timestamp())
                self.redis_client.zadd(key, {json.dumps(record): score})
            
            self.logger.info(f"Stored {len(data)} records for {symbol}:{timeframe}")
            
        except Exception as e:
            self.logger.error(f"Error storing market data for {symbol}: {e}")

    def get_latest_price(self, symbol: str, timeframe: str = '1min') -> Optional[Dict]:
        """
        Get the latest price data for a symbol.
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval
            
        Returns:
            Latest price record or None
        """
        try:
            key = f"market_data:{symbol}:{timeframe}"
            # Get the most recent record
            latest_data = self.redis_client.zrange(key, -1, -1)
            
            if latest_data:
                return json.loads(latest_data[0])
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def _get_cached_training_data(self, symbol: str, days: int, data_source: str = 'alpha') -> pd.DataFrame:
        """
        Get cached training data from Redis if available.
        Enhanced to properly handle IB data published by data_publisher.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            data_source: Data source ('alpha', 'ib')
            
        Returns:
            DataFrame with cached training data or empty DataFrame
        """
        try:
            # Method 1: Check for recent training data in Redis streams
            stream_key = f"training_stream:{symbol}"
            
            try:
                messages = self.redis_client.xrevrange(stream_key, count=20)
                
                for msg_id, fields in messages:
                    if fields.get('data_source') == data_source:
                        # Check if we have the data chunks
                        data_points = int(fields.get('data_points', 0))
                        
                        # If we have recent data with sufficient points, try to reconstruct it
                        if data_points > 100:  # Minimum data threshold
                            # Get the most recent training data key pattern
                            pattern = f"training_data:{symbol}:*:metadata"
                            metadata_keys = self.redis_client.keys(pattern)
                            
                            if metadata_keys:
                                # Sort by timestamp in key name and get the most recent
                                latest_metadata_key = sorted(metadata_keys)[-1]
                                metadata_str = self.redis_client.get(latest_metadata_key)
                                
                                if metadata_str:
                                    metadata = json.loads(metadata_str)
                                    
                                    # Check if this matches our data source requirement
                                    if metadata.get('data_source') == data_source:
                                        print(f"Found cached {data_source} data for {symbol}: {metadata.get('total_records')} records")
                                        return self._reconstruct_dataframe_from_chunks(metadata)
                            
            except Exception as e:
                self.logger.debug(f"Error checking training stream for {symbol}: {e}")

            # Method 2: Direct search for training data keys
            pattern = f"training_data:{symbol}:*:metadata"
            metadata_keys = self.redis_client.keys(pattern)
            
            if metadata_keys:
                # Sort by key name (which contains timestamp) and get most recent
                latest_metadata_key = sorted(metadata_keys)[-1]
                metadata_str = self.redis_client.get(latest_metadata_key)
                
                if metadata_str:
                    metadata = json.loads(metadata_str)
                    
                    # Check data source and freshness
                    if metadata.get('data_source') == data_source:
                        total_records = metadata.get('total_records', 0)
                        if total_records > 100:
                            print(f"Found direct cached {data_source} data for {symbol}: {total_records} records")
                            return self._reconstruct_dataframe_from_chunks(metadata)

            # Method 3: Fallback to market_data for alpha data
            if data_source == 'alpha':
                return self._get_market_data_fallback(symbol, days)

            print(f"No cached {data_source} data found for {symbol}")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error getting cached training data for {symbol}: {e}")
            return pd.DataFrame()

    def _reconstruct_dataframe_from_chunks(self, metadata: Dict) -> pd.DataFrame:
        """
        Reconstruct DataFrame from Redis chunks using the metadata.
        Enhanced with better error handling and data validation.
        """
        try:
            base_key = metadata.get('base_key', '')
            total_chunks = metadata.get('total_chunks', 0)
            expected_records = metadata.get('total_records', 0)
            
            if not base_key or total_chunks == 0:
                print(f"Invalid metadata: base_key='{base_key}', total_chunks={total_chunks}")
                return pd.DataFrame()
            
            print(f"Reconstructing data from {total_chunks} chunks, expecting {expected_records} records")
            
            all_data = []
            successful_chunks = 0
            
            for i in range(total_chunks):
                chunk_key = f"{base_key}:chunk_{i}"
                chunk_data = self.redis_client.get(chunk_key)
                
                if chunk_data:
                    try:
                        chunk_json = json.loads(chunk_data)
                        if isinstance(chunk_json, list) and len(chunk_json) > 0:
                            chunk_df = pd.DataFrame(chunk_json)
                            if not chunk_df.empty:
                                all_data.append(chunk_df)
                                successful_chunks += 1
                    except json.JSONDecodeError as e:
                        print(f"Error decoding chunk {i}: {e}")
                        continue
                else:
                    print(f"Missing chunk {i} at key: {chunk_key}")

            if all_data:
                print(f"Successfully loaded {successful_chunks}/{total_chunks} chunks")
                df = pd.concat(all_data, ignore_index=True)
                
                # Ensure timestamp column exists and convert to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    
                    print(f"Reconstructed DataFrame: {len(df)} records from {df.index.min()} to {df.index.max()}")
                    return df
                else:
                    print(f"Error: No timestamp column found in reconstructed data")
                    print(f"Available columns: {df.columns.tolist()}")
            else:
                print(f"No data chunks could be loaded")

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error reconstructing DataFrame from chunks: {e}")
            print(f"Reconstruction error: {e}")
            return pd.DataFrame()

    def _get_market_data_fallback(self, symbol: str, days: int) -> pd.DataFrame:
        """Fallback to get data from market_data keys."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            return self.get_market_data(
                symbol,
                '1min',  # Default to 1min data
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        except Exception as e:
            self.logger.error(f"Error in market data fallback for {symbol}: {e}")
            return pd.DataFrame()

    async def get_training_data(self, symbol: str, days: int = 120,
                              data_source: str = 'alpha', **kwargs) -> pd.DataFrame:
        """
        Enhanced training data retrieval with support for multiple data sources.
        First checks Redis cache, then fetches from specified source if needed.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            data_source: Data source ('alpha' for Alpha Vantage, 'ib' for Interactive Brokers)
            **kwargs: Additional arguments (e.g., bar_size for IB data)
            
        Returns:
            DataFrame with training data
        """
        symbol = symbol.upper()
        print(f"Getting training data for {symbol}: {days} days from {data_source}")
        
        # First check if data exists in Redis cache
        data = self._get_cached_training_data(symbol, days, data_source)
        
        if not data.empty:
            print(f"Found cached {data_source} training data for {symbol}: {len(data)} records")
            return data

        # If no cached data found, fetch from source
        print(f"No cached {data_source} data found for {symbol}, fetching from source...")
        
        try:
            # Import here to avoid circular imports
            from .data_publisher import fetch_and_publish_data
            
            # Prepare kwargs for the data publisher
            publisher_kwargs = {
                'host': self.host,
                'port': self.port,
                'db': self.db
            }
            
            # Add IB-specific arguments if using IB data source
            if data_source == 'ib':
                publisher_kwargs.update({
                    'ib_host': kwargs.get('ib_host', '127.0.0.1'),
                    'ib_port': kwargs.get('ib_port', 4002),
                    'ib_client_id': kwargs.get('ib_client_id', 3)
                })

            # Fetch and publish data
            await fetch_and_publish_data(
                symbol=symbol,
                days=days,
                data_source=data_source,
                bar_size=kwargs.get('bar_size', '1 min'),
                **publisher_kwargs
            )

            # Wait a moment for data to be published and try again
            print(f"Waiting for data to be published to Redis...")
            await asyncio.sleep(5)  # Increased wait time
            
            data = self._get_cached_training_data(symbol, days, data_source)
            
            if not data.empty:
                print(f"Successfully fetched and cached {len(data)} records for {symbol} from {data_source}")
                return data
            else:
                print(f"Failed to retrieve training data for {symbol} from {data_source}")
                
                # If IB failed, try fallback to cached alpha data
                if data_source == 'ib':
                    print(f"Attempting fallback to Alpha Vantage data for {symbol}")
                    fallback_data = self._get_cached_training_data(symbol, days, 'alpha')
                    if not fallback_data.empty:
                        print(f"Using Alpha Vantage fallback data: {len(fallback_data)} records")
                        return fallback_data

                return pd.DataFrame()

        except ImportError:
            self.logger.error("Could not import data_publisher. Using fallback method.")
            return self._get_market_data_fallback(symbol, days)
            
        except Exception as e:
            self.logger.error(f"Error fetching training data for {symbol} from {data_source}: {e}")
            print(f"Error fetching data: {e}")
            
            # Try fallback to existing market data
            fallback_data = self._get_market_data_fallback(symbol, days)
            if not fallback_data.empty:
                print(f"Using fallback market data: {len(fallback_data)} records")
                return fallback_data
            
            return pd.DataFrame()

    def get_training_data_sync(self, symbol: str, days: int = 30,
                              timeframe: str = '1min') -> pd.DataFrame:
        """
        Synchronous version of get_training_data for backward compatibility.
        Uses the original implementation for existing code.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            timeframe: Time interval
            
        Returns:
            DataFrame with training data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_market_data(
            symbol,
            timeframe,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

    async def get_ib_training_data(self, symbol: str, days: int = 120,
                                 bar_size: str = '1 min', **kwargs) -> pd.DataFrame:
        """
        Convenience method to get IB training data.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            bar_size: IB bar size ('1 min', '5 mins', '1 hour', etc.)
            **kwargs: Additional IB connection parameters
            
        Returns:
            DataFrame with IB training data
        """
        return await self.get_training_data(
            symbol=symbol,
            days=days,
            data_source='ib',
            bar_size=bar_size,
            **kwargs
        )

    async def get_alpha_training_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """
        Convenience method to get Alpha Vantage training data.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with Alpha Vantage training data
        """
        return await self.get_training_data(
            symbol=symbol,
            days=days,
            data_source='alpha'
        )

    def store_training_result(self, symbol: str, model_type: str,
                            metrics: Dict, model_path: str):
        """
        Store training results and model metadata.
        
        Args:
            symbol: Stock symbol
            model_type: Type of model ('entry', 'exit', 'hybrid')
            metrics: Training metrics dictionary
            model_path: Path to saved model file
        """
        try:
            key = f"training_results:{symbol}:{model_type}"
            result = {
                'symbol': symbol,
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'model_path': model_path,
                'status': 'completed'
            }
            
            self.redis_client.setex(key, 86400 * 7, json.dumps(result))  # Keep for 7 days
            self.logger.info(f"Stored training results for {symbol}:{model_type}")
            
        except Exception as e:
            self.logger.error(f"Error storing training results: {e}")

    def get_training_results(self, symbol: str, model_type: str = None) -> Dict:
        """
        Retrieve training results for a symbol.
        
        Args:
            symbol: Stock symbol
            model_type: Specific model type or None for all
            
        Returns:
            Training results dictionary
        """
        try:
            if model_type:
                key = f"training_results:{symbol}:{model_type}"
                result = self.redis_client.get(key)
                return json.loads(result) if result else {}
            else:
                # Get all training results for symbol
                pattern = f"training_results:{symbol}:*"
                keys = self.redis_client.keys(pattern)
                results = {}
                
                for key in keys:
                    model_type = key.split(':')[-1]
                    result = self.redis_client.get(key)
                    if result:
                        results[model_type] = json.loads(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error retrieving training results: {e}")
            return {}

    def store_live_signal(self, symbol: str, signal_data: Dict):
        """
        Store live trading signals.
        
        Args:
            symbol: Stock symbol
            signal_data: Signal data dictionary
        """
        try:
            key = f"live_signals:{symbol}"
            signal_data['timestamp'] = datetime.now().isoformat()
            
            # Use timestamp as score
            score = int(datetime.now().timestamp())
            self.redis_client.zadd(key, {json.dumps(signal_data): score})
            
            # Keep only last 1000 signals
            self.redis_client.zremrangebyrank(key, 0, -1001)
            
            self.logger.debug(f"Stored live signal for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error storing live signal: {e}")

    def get_live_signals(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent live signals for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of signals to return
            
        Returns:
            List of signal dictionaries
        """
        try:
            key = f"live_signals:{symbol}"
            signals = self.redis_client.zrange(key, -limit, -1)
            return [json.loads(signal) for signal in signals]
            
        except Exception as e:
            self.logger.error(f"Error getting live signals: {e}")
            return []

    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Clean up old data to manage Redis memory usage.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        try:
            cutoff_timestamp = int((datetime.now() - timedelta(days=days_to_keep)).timestamp())
            
            # Clean up market data
            market_data_keys = self.redis_client.keys("market_data:*")
            for key in market_data_keys:
                self.redis_client.zremrangebyscore(key, 0, cutoff_timestamp)
            
            # Clean up live signals
            signal_keys = self.redis_client.keys("live_signals:*")
            for key in signal_keys:
                self.redis_client.zremrangebyscore(key, 0, cutoff_timestamp)
            
            # Clean up old training data chunks
            training_keys = self.redis_client.keys("training_data:*")
            current_time = datetime.now()
            
            for key in training_keys:
                try:
                    # Parse timestamp from key name for cleanup
                    if ':metadata' in key:
                        metadata_str = self.redis_client.get(key)
                        if metadata_str:
                            # Check age and clean up if too old
                            # This is a simplified cleanup - could be enhanced
                            pass
                    elif ':chunk_' not in key:
                        # Base training data key - check if it has metadata
                        metadata_key = f"{key}:metadata"
                        if not self.redis_client.exists(metadata_key):
                            # Orphaned key, delete it
                            self.redis_client.delete(key)
                except:
                    continue
            
            self.logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    def get_data_stats(self, symbol: str) -> Dict:
        """
        Get statistics about available data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with data statistics
        """
        try:
            stats = {}
            
            # Check different timeframes
            timeframes = ['1min', '5min', '1h', '1d']
            for tf in timeframes:
                key = f"market_data:{symbol}:{tf}"
                count = self.redis_client.zcard(key)
                
                if count > 0:
                    # Get date range
                    oldest = self.redis_client.zrange(key, 0, 0)
                    newest = self.redis_client.zrange(key, -1, -1)
                    
                    if oldest and newest:
                        oldest_record = json.loads(oldest[0])
                        newest_record = json.loads(newest[0])
                        
                        stats[tf] = {
                            'count': count,
                            'oldest': oldest_record['timestamp'],
                            'newest': newest_record['timestamp']
                        }
                else:
                    stats[tf] = {'count': 0}
            
            # Add training data stats
            training_stream_key = f"training_stream:{symbol}"
            
            try:
                training_info = self.redis_client.xinfo_stream(training_stream_key)
                stats['training_data'] = {
                    'stream_length': training_info.get('length', 0),
                    'data_sources': []
                }
                
                # Check for different data sources
                messages = self.redis_client.xrevrange(training_stream_key, count=10)
                sources = set()
                for _, fields in messages:
                    source = fields.get('data_source', 'unknown')
                    sources.add(source)
                
                stats['training_data']['data_sources'] = list(sources)
                
            except:
                stats['training_data'] = {'stream_length': 0, 'data_sources': []}
            
            # Add training data chunk stats
            training_pattern = f"training_data:{symbol}:*:metadata"
            training_keys = self.redis_client.keys(training_pattern)
            stats['training_chunks'] = {
                'available_datasets': len(training_keys),
                'datasets': []
            }
            
            for key in sorted(training_keys)[-5:]:  # Last 5 datasets
                metadata_str = self.redis_client.get(key)
                if metadata_str:
                    metadata = json.loads(metadata_str)
                    stats['training_chunks']['datasets'].append({
                        'timestamp': key.split(':')[2],
                        'data_source': metadata.get('data_source', 'unknown'),
                        'records': metadata.get('total_records', 0),
                        'business_days': metadata.get('business_days', 0)
                    })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting data stats: {e}")
            return {}

    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            result = self.redis_client.ping()
            self.logger.info(f"Redis health check successful: {self.host}:{self.port}")
            return result
            
        except Exception as e:
            self.logger.error(f"Redis health check failed for {self.host}:{self.port}: {e}")
            return False

    def debug_training_data(self, symbol: str) -> Dict:
        """
        Debug method to check what training data is available for a symbol.
        Useful for troubleshooting data retrieval issues.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'streams': {},
            'chunks': {},
            'market_data': {}
        }
        
        try:
            # Check training stream
            stream_key = f"training_stream:{symbol}"
            try:
                stream_info = self.redis_client.xinfo_stream(stream_key)
                debug_info['streams']['training_stream'] = {
                    'exists': True,
                    'length': stream_info.get('length', 0)
                }
                
                # Get recent messages
                messages = self.redis_client.xrevrange(stream_key, count=5)
                debug_info['streams']['recent_messages'] = []
                
                for msg_id, fields in messages:
                    debug_info['streams']['recent_messages'].append({
                        'id': msg_id,
                        'data_source': fields.get('data_source'),
                        'data_points': fields.get('data_points'),
                        'timestamp': fields.get('timestamp')
                    })
                    
            except Exception as e:
                debug_info['streams']['training_stream'] = {
                    'exists': False,
                    'error': str(e)
                }
            
            # Check for training data chunks
            chunk_pattern = f"training_data:{symbol}:*"
            chunk_keys = self.redis_client.keys(chunk_pattern)
            debug_info['chunks']['total_keys'] = len(chunk_keys)
            debug_info['chunks']['keys'] = sorted(chunk_keys)
            
            # Check metadata keys specifically
            metadata_pattern = f"training_data:{symbol}:*:metadata"
            metadata_keys = self.redis_client.keys(metadata_pattern)
            debug_info['chunks']['metadata_keys'] = len(metadata_keys)
            
            for key in sorted(metadata_keys)[-3:]:  # Last 3 metadata keys
                metadata_str = self.redis_client.get(key)
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                        debug_info['chunks'][key] = {
                            'data_source': metadata.get('data_source'),
                            'total_records': metadata.get('total_records'),
                            'total_chunks': metadata.get('total_chunks'),
                            'business_days': metadata.get('business_days')
                        }
                    except json.JSONDecodeError:
                        debug_info['chunks'][key] = {'error': 'Invalid JSON'}
            
            # Check market data
            market_key = f"market_data:{symbol}:1min"
            market_count = self.redis_client.zcard(market_key)
            debug_info['market_data']['1min_count'] = market_count
            
            print("=== TRAINING DATA DEBUG INFO ===")
            print(json.dumps(debug_info, indent=2))
            
            return debug_info
            
        except Exception as e:
            debug_info['error'] = str(e)
            print(f"Debug error: {e}")
            return debug_info

