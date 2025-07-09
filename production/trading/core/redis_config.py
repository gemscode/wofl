#!/usr/bin/env python3
"""
Redis Configuration Manager
Handles Redis connection setup and management
"""

import redis
import sys

class RedisConfig:
    """Redis connection configuration and management"""
    
    def __init__(self, config):
        self.host = config.get('redis_host', 'trader.wolfx0.com')
        self.port = config.get('redis_port', 6379)
        self.password = config.get('redis_password')
        
        # Load password if not provided
        if not self.password:
            self.password = self.load_password_from_file()
    
    def load_password_from_file(self):
        """Load Redis password from .redis_passwd file"""
        try:
            with open('.redis_passwd', 'r') as f:
                password = f.read().strip()
            print("[CONFIG] Redis password loaded from .redis_passwd file")
            return password
        except FileNotFoundError:
            print("[ERROR] Redis password not provided and .redis_passwd file not found")
            sys.exit(1)
    
    def get_client(self):
        """Get configured Redis client"""
        client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=10,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        try:
            pong = client.ping()
            print(f"[CONNECTION] Redis server status: {pong}")
            return client
        except Exception as e:
            print(f"[ERROR] Redis connection failed: {e}")
            sys.exit(1)
