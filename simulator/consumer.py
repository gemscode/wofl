#!/usr/bin/env python3
"""
Consumer script to subscribe to S_LIVE_ABC and provide trading recommendations
"""

import redis
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

class TradingConsumer:
    def __init__(self, redis_host='trader.wolfx0.com', redis_port=6379):
        """Initialize consumer with Redis connection"""
        try:
            with open('.redis_passwd', 'r') as f:
                password = f.read().strip()
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=password,
                decode_responses=True
            )
            
            self.redis_client.ping()
            print(f"✓ Connected to Redis at {redis_host}:{redis_port}")
            
        except Exception as e:
            print(f"✗ Redis connection failed: {e}")
            raise
        
        self.recommendations = []
        self.price_history = []
        self.optimal_strategy = None
        self.strategy_params = {}
    
    def load_optimal_strategy(self, symbol, source='alpha'):
        """Load optimal strategy from optimization results"""
        try:
            results_file = Path(f'simple_hindsight_rl_results_{source}.json')
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                symbol_key = f'S_{symbol}_{source.upper()}'
                if symbol_key in results:
                    self.optimal_strategy = results[symbol_key]['best_strategy']
                    strategy_data = results[symbol_key]['strategies'][self.optimal_strategy]
                    self.strategy_params = strategy_data
                    
                    print(f"✓ Loaded optimal strategy: {self.optimal_strategy}")
                    print(f"  📊 Train Return: {strategy_data['train_return']:.4f}")
                    print(f"  📈 Test Return: {strategy_data['test_return']:.4f}")
                    print(f"  📡 Data Source: {strategy_data.get('data_source', 'Unknown')}")
                    return True
                else:
                    print(f"⚠️  No optimization results found for {symbol_key}")
            else:
                print(f"⚠️  No optimization results file found: {results_file}")
        
        except Exception as e:
            print(f"✗ Error loading strategy: {e}")
        
        # Use default strategy
        self.optimal_strategy = 'LONG_ONLY'
        self.strategy_params = {'train_return': 0.0, 'test_return': 0.0}
        print(f"📋 Using default strategy: {self.optimal_strategy}")
        return False
    
    def generate_recommendation(self, price_data):
        """Generate trading recommendation based on optimal strategy"""
        try:
            current_price = float(price_data['price'])
            self.price_history.append(current_price)
            
            # Keep only last 50 prices for analysis
            if len(self.price_history) > 50:
                self.price_history.pop(0)
            
            # Need at least 10 data points for analysis
            if len(self.price_history) < 10:
                return {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'reasoning': 'Insufficient price history for analysis'
                }
            
            # Calculate technical indicators
            prices = np.array(self.price_history)
            sma_5 = np.mean(prices[-5:])
            sma_10 = np.mean(prices[-10:])
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else sma_10
            
            # Calculate momentum
            momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            
            # Calculate volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # Strategy-based decision making
            action = 'HOLD'
            confidence = 0.5
            reasoning = 'Market analysis'
            
            if self.optimal_strategy == 'LONG_ONLY':
                if sma_5 > sma_10 and momentum > 0.01:
                    action = 'BUY'
                    confidence = min(0.5 + momentum * 10, 0.95)
                    reasoning = 'Bullish momentum with SMA crossover'
                elif momentum < -0.015:
                    action = 'SELL'
                    confidence = min(0.5 + abs(momentum) * 10, 0.95)
                    reasoning = 'Strong bearish momentum'
                    
            elif self.optimal_strategy == 'SHORT_ONLY':
                if sma_5 < sma_10 and momentum < -0.01:
                    action = 'SELL'
                    confidence = min(0.5 + abs(momentum) * 10, 0.95)
                    reasoning = 'Bearish momentum with SMA crossover'
                elif momentum > 0.015:
                    action = 'BUY'
                    confidence = min(0.5 + momentum * 10, 0.95)
                    reasoning = 'Strong bullish momentum (cover short)'
                    
            elif self.optimal_strategy == 'LONG_SHORT':
                if sma_5 > sma_10 and momentum > 0.008:
                    action = 'BUY'
                    confidence = min(0.5 + momentum * 12, 0.95)
                    reasoning = 'Long signal: bullish momentum'
                elif sma_5 < sma_10 and momentum < -0.008:
                    action = 'SELL'
                    confidence = min(0.5 + abs(momentum) * 12, 0.95)
                    reasoning = 'Short signal: bearish momentum'
            
            # Adjust confidence based on volatility
            if volatility > 0.02:  # High volatility
                confidence *= 0.8
                reasoning += ' (reduced confidence due to high volatility)'
            
            return {
                'action': action,
                'confidence': round(confidence, 3),
                'reasoning': reasoning,
                'technical_data': {
                    'price': current_price,
                    'sma_5': round(sma_5, 4),
                    'sma_10': round(sma_10, 4),
                    'momentum': round(momentum, 4),
                    'volatility': round(volatility, 4)
                }
            }
            
        except Exception as e:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': f'Error in analysis: {str(e)}'
            }
    
    def consume_stream(self, symbol, stream_name=None, save_recommendations=True):
        """Consume data from Redis stream and generate recommendations"""
        if stream_name is None:
            stream_name = f'S_LIVE_{symbol}'
        
        print(f"\n🎯 Starting consumer for {symbol}")
        print(f"📡 Listening to stream: {stream_name}")
        print(f"🧠 Strategy: {self.optimal_strategy}")
        print("=" * 80)
        
        try:
            last_id = '0'
            recommendation_count = 0
            
            while True:
                try:
                    # Read from stream
                    messages = self.redis_client.xread({stream_name: last_id}, count=1, block=1000)
                    
                    if not messages:
                        continue
                    
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            last_id = msg_id
                            
                            # Generate recommendation
                            recommendation = self.generate_recommendation(fields)
                            
                            # Create complete recommendation record
                            rec_record = {
                                'timestamp': fields.get('timestamp', datetime.now().isoformat()),
                                'symbol': symbol,
                                'price': float(fields.get('price', 0)),
                                'volume': int(fields.get('volume', 0)),
                                'action': recommendation['action'],
                                'confidence': recommendation['confidence'],
                                'reasoning': recommendation['reasoning'],
                                'strategy': self.optimal_strategy,
                                'technical_data': recommendation.get('technical_data', {})
                            }
                            
                            self.recommendations.append(rec_record)
                            recommendation_count += 1
                            
                            # Display recommendation
                            timestamp = datetime.fromisoformat(fields['timestamp'].replace('Z', '+00:00'))
                            print(f"🤖 [{recommendation_count:3d}] {timestamp.strftime('%H:%M:%S')} | "
                                  f"${float(fields['price']):7.2f} | "
                                  f"{recommendation['action']:4s} | "
                                  f"Conf: {recommendation['confidence']:.2f} | "
                                  f"{recommendation['reasoning'][:40]}...")
                
                except redis.ResponseError as e:
                    if "NOGROUP" in str(e):
                        print(f"⚠️  Stream {stream_name} not found, waiting...")
                        time.sleep(2)
                        continue
                    else:
                        raise
                        
        except KeyboardInterrupt:
            print(f"\n⏹️  Consumer stopped by user")
        except Exception as e:
            print(f"\n💥 Consumer error: {e}")
        
        # Save recommendations
        if save_recommendations and self.recommendations:
            self.save_recommendations(symbol)
        
        print(f"\n📊 Generated {len(self.recommendations)} recommendations")
        return self.recommendations
    
    def save_recommendations(self, symbol):
        """Save recommendations to JSON file"""
        try:
            filename = f'recommendations_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            summary = {
                'symbol': symbol,
                'strategy': self.optimal_strategy,
                'strategy_params': self.strategy_params,
                'total_recommendations': len(self.recommendations),
                'actions_summary': {
                    'BUY': len([r for r in self.recommendations if r['action'] == 'BUY']),
                    'SELL': len([r for r in self.recommendations if r['action'] == 'SELL']),
                    'HOLD': len([r for r in self.recommendations if r['action'] == 'HOLD'])
                },
                'avg_confidence': np.mean([r['confidence'] for r in self.recommendations]),
                'generated_at': datetime.now().isoformat(),
                'recommendations': self.recommendations
            }
            
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"💾 Saved {len(self.recommendations)} recommendations to {filename}")
            print(f"📈 Actions: {summary['actions_summary']}")
            print(f"🎯 Avg Confidence: {summary['avg_confidence']:.3f}")
            
        except Exception as e:
            print(f"✗ Error saving recommendations: {e}")
    
    def get_recommendations_json(self):
        """Return recommendations as JSON string"""
        return json.dumps(self.recommendations, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Trading Recommendation Consumer')
    parser.add_argument('--symbol', default='ABC', help='Stock symbol to consume (default: ABC)')
    parser.add_argument('--source', default='alpha', choices=['yahoo', 'alpha'], 
                       help='Optimization source (default: alpha)')
    parser.add_argument('--stream', help='Custom stream name (default: S_LIVE_{symbol})')
    parser.add_argument('--no-save', action='store_true', help='Do not save recommendations to file')
    parser.add_argument('--host', default='trader.wolfx0.com', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    try:
        # Create consumer
        consumer = TradingConsumer(args.host, args.port)
        
        # Load optimal strategy
        consumer.load_optimal_strategy(args.symbol, args.source)
        
        # Start consuming
        recommendations = consumer.consume_stream(
            symbol=args.symbol,
            stream_name=args.stream,
            save_recommendations=not args.no_save
        )
        
        print(f"\n✅ Consumer completed for {args.symbol}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Consumer stopped by user")
    except Exception as e:
        print(f"\n💥 Consumer error: {e}")

if __name__ == '__main__':
    main()

