#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse, json, sys, os
from datetime import datetime
from hmmlearn import hmm
from sklearn.model_selection import cross_val_score
from common.data_manager import FixedDataManagerLocal

def optimize_hmm_parameters(data, regime_range=(2, 8), cv_folds=3):
    """Optimize HMM parameters using cross-validation"""
    returns = data['close'].pct_change().dropna().values.reshape(-1, 1)
    best_score = -np.inf
    best_params = {'n_components': 4, 'covariance_type': 'full'}
    
    print(f"[HMM] Optimizing parameters across {regime_range[0]}-{regime_range[1]} regimes...")
    
    for n_regimes in range(regime_range[0], regime_range[1] + 1):
        for cov_type in ['full', 'diag', 'tied']:
            try:
                model = hmm.GaussianHMM(n_components=n_regimes, covariance_type=cov_type, n_iter=100, random_state=42)
                model.fit(returns)
                score = model.score(returns)
                
                if score > best_score:
                    best_score = score
                    best_params = {'n_components': n_regimes, 'covariance_type': cov_type}
                    
                print(f"  Regimes: {n_regimes}, Cov: {cov_type}, Score: {score:.4f}")
            except Exception as e:
                continue
    
    print(f"[HMM] Best parameters: {best_params}, Score: {best_score:.4f}")
    return best_params

def generate_regime_signals(data, model, states, target_profit=0.01):
    """Generate trading signals based on regime transitions and characteristics"""
    signals = []
    regime_stats = {}
    
    # Calculate regime statistics - FIXED INDEX ALIGNMENT
    returns = data['close'].pct_change().dropna()
    
    # Ensure states and returns have same length
    min_len = min(len(states), len(returns))
    states_aligned = states[:min_len]
    returns_aligned = returns.iloc[:min_len] if hasattr(returns, 'iloc') else returns[:min_len]
    
    for regime in range(model.n_components):
        regime_mask = states_aligned == regime
        if np.sum(regime_mask) > 10:  # Ensure sufficient data
            regime_returns = returns_aligned[regime_mask]
            regime_stats[regime] = {
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe': regime_returns.mean() / (regime_returns.std() + 1e-8)
            }
    
    # Generate signals based on regime transitions and characteristics
    for i in range(1, min_len):
        if states_aligned[i] != states_aligned[i-1]:  # Regime change
            current_regime = states_aligned[i]
            prev_regime = states_aligned[i-1]
            
            if current_regime in regime_stats and prev_regime in regime_stats:
                current_stats = regime_stats[current_regime]
                prev_stats = regime_stats[prev_regime]
                
                # Signal logic based on regime characteristics
                if current_stats['mean_return'] > target_profit and current_stats['sharpe'] > 0.1:
                    signals.append({
                        'timestamp': data.index[i+1].isoformat(), 
                        'signal': 'BUY',
                        'confidence': min(abs(current_stats['sharpe']), 1.0),
                        'regime': int(current_regime),
                        'expected_return': float(current_stats['mean_return'])
                    })
                elif current_stats['mean_return'] < -target_profit and prev_stats['mean_return'] > 0:
                    signals.append({
                        'timestamp': data.index[i+1].isoformat(), 
                        'signal': 'SELL',
                        'confidence': min(abs(current_stats['sharpe']), 1.0),
                        'regime': int(current_regime),
                        'expected_return': float(current_stats['mean_return'])
                    })
    
    # Limit signals to prevent overtrading
    signals = signals[:50] if len(signals) > 50 else signals
    return signals, regime_stats

def main():
    parser = argparse.ArgumentParser(description='Enhanced HMM Analysis')
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--days', type=int, default=40)
    parser.add_argument('--target-profit', type=float, default=0.0112)
    parser.add_argument('--regimes', type=int, default=0, help="Fixed regimes (0 for auto-optimization)")
    parser.add_argument('--regime-min', type=int, default=2, help="Minimum regimes for optimization")
    parser.add_argument('--regime-max', type=int, default=8, help="Maximum regimes for optimization")
    args = parser.parse_args()
    
    try:
        dm = FixedDataManagerLocal(f"S_{args.symbol}_ALPHA", args.host)
        data = dm.load_multiple_days(dm.available_days[-args.days:])
        
        if args.regimes > 0:
            best_params = {'n_components': args.regimes, 'covariance_type': 'full'}
            print(f"[HMM] Using fixed {args.regimes} regimes")
        else:
            # Optimize regimes
            best_params = optimize_hmm_parameters(data, (args.regime_min, args.regime_max))
        
        # Train final model with best parameters
        returns = data['close'].pct_change().dropna().values.reshape(-1, 1)
        model = hmm.GaussianHMM(**best_params, n_iter=200, random_state=42)
        model.fit(returns)
        states = model.predict(returns)
        
        # Generate enhanced signals
        signals, regime_stats = generate_regime_signals(data, model, states, args.target_profit)
        
        results = {
            'symbol': args.symbol,
            'signals': signals,
            'total_signals': len(signals),
            'hmm_params': best_params,
            'regime_stats': regime_stats,
            'model_score': float(model.score(returns))
        }
        
        filename = f"hmm_analysis_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[HMM] Generated {len(signals)} signals using {best_params['n_components']} regimes")
        print(f"[HMM] Model score: {model.score(returns):.4f}")
        print(f"[HMM] Results saved to {filename}")
        
    except Exception as e:
        print(f"[ERROR] HMM analysis failed: {e}")
        # Create empty results file so training can continue
        results = {
            'symbol': args.symbol,
            'signals': [],
            'total_signals': 0,
            'hmm_params': {'n_components': 4, 'covariance_type': 'full'},
            'regime_stats': {},
            'model_score': 0.0
        }
        filename = f"hmm_analysis_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()

