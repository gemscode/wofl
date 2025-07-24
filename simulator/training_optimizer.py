# File: training_optimizer.py
import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# FIX: Import the agent correctly.
from training_agent import run_hybrid_analysis

def run_optimizer(symbol, search_type='normal', max_workers=4):
    """
    Hyperparameter search space and execution logic.
    """
    print(f"[OPTIMIZER] Starting hyperparameter search for {symbol}...")

    # Expanded search space for more robust tuning
    param_grid = {
        'epochs': [20, 30, 40],
        'window_size': [3, 5, 8],
        'target_profit': [0.010, 0.015, 0.020]
    }
    if search_type == 'aggressive':
        param_grid['epochs'] = [20, 30, 50, 70]
        param_grid['window_size'] = [3, 5, 8, 12]

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    print(f"[OPTIMIZER] Testing {len(all_params)} different configurations with {max_workers} workers.")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(run_hybrid_analysis, symbol, "localhost", 40, p['target_profit'], p['epochs'], p['window_size']): p
            for p in all_params
        }
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                if result and 'strategies' in result:
                    # Capture the primary performance metric for ranking
                    neural_perf = result['strategies'].get('neural', {}).get('test', {})
                    final_balance = neural_perf.get('final_balance', 0)
                    total_return = neural_perf.get('total_return', 0)
                    
                    results.append({
                        'params': params,
                        'final_balance': final_balance,
                        'total_return': total_return,
                        'full_results': result # Store everything for detailed review
                    })
                    print(f"[OPTIMIZER] Run with params {tuple(params.values())} completed. Return: {total_return:.2%}")
                else:
                    print(f"[OPTIMIZER] Run with params {tuple(params.values())} returned no valid result.")
            except Exception as exc:
                print(f'[OPTIMIZER] Run with params {tuple(params.values())} failed: {exc}')

    if not results:
        print("[OPTIMIZER] No successful runs completed. Exiting.")
        return

    # Find the best parameters based on final balance
    best_run = max(results, key=lambda x: x['final_balance'])
    print("\n" + "="*40 + " OPTIMIZATION COMPLETE " + "="*40)
    print(f"Best Parameters Found: {best_run['params']}")
    print(f"  -> Final Balance: ${best_run['final_balance']:,.2f}")
    print(f"  -> Total Return: {best_run['total_return']:.2%}")
    print("="*103)

    # Save the results
    filename = f"optimizer_results_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(best_run, f, indent=2, default=str)
    print(f"[OPTIMIZER] Detailed results for the best run saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Strategy Hyperparameter Optimizer')
    parser.add_argument('--symbol', required=True, help='Stock symbol to optimize (e.g., META)')
    parser.add_argument('--search-type', default='normal', choices=['normal', 'aggressive'])
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()
    run_optimizer(args.symbol, args.search_type, args.max_workers)

