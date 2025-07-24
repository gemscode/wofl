# File: training_agent.py
import json
import sys
import os
import subprocess
import time
from datetime import datetime

import torch
import pandas as pd

from common.data_manager import FixedDataManagerLocal
from common.backtester import UniversalBacktester
from strategies.neural_strategy import NeuralStrategy


def get_device():
    """Checks for available hardware acceleration and returns the appropriate torch device."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_external_analysis(script, symbol, host, days, target_profit, **kwargs):
    """
    Runs an external analysis script (e.g., hmm_analyzer.py) as a robust subprocess.

    This function handles:
    - Building the command with necessary arguments.
    - Capturing stdout and stderr from the script.
    - Reliably parsing the temporary filename from the LAST line of stdout.
    - A retry loop to prevent filesystem race conditions on file creation.
    - Graceful error handling for script crashes or other exceptions.
    """
    name = script.split('_')[0].upper()
    print(f"\n[{name}] Running analysis using temporary file method...")

    cmd = [
        sys.executable, script,
        '--symbol', symbol,
        '--host', host,
        '--days', str(days),
        '--target-profit', str(target_profit)
    ]
    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)

        # Get all non-empty lines from the script's output
        output_lines = [line for line in proc.stdout.strip().split('\n') if line.strip()]
        if not output_lines:
            raise ValueError("Analyzer script did not print a filename.")

        # The filename is the VERY LAST line printed. This makes it robust to logging.
        temp_filename = output_lines[-1]

        # Retry loop to handle filesystem race conditions, especially in parallel runs.
        results = None
        for _ in range(5):
            if os.path.exists(temp_filename):
                with open(temp_filename, 'r') as f:
                    results = json.load(f)
                os.remove(temp_filename)
                break
            time.sleep(0.1)  # Wait 100ms before retrying

        if results is None:
            raise FileNotFoundError(f"Analyzer file '{temp_filename}' did not appear in time.")

        return results

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {name} script failed. Stderr:\n{e.stderr.strip()}")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error with {name}: {e}")
        return None


def run_hybrid_analysis(symbol, host, days, target_profit, epochs, window_size):
    """
    Orchestrates the entire hybrid analysis pipeline.
    
    1. Loads data.
    2. Runs external HMM and TimeSeries analyzers.
    3. Integrates HMM results as a feature for the Neural Strategy.
    4. Runs the Neural Strategy's comprehensive analysis.
    5. Backtests all strategies on a consistent test set.
    """
    print(f"[AGENT] Starting analysis with window={window_size}, epochs={epochs}, target_profit={target_profit:.3%}")

    try:
        dm = FixedDataManagerLocal(f"S_{symbol.upper()}_ALPHA", host)
        # Load extra days to provide enough data for TA feature lookback periods
        data = dm.load_multiple_days(dm.available_days[-(days + 20):])
        # Keep original close price for accurate backtesting
        data['close_raw'] = data['close'].copy()
    except Exception as e:
        print(f"[FATAL] Data loading failed: {e}")
        return None

    device = get_device()
    print(f"[SYSTEM] Using device: {device}")

    # Run external analyzers to get signals and features
    hmm_res = run_external_analysis('hmm_analyzer.py', symbol, host, days, target_profit, regimes=4)
    ts_res = run_external_analysis('timeseries_analyzer.py', symbol, host, days, target_profit)

    # --- Robustly merge HMM data while preserving the DatetimeIndex ---
    if hmm_res and 'regime_states' in hmm_res and hmm_res['regime_states']:
        hmm_df = pd.DataFrame(hmm_res['regime_states'])
        hmm_df['timestamp'] = pd.to_datetime(hmm_df['timestamp'])
        
        # Use a left merge on the original dataframe's index to preserve it
        data = data.merge(hmm_df.set_index('timestamp'), left_index=True, right_index=True, how='left')
        
        # Forward fill and then backfill to handle any gaps in regime data
        data['regime_state'] = data['regime_state'].ffill().fillna(0)
        print("[AGENT] Successfully integrated HMM regime states as a feature.")
    else:
        # Create a default regime column if HMM analysis fails
        data['regime_state'] = 0

    # Run the main neural strategy
    strategy_nn = NeuralStrategy(symbol, device, window_size)
    neural_results = strategy_nn.run_comprehensive_analysis(data=data.copy(), epochs=epochs)

    # --- Backtest all strategies on a consistent test set ---
    # Use the last 20% of the data period for the final evaluation
    test_data_start = data.index.max() - pd.Timedelta(days=int(days * 0.2))
    test_data = data[data.index >= test_data_start]

    results = {'strategies': {'neural': neural_results}}
    for name, analysis_res in [('HMM', hmm_res), ('TIMESERIES', ts_res)]:
        if analysis_res and analysis_res.get('signals'):
            backtester = UniversalBacktester(test_data.copy(), initial_cash=25000)
            analysis_res['test'] = backtester.run(analysis_res['signals'])
        else:
            # Provide a default result for failed or signal-less strategies
            analysis_res = analysis_res or {}
            analysis_res['test'] = {'final_balance': 25000, 'total_return': 0}
        results['strategies'][name.lower()] = analysis_res

    return results

