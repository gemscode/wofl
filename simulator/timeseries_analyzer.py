import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
from datetime import datetime
from common.data_manager import FixedDataManagerLocal

def main():
    """
    Performs time-series analysis (currently generates random signals as a placeholder).
    This script is called as a subprocess by the training_agent.
    """
    parser = argparse.ArgumentParser(description='Time Series Analysis')
    # This script correctly accepts all arguments passed by the agent.
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--days', type=int, default=40)
    parser.add_argument('--target-profit', type=float, default=0.01)
    args = parser.parse_args()

    # Create a unique filename to prevent conflicts during parallel runs.
    filename = f"ts_{args.symbol}_{os.getpid()}_{datetime.now().strftime('%f')}.json"

    results = {}
    try:
        dm = FixedDataManagerLocal(f"S_{args.symbol}_ALPHA", args.host)
        data = dm.load_multiple_days(dm.available_days[-args.days:])

        # Generate some random signals for demonstration purposes
        indices = np.random.choice(data.index, size=np.random.randint(20, 50), replace=False)
        signals = [{'timestamp': pd.to_datetime(ts).isoformat(), 'signal': np.random.choice(['BUY', 'SELL'])} for ts in indices]

        results = {
            'symbol': args.symbol,
            'signals': signals
        }
        # Use stderr for logging to keep stdout clean for the filename
        sys.stderr.write(f"[TIMESERIES] Analysis complete for {args.symbol}.\n")

    except Exception as e:
        sys.stderr.write(f"[ERROR] TIMESERIES analysis failed: {e}\n")
        # Ensure the error result dict still has the keys the agent expects
        results = {'symbol': args.symbol, 'signals': []}

    # --- Communication Protocol ---
    # Always create the results file and print its name LAST.
    with open(filename, 'w') as f:
        json.dump(results, f, default=str)

    # This MUST be the only line printed to stdout for the agent to read.
    print(filename)

if __name__ == "__main__":
    main()

