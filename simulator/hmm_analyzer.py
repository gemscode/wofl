import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
from datetime import datetime
from hmmlearn import hmm
from common.data_manager import FixedDataManagerLocal

def main():
    """
    Performs Hidden Markov Model (HMM) analysis to identify market regimes.
    This script is called as a subprocess by the training_agent.
    """
    parser = argparse.ArgumentParser(description='HMM Analysis for Trading Regimes')
    # This script correctly accepts all arguments passed by the agent.
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--days', type=int, default=40)
    parser.add_argument('--target-profit', type=float, default=0.01)
    parser.add_argument('--regimes', type=int, default=4)
    args = parser.parse_args()

    # Create a unique filename to prevent conflicts during parallel runs.
    filename = f"hmm_{args.symbol}_{os.getpid()}_{datetime.now().strftime('%f')}.json"

    results = {}
    try:
        # Load data
        dm = FixedDataManagerLocal(f"S_{args.symbol}_ALPHA", args.host)
        data = dm.load_multiple_days(dm.available_days[-args.days:])
        returns = data['close'].pct_change().dropna().values.reshape(-1, 1)

        # Add tiny noise to prevent zero-variance errors in HMM
        if np.std(returns) < 1e-5:
            returns += np.random.normal(0, 1e-4, returns.shape)

        # Fit the HMM model to identify regimes
        model = hmm.GaussianHMM(n_components=args.regimes, covariance_type="diag", n_iter=1000, random_state=42)
        model.fit(returns)
        states = model.predict(returns)

        # Align the predicted states with the correct timestamps from the original data
        start_index = len(data) - len(states)
        relevant_timestamps = data.index[start_index:]
        states_df = pd.DataFrame({'timestamp': relevant_timestamps, 'regime_state': states})

        # Prepare results dictionary. The 'regime_states' are crucial for the neural network.
        results = {
            'symbol': args.symbol,
            'signals': [],  # This script now focuses on providing states as features
            'regime_states': states_df.to_dict('records')
        }
        # Use stderr for logging to keep stdout clean for the filename
        sys.stderr.write(f"[HMM] Analysis complete for {args.symbol}.\n")

    except Exception as e:
        sys.stderr.write(f"[ERROR] HMM analysis failed: {e}\n")
        # Ensure the error result dict still has the keys the agent expects
        results = {'symbol': args.symbol, 'signals': [], 'regime_states': []}

    # --- Communication Protocol ---
    # Always create the results file and print its name LAST.
    with open(filename, 'w') as f:
        # Use default=str to handle any non-serializable types gracefully (like datetimes).
        json.dump(results, f, default=str)

    # This MUST be the only line printed to stdout for the agent to read.
    print(filename)

if __name__ == "__main__":
    main()

