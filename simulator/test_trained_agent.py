#!/usr/bin/env python3
"""
test_trained_agent.py – evaluate a trained intraday agent on unseen data
-----------------------------------------------------------------------
CLI
$ python test_trained_agent.py \
      --symbol GERN \
      --model  models/gern_model.pt \
      --days   10 \
      --capital 25000
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------ #
#  IMPORT YOUR OWN PROJECT MODULES – adjust paths if necessary       #
# ------------------------------------------------------------------ #
from data_manager           import FixedDataManager
from pattern_explorer       import PatternExplorer
from intraday_preprocessor  import IntradayDataPreprocessor          # <- the class used at training time
from fixed_cnn_bilstm_model import FixedCNNBiLSTMModel               # <- the model class used at training time
# ------------------------------------------------------------------ #

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    
    # ­­­­­restore architecture  – hyper-parameters were saved in the checkpoint
    net_cfg   = checkpoint["net_cfg"]          # dict saved during training
    model     = FixedCNNBiLSTMModel(**net_cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # restore pre-processor scalers if present
    if "preproc_state" in checkpoint:
        preproc = IntradayDataPreprocessor(**checkpoint["preproc_state"]["init_args"])
        preproc.scalers = checkpoint["preproc_state"]["scalers"]
    else:
        preproc = IntradayDataPreprocessor()    # fallback (will fit on test data)
    
    return model, preproc


def build_test_loader(symbol: str, days: int, preproc: IntradayDataPreprocessor):
    dm   = FixedDataManager(f"S_{symbol.upper()}_ALPHA")
    data = dm.load_multiple_days(dm.available_days[-days:])          # newest N days
    
    explorer = PatternExplorer(data)                                 # add indicators identical to training
    data     = explorer.preprocess_data()
    
    seqs, labels = preproc.transform(data)[:2]                       # ignore weights
    
    # align structures for torch
    x_tensor = torch.from_numpy(seqs).float()
    y_tensor = torch.from_numpy(labels).long()
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=256, shuffle=False), data


def run_backtest(model, loader, raw_df, preproc, initial_cash: float):
    device = next(model.parameters()).device
    
    predictions, confidences = [], []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            probs  = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            predictions.extend(pred.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
    
    # align test window with dataframe
    start = preproc.sequence_length
    closes = raw_df["close"].values[start : start + len(predictions)]
    
    cash, shares, trades = initial_cash, 0, 0
    equity_curve = []
    for price, pred, conf in zip(closes, predictions, confidences):
        # 0 = HOLD, 1 = BUY, 2 = SELL
        if conf > 0.6:
            if pred == 1 and shares == 0:                       # enter long
                qty   = int(cash // price)
                shares += qty
                cash   -= qty * price
                trades += 1
            elif pred == 2 and shares > 0:                      # exit
                cash   += shares * price
                shares  = 0
                trades += 1
        equity_curve.append(cash + shares * price)
    
    final_ret = (equity_curve[-1] - initial_cash) / initial_cash * 100
    return {
        "trades"       : trades,
        "return_pct"   : final_ret,
        "equity_curve" : equity_curve
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",   required=True)
    parser.add_argument("--model",    required=True, help="path to .pt checkpoint")
    parser.add_argument("--days",     type=int, default=10, help="test days")
    parser.add_argument("--capital",  type=float, default=25000, help="starting cash")
    args = parser.parse_args()
    
    device = get_device()
    
    model, preproc = load_model(args.model, device)
    loader, df     = build_test_loader(args.symbol, args.days, preproc)
    
    result = run_backtest(model, loader, df, preproc, args.capital)
    
    # ---------- summary ---------- #
    print("\n===== TEST SUMMARY =====")
    print(f"Symbol          : {args.symbol}")
    print(f"Test days       : {args.days}")
    print(f"Total trades    : {result['trades']}")
    print(f"Return (%)      : {result['return_pct']:.2f}")
    
    # Save equity curve & metadata
    out = {
        "symbol"       : args.symbol,
        "test_days"    : args.days,
        "trades"       : result["trades"],
        "return_pct"   : result["return_pct"],
        "equity_curve" : result["equity_curve"],
        "timestamp"    : datetime.utcnow().isoformat()
    }
    out_file = f"test_{args.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    Path("reports").mkdir(exist_ok=True)
    with open(Path("reports") / out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Report written → reports/{out_file}")


if __name__ == "__main__":
    main()

