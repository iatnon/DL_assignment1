import json
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.io
import torch

from model import (MinMaxScaler, ResidualRNN, make_windows, set_seed,
                   find_collapse_windows, find_posttail_windows,
                   train, recursive_forecast)

# Define variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
raw_series = scipy.io.loadmat('Xtrain.mat')['Xtrain'].flatten().astype(np.float64)
HORIZON = 200

# Pick two random validation windows from the valid range [200, 800].
rng = np.random.default_rng(20260511)
SPLITS = sorted(int(x) for x in rng.integers(200, 801, size=2))
print(SPLITS)

# Hyperparameters initizialization
HIDDEN_SIZE = 192
NUM_LAYERS = 3
EPOCHS = 200                  
SWA_START = 140               
SWA_FREQUENCY = 4
NOISE_STD = 0.03
OVERSAMPLE_FACTOR = 4.0
PERIOD_REG_WEIGHT = 0.05
LOOKBACK_CANDIDATES = [40, 50, 60, 70, 80]
SEEDS = [('gru', 42), ('lstm', 43)]

all_runs = []
log_lines = []

# Loop over the window size values
for lookback in LOOKBACK_CANDIDATES:
    per_lookback = []
    for split_s in SPLITS:
        training_part = raw_series[:split_s]
        validation_part = raw_series[split_s:split_s + HORIZON]
        scaler = MinMaxScaler().fit(training_part)
        training_scaled = scaler.transform(training_part).astype(np.float32)

        # Make the windows and split the data into training and validation sets
        all_windows_x, all_windows_y = make_windows(training_scaled, lookback)
        inner_validation_size = max(40, int(len(all_windows_x) * 0.1))
        x_train = all_windows_x[:-inner_validation_size]
        y_train = all_windows_y[:-inner_validation_size]
        x_validation = all_windows_x[-inner_validation_size:]
        y_validation = all_windows_y[-inner_validation_size:]

        # Identify collapse and post-tail windows in the training split only
        collapse_indices = find_collapse_windows(training_scaled, lookback)
        collapse_indices = collapse_indices[collapse_indices < len(x_train)]
        posttail_indices = find_posttail_windows(training_scaled, lookback)
        posttail_indices = posttail_indices[posttail_indices < len(x_train)]

        for cell_type, seed in SEEDS:
            set_seed(seed)
            t0 = time.time()
            model = ResidualRNN(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                                dropout=0.2, cell_type=cell_type, predict_residual=False)

            # Model training
            model, inner_val_mse = train(
                model, x_train, y_train, x_validation, y_validation,
                epochs=EPOCHS, swa_start=SWA_START, swa_frequency=SWA_FREQUENCY,
                batch_size=64, learning_rate=2e-3, weight_decay=1e-5,
                noise_std=NOISE_STD, patience=9999,
                collapse_indices=collapse_indices, posttail_indices=posttail_indices,
                oversample_factor=OVERSAMPLE_FACTOR, period_reg_weight=PERIOD_REG_WEIGHT,
                device=device, clip_predictions=True)

            # Recursive 200-step forecast against the held-out region
            predictions_scaled = recursive_forecast(model, training_scaled, HORIZON, lookback, device, clip=True)
            predictions = scaler.inverse(predictions_scaled)
            errors = predictions - validation_part
            mse = float(np.mean(errors ** 2))
            mae = float(np.mean(np.abs(errors)))
            duration = time.time() - t0

            all_runs.append({'lookback': lookback, 'split': split_s, 'cell': cell_type,
                             'mse': mse, 'mae': mae, 'inner_val_mse': inner_val_mse,
                             'seconds': duration})
            per_lookback.append((mse, mae))


    # mean mse and mae
    mean_mse = float(np.mean([m for m, _ in per_lookback]))
    mean_mae = float(np.mean([a for _, a in per_lookback]))

    # prints the mean mse and mae for each lookback
    print(lookback, mean_mse, mean_mae)
    
# Aggregate summary
agg = defaultdict(list)
for r in all_runs:
    agg[r['lookback']].append((r['mse'], r['mae']))


summary = []
for L, vals in agg.items():
    mses = [v[0] for v in vals]
    maes = [v[1] for v in vals]
    summary.append({'lookback': L,
                    'mean_mse': float(np.mean(mses)),
                    'mean_mae': float(np.mean(maes)),
                    'std_mse': float(np.std(mses)),
                    'std_mae': float(np.std(maes))})
for s in sorted(summary, key=lambda x: x['mean_mse']):
    print(s)

with open('tune.json', 'w') as results_file:
    json.dump({'splits': SPLITS, 'runs': all_runs, 'summary': summary}, results_file, indent=2)
