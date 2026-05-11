import json
import os
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt

from model import (MinMaxScaler, ResidualRNN, make_windows, set_seed,
                   find_collapse_windows, find_posttail_windows,
                   ensemble_recursive_forecast, train, phase_warp)

# Initialize values
LOOKBACK = 60                
NUM_SEEDS = 16               
HIDDEN_SIZE = 192           
NUM_LAYERS = 3
EPOCHS = 400
SWA_START = 280             
SWA_FREQUENCY = 4
NOISE_STD = 0.03          
OVERSAMPLE_FACTOR = 4.0      
PERIOD_REG_WEIGHT = 0.05   
HORIZON = 200
AGGREGATOR = 'mean'
PHASE_WARP_WINDOW = 40
OUTPUT_DIRECTORY = 'final_run'
REPORT_VALIDATION = False 


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_series = scipy.io.loadmat('Xtrain.mat')['Xtrain'].flatten().astype(np.float64)
    print(f"Loaded {len(raw_series)} samples")

    # Split the data into training and validation sets
    if REPORT_VALIDATION:
        training_part = raw_series[:-HORIZON]
        validation_part = raw_series[-HORIZON:]
    else:
        training_part = raw_series
        validation_part = None

    scaler = MinMaxScaler().fit(training_part)
    training_scaled = scaler.transform(training_part).astype(np.float32)

    # Build sliding windows and split off an inner validation set for early stopping
    all_windows_x, all_windows_y = make_windows(training_scaled, LOOKBACK)
    inner_validation_size = max(60, int(len(all_windows_x) * 0.1))
    x_train = all_windows_x[:-inner_validation_size]
    y_train = all_windows_y[:-inner_validation_size]
    x_validation = all_windows_x[-inner_validation_size:]
    y_validation = all_windows_y[-inner_validation_size:]

    # Identify collapse and post-tail windows in the training split only
    collapse_indices = find_collapse_windows(training_scaled, LOOKBACK)
    collapse_indices = collapse_indices[collapse_indices < len(x_train)]
    posttail_indices = find_posttail_windows(training_scaled, LOOKBACK)
    posttail_indices = posttail_indices[posttail_indices < len(x_train)]

    models = []
    for seed in range(NUM_SEEDS):
        set_seed(20000 + seed)

        # Switch between GRU and LSTM for each model
        cell_type = 'gru' if seed % 2 == 0 else 'lstm'

        # Initialize model
        model = ResidualRNN(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                            dropout=0.2, cell_type=cell_type, predict_residual=False)

        # Train the model
        model, validation_loss = train(
            model, x_train, y_train, x_validation, y_validation,
            epochs=EPOCHS, swa_start=SWA_START, swa_frequency=SWA_FREQUENCY,
            batch_size=64, learning_rate=2e-3, weight_decay=1e-5,
            noise_std=NOISE_STD, patience=9999,
            collapse_indices=collapse_indices, posttail_indices=posttail_indices,
            oversample_factor=OVERSAMPLE_FACTOR, period_reg_weight=PERIOD_REG_WEIGHT,
            device=device, clip_predictions=True)
        models.append(model)

    # Save all model weights
    for index, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIRECTORY, f'model_{index:02d}.pt'))

    # Save the configuration to a config file
    config = {
        'lookback': LOOKBACK,
        'num_seeds': NUM_SEEDS,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'epochs': EPOCHS,
        'swa_start': SWA_START,
        'swa_frequency': SWA_FREQUENCY,
        'noise_std': NOISE_STD,
        'oversample_factor': OVERSAMPLE_FACTOR,
        'period_reg_weight': PERIOD_REG_WEIGHT,
        'aggregator': AGGREGATOR,
        'scaler_low': scaler.low,
        'scaler_high': scaler.high,
        'horizon': HORIZON,
        'used_full_train': not REPORT_VALIDATION,
        'cells': ['gru' if seed % 2 == 0 else 'lstm' for seed in range(NUM_SEEDS)],
    }
    with open(os.path.join(OUTPUT_DIRECTORY, 'config.json'), 'w') as config_file:
        json.dump(config, config_file, indent=2)

    # Generate forecast from training data
    predictions_scaled = ensemble_recursive_forecast(models, training_scaled, HORIZON, LOOKBACK,
                                                     device=device, aggregator=AGGREGATOR)

    # Apply phase warp to correct potential drift.
    predictions_scaled_warped = phase_warp(predictions_scaled, window=PHASE_WARP_WINDOW)
    predictions = scaler.inverse(predictions_scaled_warped)

    np.save(os.path.join(OUTPUT_DIRECTORY, 'recursive_forecast.npy'), predictions)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'recursive_forecast.csv'), predictions, fmt='%.6f')
    print(f"Saved recursive forecast (horizon={HORIZON}) to {OUTPUT_DIRECTORY}/")

    # Report results
    if validation_part is not None:
        errors = predictions - validation_part
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        print(f"FINAL ENSEMBLE: MSE={mse}  MAE={mae}")
        config['final_mse'] = mse
        config['final_mae'] = mae
        with open(os.path.join(OUTPUT_DIRECTORY, 'config.json'), 'w') as config_file:
            json.dump(config, config_file, indent=2)

    # Get the number of steps
    num_steps = len(predictions)

    # Make the plot
    figure, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    axis = axes[0]
    axis.plot(np.arange(len(training_part)), training_part, color='0.5', lw=0.8, label='Xtrain (history)')
    forecast_x_axis = np.arange(len(training_part), len(training_part) + num_steps)
    if validation_part is not None:
        axis.plot(forecast_x_axis, validation_part, color='black', lw=1.2, label='Xtest (truth)')
    axis.plot(forecast_x_axis, predictions, color='red', lw=1.2, label='forecast')
    axis.axvline(len(training_part), color='k', ls=':')
    axis.legend(loc='upper left')
    if validation_part is not None:
        axis.set_title(f'Recursive {num_steps}-step forecast, MSE={mse:.2f}   MAE={mae:.2f}')
    else:
        axis.set_title(f'Recursive {num_steps}-step forecast')

    # Zoomed view
    axis = axes[1]
    if validation_part is not None:
        axis.plot(validation_part, color='black', label='truth')
        axis.plot(predictions, color='red', label='forecast')
        axis.fill_between(np.arange(num_steps), validation_part, predictions, color='red', alpha=0.15)
    else:
        axis.plot(predictions, color='red', label='forecast')
    axis.legend(loc='upper left')
    axis.set_xlabel('forecast step')
    axis.set_title('Zoomed: forecast only')

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'forecast.png'), dpi=110)
    print(f"Saved {OUTPUT_DIRECTORY}/forecast.png")


if __name__ == '__main__':
    main()
