import json
import os
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt

from model import (MinMaxScaler, ResidualRNN, ensemble_recursive_forecast, phase_warp)

INPUT_DIRECTORY = 'final_run'
NUM_STEPS = 200
SEED_FROM_FILE = 'Xtrain.mat'
SEED_KEY = 'Xtrain'
OUTPUT_PATH = None


def load_ensemble(input_directory):
    config = json.load(open(os.path.join(input_directory, 'config.json')))
    scaler = MinMaxScaler()
    scaler.low = config['scaler_low']
    scaler.high = config['scaler_high']
    lookback = config['lookback']
    cells = config['cells']
    models = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for index, cell_type in enumerate(cells):
        model = ResidualRNN(hidden_size=config['hidden_size'], num_layers=config['num_layers'],
                            dropout=0.2, cell_type=cell_type, predict_residual=False)
        model.load_state_dict(torch.load(os.path.join(input_directory, f'model_{index:02d}.pt'),
                                         map_location=device))
        model.to(device).eval()
        models.append(model)
    return models, scaler, lookback, config


def main():
    # Load the ensemble
    models, scaler, lookback, config = load_ensemble(INPUT_DIRECTORY)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the training data (used as the seed history for the recursive rollout)
    training_series = scipy.io.loadmat(SEED_FROM_FILE)[SEED_KEY].flatten().astype(np.float64)
    history_scaled = scaler.transform(training_series).astype(np.float32)

    # Generate the raw recursive forecast
    aggregator = config.get('aggregator', 'mean')
    predictions_scaled = ensemble_recursive_forecast(models, history_scaled, NUM_STEPS, lookback,
                                                     device=device, aggregator=aggregator)

    # Apply blind phase warp to correct accumulated period drift in the rollout
    warp_window = config.get('phase_warp_window', 40)
    predictions_scaled_warped = phase_warp(predictions_scaled, window=warp_window)
    predictions = scaler.inverse(predictions_scaled_warped)

    # Save the forecast as csv and npy
    output_path = OUTPUT_PATH or os.path.join(INPUT_DIRECTORY, f'prediction_{NUM_STEPS}.csv')
    np.savetxt(output_path, predictions, fmt='%.6f')
    np.save(output_path.replace('.csv', '.npy'), predictions)

    # Get the number of steps
    num_steps = len(predictions)

    # Make the plot
    figure, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    axis = axes[0]
    axis.plot(np.arange(len(training_series)), training_series, color='0.5', lw=0.8, label='Xtrain (history)')
    forecast_x_axis = np.arange(len(training_series), len(training_series) + num_steps)
    axis.plot(forecast_x_axis, predictions, color='red', lw=1.2, label='forecast')
    axis.axvline(len(training_series), color='k', ls=':')
    axis.legend(loc='upper left')
    axis.set_title(f'Recursive {num_steps}-step forecast (with phase warp)')

    # Zoomed view
    axis = axes[1]
    axis.plot(predictions, color='red', label='forecast')
    axis.legend(loc='upper left')
    axis.set_xlabel('forecast step')
    axis.set_title('Zoomed: forecast only')

    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path.replace('.csv', '.png'), dpi=110)


if __name__ == '__main__':
    main()
