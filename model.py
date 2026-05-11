from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Scaler, scales the data between -1 and 1.
class MinMaxScaler:
    def __init__(self):
        self.low = None
        self.high = None

    # To get the maximum and minimum values of our data points
    def fit(self, data: np.ndarray):
        self.low = float(data.min())
        self.high = float(data.max())
        return self

    # To transform our data points into [-1,+1] range
    def transform(self, data):
        return 2.0 * (data - self.low) / (self.high - self.low) - 1.0

    # To transform our scaled [-1,+1] range data points back to its original values
    def inverse(self, data):
        return (data + 1.0) * 0.5 * (self.high - self.low) + self.low


# Sliding-window dataset
def make_windows(series: np.ndarray, lookback: int):
    """series: 1D scaled array. Returns inputs (N, L, 1), targets (N,)."""
    inputs, targets = [], []
    for i in range(lookback, len(series)):
        inputs.append(series[i - lookback:i])
        targets.append(series[i])
    inputs = np.asarray(inputs, dtype=np.float32)[..., None]
    targets = np.asarray(targets, dtype=np.float32)
    return inputs, targets


# Collapse window detection
def find_collapse_windows(series: np.ndarray, lookback: int,
                          peak_window: int = 15, low_quantile: float = 0.20):
    """
    Return indices into the windows produced by make_windows whose target
    sits in the bottom low_quantile of local peak-to-peak amplitude.
    These are the collapse onset and trough windows — the rare transitions
    to near-zero amplitude that the model must learn to predict.
    """
    # Compute local amplitude at every timestep
    envelope = np.zeros(len(series))
    for index in range(len(series)):
        start = max(0, index - peak_window + 1)
        chunk = series[start:index + 1]
        envelope[index] = chunk.max() - chunk.min()

    # Window targets start at index `lookback`
    target_indices = np.arange(lookback, len(series))
    threshold = np.quantile(envelope, low_quantile)
    return np.where(envelope[target_indices] <= threshold)[0]


def find_posttail_windows(series: np.ndarray, lookback: int,
                          peak_window: int = 15,
                          collapse_quantile: float = 0.20,
                          posttail_quantile: float = 0.40):
    """
    Return indices of windows in the low-amplitude recovery tail that
    immediately follows a collapse — i.e. where the local amplitude is
    moderate-low AND there has been a collapse trough within the last 30 steps.
    Training on these teaches the model what happens during recovery, not
    just how to enter a collapse.
    """
    envelope = np.zeros(len(series))
    for index in range(len(series)):
        start = max(0, index - peak_window + 1)
        chunk = series[start:index + 1]
        envelope[index] = chunk.max() - chunk.min()

    target_indices = np.arange(lookback, len(series))
    target_env = envelope[target_indices]
    collapse_threshold = np.quantile(envelope, collapse_quantile)
    posttail_threshold = np.quantile(envelope, posttail_quantile)

    # In recovery: amplitude is between collapse and posttail thresholds
    is_low_to_mid = (target_env > collapse_threshold) & (target_env <= posttail_threshold)

    # AND there is a recent collapse trough behind us (within 30 steps)
    has_recent_collapse = np.zeros(len(target_indices), dtype=bool)
    for i in range(len(target_indices)):
        absolute_t = target_indices[i]
        lookback_start = max(0, absolute_t - 30)
        if envelope[lookback_start:absolute_t].min() <= collapse_threshold:
            has_recent_collapse[i] = True

    return np.where(is_low_to_mid & has_recent_collapse)[0]


# Creating the Neural Network
# RNN with either GRU or LSTM
class ResidualRNN(nn.Module):
    # RNN with residual connections and fully connected output layers.
    # This defines the hyperparameters
    def __init__(self, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2,
                 cell_type: str = 'gru', predict_residual: bool = False):
        super().__init__()

        # Transforming each datapoint into a vector of hidden_size dimensions
        self.input_projection = nn.Linear(1, hidden_size)

        # Setting if we use a RNN with GRU or with LSTM
        self.cell_type = cell_type
        # Which kind of RNN we end up choosing from the nn library. GRU or LSTM
        rnn_class = nn.GRU if cell_type == 'gru' else nn.LSTM

        # This creates the Recurrent Neural Network
        self.rnn = rnn_class(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Adding a normalization layer to normalize the internal activation values
        # (more stability, less exploding gradients)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # The output head is a fully connected network that maps the RNN output to a scalar prediction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.predict_residual = predict_residual

    # Forward pass
    def forward(self, inputs):
        # inputs has shape (batch_size, sequence_length, 1)
        # last_value is kept so we can optionally predict a delta instead of the absolute next value
        last_value = inputs[:, -1, 0]
        hidden = self.input_projection(inputs)

        rnn_output, _ = self.rnn(hidden)

        # Skip connection: add the projected input back to the RNN output
        rnn_output = rnn_output + hidden
        rnn_output = self.layer_norm(rnn_output[:, -1])

        # Pass the last hidden state through the fully connected output layers
        # We can set predict_residual to have the network predict delta (t+1 - t) instead of t+1 directly
        delta = self.output_head(rnn_output).squeeze(-1)
        if self.predict_residual:
            return last_value + delta

        # Get the prediction
        return delta


# Training with SWA (Stochastic Weight Averaging) and collapse-aware oversampling
def train(model: nn.Module, x_train, y_train, x_validation, y_validation, *,
                   epochs: int = 300, swa_start: int = 220, swa_frequency: int = 4,
                   batch_size: int = 64, learning_rate: float = 2e-3, weight_decay: float = 1e-5,
                   noise_std: float = 0.03, patience: int = 9999,
                   collapse_indices: np.ndarray = None, posttail_indices: np.ndarray = None,
                   oversample_factor: float = 4.0, period_reg_weight: float = 0.01,
                   device: str = 'cuda', clip_predictions: bool = True):

    # Initialize model and training components
    model = model.to(device)
    x_train_tensor = torch.tensor(x_train, device=device)
    y_train_tensor = torch.tensor(y_train, device=device)
    x_validation_tensor = torch.tensor(x_validation, device=device)
    y_validation_tensor = torch.tensor(y_validation, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.05)

    # Initialize training variables
    num_samples = x_train_tensor.shape[0]
    swa_state = None
    swa_count = 0
    best_validation_loss = float('inf')
    best_state = None

    # Start the training loop
    for epoch in range(epochs):
        model.train()

        # Randomly shuffle the training data at the start of each epoch
        permutation = torch.randperm(num_samples, device=device)

        for start in range(0, num_samples, batch_size):
            batch_indices = permutation[start:start + batch_size].cpu().numpy()

            # Oversample collapse-onset windows so the model sees them 4x more often
            if collapse_indices is not None and len(collapse_indices) > 0:
                num_extra = int(oversample_factor * len(batch_indices))
                extra = np.random.choice(collapse_indices, size=num_extra, replace=True)
                batch_indices = np.concatenate([batch_indices, extra])

            # Oversample post-collapse recovery windows so the model learns to find the right pattern after collapse better.
            if posttail_indices is not None and len(posttail_indices) > 0:
                num_extra = int(oversample_factor * len(batch_indices))
                extra = np.random.choice(posttail_indices, size=num_extra, replace=True)
                batch_indices = np.concatenate([batch_indices, extra])

            batch_indices_t = torch.tensor(batch_indices, dtype=torch.long, device=device)
            x_batch = x_train_tensor[batch_indices_t].clone()
            y_batch = y_train_tensor[batch_indices_t]

            # Adding a small noise to the input makes the model more robust in the chaotic setting of the data.
            if noise_std > 0:
                x_batch = x_batch + torch.randn_like(x_batch) * noise_std

            optimizer.zero_grad()
            predictions = model(x_batch)

            # Clamp the predictions from -1.5 to 1.5 to prevent the model from predicting values outside the range of the training data.
            if clip_predictions:
                predictions = torch.clamp(predictions, -1.5, 1.5)

            # We use Smooth L1 Loss because it is less sensitive to outliers than Mean Squared Error.
            loss = F.smooth_l1_loss(predictions, y_batch)

            # Period regularisation: penalise when the batch ACF at lag 8 disagrees with targets prevents the model from slowly drifting to the wrong period.

            if period_reg_weight > 0 and len(predictions) > 12:
                p_centered = predictions - predictions.mean()
                y_centered = y_batch - y_batch.mean()
                ac_pred   = (p_centered[:-8] * p_centered[8:]).mean()
                ac_target = (y_centered[:-8] * y_centered[8:]).mean().detach()
                loss = loss + period_reg_weight * (ac_pred - ac_target).pow(2)

            # Backpropagate the loss and update the weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Advance the learning rate schedule
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            validation_predictions = model(x_validation_tensor)
            if clip_predictions:
                validation_predictions = torch.clamp(validation_predictions, -1.5, 1.5)
            validation_loss = F.mse_loss(validation_predictions, y_validation_tensor).item()
        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        # Stochstic weight averaging where the averaged weights generalise better for long recursive rollouts.
        if epoch >= swa_start and (epoch - swa_start) % swa_frequency == 0:
            current_state = {key: value.detach().float().clone() for key, value in model.state_dict().items()}
            if swa_state is None:
                swa_state = current_state
                swa_count = 1
            else:
                swa_count += 1
                for key in swa_state:
                    swa_state[key] = swa_state[key] + (current_state[key] - swa_state[key]) / swa_count

    # Load the weights
    if swa_state is not None:
        model.load_state_dict(swa_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    # Return the best model and the best validation loss
    return model, best_validation_loss


# Recursive forecasting
@torch.no_grad()
def recursive_forecast(model: nn.Module, history_scaled: np.ndarray,
                       steps: int, lookback: int, device: str = 'cuda',
                       clip: bool = True) -> np.ndarray:
    model.eval()
    history_buffer = list(map(float, history_scaled[-lookback:]))
    predictions = []
    for _ in range(steps):

        # Create a window from the history buffer
        input_window = torch.tensor(history_buffer[-lookback:], dtype=torch.float32, device=device).view(1, lookback, 1)
        next_value = float(model(input_window).item())
        if clip:
            next_value = float(np.clip(next_value, -1.0, 1.0))
        predictions.append(next_value)
        history_buffer.append(next_value)
    return np.asarray(predictions, dtype=np.float32)


@torch.no_grad()
def ensemble_recursive_forecast(models, history_scaled, steps, lookback,
                                device='cuda', clip: bool = True,
                                aggregator: str = 'mean') -> np.ndarray:

    # Average all model predictions at each step
    for model in models:
        model.eval()

    # Create a window from the history buffer
    history_buffer = list(map(float, history_scaled[-lookback:]))
    predictions = []

    # Recursive forecasting
    for _ in range(steps):
        input_window = torch.tensor(history_buffer[-lookback:], dtype=torch.float32, device=device).view(1, lookback, 1)
        model_predictions = np.array([float(model(input_window).item()) for model in models])
        if aggregator == 'median':
            next_value = float(np.median(model_predictions))
        else:
            next_value = float(np.mean(model_predictions))
        if clip:
            next_value = float(np.clip(next_value, -1.0, 1.0))
        predictions.append(next_value)
        history_buffer.append(next_value)
    return np.asarray(predictions, dtype=np.float32)


# Phase warp correcting potential period mismatch between forecast as recursive prediction continues
def phase_warp(forecast: np.ndarray, window: int = 50) -> np.ndarray:
    fc = forecast.copy().astype(np.float64)
    n = len(fc)

    # Measure period from the strictly at the start of the forecast
    pre = fc[:window]
    s = pre - pre.mean()
    var = float(np.dot(s, s))
    best_ac, ref_period = -1.0, 8
    for lag in range(4, 16):
        ac = float(np.dot(s[:-lag], s[lag:])) / var
        if ac > best_ac:
            best_ac = ac
            ref_period = lag

    corrected = fc.copy()
    corrected[:window] = fc[:window]

    for t in range(window, n):

        # Rolling measured period in a trailing window
        chunk = fc[max(0, t - window):t]
        chunk_s = chunk - chunk.mean()
        chunk_var = float(np.dot(chunk_s, chunk_s))
        measured_period = ref_period

        # Find the dominant period in the trailing window
        if chunk_var > 1e-10:
            best_ac_roll = -1.0
            for lag in range(4, 16):
                if lag >= len(chunk_s):
                    break

                # Compare autocorrelation at each possible lag () with the best one found so far
                ac = float(np.dot(chunk_s[:-lag], chunk_s[lag:])) / chunk_var
                if ac > best_ac_roll:
                    best_ac_roll = ac
                    measured_period = lag

        # If the forecast runs faster than refference period at the start of the forecast, stretch it back
        drift = (measured_period - ref_period) / max(measured_period, 1)
        src = t + drift * (t - window)
        src = float(np.clip(src, 0, n - 1))
        lo = int(src)
        frac = src - lo

        # Interpolate between the two closest points in the forecast
        if lo + 1 < n:
            corrected[t] = fc[lo] * (1.0 - frac) + fc[lo + 1] * frac
        else:
            corrected[t] = fc[lo]

    return corrected.astype(np.float32)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
