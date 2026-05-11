import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the training data
series = scipy.io.loadmat('Xtrain.mat')['Xtrain'].flatten().astype(np.float64)

# Function to get the autocorrelation in the data
def autocorrelation(values, max_lag):
    centered = values - values.mean()
    variance = (centered * centered).mean()
    output = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            output[lag] = 1.0
        else:
            output[lag] = (centered[:-lag] * centered[lag:]).mean() / variance
    return output


# Get the autocorrelation values
autocorrelation_values = autocorrelation(series, 200)

# Look for local maxima above 0.3 these are the candidate dominant lags
peaks = []
for index in range(2, len(autocorrelation_values) - 1):
    if (autocorrelation_values[index] > autocorrelation_values[index - 1]
            and autocorrelation_values[index] > autocorrelation_values[index + 1]
            and autocorrelation_values[index] > 0.3):
        peaks.append((index, autocorrelation_values[index]))
for peak in peaks[:10]:
    print(f"  lag {peak[0]}: {peak[1]:.4f}")

# Pick the strongest peak as the lag used in the phase plot
phase_lag = peaks[0][0] if peaks else 7

# Make the plot
figure, axes = plt.subplots(3, 1, figsize=(13, 7.5))
axis = axes[0]
axis.plot(np.arange(len(series)), series, color='black', lw=1.2, label='Data')
axis.legend(loc='upper left')
axis.set_xlabel('t')
axis.set_ylabel('x(t)')
axis.set_title('Data')

# Autocorrelation
axis = axes[1]
axis.plot(np.arange(len(autocorrelation_values)), autocorrelation_values, color='black', lw=1.2, label='autocorrelation')
axis.axhline(0, color='k', ls=':')
axis.legend(loc='upper left')
axis.set_xlabel('lag')
axis.set_ylabel('autocorrelation')
axis.set_title('Autocorrelation')

# Phase plot
axis = axes[2]
axis.plot(series[:-phase_lag], series[phase_lag:], '.', color='red', ms=2, label=f'x(t) vs x(t-{phase_lag})')
axis.legend(loc='upper left')
axis.set_xlabel(f'x(t-{phase_lag})')
axis.set_ylabel('x(t)')
axis.set_title(f'Phase plot x(t) vs x(t-{phase_lag})')

plt.tight_layout()

# Save the plot
plt.savefig('analysis.png', dpi=110)
