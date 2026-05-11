import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the model predictions
predictions = np.load('final_run/recursive_forecast.npy')
# Load the test data
true_values = scipy.io.loadmat('Xtest.mat')['Xtest'].flatten().astype(np.float64)

# Load the training data
training_series = scipy.io.loadmat('Xtrain.mat')['Xtrain'].flatten().astype(np.float64)

# Get the number of steps
num_steps = len(predictions)

# Calculate the errors
errors = predictions - true_values
mse = np.mean(errors ** 2)
mae = np.mean(np.abs(errors))

# Make the plot
figure, axes = plt.subplots(2, 1, figsize=(13, 7.5))
axis = axes[0]
axis.plot(np.arange(len(training_series)), training_series, color='0.5', lw=0.8, label='Xtrain (history)')
test_x_axis = np.arange(len(training_series), len(training_series) + num_steps)
axis.plot(test_x_axis, true_values, color='black', lw=1.2, label='Xtest (truth)')
axis.plot(test_x_axis, predictions, color='red', lw=1.2, label='forecast')
axis.axvline(len(training_series), color='k', ls=':')
axis.legend(loc='upper left')
axis.set_title(f'Recursive 200-step evaluation, MSE={mse:.2f}   MAE={mae:.2f}')

# Zoomed view
axis = axes[1]
axis.plot(true_values, color='black', label='truth')
axis.plot(predictions, color='red', label='forecast')
axis.fill_between(np.arange(num_steps), true_values, predictions, color='red', alpha=0.15)
axis.legend(loc='upper left')
axis.set_xlabel('forecast step')
axis.set_title('Zoomed: forecast only')

plt.tight_layout()

# Save the plot
plt.savefig('final_run/test_eval.png', dpi=110)
