import elfi
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from da_handy_model_improved import ImprovedHandyModel
from da_utils import get_three_ranges
from simulation_constants import N, batch_size_c, width, first_x_range, second_x_range, third_x_range, sampling, start, \
    stop, quantile

seed = 20170530
np.random.seed(seed)

# We set true parameters
true_values = {'xC': 100, 'xE': 25, 'y': 100, 'w': 100}

number_of_samples = int(1/sampling)

full_range = np.arange(start, stop, sampling)
all_x_range = np.array(full_range).reshape((1, len(full_range)))

first_x_data, second_x_data, third_x_data = get_three_ranges(start,
                                                             first_x_range,
                                                             second_x_range,
                                                             third_x_range,
                                                             number_of_samples,
                                                             full_range)

pred_prey_model = ImprovedHandyModel()

# Plot the observed sequence for whole range
y_obs = pred_prey_model.calculate_model(true_values, all_x_range)
plt.figure(figsize=(11, 6))
plt.plot(all_x_range[0, :], y_obs[0, :])

# Points between these lines are training points
plt.axvline(x=first_x_range, color='r')
plt.axvline(x=second_x_range,  color='r')

plt.xlabel('X value as an argument for model')
plt.ylabel('Y value of the model')


# We plot only training part
train_data = pred_prey_model.calculate_model(true_values, second_x_data)
plt.figure(figsize=(11, 6))

plt.xticks(np.arange(first_x_range, second_x_range, 1.0))

plt.plot(second_x_data[0, :], train_data[0, :])
plt.xlabel('X value as an argument for function')
plt.ylabel('Y value of the function')
plt.show()

# MAGIC
pred_prey_model.second_x_data = second_x_data

# has to be this way, so the keys in result dict have parameter names
xC = elfi.Prior(scipy.stats.uniform, true_values['xC']-width*true_values['xC'], 2 * width*true_values['xC'])
xE = elfi.Prior(scipy.stats.uniform, true_values['xE']-width*true_values['xE'], 2 * width*true_values['xE'])
y = elfi.Prior(scipy.stats.uniform, true_values['y']-width*true_values['y'], 2 * width*true_values['y'])
w = elfi.Prior(scipy.stats.uniform, true_values['w']-width*true_values['w'], 2 * width*true_values['w'])

# Define the simulator node with the MA2 model ,give the priors to it as arguments.
Y = elfi.Simulator(pred_prey_model.model, xC, xE, y, w, observed=train_data)


# Autocovariances as the summary statistics
def autocov(x, lag=1):
    c = np.mean(x[:, lag:] * x[:, :-lag], axis=1)
    return c


# Summary node is defined by giving the autocovariance function and the simulated data (also includes observed data)
S1 = elfi.Summary(autocov, Y)
S2 = elfi.Summary(autocov, Y, 2)

# Calculating the squared distance (S1_sim-S1_obs)**2 + (S2_sim-S2_obs)**2
d = elfi.Distance('euclidean', S1, S2)

# Instantiation of the Rejection Algorithm
rej = elfi.Rejection(d, batch_size=batch_size_c, seed=seed)

result = rej.sample(N, quantile=quantile)
# Print sampled means of parameters

print(result)

# Final result of mean samples
mean_results = {k: v.mean() for k, v in result.samples.items()}

for key, value in mean_results.items():
    print('{0}: {1}'.format(key, value))

y_obs = pred_prey_model.calculate_model(true_values, second_x_data)
plt.figure(figsize=(11, 6))
plt.plot(y_obs.ravel(), label="observed")
plt.plot(pred_prey_model.calculate_model(mean_results, second_x_data).ravel(),
         label="simulated")
plt.legend(loc="upper left")
plt.show()

y_obs = pred_prey_model.calculate_model(true_values, all_x_range)
plt.figure(figsize=(11, 6))
plt.plot(y_obs.ravel(), label="observed")
all_results_predicted = pred_prey_model.calculate_model(mean_results, all_x_range)
plt.plot(all_results_predicted.ravel(), label="simulated")
plt.legend(loc="upper left")
plt.show()


def calculate_error(start, stop):
    calculate = 0
    for i in range(start, stop, 1):
        calculate += (y_obs[0][i] - all_results_predicted[0][i])**2
    return calculate


print(calculate_error(int(start + 1), int(first_x_range * number_of_samples)))
print(calculate_error(int(first_x_range * number_of_samples), int(second_x_range * number_of_samples)))
print(calculate_error(int(second_x_range * number_of_samples), int(third_x_range * number_of_samples)))
