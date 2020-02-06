import elfi
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from data_assimilation_models import PredatorPreyModelDA
from data_assimilation_utils import get_three_ranges
from simulation_constants import N, batch_size_c, width, first_x_range, second_x_range, third_x_range, sampling, start, \
    stop

seed = 20170530
np.random.seed(seed)

# We set true parameters
a = 1.
b = 0.5

number_of_samples = int(1/sampling)

full_range = np.arange(start, stop, sampling)
all_x_range = np.array(full_range).reshape((1, len(full_range)))

first_x_data, second_x_data, third_x_data = get_three_ranges(start,
                                                             first_x_range,
                                                             second_x_range,
                                                             third_x_range,
                                                             number_of_samples,
                                                             full_range)

pred_prey_model = PredatorPreyModelDA()

# Plot the observed sequence for whole range
y_obs = pred_prey_model.calculate_model(a, b, all_x_range)
plt.figure(figsize=(11, 6))
plt.plot(all_x_range[0, :], y_obs[0, :])

# Points between these lines are training points
plt.axvline(x=first_x_range, color='r')
plt.axvline(x=second_x_range,  color='r')

plt.xlabel('X value as an argument for model')
plt.ylabel('Y value of the model')


# We plot only training part
train_data = pred_prey_model.calculate_model(a, b, second_x_data)
plt.figure(figsize=(11, 6))

plt.xticks(np.arange(first_x_range, second_x_range, 1.0))

plt.plot(second_x_data[0, :], train_data[0, :])
plt.xlabel('X value as an argument for function')
plt.ylabel('Y value of the function')
plt.show()

# MAGIC
pred_prey_model.second_x_data = second_x_data

a_param = elfi.Prior(scipy.stats.uniform, a-width, 2 * width)
b_param = elfi.Prior(scipy.stats.uniform, b-width, 2 * width)

# Define the simulator node with the MA2 model ,give the priors to it as arguments.
Y = elfi.Simulator(pred_prey_model.model, a_param, b_param, observed=train_data)


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

result = rej.sample(N, quantile=0.001)
# Print sampled means of parameters

print(result)

# Final result of mean samples
b_result_last = result.samples['b_param'].mean()
a_result_last = result.samples['a_param'].mean()

print(a_result_last)
print(b_result_last)

y_obs = pred_prey_model.calculate_model(a, b, second_x_data)

plt.figure(figsize=(11, 6))
plt.plot(y_obs.ravel(), label="observed")
plt.plot(pred_prey_model.calculate_model(a_result_last, b_result_last, second_x_data).ravel(), label="simulated")
plt.legend(loc="upper left")

plt.show()
