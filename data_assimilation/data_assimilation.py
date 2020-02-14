import elfi
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math

from da_handy_model_improved import ImprovedHandyModel
from da_utils import get_three_ranges
from simulation_constants import N, batch_size_c, width, first_x_range, second_x_range, \
    third_x_range, sampling, start, stop


def start_all(variable_name, variable_num, quantile, f):
    f.write('{0} {1}\n'.format(variable_name, quantile))
    seed = 20170530
    np.random.seed(seed)

    # We set true parameters
    true_values = {'p': 0.005, 'k': 1, 's': 0.0005, 'xC': 100, 'xE': 25, 'y': 100, 'yy': 0.01,
                   'n': 100, 'w': 100, 'd': 0.00005, 'bC': 0.03, 'bE': 0.03, 'am': 0.01, 'aM': 0.07}

    number_of_samples = int(1 / sampling)

    full_range = np.arange(start, stop, sampling)
    all_x_range = np.array(full_range).reshape((1, len(full_range)))

    first_x_data, second_x_data, third_x_data = get_three_ranges(start,
                                                                 first_x_range,
                                                                 second_x_range,
                                                                 third_x_range,
                                                                 number_of_samples,
                                                                 full_range)

    pred_prey_model = ImprovedHandyModel()
    pred_prey_model.variable_num = variable_num

    # Plot the observed sequence for whole range
    y_obs = pred_prey_model.calculate_model(true_values, all_x_range)
    plt.figure(figsize=(11, 6))
    plt.plot(all_x_range[0, :], y_obs[0, :])

    # Points between these lines are training points
    plt.axvline(x=first_x_range, color='r')
    plt.axvline(x=second_x_range, color='r')

    plt.xlabel('X value as an argument for model')
    plt.ylabel('Y value of the model')
    plt.savefig('{0}_{1}_all.png'.format(variable_name, quantile))
    plt.close()

    # We plot only training part
    train_data = pred_prey_model.calculate_model(true_values, second_x_data)
    plt.figure(figsize=(11, 6))

    plt.xticks(np.arange(first_x_range, second_x_range, 1.0))

    plt.plot(second_x_data[0, :], train_data[0, :])
    plt.xlabel('X value as an argument for function')
    plt.ylabel('Y value of the function')
    plt.savefig('{0}_{1}_part.png'.format(variable_name, quantile))
    plt.close()

    # MAGIC
    pred_prey_model.second_x_data = second_x_data
    elfi.new_model()
    # has to be this way, so the keys in result dict have parameter names
    xC = elfi.Prior(scipy.stats.uniform, true_values['xC'] - width * true_values['xC'], 2 * width * true_values['xC'])
    xE = elfi.Prior(scipy.stats.uniform, true_values['xE'] - width * true_values['xE'], 2 * width * true_values['xE'])
    y = elfi.Prior(scipy.stats.uniform, true_values['y'] - width * true_values['y'], 2 * width * true_values['y'])
    w = elfi.Prior(scipy.stats.uniform, true_values['w'] - width * true_values['w'], 2 * width * true_values['w'])
    k = elfi.Prior(scipy.stats.uniform, true_values['k'] - width * true_values['k'], 2 * width * true_values['k'])
    s = elfi.Prior(scipy.stats.uniform, true_values['s'] - width * true_values['s'], 2 * width * true_values['s'])
    p = elfi.Prior(scipy.stats.uniform, true_values['p'] - width * true_values['p'], 2 * width * true_values['p'])
    yy = elfi.Prior(scipy.stats.uniform, true_values['yy'] - width * true_values['yy'], 2 * width * true_values['yy'])
    n = elfi.Prior(scipy.stats.uniform, true_values['n'] - width * true_values['n'], 2 * width * true_values['n'])
    d = elfi.Prior(scipy.stats.uniform, true_values['d'] - width * true_values['d'], 2 * width * true_values['d'])
    bC = elfi.Prior(scipy.stats.uniform, true_values['bC'] - width * true_values['bC'], 2 * width * true_values['bC'])
    bE = elfi.Prior(scipy.stats.uniform, true_values['bE'] - width * true_values['bE'], 2 * width * true_values['bE'])
    am = elfi.Prior(scipy.stats.uniform, true_values['am'] - width * true_values['am'], 2 * width * true_values['am'])
    aM = elfi.Prior(scipy.stats.uniform, true_values['aM'] - width * true_values['aM'], 2 * width * true_values['aM'])

    # Define the simulator node with the MA2 model ,give the priors to it as arguments.
    Y = elfi.Simulator(pred_prey_model.model, p, k, s, xC, xE, y, yy, n, d, w, bC, bE, am, aM, observed=train_data)

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
        f.write('{0}: {1}\n'.format(key, value))

    y_obs = pred_prey_model.calculate_model(true_values, second_x_data)
    plt.figure(figsize=(11, 6))
    plt.plot(y_obs.ravel(), label="observed")
    plt.plot(pred_prey_model.calculate_model(mean_results, second_x_data).ravel(),
             label="simulated")
    plt.legend(loc="upper left")
    plt.savefig('{0}_{1}_final_part.png'.format(variable_name, quantile))
    plt.close()

    y_obs = pred_prey_model.calculate_model(true_values, all_x_range)
    plt.figure(figsize=(11, 6))
    plt.plot(y_obs.ravel(), label="observed")
    all_results_predicted = pred_prey_model.calculate_model(mean_results, all_x_range)
    plt.plot(all_results_predicted.ravel(), label="simulated")
    plt.legend(loc="upper left")
    plt.savefig('{0}_{1}_final_all.png'.format(variable_name, quantile))
    plt.close()

    def calculate_error(start, stop):
        calculate = 0
        for i in range(start, stop, 1):
            calculate += (y_obs[0][i] - all_results_predicted[0][i]) ** 2
        return calculate

    aa = calculate_error(int(start + 1), int(first_x_range * number_of_samples))
    bb = calculate_error(int(first_x_range * number_of_samples), int(second_x_range * number_of_samples))
    cc = calculate_error(int(second_x_range * number_of_samples), int(third_x_range * number_of_samples))
    f.write('{0}\n{1}\n{2}\n\n'.format(aa, bb, cc))
