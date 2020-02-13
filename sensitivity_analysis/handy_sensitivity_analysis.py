import time

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from sensitivity_analysis.handy_sa_model import Handy_SA
from simulation_constants import simulation_steps, start_simulation_time, end_simulation_time, sample_num
import pylab as p
import csv

names = ['p', 'k', 's', 'xC', 'xE', 'y', 'yy', 'n', 'w', 'd', 'bC', 'bE', 'am', 'aM']

# Define the model inputs
problem = {
    'num_vars': 14,
    'names': names,
    'bounds': [[0.0001, 0.01],
               [0, 20],
               [0.00001, 0.001],
               [25.0, 175.0],
               [10.0, 120.0],
               [25, 175],
               [0.001, 0.1],
               [25, 175.0],
               [25, 175.0],
               [0.0000001, 0.0001],
               [0.001,0.1],
               [0.001,0.1],
               [0.001,0.1],
               [0.001,0.2]]
}

# Generate samples
param_values = saltelli.sample(problem, sample_num)

# Calculate sensivitness for every simulation step (model setup)
sensitivity = []

model = Handy_SA()
model.setup_model(param_values)


# Evaluate model for every step of simulation
def evaluate(variable):
    for i in range(simulation_steps):
        print("Simulation step number: {}".format(i))
        # get array of model values at time step 'i'
        Y = model.get_array_of_model_values(i, variable)
        # analyze these values using sobol method
        Si = sobol.analyze(problem, Y, print_to_console=False)

        firstVariableSensitivity.append(Si['ST'][0])
        secondVariableSensitivity.append(Si['ST'][1])
        thirdVariableSensitivity.append(Si['ST'][2])
        forthVariableSensitivity.append(Si['ST'][3])
        fifthVariableSensitivity.append(Si['ST'][4])
        sixthVariableSensitivity.append(Si['ST'][5])
        seventhVariableSensitivity.append(Si['ST'][6])
        eighthVariableSensitivity.append(Si['ST'][7])
        ninthVariableSensitivity.append(Si['ST'][8])
        tenthVariableSensitivity.append(Si['ST'][9])
        eleventhVariableSensitivity.append(Si['ST'][10])
        twelfthVariableSensitivity.append(Si['ST'][11])
        thirteenthVariableSensitivity.append(Si['ST'][12])
        fourteenthVariableSensitivity.append(Si['ST'][13])

        # mean sensitivity values
    print("############ idicies for {} ###############".format(variable))
    p_average = sum(firstVariableSensitivity[1:]) / len(firstVariableSensitivity[1:])
    k_average = sum(secondVariableSensitivity[1:]) / len(secondVariableSensitivity[1:])
    s_average = sum(thirdVariableSensitivity[1:]) / len(thirdVariableSensitivity[1:])
    xC_average = sum(forthVariableSensitivity[1:]) / len(forthVariableSensitivity[1:])
    xE_average = sum(fifthVariableSensitivity[1:]) / len(fifthVariableSensitivity[1:])
    y_average = sum(sixthVariableSensitivity[1:]) / len(sixthVariableSensitivity[1:])
    yy_average = sum(seventhVariableSensitivity[1:]) / len(seventhVariableSensitivity[1:])
    n_average = sum(eighthVariableSensitivity[1:]) / len(eighthVariableSensitivity[1:])
    w_average = sum(ninthVariableSensitivity[1:]) / len(ninthVariableSensitivity[1:])
    d_average = sum(tenthVariableSensitivity[1:]) / len(tenthVariableSensitivity[1:])
    bC_average = sum(eleventhVariableSensitivity[1:]) / len(eleventhVariableSensitivity[1:])
    bE_average = sum(twelfthVariableSensitivity[1:]) / len(twelfthVariableSensitivity[1:])
    am_average = sum(thirdVariableSensitivity[1:]) / len(thirdVariableSensitivity[1:])
    aM_average = sum(fourteenthVariableSensitivity[1:]) / len(fourteenthVariableSensitivity[1:])

    print(p_average)
    print(k_average)
    print(s_average)
    print(xC_average)
    print(xE_average)
    print(y_average)
    print(yy_average)
    print(n_average)
    print(w_average)
    print(d_average)
    print(bC_average)
    print(bE_average)
    print(am_average)
    print(aM_average)

    results = {
    "p_average":p_average,
    "k_average":k_average,
    "s_average":s_average,
    "xC_average":xC_average,
    "xE_average":xE_average,
    "y_average":y_average,
    "yy_average":yy_average,
    "n_average":n_average,
    "w_average":w_average,
    "d_average":d_average,
    "bC_average":bC_average,
    "bE_average":bE_average,
    "am_average":am_average,
    "aM_average":aM_average,
    }

    csv_columns = list(results.keys())

    with open("data/SA_results_{}.csv".format(variable), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(results)


# plot results
def plot_results(v):
    t = np.linspace(start_simulation_time, end_simulation_time, simulation_steps)
    f1 = p.figure()
    p.plot(t, firstVariableSensitivity, 'b-', label=names[0])
    p.plot(t, secondVariableSensitivity, 'g-', label=names[1])
    p.plot(t, thirdVariableSensitivity, 'r-', label=names[2])
    p.plot(t, fourteenthVariableSensitivity, 'c-', label=names[3])
    p.plot(t, fifthVariableSensitivity, 'm-', label=names[4])
    p.plot(t, sixthVariableSensitivity, 'k-', label=names[5])
    p.plot(t, seventhVariableSensitivity, 'y-', label=names[6])
    p.plot(t, eighthVariableSensitivity, 'b^', label=names[7])
    p.plot(t, ninthVariableSensitivity, 'g^', label=names[8])
    p.plot(t, tenthVariableSensitivity, 'r^', label=names[9])
    p.plot(t, eleventhVariableSensitivity, 'c^', label=names[10])
    p.plot(t, twelfthVariableSensitivity, 'm^', label=names[11])
    p.plot(t, thirteenthVariableSensitivity, 'k^', label=names[12])
    p.plot(t, fourteenthVariableSensitivity , 'y^', label=names[13])
    p.grid()
    p.legend(loc='best')
    p.xlabel('time')
    p.ylabel('sensitivness')
    p.title('Sensitivity for: {}'.format(v))
    fig = p.gcf()
    p.show()
    fig.savefig("plot/sesitivity_{}.png".format(v), bbox_inches='tight')


if __name__ == "__main__":
    variables = ['xC', 'xE', 'y', 'w']
    start_time = time.time()
    for v in variables:
        firstVariableSensitivity = []
        secondVariableSensitivity = []
        thirdVariableSensitivity = []
        forthVariableSensitivity = []
        fifthVariableSensitivity = []
        sixthVariableSensitivity = []
        seventhVariableSensitivity = []
        eighthVariableSensitivity = []
        ninthVariableSensitivity = []
        tenthVariableSensitivity = []
        eleventhVariableSensitivity = []
        twelfthVariableSensitivity = []
        thirteenthVariableSensitivity = []
        fourteenthVariableSensitivity = []

        evaluate(v)
        plot_results(v)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
