from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from sensitivity_analysis.handy_sa_model import Handy_SA
from simulation_constants import simulation_steps, start_simulation_time, end_simulation_time, sample_num
import pylab as p


# Define the model inputs
problem = {
    'num_vars': 4,
    'names': ['xC', 'xE', 'y', 'w'],
    'bounds': [[50.0, 150.0],
               [12.5, 37.5],
               [50.0, 150.0],
               [50.0, 150.0]]
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
        # get array of model values at time step 'i'
        Y = model.get_array_of_model_values(i, variable)
        # analyze these values using sobol method
        Si = sobol.analyze(problem, Y, print_to_console=False)

        firstVariableSensitivity.append(Si['S1'][0])
        secondVariableSensitivity.append(Si['S1'][1])
        thirdVariableSensitivity.append(Si['S1'][2])
        forthVariableSensitivity.append(Si['S1'][3])

        # mean sensitivity values
    print("############ idicies for {} ###############".format(variable))
    print(sum(firstVariableSensitivity[1:])/len(firstVariableSensitivity[1:]))
    print(sum(secondVariableSensitivity[1:])/len(secondVariableSensitivity[1:]))
    print(sum(thirdVariableSensitivity[1:])/len(thirdVariableSensitivity[1:]))
    print(sum(forthVariableSensitivity[1:])/len(forthVariableSensitivity[1:]))

# plot results
def plot_results(v):
    t = np.linspace(start_simulation_time, end_simulation_time, simulation_steps)
    f1 = p.figure()
    p.plot(t, firstVariableSensitivity, 'r-', label='xC')
    p.plot(t, secondVariableSensitivity, 'b-', label='xE')
    p.plot(t, thirdVariableSensitivity, 'c-', label='y')
    p.plot(t, forthVariableSensitivity, 'm-', label='w')
    p.grid()
    p.legend(loc='best')
    p.xlabel('time')
    p.ylabel('sensitivness')
    p.title('Sensitivity for'.format())
    p.show()

if __name__ == "__main__":
    variables = ["xC", "xE", "y", "w"]
    for v in variables:

        firstVariableSensitivity = []
        secondVariableSensitivity = []
        thirdVariableSensitivity = []
        forthVariableSensitivity = []
        evaluate(v)
        plot_results(v)
