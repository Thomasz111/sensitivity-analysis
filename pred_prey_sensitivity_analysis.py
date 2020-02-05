from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from predator_prey_model import PredatorPreyModel
from simulation_constants import simulation_steps, start_simulation_time, end_simulation_time, sample_num
import pylab as p


# Define the model inputs
problem = {
    'num_vars': 2,
    'names': ['alfa', 'beta'],
    'bounds': [[0.75, 1.25],
               [0.15, 0.25]]
}

# Generate samples
param_values = saltelli.sample(problem, sample_num)

# Calculate sensivitness for every simulation step (model setup)
sensitivity = []
firstVariableSensitivity = []
secondVariableSensitivity = []
firstAndSecondVariableSensitivity = []

model = PredatorPreyModel()
model.setup_model(param_values)

# Evaluate model for every step of simulation
for i in range(simulation_steps):
    # get array of model values at time step 'i'
    Y = model.get_array_of_model_values(i)
    # analyze these values using sobol method
    Si = sobol.analyze(problem, Y, print_to_console=False)

    firstVariableSensitivity.append(Si['S1'][0])
    secondVariableSensitivity.append(Si['S1'][1])
    firstAndSecondVariableSensitivity.append(Si['S2'][0,1])

# plot results
t = np.linspace(start_simulation_time, end_simulation_time, simulation_steps)
f1 = p.figure()
p.plot(t, firstVariableSensitivity, 'r-', label='alfa')
p.plot(t, secondVariableSensitivity, 'b-', label='beta')
p.plot(t, firstAndSecondVariableSensitivity, 'g-', label='alfaAndBeta')
p.grid()
p.legend(loc='best')
p.xlabel('time')
p.ylabel('sensitivness')
p.title('Sensitivness of alfa & beta in time for prey')
p.show()
