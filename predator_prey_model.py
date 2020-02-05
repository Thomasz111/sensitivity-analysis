import numpy as np
from scipy import integrate

from simulation_constants import start_simulation_time, end_simulation_time, simulation_steps


class PredatorPreyModel:
    simulations_results = []
    initial_prey_population = 10
    initial_predator_population = 5
    alfa = 1.
    beta = 0.2
    gamma = 1.5
    delta = 0.75

    def dX_dt(self, X, t):
        return np.array([self.alfa * X[0] - self.beta * X[0] * X[1],
                         -self.gamma * X[1] + self.delta * self.beta * X[0] * X[1]])

    def setup_model(self, values):
        for i, X in enumerate(values):
            # prey and pred populations at beginning
            predator_prey_populations = np.array([self.initial_prey_population, self.initial_predator_population])
            t = np.linspace(start_simulation_time, end_simulation_time, simulation_steps)
            self.alfa = X[0]
            self.beta = X[1]

            # calculate predator-prey
            populations_at_step, info_dict = integrate.odeint(self.dX_dt, predator_prey_populations, t, full_output=True)
            self.simulations_results.append(populations_at_step)

    def get_array_of_model_values(self, step):
        Y = np.zeros([len(self.simulations_results)])
        for i in range(len(self.simulations_results)):
            Y[i] = self.simulations_results[i][step][0]  # !!! [0] for preys, [1] for predators
        return Y
