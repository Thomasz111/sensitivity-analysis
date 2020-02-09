import numpy as np

from handy_model_improved import create_society_from_values
from simulation_constants import end_simulation_time


class Handy_SA:
    simulations_results = []

    params = {"xC": 0, "xE": 1, "y": 2, "w": 3}

    def setup_model(self, values):
            for i, X in enumerate(values):

                society = create_society_from_values(
                    (5e-3, 1, 5e-4, X[0], X[1], X[2], 1e-2, 100, 3.0e-5, X[3], 3e-2, 3e-2, 1e-2, 7e-2, 0))
                time, commoner_population, elite_population, nature, wealth, carrying_capacity = \
                    society.evolve(end_simulation_time)

                self.simulations_results.append([commoner_population, elite_population, nature, wealth])

    def get_array_of_model_values(self, step, var):
        Y = np.zeros([len(self.simulations_results)])
        for i in range(len(self.simulations_results)):
            Y[i] = self.simulations_results[i][self.params.get(var)][step]  # xC [0], xE [1], y [2], w [3]
        return Y
