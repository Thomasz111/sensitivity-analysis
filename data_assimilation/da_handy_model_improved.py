import numpy as np

from handy_model_improved import create_society_from_values
from simulation_constants import batch_size_c


class ImprovedHandyModel:
    def calculate_model(self, xC, xE, data):
        t = data[0]
        start_sim = int(t[0])
        end_sim = int(t[len(t)-1])+1
        sim_length = end_sim - start_sim

        society = create_society_from_values(
            (5e-3, 1, 5e-4, xC, xE, 100, 1e-2, 100, 3.0e-5, 0, 3e-2, 3e-2, 1e-2, 7e-2, 0))
        result = society.evolve(end_sim)
        # plot_society(*result)
        result = result[1][-sim_length:]
        return np.array([result])

    def model(self, xC, xE, batch_size=1, random_state=None):
        t = self.second_x_data[0]
        start_sim = int(t[0])
        end_sim = int(t[len(t) - 1]) + 1
        sim_length = end_sim - start_sim

        x = np.zeros((batch_size_c, sim_length))
        for i in range(batch_size_c):
            society = create_society_from_values(
                (5e-3, 1, 5e-4, xC[i], xE[i], 100, 1e-2, 100, 3.0e-5, 0, 3e-2, 3e-2, 1e-2, 7e-2, 0))
            result = society.evolve(end_sim)
            result = result[1][-sim_length:]
            x[i] = np.array([result])
        print("x")
        return x
