import numpy as np

from handy_model_improved import create_society_from_values
from simulation_constants import batch_size_c


class ImprovedHandyModel:
    def calculate_model(self, values, data):
        t = data[0]
        start_sim = int(t[0])
        end_sim = int(t[len(t)-1])+1
        sim_length = end_sim - start_sim

        society = create_society_from_values(
            (values['p'], values['k'], values['s'], values['xC'], values['xE'], values['y'], values['yy'],
             values['n'], values['d'], values['w'], values['bC'], values['bE'], values['am'], values['aM'], 0))
        result = society.evolve(end_sim)
        # plot_society(*result)
        result = result[1][-sim_length:]
        return np.array([result])

    def model(self, p, k, s, xC, xE, y, yy, n, d, w, bC, bE, am, aM, batch_size=1, random_state=None):
        t = self.second_x_data[0]
        start_sim = int(t[0])
        end_sim = int(t[len(t) - 1]) + 1
        sim_length = end_sim - start_sim

        x = np.zeros((batch_size_c, sim_length))
        for i in range(batch_size_c):
            society = create_society_from_values(
                (p[i], k[i], s[i], xC[i], xE[i], y[i], yy[i], n[i], d[i], w[i], bC[i], bE[i], am[i], aM[i], 0))
            result = society.evolve(end_sim)
            result = result[1][-sim_length:]
            x[i] = np.array([result])
        print("x")
        return x
