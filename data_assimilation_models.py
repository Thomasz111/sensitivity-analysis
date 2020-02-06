import numpy as np
from scipy import integrate


class PredatorPreyModelDA:
    initial_prey_population = 10
    initial_predator_population = 5
    a = 1.
    b = 0.5
    gamma = 1.5
    delta = 0.75

    def dX_dt(self, X, t):
        return np.array([self.a * X[0] - self.b * X[0] * X[1],
                         -self.gamma * X[1] + self.delta * self.b * X[0] * X[1]])

    def calculate_model(self, a, b, data):
        self.a = a
        self.b = b
        # prey and pred populations at beginning
        predator_prey_populations = np.array([self.initial_prey_population, self.initial_predator_population])
        t = data[0]
        proper_sim_steps = len(t)
        start_sim = int(t[0])
        end_sim = int(t[len(t)-1])+1
        steps = int((end_sim * len(t)) / (end_sim - start_sim))
        t = np.linspace(0, end_sim, steps)

        # calculate predator-prey
        populations_at_step, info_dict = integrate.odeint(self.dX_dt, predator_prey_populations, t, full_output=True)
        populations_at_step = populations_at_step[-proper_sim_steps:]
        return np.array([populations_at_step[:, 0]])

    def model(self, a, b, batch_size=1, random_state=None):
        # prey and pred populations at beginning
        predator_prey_populations = np.array([self.initial_prey_population, self.initial_predator_population])
        t = self.second_x_data[0]
        proper_sim_steps = len(t)
        start_sim = int(t[0])
        end_sim = int(t[len(t) - 1]) + 1
        steps = int((end_sim * len(t)) / (end_sim - start_sim))
        t = np.linspace(0, end_sim, steps)

        x = np.zeros((100, 200))
        for i in range(100):
            self.a = a[i]
            self.b = b[i]

            # calculate predator-prey
            populations_at_step, info_dict = integrate.odeint(self.dX_dt, predator_prey_populations, t, full_output=True)
            populations_at_step = populations_at_step[-proper_sim_steps:]
            x[i] = np.array([populations_at_step[:, 0]])
        return x
