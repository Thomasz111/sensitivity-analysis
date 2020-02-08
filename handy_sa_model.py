import numpy as np
from simulation_constants import simulation_steps, start_simulation_time, end_simulation_time, sample_num
from scipy import integrate

class Handy_SA:
    simulations_results = []
    initial_prey_population = 10
    initial_predator_population = 5
    xC = 10000  # X[0]
    xE = 3000  # X[1]
    y = 100  # X[2]
    w = 100  # X[3]
    initial_populations = np.array([xC, xE, y, w])
    am = 0.01
    aM = 0.07
    bC = 0.03
    bE = 0.03
    s = 0.0005
    p = 0.005
    yy = 0.01
    n = 100
    k = 1
    d = 0

    params = {"xC": 0, "xE": 1, "y": 2, "w": 3}

    def CC(self, w, xC, xE):
        return min(1, w/self.wth(xC, xE))*self.s*xC

    def CE(self, w, xC, xE):
        return min(1, w/self.wth(xC, xE))*self.k*self.s*xE

    def wth(self, xC, xE):
        return self.p*xC+self.k*self.p*xE

    def aC(self, w, xC, xE):
        return self.am + max(0, 1-self.CC(w, xC, xE)/(self.s*xC))*(self.aM-self.am)

    def aE(self, w, xC, xE):
        return self.am + max(0, 1-self.CE(w, xC, xE)/(self.s*xE))*(self.aM-self.am)

    def dX_dt(self, X, t):
        return np.array([self.bC*X[0]-self.aC(X[3], X[0], X[1])*X[0],
                         self.bE*X[1]-self.aE(X[3], X[0], X[1])*X[1],
                         self.yy*X[2]*(self.n-X[2])-self.d*X[0]*X[2],
                         self.d*X[0]*X[2]-self.CC(X[3], X[0], X[1])-self.CE(X[3], X[0], X[1])])


    def setup_model(self, values):
            for i, X in enumerate(values):

                initial_populations = np.array([self.xC, self.xE, self.y, self.w])
                t = np.linspace(start_simulation_time, end_simulation_time, simulation_steps)

                self.xC = X[0]
                self.xE = X[1]
                self.y = X[2]
                self.w = X[3]

                populations_at_step, info_dict = integrate.odeint(self.dX_dt, initial_populations, t, full_output=True)
                self.simulations_results.append(populations_at_step)

    def get_array_of_model_values(self, step, var):
        Y = np.zeros([len(self.simulations_results)])
        for i in range(len(self.simulations_results)):
            Y[i] = self.simulations_results[i][step][self.params.get(var)]  # xC [0], xE [1], y [2], w [3]
        return Y
