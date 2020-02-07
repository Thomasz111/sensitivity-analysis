import numpy as np
from scipy import integrate

from simulation_constants import batch_size_c


class HANDYDA:
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

    def calculate_model(self, a, b, data):
        self.xC = a
        self.xE = b
        self.initial_populations = [self.xC, self.xE, self.y, self.w]
        # prey and pred populations at beginning
        t = data[0]
        proper_sim_steps = len(t)
        start_sim = int(t[0])
        end_sim = int(t[len(t)-1])+1
        steps = int((end_sim * len(t)) / (end_sim - start_sim))
        t = np.linspace(0, end_sim, steps)

        # calculate predator-prey
        populations_at_step, info_dict = integrate.odeint(self.dX_dt, self.initial_populations, t, full_output=True)
        populations_at_step = populations_at_step[-proper_sim_steps:]
        return np.array([populations_at_step[:, 0]])

    def model(self, a, b, batch_size=1, random_state=None):
        # prey and pred populations at beginning
        t = self.second_x_data[0]
        proper_sim_steps = len(t)
        start_sim = int(t[0])
        end_sim = int(t[len(t) - 1]) + 1
        steps = int((end_sim * len(t)) / (end_sim - start_sim))
        t = np.linspace(0, end_sim, steps)

        x = np.zeros((batch_size_c, proper_sim_steps))
        for i in range(batch_size_c):
            self.a = a[i]
            self.b = b[i]

            # calculate predator-prey
            populations_at_step, info_dict = integrate.odeint(self.dX_dt, self.initial_populations, t, full_output=True)
            populations_at_step = populations_at_step[-proper_sim_steps:]
            x[i] = np.array([populations_at_step[:, 0]])
        print("x")
        return x
