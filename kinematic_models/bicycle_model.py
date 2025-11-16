import numpy as np

class BicycleModel:

    """ 
    x is the state vector comprised of [x pos; y pos; yaw angle; velocity]
    
    """



    # x0 is the initial state vector for the model
    def __init__(self, params):

        self.L_f = params["L_f"]
        self.L_r = params["L_r"]
        self.a_lim = params["a_lim"]
        self.delta_lim = params["delta_lim"]

        pass

    # calculates state dynamics for the bicycle model
    def Fun_dynamics_dt(self, x, u):

        xdot = np.zeros(4)        
        beta = np.arctan(self.L_r / (self.L_r + self.L_f) * np.arctan(u[1]))

        xdot[0] = x[3] * np.cos(x[2] + beta)
        xdot[1] = x[3] * np.sin(x[2] + beta)
        xdot[2] = x[3] / self.L_r * np.sin(beta)
        xdot[3] = u[0]
    
        xkp1 = x + xdot * self.h

        return xkp1
    