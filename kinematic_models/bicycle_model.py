import numpy as np
import casadi as ca

class BicycleModel:

    """ 
    This is my attempt at modeling a Tesla Model 3 with a simplified bicycle model approach.
    Instead of tracking two separate parallel links from front to rear wheels, we can simpify the approach with one central link.
    This works well for highway driving.

    """
    def __init__(self, params):
        self.Dim_state = 4
        self.Dim_ctrl  = 2

        self.wheelbase = params["wheelbase"]
        self.L_f = params["L_f"]
        self.L_r = params["L_r"]
        self.a_lim = params["a_lim"]
        self.delta_lim = params["delta_lim"]
        self.h = params["dt"]

        self.initial_state = params["initial_state"]

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
    
    def Fun_dynamics_dt_casadi(self, state, ctrl):
        x, y, yaw, v = state[0], state[1], state[2], state[3]
        a, delta = ctrl[0], ctrl[1]

        beta = ca.arctan(self.L_r / (self.L_r + self.L_f) * ca.arctan(delta))

        x_next = x + v*ca.cos(yaw+beta) * self.h
        y_next = y + v*ca.sin(yaw+beta) * self.h
        yaw_next = yaw + v/self.L_r * ca.sin(beta) * self.h
        v_next = v + a * self.h

        return ca.vertcat(x_next, y_next, yaw_next, v_next)
    