import numpy as np
import cvxpy as cp
from casadi import *


class MPCController:
    def __init__(self, model):
        self.model = model
        self.horizon = 10

    def calc_Jacobian(self, x, u):
        L_f = self.model.L_f
        L_r = self.model.L_r
        dt = 0.05

        psi = x[2]
        v   = x[3]
        delta = u[1]
        a   = u[0]

        # Jacobian of the system dynamics
        A = np.zeros((4, 4))
        B = np.zeros((4, 2))

        A = np.array([[1, 0, -dt*v*np.sin(psi+np.arctan((L_r*np.arctan(delta))/(L_f+L_r))), dt*np.cos(psi+np.arctan((L_r*np.arctan(delta))/(L_f+L_r)))],
                    [0, 1, dt*v*np.cos(psi+np.arctan((L_r*np.arctan(delta))/(L_f+L_r))), dt*np.sin(psi+np.arctan((L_r*np.arctan(delta))/(L_f+L_r)))],
                    [0, 0, 1, (dt*np.arctan(delta))/((((L_r**2*np.arctan(delta)**2))/((L_f+L_r)**2)+1)**(1/2)*(L_f+L_r))],
                    [0, 0, 0, 1]])
        
        
        B = np.array([
        [0, -(dt*L_r*v*np.sin(psi+np.arctan((L_r*np.arctan(delta))/(L_f+L_r))))/((delta**2+1)*((L_r**2*np.arctan(delta)**2)/((L_f+L_r)**2)+1)*(L_f+L_r))],
        [0, (dt*L_r*v*np.cos(psi+np.arctan((L_r*np.arctan(delta))/(L_f+L_r))))/((delta**2+1)*((L_r**2*np.arctan(delta)**2)/((L_f+L_r)**2)+1)*(L_f+L_r))],
        [0, (dt*v)/((delta**2+1)*((L_r**2*np.arctan(delta)**2)/((L_f+L_r)**2)+1)**(3/2)*(L_f+L_r))],
        [dt, 0]])

        return [A, B]


    def compute_control(self, state, ctrl, x0):
        len_state = state.shape[0]
        len_ctrl  = ctrl.shape[0]
        dim_state = state.shape[1]
        dim_ctrl  = ctrl.shape[1]

        n_u = len_ctrl * dim_ctrl
        n_x = len_state * dim_state
        n_var = n_u + n_x

        x = cp.Variable(n_var)

        states = x[:n_x].reshape((len_state, dim_state), order="C") 
        controls = x[n_x:].reshape((len_ctrl, dim_ctrl), order="C")   

        ### WEIGHT MATRICIES ###
        Q = np.diag([1000, 1000, 10, 1000])  # weights on [x, y, yaw, v]
        R = np.diag([0.5, 0.1])      # weights on [a, delta]
        Pt = np.diag([10, 10, 20, 50])  # terminal cost

        P = cp.quad_form(states[len_state-1], Pt)
        q = 0

        # build q from 0 to N-1
        for i in range(len_state-1):

            Q_term = cp.quad_form(states[i, :], Q)

            R_term = cp.quad_form(controls[i, :], R)

            q += (Q_term + R_term)

        # cost is running + terminal
        cost = P + q


        # a_limit and delta_limit are vectors containing the upper and lower constraints
        # upper bounds 
        ub = np.array([self.model.a_lim[1], self.model.delta_lim[1]])
        # lower bounds
        lb = np.array([self.model.a_lim[0], self.model.delta_lim[0]])

        # define constraints
        constraints = []

        for i in range(len_state-1):
            a, b = self.calc_Jacobian(state[i, :], ctrl[i, :])

            constraints.append(states[i+1, :] == a @ states[i, :] + b @ controls[i, :])
            
            # append new control constraint for acceleration and steering angle
            constraints.append(controls[i, :] >= lb - ctrl[i, :])
            constraints.append(controls[i, :] <= ub - ctrl[i, :])
            
        # add the last initial difference constraint
        state_diff_init = x0 - state[0, :]
        constraints.append(states[0, :] == state_diff_init)

        # solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints=constraints)
        prob.solve(verbose=False, max_iter = 10000)

        u_act = x.value[n_x:n_x + dim_ctrl] + ctrl[0, :]
        return u_act

    def compute_control_casadi_nonlinear(self, waypoints, x0):
        """
        This function is my practice implementing a nonlinear MPC controller
        waypoints: target waypoints to track (N x 4) [x, y, yaw, v]
        x0: current state (4,)
        """
        N = waypoints.shape[0] - 1  # horizon length
        dim_state = 4
        dim_ctrl = 2
        
        # Decision variables (absolute, not deviations)v
        # state vector is [x pos, y pos, yaw angle, velocity]
        # 4xN vector where N is the horizon waypoint count
        x = SX.sym("state", dim_state, N+1)
        u = SX.sym("ctrl", dim_ctrl, N)
        
        ### WEIGHT MATRICIES ###
        Q = np.diag([1000, 1000, 100, 200])   # weights on [x, y, yaw, v]
        R = np.diag([1.0, 0.5])      # weights on [a, delta]
        P = np.diag([50, 50, 200, 100])  # terminal cost
        
        cost = 0
        # build cost from 0 to N
        for k in range(N):
            state_error = x[:, k] - SX(waypoints[k, :])
            cost += dot(state_error, mtimes(Q, state_error))
            cost += dot(u[:, k], mtimes(R, u[:, k]))
        
        # terminal cost
        terminal_error = x[:, N] - SX(waypoints[N, :])
        cost += dot(terminal_error, mtimes(Q, terminal_error))

        constraints = []
        
        # initial condition
        constraints.append(x[:, 0] - SX(x0))
        
        # Nonlinear dynamics constraints
        for k in range(N):
            x_next = self.model.Fun_dynamics_dt_casadi(x[:, k], u[:, k])
            constraints.append(x[:, k+1] - x_next)
        
        decision_vars = vertcat(reshape(x, -1, 1), reshape(u, -1, 1))
        g = vertcat(*constraints)
        nlp = {'x': decision_vars, 'f': cost, 'g': g}
        
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        S = nlpsol('S', 'ipopt', nlp, opts)
        
        # Bounds
        n_x = (N+1) * dim_state
        n_u = N * dim_ctrl
        
        lbg = np.zeros(g.shape[0])
        ubg = np.zeros(g.shape[0])
        
        lbx = np.full(n_x + n_u, -np.inf)
        ubx = np.full(n_x + n_u, np.inf)
        
        # Control bounds (absolute, not relative to reference)
        lbx[n_x:] = np.tile([self.model.a_lim[0], self.model.delta_lim[0]], N)
        ubx[n_x:] = np.tile([self.model.a_lim[1], self.model.delta_lim[1]], N)
        
        # initial guess just using waypoints and no control guess (maybe tweak this for faster convergence?)
        x0_guess = np.concatenate([waypoints.flatten('F'), np.zeros(n_u)])
        
        # Solve
        r = S(x0=x0_guess, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        # In compute_control_casadi_nonlinear, after solving:
        
        u_opt = r['x'][n_x:n_x+dim_ctrl].full().flatten()
        
        return u_opt