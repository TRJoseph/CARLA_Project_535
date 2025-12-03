import carla
import cvxpy
import numpy as np
import math

from agents.navigation.basic_agent import BasicAgent

from agents.navigation.global_route_planner import GlobalRoutePlanner

class EgoAgent(BasicAgent):
    def __init__(self, vehicle, controller, model, target_speed=20, debug=False):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param target_speed: speed (in Km/h) at which the vehicle will move
        """
        self.owner_vehicle_reference = vehicle
        self.route = None
        self.controller = controller
        self.model = model
        self.horizon = controller.horizon
        
        super().__init__(vehicle.actor, target_speed)


    def set_route(self, start_location, end_location):
        # Use global planner to get route
        self.route = self._global_planner.trace_route(start_location, end_location)
        
        # Feed route to local planner
        self._local_planner.set_global_plan(self.route)
    

    ## TODO: Maybe put all debug-related stuff in its own class?
    def draw_route_debug(self):
        try:
            for idx, waypoint in enumerate(self.route):
                self._world.debug.draw_string(waypoint[0].transform.location, f"WP {idx+1}", life_time=100)
        except Exception as e:
            print("Please ensure a route is set first.")

    def mpc_to_carla_control(self, u):
        control = carla.VehicleControl()
    
        a = u[0]      # Acceleration in m/s²
        delta = u[1]  # Steering angle in radians
        
        # Convert acceleration to throttle/brake
        if a > 0:
            # Positive acceleration -> throttle
            control.throttle = float(np.clip(a / 4.0, 0.0, 1.0))
            control.brake = 0.0
        else:
            # Negative acceleration -> brake
            control.throttle = 0.0
            control.brake = float(np.clip(-a / 8.0, 0.0, 1.0))
        
        # Convert steering angle to CARLA steer [-1, 1]
        max_steer_angle = self.model.delta_lim[1]  # Max steering angle for Tesla Model 3
        control.steer = float(np.clip(delta / max_steer_angle, -1.0, 1.0))
        
        # Other defaults
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def wrap_to_pi(self, angle):
        """Wrap angle to [-pi, pi)"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def compute_nominal_guess(self, xk, ref_k):
        v = xk[3]
        v_ref = ref_k[3]
        yaw = xk[2]
        yaw_ref = ref_k[2]

        # speed error
        accel = 0.5 * (v_ref - v)

        # heading error
        heading_error = self.wrap_to_pi(yaw_ref - yaw)
        delta = 0.5 * heading_error

        # clamp to model limits
        accel = np.clip(accel, self.model.a_lim[0], self.model.a_lim[1])
        delta = np.clip(delta, self.model.delta_lim[0], self.model.delta_lim[1])

        return np.array([accel, delta])

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        N = self.horizon

        veh_location = self._vehicle.get_location()
        vehicle_speed = self._vehicle.get_velocity()
        vehicle_speed = math.sqrt(vehicle_speed.x ** 2 + vehicle_speed.y ** 2 + vehicle_speed.z ** 2)
        self._min_distance = self._local_planner._base_min_distance + self._local_planner._distance_ratio * vehicle_speed

        
        self._local_planner._sampling_radius = self._target_speed * self.model.h

        while len(self._local_planner._waypoints_queue) < self.horizon:
                self._local_planner._compute_next_waypoints(k=self.horizon)

        waypoint_buffer = self._local_planner.get_plan()

        x0 = self.owner_vehicle_reference.get_current_state()

        wp_horizon = []

        for k in range(N):
            waypoint, _ = waypoint_buffer[k]
            
            ref_state = np.array([
                waypoint.transform.location.x,
                waypoint.transform.location.y,
                self.wrap_to_pi(np.deg2rad(waypoint.transform.rotation.yaw)),
                self._target_speed / 3.6 # km/h to meters per second
            ])

            wp_horizon.append(ref_state)

        num_waypoint_removed = 0
        for waypoint, _ in self._local_planner._waypoints_queue:

            if len(self._local_planner._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._local_planner._waypoints_queue.popleft()

        #ctrl_result = self.controller.compute_control(x_bar, u_bar, x0)
        ctrl_result = self.controller.compute_control_casadi_nonlinear(np.array(wp_horizon), x0)

        ctrl_action = self.mpc_to_carla_control(ctrl_result)

        # Actions to take during each simulation step
        return ctrl_action