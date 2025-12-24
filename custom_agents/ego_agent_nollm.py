import carla
import cvxpy
import numpy as np
import math

from agents.navigation.basic_agent import BasicAgent

from agents.navigation.global_route_planner import GlobalRoutePlanner

class EgoAgentNoLLM(BasicAgent):
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


    ## TODO: Maybe put all debug-related stuff in its own class?
    def draw_route_debug(self):
        try:
            for idx, waypoint in enumerate(self.route):
                self._world.debug.draw_string(waypoint[0].transform.location, f"WP {idx+1}", life_time=100)
        except Exception as e:
            print("Please ensure a route is set first.")

    
    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        # Actions to take during each simulation step
        control = super().run_step()
        return control