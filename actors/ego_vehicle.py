import carla
import random

from actors.vehicle import Vehicle
from custom_agents.ego_agent import EgoAgent
from kinematic_models.bicycle_model import BicycleModel


class EgoVehicle(Vehicle):
    def __init__(self, world, bp=None, spawn_point=None):
        super().__init__(world, bp, spawn_point)
        super().spawn()

        # different agents can impose different control schemas to the vehicle
        super().set_agent(EgoAgent)

        #self.model = BicycleModel(self.actor)

