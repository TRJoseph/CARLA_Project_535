import carla
import random

class Vehicle:
    def __init__(self, world, bp=None, spawn_point=None):
        self.world = world
        self.bp_lib = world.get_blueprint_library()

        if bp is None:
            self.bp = random.choice(self.bp_lib.filter('*vehicle*'))
        else:
            if isinstance(bp, str):
                self.bp = self.bp_lib.find(bp)
            else:
                self.bp = bp

        self.spawn_point = spawn_point or random.choice(world.get_map().get_spawn_points())
        self.actor = None
        self.agent = None

    def set_attribute(self, id: str, value: str) -> None:
        self.bp.set_attribute(id, value)

    def spawn(self) -> carla.Actor:
        self.actor = self.world.try_spawn_actor(self.bp, self.spawn_point)
        return self.actor
    
    def set_agent(self, agent):
        self.agent = agent(self.actor)

    def set_vehicle_route(self, start_location, end_location):
        self.agent.set_route(start_location, end_location)
    
    def enable_autopilot(self, enabled=True) -> None:
        if self.actor:
            self.actor.set_autopilot(enabled)

    def apply_control(self, VehicleControl: carla.VehicleControl) -> None:
        self.actor.apply_control(VehicleControl)

    def get_transform(self):
        return self.actor.get_transform()
    
    def step_vehicle(self):
        control = self.agent.run_step()
        self.apply_control(control)

    def destroy(self):
        if self.actor:
            print(f"Destroying actor id={self.actor.id}")
            self.actor.destroy()
            self.actor = None

