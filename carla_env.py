import carla
import random
import yaml
import time

from core.cleanup import cleanup as clu

from actors.vehicle import Vehicle
from actors.ego_vehicle import EgoVehicle

from actors.rgbcam import RGBCam

from custom_agents.ego_agent import EgoAgent

class CarlaEnv:

    def load_carla_config(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def connect_client(self, host='localhost', port=2000):
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        return client
    
    def set_world_settings(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = self.config["simulation"]["synchronous_mode"]
        settings.fixed_delta_seconds = self.config["simulation"]["time_step"]
        self.world.apply_settings(settings)

    def __init__(self):
        self.config = self.load_carla_config()

        self.client = self.connect_client()
        self.client.load_world(self.config["world"])
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.world_spawn_points = self.map.get_spawn_points()
        #self.waypoints = self.map.generate_waypoints(2)

        # to change vehicle change config bp id
        self.ego_vehicle = EgoVehicle(self.world, self.config["simulation"]["ego_vehicle_bp_id"], self.world_spawn_points[self.config["simulation"]["ego_vehicle_spawn_point"]])

        self.set_world_settings()

    def reset(self):
        self.collision_hist = []

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        # camera bp from config
        self.rgb_cam = RGBCam(self.world, self.config["simulation"]["rgb_camera_id"], host_actor=self.ego_vehicle.actor, spawn_transform=transform)
        self.rgb_cam.set_attribute("image_size_x", self.rgb_cam.IM_WIDTH)
        self.rgb_cam.set_attribute("image_size_y", self.rgb_cam.IM_HEIGHT)
        self.rgb_cam.spawn()
        #self.actor_list.append(self.rgb_cam.actor)

        # tells camera sensor to start processing image data
        self.rgb_cam.start_listening()

        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # TODO: maybe make a sensor class? have different sensor inherit from it
        col_sensor = self.blueprint_library.find(self.config["simulation"]["collision_sensor_id"])
        self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.ego_vehicle.actor)
        #self.actor_list.append(self.col_sensor)

        # TODO: take action on collision
        self.col_sensor.listen(lambda event: self.collision_data(event))

        #self.ego_vehicle.enable_autopilot()

        self.ego_vehicle.set_vehicle_route(self.ego_vehicle.get_transform().location, self.world_spawn_points[self.config["simulation"]["ego_vehicle_target_point"]].location)

        self.ego_vehicle.agent.draw_route_debug()
    
    def get_actor_list(self) -> carla.ActorList:
        return self.world.get_actors()

    def step_forward(self):
        # applies the Vehicle's Agent run_step method
        self.ego_vehicle.step_vehicle()

        self.world.tick()

    def draw_world_spawn_points(self):
        for i, spawn_point in enumerate(self.world_spawn_points):
            # Draw in the spectator window the spawn point index
            self.world.debug.draw_string(spawn_point.location, str(i), life_time=100)
            # We can also draw an arrow to see the orientation of the spawn point
            # (i.e. which way the vehicle will be facing when spawned)
            self.world.debug.draw_arrow(spawn_point.location, spawn_point.location + spawn_point.get_forward_vector(), life_time=100)

    def cleanup(self):
        clu(self.client, self.get_actor_list())
        self.collision_hist = []

