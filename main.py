import random
import carla
import time
import yaml
from core.spawn_utils import *
from core.autopilot import *
from core.cleanup import *
from carla_env import CarlaEnv

def main():
    c_env = CarlaEnv()
    try:
        if c_env.config["debug"]["draw_spawn_points"]:
            c_env.draw_world_spawn_points()

        c_env.reset()

        spectator = c_env.world.get_spectator()
        transform = c_env.ego_vehicle.spawn_point
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))

        print("Simulation running... Press Ctrl+C to stop.")

        while True:
            #time.sleep(0.1)
            c_env.step_forward()
    except KeyboardInterrupt:
        print("Stopping simulation...")
    finally:
        print("Cleaning up...")
        c_env.cleanup()
        c_env.step_forward()


if __name__ == '__main__':
    main()